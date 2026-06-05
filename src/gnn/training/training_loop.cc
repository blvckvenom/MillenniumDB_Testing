#include "gnn/training/training_loop.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "gnn/training/async_batch_prefetcher.h"

#ifdef ENABLE_CUDA_ASSEMBLER
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#ifdef GNN_CUDA_ENABLED
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace mdb::gnn {

// =============================================================================
// Construction
// =============================================================================

TrainingLoop::TrainingLoop(
    GraphSAGEModel&      model,
    BatchAssembler&      assembler,
    const SampleCatalog& catalog,
    torch::optim::Adam&  optimizer,
    Config               config)
    : model_(model)
    , assembler_(assembler)
    , catalog_(catalog)
    , optimizer_(optimizer)
    , config_(std::move(config))
{
    // Phase 0 (2026-05-17): open per-batch profile CSV iff user opted in via
    // Config::profile_log_path. BatchTimingLog truncates the target, writes
    // the CSV header, and flushes on each `flush_interval` records (default
    // 64). The unique_ptr lifetime matches TrainingLoop's, so the destructor
    // flushes any tail records.
    if (!config_.profile_log_path.empty()) {
        profile_log_ = std::make_unique<BatchTimingLog>(config_.profile_log_path);
    }
}

// =============================================================================
// Training
// =============================================================================

TrainingLoop::Result TrainingLoop::train()
{
    Result result;

    // --- Reproducibility -------------------------------------------------
    if (config_.random_seed >= 0) {
        torch::manual_seed(static_cast<uint64_t>(config_.random_seed));
    }

    // --- Device detection ---------------------------------------------------
    // Probe the first batch to discover whether the FeatureAssembler returns
    // CUDA tensors (L1 cache in GPU VRAM) or CPU tensors. If CUDA, move
    // the model so all parameters are on the same device as features.
    torch::Device device(torch::kCPU);
    {
        MiniBatch probe = assembler_.assemble(0);
        device = probe.features.device();
    }
    if (!device.is_cpu()) {
        model_.to(device);
    }

    // Optimizer is owned by the caller (accessible via optimizer_). Do NOT
    // re-create it here — doing so would discard Adam momenta carried over
    // from a resumed checkpoint.
    auto& optimizer = optimizer_;

    // Seed from resume state (defaults to fresh training)
    double   best_val_acc     = config_.start_best_val;
    uint64_t patience_counter = config_.start_patience;

    // Spec B2 (2026-04-27): seed prev_disk before the loop so the first
    // epoch's delta == bytes accrued during epoch 0 (not since process start).
    // When provider is unset, prev/cur/delta stay zero and the conditional
    // print below is suppressed, preserving pre-B2 line format.
    uint64_t prev_disk = 0;
    if (config_.cumulative_disk_bytes_provider) {
        prev_disk = config_.cumulative_disk_bytes_provider();
    }

    // Prepend seed_losses into result so epoch_losses history is contiguous
    for (double l : config_.seed_losses) {
        result.epoch_losses.push_back(l);
    }

    const uint64_t train_batches = catalog_.train_batches;
    const uint64_t val_batches   = catalog_.validation_batches;

    auto wall_start = std::chrono::steady_clock::now();

    // === Training start banner ============================================
    const std::string device_str = device.is_cpu() ? "CPU" : "CUDA";
    std::cout << "\n"
              << "========================================================\n"
              << "[TrainingLoop] STARTING TRAINING\n"
              << "========================================================\n"
              << "  device:           " << device_str                     << "\n"
              << "  start_epoch:      " << config_.start_epoch            << "\n"
              << "  total_epochs:     " << config_.epochs                 << "\n"
              << "  patience:         " << config_.patience               << "\n"
              << "  train_batches:    " << train_batches                  << "\n"
              << "  val_batches:      " << val_batches                    << "\n"
              << "  seed_best_val:    " << config_.start_best_val         << "\n"
              << "  seed_patience:    " << config_.start_patience         << "\n"
              << "========================================================\n"
              << std::flush;

    // Round 3B (2026-05-15): resolve effective worker count ONCE before the
    // epoch loop so the stderr warning (if any) and the result yield are
    // emitted exactly once per train() invocation rather than per-epoch.
    // Multi-worker (N>1) is supported ONLY for FeatureMatrix-fallback
    // BatchAssemblers; the FourLevelStore path has shared state inside
    // load_batch_features (DirectIoReader io_uring rings + pinned host
    // buffer) that races silently under concurrent calls. CLAMP to 1
    // rather than throw — keeping the API "more is fine, we just may not
    // honor it" so callers can pass aggressive defaults without
    // per-deployment branching.
    unsigned effective_workers = config_.prefetch_num_workers;
    if (effective_workers == 0) {
        effective_workers = 1;
    }
    if (config_.use_async_prefetcher && assembler_.uses_feature_store()
        && effective_workers > 1) {
        // Round 3B-mw (2026-06-01): the FourLevelStore path is now
        // multi-worker-safe — each worker owns a private DirectIoReader (its
        // own io_uring rings) and a private pinned staging buffer, so there is
        // no shared mutable state on the feature hot path. Provision those
        // resources up front. If per-worker O_DIRECT readers cannot be opened
        // (e.g. fd exhaustion) we clamp to 1 rather than let a worker read
        // zeros — fail safe, never silently corrupt features.
        try {
            assembler_.prepare_feature_store_workers(effective_workers);
            fls_prefetch_workers_ = effective_workers;  // evaluate() reuses this
            std::cerr
                << "[TrainingLoop] prefetchNumWorkers=" << effective_workers
                << " on the FourLevelStore path (per-worker DirectIoReader + "
                   "pinned buffer enabled)." << std::endl;
        } catch (const std::exception& e) {
            std::cerr
                << "[TrainingLoop] could not enable multi-worker FourLevelStore"
                   " prefetch (" << e.what() << "); clamping to 1."
                << std::endl;
            effective_workers = 1;
        }
    } else if (assembler_.uses_feature_store() && effective_workers > 1) {
        // No async prefetcher: features load on the calling (main) thread, so
        // N>1 is moot. Report the honest effective worker count.
        effective_workers = 1;
    }
    result.effective_prefetch_workers = effective_workers;

    for (uint64_t epoch = config_.start_epoch;
         epoch < config_.start_epoch + config_.epochs;
         ++epoch)
    {
        auto epoch_start = std::chrono::steady_clock::now();

        // === Training phase ===
        model_.train();

        // Round 1E (2026-05-15): on-device loss accumulator. Replaces the
        // per-batch `loss.item<double>()` (GPU→CPU sync) with a single
        // `.item()` at end-of-epoch. Stays on `device` so loss.detach()
        // accumulates without crossing the PCIe bus.
        torch::Tensor epoch_loss_sum;
        if (device.is_cpu()) {
            epoch_loss_sum = torch::zeros({}, torch::kFloat64);
        } else {
            epoch_loss_sum = torch::zeros(
                {},
                torch::TensorOptions().dtype(torch::kFloat64).device(device));
        }
        uint64_t num_labeled_batches = 0;
        uint64_t num_train_batches   = 0;

        // Spec C3 stage 1.B: optional async prefetcher fronted by a single
        // worker thread (DiskGNN paper §5.3 producer-consumer pattern).
        // The prefetcher is per-epoch — destroyed at scope exit, joining
        // its worker. Cost is ~100 μs per epoch (Linux thread spawn),
        // negligible vs the per-batch assembly cost it hides.
        //
        // Spec C3 stage 3: when stage3_active, the prefetcher worker uses
        // its own pool stream and records a CUDAEvent into MiniBatch.
        // The training thread (this loop) uses a separate train_stream and
        // event.block()s on it before forward — letting the GPU schedule
        // assemble_kernel and forward+backward concurrently when SMs allow.
#ifdef ENABLE_CUDA_ASSEMBLER
        const bool stage3_active = config_.use_async_prefetcher
                                && config_.use_cuda_streams
                                && !device.is_cpu();
        std::optional<c10::cuda::CUDAStream> train_stream;
        if (stage3_active) {
            train_stream.emplace(c10::cuda::getStreamFromPool());
        }
#else
        constexpr bool stage3_active = false;
#endif

        std::unique_ptr<AsyncBatchPrefetcher> prefetcher;
        if (config_.use_async_prefetcher) {
            prefetcher = std::make_unique<AsyncBatchPrefetcher>(
                assembler_, config_.prefetch_queue_size, stage3_active,
                effective_workers);
            const size_t prime = std::min<size_t>(
                config_.prefetch_queue_size, train_batches);
            for (size_t i = 0; i < prime; ++i) {
                prefetcher->prefetch(i);
            }
        }

        for (uint64_t bid = 0; bid < train_batches; ++bid) {
            // Phase 0 (2026-05-17) profile record. Populated incrementally
            // through the per-batch stages; appended once at end-of-iteration
            // when profile_log_ is active. Zero-initialised so fields we
            // can't cleanly time on the prefetcher path (per-tier counters,
            // sample_read_us / active_us / edge_us / h2d_us) stay at 0.
            BatchTiming bt{};
            bt.batch_id = bid;
            bt.split    = 0;  // TRAIN

            // Spec C3 stage 0: assemble + device transfer is the work an
            // async prefetcher (stage 1) would hide behind compute.
            // When the prefetcher is on, this measurement reflects only
            // the wait-for-ready-batch + device transfer time, not the
            // actual disk + CPU assembly which has overlapped.
            auto t_assem_start = std::chrono::steady_clock::now();

            MiniBatch mini;
            if (prefetcher) {
                // Order matters: next() FIRST to free a queue slot. The prime
                // loop above filled the queue to its full capacity, so calling
                // prefetch(lookahead) before next() would block on backpressure
                // (in_flight == queue_size, no space to enqueue). After next()
                // releases one slot, prefetch(lookahead) is non-blocking.
                mini = prefetcher->next();
                const uint64_t lookahead = bid + config_.prefetch_queue_size;
                if (lookahead < train_batches) {
                    prefetcher->prefetch(lookahead);
                }
            } else {
                mini = assembler_.assemble(bid);
            }

            // Phase A (2026-05-19): sub-stage timings populated by
            // BatchAssembler are now propagated via MiniBatch::timing on
            // BOTH paths (sequential and async prefetcher). The worker
            // stamps mini.timing before pushing into the queue, so the
            // consumer reads safely after next() returns.
            bt.sample_read_us      = mini.timing.sample_read_ns      / 1000;
            bt.active_us           = mini.timing.active_ns           / 1000;
            bt.assembler_kernel_us = mini.timing.assembler_kernel_ns / 1000;
            bt.edge_us             = mini.timing.edge_ns             / 1000;

            // Per-tier sub-counters are still ONLY attributable on the
            // sequential path — FourLevelStore::last_*_ns_ is reset by the
            // next batch's load, so by the time we read it on the prefetcher
            // path the values would belong to the lookahead batch. With the
            // prefetcher, l1/l2/l3/l4/rmap stay at 0 (the assembler_kernel_us
            // value above is the umbrella replacement).
            if (!prefetcher) {
                if (const auto* fs = assembler_.feature_store()) {
                    bt.l1_us         = fs->last_l1_us();
                    bt.l2_us         = fs->last_l2_us();
                    bt.l3_us         = fs->last_l3_us();
                    bt.l4_us         = fs->last_l4_us();
                    bt.rmap_lookup_us= fs->last_rmap_us();
                }
            }

            // Path 4 / STEP 6 (2026-05-31): v2 addr_table fast-path telemetry,
            // read from the MiniBatch (stamped by BatchAssembler immediately
            // after the load) rather than from the FourLevelStore. This makes it
            // correct on BOTH the sequential and async-prefetcher paths — under
            // the prefetcher the store's last_used_addr_tables()/last_addr_load_us()
            // belong to the lookahead batch, which is why the old code only
            // recorded it on the sequential path (so useAddrTablesEffective was
            // always false on the default prefetcher-on config).
            if (mini.timing.used_addr_tables) {
                result.addr_tables_used_ever = true;
                ++result.addr_table_load_us_count;
                // Welford-style running mean — avoids overflow vs. sum.
                const double load_us =
                    static_cast<double>(mini.timing.addr_load_ns / 1000ULL);
                result.addr_table_load_us_mean +=
                    (load_us - result.addr_table_load_us_mean) /
                    static_cast<double>(result.addr_table_load_us_count);
            }

            // Spec C3 stage 3: cross-stream sync + record_stream BEFORE any
            // tensor reads on the train stream. event.block makes train
            // stream wait for worker's assemble_kernel + .to(device); it
            // does NOT block the host. record_stream prevents the caching
            // allocator from freeing the worker-allocated tensors while
            // train_stream is still using them.
#ifdef ENABLE_CUDA_ASSEMBLER
            std::optional<c10::cuda::CUDAStreamGuard> stage3_guard;
            if (stage3_active && train_stream) {
                if (mini.ready_event.isCreated()) {
                    mini.ready_event.block(*train_stream);
                }
                stage3_guard.emplace(*train_stream);
                if (mini.features.is_cuda()) {
                    mini.features.record_stream(*train_stream);
                }
                for (auto& ei : mini.edge_indices) {
                    if (ei.is_cuda()) ei.record_stream(*train_stream);
                }
                for (auto& ai : mini.active_indices_per_layer) {
                    if (ai.is_cuda()) ai.record_stream(*train_stream);
                }
                if (mini.labels.is_cuda()) {
                    mini.labels.record_stream(*train_stream);
                }
                if (mini.label_mask.is_cuda()) {
                    mini.label_mask.record_stream(*train_stream);
                }
            }
#endif

            auto t_h2d_start = std::chrono::steady_clock::now();
            if (!device.is_cpu()) {
                mini.features = mini.features.to(device);
                for (auto& ei : mini.edge_indices) {
                    ei = ei.to(device);
                }
                for (auto& ai : mini.active_indices_per_layer) {
                    ai = ai.to(device);
                }
                mini.labels     = mini.labels.to(device);
                mini.label_mask = mini.label_mask.to(device);
            }
            auto t_assem_end = std::chrono::steady_clock::now();
            result.assemble_seconds += std::chrono::duration<double>(
                t_assem_end - t_assem_start).count();
            // load_features_us = umbrella for (assemble + h→d) — same as Phase 0.
            // Phase A now attributes its 50% uninstrumented mass via
            // sample_read/active/assembler_kernel/edge/h2d sub-stages above.
            bt.load_features_us = std::chrono::duration_cast<std::chrono::microseconds>(
                t_assem_end - t_assem_start).count();
            bt.h2d_us = std::chrono::duration_cast<std::chrono::microseconds>(
                t_assem_end - t_h2d_start).count();

            // Read-only isolation bench (2026-06-05): the producer path
            // (read_sample + load_batch_features + GPU assemble) has already
            // run to produce `mini`. Skip the model forward/backward/optimizer
            // so the prefetch workers run UNTHROTTLED by GPU compute; the
            // per-epoch io_disk/epoch_t then measures the read+assemble path's
            // throughput when compute does not pace it (handoff 2026-06-05 §3).
            if (config_.read_only_bench) {
                ++num_train_batches;
                if (profile_log_) {
                    profile_log_->append(bt);
                }
#ifdef GNN_CUDA_ENABLED
                if (config_.empty_cache_every_n_batches > 0 &&
                    num_train_batches % config_.empty_cache_every_n_batches == 0 &&
                    !device.is_cpu())
                {
                    c10::cuda::CUDACachingAllocator::emptyCache();
                }
#endif
                continue;
            }

            optimizer.zero_grad();

            auto t_fwd_start = std::chrono::steady_clock::now();
            auto logits = model_.forward(
                mini.features,
                mini.edge_indices,
                mini.active_sizes_per_layer
            );
            auto t_fwd_end = std::chrono::steady_clock::now();
            result.forward_seconds += std::chrono::duration<double>(
                t_fwd_end - t_fwd_start).count();
            bt.forward_us = std::chrono::duration_cast<std::chrono::microseconds>(
                t_fwd_end - t_fwd_start).count();

            // Round 1E (2026-05-15): guard backward on CPU-side num_labeled
            // (computed in BatchAssembler) instead of `label_mask.any().item<bool>()`
            // — the latter incurs a per-batch GPU→CPU sync.
            if (mini.num_labeled > 0) {
                auto masked_logits = logits.index({mini.label_mask});
                auto masked_labels = mini.labels.index({mini.label_mask});

                auto loss = torch::nn::functional::cross_entropy(
                    masked_logits, masked_labels
                );

                auto t_bwd_start = std::chrono::steady_clock::now();
                loss.backward();
                optimizer.step();

                auto t_bwd_end = std::chrono::steady_clock::now();
                result.backward_seconds += std::chrono::duration<double>(
                    t_bwd_end - t_bwd_start).count();
                bt.backward_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    t_bwd_end - t_bwd_start).count();

                // Round 1E: defer .item() to end-of-epoch. loss.detach() keeps
                // the scalar on-device; the add is a tiny GPU kernel. One sync
                // per epoch instead of ~train_batches per epoch.
                epoch_loss_sum += loss.detach().to(epoch_loss_sum.dtype());
                ++num_labeled_batches;
            }

            ++num_train_batches;

            // Phase 0: emit this batch's timing record. No-op when
            // profile_log_path was empty (profile_log_ is nullptr).
            if (profile_log_) {
                profile_log_->append(bt);
            }

            // Periodic fragmentation reclaim. Variable batch receptive
            // fields leak holes into the caching allocator pool; without
            // this call the OOM at ~12 min observed on UNDIRECTED 16 GB
            // GPUs hits well before epoch boundaries.
#ifdef GNN_CUDA_ENABLED
            if (config_.empty_cache_every_n_batches > 0 &&
                num_train_batches % config_.empty_cache_every_n_batches == 0 &&
                !device.is_cpu())
            {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
#endif
        }

        // Spec C3 stage 3: synchronize train_stream before validation reads
        // any tensors. evaluate() runs on the default stream (sequentially)
        // and may read tensors that were last written on train_stream.
#ifdef ENABLE_CUDA_ASSEMBLER
        if (train_stream) {
            train_stream->synchronize();
        }
#endif

#ifdef GNN_CUDA_ENABLED
        // End-of-epoch reclaim so validation runs against a defragged pool
        // and the next epoch starts clean.
        if (!device.is_cpu()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
#endif

        // Round 1E (2026-05-15): single end-of-epoch sync to read accumulated
        // loss off-device. The .item() fires exactly once per epoch instead
        // of once per labeled batch (~1300× on papers100M scale). Division
        // by num_train_batches preserves the previous behavior exactly —
        // total_loss only accrues from labeled batches, but the average is
        // computed across all batches (including those skipped due to no
        // labels).
        const double total_loss = (num_labeled_batches > 0)
            ? epoch_loss_sum.item<double>()
            : 0.0;
        double avg_loss = (num_train_batches > 0)
            ? (total_loss / static_cast<double>(num_train_batches))
            : 0.0;
        result.epoch_losses.push_back(avg_loss);

        // === Validation phase ===
        // Phase-split instrumentation: wall up to here is the training phase;
        // the delta across evaluate() is the validation phase.
        auto t_train_end = std::chrono::steady_clock::now();
        double train_phase_s = std::chrono::duration<double>(
            t_train_end - epoch_start).count();
        // Read-only bench skips validation — the train phase is the whole
        // measurement (workers read+assemble all batches with no compute).
        double val_accuracy = config_.read_only_bench
            ? 0.0
            : evaluate(train_batches, val_batches);
        double val_phase_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_train_end).count();

        // Phase 0: flush per-batch CSV at epoch boundary so partial logs
        // survive an early termination (Ctrl-C, OOM, etc.) and post-hoc
        // analysis sees epoch-N records even if the process never reached
        // the destructor.
        if (profile_log_) {
            profile_log_->flush();
        }

        // Spec B2: capture cumulative L3+L4 disk bytes post-validation so
        // the delta covers train + eval activity for this epoch. The
        // single uint64_t return keeps the Config callback ABI-stable.
        uint64_t cur_disk = 0;
        uint64_t delta_disk = 0;
        if (config_.cumulative_disk_bytes_provider) {
            cur_disk    = config_.cumulative_disk_bytes_provider();
            delta_disk  = cur_disk - prev_disk;
            prev_disk   = cur_disk;
        }

        // === Check if this epoch improved (BEFORE updating best_val_acc) ===
        // Capture is_best with the same strict comparator used below to update
        // best_val_acc, so callback consumers see a consistent "improved?" flag.
        const bool is_best = (val_accuracy > best_val_acc);

        if (is_best) {
            best_val_acc = val_accuracy;
            patience_counter = 0;
            // Checkpoint persistence is delegated to the on_epoch_end callback
            // (see AutoCheckpointer in Phase 3).
        } else {
            ++patience_counter;
        }

        // Fire per-epoch callback BEFORE any early-return. Consumers observe
        // every epoch including the final one that triggers patience-stop or
        // convergence.
        if (config_.on_epoch_end) {
            config_.on_epoch_end(EpochEvent{
                epoch,
                avg_loss,
                val_accuracy,
                patience_counter,
                is_best
            });
        }

        // === Per-epoch progress print =====================================
        auto epoch_end = std::chrono::steady_clock::now();
        double epoch_seconds = std::chrono::duration<double>(
            epoch_end - epoch_start).count();
        auto wall_so_far = std::chrono::duration<double>(
            epoch_end - wall_start).count();
        std::cout << "[TrainingLoop] epoch=" << std::setw(3) << (epoch + 1)
                  << "/" << config_.epochs
                  << "  loss=" << std::fixed << std::setprecision(4) << avg_loss
                  << "  val_acc=" << std::fixed << std::setprecision(4) << val_accuracy
                  << "  best_val=" << std::fixed << std::setprecision(4) << best_val_acc
                  << "  patience=" << patience_counter << "/" << config_.patience
                  << "  epoch_t=" << std::fixed << std::setprecision(1) << epoch_seconds << "s"
                  << "  (train=" << std::fixed << std::setprecision(1) << train_phase_s
                  << "s val=" << std::fixed << std::setprecision(1) << val_phase_s << "s)"
                  << "  total_t=" << std::fixed << std::setprecision(0) << wall_so_far << "s";
        // Spec B2: per-epoch L3+L4 disk-traffic delta inline. Suppressed
        // when provider is unset (delta stays zero) or the delta is
        // < 1 MB (dominant case for small datasets where the line would
        // just clutter).
        if (delta_disk >= (1ULL << 20)) {  // >= 1 MB
            constexpr double GB = 1024.0 * 1024.0 * 1024.0;
            std::cout << "  io_disk="
                      << std::fixed << std::setprecision(2)
                      << (static_cast<double>(delta_disk) / GB) << "GB";
        }
        std::cout << (is_best ? "  ★" : "") << std::endl;

        // === Patience check ===
        if (!is_best && patience_counter >= config_.patience) {
            std::cout << "[TrainingLoop] EARLY STOP — patience "
                      << patience_counter << " >= " << config_.patience
                      << " (no val improvement for " << patience_counter
                      << " epochs).\n"
                      << "  best_val_acc reached: " << best_val_acc << "\n"
                      << std::flush;

            result.converged = false;
            ++epoch;  // account for this epoch before break
            result.ran_epochs = epoch - config_.start_epoch;
            result.best_val_accuracy = best_val_acc;

            auto wall_end = std::chrono::steady_clock::now();
            result.train_seconds = std::chrono::duration<double>(
                wall_end - wall_start).count();
            return result;
        }

        // === Convergence check ===
        const size_t n = result.epoch_losses.size();
        if (n >= 2) {
            double delta = std::abs(
                result.epoch_losses[n - 1] - result.epoch_losses[n - 2]
            );
            if (delta < config_.tolerance) {
                std::cout << "[TrainingLoop] CONVERGED — loss delta "
                          << std::scientific << delta
                          << " < tolerance " << config_.tolerance << "\n"
                          << "  best_val_acc reached: "
                          << std::fixed << std::setprecision(4) << best_val_acc << "\n"
                          << std::flush;

                result.converged  = true;
                result.ran_epochs = (epoch - config_.start_epoch) + 1;
                result.best_val_accuracy = best_val_acc;

                auto wall_end = std::chrono::steady_clock::now();
                result.train_seconds = std::chrono::duration<double>(
                    wall_end - wall_start).count();
                return result;
            }
        }
    }

    // Completed all epochs without early stopping
    std::cout << "[TrainingLoop] EPOCHS COMPLETED — ran all "
              << config_.epochs << " epochs without early-stop or convergence.\n"
              << "  best_val_acc reached: "
              << std::fixed << std::setprecision(4) << best_val_acc << "\n"
              << std::flush;

    result.ran_epochs        = config_.epochs;
    result.best_val_accuracy = best_val_acc;

    auto wall_end = std::chrono::steady_clock::now();
    result.train_seconds = std::chrono::duration<double>(
        wall_end - wall_start).count();
    return result;
}

// =============================================================================
// Evaluation
// =============================================================================

double TrainingLoop::evaluate(uint64_t start_batch, uint64_t count)
{
    if (count == 0) {
        return 0.0;
    }

    torch::NoGradGuard no_grad;
    model_.eval();

    int64_t correct = 0;
    int64_t total   = 0;

    // Hoist device probe out of the inner loop. Mirrors train()'s pattern at
    // lines 63-66 — model_ is already on its final device by the time
    // train() reaches the validation phase (probe + .to() happens once at
    // the top of train()). When evaluate() is called standalone (e.g.,
    // tests) the model's first parameter still carries the canonical device.
    const torch::Device device = model_.parameters().begin()->device();

    // Round 1D (2026-05-15): mirror train()'s AsyncBatchPrefetcher to hide
    // validation-batch assemble + host→device cost behind the model forward
    // on GPU. evaluate() is called once per epoch from train() AFTER the
    // training prefetcher has drained (last train batch consumed, no more
    // prefetch() calls issued), so the train prefetcher's worker is idle
    // on its req_queue cv and there is no concurrent BatchAssembler access.
    // The eval prefetcher owns its own worker that is destructed (join) at
    // scope exit.
    //
    // Spec C3 stage 3: when enabled, the eval prefetcher's worker runs
    // assemble on a pool stream and records a CUDAEvent into MiniBatch;
    // this thread uses a separate eval_stream and event.block()s before
    // forward, mirroring train()'s lines 193-216 exactly.
#ifdef ENABLE_CUDA_ASSEMBLER
    const bool stage3_active = config_.use_async_prefetcher
                            && config_.use_cuda_streams
                            && !device.is_cpu();
    std::optional<c10::cuda::CUDAStream> eval_stream;
    if (stage3_active) {
        eval_stream.emplace(c10::cuda::getStreamFromPool());
    }
#else
    constexpr bool stage3_active = false;
#endif

    // Round 3B (2026-05-15): same FourLevelStore guard as train(). See
    // train()'s comment for rationale. evaluate() is called once per epoch
    // so the warning would be spammy — we suppress it here (train() already
    // emitted it once on the first epoch).
    unsigned effective_eval_workers = config_.prefetch_num_workers;
    if (effective_eval_workers == 0) {
        effective_eval_workers = 1;
    }
    if (assembler_.uses_feature_store()) {
        // Round 3B-mw: reuse EXACTLY the per-worker IO count train() already
        // provisioned (fls_prefetch_workers_; 1 if multi-worker was not
        // enabled). Never exceed it — a worker id beyond the provisioned slots
        // would fall back to the shared primary DirectIoReader and race.
        // prepare_feature_store_workers() ran in train() before the epoch loop
        // (single-threaded), so the slots already exist here.
        effective_eval_workers = fls_prefetch_workers_;
    }

    std::unique_ptr<AsyncBatchPrefetcher> prefetcher;
    if (config_.use_async_prefetcher) {
        prefetcher = std::make_unique<AsyncBatchPrefetcher>(
            assembler_, config_.prefetch_queue_size, stage3_active,
            effective_eval_workers);
        const size_t prime = std::min<size_t>(
            config_.prefetch_queue_size, count);
        for (size_t i = 0; i < prime; ++i) {
            prefetcher->prefetch(start_batch + i);
        }
    }

    // Phase 0 (2026-05-17): infer split from start_batch using the catalog
    // layout (train batches first, then validation, then test). Anything
    // landing at or after train+validation counts as TEST; otherwise VAL.
    // Matches the procedure-level convention used by gnn_train_procedure.
    const uint8_t eval_split =
        (start_batch >= catalog_.train_batches + catalog_.validation_batches)
            ? 2u   // TEST
            : 1u;  // VAL

    for (uint64_t i = 0; i < count; ++i) {
        BatchTiming bt{};
        bt.batch_id = start_batch + i;
        bt.split    = eval_split;

        auto t_load_start = std::chrono::steady_clock::now();

        MiniBatch mini;
        if (prefetcher) {
            // Same backpressure-safe ordering as train(): next() first to
            // free a queue slot, then prefetch the next-after-lookahead
            // batch (if any). The prime loop above filled the queue to
            // capacity, so prefetching before next() would block.
            mini = prefetcher->next();
            const uint64_t lookahead = i + config_.prefetch_queue_size;
            if (lookahead < count) {
                prefetcher->prefetch(start_batch + lookahead);
            }
        } else {
            mini = assembler_.assemble(start_batch + i);
        }

        // Phase A (2026-05-19): sub-stage timings propagated via mini.timing
        // on both paths — see train()'s sibling comment.
        bt.sample_read_us      = mini.timing.sample_read_ns      / 1000;
        bt.active_us           = mini.timing.active_ns           / 1000;
        bt.assembler_kernel_us = mini.timing.assembler_kernel_ns / 1000;
        bt.edge_us             = mini.timing.edge_ns             / 1000;

        if (!prefetcher) {
            if (const auto* fs = assembler_.feature_store()) {
                bt.l1_us         = fs->last_l1_us();
                bt.l2_us         = fs->last_l2_us();
                bt.l3_us         = fs->last_l3_us();
                bt.l4_us         = fs->last_l4_us();
                bt.rmap_lookup_us= fs->last_rmap_us();
            }
        }

        // Spec C3 stage 3: cross-stream sync + record_stream BEFORE any
        // tensor reads on the eval stream. Mirrors train()'s lines 193-216.
#ifdef ENABLE_CUDA_ASSEMBLER
        std::optional<c10::cuda::CUDAStreamGuard> stage3_guard;
        if (stage3_active && eval_stream) {
            if (mini.ready_event.isCreated()) {
                mini.ready_event.block(*eval_stream);
            }
            stage3_guard.emplace(*eval_stream);
            if (mini.features.is_cuda()) {
                mini.features.record_stream(*eval_stream);
            }
            for (auto& ei : mini.edge_indices) {
                if (ei.is_cuda()) ei.record_stream(*eval_stream);
            }
            for (auto& ai : mini.active_indices_per_layer) {
                if (ai.is_cuda()) ai.record_stream(*eval_stream);
            }
            if (mini.labels.is_cuda()) {
                mini.labels.record_stream(*eval_stream);
            }
            if (mini.label_mask.is_cuda()) {
                mini.label_mask.record_stream(*eval_stream);
            }
        }
#endif

        // Move all batch tensors to the model's device (Phase A: time h→d)
        auto t_h2d_start = std::chrono::steady_clock::now();
        if (!device.is_cpu()) {
            mini.features = mini.features.to(device);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(device);
            }
            for (auto& ai : mini.active_indices_per_layer) {
                ai = ai.to(device);
            }
            mini.labels     = mini.labels.to(device);
            mini.label_mask = mini.label_mask.to(device);
        }

        auto t_load_end = std::chrono::steady_clock::now();
        bt.load_features_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_load_end - t_load_start).count();
        bt.h2d_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_load_end - t_h2d_start).count();

        auto t_fwd_start = std::chrono::steady_clock::now();
        auto logits = model_.forward(
            mini.features,
            mini.edge_indices,
            mini.active_sizes_per_layer
        );
        auto t_fwd_end = std::chrono::steady_clock::now();
        bt.forward_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_fwd_end - t_fwd_start).count();

        // Round 1E (2026-05-15): guard on CPU-side num_labeled (computed in
        // BatchAssembler) to avoid the `label_mask.any().item<bool>()`
        // per-batch GPU→CPU sync. The remaining `.sum().item<int64_t>()`
        // collects validation accuracy — out of scope for Round 1E.
        if (mini.num_labeled > 0) {
            auto masked_logits = logits.index({mini.label_mask});
            auto masked_labels = mini.labels.index({mini.label_mask});

            auto predicted = masked_logits.argmax(1);
            int64_t batch_correct = (predicted == masked_labels).sum().item<int64_t>();
            correct += batch_correct;
            total   += masked_labels.size(0);
        }

        // Phase 0: emit this val/test batch's timing record. backward_us
        // stays 0 (no backward in eval). Guarded by profile_log_ so
        // tests / standalone evaluate() calls without an opt-in path are
        // unaffected.
        if (profile_log_) {
            profile_log_->append(bt);
        }
    }

    // Phase 0: flush after the val/test loop so the eval records are durable
    // even if a later phase (e.g., callback, test cleanup) throws.
    if (profile_log_) {
        profile_log_->flush();
    }

    // Spec C3 stage 3: synchronize eval_stream before returning so any
    // tensor reads following this call (e.g., the next epoch's train
    // prefetcher reading model state) observe a coherent view. The
    // per-batch .item<int64_t>() on `correct` is already an implicit sync
    // point, but the explicit synchronize matches train()'s lines 283-287
    // and survives future refactors that might remove the .item() calls.
#ifdef ENABLE_CUDA_ASSEMBLER
    if (eval_stream) {
        eval_stream->synchronize();
    }
#endif

    // Restore training mode for subsequent use. Note: not RAII — an
    // exception escaping the loop would leave model_ in eval mode. The
    // AsyncBatchPrefetcher destructor still runs (unique_ptr) and joins
    // the worker, but the training-mode flip is the caller's problem in
    // the exception path. Out of scope for this change.
    model_.train();

    return (total > 0) ? (static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
}

} // namespace mdb::gnn
