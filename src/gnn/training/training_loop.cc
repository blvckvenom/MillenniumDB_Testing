#include "gnn/training/training_loop.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "gnn/training/async_batch_prefetcher.h"

#ifdef ENABLE_CUDA_ASSEMBLER
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
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

    for (uint64_t epoch = config_.start_epoch;
         epoch < config_.start_epoch + config_.epochs;
         ++epoch)
    {
        auto epoch_start = std::chrono::steady_clock::now();

        // === Training phase ===
        model_.train();

        double   total_loss       = 0.0;
        uint64_t num_train_batches = 0;

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
                assembler_, config_.prefetch_queue_size, stage3_active);
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

            // Phase 0: per-tier sub-counters are attributable ONLY on the
            // sequential (non-prefetcher) path. With the prefetcher on, by
            // the time next() returns batch N the worker has already begun
            // load_batch_features() for batch N+1 (which resets last_*_ns_),
            // so the FourLevelStore's per-call timers race silently and
            // cannot be associated with this batch. Leaving at 0 is the
            // documented Phase 0 contract; per-batch tier propagation would
            // require attaching the timings to MiniBatch (deferred refactor).
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
                if (mini.labels.is_cuda()) {
                    mini.labels.record_stream(*train_stream);
                }
                if (mini.label_mask.is_cuda()) {
                    mini.label_mask.record_stream(*train_stream);
                }
            }
#endif

            if (!device.is_cpu()) {
                mini.features = mini.features.to(device);
                for (auto& ei : mini.edge_indices) {
                    ei = ei.to(device);
                }
                mini.labels     = mini.labels.to(device);
                mini.label_mask = mini.label_mask.to(device);
            }
            auto t_assem_end = std::chrono::steady_clock::now();
            result.assemble_seconds += std::chrono::duration<double>(
                t_assem_end - t_assem_start).count();
            // Phase 0: assemble + (optional) host→device transfer rolled into
            // a single bucket. sample_read / active / edge / h2d are subsumed
            // here until BatchAssembler exposes finer-grained sub-timers.
            bt.load_features_us = std::chrono::duration_cast<std::chrono::microseconds>(
                t_assem_end - t_assem_start).count();

            optimizer.zero_grad();

            auto t_fwd_start = std::chrono::steady_clock::now();
            auto logits = model_.forward(
                mini.features,
                mini.edge_indices,
                static_cast<int64_t>(mini.num_seeds)
            );
            auto t_fwd_end = std::chrono::steady_clock::now();
            result.forward_seconds += std::chrono::duration<double>(
                t_fwd_end - t_fwd_start).count();
            bt.forward_us = std::chrono::duration_cast<std::chrono::microseconds>(
                t_fwd_end - t_fwd_start).count();

            // Only back-prop if at least one labeled seed exists in this batch
            if (mini.label_mask.any().item<bool>()) {
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

                total_loss += loss.item<double>();
            }

            ++num_train_batches;

            // Phase 0: emit this batch's timing record. No-op when
            // profile_log_path was empty (profile_log_ is nullptr).
            if (profile_log_) {
                profile_log_->append(bt);
            }
        }

        // Spec C3 stage 3: synchronize train_stream before validation reads
        // any tensors. evaluate() runs on the default stream (sequentially)
        // and may read tensors that were last written on train_stream.
#ifdef ENABLE_CUDA_ASSEMBLER
        if (train_stream) {
            train_stream->synchronize();
        }
#endif

        double avg_loss = (num_train_batches > 0)
            ? (total_loss / static_cast<double>(num_train_batches))
            : 0.0;
        result.epoch_losses.push_back(avg_loss);

        // === Validation phase ===
        double val_accuracy = evaluate(train_batches, val_batches);

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
        auto mini = assembler_.assemble(start_batch + i);

        // Phase 0: per-tier sub-counters — evaluate() does not use the
        // prefetcher, so the FourLevelStore's last_*_us() can be attributed
        // directly to this batch.
        if (const auto* fs = assembler_.feature_store()) {
            bt.l1_us         = fs->last_l1_us();
            bt.l2_us         = fs->last_l2_us();
            bt.l3_us         = fs->last_l3_us();
            bt.l4_us         = fs->last_l4_us();
            bt.rmap_lookup_us= fs->last_rmap_us();
        }

        // Move all batch tensors to the model's device
        auto dev = model_.parameters().begin()->device();
        if (!dev.is_cpu()) {
            mini.features = mini.features.to(dev);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(dev);
            }
            mini.labels     = mini.labels.to(dev);
            mini.label_mask = mini.label_mask.to(dev);
        }
        auto t_load_end = std::chrono::steady_clock::now();
        bt.load_features_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_load_end - t_load_start).count();

        auto t_fwd_start = std::chrono::steady_clock::now();
        auto logits = model_.forward(
            mini.features,
            mini.edge_indices,
            static_cast<int64_t>(mini.num_seeds)
        );
        auto t_fwd_end = std::chrono::steady_clock::now();
        bt.forward_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_fwd_end - t_fwd_start).count();

        if (mini.label_mask.any().item<bool>()) {
            auto masked_logits = logits.index({mini.label_mask});
            auto masked_labels = mini.labels.index({mini.label_mask});

            auto predicted = masked_logits.argmax(1);
            correct += (predicted == masked_labels).sum().item<int64_t>();
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

    // Restore training mode for subsequent use
    model_.train();

    return (total > 0) ? (static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
}

} // namespace mdb::gnn
