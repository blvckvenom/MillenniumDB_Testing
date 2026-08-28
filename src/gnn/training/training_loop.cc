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

namespace {

// Folds one batch's v2 addr-table telemetry into the running result.
// Assumes mini.timing was stamped by BatchAssembler immediately after the
// load (correct on BOTH the sequential and async-prefetcher paths). Mutates
// only the three addr_table_* fields of result.
void record_addr_table_telemetry(const MiniBatch& mini,
                                 TrainingLoop::Result& result)
{
    // Address-table fast-path telemetry: v2 addr_table
    // metrics, read from the MiniBatch (stamped by BatchAssembler immediately
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
}

// Moves every tensor of the batch to the given device. No-op when the
// device is CPU (assembler output is already host-resident). Mutates the
// batch's tensor members in place; call before any forward on `device`.
void move_batch_to_device(MiniBatch& mini, const torch::Device& device)
{
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
}

// Stamps the metrics shared by every train() exit path (early stop,
// convergence, all-epochs-completed) into the result: best-val tracking
// plus the total wall-clock. Assumes result.converged and result.ran_epochs
// were already set by the caller (they differ per exit path). Reads the
// clock at call time, so call it at the point the exit path returns.
void finalize_result(TrainingLoop::Result&                 result,
                     double                                best_val_acc,
                     double                                best_val_test_acc,
                     uint64_t                              best_val_epoch_seen,
                     std::chrono::steady_clock::time_point wall_start)
{
    result.best_val_accuracy         = best_val_acc;
    result.test_accuracy_at_best_val = best_val_test_acc;
    result.best_val_epoch            = best_val_epoch_seen;

    auto wall_end = std::chrono::steady_clock::now();
    result.train_seconds = std::chrono::duration<double>(
        wall_end - wall_start).count();
}

} // anonymous namespace

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
    // Open the per-batch profile CSV iff the user opted in via
    // Config::profile_log_path. BatchTimingLog truncates the target, writes
    // the CSV header, and flushes on each `flush_interval` records (default
    // 64). The unique_ptr lifetime matches TrainingLoop's, so the destructor
    // flushes any tail records.
    if (!config_.profile_log_path.empty()) {
        profile_log_ = std::make_unique<BatchTimingLog>(config_.profile_log_path);
    }
}

// =============================================================================
// Training — stage helpers
// =============================================================================

void TrainingLoop::print_training_banner_(const torch::Device& device,
                                          uint64_t train_batches,
                                          uint64_t val_batches) const
{
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
}

unsigned TrainingLoop::resolve_effective_prefetch_workers_()
{
    // Resolve effective worker count ONCE before the epoch loop so the
    // stderr warning (if any) and the result yield are emitted exactly
    // once per train() invocation rather than per-epoch. Multi-worker
    // (N>1) is supported ONLY for FeatureMatrix-fallback BatchAssemblers;
    // the Four-Level Feature Store path has shared state inside
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
        // The Four-Level Feature Store path is multi-worker-safe — each
        // worker owns a private DirectIoReader (its own io_uring rings) and
        // a private pinned staging buffer, so there is no shared mutable
        // state on the feature hot path. Provision those resources up front.
        // If per-worker O_DIRECT readers cannot be opened (e.g. fd
        // exhaustion) we clamp to 1 rather than let a worker read zeros —
        // fail safe, never silently corrupt features.
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
    return effective_workers;
}

void TrainingLoop::apply_cosine_lr_schedule_(uint64_t epoch)
{
    // Cosine LR schedule: set the Adam lr for this epoch BEFORE
    // any forward/backward. Default ("") leaves the optimizer's lr untouched
    // (constant, canonical). "cosine" anneals learning_rate -> ~0 over the
    // run: lr(t) = learning_rate * 0.5 * (1 + cos(pi * t/T)).
    if (config_.lr_schedule == "cosine") {
        const double pi   = 3.14159265358979323846;
        const double T    = static_cast<double>(config_.epochs > 0 ? config_.epochs : 1);
        const double t    = static_cast<double>(epoch - config_.start_epoch);
        const double frac = (t < T) ? (t / T) : 1.0;
        const double new_lr = config_.learning_rate * 0.5 * (1.0 + std::cos(pi * frac));
        for (auto& group : optimizer_.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(group.options()).lr(new_lr);
        }
        std::cout << "[TrainingLoop] lr_schedule=cosine epoch=" << (epoch + 1)
                  << "/" << config_.epochs << "  lr="
                  << std::scientific << std::setprecision(4) << new_lr
                  << std::fixed << std::endl;
    }
}

void TrainingLoop::record_batch_stage_timings_(BatchTiming&     bt,
                                               const MiniBatch& mini,
                                               bool             sequential_path) const
{
    // Per-batch sub-stage timing breakdown: sub-stage
    // timings populated by BatchAssembler are propagated via
    // MiniBatch::timing on BOTH paths (sequential and async
    // prefetcher). The worker stamps mini.timing before pushing into
    // the queue, so the consumer reads safely after next() returns.
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
    if (sequential_path) {
        if (const auto* fs = assembler_.feature_store()) {
            bt.l1_us         = fs->last_l1_us();
            bt.l2_us         = fs->last_l2_us();
            bt.l3_us         = fs->last_l3_us();
            bt.l4_us         = fs->last_l4_us();
            bt.rmap_lookup_us= fs->last_rmap_us();
        }
    }
}

void TrainingLoop::forward_backward_step_(MiniBatch&     mini,
                                          BatchTiming&   bt,
                                          torch::Tensor& epoch_loss_sum,
                                          uint64_t&      num_labeled_batches,
                                          Result&        result)
{
    // Optimizer is owned by the caller (accessible via optimizer_). Do NOT
    // re-create it here — doing so would discard Adam momenta carried over
    // from a resumed checkpoint.
    auto& optimizer = optimizer_;

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

    // Sync-avoidance optimization: guard backward on
    // CPU-side num_labeled (computed in BatchAssembler) instead of
    // `label_mask.any().item<bool>()` — the latter incurs a per-batch
    // GPU→CPU sync.
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

        // Defer .item() to end-of-epoch (sync-avoidance). loss.detach()
        // keeps the scalar on-device; the add is a tiny GPU kernel. One
        // sync per epoch instead of ~train_batches per epoch.
        epoch_loss_sum += loss.detach().to(epoch_loss_sum.dtype());
        ++num_labeled_batches;
    }
}

double TrainingLoop::run_training_phase_(const torch::Device& device,
                                         unsigned             effective_workers,
                                         uint64_t             train_batches,
                                         Result&              result)
{
    model_.train();

    // On-device loss accumulator (GPU-sync reduction).
    // Replaces the per-batch `loss.item<double>()` (GPU→CPU sync) with a
    // single `.item()` at end-of-epoch. Stays on `device` so loss.detach()
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

    // Optional async prefetcher (training pipeline overlap): a single
    // producer-consumer worker thread overlaps each batch's assemble +
    // host→device transfer with the previous batch's forward+backward
    // (DiskGNN paper §5.3 pattern). The prefetcher is per-epoch —
    // destroyed at scope exit, joining its worker. Cost is ~100 μs per
    // epoch (Linux thread spawn), negligible vs the per-batch assembly
    // cost it hides.
    //
    // When stage3_active (dual CUDA streams), the prefetcher worker uses
    // its own pool stream and records a CUDAEvent into MiniBatch. The
    // training thread (this loop) uses a separate train_stream and
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
        // Profile record. Populated incrementally
        // through the per-batch stages; appended once at end-of-iteration
        // when profile_log_ is active. Zero-initialised so fields we
        // can't cleanly time on the prefetcher path (per-tier counters,
        // sample_read_us / active_us / edge_us / h2d_us) stay at 0.
        BatchTiming bt{};
        bt.batch_id = bid;
        bt.split    = 0;  // TRAIN

        // Assemble + device transfer is the work the async prefetcher
        // hides behind compute (pipeline overlap). When the prefetcher
        // is on, this measurement reflects only the wait-for-ready-batch
        // + device transfer time, not the actual disk + CPU assembly
        // which has already overlapped with the previous iteration.
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

        record_batch_stage_timings_(bt, mini, /*sequential_path=*/!prefetcher);
        record_addr_table_telemetry(mini, result);

        // Dual-stream CUDA overlap: cross-stream sync + record_stream
        // BEFORE any tensor reads on the train stream. event.block makes
        // train_stream wait for the prefetch worker's assemble_kernel +
        // .to(device); it does NOT block the host. record_stream prevents
        // the caching allocator from freeing the worker-allocated tensors
        // while train_stream is still using them.
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
        move_batch_to_device(mini, device);
        auto t_assem_end = std::chrono::steady_clock::now();
        result.assemble_seconds += std::chrono::duration<double>(
            t_assem_end - t_assem_start).count();
        // load_features_us = umbrella for (assemble + h→d). The
        // sample_read/active/assembler_kernel/edge/h2d sub-stages above
        // attribute the mass this umbrella would otherwise hide.
        bt.load_features_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_assem_end - t_assem_start).count();
        bt.h2d_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_assem_end - t_h2d_start).count();

        // Read-only isolation bench: the producer path
        // (read_sample + load_batch_features + GPU assemble) has already
        // run to produce `mini`. Skip the model forward/backward/optimizer
        // so the prefetch workers run UNTHROTTLED by GPU compute; the
        // per-epoch io_disk/epoch_t then measures the read+assemble path's
        // throughput when compute does not pace it.
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

        forward_backward_step_(mini, bt, epoch_loss_sum,
                               num_labeled_batches, result);

        ++num_train_batches;

        // Emit this batch's timing record. No-op when
        // profile_log_path was empty (profile_log_ is nullptr).
        if (profile_log_) {
            profile_log_->append(bt);
        }

        // Periodic fragmentation reclaim. Variable batch receptive
        // fields leak holes into the caching allocator pool; without
        // this call, deep-fanout runs on a 16 GB GPU have hit
        // fragmentation-driven OOM mid-epoch, well before any epoch
        // boundary where a full reclaim would otherwise run.
#ifdef GNN_CUDA_ENABLED
        if (config_.empty_cache_every_n_batches > 0 &&
            num_train_batches % config_.empty_cache_every_n_batches == 0 &&
            !device.is_cpu())
        {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
#endif
    }

    // Synchronize train_stream before validation reads any tensors.
    // evaluate() runs on the default stream (sequentially) and may read
    // tensors that were last written on train_stream by the dual-stream
    // overlap path.
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

    // Sync-avoidance optimization: single end-of-epoch sync to
    // read accumulated loss off-device. The .item() fires exactly once per epoch instead
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
    return avg_loss;
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

    // Seed from resume state (defaults to fresh training)
    double   best_val_acc     = config_.start_best_val;
    uint64_t patience_counter = config_.start_patience;

    // Test-at-best-val protocol: captured each time validation
    // strictly improves, when config_.track_test_at_best_val is on. Stays at
    // its sentinel (-1.0 / start_epoch) when the flag is off so the procedure
    // can tell "not tracked" from a genuine 0.0 test accuracy.
    double   best_val_test_acc   = -1.0;
    uint64_t best_val_epoch_seen = config_.start_epoch;

    // Per-epoch disk-traffic accounting: seed prev_disk before
    // the loop so the first epoch's delta == bytes accrued during epoch 0 (not
    // since process start). When provider is unset, prev/cur/delta stay zero
    // and the conditional print below is suppressed, preserving the original
    // line format (the one without the per-epoch disk-bytes column).
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
    print_training_banner_(device, train_batches, val_batches);

    // See resolve_effective_prefetch_workers_ for the once-per-train()
    // rationale and the FourLevelStore multi-worker provisioning rules.
    unsigned effective_workers = resolve_effective_prefetch_workers_();
    result.effective_prefetch_workers = effective_workers;

    for (uint64_t epoch = config_.start_epoch;
         epoch < config_.start_epoch + config_.epochs;
         ++epoch)
    {
        auto epoch_start = std::chrono::steady_clock::now();

        apply_cosine_lr_schedule_(epoch);

        // === Training phase ===
        double avg_loss = run_training_phase_(
            device, effective_workers, train_batches, result);
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

        // Flush the per-batch CSV at the epoch boundary so partial logs
        // survive an early termination (Ctrl-C, OOM, etc.) and post-hoc
        // analysis sees epoch-N records even if the process never reached
        // the destructor.
        if (profile_log_) {
            profile_log_->flush();
        }

        // Capture cumulative L3+L4 disk bytes post-validation so the delta
        // covers train + eval activity for this epoch. The single uint64_t
        // return keeps the Config callback ABI-stable.
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
            best_val_epoch_seen = epoch;
            // Checkpoint persistence is delegated to the on_epoch_end callback
            // (see AutoCheckpointer).

            // Test-at-best-val protocol: evaluate the test split
            // on the current (best-val) weights so the procedure can report
            // test-at-best-val (the reporting protocol common in GNN papers,
            // e.g. DiskGNN SIGMOD'25 §7.1) alongside the
            // test@final-epoch number. Skipped under read_only_bench (no
            // compute) and when there are no test batches.
            if (config_.track_test_at_best_val && !config_.read_only_bench
                && catalog_.test_batches > 0) {
                best_val_test_acc = evaluate(
                    train_batches + val_batches, catalog_.test_batches);
                std::cout << "[TrainingLoop]   test@best_val(epoch "
                          << (epoch + 1) << ")="
                          << std::fixed << std::setprecision(4)
                          << best_val_test_acc << std::endl;
            }
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
        // Per-epoch L3+L4 disk-traffic delta inline (bytes read from the
        // on-disk reordered feature matrix and per-batch packed store).
        // Suppressed when provider is unset (delta stays zero) or the delta
        // is < 1 MB (dominant case for small datasets where the line would
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
            finalize_result(result, best_val_acc, best_val_test_acc,
                            best_val_epoch_seen, wall_start);
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
                finalize_result(result, best_val_acc, best_val_test_acc,
                                best_val_epoch_seen, wall_start);
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

    result.ran_epochs = config_.epochs;
    finalize_result(result, best_val_acc, best_val_test_acc,
                    best_val_epoch_seen, wall_start);
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

    // Hoist device probe out of the inner loop. Mirrors train()'s device
    // detection — model_ is already on its final device by the time
    // train() reaches the validation phase (probe + .to() happens once at
    // the top of train()). When evaluate() is called standalone (e.g.,
    // tests) the model's first parameter still carries the canonical device.
    const torch::Device device = model_.parameters().begin()->device();

    // Mirror train()'s async prefetcher to hide validation-batch assemble +
    // host→device cost behind the model forward on GPU (pipeline overlap).
    // evaluate() is called once per epoch from train() AFTER the training
    // prefetcher has drained (last train batch consumed, no more prefetch()
    // calls issued), so the train prefetcher's worker is idle on its
    // req_queue cv and there is no concurrent BatchAssembler access. The
    // eval prefetcher owns its own worker that is destructed (join) at
    // scope exit.
    //
    // When dual CUDA streams are enabled, the eval prefetcher's worker runs
    // assemble on a pool stream and records a CUDAEvent into MiniBatch; this
    // thread uses a separate eval_stream and event.block()s before forward,
    // mirroring train()'s cross-stream sync block exactly.
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

    // Same Four-Level Feature Store worker guard as train(). See train()'s
    // comment for rationale. evaluate() is called once per epoch so the
    // warning would be spammy — we suppress it here (train() already emitted
    // it once on the first epoch).
    unsigned effective_eval_workers = config_.prefetch_num_workers;
    if (effective_eval_workers == 0) {
        effective_eval_workers = 1;
    }
    if (assembler_.uses_feature_store()) {
        // Reuse EXACTLY the per-worker IO count train() already provisioned
        // (fls_prefetch_workers_; 1 if multi-worker was not enabled). Never
        // exceed it — a worker id beyond the provisioned slots would fall
        // back to the shared primary DirectIoReader and race.
        // prepare_feature_store_workers() ran in train() before the epoch
        // loop (single-threaded), so the slots already exist here.
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

    // Infer split from start_batch using the catalog
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

        // Sub-stage timings propagated via mini.timing
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

        // Dual-stream CUDA overlap: cross-stream sync + record_stream BEFORE
        // any tensor reads on the eval stream. Mirrors train()'s cross-stream
        // sync block for the training loop.
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

        // Sync-avoidance optimization: guard on CPU-side
        // num_labeled (computed in BatchAssembler) to avoid the
        // `label_mask.any().item<bool>()` per-batch GPU→CPU sync. The remaining
        // `.sum().item<int64_t>()` collects validation accuracy — out of scope
        // for this sync-avoidance change.
        if (mini.num_labeled > 0) {
            auto masked_logits = logits.index({mini.label_mask});
            auto masked_labels = mini.labels.index({mini.label_mask});

            auto predicted = masked_logits.argmax(1);
            int64_t batch_correct = (predicted == masked_labels).sum().item<int64_t>();
            correct += batch_correct;
            total   += masked_labels.size(0);
        }

        // Emit this val/test batch's timing record. backward_us
        // stays 0 (no backward in eval). Guarded by profile_log_ so
        // tests / standalone evaluate() calls without an opt-in path are
        // unaffected.
        if (profile_log_) {
            profile_log_->append(bt);
        }
    }

    // Flush after the val/test loop so the eval records are durable
    // even if a later phase (e.g., callback, test cleanup) throws.
    if (profile_log_) {
        profile_log_->flush();
    }

    // Synchronize eval_stream before returning so any tensor reads following
    // this call (e.g., the next epoch's train prefetcher reading model state)
    // observe a coherent view. The per-batch .item<int64_t>() on `correct`
    // is already an implicit sync point, but the explicit synchronize mirrors
    // train()'s train_stream synchronize and survives future refactors that
    // might remove the .item() calls.
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
