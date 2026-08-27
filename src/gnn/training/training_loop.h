#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "gnn/models/graphsage_model.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/training/batch_assembler.h"
#include "gnn/training/batch_timing_log.h"

namespace mdb::gnn {

/**
 * @brief Orchestrates GNN training: epoch iteration, forward/backward,
 *        validation, and early stopping.
 *
 * Consumes MiniBatches from a BatchAssembler and drives a GraphSAGEModel
 * through supervised cross-entropy training.
 *
 * ## Early-stopping policy
 *
 * Two distinct termination conditions are tracked:
 *   1. **Convergence** — two consecutive epochs differ in average loss by
 *      less than `config.tolerance`.  Sets `Result::converged = true`.
 *   2. **Patience** — validation accuracy fails to improve for
 *      `config.patience` consecutive epochs.  Sets `Result::converged = false`
 *      and `Result::ran_epochs` reflects how many epochs actually ran.
 *
 * Per-epoch checkpoint persistence is delegated to an optional `on_epoch_end`
 * callback (see `Config::on_epoch_end` and `EpochEvent`). `TrainingLoop` no
 * longer performs any direct file I/O. `config.output_dir` is retained for
 * compatibility but is no longer consumed by this class.
 *
 * ## Batch layout contract (SampleCatalog)
 *
 * - Training batches:   batch_id in [0,               train_batches)
 * - Validation batches: batch_id in [train_batches,   train_batches + validation_batches)
 * - Test batches:       batch_id in [train_batches + validation_batches, total_batches)
 */
class TrainingLoop {
public:
    // =========================================================================
    // EpochEvent — emitted once per epoch via on_epoch_end callback
    // =========================================================================
    struct EpochEvent {
        uint64_t epoch;              ///< 0-indexed, absolute (includes resume offset)
        double   train_loss;         ///< average training loss this epoch
        double   val_accuracy;       ///< validation accuracy this epoch
        uint64_t patience_counter;   ///< patience state AFTER this epoch
        bool     is_best;            ///< val_accuracy > best_val_acc at this point
    };

    // =========================================================================
    // Configuration
    // =========================================================================
    struct Config {
        // Existing fields — semantics unchanged
        uint64_t    epochs        = 50;
        double      learning_rate = 0.01;
        double      weight_decay  = 0.0;
        double      tolerance     = 1e-4;
        uint64_t    patience      = 5;
        int64_t     random_seed   = -1;     ///< -1 = non-deterministic
        std::string output_dir;             ///< legacy: retained for existing tests

        // NEW — optional (defaults = fresh training, no callback)
        std::function<void(const EpochEvent&)> on_epoch_end;
        uint64_t             start_epoch     = 0;
        uint64_t             start_patience  = 0;
        double               start_best_val  = 0.0;
        std::vector<double>  seed_losses;

        // Optional cumulative-disk-bytes provider.
        // If set, train() invokes it once before the first epoch and once
        // at the end of each epoch, computes the per-epoch delta, and
        // prints the L3+L4 disk-traffic delta inline on the per-epoch
        // progress line. Returns a single uint64_t (not a struct) to keep
        // the Config struct's ABI stable across translation units —
        // CacheStatsSnapshot lives in a separate header that callers
        // can include without forcing TrainingLoop's TU to depend on
        // the four-level cache implementation.
        std::function<uint64_t()> cumulative_disk_bytes_provider;

        // Training pipeline overlap — async batch prefetcher.
        // When true, the inner train loop uses AsyncBatchPrefetcher to run
        // BatchAssembler::assemble() + host→device transfer on a background
        // thread, overlapping with model forward+backward. The validation
        // phase remains sequential. Queue size matches DiskGNN paper §6
        // default of 2.
        //
        // Default true: overlapping assembly with compute measured a ~1.6×
        // wall-clock speedup on a papers100M-scale sample, with accuracy
        // identical within noise. Set false only for debugging or to compare
        // against the legacy serial path.
        bool        use_async_prefetcher = true;
        size_t      prefetch_queue_size  = 2;

        // Number of background worker threads inside AsyncBatchPrefetcher.
        // Default 1: a single producer thread assembles each batch and
        // transfers it to the device while the main thread runs
        // forward+backward on the previous batch. Values >1 let multiple
        // BatchAssembler::assemble() calls run concurrently (CPU-side prep
        // work overlaps with the previous batch's GPU forward+backward).
        //
        // IMPORTANT: multi-worker mode is correct ONLY when the
        // BatchAssembler runs in FeatureMatrix-fallback mode. In
        // FourLevelStore (full) mode, shared state inside
        // load_batch_features (DirectIoReader io_uring rings + the
        // pinned_ptr_ host-pinned buffer reused across calls) makes
        // concurrent calls race silently. TrainingLoop enforces
        // num_workers=1 in that case via the runtime guard below.
        // See async_batch_prefetcher.h for full constraints.
        unsigned    prefetch_num_workers = 1;

        // Dual-CUDA-stream training overlap: split assemble_kernel and
        // model.forward+backward onto separate CUDA streams so the GPU can
        // execute them concurrently when SMs are free (DiskGNN SIGMOD'25
        // §5.3 "we run the model trainer and feature assembler on separate
        // CUDA streams to improve GPU utilization"). Uses c10::cuda::CUDAStream
        // pool acquisition and at::cuda::CUDAEvent for cross-stream
        // synchronization.
        //
        // Effective only when use_async_prefetcher=true AND CUDA is available.
        // Empirically measured as neutral on RTX 5070 Ti (1.014× over
        // async-prefetcher-only); default false because the assemble_kernel
        // is small (microseconds) and host-blocking PyTorch syncs consume
        // the concurrency window. Larger models or more SMs may benefit.
        bool        use_cuda_streams = false;

        // Periodic CUDA caching-allocator reclaim. When the receptive
        // field size varies across batches (e.g., fanout [10,15,20] on
        // UNDIRECTED orientation produces L3 layers swinging from 1M to
        // 3M nodes), the allocator pool fragments and "reserved but
        // unallocated" memory grows until the next alloc OOMs. Calling
        // c10::cuda::CUDACachingAllocator::emptyCache() periodically
        // releases unused blocks back to CUDA. Cost: ~1-2 ms per call on
        // RTX 5070 Ti; default 100 ⇒ ~24 ms overhead per 1200-batch
        // epoch (<0.05%). Set to 0 to disable.
        uint64_t empty_cache_every_n_batches = 100;

        // Profile instrumentation: when non-empty,
        // per-batch timings are persisted to this CSV path (train + val +
        // test). Disabled when empty (default). Captures: forward / backward
        // wall-times always; per-tier L1/L2/L3/L4/rmap sub-counters only on
        // the sequential (non-prefetcher) path where the FourLevelStore's
        // last_*_us() values can be safely attributed to the current batch.
        // Under the AsyncBatchPrefetcher (the default) tier
        // sub-counters are left at 0 — the prefetcher's worker has already
        // begun assembling batch N+1 by the time the consumer reads N, so
        // last_*_us() is racy/unattributable without MiniBatch-level
        // propagation (deferred refactor).
        std::string profile_log_path = "";

        // Read-only isolation bench. When true, each train batch
        // runs the full producer path (read_sample + load_batch_features +
        // GPU assemble, via the prefetcher when enabled) but SKIPS the model
        // forward/backward/optimizer and the validation phase. The prefetch
        // workers therefore run unthrottled by GPU compute, so the per-epoch
        // io_disk / epoch_t measures the read+assemble path's throughput when
        // compute does not pace it. This settles whether the in-train read is
        // limited by compute-pacing or by the read path itself, and yields a
        // number comparable to systems that benchmark their feature-load
        // stage in isolation.
        bool        read_only_bench = false;

        // Test-at-best-val protocol. When true, every time the
        // validation accuracy strictly improves (is_best), the test split is
        // ALSO evaluated and the value is captured into
        // Result::test_accuracy_at_best_val (the test accuracy of the model
        // weights at the best-validation epoch). This matches the DiskGNN
        // paper §7.1 reporting protocol ("test accuracy at the epoch with the
        // best validation accuracy"), as opposed to the test-at-final-epoch
        // number that gnn_train reports by default. ADDITIVE: the final-epoch
        // testAccuracy yield is unchanged, so existing bit-identical
        // regression gates are preserved. Default false to keep clean
        // benchmark timing (each improving epoch pays one extra test-split
        // eval pass); enable per-run via the trackTestAtBestVal procedure
        // parameter.
        bool        track_test_at_best_val = false;

        // Learning-rate schedule. "" = constant lr (default,
        // canonical). "cosine" = cosine annealing from learning_rate down to ~0
        // over [start_epoch, start_epoch+epochs): at relative epoch t of T total,
        // lr(t) = learning_rate * 0.5 * (1 + cos(pi * t / T)). Set on all Adam
        // param groups at the top of each epoch. Targets the late-epoch
        // generalization gap: with constant lr the validation accuracy
        // plateaus late in training and then drifts down, which annealing
        // avoids. Opt-in via the lrSchedule procedure parameter; default
        // "" preserves the canonical constant-lr trajectory and bit-identical
        // gates.
        std::string lr_schedule = "";
    };

    // =========================================================================
    // Result
    // =========================================================================

    struct Result {
        uint64_t             ran_epochs       = 0;
        bool                 converged        = false;
        double               best_val_accuracy= 0.0;
        std::vector<double>  epoch_losses;
        double               train_seconds    = 0.0;

        // Per-stage cumulative wall-time breakdown of the inner training loop.
        // All values are summed across all train batches across all epochs;
        // validation/eval time is NOT counted here (only the train phase).
        // Used to baseline the gain of training pipeline-overlap work.
        //
        // Invariant: assemble_seconds + forward_seconds + backward_seconds
        // ≈ time spent inside the inner train loop (i.e., excluding
        // validation, callbacks, and stoppage logic).
        //
        // - assemble_seconds: BatchAssembler::assemble() + per-batch
        //                     CPU→GPU device transfer
        // - forward_seconds:  model_.forward() (logits computation)
        // - backward_seconds: loss.backward() + optimizer.step()
        double               assemble_seconds = 0.0;
        double               forward_seconds  = 0.0;
        double               backward_seconds = 0.0;

        // The actual number of AsyncBatchPrefetcher workers used during
        // train(). Equals config.prefetch_num_workers unless (a) it was 0
        // (resolved to 1) or (b) the BatchAssembler runs in FourLevelStore
        // mode and N>1 was requested (clamped to 1; a stderr warning is also
        // emitted, see train() implementation).
        //
        // Surfaced from procedures (e.g., gnn_train) to make the multi-worker
        // activation diagnostic without parsing stderr.
        unsigned             effective_prefetch_workers = 1;

        // Track whether the v2 addr_table fast path was
        // used at least once during the run, and the mean per-batch cost of
        // reading+parsing the addr_table sidecar (μs).  Only attributable on
        // the sequential (non-prefetcher) path — see the sampling site in
        // training_loop.cc.  With the async prefetcher on, both fields stay at
        // their zero-initialized defaults (false / 0.0).
        bool                 addr_tables_used_ever    = false;
        double               addr_table_load_us_mean  = 0.0;
        uint64_t             addr_table_load_us_count = 0;

        // Test-at-best-val protocol. Populated only when
        // Config::track_test_at_best_val is true; otherwise stays at -1.0.
        // test_accuracy_at_best_val is the test-split accuracy of the model
        // weights at the epoch where validation accuracy last strictly
        // improved (the "best-val" epoch), i.e. the DiskGNN paper §7.1
        // reporting protocol. best_val_epoch is the 0-indexed absolute epoch
        // that produced best_val_accuracy. These are distinct from the
        // procedure's end-of-train testAccuracy (test at the FINAL epoch).
        double               test_accuracy_at_best_val = -1.0;
        uint64_t             best_val_epoch            = 0;
    };

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @param model     GraphSAGE model (must outlive TrainingLoop)
     * @param assembler BatchAssembler (must outlive TrainingLoop)
     * @param catalog   SampleCatalog describing train/val/test split counts
     * @param config    Hyper-parameters and stopping criteria
     */
    TrainingLoop(
        GraphSAGEModel&      model,
        BatchAssembler&      assembler,
        const SampleCatalog& catalog,
        torch::optim::Adam&  optimizer,       ///< owned by caller
        Config               config
    );

    // =========================================================================
    // Training
    // =========================================================================

    /**
     * @brief Run the full training loop.
     *
     * Iterates up to config.epochs epochs.  Each epoch:
     *   1. Forward/backward over all training batches (cross-entropy loss).
     *   2. Accuracy evaluation on validation batches.
     *   3. Early-stopping check (patience + convergence).
     *
     * @return Summary of the training run.
     */
    Result train();

    // =========================================================================
    // Evaluation
    // =========================================================================

    /**
     * @brief Evaluate classification accuracy over a contiguous range of batches.
     *
     * Runs under torch::NoGradGuard and temporarily sets the model to eval mode.
     * Restores training mode before returning.
     *
     * @param start_batch First batch id to evaluate.
     * @param count       Number of consecutive batches to evaluate.
     * @return Fraction of correctly classified labeled seed nodes in [0.0, 1.0].
     *         Returns 0.0 if no labeled seeds exist across all batches.
     */
    double evaluate(uint64_t start_batch, uint64_t count);

private:
    GraphSAGEModel&      model_;
    BatchAssembler&      assembler_;
    const SampleCatalog& catalog_;
    torch::optim::Adam&  optimizer_;
    Config               config_;

    // Per-batch CSV profile log. Constructed only when
    // config_.profile_log_path is non-empty; nullptr otherwise. Owned by
    // TrainingLoop; flushed on epoch boundaries and on destruction.
    std::unique_ptr<BatchTimingLog> profile_log_;

    // The FourLevelStore per-worker IO count that train() successfully
    // provisioned (via prepare_feature_store_workers). Each provisioned slot
    // receives its own DirectIoReader io_uring ring and pinned host buffer so
    // concurrent assemble() calls do not share mutable IO state. 1 =
    // single-worker / multi-worker not enabled. evaluate() reuses EXACTLY
    // this count for its own prefetcher so a validation worker can never
    // exceed the provisioned slots and fall back to the shared primary reader
    // (which would race). Set in train(); 1 if evaluate() is ever called
    // standalone.
    unsigned fls_prefetch_workers_ = 1;
};

} // namespace mdb::gnn
