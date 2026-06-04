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

        // Spec B2 (2026-04-27): optional cumulative-disk-bytes provider.
        // If set, train() invokes it once before the first epoch and once
        // at the end of each epoch, computes the per-epoch delta, and
        // prints the L3+L4 disk-traffic delta inline on the per-epoch
        // progress line. Returns a single uint64_t (not a struct) to keep
        // the Config struct's ABI stable across translation units —
        // CacheStatsSnapshot lives in a separate header that callers
        // can include without forcing TrainingLoop's TU to depend on
        // the four-level cache implementation.
        std::function<uint64_t()> cumulative_disk_bytes_provider;

        // Spec C3 stage 1 (delivered 2026-05-07): async batch prefetcher.
        // When true, the inner train loop uses AsyncBatchPrefetcher to run
        // BatchAssembler::assemble() + host→device transfer on a background
        // thread, overlapping with model forward+backward. The validation
        // phase remains sequential. queue size matches DiskGNN paper §6
        // default of 2.
        //
        // Default true since 2026-05-07: empirical 1.609× wall-clock
        // speedup measured on papers100M_caminoD_sample with bit-identical
        // accuracy (0.5942 vs 0.5940, within noise). Set false only for
        // debugging or to compare against the legacy serial path.
        bool        use_async_prefetcher = true;
        size_t      prefetch_queue_size  = 2;

        // Round 3B (2026-05-15): number of background worker threads inside
        // AsyncBatchPrefetcher. Default 1 preserves Stage 1 semantics
        // exactly. >1 lets multiple BatchAssembler::assemble() calls run
        // concurrently (CPU-side prep work overlaps with the previous
        // batch's GPU forward+backward).
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

        // Spec C3 stage 3 (started 2026-05-08): split assemble_kernel and
        // model.forward+backward onto separate CUDA streams so the GPU can
        // execute them concurrently when SMs are free (DiskGNN SIGMOD'25
        // §5.3 "we run the model trainer and feature assembler on separate
        // CUDA streams to improve GPU utilization").
        //
        // Effective only when use_async_prefetcher=true AND CUDA is available.
        // Default false until empirical validation completes; flip to true
        // once Module 6 confirms speedup + bit-identical accuracy.
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

        // Phase 0 (2026-05-17) profile instrumentation: when non-empty,
        // per-batch timings are persisted to this CSV path (train + val +
        // test). Disabled when empty (default). Captures: forward / backward
        // wall-times always; per-tier L1/L2/L3/L4/rmap sub-counters only on
        // the sequential (non-prefetcher) path where the FourLevelStore's
        // last_*_us() values can be safely attributed to the current batch.
        // Under the AsyncBatchPrefetcher (default-on since 2026-05-07) tier
        // sub-counters are left at 0 — the prefetcher's worker has already
        // begun assembling batch N+1 by the time the consumer reads N, so
        // last_*_us() is racy/unattributable without MiniBatch-level
        // propagation (deferred refactor).
        std::string profile_log_path = "";
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

        // Spec C3 stage 0 (2026-05-07): per-stage cumulative wall-time
        // breakdown of the inner training loop. All values are summed
        // across all train batches across all epochs; validation/eval
        // time is NOT counted here (only the train phase). Used to
        // baseline the gain of subsequent pipeline-overlap stages.
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

        // Round 3B (2026-05-15): the actual number of AsyncBatchPrefetcher
        // workers used during train(). Equals config.prefetch_num_workers
        // unless (a) it was 0 (resolved to 1) or (b) the BatchAssembler runs
        // in FourLevelStore mode and N>1 was requested (clamped to 1; a
        // stderr warning is also emitted, see train() implementation).
        //
        // Surface this from procedures (e.g., gnn_train) to make the
        // multi-worker activation diagnostic without parsing stderr.
        unsigned             effective_prefetch_workers = 1;

        // Path 4 (2026-05-19): Track whether the v2 addr_table fast path was
        // used at least once during the run, and the mean per-batch cost of
        // reading+parsing the addr_table sidecar (μs).  Only attributable on
        // the sequential (non-prefetcher) path — see the sampling site in
        // training_loop.cc.  With the async prefetcher on, both fields stay at
        // their zero-initialized defaults (false / 0.0).
        bool                 addr_tables_used_ever    = false;
        double               addr_table_load_us_mean  = 0.0;
        uint64_t             addr_table_load_us_count = 0;
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

    // Phase 0 (2026-05-17): per-batch CSV profile log. Constructed only when
    // config_.profile_log_path is non-empty; nullptr otherwise. Owned by
    // TrainingLoop; flushed on epoch boundaries and on destruction.
    std::unique_ptr<BatchTimingLog> profile_log_;

    // Round 3B-mw (2026-06-01): the FourLevelStore per-worker IO count that
    // train() successfully provisioned (via prepare_feature_store_workers).
    // 1 = single-worker / multi-worker not enabled. evaluate() reuses EXACTLY
    // this count for its own prefetcher so a validation worker can never
    // exceed the provisioned slots and fall back to the shared primary reader
    // (which would race). Set in train(); 1 if evaluate() is ever called
    // standalone.
    unsigned fls_prefetch_workers_ = 1;
};

} // namespace mdb::gnn
