#pragma once

#include <cstdint>
#include <vector>
#include <torch/torch.h>

#ifdef ENABLE_CUDA_ASSEMBLER
#include <ATen/cuda/CUDAEvent.h>
#endif

#include "gnn/sampling/graph_sample.h"  // for SplitType enum

namespace mdb::gnn {

/**
 * @brief Per-stage timings populated by BatchAssembler during assembly.
 *
 * Stored in nanoseconds internally to avoid sub-μs truncation of fast
 * stages (see Phase 0 μs-truncation bug fix c25ec330). TrainingLoop
 * divides by 1000 at the API boundary to write μs into BatchTiming.
 *
 * All fields default to 0 — populated only when the assembler chooses
 * to time its work. Zero means "not measured" (consistent with the
 * BatchTimingLog convention).
 *
 * h2d_ns is NOT stored here — it is measured by TrainingLoop around
 * the .to(device) call, which happens after the assembler returns.
 */
struct BatchTimingSubrecord {
    uint64_t sample_read_ns      = 0;  ///< samples_.read_sample(bid)
    uint64_t active_ns           = 0;  ///< build_active_indices(...)
    uint64_t assembler_kernel_ns = 0;  ///< feature_store->load_batch_features(sample)
    uint64_t edge_ns             = 0;  ///< build_edge_indices(...)

    /// v2 addr-table dispatch result for THIS
    /// batch, captured by BatchAssembler immediately after load_batch_features
    /// (while FourLevelStore::last_used_addr_tables() still refers to it) and
    /// carried with the batch. This makes the v2 telemetry correct on the async
    /// prefetcher path, where reading the FourLevelStore flags on the consumer
    /// thread would race against the worker's lookahead load.
    bool     used_addr_tables    = false;  ///< true if v2 fast path served it
    uint64_t addr_load_ns        = 0;      ///< addr_table sidecar open+parse time
};

/// Data contract between BatchAssembler and TrainingLoop.
/// Contains everything needed for one forward/backward pass.
///
/// Cross-stream synchronization: when populated by an AsyncBatchPrefetcher
/// running with use_cuda_streams=true, ready_event is recorded on the worker's stream
/// after assembly completes. Consumers (training loop) must call
/// ready_event.block(consumer_stream) before reading any GPU tensors,
/// otherwise they risk reading stale memory or racing with the worker.
struct MiniBatch {
    torch::Tensor features;                        // [N_batch, D] float32 — all nodes in subgraph
    std::vector<torch::Tensor> edge_indices;       // each [2, E_k] int64 — per GNN layer

    /**
     * @brief Per-layer active-set gather indices.
     *
     * `active_indices_per_layer[k]` is a 1-D Long tensor on CPU (later moved
     * to device by TrainingLoop) containing the global positions (into
     * `features`) of the nodes in `A_k = ∪_{j<=k} nodes_per_layer[j]`.
     *
     * Used by `GraphSAGEModel::forward` to shrink `x` between layers. The
     * tensor at index `K` (where K = num_layers) is the identity permutation
     * [0, 1, ..., N_total-1] since A_K = all_unique_nodes.
     *
     * Size: K+1 (one per "active set boundary"). For 3-layer SAGE this is
     * 4 tensors: A_0 (seeds), A_1 (L0 ∪ L1), A_2 (L0..L2), A_3 (all).
     */
    std::vector<torch::Tensor> active_indices_per_layer;

    /**
     * @brief Cached sizes of each active set (= active_indices_per_layer[k].size(0)).
     *
     * Stored separately so the model can pass int64_t sizes to scatter_sum
     * without invoking .size() on a tensor (which would be a host sync).
     *
     * Size: K+1.
     */
    std::vector<int64_t> active_sizes_per_layer;

    torch::Tensor labels;                          // [num_seeds] int64 — seed node labels
    torch::Tensor label_mask;                      // [num_seeds] bool — true if label != -1

    /// Identity of the seed nodes (layer 0), aligned with `labels`.
    /// seed_ids carries the raw node ids as stored in the sample (projection
    /// ObjectId bits); seed_rows carries the feature-matrix row of each seed
    /// (-1 when the node has no row). Consumed by the embedding exporter so
    /// external tools can join exported rows back to nodes/labels/splits.
    torch::Tensor seed_ids;                        // [num_seeds] int64
    torch::Tensor seed_rows;                       // [num_seeds] int64
    uint64_t num_seeds = 0;                        // target nodes (layer 0)
    uint64_t num_nodes = 0;                        // total nodes in computational subgraph
    /// Number of seeds with label != -1, computed
    /// on the CPU during assembly so the training loop can decide whether
    /// to run backward without a `.item<bool>()` GPU sync per batch.
    uint64_t num_labeled = 0;
    SplitType split = SplitType::TRAIN;            // batch split assignment
    uint64_t batch_id = 0;                         // identifier for this batch

    /// Phase A (2026-05-19): per-stage timings populated by BatchAssembler
    /// during assemble_from_sample(). Read by TrainingLoop into BatchTiming.
    /// Zero-initialized; "0" means "not measured by this assembler call".
    BatchTimingSubrecord timing;

#ifdef ENABLE_CUDA_ASSEMBLER
    /// Cross-stream sync handle. Empty (uncreated) when the producer used
    /// the default stream — consumer can ignore. When the prefetcher used
    /// a worker stream, ready_event.isCreated() is true and consumer should
    /// call ready_event.block(consumer_stream) before any GPU reads.
    /// CUDAEvent is move-only, so MiniBatch is also move-only on CUDA builds.
    at::cuda::CUDAEvent ready_event;
#endif
};

} // namespace mdb::gnn
