#pragma once

#include <cstdint>
#include <vector>
#include <torch/torch.h>

#ifdef ENABLE_CUDA_ASSEMBLER
#include <ATen/cuda/CUDAEvent.h>
#endif

#include "gnn/sampling/graph_sample.h"  // for SplitType enum

namespace mdb::gnn {

/// Data contract between BatchAssembler and TrainingLoop.
/// Contains everything needed for one forward/backward pass.
///
/// Spec C3 stage 3: when populated by an AsyncBatchPrefetcher running with
/// use_cuda_streams=true, ready_event is recorded on the worker's stream
/// after assembly completes. Consumers (training loop) must call
/// ready_event.block(consumer_stream) before reading any GPU tensors,
/// otherwise they risk reading stale memory or racing with the worker.
struct MiniBatch {
    torch::Tensor features;                        // [N_batch, D] float32 — all nodes in subgraph
    std::vector<torch::Tensor> edge_indices;       // each [2, E_k] int64 — per GNN layer
    torch::Tensor labels;                          // [num_seeds] int64 — seed node labels
    torch::Tensor label_mask;                      // [num_seeds] bool — true if label != -1
    uint64_t num_seeds = 0;                        // target nodes (layer 0)
    uint64_t num_nodes = 0;                        // total nodes in computational subgraph
    SplitType split = SplitType::TRAIN;            // batch split assignment
    uint64_t batch_id = 0;                         // identifier for this batch

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
