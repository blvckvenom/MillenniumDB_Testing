#pragma once

#include <cstdint>
#include <vector>
#include <torch/torch.h>
#include "gnn/sampling/graph_sample.h"  // for SplitType enum

namespace mdb::gnn {

/// Data contract between BatchAssembler and TrainingLoop.
/// Contains everything needed for one forward/backward pass.
struct MiniBatch {
    torch::Tensor features;                        // [N_batch, D] float32 — all nodes in subgraph
    std::vector<torch::Tensor> edge_indices;       // each [2, E_k] int64 — per GNN layer
    torch::Tensor labels;                          // [num_seeds] int64 — seed node labels
    torch::Tensor label_mask;                      // [num_seeds] bool — true if label != -1
    uint64_t num_seeds = 0;                        // target nodes (layer 0)
    uint64_t num_nodes = 0;                        // total nodes in computational subgraph
    SplitType split = SplitType::TRAIN;            // batch split assignment
    uint64_t batch_id = 0;                         // identifier for this batch
};

} // namespace mdb::gnn
