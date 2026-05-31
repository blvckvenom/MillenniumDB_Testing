#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

namespace mdb::gnn {

/**
 * @brief Configuration for the GraphSAGE MEAN model.
 *
 * Controls network depth, width, regularization and optional L2 normalization
 * of per-layer embeddings.
 */
struct GraphSAGEConfig {
    int64_t input_dim;
    int64_t hidden_dim  = 256;
    int64_t num_classes;
    int64_t num_layers;
    double  dropout     = 0.5;
    bool    normalize   = false;  ///< L2 normalize per-layer output (PyG convention, default off)
};

/**
 * @brief GraphSAGE MEAN model with configurable depth and dropout.
 *
 * Implements the inductive representation learning approach from Hamilton et al.
 * (NeurIPS 2017) using MEAN aggregation across sampled neighborhoods.
 *
 * Architecture:
 *   - num_layers SAGEConv layers (each CONCAT self + MEAN neighbors → Linear)
 *   - ReLU + Dropout between hidden layers
 *   - Final linear classifier head producing logits
 *
 * Forward convention (mini-batch, outside-in, with active-set shrinking):
 *   - x                      : node features [|A_K|, input_dim] for the
 *                              deepest active set A_K (all unique nodes).
 *   - edge_indices           : per-layer edge tensors [2, E_k], k = 0..num_layers-1.
 *                              src local in A_{k+1}, dst local in A_k.
 *                              edge_indices[num_layers-1] connects outermost hop
 *                              neighbors to the next-inner layer; edge_indices[0]
 *                              connects the 1-hop neighborhood to seed nodes.
 *   - active_sizes_per_layer : K+1 monotone-nondecreasing sizes (|A_0|, ..., |A_K|),
 *                              where active_sizes_per_layer[K] == x.size(0).
 *                              The first |A_0| rows of x are the seeds and form a
 *                              prefix of every A_k (BatchAssembler invariant).
 *
 * Returns logits [num_seeds, num_classes] where num_seeds = active_sizes_per_layer[0].
 */
class GraphSAGEModel : public torch::nn::Module {
public:
    explicit GraphSAGEModel(const GraphSAGEConfig& config);

    /**
     * @brief Run the full GraphSAGE forward pass with active-set shrinking.
     *
     * Processes layers from outermost (convs_[num_layers-1]) down to innermost
     * (convs_[0]). At each conv, the dst active-set size = active_sizes_per_layer[k]
     * is passed to sage_conv, which produces output sized to that. After the
     * final conv at k=0, x has shape [num_seeds, hidden_dim].
     *
     * @param x                          Features for the deepest active set
     *                                   A_K (= all_unique_nodes), shape
     *                                   [|A_K|, input_dim].
     * @param edge_indices               Per-layer edges, length == num_layers.
     *                                   src local in A_{k+1}, dst local in A_k.
     * @param active_sizes_per_layer     K+1 sizes (= |A_0|, |A_1|, ..., |A_K|).
     *                                   active_sizes_per_layer[K] must equal
     *                                   x.size(0).
     * @return Logits [num_seeds, num_classes] where num_seeds = active_sizes_per_layer[0].
     */
    torch::Tensor forward(
        torch::Tensor x,
        const std::vector<torch::Tensor>& edge_indices,
        const std::vector<int64_t>& active_sizes_per_layer
    );

    /**
     * @brief Same as forward() but returns hidden-dim embeddings (no classifier).
     *
     * @param x                          Features [|A_K|, input_dim].
     * @param edge_indices               Per-layer edges, length == num_layers.
     * @param active_sizes_per_layer     K+1 sizes.
     * @return Embeddings [num_seeds, hidden_dim].
     */
    torch::Tensor get_embeddings(
        torch::Tensor x,
        const std::vector<torch::Tensor>& edge_indices,
        const std::vector<int64_t>& active_sizes_per_layer
    );

    const GraphSAGEConfig& config() const { return config_; }

private:
    /**
     * @brief One MEAN-aggregation sage layer with active-set output.
     *
     * @param x           [|A_{k+1}|, D_in] features for the src active set.
     * @param edge_index  [2, E] with src in [0, |A_{k+1}|), dst in [0, |A_k|).
     * @param num_dst     = |A_k|. Output size at dim 0.
     * @param linear      The layer's Linear module (2*D_in -> D_out).
     * @return            [|A_k|, D_out].
     *
     * Self-feature: uses x.slice(0, 0, num_dst) — relies on the prefix
     * invariant established by BatchAssembler::build_active_indices.
     */
    torch::Tensor sage_conv(
        torch::Tensor x,
        torch::Tensor edge_index,
        int64_t num_dst,
        torch::nn::Linear& linear
    );

    std::vector<torch::nn::Linear> convs_;
    torch::nn::Linear classifier_{nullptr};
    GraphSAGEConfig config_;
};

} // namespace mdb::gnn
