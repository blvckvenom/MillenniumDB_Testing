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
 * Forward convention (mini-batch, outside-in):
 *   - x            : node features [N, input_dim].  N includes seed + multi-hop neighbors.
 *   - edge_indices : per-layer edge tensors [2, E_k], k = 0..num_layers-1.
 *                    edge_indices[num_layers-1] connects outermost hop neighbors to
 *                    the next-inner layer; edge_indices[0] connects the 1-hop
 *                    neighborhood to seed nodes.
 *   - num_seeds    : first num_seeds rows of final x are the seed nodes.
 *
 * Returns logits [num_seeds, num_classes].
 */
class GraphSAGEModel : public torch::nn::Module {
public:
    explicit GraphSAGEModel(const GraphSAGEConfig& config);

    /**
     * @brief Run the full GraphSAGE forward pass.
     *
     * Processes layers from outermost (convs_[num_layers-1]) to innermost
     * (convs_[0]), then classifies only the first num_seeds rows.
     *
     * @param x            Node features [N, input_dim]
     * @param edge_indices Per-layer edge index tensors, length == num_layers
     * @param num_seeds    Number of seed nodes (first rows of x after all passes)
     * @return Logits [num_seeds, num_classes]
     */
    torch::Tensor forward(
        torch::Tensor x,
        const std::vector<torch::Tensor>& edge_indices,
        int64_t num_seeds
    );

    /**
     * @brief Run the GNN message-passing layers WITHOUT the final classifier.
     *
     * Same as forward() but returns the hidden-dim embeddings for seed nodes
     * instead of logits.  Used for embedding export after training.
     *
     * @param x            Node features [N, input_dim]
     * @param edge_indices Per-layer edge index tensors, length == num_layers
     * @param num_seeds    Number of seed nodes (first rows of x after all passes)
     * @return Embeddings [num_seeds, hidden_dim]
     */
    torch::Tensor get_embeddings(
        torch::Tensor x,
        const std::vector<torch::Tensor>& edge_indices,
        int64_t num_seeds
    );

    const GraphSAGEConfig& config() const { return config_; }

private:
    /**
     * @brief Single GraphSAGE MEAN convolution layer.
     *
     * Aggregates neighbor features by MEAN, concatenates with self features,
     * then applies a linear transformation.
     *
     * @param x          Current node features [N, D_in]
     * @param edge_index Edge index [2, E] — row 0: src, row 1: dst
     * @param linear     Weight matrix [2*D_in, D_out]
     * @return Updated features [N, D_out]
     */
    torch::Tensor sage_conv(
        torch::Tensor x,
        torch::Tensor edge_index,
        torch::nn::Linear& linear
    );

    std::vector<torch::nn::Linear> convs_;
    torch::nn::Linear classifier_{nullptr};
    GraphSAGEConfig config_;
};

} // namespace mdb::gnn
