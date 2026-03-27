#include "gnn/models/graphsage_model.h"

#include <stdexcept>
#include <string>

#include "gnn/core/sparse_ops.h"

namespace mdb::gnn {

// ============================================================================
// Constructor
// ============================================================================

GraphSAGEModel::GraphSAGEModel(const GraphSAGEConfig& config) : config_(config) {
    if (config.num_layers < 1) {
        throw std::invalid_argument("GraphSAGEConfig: num_layers must be >= 1");
    }
    if (config.input_dim <= 0 || config.hidden_dim <= 0 || config.num_classes <= 0) {
        throw std::invalid_argument(
            "GraphSAGEConfig: input_dim, hidden_dim and num_classes must be positive");
    }

    // convs_[k] is applied at iteration step k (k runs from num_layers-1 down to 0).
    //   k == num_layers-1 (first applied): input features have dim input_dim
    //   k <  num_layers-1 (subsequent):    input features have dim hidden_dim
    //
    // Each conv takes CONCAT(self, agg) → 2*D_in, produces hidden_dim.
    for (int64_t i = 0; i < config.num_layers; i++) {
        int64_t in_dim = (i == config.num_layers - 1)
                             ? config.input_dim
                             : config.hidden_dim;
        convs_.push_back(register_module(
            "conv_" + std::to_string(i),
            torch::nn::Linear(2 * in_dim, config.hidden_dim)));
    }

    classifier_ = register_module(
        "classifier",
        torch::nn::Linear(config.hidden_dim, config.num_classes));
}

// ============================================================================
// sage_conv — one MEAN aggregation layer
// ============================================================================

torch::Tensor GraphSAGEModel::sage_conv(
    torch::Tensor x,
    torch::Tensor edge_index,
    torch::nn::Linear& linear)
{
    auto src = edge_index[0];   // [E] — message sources
    auto dst = edge_index[1];   // [E] — message destinations
    int64_t N = x.size(0);

    // Gather neighbor features for each edge, then aggregate by destination.
    auto neighbor_feat = x.index_select(0, src);                     // [E, D_in]
    auto agg = ops::scatter_sum(neighbor_feat, dst, N);              // [N, D_in]

    // Normalize by degree (MEAN = SUM / count).  clamp_min(1) avoids divide-by-zero
    // for isolated nodes; they keep agg = 0.
    auto ones   = torch::ones({src.size(0), 1}, x.options());
    auto degree = ops::scatter_sum(ones, dst, N).clamp_min(1.0);    // [N, 1]
    agg = agg / degree;

    // Concatenate self embedding with aggregated neighborhood.
    auto combined = torch::cat({x, agg}, /*dim=*/1);                // [N, 2*D_in]

    return linear->forward(combined);                                // [N, D_out]
}

// ============================================================================
// forward — outside-in layer traversal
// ============================================================================

torch::Tensor GraphSAGEModel::forward(
    torch::Tensor x,
    const std::vector<torch::Tensor>& edge_indices,
    int64_t num_seeds)
{
    if ((int64_t)edge_indices.size() != config_.num_layers) {
        throw std::invalid_argument(
            "GraphSAGEModel::forward: edge_indices.size() must equal num_layers ("
            + std::to_string(config_.num_layers) + "), got "
            + std::to_string(edge_indices.size()));
    }
    if (num_seeds <= 0 || num_seeds > x.size(0)) {
        throw std::invalid_argument(
            "GraphSAGEModel::forward: num_seeds must be in [1, N], got "
            + std::to_string(num_seeds));
    }

    // Process layers from outermost to innermost.
    // convs_[num_layers-1] was registered for raw input features (2*input_dim → hidden_dim).
    // convs_[k < num_layers-1] expect hidden_dim inputs (2*hidden_dim → hidden_dim).
    for (int k = (int)convs_.size() - 1; k >= 0; k--) {
        x = sage_conv(x, edge_indices[k], convs_[k]);

        if (k > 0) {
            // Activation + regularization between hidden layers (not after last conv).
            x = torch::relu(x);
            if (is_training()) {
                x = torch::dropout(x, config_.dropout, /*train=*/true);
            }
            if (config_.normalize) {
                x = x / x.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-6);
            }
        }
    }

    // Classify only seed nodes (first num_seeds rows after all message-passing rounds).
    auto seed_embeddings = x.slice(/*dim=*/0, /*start=*/0, /*end=*/num_seeds);
    return classifier_->forward(seed_embeddings);   // [num_seeds, num_classes]
}

// ============================================================================
// get_embeddings — same as forward() but returns hidden representations
// ============================================================================

torch::Tensor GraphSAGEModel::get_embeddings(
    torch::Tensor x,
    const std::vector<torch::Tensor>& edge_indices,
    int64_t num_seeds)
{
    if ((int64_t)edge_indices.size() != config_.num_layers) {
        throw std::invalid_argument(
            "GraphSAGEModel::get_embeddings: edge_indices.size() must equal num_layers ("
            + std::to_string(config_.num_layers) + "), got "
            + std::to_string(edge_indices.size()));
    }
    if (num_seeds <= 0 || num_seeds > x.size(0)) {
        throw std::invalid_argument(
            "GraphSAGEModel::get_embeddings: num_seeds must be in [1, N], got "
            + std::to_string(num_seeds));
    }

    // Identical message-passing to forward()
    for (int k = (int)convs_.size() - 1; k >= 0; k--) {
        x = sage_conv(x, edge_indices[k], convs_[k]);

        if (k > 0) {
            x = torch::relu(x);
            if (is_training()) {
                x = torch::dropout(x, config_.dropout, /*train=*/true);
            }
            if (config_.normalize) {
                x = x / x.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-6);
            }
        }
    }

    // Return hidden-dim embeddings for seed nodes (skip the classifier)
    return x.slice(/*dim=*/0, /*start=*/0, /*end=*/num_seeds);  // [num_seeds, hidden_dim]
}

} // namespace mdb::gnn
