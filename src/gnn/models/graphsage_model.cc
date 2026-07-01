#include "gnn/models/graphsage_model.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/nn/init.h>

#include "gnn/core/sparse_ops.h"
#include "misc/ablation_registry.h"

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

    // Opt-in (MDB_GNN_XAVIER_INIT=1): re-initialize the Linear weights with
    // Glorot/Xavier-uniform using the ReLU gain (gain = sqrt(2)) for the conv
    // layers and gain 1 for the linear classifier head. This mirrors DGL's
    // SAGEConv.reset_parameters, which applies
    // nn.init.xavier_uniform_(w, gain=calculate_gain("relu")) to its linear
    // weights (DGL, python/dgl/nn/pytorch/conv/sageconv.py); the bias zeroing
    // below is ours -- DGL's reset_parameters does not touch biases.
    //
    // Why the switch exists: PyTorch's Linear default draws weights from
    // U(-sqrt(k), sqrt(k)) with k = 1/fan_in (torch.nn.Linear documentation),
    // i.e. Var(W) = 1/(3*fan_in). Keeping activation variance constant across
    // ReLU layers instead wants Var(W) = 2/fan_in -- the Xavier bound
    // gain*sqrt(6/(fan_in+fan_out)) (Glorot & Bengio, AISTATS 2010) combined
    // with the ReLU gain sqrt(2) (He et al. 2015) -- which is 6x larger for a
    // square layer. Each ReLU layer multiplies activation variance by roughly
    // r = fan_in*Var(W)/2, so a per-layer deficit compounds geometrically
    // (r^L): two pipelines that differ only in this fixed choice can show a
    // systematic accuracy offset (not seed noise -- every run starts from the
    // same-scaled region) that grows with depth, invisible on 2-layer runs
    // but material at 3. Default (unset) preserves PyTorch's Linear default
    // exactly, so the bit-identical gates are unaffected.
    {
        // "1", "true" and "yes" are the only spellings this switch has ever
        // acted on and remain the accepted set, so an existing run keeps its
        // meaning. This one moves the weights a run starts from, so a value the
        // code declines has to be visible: a mistyped arm that silently trained
        // with the PyTorch default would be read as evidence about the init.
        static const bool xavier =
            Ablation::choice("MDB_GNN_XAVIER_INIT", "0", {"1", "true", "yes"}) != "0";
        if (xavier) {
            torch::NoGradGuard no_grad;
            const double relu_gain = std::sqrt(2.0);  // He/DGL ReLU gain
            for (auto& conv : convs_) {
                torch::nn::init::xavier_uniform_(conv->weight, relu_gain);
                if (conv->bias.defined()) torch::nn::init::zeros_(conv->bias);
            }
            torch::nn::init::xavier_uniform_(classifier_->weight, 1.0);
            if (classifier_->bias.defined())
                torch::nn::init::zeros_(classifier_->bias);
            std::cerr << "[GraphSAGEModel] weight init: xavier_uniform "
                         "(conv gain=sqrt(2) ReLU, classifier gain=1) "
                         "[MDB_GNN_XAVIER_INIT]\n";
        }
    }
}

// ============================================================================
// sage_conv — one MEAN aggregation layer
// ============================================================================

torch::Tensor GraphSAGEModel::sage_conv(
    torch::Tensor x,
    torch::Tensor edge_index,
    int64_t num_dst,
    torch::nn::Linear& linear)
{
    // Move edge_index to the same device as features.
    auto ei  = edge_index.to(x.device());
    auto src = ei[0];           // [E] — message sources, local in A_{k+1}
    auto dst = ei[1];           // [E] — message destinations, local in A_k

    // Gather neighbor features.
    auto neighbor_feat = x.index_select(0, src);                  // [E, D_in]

    // Sum-aggregate by destination. Output [num_dst, D_in]; positions never
    // indexed in `dst` stay zero (which is the SAGE-MEAN semantic for nodes
    // with no sampled neighbors at this hop).
    auto agg = ops::scatter_sum(neighbor_feat, dst, num_dst);

    // Mean normalization. clamp_min(1) handles isolated dst (agg stays 0).
    auto ones   = torch::ones({src.size(0), 1}, x.options());
    auto degree = ops::scatter_sum(ones, dst, num_dst).clamp_min(1.0);
    agg = agg / degree;

    // Self-feature: A_k is a prefix of A_{k+1} (BatchAssembler invariant),
    // so the first num_dst rows of x are exactly the self-features for A_k.
    auto x_self = x.slice(/*dim=*/0, /*start=*/0, /*end=*/num_dst);

    // Concatenate self + aggregated.
    auto combined = torch::cat({x_self, agg}, /*dim=*/1);         // [num_dst, 2*D_in]

    auto out = linear->forward(combined);                          // [num_dst, D_out]
    return out;
}

// ============================================================================
// Shared input validation — forward() and get_embeddings() enforce the same
// invariants; `who` names the entry point in error messages.
// ============================================================================

namespace {

void validate_message_passing_inputs(
    const char* who,
    const GraphSAGEConfig& config,
    const torch::Tensor& x,
    const std::vector<torch::Tensor>& edge_indices,
    const std::vector<int64_t>& active_sizes_per_layer)
{
    if ((int64_t)edge_indices.size() != config.num_layers) {
        throw std::invalid_argument(
            std::string(who) + ": edge_indices.size() must equal num_layers ("
            + std::to_string(config.num_layers) + "), got "
            + std::to_string(edge_indices.size()));
    }
    if ((int64_t)active_sizes_per_layer.size() != config.num_layers + 1) {
        throw std::invalid_argument(
            std::string(who) + ": active_sizes_per_layer.size() must be "
            "num_layers+1 (" + std::to_string(config.num_layers + 1) + "), got "
            + std::to_string(active_sizes_per_layer.size()));
    }
    if (active_sizes_per_layer.back() != x.size(0)) {
        throw std::invalid_argument(
            std::string(who) + ": active_sizes_per_layer.back() ("
            + std::to_string(active_sizes_per_layer.back())
            + ") must equal x.size(0) ("
            + std::to_string(x.size(0)) + ")");
    }
    const int64_t num_seeds = active_sizes_per_layer[0];
    if (num_seeds <= 0) {
        throw std::invalid_argument(
            std::string(who) + ": num_seeds (= active_sizes_per_layer[0]) "
            "must be > 0, got " + std::to_string(num_seeds));
    }
}

} // namespace

// ============================================================================
// forward — outside-in layer traversal
// ============================================================================

torch::Tensor GraphSAGEModel::forward(
    torch::Tensor x,
    const std::vector<torch::Tensor>& edge_indices,
    const std::vector<int64_t>& active_sizes_per_layer)
{
    validate_message_passing_inputs(
        "GraphSAGEModel::forward", config_, x, edge_indices, active_sizes_per_layer);

    // Message passing is shared with get_embeddings; after the final conv at
    // k=0, the result has shape [num_seeds, hidden_dim].
    auto hidden = get_embeddings(std::move(x), edge_indices, active_sizes_per_layer);

    // The classifier maps to num_classes.
    auto logits = classifier_->forward(hidden);   // [num_seeds, num_classes]
    return logits;
}

// ============================================================================
// get_embeddings — same as forward() but returns hidden representations
// ============================================================================

torch::Tensor GraphSAGEModel::get_embeddings(
    torch::Tensor x,
    const std::vector<torch::Tensor>& edge_indices,
    const std::vector<int64_t>& active_sizes_per_layer)
{
    validate_message_passing_inputs(
        "GraphSAGEModel::get_embeddings", config_, x, edge_indices, active_sizes_per_layer);

    // Process layers from outermost (convs_[num_layers-1], deepest applied)
    // down to innermost (convs_[0], seeds dst).
    // convs_[k] consumes A_{k+1} features and produces A_k features.
    for (int k = (int)convs_.size() - 1; k >= 0; k--) {
        const int64_t num_dst = active_sizes_per_layer[k];
        x = sage_conv(x, edge_indices[k], num_dst, convs_[k]);

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

    // After the final conv at k=0, x has shape [num_seeds, hidden_dim].
    return x;
}

} // namespace mdb::gnn
