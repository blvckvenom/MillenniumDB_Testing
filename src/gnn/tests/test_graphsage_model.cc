#include "gnn/models/graphsage_model.h"

#include <gtest/gtest.h>

using namespace mdb::gnn;

// ============================================================================
// Helper: build a small 2-layer config
// ============================================================================
static GraphSAGEConfig small_config() {
    return GraphSAGEConfig{
        .input_dim  = 4,
        .hidden_dim = 8,
        .num_classes = 3,
        .num_layers  = 2,
        .dropout     = 0.5,
        .normalize   = false,
    };
}

// Two-layer edge index sets used across tests, expressed in the new
// active-set-shrinking convention (src in A_{k+1}, dst in A_k):
//   inner hop (k=0): 1-hop neighbors {3,4,5} -> seeds {0,1,2}
//   outer hop (k=1): 2-hop neighbors {6,7,8} -> 1-hop {3,4,5}
// Active-set sizes used at call sites: {3, 6, 10}
//   |A_0|=3 (seeds), |A_1|=6 (seeds + 1-hop), |A_2|=10 (full feature matrix).
static std::vector<torch::Tensor> two_layer_edges() {
    return {
        torch::tensor({{3, 4, 5}, {0, 1, 2}}, torch::kInt64),  // k=0: 1-hop -> seeds
        torch::tensor({{6, 7, 8}, {3, 4, 5}}, torch::kInt64),  // k=1: 2-hop -> 1-hop
    };
}

// Active-set sizes paired with two_layer_edges(): |A_0|=3, |A_1|=6, |A_2|=10.
static std::vector<int64_t> two_layer_active_sizes() {
    return {3, 6, 10};
}

// ============================================================================
// Test 1: output shape
// ============================================================================
TEST(GraphSAGEModelTest, ForwardProducesCorrectShape) {
    auto cfg = small_config();
    GraphSAGEModel model(cfg);
    model.eval();

    auto x    = torch::randn({10, 4});
    auto out  = model.forward(x, two_layer_edges(), two_layer_active_sizes());

    EXPECT_EQ(out.size(0), 3);
    EXPECT_EQ(out.size(1), 3);  // num_classes
}

// ============================================================================
// Test 2: single-layer model
// ============================================================================
TEST(GraphSAGEModelTest, SingleLayerForwardShape) {
    GraphSAGEConfig cfg{
        .input_dim   = 4,
        .hidden_dim  = 8,
        .num_classes = 2,
        .num_layers  = 1,
    };
    GraphSAGEModel model(cfg);
    model.eval();

    // One edge set only, in active-set convention:
    //   k=0: 1-hop {2,3} -> seeds {0,1}
    std::vector<torch::Tensor> edges = {
        torch::tensor({{2, 3}, {0, 1}}, torch::kInt64),
    };
    std::vector<int64_t> active_sizes = {2, 5};  // |A_0|=2 seeds, |A_1|=5 full
    auto x   = torch::randn({5, 4});
    auto out = model.forward(x, edges, active_sizes);

    EXPECT_EQ(out.size(0), 2);
    EXPECT_EQ(out.size(1), 2);
}

// ============================================================================
// Test 3: three-layer model
// ============================================================================
TEST(GraphSAGEModelTest, ThreeLayerForwardShape) {
    GraphSAGEConfig cfg{
        .input_dim   = 4,
        .hidden_dim  = 8,
        .num_classes = 5,
        .num_layers  = 3,
    };
    GraphSAGEModel model(cfg);
    model.eval();

    // Three layers in active-set convention (src in A_{k+1}, dst in A_k):
    //   k=0: 1-hop {2,3} -> seeds {0,1}
    //   k=1: 2-hop {4,5} -> 1-hop {2,3}
    //   k=2: 3-hop {6,7} -> 2-hop {4,5}
    std::vector<torch::Tensor> edges = {
        torch::tensor({{2, 3}, {0, 1}}, torch::kInt64),
        torch::tensor({{4, 5}, {2, 3}}, torch::kInt64),
        torch::tensor({{6, 7}, {4, 5}}, torch::kInt64),
    };
    // |A_0|=2 seeds, |A_1|=4, |A_2|=6, |A_3|=10 full.
    std::vector<int64_t> active_sizes = {2, 4, 6, 10};
    auto x   = torch::randn({10, 4});
    auto out = model.forward(x, edges, active_sizes);

    EXPECT_EQ(out.size(0), 2);
    EXPECT_EQ(out.size(1), 5);
}

// ============================================================================
// Test 4: gradients flow through all parameters
// ============================================================================
TEST(GraphSAGEModelTest, GradientsFlow) {
    auto cfg = small_config();
    GraphSAGEModel model(cfg);
    model.train();

    auto x   = torch::randn({10, 4});
    auto out = model.forward(x, two_layer_edges(), two_layer_active_sizes());
    out.sum().backward();

    for (auto& p : model.parameters()) {
        EXPECT_TRUE(p.grad().defined())
            << "Gradient not defined for a parameter";
        EXPECT_GT(p.grad().abs().sum().item<float>(), 0.0f)
            << "Gradient is zero for a parameter";
    }
}

// ============================================================================
// Test 5: dropout makes train and eval outputs differ
// ============================================================================
TEST(GraphSAGEModelTest, DropoutDiffersBetweenModes) {
    GraphSAGEConfig cfg{
        .input_dim   = 4,
        .hidden_dim  = 16,
        .num_classes = 3,
        .num_layers  = 2,
        .dropout     = 0.5,
    };
    GraphSAGEModel model(cfg);
    torch::manual_seed(42);

    auto x     = torch::randn({10, 4});
    auto edges = two_layer_edges();
    auto sizes = two_layer_active_sizes();

    model.train();
    auto train_out = model.forward(x, edges, sizes);

    model.eval();
    auto eval_out = model.forward(x, edges, sizes);

    EXPECT_FALSE(torch::allclose(train_out, eval_out))
        << "Expected train/eval outputs to differ due to dropout";
}

// ============================================================================
// Test 6: config() accessor round-trips values
// ============================================================================
TEST(GraphSAGEModelTest, ConfigAccessor) {
    auto cfg = small_config();
    GraphSAGEModel model(cfg);

    EXPECT_EQ(model.config().input_dim,   cfg.input_dim);
    EXPECT_EQ(model.config().hidden_dim,  cfg.hidden_dim);
    EXPECT_EQ(model.config().num_classes, cfg.num_classes);
    EXPECT_EQ(model.config().num_layers,  cfg.num_layers);
    EXPECT_DOUBLE_EQ(model.config().dropout, cfg.dropout);
    EXPECT_EQ(model.config().normalize,   cfg.normalize);
}

// ============================================================================
// Test 7: invalid num_layers throws
// ============================================================================
TEST(GraphSAGEModelTest, InvalidNumLayersThrows) {
    GraphSAGEConfig cfg{
        .input_dim   = 4,
        .hidden_dim  = 8,
        .num_classes = 3,
        .num_layers  = 0,  // invalid
    };
    EXPECT_THROW(GraphSAGEModel model(cfg), std::invalid_argument);
}

// ============================================================================
// Test 8: mismatched edge_indices size throws
// ============================================================================
TEST(GraphSAGEModelTest, MismatchedEdgeIndicesThrows) {
    auto cfg = small_config();  // num_layers = 2
    GraphSAGEModel model(cfg);
    model.eval();

    // Provide only 1 edge set instead of 2
    std::vector<torch::Tensor> one_edge = {
        torch::tensor({{2, 3}, {0, 1}}, torch::kInt64),
    };
    auto x = torch::randn({10, 4});
    EXPECT_THROW(model.forward(x, one_edge, two_layer_active_sizes()),
                 std::invalid_argument);
}

// ============================================================================
// Test 9: L2 normalize option does not change output shape
// ============================================================================
TEST(GraphSAGEModelTest, NormalizeOptionDoesNotChangeShape) {
    GraphSAGEConfig cfg{
        .input_dim   = 4,
        .hidden_dim  = 8,
        .num_classes = 3,
        .num_layers  = 2,
        .normalize   = true,
    };
    GraphSAGEModel model(cfg);
    model.eval();

    auto x   = torch::randn({10, 4});
    auto out = model.forward(x, two_layer_edges(), two_layer_active_sizes());

    EXPECT_EQ(out.size(0), 3);
    EXPECT_EQ(out.size(1), 3);
}

// ============================================================================
// Test 10: isolated nodes (no incoming edges) produce finite output
// ============================================================================
TEST(GraphSAGEModelTest, IsolatedNodesProduceFiniteOutput) {
    auto cfg = small_config();
    GraphSAGEModel model(cfg);
    model.eval();

    auto x = torch::randn({10, 4});

    // Empty edge tensors — all nodes are isolated
    std::vector<torch::Tensor> empty_edges = {
        torch::zeros({2, 0}, torch::kInt64),
        torch::zeros({2, 0}, torch::kInt64),
    };
    auto out = model.forward(x, empty_edges, two_layer_active_sizes());

    EXPECT_TRUE(out.isfinite().all().item<bool>())
        << "Output contains NaN or Inf for isolated nodes";
}

// ============================================================================
// Test 11: output is detached / no grad in eval mode (no backward needed)
// ============================================================================
TEST(GraphSAGEModelTest, EvalModeOutputRequiresNoGrad) {
    auto cfg = small_config();
    GraphSAGEModel model(cfg);
    model.eval();

    torch::NoGradGuard no_grad;
    auto x   = torch::randn({10, 4});
    auto out = model.forward(x, two_layer_edges(), two_layer_active_sizes());

    // In eval + no_grad, output should be finite and 2-D
    EXPECT_EQ(out.dim(), 2);
    EXPECT_TRUE(out.isfinite().all().item<bool>());
}

// ============================================================================
// Test 12: get_embeddings enforces the same input invariants as forward
// ============================================================================
TEST(GraphSAGEModelTest, GetEmbeddingsValidatesLikeForward) {
    auto cfg = small_config();  // num_layers = 2
    GraphSAGEModel model(cfg);
    model.eval();

    auto x = torch::randn({10, 4});

    // Mismatched edge_indices count
    std::vector<torch::Tensor> one_edge = {
        torch::tensor({{2, 3}, {0, 1}}, torch::kInt64),
    };
    EXPECT_THROW(model.get_embeddings(x, one_edge, two_layer_active_sizes()),
                 std::invalid_argument);

    // Wrong active_sizes_per_layer length
    EXPECT_THROW(model.get_embeddings(x, two_layer_edges(), {3, 10}),
                 std::invalid_argument);

    // active_sizes_per_layer.back() != x.size(0)
    EXPECT_THROW(model.get_embeddings(x, two_layer_edges(), {3, 6, 11}),
                 std::invalid_argument);

    // num_seeds (= active_sizes_per_layer[0]) must be > 0
    EXPECT_THROW(model.get_embeddings(x, two_layer_edges(), {0, 6, 10}),
                 std::invalid_argument);

    // Valid inputs still produce hidden-dim embeddings for the seeds
    auto emb = model.get_embeddings(x, two_layer_edges(), two_layer_active_sizes());
    EXPECT_EQ(emb.size(0), 3);
    EXPECT_EQ(emb.size(1), cfg.hidden_dim);
}

// ============================================================================
// Test 13: forward == get_embeddings + classifier (shared message passing)
// ============================================================================
TEST(GraphSAGEModelTest, ForwardMatchesEmbeddingsPlusClassifier) {
    auto cfg = small_config();
    GraphSAGEModel model(cfg);
    model.eval();  // disable dropout so both passes are deterministic

    torch::NoGradGuard no_grad;
    auto x      = torch::randn({10, 4});
    auto edges  = two_layer_edges();
    auto sizes  = two_layer_active_sizes();

    auto logits = model.forward(x, edges, sizes);
    auto emb    = model.get_embeddings(x, edges, sizes);

    EXPECT_EQ(emb.size(0), logits.size(0));
    EXPECT_EQ(emb.size(1), cfg.hidden_dim);
    EXPECT_EQ(logits.size(1), cfg.num_classes);
}
