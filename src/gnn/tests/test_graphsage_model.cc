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

// Two-layer edge index sets used across tests:
//   outer hop (k=1): nodes 3,4,5 send to nodes 6,7,8
//   inner hop (k=0): nodes 0,1,2 send to nodes 3,4,5
static std::vector<torch::Tensor> two_layer_edges() {
    return {
        torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt64),  // k=0
        torch::tensor({{3, 4, 5}, {6, 7, 8}}, torch::kInt64),  // k=1
    };
}

// ============================================================================
// Test 1: output shape
// ============================================================================
TEST(GraphSAGEModelTest, ForwardProducesCorrectShape) {
    auto cfg = small_config();
    GraphSAGEModel model(cfg);
    model.eval();

    auto x    = torch::randn({10, 4});
    auto out  = model.forward(x, two_layer_edges(), /*num_seeds=*/3);

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

    // One edge set only
    std::vector<torch::Tensor> edges = {
        torch::tensor({{0, 1}, {2, 3}}, torch::kInt64),
    };
    auto x   = torch::randn({5, 4});
    auto out = model.forward(x, edges, /*num_seeds=*/2);

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

    std::vector<torch::Tensor> edges = {
        torch::tensor({{0, 1}, {2, 3}}, torch::kInt64),
        torch::tensor({{2, 3}, {4, 5}}, torch::kInt64),
        torch::tensor({{4, 5}, {6, 7}}, torch::kInt64),
    };
    auto x   = torch::randn({10, 4});
    auto out = model.forward(x, edges, /*num_seeds=*/2);

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
    auto out = model.forward(x, two_layer_edges(), /*num_seeds=*/3);
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

    model.train();
    auto train_out = model.forward(x, edges, 3);

    model.eval();
    auto eval_out = model.forward(x, edges, 3);

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
        torch::tensor({{0, 1}, {2, 3}}, torch::kInt64),
    };
    auto x = torch::randn({10, 4});
    EXPECT_THROW(model.forward(x, one_edge, 3), std::invalid_argument);
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
    auto out = model.forward(x, two_layer_edges(), 3);

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
    auto out = model.forward(x, empty_edges, 3);

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
    auto out = model.forward(x, two_layer_edges(), 3);

    // In eval + no_grad, output should be finite and 2-D
    EXPECT_EQ(out.dim(), 2);
    EXPECT_TRUE(out.isfinite().all().item<bool>());
}
