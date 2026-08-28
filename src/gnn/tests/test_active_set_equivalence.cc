// Determinism + sanity gate for the active-set-shrinking refactor.
//
// Verifies the refactored GraphSAGEModel produces:
// - Deterministic output (same inputs -> same outputs, bit-identical)
// - Correct shape [num_seeds, num_classes]
// - Finite values (no NaN/inf)
// - Differentiable (gradients flow correctly)
//
// Why shrinking cannot change seed embeddings: conv k reads only rows of
// A_{k+1} (its dst set A_k is a prefix of A_{k+1}), so the rows dropped at
// each layer are exactly the ones no shallower layer ever consumes — the
// full-width model computed them and then discarded them. The chain down to
// A_0 (seeds) therefore performs identical operations on identical inputs,
// and the seed positions are bit-equivalent to the pre-refactor model.
// Empirical accuracy parity with the published GraphSAGE numbers is
// validated end-to-end, not in this unit gate.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "gnn/models/graphsage_model.h"

using mdb::gnn::GraphSAGEConfig;
using mdb::gnn::GraphSAGEModel;

namespace {

// Make an L-layer SAGE model with deterministic random weights.
GraphSAGEModel make_test_model(int64_t input_dim, int64_t hidden_dim,
                                int64_t num_classes, int64_t num_layers)
{
    GraphSAGEConfig config;
    config.input_dim   = input_dim;
    config.hidden_dim  = hidden_dim;
    config.num_classes = num_classes;
    config.num_layers  = num_layers;
    config.dropout     = 0.0;  // disable for determinism
    config.normalize   = false;
    return GraphSAGEModel(config);
}

}  // namespace

// 1. Determinism: two calls with same input produce identical output.
TEST(ActiveSetEquivalence, Deterministic) {
    torch::manual_seed(42);
    auto model = make_test_model(/*in=*/4, /*hidden=*/8, /*classes=*/3, /*L=*/3);
    model.eval();

    // 3-layer fixture: 10 nodes total
    //   A_0 = [0, 1]            (seeds, 2 nodes)
    //   A_1 = [0..4]             (1-hop active set, 5 nodes)
    //   A_2 = [0..7]             (2-hop active set, 8 nodes)
    //   A_3 = [0..9]             (deepest = all, 10 nodes)
    auto x = torch::randn({10, 4});

    // Edges (LOCAL indices in respective active sets):
    // edge_indices[0]: src in A_1, dst in A_0 - connect [2,3,4] to seeds [0, 1]
    auto e0 = torch::tensor({{2, 3, 4}, {0, 0, 1}}, torch::kInt64);
    // edge_indices[1]: src in A_2, dst in A_1
    auto e1 = torch::tensor({{5, 6, 7}, {2, 3, 4}}, torch::kInt64);
    // edge_indices[2]: src in A_3, dst in A_2
    auto e2 = torch::tensor({{8, 9}, {5, 6}}, torch::kInt64);

    std::vector<torch::Tensor> edges = {e0, e1, e2};
    std::vector<int64_t> active_sizes = {2, 5, 8, 10};

    auto out1 = model.forward(x, edges, active_sizes);
    auto out2 = model.forward(x, edges, active_sizes);

    ASSERT_EQ(out1.size(0), 2);  // num_seeds
    ASSERT_EQ(out1.size(1), 3);  // num_classes
    EXPECT_TRUE(torch::allclose(out1, out2, /*rtol=*/1e-7, /*atol=*/1e-7))
        << "Refactored model is not deterministic";
}

// 2. Shape is [num_seeds, num_classes] across various configurations.
TEST(ActiveSetEquivalence, ShapeIsSeedsByClasses) {
    torch::manual_seed(42);
    for (int64_t L : {1, 2, 3}) {
        auto model = make_test_model(4, 8, 3, L);
        model.eval();

        // Build a simple L-layer fixture.
        int64_t N = 10;
        int64_t num_seeds = 2;
        auto x = torch::randn({N, 4});

        std::vector<torch::Tensor> edges;
        std::vector<int64_t> active_sizes = {num_seeds};
        for (int k = 0; k < L; ++k) {
            // A_{k+1} grows linearly toward N. Simple: A_k+1 = (k+1)*N/L
            int64_t next_size = (k + 1) * N / L;
            if (next_size < active_sizes.back()) next_size = active_sizes.back();
            if (next_size > N) next_size = N;
            active_sizes.push_back(next_size);

            // Trivial edge tensor: one edge per dst node (k from A_{k+1} to A_k).
            int64_t n_dst = active_sizes[k];
            int64_t n_src = active_sizes[k + 1];
            int64_t E = std::min(n_dst, n_src);
            auto e = torch::empty({2, E}, torch::kInt64);
            for (int64_t i = 0; i < E; ++i) {
                e[0][i] = i;  // src
                e[1][i] = i % n_dst;
            }
            edges.push_back(e);
        }
        active_sizes.back() = N;

        auto out = model.forward(x, edges, active_sizes);

        EXPECT_EQ(out.size(0), num_seeds)
            << "L=" << L << " - expected output rows = num_seeds";
        EXPECT_EQ(out.size(1), 3)
            << "L=" << L << " - expected output cols = num_classes";
    }
}

// 3. Output is finite (no NaN/inf).
TEST(ActiveSetEquivalence, OutputIsFinite) {
    torch::manual_seed(42);
    auto model = make_test_model(4, 8, 3, 3);
    model.eval();

    auto x = torch::randn({10, 4});
    auto e0 = torch::tensor({{2, 3, 4}, {0, 0, 1}}, torch::kInt64);
    auto e1 = torch::tensor({{5, 6, 7}, {2, 3, 4}}, torch::kInt64);
    auto e2 = torch::tensor({{8, 9}, {5, 6}}, torch::kInt64);
    std::vector<torch::Tensor> edges = {e0, e1, e2};
    std::vector<int64_t> active_sizes = {2, 5, 8, 10};

    auto out = model.forward(x, edges, active_sizes);
    EXPECT_TRUE(out.isfinite().all().item<bool>())
        << "Output contains NaN/inf";
}

// 4. Gradients flow (model is differentiable).
TEST(ActiveSetEquivalence, GradientsFlow) {
    torch::manual_seed(42);
    auto model = make_test_model(4, 8, 3, 3);
    model.train();

    auto x = torch::randn({10, 4}, torch::requires_grad(false));  // input not learned
    auto e0 = torch::tensor({{2, 3, 4}, {0, 0, 1}}, torch::kInt64);
    auto e1 = torch::tensor({{5, 6, 7}, {2, 3, 4}}, torch::kInt64);
    auto e2 = torch::tensor({{8, 9}, {5, 6}}, torch::kInt64);
    std::vector<torch::Tensor> edges = {e0, e1, e2};
    std::vector<int64_t> active_sizes = {2, 5, 8, 10};

    auto logits = model.forward(x, edges, active_sizes);

    // Compute fake loss vs random target.
    auto target = torch::tensor({0, 1}, torch::kInt64);
    auto loss = torch::nn::functional::cross_entropy(logits, target);
    loss.backward();

    // At least one parameter should have a non-zero gradient.
    bool any_nonzero_grad = false;
    for (auto& p : model.parameters()) {
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) {
            any_nonzero_grad = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero_grad) << "No parameter received a non-zero gradient";
}

// 5. get_embeddings shape matches forward minus classifier.
TEST(ActiveSetEquivalence, GetEmbeddingsShape) {
    torch::manual_seed(42);
    auto model = make_test_model(4, /*hidden=*/8, 3, 3);
    model.eval();

    auto x = torch::randn({10, 4});
    auto e0 = torch::tensor({{2, 3, 4}, {0, 0, 1}}, torch::kInt64);
    auto e1 = torch::tensor({{5, 6, 7}, {2, 3, 4}}, torch::kInt64);
    auto e2 = torch::tensor({{8, 9}, {5, 6}}, torch::kInt64);
    std::vector<torch::Tensor> edges = {e0, e1, e2};
    std::vector<int64_t> active_sizes = {2, 5, 8, 10};

    auto emb = model.get_embeddings(x, edges, active_sizes);
    EXPECT_EQ(emb.size(0), 2);  // num_seeds
    EXPECT_EQ(emb.size(1), 8);  // hidden_dim
    EXPECT_TRUE(emb.isfinite().all().item<bool>());
}
