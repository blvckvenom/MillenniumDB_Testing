// gpu_khop_sampler_test.cc
//
// Statistical validation of the GPU k-hop sampling primitive (Phase 3). These
// exercise the reservoir + Philox kernel via the test-only entry
// gpu_sample_neighbors_for_test. They confirm the kernel honors the SAME
// distribution as the CPU sampler (uniform without replacement, k=min(fanout,deg),
// k==deg short-circuit) and is deterministic / launch-invariant — without
// requiring bit-identity to the CPU (Philox != mt19937_64).
//
// CI-friendly: when the build has no CUDA or no GPU is present at runtime, every
// test GTEST_SKIPs.

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/pinned_topology_view.h"
#include "gnn/sampling/gpu_khop_sampler.h"
#include "gnn/sampling/sampling_backend_plan.h"
#include "graph_models/object_id.h"

#ifdef GNN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace mdb::gnn {
namespace {

#ifdef GNN_CUDA_ENABLED
bool gpu_present() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}
#else
bool gpu_present() { return false; }
#endif

// Build a CSR from per-node adjacency lists. row_ptr[i] = prefix sum.
struct Csr {
    std::vector<uint64_t> row_ptr;
    std::vector<uint32_t> col_idx;
};

Csr make_csr(const std::vector<std::vector<uint32_t>>& adj) {
    Csr c;
    c.row_ptr.reserve(adj.size() + 1);
    c.row_ptr.push_back(0);
    for (const auto& nbrs : adj) {
        for (uint32_t n : nbrs) c.col_idx.push_back(n);
        c.row_ptr.push_back(c.col_idx.size());
    }
    return c;
}

#define SKIP_IF_NO_GPU()                                            \
    do {                                                            \
        if (!gpu_present()) GTEST_SKIP() << "no GPU at runtime";    \
    } while (0)

// ---------------------------------------------------------------------------

// (a) k == deg short-circuit: a node with degree <= fanout returns ALL its
// neighbors, in CSR order, drawing no randoms (so the result is identical for
// any batch_seed).
TEST(GpuKhopSampler, ShortCircuit_EmitsAllNeighborsNoDraw) {
    SKIP_IF_NO_GPU();
    Csr csr = make_csr({{10, 11, 12}});  // node 0: degree 3
    const int fanout = 5;                // fanout > degree => take all

    auto s1 = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx, {0}, fanout,
                                            /*batch_seed=*/123, /*layer=*/0);
    auto s2 = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx, {0}, fanout,
                                            /*batch_seed=*/999999, /*layer=*/0);
    ASSERT_EQ(s1.size(), 1u);
    EXPECT_EQ(s1[0], (std::vector<uint32_t>{10, 11, 12}));  // CSR order, all
    EXPECT_EQ(s1[0], s2[0]);  // no RNG => seed-independent
}

// A node with exactly degree == fanout also short-circuits (k == deg).
TEST(GpuKhopSampler, ExactlyFanout_EmitsAll) {
    SKIP_IF_NO_GPU();
    Csr csr = make_csr({{10, 11, 12, 13}});  // degree 4
    auto s = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx, {0},
                                           /*fanout=*/4, 7, 0);
    EXPECT_EQ(s[0], (std::vector<uint32_t>{10, 11, 12, 13}));
}

// Isolated node (degree 0) returns an empty sample.
TEST(GpuKhopSampler, IsolatedNode_Empty) {
    SKIP_IF_NO_GPU();
    Csr csr = make_csr({{}, {5, 6}});  // node 0 isolated
    auto s = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx, {0}, 5, 1, 0);
    EXPECT_TRUE(s[0].empty());
}

// (c) No replacement: a reservoir sample (deg > fanout) has no repeated id.
TEST(GpuKhopSampler, Reservoir_NoReplacement) {
    SKIP_IF_NO_GPU();
    std::vector<uint32_t> nbrs(100);
    for (uint32_t i = 0; i < 100; ++i) nbrs[i] = 1000 + i;
    Csr csr = make_csr({nbrs});
    const int fanout = 10;

    auto s = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx, {0}, fanout,
                                           42, 0);
    ASSERT_EQ(s[0].size(), static_cast<size_t>(fanout));
    std::set<uint32_t> uniq(s[0].begin(), s[0].end());
    EXPECT_EQ(uniq.size(), s[0].size());  // all distinct
    for (uint32_t v : s[0]) {
        EXPECT_GE(v, 1000u);
        EXPECT_LT(v, 1100u);  // every pick is a real neighbor
    }
}

// (d) Determinism + launch invariance: the sample of a node depends only on
// (batch_seed, node, layer), not on the frontier composition or grid layout.
TEST(GpuKhopSampler, Determinism_LaunchInvariant) {
    SKIP_IF_NO_GPU();
    std::vector<std::vector<uint32_t>> adj(2000);
    std::vector<uint32_t> nbrs(100);
    for (uint32_t i = 0; i < 100; ++i) nbrs[i] = 5000 + i;
    for (auto& a : adj) a = nbrs;  // every node has the same 100 neighbors
    Csr csr = make_csr(adj);
    const int fanout = 12;

    // Node 7 sampled alone.
    auto alone = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx, {7},
                                               fanout, /*seed=*/0xABCDEF, 1);
    // Node 7 embedded in a large frontier (spans many blocks => different grid).
    std::vector<uint32_t> big(2000);
    for (uint32_t i = 0; i < 2000; ++i) big[i] = i;
    auto crowd = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx, big,
                                               fanout, /*seed=*/0xABCDEF, 1);
    EXPECT_EQ(alone[0], crowd[7]);  // identical regardless of launch shape

    // Re-running the exact call reproduces it.
    auto again = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx, {7},
                                               fanout, 0xABCDEF, 1);
    EXPECT_EQ(alone[0], again[0]);

    // A different layer changes the draw (counter differs).
    auto other_layer = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx,
                                                     {7}, fanout, 0xABCDEF, 2);
    EXPECT_NE(alone[0], other_layer[0]);
}

// (b) Uniformity: over many independent samples (distinct node ids sharing the
// same neighbor list => independent Philox counters), each of the 100 candidates
// is selected with empirical frequency ~ fanout/degree = 0.1.
TEST(GpuKhopSampler, Reservoir_UniformSelectionFrequency) {
    SKIP_IF_NO_GPU();
    const uint32_t kTrials = 4000;
    const uint32_t kDeg = 100;
    const int      fanout = 10;
    std::vector<uint32_t> nbrs(kDeg);
    for (uint32_t i = 0; i < kDeg; ++i) nbrs[i] = 2000 + i;

    std::vector<std::vector<uint32_t>> adj(kTrials, nbrs);  // independent counters
    Csr csr = make_csr(adj);
    std::vector<uint32_t> frontier(kTrials);
    for (uint32_t i = 0; i < kTrials; ++i) frontier[i] = i;

    auto samples = gpu_sample_neighbors_for_test(csr.row_ptr, csr.col_idx,
                                                 frontier, fanout,
                                                 /*batch_seed=*/0x5151, 0);
    ASSERT_EQ(samples.size(), kTrials);

    std::vector<uint32_t> hits(kDeg, 0);
    for (const auto& s : samples) {
        EXPECT_EQ(s.size(), static_cast<size_t>(fanout));
        std::set<uint32_t> uniq(s.begin(), s.end());
        EXPECT_EQ(uniq.size(), s.size());  // no replacement, every trial
        for (uint32_t v : s) {
            ASSERT_GE(v, 2000u);
            ASSERT_LT(v, 2000u + kDeg);
            hits[v - 2000]++;
        }
    }

    const double expected = static_cast<double>(fanout) / kDeg;  // 0.1
    for (uint32_t c = 0; c < kDeg; ++c) {
        double freq = static_cast<double>(hits[c]) / kTrials;
        EXPECT_NEAR(freq, expected, 0.03) << "candidate " << c << " freq " << freq;
    }
}

// ---------------------------------------------------------------------------
// Orchestrator: sample_khop_gpu produces a GraphSample with the same SHAPE as
// the CPU sampler (seeds-first layer 0, valid local edge indices, first-
// appearance all_unique_nodes) — only the sampled neighbors differ (Philox).
// ---------------------------------------------------------------------------

TEST(GpuKhopSampler, Orchestrator_ProducesValidGraphSample) {
    SKIP_IF_NO_GPU();
    // CSR (forward / out-neighbors), tag-stripped dense ids:
    //   0->{2,3}  1->{3,4}  2->{5}  3->{5}  4->{}  5->{}
    // Reserve large capacity so each backing array gets its own mmap page
    // (the view's two cudaHostRegister calls must not share a 4 KiB page).
    std::vector<uint64_t> row_ptr;
    std::vector<uint32_t> col_idx;
    row_ptr.reserve(64 * 1024);
    col_idx.reserve(64 * 1024);
    Csr built = make_csr({{2, 3}, {3, 4}, {5}, {5}, {}, {}});
    row_ptr.assign(built.row_ptr.begin(), built.row_ptr.end());
    col_idx.assign(built.col_idx.begin(), built.col_idx.end());

    PinnedTopologyView view;
    HostCsrArrays fwd;
    fwd.row_ptr      = row_ptr.data();
    fwd.col_idx      = col_idx.data();
    fwd.n_rows       = row_ptr.size() - 1;  // 6
    fwd.n_edges      = col_idx.size();      // 6
    fwd.dst_type_tag = 0;
    view.build_and_register(fwd, HostCsrArrays{});
    ASSERT_TRUE(view.is_registered());

    SamplingBackendPlan plan;
    plan.backend    = SamplingBackend::GPU_UVA;
    plan.directions = GpuDirections::FORWARD_ONLY;

    std::vector<ObjectId> seeds{ObjectId(0), ObjectId(1)};
    GraphSample s = sample_khop_gpu(seeds, /*batch_id=*/7, SplitType::TRAIN,
                                    /*fanouts=*/{2, 2}, view, plan,
                                    /*random_seed=*/42);

    EXPECT_EQ(s.batch_id, 7u);
    EXPECT_EQ(s.split, SplitType::TRAIN);
    ASSERT_EQ(s.nodes_per_layer.size(), 3u);  // K+1 = 3
    // Layer 0 is the seeds verbatim.
    ASSERT_EQ(s.nodes_per_layer[0].size(), 2u);
    EXPECT_EQ(s.nodes_per_layer[0][0].id, 0u);
    EXPECT_EQ(s.nodes_per_layer[0][1].id, 1u);
    // Structural validity (edge indices within layer bounds).
    EXPECT_NO_THROW(s.validate());
    // all_unique_nodes is the dedup union, seeds first.
    EXPECT_EQ(s.all_unique_nodes[0].id, 0u);
    EXPECT_EQ(s.all_unique_nodes[1].id, 1u);
    std::set<uint64_t> uniq;
    for (const auto& n : s.all_unique_nodes) uniq.insert(n.id);
    EXPECT_EQ(uniq, (std::set<uint64_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_GT(s.total_edges(), 0u);

    // Determinism: same (seed, batch_id) reproduces the sample shape.
    GraphSample s2 = sample_khop_gpu(seeds, 7, SplitType::TRAIN, {2, 2}, view,
                                     plan, 42);
    EXPECT_EQ(s2.all_unique_nodes.size(), s.all_unique_nodes.size());
}

// ---------------------------------------------------------------------------
// Tiled COL_IDX windowing == whole-CSR sampling, bit-for-bit.
//
// The tiled seam pins ROW_PTR whole and streams COL_IDX in node-aligned windows.
// Because every node is sampled wholly within one window and the kernel's Philox
// draws key on (batch_seed, node, layer, j) over the node's own degree, the
// windowed picks must equal the whole-CSR picks element-for-element AND in order,
// for ANY window cap. We assert that across caps (incl cap=1 forcing the deg-50
// hub into its own window) x fanouts (reservoir, k==deg short-circuit, k>deg).
// ---------------------------------------------------------------------------

// deg = [2,1,50,0,3,4,1,7]; node 2 is a 50-edge super-hub, node 3 is isolated.
// Distinct neighbour ids per node so equality is meaningful (no coincidences).
// (Already inside the file's anonymous namespace -> internal linkage.)
inline Csr make_tiled_fixture_csr() {
    std::vector<std::vector<uint32_t>> adj(8);
    auto fill = [&](uint32_t u, uint32_t base, uint32_t deg) {
        for (uint32_t i = 0; i < deg; ++i) adj[u].push_back(base + i);
    };
    fill(0, 100, 2);
    fill(1, 200, 1);
    fill(2, 1000, 50);
    // node 3: isolated (deg 0)
    fill(4, 300, 3);
    fill(5, 400, 4);
    fill(6, 500, 1);
    fill(7, 700, 7);
    return make_csr(adj);
}

TEST(GpuKhopSampler, Tiled_EqualsWholeCsr) {
    SKIP_IF_NO_GPU();
    Csr csr = make_tiled_fixture_csr();
    // Shuffled full frontier — proves window partition + scatter-back preserves
    // per-frontier-index order regardless of input order.
    const std::vector<uint32_t> frontier{5, 2, 0, 7, 3, 1, 6, 4};
    const std::size_t caps[] = {1, 2, 3, 8, 1000};
    const int fanouts[] = {1, 2, 10, 60};
    for (int fanout : fanouts) {
        auto whole = gpu_sample_neighbors_for_test(
            csr.row_ptr, csr.col_idx, frontier, fanout,
            /*batch_seed=*/0xC0FFEEu, /*layer=*/1);
        for (std::size_t cap : caps) {
            auto tiled = gpu_sample_neighbors_tiled_for_test(
                csr.row_ptr, csr.col_idx, frontier, fanout,
                /*batch_seed=*/0xC0FFEEu, /*layer=*/1, cap);
            ASSERT_EQ(tiled.size(), whole.size())
                << "fanout=" << fanout << " cap=" << cap;
            for (std::size_t i = 0; i < whole.size(); ++i) {
                EXPECT_EQ(tiled[i], whole[i])
                    << "fanout=" << fanout << " cap=" << cap
                    << " frontier_idx=" << i << " node=" << frontier[i];
            }
        }
    }
}

// A frontier subset whose nodes land in DIFFERENT windows (the hub node 2 in its
// own window under a small cap, node 7 in another) still equals the whole-CSR
// result for that subset.
TEST(GpuKhopSampler, Tiled_FrontierSpanningWindows) {
    SKIP_IF_NO_GPU();
    Csr csr = make_tiled_fixture_csr();
    const std::vector<uint32_t> frontier{2, 7};
    for (int fanout : {3, 40}) {
        auto whole = gpu_sample_neighbors_for_test(
            csr.row_ptr, csr.col_idx, frontier, fanout, 0x5151u, 0);
        auto tiled = gpu_sample_neighbors_tiled_for_test(
            csr.row_ptr, csr.col_idx, frontier, fanout, 0x5151u, 0,
            /*window_cap_edges=*/3);
        ASSERT_EQ(tiled, whole) << "fanout=" << fanout;
    }
}

// Determinism + layer sensitivity through the tiled seam.
TEST(GpuKhopSampler, Tiled_Determinism) {
    SKIP_IF_NO_GPU();
    Csr csr = make_tiled_fixture_csr();
    const std::vector<uint32_t> frontier{2};  // the deg-50 hub draws randoms
    auto a = gpu_sample_neighbors_tiled_for_test(csr.row_ptr, csr.col_idx,
                                                 frontier, 10, 0xABCDEFu, 1, 8);
    auto b = gpu_sample_neighbors_tiled_for_test(csr.row_ptr, csr.col_idx,
                                                 frontier, 10, 0xABCDEFu, 1, 8);
    EXPECT_EQ(a, b);  // same (seed,node,layer) -> identical
    auto other_layer = gpu_sample_neighbors_tiled_for_test(
        csr.row_ptr, csr.col_idx, frontier, 10, 0xABCDEFu, 2, 8);
    EXPECT_NE(a, other_layer);  // layer enters the Philox counter
}

// Device-resident CSR == whole-UVA CSR: build the SAME small graph as a
// device-resident view (cudaMalloc'd VRAM) and as a host-registered (UVA) view,
// and assert sample_khop_gpu produces an identical GraphSample. Also assert the
// resident view's col_idx is REAL device memory (not a host-mapped pointer),
// which is what distinguishes the resident path from build_and_register.
TEST(GpuKhopSampler, Resident_EqualsWholeCsr) {
    SKIP_IF_NO_GPU();
    std::vector<uint64_t> row_ptr;
    std::vector<uint32_t> col_idx;
    row_ptr.reserve(64 * 1024);
    col_idx.reserve(64 * 1024);
    Csr built = make_csr({{2, 3}, {3, 4}, {5}, {5}, {}, {}});
    row_ptr.assign(built.row_ptr.begin(), built.row_ptr.end());
    col_idx.assign(built.col_idx.begin(), built.col_idx.end());

    HostCsrArrays fwd;
    fwd.row_ptr      = row_ptr.data();
    fwd.col_idx      = col_idx.data();
    fwd.n_rows       = row_ptr.size() - 1;
    fwd.n_edges      = col_idx.size();
    fwd.dst_type_tag = 0;

    SamplingBackendPlan plan;
    plan.backend    = SamplingBackend::GPU_VRAM_COPY;
    plan.directions = GpuDirections::FORWARD_ONLY;
    std::vector<ObjectId> seeds{ObjectId(0), ObjectId(1)};

    PinnedTopologyView resident;
    resident.build_and_register_resident(fwd, HostCsrArrays{});
    ASSERT_TRUE(resident.is_registered());
#ifdef GNN_CUDA_ENABLED
    cudaPointerAttributes attr{};
    ASSERT_EQ(cudaPointerGetAttributes(&attr, resident.fwd()->d_col_idx),
              cudaSuccess);
    EXPECT_EQ(attr.type, cudaMemoryTypeDevice);  // real VRAM, not host-mapped
#endif
    GraphSample rs = sample_khop_gpu(seeds, /*batch_id=*/7, SplitType::TRAIN,
                                     /*fanouts=*/{2, 2}, resident, plan,
                                     /*random_seed=*/42);
    EXPECT_NO_THROW(rs.validate());

    PinnedTopologyView uva;
    uva.build_and_register(fwd, HostCsrArrays{});
    ASSERT_TRUE(uva.is_registered());
    GraphSample us = sample_khop_gpu(seeds, 7, SplitType::TRAIN, {2, 2}, uva, plan,
                                     42);

    // Bit-identical: same kernel, same data, only the memory tier differs.
    ASSERT_EQ(rs.all_unique_nodes.size(), us.all_unique_nodes.size());
    for (size_t i = 0; i < us.all_unique_nodes.size(); ++i) {
        EXPECT_EQ(rs.all_unique_nodes[i].id, us.all_unique_nodes[i].id) << "i=" << i;
    }
    EXPECT_EQ(rs.total_edges(), us.total_edges());
}

}  // namespace
}  // namespace mdb::gnn
