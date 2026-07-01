// gnn_sampling_snapshot_test.cc
//
// Integration tests for multi-layer k-hop sampling determinism across the
// topology CSR sidecar fast-path (mmap-backed topology_{fwd,rev}.csr files
// that provide O(1) neighbor slices) and the B+Tree fallback path.
//
// Complements the single-layer `DeterministicSampleMatchesBpt` test in
// topology_accessor_csr_path_test.cc. Where that test locked down one call to
// `sample_neighbors`, this file exercises `sample_khop_neighbors` across
// realistic GraphSAGE fanout shapes and orientations so downstream training
// reproducibility is guarded end-to-end.
//
// Matrix covered (9 combinations possible, 7 implemented here):
//   Orientation {NATURAL, REVERSE, UNDIRECTED} × Fanouts {[5,5], [3,2], [2,2,1]}
// Plus:
//   - Partial-fallback scenario: topology_rev.csr is deleted, UNDIRECTED k-hop
//     sampling must still match a pure-BPT baseline.
//   - Out-of-range seed scenario: sampling should not crash and should produce
//     the same result as the pure-BPT path.
//
// Invariants asserted per test:
//   - Same number of layers produced.
//   - Per-layer: same src_nodes SET (sorted compare — iteration order across
//     `unordered_set` differs deterministically between builds but content is
//     bit-identical for the same RNG seed).
//   - Per-layer: same dst_nodes SET (sorted compare).
//   - Per-layer: same edge coverage — the multiset of (src_id, dst_id) pairs
//     obtained by mapping local edge_index indices back through src_nodes /
//     dst_nodes must match across paths. Local indexing differs in lock-step
//     with `unordered_set` iteration order so we compare GLOBAL ObjectId
//     pairs, not raw tensor contents.
//   - Per-layer: same num_src_nodes / num_dst_nodes.
//
// All sampling uses a fixed RNG seed (42) on both accessors prior to each call.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "gnn/projection/topology_accessor.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Process-lifetime System + ProjectionManager fixture. MDB's buffer pool can
// only be bound once per process, so all tests share the same singleton.
// ---------------------------------------------------------------------------
class MdbFixture {
public:
    static MdbFixture& instance() {
        static MdbFixture f;
        return f;
    }

    const std::string& db_folder() const { return db_folder_; }

private:
    MdbFixture() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        db_folder_ = "test_db_gnn_samp_snap_" + std::to_string(rng());
        fs::remove_all(db_folder_);

        system_.reset(new System(
            db_folder_,
            1024 * 1024,        // str_static_size
            1024 * 1024,        // str_dynamic_size
            64 * 1024 * 1024,   // shared_buffer_size
            32 * 1024 * 1024,   // private_buffer_size
            1024 * 1024,        // tensor_static_size
            1024 * 1024,        // tensor_dynamic_size
            1                   // workers
        ));

        query_ctx_.reset(new QueryContext());
        QueryContext::set_query_ctx(query_ctx_.get());

        auto& manager = GQL::ProjectionManager::get_instance();
        manager.init(db_folder_);
    }

    std::string                     db_folder_;
    std::unique_ptr<System>         system_;
    std::unique_ptr<QueryContext>   query_ctx_;
};

// ---------------------------------------------------------------------------
// Richer fixture graph (k-hop sampling needs several layers of branching).
//
// 16 nodes (0..15). Topology: a two-level tree rooted at 0 with extra
// cross-edges so out-degree varies from 0 to 5, reverse degree varies from
// 0 to 2, and every layer of a 3-hop expansion reaches at least one new
// node with fanout options large enough to force actual sampling decisions.
//
//   0 -> {1, 2, 3, 4, 5}      deg=5    (will sample fanout=2 or 3)
//   1 -> {6, 7, 8}            deg=3
//   2 -> {6, 9}               deg=2
//   3 -> {9, 10, 11}          deg=3
//   4 -> {10, 11, 12, 13}     deg=4
//   5 -> {13, 14, 15}         deg=3
//   6 -> {12, 14}             deg=2
//   7 -> {15}                 deg=1
//   8, 9, 10, 11, 12, 13, 14, 15 -> {} (leaves)
// ---------------------------------------------------------------------------
struct FixtureGraph {
    static constexpr uint64_t kNumNodes = 16;

    static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& edges() {
        static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> E = {
            // from 0 (degree 5)
            {0, 1, 1000}, {0, 2, 1001}, {0, 3, 1002}, {0, 4, 1003}, {0, 5, 1004},
            // from 1
            {1, 6, 1005}, {1, 7, 1006}, {1, 8, 1007},
            // from 2
            {2, 6, 1008}, {2, 9, 1009},
            // from 3
            {3, 9, 1010}, {3, 10, 1011}, {3, 11, 1012},
            // from 4
            {4, 10, 1013}, {4, 11, 1014}, {4, 12, 1015}, {4, 13, 1016},
            // from 5
            {5, 13, 1017}, {5, 14, 1018}, {5, 15, 1019},
            // from 6
            {6, 12, 1020}, {6, 14, 1021},
            // from 7
            {7, 15, 1022},
        };
        return E;
    }
};

// Build a projection, populate with FixtureGraph, flush, and optionally
// produce CSR sidecars (topology_fwd.csr / topology_rev.csr) via the
// test-only hook used by the topology snapshot builder integration tests.
//
// Returns (storage, projection_dir). Caller owns storage.
std::pair<std::unique_ptr<GQL::ProjectionStorage>, std::string>
build_fixture(const std::string& projection_name, bool produce_snapshots) {
    auto& manager = GQL::ProjectionManager::get_instance();
    std::string proj_dir = manager.create_projection(projection_name);

    auto storage = std::make_unique<GQL::ProjectionStorage>(
        proj_dir,
        MdbFixture::instance().db_folder(),
        projection_name);
    storage->init();

    for (uint64_t i = 0; i < FixtureGraph::kNumNodes; ++i) {
        GQL::ProjectedNode node;
        node.node_id = ObjectId(i);
        storage->add_node(node);
    }
    for (const auto& [from, to, eid] : FixtureGraph::edges()) {
        GQL::ProjectedEdge edge;
        edge.from_node   = ObjectId(from);
        edge.to_node     = ObjectId(to);
        edge.edge_id     = ObjectId(eid);
        edge.is_directed = true;
        storage->add_edge(edge);
    }

    storage->flush();

    if (produce_snapshots) {
        GQL::detail::build_topology_snapshots_for_test(
            *storage,
            /*build_forward=*/true,
            /*build_reverse=*/true);
    }

    return {std::move(storage), proj_dir};
}

// ---------------------------------------------------------------------------
// Helpers for path-invariant comparisons.
//
// Because `TopologyAccessor::sample_neighbors` builds its src_nodes vector by
// iterating an `std::unordered_set<uint64_t>`, the local indexing order is
// implementation-defined (though stable for a given libstdc++ build). Two
// accessors running against the same graph with the same RNG seed will pick
// the same set of neighbors but the vector order may permute. Therefore we
// compare SETS of ObjectId.id, and we compare edges AFTER resolving local
// indices back to global ObjectId pairs.
// ---------------------------------------------------------------------------
std::vector<uint64_t> sorted_ids(const std::vector<ObjectId>& v) {
    std::vector<uint64_t> out;
    out.reserve(v.size());
    for (auto id : v) {
        out.push_back(id.id);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Resolve a SampledSubgraph layer's edge_index tensor back to the multiset of
// (src_global_id, dst_global_id) pairs. This is the permutation-invariant
// quantity the two paths must match on.
//
// Out-of-range local indices would indicate a TopologyAccessor bug (not a
// test bug) and hard-abort via std::abort() so the test suite surfaces the
// fault clearly rather than silently masking it with a spurious pair.
std::vector<std::pair<uint64_t, uint64_t>>
global_edges(const mdb::gnn::SampledSubgraph& sg) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    const auto& t = sg.edge_index.edge_index;
    int64_t n = t.size(1);
    if (n == 0) {
        return out;
    }
    // Ensure tensor is on CPU + int64 contiguous before accessor.
    torch::Tensor cpu = t.detach().to(torch::kCPU).to(torch::kInt64).contiguous();
    auto a = cpu.accessor<int64_t, 2>();
    out.reserve(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        int64_t src_local = a[0][i];
        int64_t dst_local = a[1][i];
        if (src_local < 0 ||
            src_local >= static_cast<int64_t>(sg.src_nodes.size()) ||
            dst_local < 0 ||
            dst_local >= static_cast<int64_t>(sg.dst_nodes.size()))
        {
            std::abort();
        }
        uint64_t src_gid = sg.src_nodes[src_local].id;
        uint64_t dst_gid = sg.dst_nodes[dst_local].id;
        out.emplace_back(src_gid, dst_gid);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Produce k-hop samples on both the CSR-enabled accessor and the BPT-only
// accessor with the same fixed RNG seed, and assert path-equivalent output
// for every layer.
void expect_khop_matches(
    mdb::gnn::TopologyAccessor& acc_csr,
    mdb::gnn::TopologyAccessor& acc_bpt,
    const std::vector<ObjectId>& seeds,
    const std::vector<int64_t>& fanouts,
    mdb::gnn::EdgeOrientation orientation,
    const char* label)
{
    acc_csr.set_random_seed(42);
    acc_bpt.set_random_seed(42);

    auto csr_layers = acc_csr.sample_khop_neighbors(
        seeds, fanouts, mdb::gnn::SamplingStrategy::UNIFORM, orientation);
    auto bpt_layers = acc_bpt.sample_khop_neighbors(
        seeds, fanouts, mdb::gnn::SamplingStrategy::UNIFORM, orientation);

    ASSERT_EQ(csr_layers.size(), bpt_layers.size())
        << "[" << label << "] layer count mismatch";
    ASSERT_EQ(csr_layers.size(), fanouts.size())
        << "[" << label << "] sample_khop_neighbors must produce one layer per fanout";

    for (std::size_t li = 0; li < csr_layers.size(); ++li) {
        const auto& csr_l = csr_layers[li];
        const auto& bpt_l = bpt_layers[li];

        EXPECT_EQ(sorted_ids(csr_l.src_nodes), sorted_ids(bpt_l.src_nodes))
            << "[" << label << "] layer " << li << " src_nodes mismatch";
        EXPECT_EQ(sorted_ids(csr_l.dst_nodes), sorted_ids(bpt_l.dst_nodes))
            << "[" << label << "] layer " << li << " dst_nodes mismatch";

        EXPECT_EQ(csr_l.edge_index.num_src_nodes, bpt_l.edge_index.num_src_nodes)
            << "[" << label << "] layer " << li << " num_src_nodes mismatch";
        EXPECT_EQ(csr_l.edge_index.num_dst_nodes, bpt_l.edge_index.num_dst_nodes)
            << "[" << label << "] layer " << li << " num_dst_nodes mismatch";
        EXPECT_EQ(csr_l.edge_index.num_edges(), bpt_l.edge_index.num_edges())
            << "[" << label << "] layer " << li << " num_edges mismatch";

        // Compare permutation-invariant global edge multisets.
        auto csr_edges = global_edges(csr_l);
        auto bpt_edges = global_edges(bpt_l);
        EXPECT_EQ(csr_edges, bpt_edges)
            << "[" << label << "] layer " << li
            << " global edge (src_gid, dst_gid) multiset mismatch";
    }
}

}  // namespace

// ===========================================================================
// Test 1 — NATURAL orientation, 2-layer fanout [5, 5].
// ===========================================================================
TEST(GnnSamplingSnapshot, MultiLayerNaturalFanout55) {
    (void)MdbFixture::instance();

    auto [s_csr, d_csr] = build_fixture("samp_snap_nat_55_csr", /*produce_snapshots=*/true);
    auto [s_bpt, d_bpt] = build_fixture("samp_snap_nat_55_bpt", /*produce_snapshots=*/false);

    ASSERT_TRUE(fs::exists(fs::path(d_csr) / "topology_fwd.csr"));
    ASSERT_TRUE(fs::exists(fs::path(d_csr) / "topology_rev.csr"));
    ASSERT_FALSE(fs::exists(fs::path(d_bpt) / "topology_fwd.csr"));

    mdb::gnn::TopologyAccessor acc_csr(*s_csr);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Seed 0: under NATURAL, out-degree 5 → sample 5 (keeps all) → their
    // children (9 unique out-edges across children of 0). Layer 2 then
    // samples fanout=5 per src. Forces at least one real sampling decision
    // (0 has exactly 5 children so no subsampling at layer 0 — but nodes 1,
    // 3, 4 each have 3+ children, creating sampling work at layer 1).
    expect_khop_matches(
        acc_csr, acc_bpt,
        {ObjectId(0)},
        {5, 5},
        mdb::gnn::EdgeOrientation::NATURAL,
        "NATURAL,[5,5]");
}

// ===========================================================================
// Test 2 — REVERSE orientation, 2-layer fanout [5, 5].
// ===========================================================================
TEST(GnnSamplingSnapshot, MultiLayerReverseFanout55) {
    (void)MdbFixture::instance();

    auto [s_csr, d_csr] = build_fixture("samp_snap_rev_55_csr", /*produce_snapshots=*/true);
    auto [s_bpt, d_bpt] = build_fixture("samp_snap_rev_55_bpt", /*produce_snapshots=*/false);

    mdb::gnn::TopologyAccessor acc_csr(*s_csr);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Seeds 9, 10, 11, 12, 13, 14, 15 — leaves with multiple in-edges, so
    // REVERSE sampling actually selects among several predecessors.
    const std::vector<ObjectId> seeds = {
        ObjectId(9), ObjectId(10), ObjectId(11),
        ObjectId(12), ObjectId(13), ObjectId(14), ObjectId(15),
    };
    expect_khop_matches(
        acc_csr, acc_bpt,
        seeds,
        {5, 5},
        mdb::gnn::EdgeOrientation::REVERSE,
        "REVERSE,[5,5]");
}

// ===========================================================================
// Test 3 — UNDIRECTED orientation, 2-layer fanout [5, 5].
// ===========================================================================
TEST(GnnSamplingSnapshot, MultiLayerUndirectedFanout55) {
    (void)MdbFixture::instance();

    auto [s_csr, d_csr] = build_fixture("samp_snap_und_55_csr", /*produce_snapshots=*/true);
    auto [s_bpt, d_bpt] = build_fixture("samp_snap_und_55_bpt", /*produce_snapshots=*/false);

    mdb::gnn::TopologyAccessor acc_csr(*s_csr);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Seeds {6, 13}: both sit mid-graph with high undirected degree (6 has
    // out={12,14} + in={1,2} = 4 undirected neighbors; 13 has in={4,5} = 2
    // undirected neighbors). Picks both paths correctly.
    expect_khop_matches(
        acc_csr, acc_bpt,
        {ObjectId(6), ObjectId(13)},
        {5, 5},
        mdb::gnn::EdgeOrientation::UNDIRECTED,
        "UNDIRECTED,[5,5]");
}

// ===========================================================================
// Test 4 — NATURAL orientation, non-uniform fanout [3, 2].
// ===========================================================================
TEST(GnnSamplingSnapshot, MultiLayerNaturalFanout32) {
    (void)MdbFixture::instance();

    auto [s_csr, d_csr] = build_fixture("samp_snap_nat_32_csr", /*produce_snapshots=*/true);
    auto [s_bpt, d_bpt] = build_fixture("samp_snap_nat_32_bpt", /*produce_snapshots=*/false);

    mdb::gnn::TopologyAccessor acc_csr(*s_csr);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Seed 0 out-deg 5 → fanout 3 subsamples to 3 (real sampling work at
    // layer 0). Selected layer-1 seeds each have varying out-degree, so
    // fanout 2 exercises the fixed-point case (degree=2) AND the sampling
    // case (degree=3 or 4).
    expect_khop_matches(
        acc_csr, acc_bpt,
        {ObjectId(0)},
        {3, 2},
        mdb::gnn::EdgeOrientation::NATURAL,
        "NATURAL,[3,2]");
}

// ===========================================================================
// Test 5 — NATURAL orientation, 3-layer fanout [2, 2, 1].
// ===========================================================================
TEST(GnnSamplingSnapshot, ThreeLayerNaturalFanout221) {
    (void)MdbFixture::instance();

    auto [s_csr, d_csr] = build_fixture("samp_snap_nat_221_csr", /*produce_snapshots=*/true);
    auto [s_bpt, d_bpt] = build_fixture("samp_snap_nat_221_bpt", /*produce_snapshots=*/false);

    mdb::gnn::TopologyAccessor acc_csr(*s_csr);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Three-hop expansion from the root. Deepest GraphSAGE typical config.
    // Every layer exercises real sampling because the fanout (1 or 2) is
    // strictly less than the max degree encountered at that layer.
    expect_khop_matches(
        acc_csr, acc_bpt,
        {ObjectId(0)},
        {2, 2, 1},
        mdb::gnn::EdgeOrientation::NATURAL,
        "NATURAL,[2,2,1]");
}

// ===========================================================================
// Test 6 — REVERSE orientation, 3-layer fanout [2, 2, 1].
// Complements test 5 with the REVERSE traversal direction.
// ===========================================================================
TEST(GnnSamplingSnapshot, ThreeLayerReverseFanout221) {
    (void)MdbFixture::instance();

    auto [s_csr, d_csr] = build_fixture("samp_snap_rev_221_csr", /*produce_snapshots=*/true);
    auto [s_bpt, d_bpt] = build_fixture("samp_snap_rev_221_bpt", /*produce_snapshots=*/false);

    mdb::gnn::TopologyAccessor acc_csr(*s_csr);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Seeds {12, 14, 15}: leaves with multiple predecessors. 12 <- {4, 6};
    // 14 <- {5, 6}; 15 <- {5, 7}. Three hops of REVERSE walk reach node 0
    // through several distinct paths, maximising sampling coverage.
    expect_khop_matches(
        acc_csr, acc_bpt,
        {ObjectId(12), ObjectId(14), ObjectId(15)},
        {2, 2, 1},
        mdb::gnn::EdgeOrientation::REVERSE,
        "REVERSE,[2,2,1]");
}

// ===========================================================================
// Test 7 — UNDIRECTED with ONLY forward CSR on disk. The reverse sidecar is
//          deleted after build, forcing TopologyAccessor into mixed-path
//          mode: CSR for forward neighbor lookups, BPT for reverse ones.
//          The aggregate UNDIRECTED k-hop sample must still match the pure
//          BPT baseline exactly.
// ===========================================================================
TEST(GnnSamplingSnapshot, UndirectedPartialFallbackStillDeterministic) {
    (void)MdbFixture::instance();

    // Mixed-path accessor: build snapshots, then delete topology_rev.csr so
    // the reverse direction falls back to BPT while forward remains on CSR.
    auto [s_mix, d_mix] = build_fixture("samp_snap_und_partial_mix", /*produce_snapshots=*/true);
    {
        std::error_code ec;
        fs::remove(fs::path(d_mix) / "topology_rev.csr", ec);
        ASSERT_TRUE(fs::exists(fs::path(d_mix) / "topology_fwd.csr"));
        ASSERT_FALSE(fs::exists(fs::path(d_mix) / "topology_rev.csr"));
    }

    // Pure-BPT baseline: no snapshots at all.
    auto [s_bpt, d_bpt] = build_fixture("samp_snap_und_partial_bpt", /*produce_snapshots=*/false);

    mdb::gnn::TopologyAccessor acc_mix(*s_mix);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Seed 6 has both in={1,2} and out={12,14} so UNDIRECTED must query both
    // directions — the fast-forward/slow-reverse split is exercised fully.
    expect_khop_matches(
        acc_mix, acc_bpt,
        {ObjectId(6)},
        {3, 2},
        mdb::gnn::EdgeOrientation::UNDIRECTED,
        "UNDIRECTED-MIXED,[3,2]");
}

// ===========================================================================
// Test 8 — Seed list containing an out-of-range node id.
// The fast path must not crash; both paths must agree (the OOB seed yields
// empty neighbors on both, and sampling for the remaining valid seeds
// continues normally).
// ===========================================================================
TEST(GnnSamplingSnapshot, OutOfRangeSeedHandledGracefully) {
    (void)MdbFixture::instance();

    auto [s_csr, d_csr] = build_fixture("samp_snap_oob_csr", /*produce_snapshots=*/true);
    auto [s_bpt, d_bpt] = build_fixture("samp_snap_oob_bpt", /*produce_snapshots=*/false);

    mdb::gnn::TopologyAccessor acc_csr(*s_csr);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Mix real + fake seeds. 99999 is well beyond num_nodes=16.
    const std::vector<ObjectId> seeds = {
        ObjectId(0), ObjectId(99999), ObjectId(3),
    };

    // NATURAL k-hop from this mixed set must not crash and both paths must
    // agree. Note: RNG state is consumed identically on both sides because
    // the OOB seed yields an empty neighbor list in both code paths (same
    // zero RNG calls), so downstream valid-seed sampling stays in sync.
    expect_khop_matches(
        acc_csr, acc_bpt,
        seeds,
        {3, 2},
        mdb::gnn::EdgeOrientation::NATURAL,
        "NATURAL+OOB,[3,2]");
}
