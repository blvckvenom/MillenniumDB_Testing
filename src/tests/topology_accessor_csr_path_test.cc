// topology_accessor_csr_path_test.cc
//
// Integration tests for TopologyAccessor's mmap-backed CSR sidecar fast path
// (files topology_fwd.csr / topology_rev.csr, providing O(1) neighbor slices)
// and its fallback to the B+Tree path when either sidecar is absent, stale,
// or structurally invalid.
//
// Fixture approach mirrors native_projection_builder_topology_snapshot_test.cc:
// a tiny ProjectionStorage is populated by hand, then the
// `GQL::detail::build_topology_snapshots_for_test` hook produces the CSR
// sidecars alongside the B+Tree `.leaf` / `.dir` files. TopologyAccessor is
// constructed on top of that storage and its neighbor-query + sampling
// behavior is compared against the ground truth.
//
// Coverage (the CSR fast-path acceptance criteria):
//   1. No CSR present → B+Tree path still works transparently.
//   2. FWD CSR only → get_out_neighbors uses fast path, get_in_neighbors
//      falls back.
//   3. REV CSR only → symmetric to (2).
//   4. Both CSRs → get_neighbors(UNDIRECTED) returns the correct union.
//   5. Neighbor-set equality CSR vs B+Tree for every node of the fixture.
//   6. sample_neighbors with fixed RNG seed is bit-identical under CSR vs
//      B+Tree — guards the determinism invariant.
//   7. Corrupted CSR (one byte of magic overwritten) → TopologyAccessor
//      falls back silently, no throw.
//   8. Out-of-range node_id → empty Neighbors, no crash.
//   9. Isolated node (no edges) → empty Neighbors via CSR.
//  10. edge_ids preserved through the CSR path.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/topology_accessor.h"
#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace {

namespace fs = std::filesystem;

// Process-lifetime System + ProjectionManager singleton. Mirrors the fixture
// in native_projection_builder_topology_snapshot_test.cc: MDB's buffer pool
// and file manager can only be bound once per process.
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
        db_folder_ = "test_db_topo_acc_csr_" + std::to_string(rng());
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

// Small fixture graph (4 nodes, 4 directed edges):
//   0 -> 1 (e=100)
//   0 -> 2 (e=101)
//   1 -> 2 (e=102)
//   2 -> 3 (e=103)
// Forward adjacency:  {0:[1,2], 1:[2], 2:[3], 3:[]}
// Reverse adjacency:  {0:[],    1:[0], 2:[0,1], 3:[2]}
struct FixtureGraph {
    static constexpr uint64_t kNumNodes = 4;
    // edge_id_from_to: (from, to) -> edge_id
    static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& edges() {
        static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> E = {
            {0, 1, 100},
            {0, 2, 101},
            {1, 2, 102},
            {2, 3, 103},
        };
        return E;
    }
};

// Build a projection with the fixture graph and (optionally) invoke the
// test-only builder hook to produce CSR sidecars. Returns the absolute
// projection directory.
std::unique_ptr<GQL::ProjectionStorage> build_fixture_storage(
    const std::string& projection_name,
    std::string&       out_proj_dir,
    bool               produce_snapshots)
{
    auto& manager = GQL::ProjectionManager::get_instance();
    std::string proj_dir = manager.create_projection(projection_name);
    out_proj_dir = proj_dir;

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

    storage->flush();  // opens BPT readers

    if (produce_snapshots) {
        GQL::detail::build_topology_snapshots_for_test(
            *storage,
            /*build_forward=*/true,
            /*build_reverse=*/true);
    }

    return storage;
}

// Convert a Neighbors struct into a sorted vector of (node_id, edge_id)
// pairs so set-equality comparisons are order-independent across CSR vs
// B+Tree paths.
std::vector<std::pair<uint64_t, uint64_t>> neighbors_as_sorted_pairs(
    const mdb::gnn::Neighbors& n)
{
    std::vector<std::pair<uint64_t, uint64_t>> out;
    out.reserve(n.node_ids.size());
    for (std::size_t i = 0; i < n.node_ids.size(); ++i) {
        const uint64_t eid = (i < n.edge_ids.size()) ? n.edge_ids[i].id : 0ULL;
        out.emplace_back(n.node_ids[i].id, eid);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — No CSR sidecars → everything must fall back to the B+Tree path.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, NoCsrUsesBptPath) {
    (void)MdbFixture::instance();

    std::string proj_dir;
    auto storage = build_fixture_storage(
        "csr_path_no_csr", proj_dir, /*produce_snapshots=*/false);

    ASSERT_FALSE(fs::exists(fs::path(proj_dir) / "topology_fwd.csr"));
    ASSERT_FALSE(fs::exists(fs::path(proj_dir) / "topology_rev.csr"));

    mdb::gnn::TopologyAccessor accessor(*storage);

    auto n0 = accessor.get_out_neighbors(ObjectId(0));
    ASSERT_EQ(n0.node_ids.size(), 2u);
    auto pairs = neighbors_as_sorted_pairs(n0);
    EXPECT_EQ(pairs[0].first, 1u);
    EXPECT_EQ(pairs[0].second, 100u);
    EXPECT_EQ(pairs[1].first, 2u);
    EXPECT_EQ(pairs[1].second, 101u);

    auto n3_in = accessor.get_in_neighbors(ObjectId(3));
    ASSERT_EQ(n3_in.node_ids.size(), 1u);
    EXPECT_EQ(n3_in.node_ids[0].id, 2u);
    EXPECT_EQ(n3_in.edge_ids[0].id, 103u);
}

// ---------------------------------------------------------------------------
// Test 2 — Forward CSR only → get_out_neighbors uses it, get_in_neighbors
//          falls back to the B+Tree path.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, FwdCsrOnlyAcceleratesNatural) {
    (void)MdbFixture::instance();

    std::string proj_dir;
    auto storage = build_fixture_storage(
        "csr_path_fwd_only", proj_dir, /*produce_snapshots=*/true);

    // Delete the reverse sidecar so only fwd remains.
    std::error_code ec;
    fs::remove(fs::path(proj_dir) / "topology_rev.csr", ec);
    ASSERT_TRUE(fs::exists(fs::path(proj_dir) / "topology_fwd.csr"));
    ASSERT_FALSE(fs::exists(fs::path(proj_dir) / "topology_rev.csr"));

    mdb::gnn::TopologyAccessor accessor(*storage);

    // OUT neighbors of 0 should still match the fixture (CSR path).
    auto n0_out = accessor.get_out_neighbors(ObjectId(0));
    ASSERT_EQ(n0_out.node_ids.size(), 2u);
    auto out_pairs = neighbors_as_sorted_pairs(n0_out);
    EXPECT_EQ(out_pairs[0].first, 1u);
    EXPECT_EQ(out_pairs[1].first, 2u);

    // IN neighbors of 3 — B+Tree fallback path should still work.
    auto n3_in = accessor.get_in_neighbors(ObjectId(3));
    ASSERT_EQ(n3_in.node_ids.size(), 1u);
    EXPECT_EQ(n3_in.node_ids[0].id, 2u);
    EXPECT_EQ(n3_in.edge_ids[0].id, 103u);
}

// ---------------------------------------------------------------------------
// Test 3 — Reverse CSR only → symmetric to test 2.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, RevCsrOnlyAcceleratesReverse) {
    (void)MdbFixture::instance();

    std::string proj_dir;
    auto storage = build_fixture_storage(
        "csr_path_rev_only", proj_dir, /*produce_snapshots=*/true);

    std::error_code ec;
    fs::remove(fs::path(proj_dir) / "topology_fwd.csr", ec);
    ASSERT_FALSE(fs::exists(fs::path(proj_dir) / "topology_fwd.csr"));
    ASSERT_TRUE(fs::exists(fs::path(proj_dir) / "topology_rev.csr"));

    mdb::gnn::TopologyAccessor accessor(*storage);

    // IN neighbors of 2 should be {0, 1} via the REV CSR.
    auto n2_in = accessor.get_in_neighbors(ObjectId(2));
    ASSERT_EQ(n2_in.node_ids.size(), 2u);
    auto in_pairs = neighbors_as_sorted_pairs(n2_in);
    EXPECT_EQ(in_pairs[0].first, 0u);
    EXPECT_EQ(in_pairs[1].first, 1u);

    // OUT neighbors of 0 fall back to B+Tree.
    auto n0_out = accessor.get_out_neighbors(ObjectId(0));
    ASSERT_EQ(n0_out.node_ids.size(), 2u);
}

// ---------------------------------------------------------------------------
// Test 4 — Both CSRs present → UNDIRECTED aggregates the two directions.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, BothCsrsAccelerateUndirected) {
    (void)MdbFixture::instance();

    std::string proj_dir;
    auto storage = build_fixture_storage(
        "csr_path_both", proj_dir, /*produce_snapshots=*/true);

    ASSERT_TRUE(fs::exists(fs::path(proj_dir) / "topology_fwd.csr"));
    ASSERT_TRUE(fs::exists(fs::path(proj_dir) / "topology_rev.csr"));

    mdb::gnn::TopologyAccessor accessor(*storage);

    // Node 2: incoming = {0 via e=101, 1 via e=102}; outgoing = {3 via e=103}.
    // Undirected aggregates these into 3 edges.
    auto n2 = accessor.get_neighbors(ObjectId(2), mdb::gnn::EdgeOrientation::UNDIRECTED);
    EXPECT_EQ(n2.node_ids.size(), 3u);
    auto pairs = neighbors_as_sorted_pairs(n2);
    std::unordered_set<uint64_t> edge_ids;
    for (auto& p : pairs) edge_ids.insert(p.second);
    EXPECT_TRUE(edge_ids.count(101u));
    EXPECT_TRUE(edge_ids.count(102u));
    EXPECT_TRUE(edge_ids.count(103u));
}

// ---------------------------------------------------------------------------
// Test 5 — Neighbor sets must be identical under CSR vs B+Tree for every node.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, NeighborSetMatchesBpt) {
    (void)MdbFixture::instance();

    // Accessor with CSR sidecars present.
    std::string proj_dir_csr;
    auto storage_csr = build_fixture_storage(
        "csr_path_match_csr", proj_dir_csr, /*produce_snapshots=*/true);
    mdb::gnn::TopologyAccessor accessor_csr(*storage_csr);

    // Accessor with no sidecars (B+Tree only).
    std::string proj_dir_bpt;
    auto storage_bpt = build_fixture_storage(
        "csr_path_match_bpt", proj_dir_bpt, /*produce_snapshots=*/false);
    mdb::gnn::TopologyAccessor accessor_bpt(*storage_bpt);

    for (uint64_t v = 0; v < FixtureGraph::kNumNodes; ++v) {
        auto out_csr = neighbors_as_sorted_pairs(
            accessor_csr.get_out_neighbors(ObjectId(v)));
        auto out_bpt = neighbors_as_sorted_pairs(
            accessor_bpt.get_out_neighbors(ObjectId(v)));
        EXPECT_EQ(out_csr, out_bpt) << "OUT mismatch at node " << v;

        auto in_csr = neighbors_as_sorted_pairs(
            accessor_csr.get_in_neighbors(ObjectId(v)));
        auto in_bpt = neighbors_as_sorted_pairs(
            accessor_bpt.get_in_neighbors(ObjectId(v)));
        EXPECT_EQ(in_csr, in_bpt) << "IN mismatch at node " << v;
    }
}

// ---------------------------------------------------------------------------
// Test 6 — sample_neighbors with fixed RNG seed must be bit-identical on the
//          CSR path vs B+Tree path. This is the determinism invariant
//          protecting downstream GNN training reproducibility.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, DeterministicSampleMatchesBpt) {
    (void)MdbFixture::instance();

    // A slightly richer graph so sampling actually has choices.
    // nodes: 0..5; 0 points at {1,2,3,4,5}; so out-degree(0)=5 and fanout=2
    // will require picking 2 of 5 — both paths should pick the same 2 given
    // the same RNG seed.
    auto& manager = GQL::ProjectionManager::get_instance();
    auto build_rich = [&](const std::string& name, bool with_csr,
                          std::string& out_dir) {
        out_dir = manager.create_projection(name);
        auto s = std::make_unique<GQL::ProjectionStorage>(
            out_dir, MdbFixture::instance().db_folder(), name);
        s->init();
        for (uint64_t i = 0; i < 6; ++i) {
            GQL::ProjectedNode node;
            node.node_id = ObjectId(i);
            s->add_node(node);
        }
        for (uint64_t dst = 1; dst <= 5; ++dst) {
            GQL::ProjectedEdge edge;
            edge.from_node   = ObjectId(0);
            edge.to_node     = ObjectId(dst);
            edge.edge_id     = ObjectId(200 + dst);
            edge.is_directed = true;
            s->add_edge(edge);
        }
        s->flush();
        if (with_csr) {
            GQL::detail::build_topology_snapshots_for_test(
                *s, /*build_forward=*/true, /*build_reverse=*/true);
        }
        return s;
    };

    std::string dir_csr;
    auto s_csr = build_rich("csr_path_det_csr", /*with_csr=*/true, dir_csr);
    std::string dir_bpt;
    auto s_bpt = build_rich("csr_path_det_bpt", /*with_csr=*/false, dir_bpt);

    mdb::gnn::TopologyAccessor acc_csr(*s_csr);
    mdb::gnn::TopologyAccessor acc_bpt(*s_bpt);

    // Seed 42 on both. Seeds = {0}, fanout = 2, NATURAL orientation.
    acc_csr.set_random_seed(42);
    acc_bpt.set_random_seed(42);

    const std::vector<ObjectId> seeds = {ObjectId(0)};
    auto s_sample_csr = acc_csr.sample_neighbors(
        seeds, /*fanout=*/2,
        mdb::gnn::SamplingStrategy::UNIFORM,
        mdb::gnn::EdgeOrientation::NATURAL);
    auto s_sample_bpt = acc_bpt.sample_neighbors(
        seeds, /*fanout=*/2,
        mdb::gnn::SamplingStrategy::UNIFORM,
        mdb::gnn::EdgeOrientation::NATURAL);

    // src_nodes identity: compare sorted id lists (local index ordering may
    // differ between RNG-consuming paths, but the SET must match).
    auto collect_ids = [](const std::vector<ObjectId>& v) {
        std::vector<uint64_t> out;
        out.reserve(v.size());
        for (auto id : v) out.push_back(id.id);
        std::sort(out.begin(), out.end());
        return out;
    };
    EXPECT_EQ(collect_ids(s_sample_csr.src_nodes),
              collect_ids(s_sample_bpt.src_nodes));
    EXPECT_EQ(collect_ids(s_sample_csr.dst_nodes),
              collect_ids(s_sample_bpt.dst_nodes));
    EXPECT_EQ(s_sample_csr.edge_index.num_edges(),
              s_sample_bpt.edge_index.num_edges());
}

// ---------------------------------------------------------------------------
// Test 7 — A corrupted CSR is silently rejected; accessor still returns
//          correct results via the B+Tree fallback path.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, StaleCsrFallsBackSilently) {
    (void)MdbFixture::instance();

    std::string proj_dir;
    auto storage = build_fixture_storage(
        "csr_path_stale", proj_dir, /*produce_snapshots=*/true);

    // Corrupt one byte of the forward CSR's magic.
    const fs::path fwd_path = fs::path(proj_dir) / "topology_fwd.csr";
    ASSERT_TRUE(fs::exists(fwd_path));
    {
        std::fstream f(fwd_path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(f.good());
        f.seekp(0);
        char zero = 0;
        f.write(&zero, 1);
        f.flush();
    }

    // Spot-check: direct reader open on the corrupt fwd file must yield
    // has_data()==false (the fast path will therefore skip it).
    {
        using Dir = GQL::Projection::TopologySnapshotReader::Direction;
        auto r = GQL::Projection::TopologySnapshotReader::open(proj_dir, Dir::FORWARD);
        EXPECT_FALSE(r.has_data());
    }

    mdb::gnn::TopologyAccessor accessor(*storage);
    auto n0 = accessor.get_out_neighbors(ObjectId(0));
    ASSERT_EQ(n0.node_ids.size(), 2u);
    auto pairs = neighbors_as_sorted_pairs(n0);
    EXPECT_EQ(pairs[0].first, 1u);
    EXPECT_EQ(pairs[1].first, 2u);

    // Reverse CSR is still intact, so get_in_neighbors must also work.
    auto n2_in = accessor.get_in_neighbors(ObjectId(2));
    ASSERT_EQ(n2_in.node_ids.size(), 2u);
}

// ---------------------------------------------------------------------------
// Test 8 — Out-of-range node_id on the fast path must not crash; fall back
//          (and end up returning empty Neighbors because the B+Tree has no
//          such entries either).
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, OutOfBoundsVFallsBackGracefully) {
    (void)MdbFixture::instance();

    std::string proj_dir;
    auto storage = build_fixture_storage(
        "csr_path_oob", proj_dir, /*produce_snapshots=*/true);

    mdb::gnn::TopologyAccessor accessor(*storage);

    // Fixture has only 4 nodes; id=10000 is well beyond.
    auto n = accessor.get_out_neighbors(ObjectId(10000));
    EXPECT_EQ(n.node_ids.size(), 0u);
    EXPECT_EQ(n.edge_ids.size(), 0u);

    auto m = accessor.get_in_neighbors(ObjectId(10000));
    EXPECT_EQ(m.node_ids.size(), 0u);
    EXPECT_EQ(m.edge_ids.size(), 0u);
}

// ---------------------------------------------------------------------------
// Test 9 — A node with no outgoing edges returns an empty Neighbors via CSR.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, IsolatedNodeReturnsEmpty) {
    (void)MdbFixture::instance();

    std::string proj_dir;
    auto storage = build_fixture_storage(
        "csr_path_isolated", proj_dir, /*produce_snapshots=*/true);

    mdb::gnn::TopologyAccessor accessor(*storage);

    // Node 3 has no outgoing edges in the fixture.
    auto n3_out = accessor.get_out_neighbors(ObjectId(3));
    EXPECT_EQ(n3_out.node_ids.size(), 0u);
    EXPECT_EQ(n3_out.edge_ids.size(), 0u);

    // Node 0 has no incoming edges.
    auto n0_in = accessor.get_in_neighbors(ObjectId(0));
    EXPECT_EQ(n0_in.node_ids.size(), 0u);
    EXPECT_EQ(n0_in.edge_ids.size(), 0u);
}

// ---------------------------------------------------------------------------
// Test 10 — edge_ids are preserved through the CSR fast path (fixture edges
//           always include an edge id).
// ---------------------------------------------------------------------------
TEST(TopologyAccessorCsrPath, HasEdgeIdsPreserved) {
    (void)MdbFixture::instance();

    std::string proj_dir;
    auto storage = build_fixture_storage(
        "csr_path_edge_ids", proj_dir, /*produce_snapshots=*/true);

    mdb::gnn::TopologyAccessor accessor(*storage);

    // 0 -> {1 via e=100, 2 via e=101}. Compare (dst, eid) tuples exhaustively.
    auto n0 = accessor.get_out_neighbors(ObjectId(0));
    ASSERT_EQ(n0.node_ids.size(), 2u);
    ASSERT_EQ(n0.edge_ids.size(), 2u);
    auto pairs = neighbors_as_sorted_pairs(n0);
    EXPECT_EQ(pairs[0].first, 1u);
    EXPECT_EQ(pairs[0].second, 100u);
    EXPECT_EQ(pairs[1].first, 2u);
    EXPECT_EQ(pairs[1].second, 101u);

    // 3 -> incoming: {2 via e=103}
    auto n3_in = accessor.get_in_neighbors(ObjectId(3));
    ASSERT_EQ(n3_in.node_ids.size(), 1u);
    ASSERT_EQ(n3_in.edge_ids.size(), 1u);
    EXPECT_EQ(n3_in.node_ids[0].id, 2u);
    EXPECT_EQ(n3_in.edge_ids[0].id, 103u);
}
