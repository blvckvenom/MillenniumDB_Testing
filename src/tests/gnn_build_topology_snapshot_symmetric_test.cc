// Unit tests for the post-hoc symmetric topology bake
// (build_symmetric_snapshot_post_hoc, exposed via
// GnnBuildTopologySnapshotProcedure::run_symmetric_for_test).
//
// The bake merges each node's out+in neighbor lists into one pre-merged
// undirected CSR (topology_sym.csr, magic "TOPOSYM1", no EDGE_IDS), replicating
// the accessor's canonical UNDIRECTED emission. Two invariants:
//   1. Every baked row equals the LIVE accessor UNDIRECTED node list (the same
//      oracle the in-build self-verify uses) — byte-identical dst sequence.
//   2. A parallel-edge multigraph is REFUSED (no file, refused flag set).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/edge_orientation.h"
#include "gnn/projection/topology_accessor.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot.h"
#include "graph_models/object_id.h"
#include "query/procedure/builtin/gnn_build_topology_snapshot_procedure.h"

#include "gnn_projection_test_fixture.h"

namespace fs = std::filesystem;
using gnn_test_fixture::build_small_projection;
using gnn_test_fixture::MdbFixture;
using gnn_test_fixture::open_projection;
using Proc = GQL::Procedures::GnnBuildTopologySnapshotProcedure;

TEST(SymmetricBake, BakedRowsMatchLiveUndirected) {
    (void)MdbFixture::instance();
    const std::string dir = build_small_projection("sym_bake_match");
    {
        auto storage = open_projection(dir);
        auto [bytes, ms, refused] = Proc::run_symmetric_for_test(*storage);
        EXPECT_GT(bytes, 64u);
        EXPECT_GE(ms, 0);
        EXPECT_FALSE(refused);
    }

    // Raw-read the baked symmetric sidecar (directional reader rejects TOPOSYM1)
    // and compare each row to the live accessor UNDIRECTED node list.
    auto storage = open_projection(dir);
    mdb::gnn::TopologyAccessor acc(*storage);

    std::ifstream f(fs::path(dir) / "topology_sym.csr", std::ios::binary);
    ASSERT_TRUE(f.good());
    uint8_t hdr[GQL::Projection::kTopologySnapshotHeaderSize];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    ASSERT_EQ(0, std::memcmp(hdr, GQL::Projection::kTopologySnapshotSymMagic.data(), 8));
    // No EDGE_IDS section in the symmetric sidecar.
    EXPECT_EQ(0u, hdr[13] & GQL::Projection::TopologySnapshotFlags::kHasEdgeIds);

    uint64_t N = 0;
    std::memcpy(&N, hdr + 16, 8);
    ASSERT_EQ(N, 4u);
    std::vector<uint64_t> row_ptr(N + 1);
    f.read(reinterpret_cast<char*>(row_ptr.data()), (N + 1) * sizeof(uint64_t));
    std::vector<uint64_t> col(row_ptr[N]);
    f.read(reinterpret_cast<char*>(col.data()), row_ptr[N] * sizeof(uint64_t));

    for (uint64_t u = 0; u < N; ++u) {
        mdb::gnn::Neighbors live;
        acc.get_neighbors_into(ObjectId{u}, mdb::gnn::EdgeOrientation::UNDIRECTED, live);
        std::vector<uint64_t> baked(col.begin() + row_ptr[u], col.begin() + row_ptr[u + 1]);
        std::vector<uint64_t> live_ids;
        for (auto& n : live.node_ids) live_ids.push_back(n.id & ObjectId::VALUE_MASK);
        EXPECT_EQ(baked, live_ids) << "row " << u;
    }
}

TEST(SymmetricBake, MultigraphRefuses) {
    (void)MdbFixture::instance();
    auto& mgr = GQL::ProjectionManager::get_instance();
    std::string dir = mgr.create_projection("sym_bake_multi");
    GQL::ProjectionStorage s(dir, MdbFixture::instance().db_folder(), "sym_bake_multi");
    s.init();
    for (uint64_t i = 0; i < 2; ++i) {
        GQL::ProjectedNode n;
        n.node_id = ObjectId(i);
        s.add_node(n);
    }
    auto e = [&](uint64_t f, uint64_t t, uint64_t id) {
        GQL::ProjectedEdge x;
        x.from_node = ObjectId(f);
        x.to_node = ObjectId(t);
        x.edge_id = ObjectId(id);
        x.is_directed = true;
        return x;
    };
    s.add_edge(e(0, 1, 100));
    s.add_edge(e(0, 1, 101));  // PARALLEL
    s.flush();

    auto storage = open_projection(dir);
    auto [bytes, ms, refused] = Proc::run_symmetric_for_test(*storage);
    (void)ms;
    EXPECT_TRUE(refused);
    EXPECT_EQ(bytes, 0u);
    EXPECT_FALSE(fs::exists(fs::path(dir) / "topology_sym.csr"));
}

// mode='symmetric' builds the sym sidecar.
TEST(SymmetricMode, SymmetricModeBuildsSym) {
    (void)MdbFixture::instance();
    const std::string dir = build_small_projection("sym_mode_sym");
    auto storage = open_projection(dir);
    auto [bytes, status, refused] = Proc::run_mode_for_test(*storage, "symmetric");
    EXPECT_EQ(status, "built");
    EXPECT_GT(bytes, 64u);
    EXPECT_FALSE(refused);
    EXPECT_TRUE(fs::exists(fs::path(dir) / "topology_sym.csr"));
}

// mode='directional' leaves the sym sidecar untouched.
TEST(SymmetricMode, DirectionalModeSkipsSym) {
    (void)MdbFixture::instance();
    const std::string dir = build_small_projection("sym_mode_dir");
    auto storage = open_projection(dir);
    auto [bytes, status, refused] = Proc::run_mode_for_test(*storage, "directional");
    EXPECT_EQ(status, "skipped");
    EXPECT_EQ(bytes, 0u);
    EXPECT_FALSE(refused);
    EXPECT_FALSE(fs::exists(fs::path(dir) / "topology_sym.csr"));
}
