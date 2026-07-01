// Unit tests for detect_parallel_edges — the guard that refuses a symmetric
// edge_id-drop bake on a meaningful multigraph (two distinct edges sharing the
// same (src,dst)). The from_to_edge BPT key layout (src,dst,edge_id) ascending
// makes such edges consecutive, so the guard is a single linear scan.

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_symmetric_merge.h"
#include "graph_models/object_id.h"

#include "gnn_projection_test_fixture.h"

using gnn_test_fixture::build_small_projection;
using gnn_test_fixture::MdbFixture;
using gnn_test_fixture::open_projection;

// Simple graph (no parallel edges) -> false.
TEST(SymmetricParallelEdgeGuard, SimpleGraphNoParallel) {
    (void)MdbFixture::instance();
    const std::string dir = build_small_projection("guard_simple");  // 0->1,0->2,1->2,2->3
    auto storage = open_projection(dir);
    EXPECT_FALSE(GQL::Projection::detect_parallel_edges(
        storage->get_from_to_edge_index(), storage->get_node_count()));
}

// Two distinct edges on the same (0,1) -> true.
TEST(SymmetricParallelEdgeGuard, MultigraphDetected) {
    (void)MdbFixture::instance();
    auto& mgr = GQL::ProjectionManager::get_instance();
    std::string dir = mgr.create_projection("guard_multi");
    GQL::ProjectionStorage s(dir, MdbFixture::instance().db_folder(), "guard_multi");
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
    s.add_edge(e(0, 1, 101));  // PARALLEL: same (0,1), distinct id
    s.flush();
    auto storage = open_projection(dir);
    EXPECT_TRUE(GQL::Projection::detect_parallel_edges(
        storage->get_from_to_edge_index(), storage->get_node_count()));
}

// Null BPT is treated as "no parallel edges" (conservative, never throws).
TEST(SymmetricParallelEdgeGuard, NullBptIsFalse) {
    EXPECT_FALSE(GQL::Projection::detect_parallel_edges(nullptr, 0));
}
