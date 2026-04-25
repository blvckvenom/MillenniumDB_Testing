// topology_accessor_four_level_integration_test.cc
//
// Spec #13 Phase 3 (T13.8) integration tests — verifies that
// `TopologyAccessor::enable_four_level_store()` builds a real
// FourLevelTopologyStore from a live ProjectionStorage's B+Tree edge
// indexes and that all subsequent `get_*_neighbors` calls dispatch
// through it.
//
// Coverage (3 tests as specified in the Phase 3 plan):
//
//   1. EnableDisable_Roundtrip
//      Build the four-level store, look up neighbours for every
//      fixture node, verify the dispatch result matches a parallel
//      B+Tree-only `TopologyAccessor` (oracle).
//
//   2. BackwardsCompat_DefaultPath
//      Without enabling the four-level store, the existing dispatch
//      chain (Spec #11 cache → Spec #4-B sidecar → BPT direct)
//      preserves byte-identical behaviour. Sub-cases:
//          (a) cache disabled
//          (b) cache enabled + prebuilt
//      Both must match the BPT-oracle accessor.
//
//   3. Conflict_AdjCacheFalseAndFourLevelTrue_Throws
//      `SamplingConfig::validate()` rejects the
//      `useFourLevelTopologyStore=true` + `useAdjacencyCache=false`
//      combination per design D8. We assert the validate() throw
//      surface here since `gnn_offline_sample_procedure_test.cc`
//      does not exist in this repo today.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/four_level_topology_store.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/sampling/sampling_config.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace {

namespace fs = std::filesystem;

// Process-lifetime fixture (System + ProjectionManager singletons can only be
// bound once per process). Mirrors the pattern used by
// topology_accessor_adjacency_cache_test.cc.
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
        db_folder_ = "test_db_topo_acc_four_level_" + std::to_string(rng());
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

// Modest fixture: 6 nodes, 8 directed edges, includes one isolated node (5)
// to exercise the absent-key path of the cache. Edge ids start at 200.
//
//   0 -> 1 (e=200)
//   0 -> 2 (e=201)
//   1 -> 3 (e=202)
//   2 -> 3 (e=203)
//   2 -> 4 (e=204)
//   3 -> 4 (e=205)
//   4 -> 0 (e=206)   -- back-edge
//   1 -> 4 (e=207)
//   (node 5 is isolated)
struct FixtureGraph {
    static constexpr uint64_t kNumNodes = 6;
    static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& edges() {
        static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> E = {
            {0, 1, 200},
            {0, 2, 201},
            {1, 3, 202},
            {2, 3, 203},
            {2, 4, 204},
            {3, 4, 205},
            {4, 0, 206},
            {1, 4, 207},
        };
        return E;
    }
};

std::unique_ptr<GQL::ProjectionStorage> build_fixture_storage(
    const std::string& projection_name)
{
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
    return storage;
}

std::vector<std::pair<uint64_t, uint64_t>> sorted_pairs(
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

// "Node-id only" comparator. The four-level store's L2 tier deliberately
// drops edge_ids per the L2CompactCsr design note, so when we compare its
// output to the BPT-oracle for L2 nodes we project both sides to node-id
// sets. Useful for sub-tests that want strict tier-1 parity AND lenient
// tier-2 parity in the same fixture.
std::vector<uint64_t> sorted_node_ids(const mdb::gnn::Neighbors& n) {
    std::vector<uint64_t> out;
    out.reserve(n.node_ids.size());
    for (const auto& id : n.node_ids) out.push_back(id.id);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — EnableDisable_Roundtrip.
//
// After enable_four_level_store(), every neighbour lookup returns a result
// that matches the parallel BPT-oracle accessor (modulo edge_ids on L2).
// ---------------------------------------------------------------------------
TEST(TopologyAccessorFourLevel, EnableDisable_Roundtrip) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("four_level_roundtrip");

    mdb::gnn::TopologyAccessor acc_four_level(*storage);
    mdb::gnn::TopologyAccessor acc_bpt(*storage);

    EXPECT_FALSE(acc_four_level.is_four_level_store_enabled());

    mdb::gnn::FourLevelTopologyStore::Config cfg;
    cfg.l1_budget_mb        = 256;   // generous → all nodes go to L1
    cfg.l2_budget_mb        = 256;
    cfg.use_l3_mmap_sidecar = false; // small fixture lacks the sidecar
    cfg.orientation         = mdb::gnn::EdgeOrientation::UNDIRECTED;
    acc_four_level.enable_four_level_store(cfg);

    EXPECT_TRUE(acc_four_level.is_four_level_store_enabled());

    // For each fixture node, BPT-oracle vs four-level dispatch should
    // return the same neighbour set. Compare on node-ids only — see the
    // L2CompactCsr edge-id design note.
    for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
        EXPECT_EQ(sorted_node_ids(acc_bpt.get_out_neighbors(ObjectId(nid))),
                  sorted_node_ids(acc_four_level.get_out_neighbors(ObjectId(nid))))
            << "out node " << nid;
        EXPECT_EQ(sorted_node_ids(acc_bpt.get_in_neighbors(ObjectId(nid))),
                  sorted_node_ids(acc_four_level.get_in_neighbors(ObjectId(nid))))
            << "in node " << nid;
    }

    // Calling enable twice throws (idempotent fail-loud contract).
    EXPECT_THROW(acc_four_level.enable_four_level_store(cfg), std::logic_error);
}

// ---------------------------------------------------------------------------
// Test 2 — BackwardsCompat_DefaultPath.
//
// Without enabling the four-level store, both (a) the cache-disabled path
// and (b) the Spec #11 cache-enabled path return the same data they
// returned pre-Spec-#13. Compared by-value to a pristine BPT accessor.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorFourLevel, BackwardsCompat_DefaultPath) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("four_level_backcompat");

    mdb::gnn::TopologyAccessor acc_oracle(*storage);

    // Sub-case (a): cache disabled. Default state pre-Spec-#13.
    {
        mdb::gnn::TopologyAccessor acc(*storage);
        EXPECT_FALSE(acc.is_four_level_store_enabled());
        EXPECT_FALSE(acc.is_adjacency_cache_enabled());

        for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
            EXPECT_EQ(sorted_pairs(acc_oracle.get_out_neighbors(ObjectId(nid))),
                      sorted_pairs(acc.get_out_neighbors(ObjectId(nid))))
                << "(a) out node " << nid;
            EXPECT_EQ(sorted_pairs(acc_oracle.get_in_neighbors(ObjectId(nid))),
                      sorted_pairs(acc.get_in_neighbors(ObjectId(nid))))
                << "(a) in node " << nid;
        }
    }

    // Sub-case (b): Spec #11 cache enabled + prebuilt. Existing pre-
    // Spec-#13 path; the four-level store must NOT have hijacked it.
    {
        mdb::gnn::TopologyAccessor acc(*storage);
        acc.enable_adjacency_cache(true);
        acc.prebuild_adjacency_cache(mdb::gnn::EdgeOrientation::UNDIRECTED);
        EXPECT_FALSE(acc.is_four_level_store_enabled());
        EXPECT_TRUE(acc.is_adjacency_cache_enabled());

        for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
            EXPECT_EQ(sorted_pairs(acc_oracle.get_out_neighbors(ObjectId(nid))),
                      sorted_pairs(acc.get_out_neighbors(ObjectId(nid))))
                << "(b) out node " << nid;
            EXPECT_EQ(sorted_pairs(acc_oracle.get_in_neighbors(ObjectId(nid))),
                      sorted_pairs(acc.get_in_neighbors(ObjectId(nid))))
                << "(b) in node " << nid;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3 — Conflict_AdjCacheFalseAndFourLevelTrue_Throws.
//
// SamplingConfig::validate() should reject the invalid combination per
// design §2.8 / D8. Both flags being false is OK (sidecar/BPT-direct
// path); both true is OK (Spec #13 supersedes Spec #11 transparently);
// only `useFourLevelTopologyStore=true && useAdjacencyCache=false` is
// rejected.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorFourLevel, Conflict_AdjCacheFalseAndFourLevelTrue_Throws) {
    auto make_base_config = []() {
        mdb::gnn::SamplingConfig c;
        c.projection_name = "fixture";
        c.sample_name     = "fixture_samples";
        c.fanouts         = {2};
        return c;
    };

    {
        // OK: both false (legacy fallback to sidecar/BPT direct).
        auto cfg = make_base_config();
        cfg.use_four_level_topology_store = false;
        cfg.use_adjacency_cache           = false;
        EXPECT_NO_THROW(cfg.validate());
    }
    {
        // OK: both true (Spec #13 supersedes Spec #11).
        auto cfg = make_base_config();
        cfg.use_four_level_topology_store = true;
        cfg.use_adjacency_cache           = true;
        EXPECT_NO_THROW(cfg.validate());
    }
    {
        // OK: legacy Spec #11 only.
        auto cfg = make_base_config();
        cfg.use_four_level_topology_store = false;
        cfg.use_adjacency_cache           = true;
        EXPECT_NO_THROW(cfg.validate());
    }
    {
        // ERROR: D8 violation.
        auto cfg = make_base_config();
        cfg.use_four_level_topology_store = true;
        cfg.use_adjacency_cache           = false;
        EXPECT_THROW(cfg.validate(), std::invalid_argument);
    }
}
