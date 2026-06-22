// topology_accessor_four_level_integration_test.cc
//
// Integration tests for `TopologyAccessor::enable_four_level_store()`.
// Verifies that building the Four-Level Topology Store (L1 RAM hash /
// L2 compact uint32 CSR / L3 mmap sidecar / L4 direct B+Tree) from a
// live ProjectionStorage's B+Tree edge indexes works correctly, and
// that all subsequent `get_*_neighbors` calls dispatch through it.
//
// Coverage (3 tests):
//
//   1. EnableDisable_Roundtrip
//      Build the four-level store, look up neighbours for every
//      fixture node, verify the dispatch result matches a parallel
//      B+Tree-only `TopologyAccessor` (oracle).
//
//   2. BackwardsCompat_DefaultPath
//      Without enabling the four-level store, the existing dispatch
//      chain (in-memory adjacency cache → topology CSR sidecar →
//      direct B+Tree) preserves byte-identical behaviour. Sub-cases:
//          (a) cache disabled
//          (b) in-memory adjacency cache enabled + prebuilt
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
#include "graph_models/gql/projection/topology_snapshot_writer.h"
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
// and (b) the in-memory adjacency cache path return the same data they
// returned before the Four-Level Topology Store was introduced.
// Compared by-value to a pristine BPT accessor.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorFourLevel, BackwardsCompat_DefaultPath) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("four_level_backcompat");

    mdb::gnn::TopologyAccessor acc_oracle(*storage);

    // Sub-case (a): cache disabled. Default BPT-direct path.
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

    // Sub-case (b): in-memory adjacency cache (one full B+Tree scan into
    // unordered_map<src, vector<AdjEntry>>) enabled + prebuilt. The
    // four-level store must NOT have hijacked this existing path.
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
// SamplingConfig::validate() should reject the invalid combination where
// `useFourLevelTopologyStore=true` but `useAdjacencyCache=false`.
// Both flags being false is OK (falls back to topology CSR sidecar or
// direct B+Tree); both true is OK (Four-Level Topology Store is built on
// top of the adjacency cache infrastructure and supersedes it); only the
// asymmetric `useFourLevelTopologyStore=true && useAdjacencyCache=false`
// combination is rejected.
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
        // OK: both false (legacy fallback to topology CSR sidecar or direct B+Tree).
        auto cfg = make_base_config();
        cfg.use_four_level_topology_store = false;
        cfg.use_adjacency_cache           = false;
        EXPECT_NO_THROW(cfg.validate());
    }
    {
        // OK: both true (Four-Level Topology Store uses and supersedes the
        // in-memory adjacency cache path).
        auto cfg = make_base_config();
        cfg.use_four_level_topology_store = true;
        cfg.use_adjacency_cache           = true;
        EXPECT_NO_THROW(cfg.validate());
    }
    {
        // OK: in-memory adjacency cache only (one full B+Tree scan, O(1)
        // hash lookup thereafter), without the frequency-tiered four-level store.
        auto cfg = make_base_config();
        cfg.use_four_level_topology_store = false;
        cfg.use_adjacency_cache           = true;
        EXPECT_NO_THROW(cfg.validate());
    }
    {
        // ERROR: Four-Level Topology Store requires the adjacency cache
        // infrastructure to be active; this combination is invalid.
        auto cfg = make_base_config();
        cfg.use_four_level_topology_store = true;
        cfg.use_adjacency_cache           = false;
        EXPECT_THROW(cfg.validate(), std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// Build_UsesSidecarWhenAvailable.
//
// When `use_l3_mmap_sidecar:true` is set on a projection that already has
// `topology_*.csr` sidecar files (the mmap-backed CSR sidecar that gives
// O(1) neighbor slices), `populate_direction_` takes the fast
// path (sidecar walk) instead of the BPT walk. Streaming determinism + L1
// node counts must remain bit-identical to the BPT path, so this test:
//
//   1. Builds the sidecar files directly using `TopologySnapshotWriter`
//      (avoiding the heavier `native_projection_builder.h` link surface
//      that is independently broken on this branch).
//   2. Constructs a four-level store with a generous L1 budget so all
//      nodes land in tier 1.
//   3. Asserts L1 holds every node in both directions and L2 / L3 are
//      empty.
//   4. Verifies per-node neighbour parity vs a BPT-only oracle accessor
//      on `(node_id, edge_id)` pairs.
//
// Build-time speedup (~5×) is measured separately by Phase 6 benchmarks;
// this unit test only proves functional equivalence.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorFourLevel, Build_UsesSidecarWhenAvailable) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("four_level_sidecar_fast_path");

    // ------------------------------------------------------------------
    // Build topology_fwd.csr + topology_rev.csr from the fixture edges.
    // Order requirement: edges must arrive in source-monotonic order
    // matching the degree histogram. We sort the fixture edges into
    // src order and feed both directions.
    // ------------------------------------------------------------------
    auto build_sidecar = [&](GQL::Projection::TopologySnapshotWriter::Direction dir) {
        std::vector<uint64_t> degrees(FixtureGraph::kNumNodes, 0);
        std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sorted_edges;
        sorted_edges.reserve(FixtureGraph::edges().size());
        for (const auto& e : FixtureGraph::edges()) {
            const uint64_t from = std::get<0>(e);
            const uint64_t to   = std::get<1>(e);
            const uint64_t eid  = std::get<2>(e);
            if (dir == GQL::Projection::TopologySnapshotWriter::Direction::FORWARD) {
                sorted_edges.emplace_back(from, to, eid);
                degrees[from]++;
            } else {
                // REVERSE: store as (to, from, eid). Source key is the
                // destination of the natural edge.
                sorted_edges.emplace_back(to, from, eid);
                degrees[to]++;
            }
        }
        std::sort(sorted_edges.begin(), sorted_edges.end());

        GQL::Projection::TopologySnapshotWriter w(
            std::filesystem::path(storage->get_projection_dir()),
            dir,
            FixtureGraph::kNumNodes,
            std::move(degrees),
            /*include_edge_ids=*/true);
        for (const auto& [src, dst, eid] : sorted_edges) {
            w.append_edge(ObjectId(src), ObjectId(dst), ObjectId(eid));
        }
        w.finalize();
    };
    build_sidecar(GQL::Projection::TopologySnapshotWriter::Direction::FORWARD);
    build_sidecar(GQL::Projection::TopologySnapshotWriter::Direction::REVERSE);

    // ------------------------------------------------------------------
    // Construct a four-level store directly (bypassing TopologyAccessor)
    // so the test can inspect tier counts. Generous L1 budget guarantees
    // every node lands in tier 1.
    // ------------------------------------------------------------------
    mdb::gnn::FourLevelTopologyStore::Config cfg;
    cfg.l1_budget_mb        = 256;
    cfg.l2_budget_mb        = 256;
    cfg.use_l3_mmap_sidecar = true;
    cfg.orientation         = mdb::gnn::EdgeOrientation::UNDIRECTED;

    mdb::gnn::FourLevelTopologyStore store(
        storage->get_from_to_edge_index(),
        storage->get_to_from_edge_index(),
        storage.get(),
        std::filesystem::path(storage->get_projection_dir()),
        cfg);
    store.build();

    // Tier 1 holds every node in both directions. UNDIRECTED orientation
    // builds both fwd and rev L1, so the count is 2 × N.
    EXPECT_EQ(2u * FixtureGraph::kNumNodes, store.l1_node_count());
    EXPECT_EQ(0u, store.l2_node_count());
    // l3_node_count() reports the sidecar's `num_nodes()` per direction
    // (it's a sidecar-presence diagnostic, not a tier-3-assignment
    // counter). Under UNDIRECTED both directions report N, totalling
    // 2 × N. The orthogonal assertion that L1 holds every node already
    // proves no node was actually served from L3 at lookup time.
    EXPECT_EQ(2u * FixtureGraph::kNumNodes, store.l3_node_count());

    // Per-node parity vs a BPT-only oracle accessor. We compare on
    // (node_id, edge_id) pairs since tier 1 retains edge_ids.
    mdb::gnn::TopologyAccessor oracle(*storage);
    for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
        auto store_n = store.get_out_neighbors(ObjectId(nid));
        std::vector<std::pair<uint64_t, uint64_t>> got;
        store_n.for_each_with_edge_id(
            [&](uint64_t dst, uint64_t eid) {
                got.emplace_back(dst, eid);
            });
        std::sort(got.begin(), got.end());

        auto oracle_n = oracle.get_out_neighbors(ObjectId(nid));
        std::vector<std::pair<uint64_t, uint64_t>> expected;
        for (std::size_t i = 0; i < oracle_n.node_ids.size(); ++i) {
            const uint64_t eid = (i < oracle_n.edge_ids.size())
                                 ? oracle_n.edge_ids[i].id : 0ULL;
            expected.emplace_back(oracle_n.node_ids[i].id, eid);
        }
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(expected, got) << "out node " << nid;
    }

    // Cold-start path: the MinHash permutation must be empty when no
    // node_counts.bin exists (every build today). Phase 5 will flip
    // the warm-start branch and populate this vector.
    EXPECT_TRUE(store.l3_reorder_permutation().empty());
}

// ---------------------------------------------------------------------------
// Build_ColdStartSkipsMinHashReorder.
//
// Confirms the cold-start branch of `compute_l3_minhash_reorder_` leaves
// the permutation empty when `node_counts.bin` is absent. When no prior
// per-node access-count file exists the MinHash reorder cannot be computed,
// so this test pins the invariant against future regressions in the
// warm-start activation logic.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorFourLevel, Build_ColdStartSkipsMinHashReorder) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("four_level_cold_start_minhash");

    mdb::gnn::FourLevelTopologyStore::Config cfg;
    cfg.l1_budget_mb        = 256;
    cfg.l2_budget_mb        = 256;
    cfg.use_l3_mmap_sidecar = false;
    cfg.orientation         = mdb::gnn::EdgeOrientation::UNDIRECTED;

    mdb::gnn::FourLevelTopologyStore store(
        storage->get_from_to_edge_index(),
        storage->get_to_from_edge_index(),
        storage.get(),
        std::filesystem::path(storage->get_projection_dir()),
        cfg);
    store.build();

    EXPECT_TRUE(store.l3_reorder_permutation().empty());
}

// ---------------------------------------------------------------------------
// Build_SymmetricTierFromDirectionalSidecars.
//
// An UNDIRECTED build with both directional sidecars present (and no on-disk
// topology_sym.csr) populates the symmetric tier by merging fwd+rev row-by-row,
// so is_symmetric_built() is true and the symmetric L1 holds every node. A
// NATURAL-orientation build leaves the symmetric tier off.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorFourLevel, Build_SymmetricTierFromDirectionalSidecars) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("four_level_symmetric_tier");

    auto build_sidecar = [&](GQL::Projection::TopologySnapshotWriter::Direction dir) {
        std::vector<uint64_t> degrees(FixtureGraph::kNumNodes, 0);
        std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sorted_edges;
        for (const auto& e : FixtureGraph::edges()) {
            const uint64_t from = std::get<0>(e), to = std::get<1>(e),
                           eid = std::get<2>(e);
            if (dir == GQL::Projection::TopologySnapshotWriter::Direction::FORWARD) {
                sorted_edges.emplace_back(from, to, eid);
                degrees[from]++;
            } else {
                sorted_edges.emplace_back(to, from, eid);
                degrees[to]++;
            }
        }
        std::sort(sorted_edges.begin(), sorted_edges.end());
        GQL::Projection::TopologySnapshotWriter w(
            std::filesystem::path(storage->get_projection_dir()), dir,
            FixtureGraph::kNumNodes, std::move(degrees), /*include_edge_ids=*/true);
        for (const auto& [src, dst, eid] : sorted_edges) {
            w.append_edge(ObjectId(src), ObjectId(dst), ObjectId(eid));
        }
        w.finalize();
    };
    build_sidecar(GQL::Projection::TopologySnapshotWriter::Direction::FORWARD);
    build_sidecar(GQL::Projection::TopologySnapshotWriter::Direction::REVERSE);

    {
        mdb::gnn::FourLevelTopologyStore::Config cfg;
        cfg.l1_budget_mb        = 256;  // all nodes -> L1
        cfg.l2_budget_mb        = 256;
        cfg.use_l3_mmap_sidecar = true;
        cfg.orientation         = mdb::gnn::EdgeOrientation::UNDIRECTED;
        mdb::gnn::FourLevelTopologyStore store(
            storage->get_from_to_edge_index(),
            storage->get_to_from_edge_index(),
            storage.get(),
            std::filesystem::path(storage->get_projection_dir()), cfg);
        store.build();
        EXPECT_TRUE(store.is_symmetric_built());
        // All 6 nodes (incl. the isolated one) land in the symmetric L1.
        EXPECT_EQ(FixtureGraph::kNumNodes, store.l1_sym_node_count());
        EXPECT_EQ(0u, store.l2_sym_node_count());
    }
    {
        mdb::gnn::FourLevelTopologyStore::Config cfg;
        cfg.l1_budget_mb        = 256;
        cfg.l2_budget_mb        = 256;
        cfg.use_l3_mmap_sidecar = true;
        cfg.orientation         = mdb::gnn::EdgeOrientation::NATURAL;
        mdb::gnn::FourLevelTopologyStore store(
            storage->get_from_to_edge_index(),
            storage->get_to_from_edge_index(),
            storage.get(),
            std::filesystem::path(storage->get_projection_dir()), cfg);
        store.build();
        EXPECT_FALSE(store.is_symmetric_built());
        EXPECT_EQ(0u, store.l1_sym_node_count());
    }
}
