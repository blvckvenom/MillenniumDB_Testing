// four_level_topology_store_test.cc
//
// Spec #13 Phase 2 — FourLevelTopologyStore dispatcher unit tests (T13.6).
//
// Phase 2 only ships the dispatch skeleton — no `build()` orchestration, no
// profiler integration, no real RowMapping. These tests therefore drive the
// dispatcher with synthetically populated L1/L2 caches plus pluggable
// L4 callables (and optionally a real `TopologySnapshotReader` for L3).
//
// To keep the suite light we DO NOT spin up a `System` / `ProjectionStorage`
// fixture (the GQL projection layer is a heavy dependency for what is
// fundamentally a pure dispatch test). Instead we drive the four-tier paths
// with std::function L4 callables and either nullptr-or-pre-built mmap
// sidecars for L3.
//
// Coverage:
//   1. Dispatch_L1_Hits — node tier=1 routes to L1HashCache.
//   2. Dispatch_L2_Hits — node tier=2 routes to L2CompactCsr.
//   3. Dispatch_L3_Sidecar_AbsentFallsThroughToL4 — tier=3 with nullptr L3
//      reader plus a wired L4 dispatches to L4 (graceful degradation path,
//      same as Phase 3 will use when buildTopologySnapshot:false).
//   4. Dispatch_L4_BptFallback — tier=4 routes to the L4 callable.
//   5. NoL4Configured_TierMismatch_Throws — tier=3/4 with no L4 callable
//      configured throws a clear runtime_error.
//   6. IsolatedNode_AllTiers — zero-degree node returns empty for every
//      tier path.
//   7. MultiTierGraph_ConsistentResults — graph with mix of tiers; lookups
//      across all four match a B+Tree-equivalent oracle (oracle = the same
//      adjacency we put into each tier).
//   8. RowIdxOutOfRange_FallsToL4_OrThrows — `row_lookup` returns
//      sentinel >= tier_lookup.size(); dispatcher routes to L4 if wired,
//      throws otherwise.
//
// Note on Tier 3 (L3 mmap sidecar): rather than build a real sidecar fixture
// per-test (Spec #4-B writer needs a fake source-leaf for SHA-256 + a
// projection_dir + OpenSSL), we exercise the absent-sidecar path symbolically
// via `nullptr` and document via an explicit comment in test 3 that the real
// L3-hit path is covered by `topology_snapshot_reader_test.cc` + Phase 3
// integration tests.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/adj_entry.h"
#include "gnn/projection/four_level_topology_store.h"
#include "gnn/projection/l1_hash_cache.h"
#include "gnn/projection/l2_compact_csr.h"
#include "graph_models/object_id.h"

using mdb::gnn::AdjEntry;
using mdb::gnn::FourLevelTopologyStore;
using mdb::gnn::L1HashCache;
using mdb::gnn::L2CompactCsr;

namespace {

// Each test scenario: how to map ObjectId -> row_idx (identity is fine for
// these tests; we use ObjectId.id directly).
mdb::gnn::FourLevelTopologyStore::RowLookup identity_row_lookup() {
    return [](ObjectId v) -> uint64_t { return v.id; };
}

// L4 callable backed by an unordered_map from src node id to its full
// adjacency list. Lets us seed tier-4 nodes' "ground truth" without spinning
// up a real BPT.
mdb::gnn::FourLevelTopologyStore::L4Lookup make_oracle_l4(
    std::shared_ptr<std::unordered_map<uint64_t, std::vector<AdjEntry>>> oracle)
{
    return [oracle](ObjectId v) -> std::vector<AdjEntry> {
        auto it = oracle->find(v.id);
        if (it == oracle->end()) return {};
        return it->second;  // copy, simulating BPT materialization
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — Dispatch_L1_Hits.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyStore, Dispatch_L1_Hits) {
    const std::vector<uint8_t> tiers = { 1, 1, 1 };
    L1HashCache l1_fwd(tiers);
    L1HashCache l1_rev(tiers);
    L2CompactCsr l2_fwd, l2_rev;
    l2_fwd.freeze();
    l2_rev.freeze();

    l1_fwd.insert(0, std::vector<AdjEntry>{ {1, 100}, {2, 101} }, /*row_idx=*/0);
    l1_fwd.insert(1, std::vector<AdjEntry>{ {2, 102} },           /*row_idx=*/1);

    FourLevelTopologyStore::Config cfg;
    FourLevelTopologyStore store(
        l1_fwd, l1_rev, l2_fwd, l2_rev,
        /*l3_fwd=*/nullptr, /*l3_rev=*/nullptr,
        /*l4_fwd=*/{},      /*l4_rev=*/{},
        tiers, identity_row_lookup(), cfg);

    auto n0 = store.get_out_neighbors(ObjectId(0));
    ASSERT_EQ(1u, n0.tier);
    ASSERT_EQ(2u, n0.size());
    EXPECT_EQ(1u, n0.l1.data[0].node_id);
    EXPECT_EQ(2u, n0.l1.data[1].node_id);

    auto n1 = store.get_out_neighbors(ObjectId(1));
    ASSERT_EQ(1u, n1.tier);
    ASSERT_EQ(1u, n1.size());
    EXPECT_EQ(2u, n1.l1.data[0].node_id);
}

// ---------------------------------------------------------------------------
// Test 2 — Dispatch_L2_Hits.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyStore, Dispatch_L2_Hits) {
    const std::vector<uint8_t> tiers = { 2, 2, 2 };
    L1HashCache l1_fwd(tiers);
    L1HashCache l1_rev(tiers);
    L2CompactCsr l2_fwd, l2_rev;

    l2_fwd.add_node(0, std::vector<AdjEntry>{ {10, 0}, {11, 0}, {12, 0} });
    l2_fwd.add_node(1, std::vector<AdjEntry>{ {20, 0} });
    l2_fwd.add_node(2, std::vector<AdjEntry>{});  // isolated
    l2_fwd.freeze();
    l2_rev.freeze();

    FourLevelTopologyStore::Config cfg;
    FourLevelTopologyStore store(
        l1_fwd, l1_rev, l2_fwd, l2_rev,
        nullptr, nullptr, {}, {},
        tiers, identity_row_lookup(), cfg);

    auto n0 = store.get_out_neighbors(ObjectId(0));
    ASSERT_EQ(2u, n0.tier);
    ASSERT_EQ(3u, n0.size());
    EXPECT_EQ(10u, n0.l2_col_idx[0]);
    EXPECT_EQ(11u, n0.l2_col_idx[1]);
    EXPECT_EQ(12u, n0.l2_col_idx[2]);

    auto n1 = store.get_out_neighbors(ObjectId(1));
    ASSERT_EQ(2u, n1.tier);
    ASSERT_EQ(1u, n1.size());
    EXPECT_EQ(20u, n1.l2_col_idx[0]);

    auto n2 = store.get_out_neighbors(ObjectId(2));
    EXPECT_EQ(2u, n2.tier);
    EXPECT_EQ(0u, n2.size());
    EXPECT_TRUE(n2.empty());
}

// ---------------------------------------------------------------------------
// Test 3 — Dispatch_L3_Sidecar_AbsentFallsThroughToL4.
//
// Real L3 mmap-sidecar hits are exercised by the existing
// topology_snapshot_reader_test suite + Phase 3 integration tests. Here we
// validate the dispatch contract: a tier=3 node with nullptr L3 reader and a
// wired L4 callable falls through to L4, *not* to a throw.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyStore, Dispatch_L3_Sidecar_AbsentFallsThroughToL4) {
    const std::vector<uint8_t> tiers = { 3, 3 };
    L1HashCache l1_fwd(tiers);
    L1HashCache l1_rev(tiers);
    L2CompactCsr l2_fwd, l2_rev;
    l2_fwd.freeze();
    l2_rev.freeze();

    auto oracle = std::make_shared<
        std::unordered_map<uint64_t, std::vector<AdjEntry>>>();
    (*oracle)[0] = std::vector<AdjEntry>{ {7, 700}, {8, 701} };
    (*oracle)[1] = std::vector<AdjEntry>{ {9, 702} };

    FourLevelTopologyStore::Config cfg;
    FourLevelTopologyStore store(
        l1_fwd, l1_rev, l2_fwd, l2_rev,
        /*l3_fwd=*/nullptr, /*l3_rev=*/nullptr,
        /*l4_fwd=*/make_oracle_l4(oracle),
        /*l4_rev=*/make_oracle_l4(oracle),
        tiers, identity_row_lookup(), cfg);

    auto n0 = store.get_out_neighbors(ObjectId(0));
    EXPECT_EQ(4u, n0.tier);
    ASSERT_EQ(2u, n0.l4_owned.size());
    EXPECT_EQ(7u, n0.l4_owned[0].node_id);
    EXPECT_EQ(8u, n0.l4_owned[1].node_id);

    auto n1 = store.get_out_neighbors(ObjectId(1));
    EXPECT_EQ(4u, n1.tier);
    ASSERT_EQ(1u, n1.l4_owned.size());
    EXPECT_EQ(9u, n1.l4_owned[0].node_id);
}

// ---------------------------------------------------------------------------
// Test 4 — Dispatch_L4_BptFallback.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyStore, Dispatch_L4_BptFallback) {
    const std::vector<uint8_t> tiers = { 4, 4 };
    L1HashCache l1_fwd(tiers);
    L1HashCache l1_rev(tiers);
    L2CompactCsr l2_fwd, l2_rev;
    l2_fwd.freeze();
    l2_rev.freeze();

    auto oracle = std::make_shared<
        std::unordered_map<uint64_t, std::vector<AdjEntry>>>();
    (*oracle)[0] = std::vector<AdjEntry>{ {100, 999}, {101, 998} };

    FourLevelTopologyStore::Config cfg;
    FourLevelTopologyStore store(
        l1_fwd, l1_rev, l2_fwd, l2_rev,
        nullptr, nullptr,
        make_oracle_l4(oracle), make_oracle_l4(oracle),
        tiers, identity_row_lookup(), cfg);

    auto n0 = store.get_out_neighbors(ObjectId(0));
    EXPECT_EQ(4u, n0.tier);
    ASSERT_EQ(2u, n0.l4_owned.size());
    EXPECT_EQ(100u, n0.l4_owned[0].node_id);
    EXPECT_EQ(999u, n0.l4_owned[0].edge_id);

    // Node not in the oracle returns empty (mirrors a BPT empty range).
    auto n1 = store.get_out_neighbors(ObjectId(1));
    EXPECT_EQ(4u, n1.tier);
    EXPECT_TRUE(n1.empty());
}

// ---------------------------------------------------------------------------
// Test 5 — NoL4Configured_TierMismatch_Throws.
//
// Defensive contract: tier 3 with no L3 sidecar AND no L4 callback, or tier
// 4 with no L4 callback, throws so callers see the misconfiguration loudly
// rather than silently producing empty neighbor sets.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyStore, NoL4Configured_TierMismatch_Throws) {
    const std::vector<uint8_t> tiers = { 3, 4 };
    L1HashCache l1_fwd(tiers);
    L1HashCache l1_rev(tiers);
    L2CompactCsr l2_fwd, l2_rev;
    l2_fwd.freeze();
    l2_rev.freeze();

    FourLevelTopologyStore::Config cfg;
    FourLevelTopologyStore store(
        l1_fwd, l1_rev, l2_fwd, l2_rev,
        nullptr, nullptr, {}, {},  // no L3, no L4
        tiers, identity_row_lookup(), cfg);

    EXPECT_THROW(store.get_out_neighbors(ObjectId(0)), std::runtime_error);
    EXPECT_THROW(store.get_out_neighbors(ObjectId(1)), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Test 6 — IsolatedNode_AllTiers.
//
// Zero-degree nodes return empty Neighbors regardless of which tier they
// were assigned to.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyStore, IsolatedNode_AllTiers) {
    // node 0 -> L1 isolated, node 1 -> L2 isolated, node 2 -> L4 isolated
    const std::vector<uint8_t> tiers = { 1, 2, 4 };
    L1HashCache l1_fwd(tiers);
    L1HashCache l1_rev(tiers);
    L2CompactCsr l2_fwd, l2_rev;

    l1_fwd.insert(0, /*neighbors=*/{}, /*row_idx=*/0);
    l2_fwd.add_node(1, /*neighbors=*/{});
    l2_fwd.freeze();
    l2_rev.freeze();

    auto empty_oracle = std::make_shared<
        std::unordered_map<uint64_t, std::vector<AdjEntry>>>();
    (*empty_oracle)[2] = {};

    FourLevelTopologyStore::Config cfg;
    FourLevelTopologyStore store(
        l1_fwd, l1_rev, l2_fwd, l2_rev,
        nullptr, nullptr,
        make_oracle_l4(empty_oracle), make_oracle_l4(empty_oracle),
        tiers, identity_row_lookup(), cfg);

    EXPECT_TRUE(store.get_out_neighbors(ObjectId(0)).empty());
    EXPECT_TRUE(store.get_out_neighbors(ObjectId(1)).empty());
    EXPECT_TRUE(store.get_out_neighbors(ObjectId(2)).empty());
}

// ---------------------------------------------------------------------------
// Test 7 — MultiTierGraph_ConsistentResults.
//
// Mixed-tier graph: one node per tier (1, 2, 3, 4). Verify that:
//  - L1-tagged node returns the L1 cache contents.
//  - L2-tagged node returns the L2 CSR contents.
//  - L3-tagged node falls through to L4 (no sidecar in this test).
//  - L4-tagged node returns the L4 oracle contents.
// All four results match the canonical adjacency we seeded into the
// per-tier sources from the same shared oracle.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyStore, MultiTierGraph_ConsistentResults) {
    // node 0 -> L1, 1 -> L2, 2 -> L3 (no sidecar -> L4), 3 -> L4 direct
    const std::vector<uint8_t> tiers = { 1, 2, 3, 4 };
    L1HashCache l1_fwd(tiers);
    L1HashCache l1_rev(tiers);
    L2CompactCsr l2_fwd, l2_rev;

    auto oracle = std::make_shared<
        std::unordered_map<uint64_t, std::vector<AdjEntry>>>();
    (*oracle)[0] = std::vector<AdjEntry>{ {10, 0}, {11, 0} };
    (*oracle)[1] = std::vector<AdjEntry>{ {20, 0}, {21, 0}, {22, 0} };
    (*oracle)[2] = std::vector<AdjEntry>{ {30, 0} };
    (*oracle)[3] = std::vector<AdjEntry>{ {40, 0}, {41, 0} };

    // Seed L1 with node 0.
    l1_fwd.insert(0, (*oracle)[0], /*row_idx=*/0);
    // Seed L2 with node 1. (Other L2 rows must be added too because
    // L2CompactCsr doesn't have a "skip" semantic; we stub them in.)
    l2_fwd.add_node(1, (*oracle)[1]);
    l2_fwd.freeze();
    l2_rev.freeze();

    FourLevelTopologyStore::Config cfg;
    FourLevelTopologyStore store(
        l1_fwd, l1_rev, l2_fwd, l2_rev,
        nullptr, nullptr,
        make_oracle_l4(oracle), make_oracle_l4(oracle),
        tiers, identity_row_lookup(), cfg);

    // Helper to extract dst node ids from any tier's Neighbors shape.
    auto dsts_of = [](const FourLevelTopologyStore::Neighbors& n) {
        std::vector<uint64_t> out;
        switch (n.tier) {
            case 1:
                for (std::size_t i = 0; i < n.l1.size; ++i)
                    out.push_back(n.l1.data[i].node_id);
                break;
            case 2:
                for (std::size_t i = 0; i < n.l2_size; ++i)
                    out.push_back(n.l2_col_idx[i]);
                break;
            case 3:
                for (std::size_t i = 0; i < n.l3_size; ++i)
                    out.push_back(n.l3_col_idx[i]);
                break;
            case 4:
                for (const auto& e : n.l4_owned) out.push_back(e.node_id);
                break;
        }
        return out;
    };

    {
        auto n = store.get_out_neighbors(ObjectId(0));
        EXPECT_EQ(1u, n.tier);
        EXPECT_EQ(std::vector<uint64_t>({10, 11}), dsts_of(n));
    }
    {
        auto n = store.get_out_neighbors(ObjectId(1));
        EXPECT_EQ(2u, n.tier);
        EXPECT_EQ(std::vector<uint64_t>({20, 21, 22}), dsts_of(n));
    }
    {
        auto n = store.get_out_neighbors(ObjectId(2));
        EXPECT_EQ(4u, n.tier);  // L3 missing -> L4 fallback
        EXPECT_EQ(std::vector<uint64_t>({30}), dsts_of(n));
    }
    {
        auto n = store.get_out_neighbors(ObjectId(3));
        EXPECT_EQ(4u, n.tier);
        EXPECT_EQ(std::vector<uint64_t>({40, 41}), dsts_of(n));
    }
}

// ---------------------------------------------------------------------------
// Test 8 — RowIdxOutOfRange_FallsToL4_OrThrows.
//
// `row_lookup` returns a sentinel >= tier_lookup.size(): the dispatcher must
// treat the node as outside the projection. With L4 wired it routes to L4;
// without L4 it throws (defensive — silent empty would mask import bugs).
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyStore, RowIdxOutOfRange_FallsToL4_OrThrows) {
    const std::vector<uint8_t> tiers = { 1, 1 };
    L1HashCache l1_fwd(tiers);
    L1HashCache l1_rev(tiers);
    L2CompactCsr l2_fwd, l2_rev;
    l2_fwd.freeze();
    l2_rev.freeze();

    // row_lookup that always returns "out-of-range" for any input.
    FourLevelTopologyStore::RowLookup oor =
        [](ObjectId) -> uint64_t { return UINT64_MAX; };

    FourLevelTopologyStore::Config cfg;

    auto oracle = std::make_shared<
        std::unordered_map<uint64_t, std::vector<AdjEntry>>>();
    (*oracle)[0] = std::vector<AdjEntry>{ {1, 0} };

    // (a) With L4 wired -> tier 4 result.
    {
        FourLevelTopologyStore store(
            l1_fwd, l1_rev, l2_fwd, l2_rev,
            nullptr, nullptr,
            make_oracle_l4(oracle), make_oracle_l4(oracle),
            tiers, oor, cfg);

        auto n = store.get_out_neighbors(ObjectId(0));
        EXPECT_EQ(4u, n.tier);
        ASSERT_EQ(1u, n.l4_owned.size());
        EXPECT_EQ(1u, n.l4_owned[0].node_id);
    }

    // (b) Without L4 -> defensive throw.
    {
        FourLevelTopologyStore store(
            l1_fwd, l1_rev, l2_fwd, l2_rev,
            nullptr, nullptr, {}, {},
            tiers, oor, cfg);

        EXPECT_THROW(store.get_out_neighbors(ObjectId(0)), std::out_of_range);
    }
}
