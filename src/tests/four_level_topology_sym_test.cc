// Unit tests for the FourLevelTopologyStore symmetric (pre-merged undirected)
// tier: the members + accessor (this file), the per-row merge populate
// (extended in later tasks), the single-dispatch collapse, and the edge_id-drop
// gate. The symmetric tier reuses the SAME per-node tier assignment as the
// directional tiers (direction-agnostic), so a node's tier (L1/L2/L3/L4) is the
// same whether read out, in, or undirected.

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

// The symmetric tier is opt-in: a dispatcher-constructed store (no build())
// never has it, so is_symmetric_built() defaults false.
TEST(FourLevelTopologySym, SymTierDefaultsOff) {
    const std::vector<uint8_t> tiers = {1, 1};
    L1HashCache l1f(tiers), l1r(tiers);
    L2CompactCsr l2f, l2r;
    l2f.freeze();
    l2r.freeze();
    FourLevelTopologyStore::Config cfg;  // orientation UNDIRECTED default
    FourLevelTopologyStore store(
        l1f, l1r, l2f, l2r,
        /*l3_fwd=*/nullptr, /*l3_rev=*/nullptr,
        /*l4_fwd=*/{}, /*l4_rev=*/{},
        tiers, [](ObjectId v) { return v.id; }, cfg);
    EXPECT_FALSE(store.is_symmetric_built());
}

// ---------------------------------------------------------------------------
// Task 11 — the per-row merge that the symmetric populate uses. Replicates the
// accessor's dedup: edge_id key when has_edge_ids (distinct -> nothing removed),
// node-id key otherwise; out(u) first, then in(u) survivors.
// ---------------------------------------------------------------------------
using mdb::gnn::detail::symmetric_merge_row;
using mdb::gnn::detail::resolve_symmetric_dst_tag;

// Distinct edge_ids -> out ++ in with NO node-id dedup (parallel/mutual edges
// are PRESERVED, byte-identical to the accessor + the design correction).
TEST(FourLevelTopologySym, MergeRule_RealEdgeIds_NoDedup) {
    std::vector<uint64_t> df = {2, 3}, ef = {10, 11};
    std::vector<uint64_t> dr = {2, 4}, er = {20, 21};  // dst 2 repeats, eids distinct
    std::vector<AdjEntry> out;
    symmetric_merge_row(df, ef, dr, er, /*has_edge_ids=*/true, out);
    ASSERT_EQ(4u, out.size());  // 2,3,2,4 — no node-id dedup
    EXPECT_EQ(2u, out[0].node_id);
    EXPECT_EQ(2u, out[2].node_id);
    EXPECT_EQ(10u, out[0].edge_id);
    EXPECT_EQ(20u, out[2].edge_id);
}

// edge_id==0 -> node-id dedup; the in-side duplicate of an out node is dropped.
TEST(FourLevelTopologySym, MergeRule_ZeroEdgeIds_NodeDedup) {
    std::vector<uint64_t> df = {2, 3}, ef = {0, 0};
    std::vector<uint64_t> dr = {2, 4}, er = {0, 0};
    std::vector<AdjEntry> out;
    symmetric_merge_row(df, ef, dr, er, /*has_edge_ids=*/false, out);
    ASSERT_EQ(3u, out.size());  // 2,3,4 — dst 2 from rev dropped
    EXPECT_EQ(2u, out[0].node_id);
    EXPECT_EQ(3u, out[1].node_id);
    EXPECT_EQ(4u, out[2].node_id);
}

// Order contract: out(u) first, then in(u) survivors.
TEST(FourLevelTopologySym, MergeRule_OrderOutThenIn) {
    std::vector<uint64_t> df = {5}, ef = {0};
    std::vector<uint64_t> dr = {4, 5}, er = {0, 0};
    std::vector<AdjEntry> out;
    symmetric_merge_row(df, ef, dr, er, /*has_edge_ids=*/false, out);
    ASSERT_EQ(2u, out.size());
    EXPECT_EQ(5u, out[0].node_id);  // from out
    EXPECT_EQ(4u, out[1].node_id);  // from in, 5-dup dropped
}

// ---------------------------------------------------------------------------
// Task 13 — get_neighbors(UNDIRECTED) with the sym tier NOT built falls back to
// the out+in merge keyed by the same rule. zero edge_ids -> node-id dedup.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologySym, Undirected_FallbackMerge_NodeDedup) {
    const std::vector<uint8_t> tiers = {1, 1, 1, 1};
    L1HashCache l1f(tiers), l1r(tiers);
    L2CompactCsr l2f, l2r;
    l2f.freeze();
    l2r.freeze();
    l1f.insert(0, std::vector<AdjEntry>{ {1, 0}, {2, 0} }, 0);  // out(0) = {1,2}
    l1r.insert(0, std::vector<AdjEntry>{ {2, 0}, {3, 0} }, 0);  // in(0)  = {2,3}
    FourLevelTopologyStore::Config cfg;  // UNDIRECTED
    FourLevelTopologyStore store(
        l1f, l1r, l2f, l2r,
        /*l3_fwd=*/nullptr, /*l3_rev=*/nullptr,
        /*l4_fwd=*/{}, /*l4_rev=*/{},
        tiers, [](ObjectId v) { return v.id; }, cfg);
    ASSERT_FALSE(store.is_symmetric_built());  // dispatcher ctor: no sym tier

    auto merged = store.get_neighbors(ObjectId(0));
    std::vector<uint64_t> dst, eid;
    merged.for_each_with_edge_id([&](uint64_t d, uint64_t e) {
        dst.push_back(d);
        eid.push_back(e);
    });
    EXPECT_EQ(dst, (std::vector<uint64_t>{1, 2, 3}));  // out{1,2} ++ in{3}
    EXPECT_EQ(eid, (std::vector<uint64_t>{0, 0, 0}));
}

// ---------------------------------------------------------------------------
// Task 15 — the edge_id-drop gate: dropping switches the dedup key to node-id
// (parallel/mutual edges collapse) and zeroes every edge_id; NOT dropping keeps
// the edge-id key (duplicates preserved).
// ---------------------------------------------------------------------------
TEST(FourLevelTopologySym, DropEdgeIds_ZerosEdgeIds_KeepsDstSet) {
    std::vector<uint64_t> df = {1, 2}, ef = {10, 11};
    std::vector<uint64_t> dr = {2, 3}, er = {12, 13};  // dst 2 repeats
    std::vector<AdjEntry> kept, dropped;
    symmetric_merge_row(df, ef, dr, er, /*has_edge_ids=*/true, kept);
    symmetric_merge_row(df, ef, dr, er, /*has_edge_ids=*/false, dropped);
    for (auto& e : dropped) e.edge_id = 0;

    // drop: node-id dedup -> {1,2,3}, eids zeroed.
    ASSERT_EQ(3u, dropped.size());
    EXPECT_EQ(1u, dropped[0].node_id);
    EXPECT_EQ(2u, dropped[1].node_id);
    EXPECT_EQ(3u, dropped[2].node_id);
    for (auto& e : dropped) EXPECT_EQ(0u, e.edge_id);

    // keep: edge-id key -> no node dedup (1,2,2,3).
    ASSERT_EQ(4u, kept.size());
}

// The symmetric snapshot persists dst_type_tag==0 (the bake feeds the writer
// tag-stripped values). The store must recover the real node type tag from a
// directional reader, else the GPU sampler reconstructs neighbor ObjectIds
// WITHOUT the tag and they miss the (tagged) feature RowMapping -> zero-filled.
TEST(FourLevelTopologySym, ResolveDstTag_FallsBackToDirectionalWhenSymZero) {
    EXPECT_EQ(0xD4u, resolve_symmetric_dst_tag(0x00, 0xD4, 0xD4));  // sym lost it -> fwd
    EXPECT_EQ(0xD4u, resolve_symmetric_dst_tag(0x00, 0x00, 0xD4));  // -> rev
    EXPECT_EQ(0xD4u, resolve_symmetric_dst_tag(0xD4, 0x00, 0x00));  // sym present is kept
    EXPECT_EQ(0xABu, resolve_symmetric_dst_tag(0xAB, 0xD4, 0xD4));  // any non-zero sym wins
    EXPECT_EQ(0x00u, resolve_symmetric_dst_tag(0x00, 0x00, 0x00));  // nothing to recover
}
