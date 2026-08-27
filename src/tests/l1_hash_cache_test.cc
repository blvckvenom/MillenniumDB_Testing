// l1_hash_cache_test.cc
//
// Four-Level Topology Store — L1HashCache unit tests.
// L1HashCache is the hottest tier of the Four-Level Topology Store
// (L1 RAM hash / L2 compact uint32 CSR / L3 mmap sidecar / L4 direct B+Tree).
// It holds the highest-frequency hub nodes in a plain unordered_map so that
// get_out_neighbors() on those nodes is an O(1) hash lookup (~10-20 ns).
//
// Coverage:
//   1. EmptyCache_GetMissReturnsEmpty — fresh cache reports miss as empty span.
//   2. InsertRespectsTierAssignment — only nodes flagged tier=1 land; tier 2/3
//      inserts silently no-op so the orchestrator can iterate uniformly.
//   3. RoundTrip_NaturalOrientation — insert then get returns the exact same
//      AdjEntry sequence.
//   4. LargeDegree_NoTruncation — node with 10_000 neighbors round-trips.
//   5. IsolatedNode_EmptyNeighbors — inserted empty list returns empty span,
//      contains() still reports true (cache holds the entry, just zero edges).
//   6. TotalBytes_MatchesL1Contract — total_bytes() honors
//      kL1NodeFixedOverhead + kL1PerEdgeBytes * degree, the sizing
//      constants declared in topology_frequency_profiler.h.
//
// The cache is a pure data structure (no DB / System / ProjectionStorage
// dependency), so these tests need none of the heavy fixtures used by
// topology_accessor_adjacency_cache_test.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/adj_entry.h"
#include "gnn/projection/l1_hash_cache.h"
#include "gnn/projection/topology_frequency_profiler.h"  // kL1* constants

using mdb::gnn::AdjEntry;
using mdb::gnn::L1HashCache;

namespace {

// Helper: build a small AdjEntry list with a deterministic shape so failures
// log readable diffs.
std::vector<AdjEntry> make_neighbors(uint64_t base, std::size_t count) {
    std::vector<AdjEntry> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(AdjEntry{ base + i, base + 1000 + i });
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — EmptyCache_GetMissReturnsEmpty.
// ---------------------------------------------------------------------------
TEST(L1HashCache, EmptyCache_GetMissReturnsEmpty) {
    const std::vector<uint8_t> tiers = { 1, 1, 1 };
    L1HashCache cache(tiers);

    EXPECT_EQ(0u, cache.node_count());
    EXPECT_EQ(0u, cache.total_bytes());
    EXPECT_EQ(0u, cache.total_edges());
    EXPECT_FALSE(cache.contains(0));

    auto miss = cache.get(42);
    EXPECT_TRUE(miss.empty());
    EXPECT_EQ(nullptr, miss.data);
    EXPECT_EQ(0u, miss.size);
}

// ---------------------------------------------------------------------------
// Test 2 — InsertRespectsTierAssignment.
//
// Tier vector: [1, 2, 3, 1, 2]. Inserting at every index — only indices 0 and 3
// (tier=1) should land. Inserts at row_idx past the end of the tier vector
// should also be silent no-ops.
// ---------------------------------------------------------------------------
TEST(L1HashCache, InsertRespectsTierAssignment) {
    const std::vector<uint8_t> tiers = { 1, 2, 3, 1, 2 };
    L1HashCache cache(tiers);

    for (std::size_t i = 0; i < tiers.size(); ++i) {
        cache.insert(/*src_node_id=*/100 + i,
                     make_neighbors(100 + i, 3),
                     /*row_idx=*/i);
    }
    // Out-of-range row_idx — silent no-op.
    cache.insert(999, make_neighbors(999, 3), /*row_idx=*/42);

    EXPECT_EQ(2u, cache.node_count());
    EXPECT_TRUE(cache.contains(100));     // tier 1 ✓
    EXPECT_FALSE(cache.contains(101));    // tier 2 — rejected
    EXPECT_FALSE(cache.contains(102));    // tier 3 — rejected
    EXPECT_TRUE(cache.contains(103));     // tier 1 ✓
    EXPECT_FALSE(cache.contains(104));    // tier 2 — rejected
    EXPECT_FALSE(cache.contains(999));    // out-of-range — rejected

    EXPECT_EQ(6u, cache.total_edges());   // 2 nodes × 3 neighbors each
}

// ---------------------------------------------------------------------------
// Test 3 — RoundTrip_NaturalOrientation.
// ---------------------------------------------------------------------------
TEST(L1HashCache, RoundTrip_NaturalOrientation) {
    const std::vector<uint8_t> tiers = { 1 };
    L1HashCache cache(tiers);

    auto expect = make_neighbors(/*base=*/200, /*count=*/5);
    cache.insert(7, expect, /*row_idx=*/0);

    ASSERT_TRUE(cache.contains(7));
    auto got = cache.get(7);
    ASSERT_EQ(expect.size(), got.size);
    for (std::size_t i = 0; i < expect.size(); ++i) {
        EXPECT_EQ(expect[i].node_id, got.data[i].node_id) << "i=" << i;
        EXPECT_EQ(expect[i].edge_id, got.data[i].edge_id) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Test 4 — LargeDegree_NoTruncation.
//
// 10_000 neighbors in a single AdjEntry vector — exercises the implicit
// re-allocation paths inside the hash bucket without crossing any of the
// uint32 truncation thresholds the L2 layout cares about.
// ---------------------------------------------------------------------------
TEST(L1HashCache, LargeDegree_NoTruncation) {
    const std::vector<uint8_t> tiers = { 1 };
    L1HashCache cache(tiers);

    auto expect = make_neighbors(/*base=*/1, /*count=*/10000);
    cache.insert(/*src_node_id=*/0xCAFE, expect, /*row_idx=*/0);

    auto got = cache.get(0xCAFE);
    ASSERT_EQ(10000u, got.size);
    EXPECT_EQ(1u,            got.data[0].node_id);
    EXPECT_EQ(1u + 1000u,    got.data[0].edge_id);
    EXPECT_EQ(1u + 9999u,    got.data[9999].node_id);
    EXPECT_EQ(1u + 1000u + 9999u, got.data[9999].edge_id);
    EXPECT_EQ(10000u, cache.total_edges());
}

// ---------------------------------------------------------------------------
// Test 5 — IsolatedNode_EmptyNeighbors.
// ---------------------------------------------------------------------------
TEST(L1HashCache, IsolatedNode_EmptyNeighbors) {
    const std::vector<uint8_t> tiers = { 1 };
    L1HashCache cache(tiers);

    cache.insert(/*src_node_id=*/77, /*neighbors=*/{}, /*row_idx=*/0);

    EXPECT_TRUE(cache.contains(77));   // entry exists, just degree 0
    EXPECT_EQ(1u, cache.node_count());
    EXPECT_EQ(0u, cache.total_edges());

    auto got = cache.get(77);
    EXPECT_EQ(0u, got.size);
    EXPECT_TRUE(got.empty());
}

// ---------------------------------------------------------------------------
// Test 6 — TotalBytes_MatchesL1Contract.
//
// `topology_frequency_profiler.h` declares:
//   bytes(node) = kL1NodeFixedOverhead + kL1PerEdgeBytes * degree
// Three inserts with distinct degrees + one tier-rejected insert verifies
// the diagnostic stays in lockstep with the profiler's sizing math.
// ---------------------------------------------------------------------------
TEST(L1HashCache, TotalBytes_MatchesL1Contract) {
    const std::vector<uint8_t> tiers = { 1, 1, 1, 2 };
    L1HashCache cache(tiers);

    cache.insert(10, make_neighbors(10, 0),  /*row_idx=*/0);    // 0 edges
    cache.insert(11, make_neighbors(11, 5),  /*row_idx=*/1);    // 5 edges
    cache.insert(12, make_neighbors(12, 17), /*row_idx=*/2);    // 17 edges
    cache.insert(13, make_neighbors(13, 99), /*row_idx=*/3);    // tier 2 → ignored

    const std::size_t expected =
        3 * mdb::gnn::kL1NodeFixedOverhead
      + (0 + 5 + 17) * mdb::gnn::kL1PerEdgeBytes;

    EXPECT_EQ(expected, cache.total_bytes());
    EXPECT_EQ(22u,      cache.total_edges());
    EXPECT_EQ(3u,       cache.node_count());
}
