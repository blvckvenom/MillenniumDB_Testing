// l2_compact_csr_test.cc
//
// L2CompactCsr unit tests — the compact uint32 CSR that holds the warm-tier
// neighbors in the Four-Level Topology Store (L1 RAM hash / L2 compact uint32
// CSR / L3 mmap sidecar / L4 direct B+Tree).  L2 is built from a single
// full-scan of the from_to_edge or to_from_edge B+Tree, storing each src's
// neighbor list as a contiguous uint32 array (stripping the 8-bit ObjectId
// type tag from every dst and re-applying it uniformly on read).
//
// Coverage:
//   1. BuildAndLookup_RoundTrip — 100 nodes with varying degrees; freeze;
//      verify each get() returns the dst list we appended.
//   2. OutOfOrderInsert_PreservesIdentity — random insert order; lookup
//      still resolves through node_to_l2_idx_.
//   3. FreezeRejectsAddNode — calling add_node after freeze() throws.
//   4. Uint32EdgeCountGuard — the freeze() edge-COUNT cap has no executable
//      negative coverage (4 B entries are unallocatable in a unit test); the
//      per-dst VALUE narrowing guard in add_node IS exercised for real in
//      tests 8/9 below.
//   5. EmptyNode_ZeroDegreeRoundTrips — node with empty neighbor list.
//   6. TotalBytes_MatchesL2Contract — assertion against the profiler's
//      sizing contract (kL2NodeFixedOverhead + kL2PerEdgeBytes * degree).
//   7. GetPreFreezeThrows — calling get() before freeze() throws
//      std::logic_error, symmetric to add_node()'s post-freeze throw.
//   8. Uint32DstOrdinalOverflow_ThrowsOnAddNode — a dst ordinal > UINT32_MAX
//      throws std::overflow_error at add_node (the actual narrowing site);
//      UINT32_MAX itself round-trips.
//   9. HeterogeneousDstTag_Throws — mixed dst type tags (including tag-0
//      mixed with tagged) throw std::invalid_argument: the read path ORs
//      dst_type_tag_ back onto every entry, so a mixed section would be
//      silently retagged.
//
// Pure data-structure tests — no DB / System / projection dependency.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/adj_entry.h"
#include "gnn/projection/l2_compact_csr.h"
#include "gnn/projection/topology_frequency_profiler.h"  // kL2* constants

using mdb::gnn::AdjEntry;
using mdb::gnn::L2CompactCsr;

namespace {

std::vector<AdjEntry> make_neighbors(uint64_t first_dst, std::size_t count) {
    std::vector<AdjEntry> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(AdjEntry{ first_dst + i, /*edge_id=*/0 });
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — BuildAndLookup_RoundTrip.
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, BuildAndLookup_RoundTrip) {
    L2CompactCsr csr(/*num_l2_nodes_hint=*/100);

    // Per-node expected dst lists, keyed by src node id.
    std::unordered_map<uint64_t, std::vector<uint32_t>> expect;

    for (uint64_t src = 0; src < 100; ++src) {
        const std::size_t deg = static_cast<std::size_t>((src * 7) % 11);
        auto neigh = make_neighbors(/*first_dst=*/src * 1000ULL, deg);
        std::vector<uint32_t> dst32;
        dst32.reserve(deg);
        for (const auto& nb : neigh) {
            dst32.push_back(static_cast<uint32_t>(nb.node_id));
        }
        expect[src] = std::move(dst32);
        csr.add_node(src, neigh);
    }

    csr.freeze();
    EXPECT_TRUE(csr.is_frozen());
    EXPECT_EQ(100u, csr.node_count());

    std::size_t expected_total_edges = 0;
    for (const auto& kv : expect) expected_total_edges += kv.second.size();
    EXPECT_EQ(expected_total_edges, csr.edge_count());

    for (const auto& kv : expect) {
        auto span = csr.get(kv.first);
        ASSERT_EQ(kv.second.size(), span.second) << "src=" << kv.first;
        for (std::size_t i = 0; i < kv.second.size(); ++i) {
            EXPECT_EQ(kv.second[i], span.first[i]) << "src=" << kv.first
                                                    << " i=" << i;
        }
    }

    // Miss path.
    auto miss = csr.get(/*src=*/99999);
    EXPECT_EQ(nullptr, miss.first);
    EXPECT_EQ(0u, miss.second);
}

// ---------------------------------------------------------------------------
// Test 2 — OutOfOrderInsert_PreservesIdentity.
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, OutOfOrderInsert_PreservesIdentity) {
    L2CompactCsr csr;

    std::vector<uint64_t> ids;
    for (uint64_t i = 0; i < 50; ++i) ids.push_back(i);
    std::shuffle(ids.begin(), ids.end(), std::mt19937_64(0xDEADBEEF));

    for (auto src : ids) {
        csr.add_node(src, make_neighbors(src * 100ULL, /*count=*/3));
    }
    csr.freeze();

    for (uint64_t src = 0; src < 50; ++src) {
        auto span = csr.get(src);
        ASSERT_EQ(3u, span.second) << "src=" << src;
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(static_cast<uint32_t>(src * 100ULL + i), span.first[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3 — FreezeRejectsAddNode.
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, FreezeRejectsAddNode) {
    L2CompactCsr csr;
    csr.add_node(1, make_neighbors(10, 2));
    csr.freeze();

    EXPECT_THROW(csr.add_node(2, make_neighbors(20, 2)), std::logic_error);

    // Duplicates pre-freeze also throw.
    L2CompactCsr csr2;
    csr2.add_node(1, make_neighbors(10, 2));
    EXPECT_THROW(csr2.add_node(1, make_neighbors(20, 2)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Test 4 — Uint32EdgeCountGuard_Documented_NoExecutableCoverage.
//
// freeze() caps the total EDGE COUNT (col_idx_.size()) at UINT32_MAX. We
// can't actually allocate 4.3 B entries in a unit test and L2CompactCsr has
// no friend hook to fake col_idx_, so the count guard itself has no
// executable negative coverage — this test only documents that a small CSR
// freezes cleanly under the cap. The per-dst VALUE narrowing (the truncation
// that would actually alias neighbour ids) IS exercised for real in
// Uint32DstOrdinalOverflow_ThrowsOnAddNode below.
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, Uint32EdgeCountGuard_Documented_NoExecutableCoverage) {
    L2CompactCsr csr;
    csr.add_node(0, make_neighbors(0, 5));
    EXPECT_NO_THROW(csr.freeze());
    EXPECT_TRUE(csr.is_frozen());
    EXPECT_LE(csr.edge_count(),
              static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()));
}

// ---------------------------------------------------------------------------
// Test 8 — Uint32DstOrdinalOverflow_ThrowsOnAddNode.
//
// col_idx_ stores each dst as a uint32 ordinal; a 56-bit payload above
// UINT32_MAX would silently alias (wrong neighbours served from L2). The
// narrowing site is add_node, so the guard must fire there — with or
// without a type tag in bits 56..63.
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, Uint32DstOrdinalOverflow_ThrowsOnAddNode) {
    const uint64_t kOverflowOrdinal = 0x1'0000'0000ULL;  // UINT32_MAX + 1

    L2CompactCsr csr;
    EXPECT_THROW(
        csr.add_node(0, std::vector<AdjEntry>{ { kOverflowOrdinal, 0 } }),
        std::overflow_error);

    // The tag bits don't excuse a payload overflow.
    const uint64_t kTag = 0xD4ULL << 56;
    L2CompactCsr csr_tagged;
    EXPECT_THROW(
        csr_tagged.add_node(0, std::vector<AdjEntry>{ { kTag | kOverflowOrdinal, 0 } }),
        std::overflow_error);

    // Boundary: UINT32_MAX itself is representable and must round-trip.
    L2CompactCsr csr_max;
    csr_max.add_node(0, std::vector<AdjEntry>{ { 0xFFFF'FFFFULL, 0 } });
    csr_max.freeze();
    auto span = csr_max.get(0);
    ASSERT_EQ(1u, span.second);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), span.first[0]);
}

// ---------------------------------------------------------------------------
// Test 9 — HeterogeneousDstTag_Throws.
//
// The read path reconstructs every dst as dst_type_tag() | col_idx_[i], so
// all dsts in one L2 section must carry the SAME tag — including tag 0:
// a raw ordinal mixed into a tagged section would be silently retagged
// (and vice versa, earlier tag-0 entries would absorb a later tag).
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, HeterogeneousDstTag_Throws) {
    const uint64_t kTagA = 0xD4ULL << 56;
    const uint64_t kTagB = 0xD5ULL << 56;

    // Two different non-zero tags.
    L2CompactCsr csr;
    csr.add_node(0, std::vector<AdjEntry>{ { kTagA | 1u, 0 } });
    EXPECT_THROW(
        csr.add_node(1, std::vector<AdjEntry>{ { kTagB | 2u, 0 } }),
        std::invalid_argument);

    // tag-0 dst after a tagged section.
    L2CompactCsr csr_tag_then_raw;
    csr_tag_then_raw.add_node(0, std::vector<AdjEntry>{ { kTagA | 1u, 0 } });
    EXPECT_THROW(
        csr_tag_then_raw.add_node(1, std::vector<AdjEntry>{ { 2u, 0 } }),
        std::invalid_argument);

    // Tagged dst after raw tag-0 entries.
    L2CompactCsr csr_raw_then_tag;
    csr_raw_then_tag.add_node(0, std::vector<AdjEntry>{ { 1u, 0 } });
    EXPECT_THROW(
        csr_raw_then_tag.add_node(1, std::vector<AdjEntry>{ { kTagA | 2u, 0 } }),
        std::invalid_argument);

    // Uniform tags (any value, including 0) are fine.
    L2CompactCsr csr_uniform;
    csr_uniform.add_node(0, std::vector<AdjEntry>{ { kTagA | 1u, 0 }, { kTagA | 2u, 0 } });
    csr_uniform.add_node(1, std::vector<AdjEntry>{ { kTagA | 3u, 0 } });
    EXPECT_NO_THROW(csr_uniform.freeze());
    EXPECT_EQ(kTagA, csr_uniform.dst_type_tag());
}

// ---------------------------------------------------------------------------
// Test 5 — EmptyNode_ZeroDegreeRoundTrips.
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, EmptyNode_ZeroDegreeRoundTrips) {
    L2CompactCsr csr;
    csr.add_node(/*src=*/42, /*neighbors=*/{});
    csr.add_node(/*src=*/43, make_neighbors(100, /*count=*/3));
    csr.add_node(/*src=*/44, /*neighbors=*/{});
    csr.freeze();

    EXPECT_EQ(3u, csr.node_count());
    EXPECT_EQ(3u, csr.edge_count());

    auto a = csr.get(42);
    EXPECT_EQ(0u, a.second);

    auto b = csr.get(43);
    ASSERT_EQ(3u, b.second);
    EXPECT_EQ(100u, b.first[0]);
    EXPECT_EQ(101u, b.first[1]);
    EXPECT_EQ(102u, b.first[2]);

    auto c = csr.get(44);
    EXPECT_EQ(0u, c.second);
}

// ---------------------------------------------------------------------------
// Test 6 — TotalBytes_MatchesL2Contract.
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, TotalBytes_MatchesL2Contract) {
    L2CompactCsr csr;
    csr.add_node(0, make_neighbors(0, 0));     // 0 edges
    csr.add_node(1, make_neighbors(10, 5));    // 5 edges
    csr.add_node(2, make_neighbors(20, 17));   // 17 edges
    csr.freeze();

    const std::size_t expected =
        3 * mdb::gnn::kL2NodeFixedOverhead
      + (0 + 5 + 17) * mdb::gnn::kL2PerEdgeBytes;

    EXPECT_EQ(expected, csr.total_bytes());
}

// ---------------------------------------------------------------------------
// Test 7 — GetPreFreezeThrows.
//
// Symmetric to FreezeRejectsAddNode: verifies that calling get() before
// freeze() throws std::logic_error rather than silently returning
// (nullptr, 0). Pre-freeze the row_ptr_ prefix sum has not been built,
// so a silent miss for an already-added node would mask orchestrator
// bugs — the project's "fail loud" discipline rejects this foot-gun.
// ---------------------------------------------------------------------------
TEST(L2CompactCsr, GetPreFreezeThrows) {
    L2CompactCsr csr;
    csr.add_node(/*src=*/7, make_neighbors(/*first_dst=*/100, /*count=*/3));

    // The src exists in node_to_l2_idx_ but the structure is not frozen,
    // so get() must throw rather than misleadingly return (nullptr, 0).
    EXPECT_THROW(csr.get(7), std::logic_error);

    // After freeze, lookup succeeds normally.
    csr.freeze();
    auto span = csr.get(7);
    ASSERT_EQ(3u, span.second);
    EXPECT_EQ(100u, span.first[0]);
}
