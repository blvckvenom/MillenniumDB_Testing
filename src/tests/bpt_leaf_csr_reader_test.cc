// Unit tests for the v3 CSR_HYBRID B+Tree leaf reader (Spec #8 T8.4).
//
// Scope: ReadTag constructor (header + offset-table validation per design
// §5.5), find_src_entry binary search, get_dst_at sequential + random-access
// decoding with cursor cache, is_chain_head accessor, and the BPTLeafBase<N>
// contract methods (get_record linear-scan + search_index).
//
// Since T8.5's writer does not yet exist, test pages are hand-synthesized:
// write the 16-byte header, then the uint16 offset table, then for each src
// entry [src_id varint, degree varint, dst0 full varint, dst1..dstk-1
// zigzag-delta varints]. Reuses BPT::varint_encode + BPT::zigzag_encode_i64
// from Spec #5 T5.4/T5.5.
//
// Spec reference: docs/superpowers/specs/2026-04-25-csr-hybrid-design.md
//   §3.1 (D1 layout), §3.3 (D3 offset table), §3.5 (D5 varint composition),
//   §5.1 (chain-head page), §5.5 (page-open validation).

#include "storage/index/bplus_tree/bplus_tree_leaf_csr.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "storage/index/bplus_tree/bplus_tree_split.h"
#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
#include "storage/index/bplus_tree/varint.h"
#include "storage/index/record.h"
#include "storage/page/page.h"

namespace {

// ----------------------------------------------------------------------------
// Test-only aligned 4 KB page buffer — same helper as the v2 reader test.
// ----------------------------------------------------------------------------
struct AlignedPageBuffer {
    alignas(64) std::array<char, Page::SIZE> bytes{};

    AlignedPageBuffer() { bytes.fill(0); }

    char*       data()       { return bytes.data(); }
    const char* data() const { return bytes.data(); }

    void set_byte(std::size_t i, uint8_t v) {
        bytes[i] = static_cast<char>(v);
    }

    uint8_t byte(std::size_t i) const {
        return static_cast<uint8_t>(bytes[i]);
    }
};

// ----------------------------------------------------------------------------
// Synthetic CSR page builder (T8.5 writer does not exist yet).
//
// Emits the v3 chain-head layout per design §5.1: 16-byte header,
// uint16 offset_table[value_count], then per-src entry
// [src_id varint, degree varint, dst0 full varint, dst1..zigzag-delta varints].
//
// This helper does NOT support has_edge_ids (flags bit 1) or chain
// continuation (flags bit 0) — those are covered by T8.5 writer tests
// once the writer lands. Simulating chain-head metadata is sufficient
// for the tests specified in the T8.4 brief.
// ----------------------------------------------------------------------------
struct SrcEntry {
    uint64_t src_id;
    std::vector<uint64_t> dsts;
};

AlignedPageBuffer build_csr_page(const std::vector<SrcEntry>& entries,
                                 uint32_t next_leaf = 0,
                                 uint32_t min_src_id_low = 0)
{
    AlignedPageBuffer page;

    const uint32_t vc = static_cast<uint32_t>(entries.size());
    const std::size_t payload_start = 16 + 2u * vc;

    // ---- Pass 1: encode each entry into a scratch stream, recording its
    //      final byte length. We need the length to compute offset_table[i].
    std::vector<std::vector<uint8_t>> entry_bytes(vc);
    for (std::size_t i = 0; i < vc; ++i) {
        const auto& e = entries[i];
        std::vector<uint8_t>& buf = entry_bytes[i];

        // src_id + degree
        uint8_t scratch[BPT::VARINT_MAX_BYTES];
        size_t n = BPT::varint_encode(e.src_id, scratch, sizeof(scratch));
        buf.insert(buf.end(), scratch, scratch + n);
        n = BPT::varint_encode(e.dsts.size(), scratch, sizeof(scratch));
        buf.insert(buf.end(), scratch, scratch + n);

        // dst list: dst[0] full varint, dst[1..] zigzag(delta) varints
        for (std::size_t k = 0; k < e.dsts.size(); ++k) {
            uint64_t to_encode;
            if (k == 0) {
                to_encode = e.dsts[k];
            } else {
                const uint64_t delta_u = e.dsts[k] - e.dsts[k - 1];
                const int64_t  delta   = static_cast<int64_t>(delta_u);
                to_encode = BPT::zigzag_encode_i64(delta);
            }
            n = BPT::varint_encode(to_encode, scratch, sizeof(scratch));
            buf.insert(buf.end(), scratch, scratch + n);
        }
    }

    // ---- Pass 2: compute offsets.
    std::vector<uint16_t> offsets(vc);
    std::size_t cursor = payload_start;
    for (std::size_t i = 0; i < vc; ++i) {
        offsets[i] = static_cast<uint16_t>(cursor);
        cursor += entry_bytes[i].size();
    }
    EXPECT_LE(cursor, Page::SIZE) << "test page overflow";

    // ---- Pass 3: serialize header.
    BPT::BPTLeafCSRHeader h{};
    h.format_version = 3;
    h.record_width   = 3;
    h.flags          = 0;
    h.reserved       = 0;
    h.value_count    = vc;
    h.next_leaf      = next_leaf;
    h.min_src_id_low = min_src_id_low;

    uint8_t raw[16];
    BPT::serialize_csr_header(h, raw);
    for (std::size_t i = 0; i < 16; ++i) {
        page.set_byte(i, raw[i]);
    }

    // ---- Pass 4: serialize offset table (uint16 LE) directly after header.
    for (std::size_t i = 0; i < vc; ++i) {
        const uint16_t o = offsets[i];
        page.set_byte(16 + 2 * i,     static_cast<uint8_t>(o & 0xFF));
        page.set_byte(16 + 2 * i + 1, static_cast<uint8_t>((o >> 8) & 0xFF));
    }

    // ---- Pass 5: write entry bodies at the computed offsets.
    for (std::size_t i = 0; i < vc; ++i) {
        const std::size_t off = offsets[i];
        for (std::size_t k = 0; k < entry_bytes[i].size(); ++k) {
            page.set_byte(off + k, entry_bytes[i][k]);
        }
    }

    return page;
}

// Build a page with zero src entries (valid — corresponds to an empty page
// in an empty projection, which is rare but well-formed).
AlignedPageBuffer build_empty_csr_page(uint32_t next_leaf = 0)
{
    return build_csr_page({}, next_leaf, 0);
}

}  // anonymous namespace


// ============================================================================
// Header validation
// ============================================================================

TEST(BPTLeafCSRReader, ReadHeader_ValidatesFormatVersion) {
    auto page = build_csr_page({{1000, {5000}}});
    page.set_byte(0, 2);   // wrong version
    EXPECT_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}

TEST(BPTLeafCSRReader, ReadHeader_ValidatesRecordWidth) {
    auto page = build_csr_page({{1000, {5000}}});
    page.set_byte(1, 2);   // record_width=2 but reader opened as N=3
    EXPECT_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}

TEST(BPTLeafCSRReader, ReadHeader_RejectsContinuationAsRoot) {
    auto page = build_csr_page({{1000, {5000}}});
    // Set the is_continuation flag bit. Reader must refuse.
    page.set_byte(2, BPT::CSRHybridFlags::kIsContinuation);
    EXPECT_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}

TEST(BPTLeafCSRReader, ReadHeader_RejectsReservedFlagBits) {
    auto page = build_csr_page({{1000, {5000}}});
    page.set_byte(2, 0x04);   // reserved bit 2 set
    EXPECT_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}

TEST(BPTLeafCSRReader, ReadHeader_RejectsNonZeroReservedByte) {
    auto page = build_csr_page({{1000, {5000}}});
    page.set_byte(3, 0x7A);   // reserved byte non-zero
    EXPECT_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}

TEST(BPTLeafCSRReader, ReadHeader_ValidPage_NoThrow) {
    auto page = build_csr_page({{1000, {5000}}});
    EXPECT_NO_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})));
}

TEST(BPTLeafCSRReader, ReadHeader_NullPointer_Rejected) {
    EXPECT_THROW(
        (BPTLeafCSR<3>(nullptr, BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}

// ============================================================================
// find_src_entry — binary search over offset table
// ============================================================================

TEST(BPTLeafCSRReader, FindSrc_SingleEntry) {
    auto page = build_csr_page({{1000, {5000, 5003}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off = 0;
    uint32_t deg = 0;
    EXPECT_TRUE(reader.find_src_entry(1000, off, deg));
    EXPECT_EQ(deg, 2u);

    // The offset returned should point past (src_id_varint, degree_varint).
    // For src=1000 and degree=2, the header is 2 bytes + 1 byte = 3 bytes
    // past the entry start. entry starts at offset 16 + 2 (offset table,
    // 1 src entry → 2 bytes) = 18. So dst stream starts at 18 + 3 = 21.
    EXPECT_EQ(off, 21u);
}

TEST(BPTLeafCSRReader, FindSrc_MultipleEntries) {
    std::vector<SrcEntry> entries = {
        {100,  {1, 2, 3}},
        {200,  {10, 11}},
        {1000, {5000, 5003}},
        {5000, {9000}},
    };
    auto page = build_csr_page(entries);
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off = 0;
    uint32_t deg = 0;

    EXPECT_TRUE(reader.find_src_entry(100, off, deg));
    EXPECT_EQ(deg, 3u);

    EXPECT_TRUE(reader.find_src_entry(200, off, deg));
    EXPECT_EQ(deg, 2u);

    EXPECT_TRUE(reader.find_src_entry(1000, off, deg));
    EXPECT_EQ(deg, 2u);

    EXPECT_TRUE(reader.find_src_entry(5000, off, deg));
    EXPECT_EQ(deg, 1u);
}

TEST(BPTLeafCSRReader, FindSrc_AbsentReturnsFalse) {
    auto page = build_csr_page({
        {100, {1}},
        {200, {2}},
        {300, {3}},
    });
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off = 0;
    uint32_t deg = 0;
    EXPECT_FALSE(reader.find_src_entry(150, off, deg));
    EXPECT_FALSE(reader.find_src_entry(50,  off, deg));
    EXPECT_FALSE(reader.find_src_entry(999, off, deg));
}

TEST(BPTLeafCSRReader, FindSrc_BoundaryCases) {
    // 8 entries: verify first, last, and a middle src each resolve.
    std::vector<SrcEntry> entries;
    for (uint64_t i = 0; i < 8; ++i) {
        entries.push_back({i * 10 + 1, {i * 100}});
    }
    auto page = build_csr_page(entries);
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off = 0;
    uint32_t deg = 0;
    // First
    EXPECT_TRUE(reader.find_src_entry(1, off, deg));
    EXPECT_EQ(deg, 1u);
    // Last
    EXPECT_TRUE(reader.find_src_entry(71, off, deg));
    EXPECT_EQ(deg, 1u);
    // Middle
    EXPECT_TRUE(reader.find_src_entry(41, off, deg));
    EXPECT_EQ(deg, 1u);
}

// ============================================================================
// get_dst_at — sequential + random access dst decoding
// ============================================================================

TEST(BPTLeafCSRReader, GetDst_FirstInAdjacency) {
    auto page = build_csr_page({{1000, {5000, 5003, 5005}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off = 0;
    uint32_t deg = 0;
    ASSERT_TRUE(reader.find_src_entry(1000, off, deg));

    uint64_t dst = 0;
    ASSERT_TRUE(reader.get_dst_at(off, deg, 0, dst));
    EXPECT_EQ(dst, 5000u);
}

TEST(BPTLeafCSRReader, GetDst_SequentialAccess) {
    auto page = build_csr_page({{1000, {5000, 5003, 5005, 5100}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off = 0;
    uint32_t deg = 0;
    ASSERT_TRUE(reader.find_src_entry(1000, off, deg));
    ASSERT_EQ(deg, 4u);

    std::vector<uint64_t> seen;
    for (uint32_t i = 0; i < deg; ++i) {
        uint64_t v = 0;
        ASSERT_TRUE(reader.get_dst_at(off, deg, i, v));
        seen.push_back(v);
    }
    EXPECT_EQ(seen, (std::vector<uint64_t>{5000, 5003, 5005, 5100}));
}

TEST(BPTLeafCSRReader, GetDst_RandomAccess_RestartsCursor) {
    auto page = build_csr_page({{1000, {5000, 5003, 5005, 5100, 6000}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off = 0;
    uint32_t deg = 0;
    ASSERT_TRUE(reader.find_src_entry(1000, off, deg));

    uint64_t v = 0;
    // Walk forward
    ASSERT_TRUE(reader.get_dst_at(off, deg, 0, v));
    EXPECT_EQ(v, 5000u);
    ASSERT_TRUE(reader.get_dst_at(off, deg, 3, v));
    EXPECT_EQ(v, 5100u);
    // Jump backwards
    ASSERT_TRUE(reader.get_dst_at(off, deg, 1, v));
    EXPECT_EQ(v, 5003u);
    // Jump forward
    ASSERT_TRUE(reader.get_dst_at(off, deg, 4, v));
    EXPECT_EQ(v, 6000u);
    // Same index (cache idempotent)
    ASSERT_TRUE(reader.get_dst_at(off, deg, 4, v));
    EXPECT_EQ(v, 6000u);
}

TEST(BPTLeafCSRReader, GetDst_OutOfRange_ReturnsFalse) {
    auto page = build_csr_page({{1000, {5000, 5003}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off = 0;
    uint32_t deg = 0;
    ASSERT_TRUE(reader.find_src_entry(1000, off, deg));

    uint64_t v = 0;
    EXPECT_FALSE(reader.get_dst_at(off, deg, 2, v));
    EXPECT_FALSE(reader.get_dst_at(off, deg, 999, v));
}

TEST(BPTLeafCSRReader, GetDst_SwitchingEntries_InvalidatesCache) {
    // Two entries. Walk through entry A's dsts, then query entry B — the
    // cursor cache must not leak A's state into B's decode.
    auto page = build_csr_page({
        {100, {50, 55, 60}},
        {200, {500, 505, 510}},
    });
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    uint32_t off_a = 0, deg_a = 0;
    ASSERT_TRUE(reader.find_src_entry(100, off_a, deg_a));
    uint64_t v = 0;
    ASSERT_TRUE(reader.get_dst_at(off_a, deg_a, 0, v));
    EXPECT_EQ(v, 50u);
    ASSERT_TRUE(reader.get_dst_at(off_a, deg_a, 1, v));
    EXPECT_EQ(v, 55u);

    uint32_t off_b = 0, deg_b = 0;
    ASSERT_TRUE(reader.find_src_entry(200, off_b, deg_b));
    ASSERT_TRUE(reader.get_dst_at(off_b, deg_b, 0, v));
    EXPECT_EQ(v, 500u);
    ASSERT_TRUE(reader.get_dst_at(off_b, deg_b, 2, v));
    EXPECT_EQ(v, 510u);
}

// ============================================================================
// Chain head accessors
// ============================================================================

TEST(BPTLeafCSRReader, ChainHead_Detection) {
    auto page = build_csr_page({{1, {2}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_TRUE(reader.is_chain_head());
}

TEST(BPTLeafCSRReader, NextLeaf_ZeroWhenLast) {
    auto page = build_csr_page({{1, {2}}}, /*next_leaf=*/0);
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_EQ(reader.next_leaf(), 0u);
    EXPECT_FALSE(reader.has_next());
}

TEST(BPTLeafCSRReader, NextLeaf_NonZero_HasNextTrue) {
    auto page = build_csr_page({{1, {2}}}, /*next_leaf=*/42);
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_EQ(reader.next_leaf(), 42u);
    EXPECT_TRUE(reader.has_next());
}

TEST(BPTLeafCSRReader, SrcEntryCount_ReportsHeaderValueCount) {
    auto page = build_csr_page({
        {1, {10}}, {2, {20}}, {3, {30}},
    });
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_EQ(reader.src_entry_count(), 3u);
}

// ============================================================================
// BPTLeafBase<N> contract: get_record / search_index iterate all tuples
// ============================================================================

TEST(BPTLeafCSRReader, GetRecord_IteratesAllTuples) {
    std::vector<SrcEntry> entries = {
        {100,  {50, 55, 60}},
        {200,  {500, 510}},
        {1000, {9999}},
    };
    auto page = build_csr_page(entries);
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    EXPECT_EQ(reader.get_value_count(), 6u);   // 3 + 2 + 1

    std::vector<std::pair<uint64_t, uint64_t>> expected = {
        {100, 50}, {100, 55}, {100, 60},
        {200, 500}, {200, 510},
        {1000, 9999},
    };
    for (uint_fast32_t pos = 0; pos < expected.size(); ++pos) {
        const auto r = reader.get_record(pos);
        EXPECT_EQ(r[0], expected[pos].first)  << "at pos " << pos;
        EXPECT_EQ(r[1], expected[pos].second) << "at pos " << pos;
        // Edge_id is 0 in T8.4 read path (flags bit 1 has_edge_ids path is
        // scope of T8.5+; design §3.4 comment in decode_tuple_).
    }
}

TEST(BPTLeafCSRReader, SearchIndex_FindsFirstGreaterOrEqual) {
    std::vector<SrcEntry> entries = {
        {100,  {50, 60}},
        {200,  {500}},
    };
    auto page = build_csr_page(entries);
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    // Exact match at first tuple
    EXPECT_EQ(reader.search_index(Record<3>{100, 50, 0}), 0u);
    // Exact match at a middle tuple
    EXPECT_EQ(reader.search_index(Record<3>{100, 60, 0}), 1u);
    // Between two tuples: 100,55 is > (100,50) and < (100,60) → pos=1
    EXPECT_EQ(reader.search_index(Record<3>{100, 55, 0}), 1u);
    // After all: returns total tuple count
    EXPECT_EQ(reader.search_index(Record<3>{999, 0, 0}), 3u);
    // Before all: returns 0
    EXPECT_EQ(reader.search_index(Record<3>{1, 0, 0}), 0u);
}

TEST(BPTLeafCSRReader, LinearScan_VisitsAllInOrder) {
    std::vector<SrcEntry> entries = {
        {1,    {100, 101, 102, 103}},
        {2,    {200}},
        {3,    {300, 301}},
        {1000, {9000, 9001, 9002, 9003, 9004}},
    };
    auto page = build_csr_page(entries);
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});

    ASSERT_EQ(reader.get_value_count(), 12u);  // 4+1+2+5

    // Flatten input for comparison.
    std::vector<std::pair<uint64_t, uint64_t>> input;
    for (const auto& e : entries) {
        for (auto d : e.dsts) {
            input.emplace_back(e.src_id, d);
        }
    }

    for (uint_fast32_t pos = 0; pos < input.size(); ++pos) {
        const auto r = reader.get_record(pos);
        EXPECT_EQ(r[0], input[pos].first)  << "at pos " << pos;
        EXPECT_EQ(r[1], input[pos].second) << "at pos " << pos;
    }
}

// ============================================================================
// Immutability (I6)
// ============================================================================

TEST(BPTLeafCSRReader, Insert_ThrowsLogicError) {
    auto page = build_csr_page({{1, {2}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    bool err = false;
    EXPECT_THROW(reader.insert(Record<3>{9, 9, 9}, err), std::logic_error);
}

TEST(BPTLeafCSRReader, DeleteRecord_ThrowsLogicError) {
    auto page = build_csr_page({{1, {2}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_THROW(reader.delete_record(Record<3>{1, 2, 0}), std::logic_error);
}

TEST(BPTLeafCSRReader, UpdateToNextLeaf_ThrowsLogicError) {
    auto page = build_csr_page({{1, {2}}});
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_THROW(reader.update_to_next_leaf(), std::logic_error);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(BPTLeafCSRReader, EmptyPage_ValueCount0_Valid) {
    auto page = build_empty_csr_page();
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_EQ(reader.src_entry_count(), 0u);
    EXPECT_EQ(reader.get_value_count(), 0u);

    uint32_t off = 0, deg = 0;
    EXPECT_FALSE(reader.find_src_entry(42, off, deg));
}

TEST(BPTLeafCSRReader, CheckRange_EmptyPage_False) {
    auto page = build_empty_csr_page();
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_FALSE(reader.check_range(Record<3>{1, 0, 0}));
}

TEST(BPTLeafCSRReader, CheckRange_InRange_True) {
    auto page = build_csr_page({
        {10, {100, 101}},
        {20, {200}},
    });
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_TRUE (reader.check_range(Record<3>{15,  0, 0}));
    EXPECT_FALSE(reader.check_range(Record<3>{ 1,  0, 0}));
    EXPECT_FALSE(reader.check_range(Record<3>{99,  0, 0}));
}

TEST(BPTLeafCSRReader, CheckDiagnostic_WellFormedPageOk) {
    auto page = build_csr_page({
        {10, {100}},
        {20, {200, 201}},
    });
    BPTLeafCSR<3> reader(page.data(), BPTLeafCSR<3>::ReadTag{});
    std::ostringstream os;
    EXPECT_TRUE(reader.check(os));
}

// ============================================================================
// Offset-table corruption detection (I8)
// ============================================================================

TEST(BPTLeafCSRReader, OffsetTable_NonMonotonic_Rejected) {
    auto page = build_csr_page({
        {10, {1}}, {20, {2}}, {30, {3}},
    });
    // Swap offsets for entries 1 and 2 so offset_table is non-monotonic.
    // Offset table lives at bytes [16, 16+2*3) = [16, 22).
    const uint8_t b1_lo = page.byte(18);
    const uint8_t b1_hi = page.byte(19);
    const uint8_t b2_lo = page.byte(20);
    const uint8_t b2_hi = page.byte(21);
    page.set_byte(18, b2_lo);
    page.set_byte(19, b2_hi);
    page.set_byte(20, b1_lo);
    page.set_byte(21, b1_hi);

    EXPECT_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}

TEST(BPTLeafCSRReader, OffsetTable_PointsBeforePayload_Rejected) {
    auto page = build_csr_page({{10, {1}}});
    // Set offset_table[0] to 10 (inside the header). Byte 16 is the LSB of
    // offset_table[0]; byte 17 is the MSB.
    page.set_byte(16, 10);
    page.set_byte(17, 0);
    EXPECT_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}

TEST(BPTLeafCSRReader, OffsetTable_PointsPastPageEnd_Rejected) {
    auto page = build_csr_page({{10, {1}}});
    // Set offset_table[0] to 5000 (> 4096).
    page.set_byte(16, static_cast<uint8_t>(5000 & 0xFF));
    page.set_byte(17, static_cast<uint8_t>((5000 >> 8) & 0xFF));
    EXPECT_THROW(
        (BPTLeafCSR<3>(page.data(), BPTLeafCSR<3>::ReadTag{})),
        BPT::BPTLeafCSRDecodeException);
}
