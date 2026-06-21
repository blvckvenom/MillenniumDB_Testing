// Unit tests for the BPTLeafV2 write-path encoder (delta + LEB128-varint
// B+Tree leaf compression).
//
// Scope: append_record + flush byte-for-byte determinism against the
// hand-computed examples in design doc §5.3, overflow handling, next_leaf
// round-trip through the 16-byte header, padding invariants, and flush
// idempotency. The read path (get_record, search_index, etc.) is stubbed
// in the writer and is NOT exercised here — those stubs are fully
// implemented in bpt_leaf_v2_reader_test.cc.
//
// Spec reference: docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md
//                 (§5.2 layout, §5.3 worked example, §5.4 padding).

#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"

#include <array>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/record.h"
#include "storage/page/page.h"

namespace {

// Test-only aligned 4 KB page buffer — production Page bytes come from
// BufferManager, but these isolated unit tests drive the writer directly
// through its raw `char*` constructor to avoid standing up a full buffer
// pool. The buffer is zero-filled so any byte that's not overwritten by
// flush() remains provably 0x00.
struct AlignedPageBuffer {
    alignas(64) std::array<char, Page::SIZE> bytes{};

    AlignedPageBuffer() { bytes.fill(0); }

    char*       data()       { return bytes.data(); }
    const char* data() const { return bytes.data(); }

    uint8_t byte(size_t i) const {
        return static_cast<uint8_t>(bytes[i]);
    }
};

// Helper: decode uint32_t little-endian from `buf + off`.
uint32_t read_u32_le(const char* buf, size_t off) {
    return static_cast<uint32_t>(static_cast<uint8_t>(buf[off]))
         | (static_cast<uint32_t>(static_cast<uint8_t>(buf[off + 1])) <<  8)
         | (static_cast<uint32_t>(static_cast<uint8_t>(buf[off + 2])) << 16)
         | (static_cast<uint32_t>(static_cast<uint8_t>(buf[off + 3])) << 24);
}

// Helper: compare a slice of the page buffer against an expected byte list.
::testing::AssertionResult BytesEqual(const AlignedPageBuffer& page,
                                      size_t offset,
                                      const std::vector<uint8_t>& expected)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        const uint8_t got = page.byte(offset + i);
        if (got != expected[i]) {
            return ::testing::AssertionFailure()
                << "byte[" << (offset + i) << "] expected 0x"
                << std::hex << static_cast<unsigned>(expected[i])
                << " but got 0x" << static_cast<unsigned>(got);
        }
    }
    return ::testing::AssertionSuccess();
}

// ====================== TESTS ================================================

TEST(BPTLeafV2Writer, WritesRecord0AsFullVarints) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    Record<3> r0 = {1000, 2000, 3000};
    ASSERT_TRUE(writer.append_record(r0));
    writer.flush();

    // Record 0 starts at offset 16 (after the 16-byte header).
    EXPECT_TRUE(BytesEqual(page, 16, {0xE8, 0x07, 0xD0, 0x0F, 0xB8, 0x17}));
    EXPECT_EQ(writer.value_count(), 1u);
}

TEST(BPTLeafV2Writer, WritesRecord1AsZigzagDeltas) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    ASSERT_TRUE(writer.append_record(Record<3>{1000, 2000, 3000}));
    ASSERT_TRUE(writer.append_record(Record<3>{1000, 2001, 3005}));
    writer.flush();

    // Record 0 is 6 bytes starting at offset 16 (0xE8 0x07 0xD0 0x0F 0xB8 0x17).
    // Record 1 follows immediately — expected delta zigzag bytes: 0x00 0x02 0x0A.
    EXPECT_TRUE(BytesEqual(page, 22, {0x00, 0x02, 0x0A}));
    EXPECT_EQ(writer.value_count(), 2u);
}

TEST(BPTLeafV2Writer, ThreeRecords_ExactBytesMatch) {
    // Golden test reproducing the worked example from design §5.3.
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    ASSERT_TRUE(writer.append_record(Record<3>{1000, 2000, 3000}));
    ASSERT_TRUE(writer.append_record(Record<3>{1000, 2001, 3005}));
    ASSERT_TRUE(writer.append_record(Record<3>{1001,  500, 3100}));
    writer.flush();

    // Header: 02 03 00 00 | 03 00 00 00 | 00 00 00 00 | 00 00 00 00
    EXPECT_TRUE(BytesEqual(page, 0, {
        0x02, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    }));

    // Record 0 (6 bytes at offset 16..21): E8 07 D0 0F B8 17
    EXPECT_TRUE(BytesEqual(page, 16, {0xE8, 0x07, 0xD0, 0x0F, 0xB8, 0x17}));
    // Record 1 (3 bytes at offset 22..24): 00 02 0A
    EXPECT_TRUE(BytesEqual(page, 22, {0x00, 0x02, 0x0A}));
    // Record 2 (5 bytes at offset 25..29): 02 B9 17 BE 01
    EXPECT_TRUE(BytesEqual(page, 25, {0x02, 0xB9, 0x17, 0xBE, 0x01}));

    // Padding: bytes 30..4095 must be zero.
    for (size_t i = 30; i < Page::SIZE; ++i) {
        ASSERT_EQ(page.byte(i), 0u) << "non-zero padding at offset " << i;
    }

    EXPECT_EQ(writer.value_count(), 3u);
    EXPECT_EQ(writer.bytes_used(), 16u + 6u + 3u + 5u);
}

TEST(BPTLeafV2Writer, OverflowFlushesPage) {
    // Fill the page until append_record returns false. Each record here has
    // a non-trivial size (records against a shifting cursor); we just need
    // enough to overflow the 4080-byte payload budget.
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());

    uint64_t i = 0;
    uint32_t accepted = 0;
    bool overflow_seen = false;
    // Hard upper bound on iterations to prevent a runaway loop if overflow
    // never occurs (it always does in under ~1500 iterations for random-ish
    // deltas, but we cap anyway).
    while (i < 10000) {
        // Use increasing records with large deltas so each encodes to
        // non-minimum bytes. Large deltas on both secondary fields keep the
        // per-record cost up.
        Record<3> r = {i, i * 1000 + 7, i * 7 + 13};
        if (!writer.append_record(r)) {
            overflow_seen = true;
            break;
        }
        ++accepted;
        ++i;
    }
    ASSERT_TRUE(overflow_seen);
    // Bytes used (header + payload) is bounded by Page::SIZE.
    EXPECT_LE(writer.bytes_used(), Page::SIZE);
    // value_count() matches the accepted-append count (no partial writes).
    EXPECT_EQ(writer.value_count(), accepted);
    writer.flush();
    // Header value_count byte range also reports exactly `accepted`.
    EXPECT_EQ(read_u32_le(page.data(), 4), accepted);
}

TEST(BPTLeafV2Writer, Record_N1_FullRange) {
    AlignedPageBuffer page;
    BPTLeafV2<1> writer(page.data());
    // Varints of these values:
    //   0           -> 0x00                        (1 byte)
    //   128         -> 0x80 0x01                   (2 bytes)
    //   16384       -> 0x80 0x80 0x01              (3 bytes)
    //   2097152     -> 0x80 0x80 0x80 0x01         (4 bytes)
    //   UINT64_MAX  -> zigzag(delta = UINT64_MAX - 2097152) = ?, deltas get
    //                  interesting but the FIRST record is the only full varint.
    ASSERT_TRUE(writer.append_record(Record<1>{0}));        // full varint: 0x00
    ASSERT_TRUE(writer.append_record(Record<1>{128}));      // delta=128  -> zig=256
    ASSERT_TRUE(writer.append_record(Record<1>{16384}));    // delta=16256 -> zig=32512
    ASSERT_TRUE(writer.append_record(Record<1>{2097152}));  // delta=2080768 -> zig=4161536
    ASSERT_TRUE(writer.append_record(Record<1>{UINT64_MAX}));
    writer.flush();

    EXPECT_EQ(writer.value_count(), 5u);
    // Header value_count field at offset 4..7 == 5.
    EXPECT_EQ(read_u32_le(page.data(), 4), 5u);
    // Record 0 first byte at offset 16 is 0x00 (varint of 0).
    EXPECT_EQ(page.byte(16), 0x00u);
}

TEST(BPTLeafV2Writer, Record_N2_SecondaryDeltaNegative) {
    AlignedPageBuffer page;
    BPTLeafV2<2> writer(page.data());
    ASSERT_TRUE(writer.append_record(Record<2>{5, 1000}));
    ASSERT_TRUE(writer.append_record(Record<2>{6,  100}));
    writer.flush();

    // Record 0: varint(5)=0x05, varint(1000)=0xE8 0x07. Total 3 bytes.
    EXPECT_TRUE(BytesEqual(page, 16, {0x05, 0xE8, 0x07}));

    // Record 1:
    //   delta_0 = 6 - 5 = 1       -> zigzag(1) = 2      -> varint 0x02.
    //   delta_1 = 100 - 1000 = -900 -> zigzag(-900) = 1799 -> varint 0x87 0x0E.
    //     1799 = 0b11100000111
    //     LEB128: low 7 bits = 0b0000111 = 0x07 | 0x80 (continuation) = 0x87
    //             next 7 bits = 0b0001110 = 0x0E (terminator).
    EXPECT_TRUE(BytesEqual(page, 19, {0x02, 0x87, 0x0E}));
    EXPECT_EQ(writer.value_count(), 2u);
}

TEST(BPTLeafV2Writer, Record_N3_AllSame) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(writer.append_record(Record<3>{42, 43, 44}));
    }
    writer.flush();

    // Record 0 full varints: 42 -> 0x2A; 43 -> 0x2B; 44 -> 0x2C. 3 bytes.
    EXPECT_TRUE(BytesEqual(page, 16, {0x2A, 0x2B, 0x2C}));

    // Records 1..4: each delta is 0 -> zigzag(0)=0 -> varint 0x00 (1 byte).
    // Each record is 3 bytes of 0x00.
    for (int rec = 1; rec <= 4; ++rec) {
        const size_t off = 16 + 3 + (rec - 1) * 3;
        EXPECT_TRUE(BytesEqual(page, off, {0x00, 0x00, 0x00}))
            << "record " << rec << " did not compress to three 0x00 bytes";
    }

    EXPECT_EQ(writer.value_count(), 5u);
    EXPECT_EQ(writer.bytes_used(), 16u + 3u + 4u * 3u);  // 31 bytes
}

TEST(BPTLeafV2Writer, Record_N3_Monotonic) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    for (uint64_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(writer.append_record(Record<3>{i, 2 * i, 3 * i}));
    }
    writer.flush();

    EXPECT_EQ(writer.value_count(), 100u);

    // Record 0: varint(0),varint(0),varint(0) = 3 bytes.
    // Records 1..99: delta=(1, 2, 3) in each field.
    //   zigzag(1)=2 -> varint 0x02
    //   zigzag(2)=4 -> varint 0x04
    //   zigzag(3)=6 -> varint 0x06
    // Each record 1..99 is exactly 3 bytes.
    const size_t expected_payload = 3u + 99u * 3u;  // 300 bytes
    EXPECT_EQ(writer.bytes_used(), 16u + expected_payload);

    // Spot-check: record 1 at offset 16+3 = 19 should be 0x02 0x04 0x06.
    EXPECT_TRUE(BytesEqual(page, 19, {0x02, 0x04, 0x06}));
}

TEST(BPTLeafV2Writer, MultiPageNextLeafChained) {
    // Single-page test: next_leaf = 42 round-trips through the header.
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data(), /*next_leaf=*/42);
    ASSERT_TRUE(writer.append_record(Record<3>{1, 2, 3}));
    writer.flush();

    // Header bytes 8..11 decode to 42 LE.
    EXPECT_EQ(read_u32_le(page.data(), 8), 42u);
    EXPECT_TRUE(writer.has_next());
}

TEST(BPTLeafV2Writer, LastPageNextLeafZero) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());  // default next_leaf = 0
    ASSERT_TRUE(writer.append_record(Record<3>{7, 8, 9}));
    writer.flush();

    // Header bytes 8..11 all zero.
    EXPECT_EQ(read_u32_le(page.data(), 8), 0u);
    EXPECT_FALSE(writer.has_next());
}

TEST(BPTLeafV2Writer, FormatVersionByteEqualsTwo) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    ASSERT_TRUE(writer.append_record(Record<3>{100, 200, 300}));
    writer.flush();

    EXPECT_EQ(page.byte(0), 0x02u);  // format_version at offset 0
    EXPECT_EQ(page.byte(1), 0x03u);  // record_width at offset 1 (N=3)
}

TEST(BPTLeafV2Writer, ValueCountMatchesInsertedCount) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    constexpr uint32_t kCount = 37;
    for (uint32_t i = 0; i < kCount; ++i) {
        ASSERT_TRUE(writer.append_record(Record<3>{i, i + 1, i + 2}));
    }
    writer.flush();

    EXPECT_EQ(writer.value_count(), kCount);
    EXPECT_EQ(read_u32_le(page.data(), 4), kCount);
}

TEST(BPTLeafV2Writer, PaddingZeroedAfterLastRecord) {
    AlignedPageBuffer page;
    // Pre-stain the buffer so we can distinguish "not touched" from
    // "explicitly zeroed by flush".
    for (size_t i = 0; i < Page::SIZE; ++i) {
        page.bytes[i] = static_cast<char>(0xAB);
    }

    BPTLeafV2<3> writer(page.data());
    ASSERT_TRUE(writer.append_record(Record<3>{1000, 2000, 3000}));
    ASSERT_TRUE(writer.append_record(Record<3>{1000, 2001, 3005}));
    ASSERT_TRUE(writer.append_record(Record<3>{1001,  500, 3100}));
    writer.flush();

    // Used bytes: 30. Every byte from 30..4095 must be zero after flush.
    for (size_t i = 30; i < Page::SIZE; ++i) {
        ASSERT_EQ(page.byte(i), 0u)
            << "non-zero padding at offset " << i
            << " (flush did not zero-fill through Page::SIZE)";
    }
}

TEST(BPTLeafV2Writer, Record_AllZero_CompressesTo_1ByteEach) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(writer.append_record(Record<3>{0, 0, 0}));
    }
    writer.flush();

    // Record 0: varint(0) x 3 = 0x00 0x00 0x00 (3 bytes).
    EXPECT_TRUE(BytesEqual(page, 16, {0x00, 0x00, 0x00}));
    // Records 1..4: each field delta is 0, zigzag(0)=0, varint 0x00. Each
    // record is also 3 bytes of 0x00.
    for (int rec = 1; rec <= 4; ++rec) {
        const size_t off = 16 + rec * 3;
        EXPECT_TRUE(BytesEqual(page, off, {0x00, 0x00, 0x00}));
    }
    // Total payload = 5 * 3 = 15 bytes; bytes_used = 16 + 15 = 31.
    EXPECT_EQ(writer.bytes_used(), 31u);
}

TEST(BPTLeafV2Writer, FlushIdempotent) {
    AlignedPageBuffer page;
    BPTLeafV2<3> writer(page.data());
    ASSERT_TRUE(writer.append_record(Record<3>{1, 2, 3}));
    writer.flush();

    // Capture page contents after the first flush.
    std::array<char, Page::SIZE> snapshot = page.bytes;

    // Second flush must be a no-op: page bytes unchanged.
    writer.flush();
    for (size_t i = 0; i < Page::SIZE; ++i) {
        ASSERT_EQ(page.bytes[i], snapshot[i])
            << "flush() was not idempotent at offset " << i;
    }

    // Post-flush appends must fail.
    EXPECT_FALSE(writer.append_record(Record<3>{4, 5, 6}));
    EXPECT_TRUE(writer.is_flushed());
}

}  // namespace


// ============================================================================
// Tests for BPTLeafV2Writer<N> — the bulk-load sibling of BPTLeafWriter<N>
// that uses the delta + LEB128-varint leaf compression format. Scope:
// file-level streaming API that wraps a BPTLeafV2<N> and emits a chained
// sequence of 4 KB pages on overflow, mirroring BPTLeafWriter's external
// contract (process_block + make_empty) but record-at-a-time instead of
// page-at-a-time.
// ============================================================================

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "storage/index/bplus_tree/bpt_mem_import.h"

namespace {

namespace fs = std::filesystem;

// Pick a unique scratch path for this test binary run.
std::string scratch_leaf_path(const char* tag) {
    const auto dir = fs::temp_directory_path()
                   / ("mdb_bpt_leaf_v2_writer_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    return (dir / (std::string(tag) + ".leaf")).string();
}

// Read entire file contents into a vector of bytes.
std::vector<uint8_t> read_all_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    const auto sz = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(sz);
    in.read(reinterpret_cast<char*>(out.data()), sz);
    return out;
}

TEST(BPTLeafV2BulkWriter, MakeEmpty_ProducesSinglePage_ByteZeroIsTwo) {
    const auto path = scratch_leaf_path("MakeEmpty");
    {
        BPTLeafV2Writer<3> w(path);
        w.make_empty();
    }
    const auto bytes = read_all_bytes(path);
    ASSERT_EQ(bytes.size(), Page::SIZE);
    EXPECT_EQ(bytes[0], 0x02);          // format_version = 2
    EXPECT_EQ(bytes[1], 3);             // record_width = N
    EXPECT_EQ(bytes[2], 0);             // flags reserved
    EXPECT_EQ(bytes[3], 0);             // reserved
    // value_count (offset 4..7) = 0
    EXPECT_EQ(bytes[4], 0); EXPECT_EQ(bytes[5], 0);
    EXPECT_EQ(bytes[6], 0); EXPECT_EQ(bytes[7], 0);
    // next_leaf (offset 8..11) = 0 (terminal page)
    EXPECT_EQ(bytes[8], 0); EXPECT_EQ(bytes[9], 0);
    EXPECT_EQ(bytes[10], 0); EXPECT_EQ(bytes[11], 0);
    fs::remove(path);
}

TEST(BPTLeafV2BulkWriter, AppendSinglePage_ValueCountAndNextLeafCorrect) {
    const auto path = scratch_leaf_path("SinglePage");
    {
        BPTLeafV2Writer<3> w(path);
        EXPECT_FALSE(w.append_record(Record<3>{1, 2, 3}));     // fits
        EXPECT_FALSE(w.append_record(Record<3>{10, 20, 30}));  // fits
        EXPECT_FALSE(w.append_record(Record<3>{100, 200, 300}));// fits
        w.finalize();
    }
    const auto bytes = read_all_bytes(path);
    ASSERT_EQ(bytes.size(), Page::SIZE);
    EXPECT_EQ(bytes[0], 0x02);
    EXPECT_EQ(bytes[1], 3);
    // value_count LE
    const uint32_t vc = static_cast<uint32_t>(bytes[4])
                      | (static_cast<uint32_t>(bytes[5]) << 8)
                      | (static_cast<uint32_t>(bytes[6]) << 16)
                      | (static_cast<uint32_t>(bytes[7]) << 24);
    EXPECT_EQ(vc, 3u);
    // next_leaf = 0 (tail page)
    const uint32_t nl = static_cast<uint32_t>(bytes[8])
                      | (static_cast<uint32_t>(bytes[9]) << 8)
                      | (static_cast<uint32_t>(bytes[10]) << 16)
                      | (static_cast<uint32_t>(bytes[11]) << 24);
    EXPECT_EQ(nl, 0u);
    fs::remove(path);
}

TEST(BPTLeafV2BulkWriter, AppendManyRecords_CrossesPageBoundaries) {
    const auto path = scratch_leaf_path("ManyRecords");
    constexpr int kNumRecords = 8000;  // forces multiple pages for N=3
    std::size_t page_breaks = 0;
    {
        BPTLeafV2Writer<3> w(path);
        for (int i = 0; i < kNumRecords; ++i) {
            // Dense sorted records with small deltas — best case for v2 compression.
            if (w.append_record(Record<3>{
                    static_cast<uint64_t>(i),
                    static_cast<uint64_t>(i * 2),
                    static_cast<uint64_t>(i * 3)})) {
                ++page_breaks;
            }
        }
        w.finalize();
    }
    EXPECT_GT(page_breaks, 0u)
        << "8000 records MUST span multiple pages; writer never crossed a page boundary";
    const auto bytes = read_all_bytes(path);
    // File must be a multiple of Page::SIZE.
    ASSERT_EQ(bytes.size() % Page::SIZE, 0u);
    const std::size_t num_pages = bytes.size() / Page::SIZE;
    EXPECT_EQ(num_pages, page_breaks + 1)
        << "num_pages (" << num_pages << ") != page_breaks + 1 ("
        << (page_breaks + 1) << ")";

    // Every page must start with byte 0x02 (v2 format_version).
    for (std::size_t p = 0; p < num_pages; ++p) {
        EXPECT_EQ(bytes[p * Page::SIZE], 0x02)
            << "page " << p << " byte 0 is not 0x02";
        EXPECT_EQ(bytes[p * Page::SIZE + 1], 3)
            << "page " << p << " record_width mismatch";
    }

    // Every non-final page must have next_leaf = p + 1; final page must have 0.
    for (std::size_t p = 0; p < num_pages; ++p) {
        const std::size_t off = p * Page::SIZE + 8;
        const uint32_t nl = static_cast<uint32_t>(bytes[off])
                          | (static_cast<uint32_t>(bytes[off + 1]) << 8)
                          | (static_cast<uint32_t>(bytes[off + 2]) << 16)
                          | (static_cast<uint32_t>(bytes[off + 3]) << 24);
        if (p + 1 < num_pages) {
            EXPECT_EQ(nl, static_cast<uint32_t>(p + 1))
                << "page " << p << " next_leaf should point to " << (p + 1);
        } else {
            EXPECT_EQ(nl, 0u)
                << "final page next_leaf must be 0 (terminator)";
        }
    }
    fs::remove(path);
}

TEST(BPTLeafV2BulkWriter, RoundtripViaReadTag_RecordsDecodeCorrectly) {
    const auto path = scratch_leaf_path("Roundtrip");
    const std::vector<Record<3>> inputs = {
        {10, 20, 30},
        {11, 21, 31},
        {100, 200, 300},
        {1000, 2000, 3000},
    };
    {
        BPTLeafV2Writer<3> w(path);
        for (const auto& r : inputs) {
            w.append_record(r);
        }
        w.finalize();
    }
    const auto bytes = read_all_bytes(path);
    ASSERT_EQ(bytes.size(), Page::SIZE);

    // Re-interpret as a V2 reader over the page bytes.
    BPTLeafV2<3> reader(reinterpret_cast<const char*>(bytes.data()),
                        BPTLeafV2<3>::ReadTag{});
    ASSERT_EQ(reader.get_value_count(), inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto r = reader.get_record(static_cast<uint_fast32_t>(i));
        EXPECT_EQ(r, inputs[i]) << "record " << i << " mismatch";
    }
    fs::remove(path);
}

}  // namespace (bulk writer)
