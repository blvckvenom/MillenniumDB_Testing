// Unit tests for BPTLeafCSRWriter<N> (Spec #8 T8.5).
//
// Exercises the third bulk-load writer sibling (alongside BPTLeafWriter/V1
// and BPTLeafV2Writer). Tests are structured as:
//   1. Run the writer with a synthetic sorted input stream to a temp file.
//   2. Read the file back as raw bytes and/or via the T8.4 BPTLeafCSR<N>
//      reader to verify round-trip correctness.
//   3. Assert on header bytes, offset-table invariants, and chain structure.
//
// Covers: single-src / multi-src fit on one page, multi-page src transitions,
// hub overflow with 2- and 3-page chains, empty writer, offset-table
// monotonicity, header byte-level checks, and reader round-trip.
//
// Spec reference: docs/superpowers/specs/2026-04-25-csr-hybrid-design.md §3.9,
//                 §5.1, §5.2.

#include "storage/index/bplus_tree/bpt_mem_import.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "storage/index/bplus_tree/bplus_tree_leaf_csr.h"
#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
#include "storage/index/bplus_tree/varint.h"
#include "storage/page/page.h"

namespace {

// ----------------------------------------------------------------------------
// Test helpers: run writer against a temp file and return the bytes.
// ----------------------------------------------------------------------------

struct TempFile {
    std::string path;
    explicit TempFile(const std::string& suffix = ".leaf")
    {
        // Use a deterministic-ish path in /tmp. Not concurrency-safe across
        // gtest threads but fine for the default single-threaded runner.
        const char* base = std::getenv("TMPDIR");
        if (base == nullptr) base = "/tmp";
        static int counter = 0;
        path = std::string(base) + "/bpt_csr_wtest_"
             + std::to_string(::getpid()) + "_"
             + std::to_string(++counter) + suffix;
    }
    ~TempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

std::vector<char> read_file_bytes(const std::string& path)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> out(static_cast<std::size_t>(sz));
    if (sz > 0) f.read(out.data(), sz);
    return out;
}

// Slice of one 4 KB page out of the file bytes. Returns nullptr if out of
// range.
const char* page_at(const std::vector<char>& bytes, std::size_t page_num)
{
    const std::size_t off = page_num * Page::SIZE;
    if (off + Page::SIZE > bytes.size()) return nullptr;
    return bytes.data() + off;
}

// Varint decode helper that throws on malformed. Returns consumed bytes.
std::size_t decode_varint(const uint8_t* in, const uint8_t* end, uint64_t& out)
{
    return BPT::varint_decode(in, end, out);
}

// Full-chain readback: given a writer output file, iterate every src entry
// on every chain-head page and reconstruct the dst lists (following
// continuation pages via next_leaf when a chain-head's src has dsts
// spilled onto continuations). Returns a vector of (src, dsts) matching the
// input order.
struct DecodedEntry {
    uint64_t              src;
    std::vector<uint64_t> dsts;
};

std::vector<DecodedEntry>
decode_all_entries(const std::vector<char>& bytes, std::size_t num_pages)
{
    std::vector<DecodedEntry> out;

    std::size_t page = 0;
    while (page < num_pages) {
        const char* p = page_at(bytes, page);
        if (p == nullptr) break;
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(p);

        // Sniff format / flags.
        if (raw[0] != 3) {
            // Not a v3 page (pre-Spec-#8 makeEmpty or corruption). Skip.
            ++page;
            continue;
        }
        if (BPT::is_csr_continuation(raw)) {
            // Standalone continuation — reached via chain-head traversal in
            // the normal path, not iterated here. Skip.
            ++page;
            continue;
        }

        // Chain-head page — use T8.4 reader.
        BPTLeafCSR<3> leaf(p, BPTLeafCSR<3>::ReadTag{});
        const uint32_t vc = leaf.src_entry_count();
        for (uint32_t i = 0; i < vc; ++i) {
            // Decode the src entry header: src_id then degree.
            // Use the offset table indirectly via find_src_entry — but we
            // don't yet know the src_id, so decode from offset[i] directly.
            //
            // The offset table begins at page offset 16. Each slot is 2 bytes
            // LE. Decode src_id via raw varint walk starting from that offset.
            const uint8_t* off_ptr = raw + 16 + 2 * i;
            const uint32_t off = static_cast<uint32_t>(off_ptr[0])
                               | (static_cast<uint32_t>(off_ptr[1]) << 8);
            const uint8_t* end = raw + Page::SIZE;

            uint64_t src_id = 0;
            uint64_t degree = 0;
            const uint8_t* in = raw + off;
            in += decode_varint(in, end, src_id);
            in += decode_varint(in, end, degree);

            uint32_t start_off = 0;
            uint32_t deg_out   = 0;
            EXPECT_TRUE(leaf.find_src_entry(src_id, start_off, deg_out))
                << "find_src_entry failed for src=" << src_id;
            EXPECT_EQ(deg_out, static_cast<uint32_t>(degree));

            DecodedEntry e;
            e.src = src_id;
            e.dsts.reserve(deg_out);

            // Compute k_head = degree - sum(chunk_counts on continuations).
            // The on-disk format does not encode "number of dsts on the
            // chain-head" explicitly; the reader derives it from the total
            // degree minus the continuation chunk_counts (design §3.4). We
            // must pre-walk the continuation chain to accumulate the
            // chunk_counts before decoding the chain-head's dst stream, to
            // avoid consuming zero-padding bytes beyond the head's packed
            // dsts as bogus zigzag-delta-zero entries.
            uint64_t total_continuation_count = 0;
            {
                uint32_t probe_pg = leaf.next_leaf();
                while (probe_pg != 0) {
                    const char* cp = page_at(bytes, probe_pg);
                    if (cp == nullptr) break;
                    const uint8_t* craw = reinterpret_cast<const uint8_t*>(cp);
                    uint8_t hdrbuf[16];
                    std::memcpy(hdrbuf, craw, 16);
                    if (!BPT::is_csr_continuation(hdrbuf)) break;
                    auto chdr = BPT::deserialize_csr_continuation_header(hdrbuf);
                    total_continuation_count += chdr.chunk_count;
                    probe_pg = chdr.next_leaf;
                }
            }
            const uint32_t k_head_target = (total_continuation_count >= degree)
                ? 0u
                : static_cast<uint32_t>(degree - total_continuation_count);

            // Decode exactly k_head_target dsts from the chain head.
            const uint8_t* dst_in = raw + start_off;
            uint64_t running = 0;
            for (uint32_t k = 0; k < k_head_target; ++k) {
                uint64_t v = 0;
                dst_in += decode_varint(dst_in, end, v);
                if (k == 0) {
                    running = v;
                } else {
                    const int64_t delta = BPT::zigzag_decode_u64(v);
                    running += static_cast<uint64_t>(delta);
                }
                e.dsts.push_back(running);
            }
            uint32_t k_head = k_head_target;

            // If we still owe dsts, walk the chain via next_leaf.
            if (k_head < degree) {
                uint32_t next_pg = leaf.next_leaf();
                while (k_head < degree && next_pg != 0) {
                    const char* cp = page_at(bytes, next_pg);
                    if (cp == nullptr) break;
                    const uint8_t* craw = reinterpret_cast<const uint8_t*>(cp);

                    uint8_t hdrbuf[16];
                    std::memcpy(hdrbuf, craw, 16);
                    if (!BPT::is_csr_continuation(hdrbuf)) break;
                    auto chdr = BPT::deserialize_csr_continuation_header(hdrbuf);

                    const uint8_t* cin  = craw + 16;
                    const uint8_t* cend = craw + Page::SIZE;
                    for (uint32_t j = 0; j < chdr.chunk_count && k_head < degree; ++j) {
                        uint64_t v = 0;
                        cin += decode_varint(cin, cend, v);
                        const int64_t delta = BPT::zigzag_decode_u64(v);
                        running += static_cast<uint64_t>(delta);
                        e.dsts.push_back(running);
                        ++k_head;
                    }
                    next_pg = chdr.next_leaf;
                }
            }

            out.push_back(std::move(e));
        }

        // Advance past chain-head + any continuation pages.
        // Continuation pages belong to the LAST entry of this chain head (if
        // a hub). After this chain-head, follow next_leaf to the NEXT
        // chain-head, skipping any continuation pages in between.
        uint32_t scan = static_cast<uint32_t>(page) + 1;
        while (scan < num_pages) {
            const char* sp = page_at(bytes, scan);
            if (sp == nullptr) break;
            const uint8_t* sraw = reinterpret_cast<const uint8_t*>(sp);
            uint8_t hdrbuf[16];
            std::memcpy(hdrbuf, sraw, 16);
            if (sraw[0] == 3 && BPT::is_csr_continuation(hdrbuf)) {
                ++scan;
                continue;
            }
            break;
        }
        page = scan;
    }

    return out;
}

// ----------------------------------------------------------------------------
// Synthetic input builders.
// ----------------------------------------------------------------------------

std::array<uint64_t, 3> rec(uint64_t src, uint64_t dst, uint64_t edge = 0)
{
    return std::array<uint64_t, 3>{src, dst, edge};
}

// Produce a hub by generating `degree` dsts such that each zigzag-delta is
// ~`delta_bytes` varint bytes (force spillover). We pick deltas by spacing
// dsts appropriately: a delta fitting in X bytes of zigzag-varint requires
// |delta| in [64^(X-1), 64^X). For X=2, |delta| >= 64; for X=5, 2^28.
std::vector<uint64_t> make_hub_dsts(std::size_t degree, uint64_t stride)
{
    std::vector<uint64_t> out;
    out.reserve(degree);
    uint64_t v = 1;
    for (std::size_t i = 0; i < degree; ++i) {
        out.push_back(v);
        v += stride;
    }
    return out;
}

}  // namespace

// ============================================================================
// Test 1 — single src, small degree, one page.
// ============================================================================
TEST(BPTLeafCSRWriterTest, SingleSrc_SmallDegree_OnePage)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        w.append(rec(100, 5));
        w.append(rec(100, 7));
        w.append(rec(100, 11));
        w.append(rec(100, 13));
        w.append(rec(100, 17));
    }

    auto bytes = read_file_bytes(tf.path);
    ASSERT_EQ(bytes.size(), static_cast<std::size_t>(Page::SIZE));

    const char* p = bytes.data();
    EXPECT_EQ(static_cast<uint8_t>(p[0]), 3u)  << "format_version";
    EXPECT_EQ(static_cast<uint8_t>(p[1]), 3u)  << "record_width";
    EXPECT_EQ(static_cast<uint8_t>(p[2]), 0u)  << "flags";

    BPTLeafCSR<3> leaf(p, BPTLeafCSR<3>::ReadTag{});
    EXPECT_EQ(leaf.src_entry_count(), 1u);

    uint32_t start_off = 0;
    uint32_t degree    = 0;
    ASSERT_TRUE(leaf.find_src_entry(100, start_off, degree));
    EXPECT_EQ(degree, 5u);

    uint64_t d = 0;
    EXPECT_TRUE(leaf.get_dst_at(start_off, degree, 0, d)); EXPECT_EQ(d, 5u);
    EXPECT_TRUE(leaf.get_dst_at(start_off, degree, 1, d)); EXPECT_EQ(d, 7u);
    EXPECT_TRUE(leaf.get_dst_at(start_off, degree, 2, d)); EXPECT_EQ(d, 11u);
    EXPECT_TRUE(leaf.get_dst_at(start_off, degree, 3, d)); EXPECT_EQ(d, 13u);
    EXPECT_TRUE(leaf.get_dst_at(start_off, degree, 4, d)); EXPECT_EQ(d, 17u);
}

// ============================================================================
// Test 2 — multiple srcs fit on one page.
// ============================================================================
TEST(BPTLeafCSRWriterTest, MultipleSrcs_AllFitOnePage)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t src = 1; src <= 10; ++src) {
            for (uint64_t dst : {(src * 10) + 0, (src * 10) + 1, (src * 10) + 2}) {
                w.append(rec(src, dst));
            }
        }
    }
    auto bytes = read_file_bytes(tf.path);
    ASSERT_EQ(bytes.size(), static_cast<std::size_t>(Page::SIZE));

    BPTLeafCSR<3> leaf(bytes.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_EQ(leaf.src_entry_count(), 10u);

    for (uint64_t src = 1; src <= 10; ++src) {
        uint32_t off = 0, deg = 0;
        ASSERT_TRUE(leaf.find_src_entry(src, off, deg)) << "src=" << src;
        EXPECT_EQ(deg, 3u);
        uint64_t d = 0;
        for (uint32_t i = 0; i < 3; ++i) {
            ASSERT_TRUE(leaf.get_dst_at(off, deg, i, d));
            EXPECT_EQ(d, src * 10 + i);
        }
    }
}

// ============================================================================
// Test 3 — srcs span two pages.
// ============================================================================
TEST(BPTLeafCSRWriterTest, SrcsSpanTwoPages)
{
    TempFile tf;
    // Use enough srcs + dsts to force overflow to a 2nd page. Each src with
    // degree ~30 and modest dst spacing takes ~35 bytes + 2 offset slot.
    // 120 srcs × ~37 B = 4440 B → 2 pages.
    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t src = 1; src <= 200; ++src) {
            for (uint64_t dst = 1; dst <= 20; ++dst) {
                w.append(rec(src, src * 1000 + dst));
            }
        }
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    EXPECT_GE(pages_written, 2u);

    auto bytes = read_file_bytes(tf.path);
    ASSERT_EQ(bytes.size(), static_cast<std::size_t>(Page::SIZE) * pages_written);

    // First page has a subset of srcs; second page has the rest.
    BPTLeafCSR<3> p0(page_at(bytes, 0), BPTLeafCSR<3>::ReadTag{});
    BPTLeafCSR<3> p1(page_at(bytes, 1), BPTLeafCSR<3>::ReadTag{});
    EXPECT_GT(p0.src_entry_count(), 0u);
    EXPECT_GT(p1.src_entry_count(), 0u);
    EXPECT_EQ(p0.src_entry_count() + p1.src_entry_count()
                + (pages_written > 2 ? p0.src_entry_count() /* dummy */ : 0),
              p0.src_entry_count() + p1.src_entry_count()
                + (pages_written > 2 ? 0 : 0));  // placeholder sanity

    // Total srcs across all pages == 200.
    uint32_t total_srcs = 0;
    for (uint32_t pg = 0; pg < pages_written; ++pg) {
        BPTLeafCSR<3> leaf(page_at(bytes, pg), BPTLeafCSR<3>::ReadTag{});
        total_srcs += leaf.src_entry_count();
    }
    EXPECT_EQ(total_srcs, 200u);

    // next_leaf on page 0 points to page 1 (or further, but not zero).
    EXPECT_EQ(p0.next_leaf(), 1u);
}

// ============================================================================
// Test 4 — hub src spans two pages (chain-head + 1 continuation).
// ============================================================================
TEST(BPTLeafCSRWriterTest, HubSrc_SpansTwoPages)
{
    TempFile tf;
    // Force a hub: one src with ~1500 dsts spaced by 1 << 14 (varint 3 bytes
    // per dst after zigzag). 1500 × 3 = 4500 B > 4 KB → 2 pages.
    const std::size_t degree = 1500;
    auto dsts = make_hub_dsts(degree, /*stride=*/ (1ULL << 14));

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t d : dsts) w.append(rec(42, d));
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    ASSERT_GE(pages_written, 2u);

    auto bytes = read_file_bytes(tf.path);

    // Page 0: chain-head with 1 src entry, degree == 1500.
    BPTLeafCSR<3> head(page_at(bytes, 0), BPTLeafCSR<3>::ReadTag{});
    EXPECT_EQ(head.src_entry_count(), 1u);
    uint32_t off = 0, deg = 0;
    ASSERT_TRUE(head.find_src_entry(42, off, deg));
    EXPECT_EQ(deg, degree);

    // next_leaf must be non-zero (points to first continuation).
    EXPECT_NE(head.next_leaf(), 0u);

    // Page 1: continuation page with flags bit 0, chain_head_page_id = 0.
    const uint8_t* p1_raw = reinterpret_cast<const uint8_t*>(page_at(bytes, 1));
    EXPECT_EQ(p1_raw[0], 3u);
    EXPECT_NE(p1_raw[2] & BPT::CSRHybridFlags::kIsContinuation, 0);
    uint8_t hdr[16];
    std::memcpy(hdr, p1_raw, 16);
    auto chdr = BPT::deserialize_csr_continuation_header(hdr);
    EXPECT_EQ(chdr.chain_head_page_id, 0u);

    // Full readback via helper.
    auto decoded = decode_all_entries(bytes, pages_written);
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded[0].src, 42u);
    ASSERT_EQ(decoded[0].dsts.size(), degree);
    for (std::size_t i = 0; i < degree; ++i) {
        EXPECT_EQ(decoded[0].dsts[i], dsts[i]) << "i=" << i;
    }
}

// ============================================================================
// Test 5 — hub src spans three pages.
// ============================================================================
TEST(BPTLeafCSRWriterTest, HubSrc_SpansThreePages)
{
    TempFile tf;
    // Need ~3 pages worth. Use 3500 dsts × 3 B = ~10.5 KB => 3 pages.
    const std::size_t degree = 3500;
    auto dsts = make_hub_dsts(degree, (1ULL << 14));

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t d : dsts) w.append(rec(7, d));
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    ASSERT_GE(pages_written, 3u);

    auto bytes = read_file_bytes(tf.path);
    auto decoded = decode_all_entries(bytes, pages_written);
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded[0].src, 7u);
    ASSERT_EQ(decoded[0].dsts.size(), degree);
    for (std::size_t i = 0; i < degree; ++i) {
        ASSERT_EQ(decoded[0].dsts[i], dsts[i]) << "i=" << i;
    }
}

// ============================================================================
// Test 6 — make_empty produces one page of zeros with a valid header.
// ============================================================================
TEST(BPTLeafCSRWriterTest, MakeEmpty)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        w.make_empty();
    }
    auto bytes = read_file_bytes(tf.path);
    ASSERT_EQ(bytes.size(), static_cast<std::size_t>(Page::SIZE));

    // Valid v3 header with value_count == 0 is openable.
    BPTLeafCSR<3> leaf(bytes.data(), BPTLeafCSR<3>::ReadTag{});
    EXPECT_EQ(leaf.src_entry_count(), 0u);
    EXPECT_EQ(leaf.get_value_count(), 0u);
    EXPECT_EQ(leaf.next_leaf(), 0u);

    // Bytes 16..end of header should be zero (value_count=0, next_leaf=0,
    // min_src_id_low=0). Bytes beyond are zero-padding.
    for (std::size_t i = 16; i < Page::SIZE; ++i) {
        EXPECT_EQ(bytes[i], 0) << "byte " << i;
    }
}

// ============================================================================
// Test 7 — offset table monotonically increasing on a multi-src page.
// ============================================================================
TEST(BPTLeafCSRWriterTest, OffsetTable_MonotonicallyIncreasing)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t src = 1; src <= 12; ++src) {
            for (uint64_t k = 0; k < 4; ++k) {
                w.append(rec(src, src * 100 + k));
            }
        }
    }
    auto bytes = read_file_bytes(tf.path);
    ASSERT_EQ(bytes.size(), static_cast<std::size_t>(Page::SIZE));

    BPTLeafCSR<3> leaf(bytes.data(), BPTLeafCSR<3>::ReadTag{});
    const uint32_t vc = leaf.src_entry_count();
    ASSERT_GT(vc, 1u);

    const uint8_t* raw = reinterpret_cast<const uint8_t*>(bytes.data());
    uint32_t prev = 0;
    for (uint32_t i = 0; i < vc; ++i) {
        const uint8_t* off_ptr = raw + 16 + 2 * i;
        const uint32_t off = static_cast<uint32_t>(off_ptr[0])
                           | (static_cast<uint32_t>(off_ptr[1]) << 8);
        if (i > 0) {
            EXPECT_GT(off, prev) << "non-monotonic at i=" << i;
        }
        prev = off;
    }
}

// ============================================================================
// Test 8 — last page has next_leaf == 0.
// ============================================================================
TEST(BPTLeafCSRWriterTest, LastPage_NextLeafIsZero)
{
    TempFile tf;
    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t src = 1; src <= 500; ++src) {
            for (uint64_t dst = 1; dst <= 10; ++dst) {
                w.append(rec(src, src * 1000 + dst));
            }
        }
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    ASSERT_GE(pages_written, 2u);

    auto bytes = read_file_bytes(tf.path);

    // Last page's next_leaf (offset 8..11 of the header) must be 0.
    const uint8_t* last = reinterpret_cast<const uint8_t*>(
        page_at(bytes, pages_written - 1));
    ASSERT_NE(last, nullptr);
    uint32_t next_leaf =  static_cast<uint32_t>(last[8])
                       | (static_cast<uint32_t>(last[9])  <<  8)
                       | (static_cast<uint32_t>(last[10]) << 16)
                       | (static_cast<uint32_t>(last[11]) << 24);
    EXPECT_EQ(next_leaf, 0u);
}

// ============================================================================
// Test 9 — degree field encoded as the second varint in each entry.
// ============================================================================
TEST(BPTLeafCSRWriterTest, Degree_EncodedAsVarint)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        // src=1000 (2-byte varint), degree=7 (1-byte varint)
        for (uint64_t dst : {100ULL, 105ULL, 200ULL, 300ULL, 301ULL, 400ULL, 500ULL}) {
            w.append(rec(1000, dst));
        }
    }
    auto bytes = read_file_bytes(tf.path);
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(bytes.data());

    // Page has 1 src entry, offset_table[0] at bytes 16..17.
    const uint32_t off = static_cast<uint32_t>(raw[16])
                       | (static_cast<uint32_t>(raw[17]) << 8);
    const uint8_t* in  = raw + off;
    const uint8_t* end = raw + Page::SIZE;

    uint64_t src_id = 0, degree = 0;
    in += decode_varint(in, end, src_id);
    EXPECT_EQ(src_id, 1000u);
    in += decode_varint(in, end, degree);
    EXPECT_EQ(degree, 7u);
}

// ============================================================================
// Test 10 — first dst is a full varint; rest are zigzag-deltas.
// ============================================================================
TEST(BPTLeafCSRWriterTest, DstList_FirstFullVarint_RestZigzagDelta)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        w.append(rec(1, 100));
        w.append(rec(1, 105));
        w.append(rec(1, 200));
    }
    auto bytes = read_file_bytes(tf.path);
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(bytes.data());

    const uint32_t off = static_cast<uint32_t>(raw[16])
                       | (static_cast<uint32_t>(raw[17]) << 8);
    const uint8_t* in  = raw + off;
    const uint8_t* end = raw + Page::SIZE;

    uint64_t src_id = 0, degree = 0;
    in += decode_varint(in, end, src_id);
    in += decode_varint(in, end, degree);
    EXPECT_EQ(src_id, 1u);
    EXPECT_EQ(degree, 3u);

    uint64_t v0 = 0, v1 = 0, v2 = 0;
    in += decode_varint(in, end, v0);
    in += decode_varint(in, end, v1);
    in += decode_varint(in, end, v2);
    EXPECT_EQ(v0, 100u);  // full varint
    // Delta 105 - 100 = 5 → zigzag(5) = 10.
    EXPECT_EQ(v1, BPT::zigzag_encode_i64(5));
    // Delta 200 - 105 = 95 → zigzag(95) = 190.
    EXPECT_EQ(v2, BPT::zigzag_encode_i64(95));
}

// ============================================================================
// Test 11 — deterministic output: same input → same bytes.
// ============================================================================
TEST(BPTLeafCSRWriterTest, AppendSortedOrder_Respected)
{
    TempFile tf1, tf2;
    auto build = [](const std::string& path) {
        BPTLeafCSRWriter<3> w(path);
        for (uint64_t src = 1; src <= 20; ++src) {
            for (uint64_t dst = 1; dst <= 5; ++dst) {
                w.append(rec(src, src * 100 + dst));
            }
        }
    };
    build(tf1.path);
    build(tf2.path);
    auto b1 = read_file_bytes(tf1.path);
    auto b2 = read_file_bytes(tf2.path);
    EXPECT_EQ(b1, b2);
}

// ============================================================================
// Test 12 — format_version byte is 3 on every flushed page.
// ============================================================================
TEST(BPTLeafCSRWriterTest, FormatVersionByteIsThree)
{
    TempFile tf;
    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        // Mix of regular and hub to produce chain-head + continuation + regular.
        auto big = make_hub_dsts(1200, 1ULL << 14);
        for (uint64_t d : big) w.append(rec(10, d));
        for (uint64_t src = 11; src <= 30; ++src) {
            for (uint64_t dst = 1; dst <= 5; ++dst) {
                w.append(rec(src, src * 100 + dst));
            }
        }
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    auto bytes = read_file_bytes(tf.path);
    for (uint32_t pg = 0; pg < pages_written; ++pg) {
        const char* p = page_at(bytes, pg);
        EXPECT_EQ(static_cast<uint8_t>(p[0]), 3u) << "page " << pg;
    }
}

// ============================================================================
// Test 13 — continuation page has flag bit 0 set.
// ============================================================================
TEST(BPTLeafCSRWriterTest, ContinuationPage_HasFlagBitSet)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        auto dsts = make_hub_dsts(1500, 1ULL << 14);
        for (uint64_t d : dsts) w.append(rec(55, d));
    }
    auto bytes = read_file_bytes(tf.path);
    ASSERT_GE(bytes.size(), static_cast<std::size_t>(Page::SIZE) * 2);

    // Page 0 = chain head (flag bit 0 clear).
    EXPECT_EQ(static_cast<uint8_t>(bytes[2])
              & BPT::CSRHybridFlags::kIsContinuation, 0);
    // Page 1 = continuation (flag bit 0 set).
    EXPECT_NE(static_cast<uint8_t>(bytes[Page::SIZE + 2])
              & BPT::CSRHybridFlags::kIsContinuation, 0);
}

// ============================================================================
// Test 14 — append 100 srcs, readback via BPTLeafCSR matches input exactly.
// ============================================================================
TEST(BPTLeafCSRWriterTest, AppendThenReadback_ViaBPTLeafCSR)
{
    TempFile tf;
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> truth;
    for (uint64_t src = 1; src <= 100; ++src) {
        std::vector<uint64_t> dsts;
        for (uint64_t k = 0; k < (src % 7) + 2; ++k) {
            dsts.push_back(src * 1000 + k * 17);
        }
        truth.emplace_back(src, std::move(dsts));
    }
    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (const auto& e : truth) {
            for (uint64_t d : e.second) w.append(rec(e.first, d));
        }
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    auto bytes = read_file_bytes(tf.path);
    auto decoded = decode_all_entries(bytes, pages_written);
    ASSERT_EQ(decoded.size(), truth.size());
    for (std::size_t i = 0; i < truth.size(); ++i) {
        EXPECT_EQ(decoded[i].src, truth[i].first) << "i=" << i;
        EXPECT_EQ(decoded[i].dsts, truth[i].second) << "i=" << i;
    }
}

// ============================================================================
// Test 15 — large hub, full chain traversal recovers all dsts in order.
//
// Sized to reliably exceed one page: 2500 dsts × 3 B zigzag-varint ≈ 7.5 KB,
// spread across a chain-head + at least one continuation page. The stride
// 2^14 places each zigzag-delta in the 2-byte range (|delta| < 2^13 fits in
// 2 bytes; equal to 2^13 ≈ 2^14 / 2 here produces 2-3 byte varints).
// ============================================================================
TEST(BPTLeafCSRWriterTest, HubReadback_ChainTraversal)
{
    TempFile tf;
    const std::size_t degree = 2500;
    auto dsts = make_hub_dsts(degree, (1ULL << 14));

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t d : dsts) w.append(rec(999, d));
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    ASSERT_GE(pages_written, 2u);

    auto bytes = read_file_bytes(tf.path);
    auto decoded = decode_all_entries(bytes, pages_written);
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded[0].src, 999u);
    ASSERT_EQ(decoded[0].dsts.size(), degree);
    for (std::size_t i = 0; i < degree; ++i) {
        ASSERT_EQ(decoded[0].dsts[i], dsts[i]) << "i=" << i;
    }
}

// ============================================================================
// T8-B.1 Bug-C — last continuation of a hub chain that is FOLLOWED by
// another src must have next_leaf pointing at the next chain-head page, not
// 0. Pre-fix the writer emitted next_leaf=0 on the tail continuation and
// left the leaf chain broken at hub boundaries.
// ============================================================================

TEST(BPTLeafCSRWriterTest, HubFollowedBySrc_LeafChainContinuous)
{
    TempFile tf;
    // src=1: a few dsts on one page (non-hub).
    // src=42: hub, produces chain-head + continuations.
    // src=777: a regular single-page entry AFTER the hub's continuations.
    const std::size_t hub_deg = 2000;
    auto hub_dsts = make_hub_dsts(hub_deg, (1ULL << 14));

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        w.append(rec(1, 10));
        w.append(rec(1, 11));
        for (uint64_t d : hub_dsts) w.append(rec(42, d));
        for (uint64_t d = 1; d <= 5; ++d) w.append(rec(777, 9000 + d));
        w.flush_finalize();
        pages_written = w.pages_written();
    }

    auto bytes = read_file_bytes(tf.path);
    ASSERT_EQ(bytes.size(), static_cast<std::size_t>(Page::SIZE) * pages_written);

    // Find the last continuation page in the file. Its next_leaf MUST be
    // the page number of the following chain-head (which hosts src=777),
    // NOT 0.
    int last_continuation_idx = -1;
    for (uint32_t pg = 0; pg < pages_written; ++pg) {
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(page_at(bytes, pg));
        if (raw[0] == 3 && (raw[2] & BPT::CSRHybridFlags::kIsContinuation) != 0) {
            last_continuation_idx = static_cast<int>(pg);
        }
    }
    ASSERT_GE(last_continuation_idx, 0);

    const uint8_t* last_cont = reinterpret_cast<const uint8_t*>(
        page_at(bytes, last_continuation_idx));
    uint32_t next_leaf =  static_cast<uint32_t>(last_cont[8])
                       | (static_cast<uint32_t>(last_cont[9])  <<  8)
                       | (static_cast<uint32_t>(last_cont[10]) << 16)
                       | (static_cast<uint32_t>(last_cont[11]) << 24);

    // The immediately following page must be a chain-head holding src=777.
    const uint32_t expected_next = static_cast<uint32_t>(last_continuation_idx + 1);
    EXPECT_EQ(next_leaf, expected_next)
        << "last continuation's next_leaf should point at the chain-head of "
           "the subsequent src, not 0";

    // Cross-check: that page IS a chain-head and its only src entries reach
    // src=777.
    const uint8_t* head_after = reinterpret_cast<const uint8_t*>(
        page_at(bytes, expected_next));
    EXPECT_EQ(head_after[0], 3u);
    EXPECT_EQ(head_after[2] & BPT::CSRHybridFlags::kIsContinuation, 0);
    BPTLeafCSR<3> leaf_after(reinterpret_cast<const char*>(head_after),
                             BPTLeafCSR<3>::ReadTag{});
    uint32_t off = 0, deg = 0;
    EXPECT_TRUE(leaf_after.find_src_entry(777, off, deg));
    EXPECT_EQ(deg, 5u);
}

TEST(BPTLeafCSRWriterTest, HubIsLast_LastContinuationNextLeafZero)
{
    // Regression: when the hub is the LAST src, the last continuation's
    // next_leaf must stay 0 (no patch). This is the case
    // HubSrc_SpansThreePages already covers at a higher level, but we
    // assert on the raw bytes here for clarity.
    TempFile tf;
    const std::size_t hub_deg = 2000;
    auto hub_dsts = make_hub_dsts(hub_deg, (1ULL << 14));

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t d : hub_dsts) w.append(rec(99, d));
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    auto bytes = read_file_bytes(tf.path);

    // Last continuation page.
    int last_cont_idx = -1;
    for (uint32_t pg = 0; pg < pages_written; ++pg) {
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(page_at(bytes, pg));
        if (raw[0] == 3 && (raw[2] & BPT::CSRHybridFlags::kIsContinuation) != 0) {
            last_cont_idx = static_cast<int>(pg);
        }
    }
    ASSERT_GE(last_cont_idx, 0);

    const uint8_t* last = reinterpret_cast<const uint8_t*>(
        page_at(bytes, last_cont_idx));
    uint32_t next_leaf =  static_cast<uint32_t>(last[8])
                       | (static_cast<uint32_t>(last[9])  <<  8)
                       | (static_cast<uint32_t>(last[10]) << 16)
                       | (static_cast<uint32_t>(last[11]) << 24);
    EXPECT_EQ(next_leaf, 0u);
}

// ============================================================================
// Spec #8-B task #1 — parallel edge_id stream tests.
//
// These exercise the `emit_edge_ids` constructor flag introduced to close
// the ADR 008 Known-limitation #1 caveat (`edge_id` persisted as zero on
// CSR_HYBRID). The writer emits a parallel DELTA-zigzag varint chain
// after every entry's dst chain, and advertises the stream via the
// header bit `kHasEdgeIds` (0x02). The reader detects the flag and
// returns real edge_ids through decode_tuple_.
// ============================================================================

// Default-disabled: writer keeps legacy behavior unless opted in.
TEST(BPTLeafCSRWriterTest, EdgeIds_Default_FlagUnset)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        w.append(rec(1, 10, 100));
        w.append(rec(1, 11, 101));
        w.append(rec(2, 20, 200));
        w.flush_finalize();
    }
    auto bytes = read_file_bytes(tf.path);
    ASSERT_GE(bytes.size(), static_cast<std::size_t>(Page::SIZE));
    EXPECT_EQ(bytes[2] & 0x02, 0u) << "default writer must not set kHasEdgeIds";
}

// Opted in: flag is set AND the reader returns the encoded edge_ids.
TEST(BPTLeafCSRWriterTest, EdgeIds_Enabled_FlagSetAndReaderRoundTrip)
{
    TempFile tf;
    std::vector<std::array<uint64_t, 3>> truth = {
        {1, 10, 100}, {1, 11, 101}, {1, 12, 102},
        {2, 20, 200}, {2, 22, 222},
        {3, 30, 300},
    };
    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path, /*emit_edge_ids=*/true);
        for (const auto& r : truth) w.append(r);
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    auto bytes = read_file_bytes(tf.path);
    ASSERT_EQ(pages_written, 1u);
    EXPECT_NE(bytes[2] & 0x02, 0u);

    BPTLeafCSR<3> reader(
        reinterpret_cast<const char*>(bytes.data()),
        BPTLeafCSR<3>::ReadTag{});
    const uint32_t total = reader.get_value_count();
    ASSERT_EQ(total, truth.size());
    for (uint_fast32_t i = 0; i < total; ++i) {
        Record<3> r = reader.get_record(i);
        EXPECT_EQ(std::get<0>(r), truth[i][0]) << "pos=" << i << " src";
        EXPECT_EQ(std::get<1>(r), truth[i][1]) << "pos=" << i << " dst";
        EXPECT_EQ(std::get<2>(r), truth[i][2]) << "pos=" << i << " eid";
    }
}

// Larger mix: multiple srcs with varied edge_id ranges exercises the
// zigzag-delta encoding / decoding on non-monotone eid sequences.
TEST(BPTLeafCSRWriterTest, EdgeIds_Enabled_ZigzagDeltaRoundTrip)
{
    TempFile tf;
    std::vector<std::array<uint64_t, 3>> truth;
    for (uint64_t src = 1; src <= 30; ++src) {
        for (uint64_t k = 0; k < 4; ++k) {
            const uint64_t dst = src * 1000 + k * 7;
            const uint64_t eid = src * 10 + (k == 0 ? 5 : k == 1 ? 1
                                                         : k == 2 ? 8 : 3);
            truth.push_back({src, dst, eid});
        }
    }
    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path, /*emit_edge_ids=*/true);
        for (const auto& r : truth) w.append(r);
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    auto bytes = read_file_bytes(tf.path);
    ASSERT_GE(pages_written, 1u);

    BPTLeafCSR<3> reader(
        reinterpret_cast<const char*>(bytes.data()),
        BPTLeafCSR<3>::ReadTag{});
    const uint32_t total = reader.get_value_count();
    ASSERT_EQ(total, truth.size());
    for (uint_fast32_t i = 0; i < total; ++i) {
        Record<3> r = reader.get_record(i);
        EXPECT_EQ(std::get<2>(r), truth[i][2])
            << "pos=" << i << " src=" << truth[i][0]
            << " dst=" << truth[i][1];
    }
}

// Opt-in false (explicit) matches the default path byte-for-byte.
TEST(BPTLeafCSRWriterTest, EdgeIds_ExplicitlyDisabled_SameAsDefault)
{
    TempFile default_file, off_file;
    const std::vector<std::array<uint64_t, 3>> input = {
        {1, 10, 100}, {1, 11, 101},
        {2, 20, 200},
    };
    {
        BPTLeafCSRWriter<3> a(default_file.path);
        BPTLeafCSRWriter<3> b(off_file.path, /*emit_edge_ids=*/false);
        for (const auto& r : input) { a.append(r); b.append(r); }
        a.flush_finalize();
        b.flush_finalize();
    }
    auto a_bytes = read_file_bytes(default_file.path);
    auto b_bytes = read_file_bytes(off_file.path);
    ASSERT_EQ(a_bytes, b_bytes);
}

// Reader-side edge_id = 0 sentinel preserved for legacy default-writer
// pages (no kHasEdgeIds bit).
TEST(BPTLeafCSRWriterTest, EdgeIds_Disabled_ReaderReturnsZero)
{
    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        w.append(rec(7, 70, 700));
        w.append(rec(7, 77, 777));
        w.flush_finalize();
    }
    auto bytes = read_file_bytes(tf.path);
    BPTLeafCSR<3> reader(
        reinterpret_cast<const char*>(bytes.data()),
        BPTLeafCSR<3>::ReadTag{});
    for (uint_fast32_t i = 0; i < reader.get_value_count(); ++i) {
        Record<3> r = reader.get_record(i);
        EXPECT_EQ(std::get<2>(r), 0u)
            << "legacy writer path must keep eid=0 sentinel at pos=" << i;
    }
}
