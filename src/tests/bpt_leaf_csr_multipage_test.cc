// End-to-end multi-page test for the CSR-hybrid graph storage bug fixes that
// resolved three composed defects: the reader was inflating total_tuples_ for
// hub nodes, the ContinuationTag constructor was missing, and the writer was
// breaking the leaf chain for hubs that span multiple pages.
//
// This test exercises the integration between BPTLeafCSRWriter<N> (which
// emits the hub chain-head + continuation pages), BPTLeafCSR<N>::ReadTag
// (which opens chain-head pages via the original path), and
// BPTLeafCSR<N>::ContinuationTag (the constructor that opens continuation
// pages as a BPTLeafBase<N> view, carrying over the src_id and prev_dst
// from the preceding chain-head or continuation page).
//
// Scenarios:
//   1. Three srcs in order: non-hub, hub, non-hub. Full-file scan returns
//      all 4020 tuples in order (equivalent to what BptIter would do).
//   2. Hub-only point lookup: mimic a get_range(src=hub) scan over just
//      the hub's pages (chain-head + all continuations), yields exactly
//      the hub's degree tuples.
//   3. Leaf-chain continuity: walking next_leaf from page 0 traverses
//      every page including the continuations, ending at a page with
//      next_leaf=0.
//
// This test runs entirely on file-backed synthetic pages without the
// BufferManager — the read path does not need it since BPTLeafCSR holds
// a raw `const char*` into the page bytes. Each page is memory-mapped
// via read_file_bytes() + pointer arithmetic, matching the
// bpt_leaf_csr_writer_test.cc pattern.

#include "storage/index/bplus_tree/bplus_tree_leaf_csr.h"
#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
#include "storage/index/bplus_tree/bpt_mem_import.h"
#include "storage/index/bplus_tree/varint.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "storage/index/record.h"
#include "storage/page/page.h"

namespace {

struct TempFile {
    std::string path;
    explicit TempFile(const std::string& suffix = ".leaf")
    {
        const char* base = std::getenv("TMPDIR");
        if (base == nullptr) base = "/tmp";
        static int counter = 0;
        path = std::string(base) + "/bpt_csr_multipage_"
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

const char* page_at(const std::vector<char>& bytes, std::size_t page_num)
{
    const std::size_t off = page_num * Page::SIZE;
    if (off + Page::SIZE > bytes.size()) return nullptr;
    return bytes.data() + off;
}

// Simulate what BptIter<3> does over a CSR_HYBRID B+Tree: walk the leaf
// chain starting at page 0, opening each page as either a chain-head
// (ReadTag) or a continuation (ContinuationTag, with carry-over src_id
// and prev_dst), emit all tuples in order.
std::vector<Record<3>>
simulate_full_scan(const std::vector<char>& bytes, uint32_t num_pages)
{
    std::vector<Record<3>> out;
    if (num_pages == 0) return out;

    uint32_t page = 0;
    uint64_t carry_src_id   = 0;
    uint64_t carry_prev_dst = 0;
    uint64_t carry_prev_eid = 0;  // carry-over edge_id for hub continuation pages (CSR-hybrid multi-page hub support)
    bool have_carry         = false;

    while (page < num_pages) {
        const char* p = page_at(bytes, page);
        if (p == nullptr) break;
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(p);

        const bool is_continuation =
            (raw[2] & BPT::CSRHybridFlags::kIsContinuation) != 0;

        // Figure out next-leaf first (header byte offset 8..11 — same in both
        // variants).
        uint32_t next_leaf =  static_cast<uint32_t>(raw[8])
                           | (static_cast<uint32_t>(raw[9])  <<  8)
                           | (static_cast<uint32_t>(raw[10]) << 16)
                           | (static_cast<uint32_t>(raw[11]) << 24);

        if (!is_continuation) {
            BPTLeafCSR<3> leaf(p, BPTLeafCSR<3>::ReadTag{});
            const uint32_t cnt = leaf.get_value_count();
            for (uint32_t i = 0; i < cnt; ++i) {
                Record<3> r = leaf.get_record(i);
                out.push_back(r);
                carry_src_id   = r[0];
                carry_prev_dst = r[1];
                carry_prev_eid = r[2];
                have_carry     = true;
            }
        } else {
            if (!have_carry) {
                // No valid carry means we walked into a continuation without
                // a preceding chain-head emit — impossible under a well-
                // formed chain.
                ADD_FAILURE() << "continuation reached without carry at page "
                              << page;
                break;
            }
            BPTLeafCSR<3> leaf(p, BPTLeafCSR<3>::ContinuationTag{
                carry_src_id, carry_prev_dst, carry_prev_eid});
            const uint32_t cnt = leaf.get_value_count();
            for (uint32_t i = 0; i < cnt; ++i) {
                Record<3> r = leaf.get_record(i);
                out.push_back(r);
                carry_prev_dst = r[1];
                carry_prev_eid = r[2];
            }
        }

        if (next_leaf == 0) break;
        // Defence: the chain MUST progress monotonically.
        if (next_leaf <= page) {
            ADD_FAILURE() << "non-monotonic next_leaf at page " << page
                          << " -> " << next_leaf;
            break;
        }
        page = next_leaf;
    }
    return out;
}

}  // anonymous namespace

// ============================================================================
// Scenario 1 — non-hub, hub, non-hub. Full scan returns every tuple.
// ============================================================================

TEST(BPTLeafCSRMultipage, NonHub_Hub_NonHub_FullScan)
{
    TempFile tf;
    std::vector<Record<3>> truth;

    // Small stride keeps zigzag-delta varints short (1-2 bytes). 4000 dsts
    // × ~2B > 4080 → at least 2 pages required for the hub.
    const uint64_t hub_src = 1;
    const std::size_t hub_deg = 4000;
    const uint64_t stride = (1ULL << 14);

    auto add = [&truth](uint64_t s, uint64_t d) {
        Record<3> r;
        r[0] = s; r[1] = d; r[2] = 0;
        truth.push_back(r);
    };

    // Note: caller contract for BPTLeafCSRWriter::append requires strictly
    // non-decreasing src on successive appends. We also choose src values
    // such that src=0 comes first, then the hub (src=1), then src=2.
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> inputs;

    // src=0 with 10 dsts.
    {
        std::vector<uint64_t> dsts;
        for (uint64_t i = 0; i < 10; ++i) dsts.push_back(100 + i * 3);
        inputs.emplace_back(0, dsts);
    }
    // src=1 is the hub — 4000 dsts.
    {
        std::vector<uint64_t> dsts;
        uint64_t v = 1;
        for (std::size_t i = 0; i < hub_deg; ++i) { dsts.push_back(v); v += stride; }
        inputs.emplace_back(hub_src, dsts);
    }
    // src=2 with 10 dsts.
    {
        std::vector<uint64_t> dsts;
        for (uint64_t i = 0; i < 10; ++i) dsts.push_back(20000 + i * 5);
        inputs.emplace_back(2, dsts);
    }

    for (const auto& kv : inputs) {
        for (uint64_t d : kv.second) add(kv.first, d);
    }

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (const auto& kv : inputs) {
            for (uint64_t d : kv.second) {
                std::array<uint64_t, 3> rec = {kv.first, d, 0};
                w.append(rec);
            }
        }
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    ASSERT_GE(pages_written, 3u);

    auto bytes = read_file_bytes(tf.path);
    auto scanned = simulate_full_scan(bytes, pages_written);

    ASSERT_EQ(scanned.size(), truth.size())
        << "full scan size mismatch; pages_written=" << pages_written;

    for (std::size_t i = 0; i < truth.size(); ++i) {
        EXPECT_EQ(scanned[i][0], truth[i][0]) << "i=" << i;
        EXPECT_EQ(scanned[i][1], truth[i][1]) << "i=" << i;
    }
}

// ============================================================================
// Scenario 2 — leaf chain continuity: next_leaf walk covers every page and
// terminates at 0.
// ============================================================================

TEST(BPTLeafCSRMultipage, LeafChainContinuous_PostHubPatch)
{
    TempFile tf;
    const std::size_t hub_deg = 3000;
    const uint64_t stride = (1ULL << 14);
    std::vector<uint64_t> hub_dsts;
    uint64_t v = 1;
    for (std::size_t i = 0; i < hub_deg; ++i) { hub_dsts.push_back(v); v += stride; }

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        // Non-hub before.
        for (uint64_t d = 1; d <= 5; ++d)  w.append({10ULL, d * 100ULL, 0ULL});
        // Hub.
        for (uint64_t d : hub_dsts)         w.append({42ULL, d, 0ULL});
        // Non-hub after — must be reachable via next_leaf walk.
        for (uint64_t d = 1; d <= 5; ++d)  w.append({100ULL, 9000ULL + d, 0ULL});
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    ASSERT_GE(pages_written, 3u);

    auto bytes = read_file_bytes(tf.path);

    // Walk the leaf chain and count distinct pages visited.
    uint32_t visited = 0;
    uint32_t page = 0;
    bool saw_post_hub = false;
    while (true) {
        ++visited;
        if (visited > pages_written) {
            ADD_FAILURE() << "leaf-chain walk exceeded page count "
                          << pages_written;
            break;
        }
        const char* p = page_at(bytes, page);
        ASSERT_NE(p, nullptr);
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(p);

        // Detect the post-hub chain-head (non-continuation, holds src=100).
        const bool is_continuation =
            (raw[2] & BPT::CSRHybridFlags::kIsContinuation) != 0;
        if (!is_continuation) {
            BPTLeafCSR<3> leaf(p, BPTLeafCSR<3>::ReadTag{});
            uint32_t off = 0, deg = 0;
            if (leaf.find_src_entry(100, off, deg)) {
                saw_post_hub = true;
            }
        }

        uint32_t next =  static_cast<uint32_t>(raw[8])
                      | (static_cast<uint32_t>(raw[9])  <<  8)
                      | (static_cast<uint32_t>(raw[10]) << 16)
                      | (static_cast<uint32_t>(raw[11]) << 24);
        if (next == 0) break;
        ASSERT_GT(next, page) << "non-monotonic next_leaf at page " << page;
        page = next;
    }
    EXPECT_EQ(visited, pages_written);
    EXPECT_TRUE(saw_post_hub)
        << "post-hub chain-head unreachable via leaf-chain walk (Bug-C)";
}

// ============================================================================
// Scenario 3 — hub adjacency reconstruction via continuation ctor.
// ============================================================================

TEST(BPTLeafCSRMultipage, HubAdjacency_ContinuationReconstruction)
{
    TempFile tf;
    const std::size_t hub_deg = 4000;
    const uint64_t stride = (1ULL << 14);
    std::vector<uint64_t> hub_dsts;
    uint64_t v = 1;
    for (std::size_t i = 0; i < hub_deg; ++i) { hub_dsts.push_back(v); v += stride; }

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (uint64_t d : hub_dsts) w.append({42ULL, d, 0ULL});
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    ASSERT_GE(pages_written, 2u);

    auto bytes = read_file_bytes(tf.path);

    // Reconstruct hub adjacency by: read chain-head physical dsts (via
    // ReadTag reader), then follow next_leaf opening each subsequent page
    // as a ContinuationTag reader.
    std::vector<uint64_t> reconstructed;

    BPTLeafCSR<3> head(page_at(bytes, 0), BPTLeafCSR<3>::ReadTag{});
    const uint32_t head_count = head.get_value_count();
    uint64_t last_dst = 0;
    for (uint32_t i = 0; i < head_count; ++i) {
        Record<3> r = head.get_record(i);
        reconstructed.push_back(r[1]);
        last_dst = r[1];
    }

    uint32_t page = head.next_leaf();
    while (page != 0) {
        const char* p = page_at(bytes, page);
        ASSERT_NE(p, nullptr);
        BPTLeafCSR<3> cont(p, BPTLeafCSR<3>::ContinuationTag{42ULL, last_dst});
        const uint32_t n = cont.get_value_count();
        for (uint32_t i = 0; i < n; ++i) {
            Record<3> r = cont.get_record(i);
            reconstructed.push_back(r[1]);
            last_dst = r[1];
        }
        page = cont.next_leaf();
        // Stop when we reach a non-continuation page (would be the next src
        // for multi-src scenarios; single-src here so loop should end at 0).
        if (page != 0) {
            const uint8_t* raw = reinterpret_cast<const uint8_t*>(page_at(bytes, page));
            if ((raw[2] & BPT::CSRHybridFlags::kIsContinuation) == 0) break;
        }
    }

    ASSERT_EQ(reconstructed.size(), hub_deg);
    for (std::size_t i = 0; i < hub_deg; ++i) {
        ASSERT_EQ(reconstructed[i], hub_dsts[i]) << "i=" << i;
    }
}

// ============================================================================
// CSR-hybrid hub completion: full-scan with edge_ids on a hub that spans
// multiple continuation pages reproduces every (src, dst, eid) triple in
// order. This validates that edge_ids are correctly persisted alongside dst
// values in the chain-head and all continuation pages.
// ============================================================================

TEST(BPTLeafCSRMultipage, HubAdjacency_WithEdgeIds_FullScanRoundtrip)
{
    TempFile tf;
    const std::size_t hub_deg = 4000;
    const uint64_t stride = (1ULL << 14);
    std::vector<uint64_t> hub_dsts;
    std::vector<uint64_t> hub_eids;
    {
        uint64_t v = 1;
        for (std::size_t i = 0; i < hub_deg; ++i) {
            hub_dsts.push_back(v); v += stride;
            hub_eids.push_back(50000ull + i * 17);
        }
    }

    uint32_t pages_written = 0;
    {
        BPTLeafCSRWriter<3> w(tf.path, /*emit_edge_ids=*/true);
        for (std::size_t i = 0; i < hub_deg; ++i) {
            w.append({42ULL, hub_dsts[i], hub_eids[i]});
        }
        w.flush_finalize();
        pages_written = w.pages_written();
    }
    ASSERT_GE(pages_written, 2u);

    auto bytes = read_file_bytes(tf.path);
    auto scanned = simulate_full_scan(bytes, pages_written);

    ASSERT_EQ(scanned.size(), hub_deg);

    // Build truth records (src, dst, eid).
    for (std::size_t i = 0; i < hub_deg; ++i) {
        EXPECT_EQ(scanned[i][0], 42u);
        EXPECT_EQ(scanned[i][1], hub_dsts[i]) << "i=" << i;
        EXPECT_EQ(scanned[i][2], hub_eids[i])
            << "eid mismatch at full-scan pos " << i;
    }
}
