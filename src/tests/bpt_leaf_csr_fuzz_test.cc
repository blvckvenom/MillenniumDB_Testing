// Randomized encode->decode fuzz harness for the Spec #8 CSR-hybrid leaf
// format (design §3.9, §5.1, §5.2). Mirrors the T5.14 v2 fuzz harness
// structurally: runs hundreds of thousands of iterations under a
// deterministic mt19937_64 seed, cycling through N in {2, 3} and four
// distribution families (dense_monotonic, sparse_random, clustered,
// hub_heavy). Every iteration encodes a sorted sequence of
// (src, dst, edge_id) records through BPTLeafCSRWriter<N> into a temp
// file and asserts that the reader-side chain walk returns the same
// (src, degree, dsts[]) tuples — edge_id is excluded from the invariant
// because T8.5's writer emits flags=0 (no edge-id stream; T8.6+ scope).
//
// The iteration budget is 500k in Release, 50k under ASan (same basis
// as T5.14 — ASan slows tight encode/decode loops by ~18x on benito_pc).
// A developer can override via -D MAIN_ITERATIONS_OVERRIDE=... for a
// pre-merge full-strength check.
//
// Dedicated subtests:
//   - BoundaryCases: ~10k explicit adversarial inputs hitting
//     single-src/single-dst, 1000-dst hubs crossing continuation pages,
//     dense offset tables, all-equal-degree pages, and max-range dsts.
//   - TamperInjection_AllDetected: flips every bit in the payload region
//     of a valid 50-record chain-head page. Every flip MUST be caught —
//     either by a thrown BPTLeafCSRDecodeException at reader construction,
//     or by a materially different (src, degree, dsts[]) readout. Silent
//     corruption is a FAIL.
//
// Design reference:
//   docs/superpowers/specs/2026-04-25-csr-hybrid-design.md §3.9 §5.1 §5.2
//
// Seed: MAIN_SEED is distinct from T5.14's (0xDE1A4A4F12345678) to
// diversify corpora. See comment on MAIN_SEED below.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

#include "storage/index/bplus_tree/bplus_tree_leaf_csr.h"
#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
#include "storage/index/bplus_tree/bpt_mem_import.h"
#include "storage/index/bplus_tree/varint.h"
#include "storage/page/page.h"

namespace {

// Hardcoded seed — reproducibility contract for the full-strength fuzz
// run. Distinct from T5.14's 0xDE1A4A4F12345678 so the two harnesses
// cover independent corpora.
constexpr uint64_t MAIN_SEED  = 0xCAFEBABEDE1A4A4FULL;
constexpr uint64_t SMOKE_SEED = MAIN_SEED + 1ULL;

// Detect sanitizer-instrumented Debug builds so we can scale iterations
// down — ASan+UBSan together slow the tight file-roundtrip loop by ~15x
// on benito_pc; a 500k-iteration main run under Debug would blow past
// the 5 min ASan budget.
#if defined(__SANITIZE_ADDRESS__)
    #define MDB_FUZZ_UNDER_ASAN 1
#elif defined(__has_feature)
    #if __has_feature(address_sanitizer)
        #define MDB_FUZZ_UNDER_ASAN 1
    #else
        #define MDB_FUZZ_UNDER_ASAN 0
    #endif
#else
    #define MDB_FUZZ_UNDER_ASAN 0
#endif

#ifdef MAIN_ITERATIONS_OVERRIDE
constexpr size_t MAIN_ITERATIONS  = MAIN_ITERATIONS_OVERRIDE;
#elif MDB_FUZZ_UNDER_ASAN
constexpr size_t MAIN_ITERATIONS  =  50'000;
#else
constexpr size_t MAIN_ITERATIONS  = 500'000;
#endif
constexpr size_t SMOKE_ITERATIONS =   1'000;
constexpr size_t BOUNDARY_ITERATIONS = 10'000;

enum class Distribution {
    DenseMonotonic,
    SparseRandom,
    Clustered,
    HubHeavy,
};

// ----------------------------------------------------------------------------
// Temp file helper (same idiom as bpt_leaf_csr_writer_test.cc).
// ----------------------------------------------------------------------------
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& suffix = ".leaf")
    {
        const char* base = std::getenv("TMPDIR");
        if (base == nullptr) base = "/tmp";
        static int counter = 0;
        path = std::string(base) + "/bpt_csr_fuzz_"
             + std::to_string(::getpid()) + "_"
             + std::to_string(++counter) + suffix;
    }
    ~TempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempFile(const TempFile&)            = delete;
    TempFile& operator=(const TempFile&) = delete;
};

// ----------------------------------------------------------------------------
// Input generator. Produces a sorted sequence of (src, dst[, edge_id])
// records conforming to the BPTLeafCSRWriter contract: records are
// non-decreasing on record[0] (src).
// ----------------------------------------------------------------------------

struct SrcGroup {
    uint64_t              src;
    std::vector<uint64_t> dsts;
};

// Pick an entry-distribution-appropriate k (number of src groups) with a
// heavy skew toward small pages so the 500k iteration budget stays under
// 60 s. 85% small (1..30 srcs, 1..5 dsts each), 13% medium (up to ~200),
// 2% hub-heavy (exactly one hub with 200-2000 dsts forcing continuation
// pages). Large hubs are expensive (≥3 pages of writer output + full
// decode); running 500k of them would take ~15 min.
struct Plan {
    size_t num_srcs;       // number of src groups
    size_t dsts_per_src;   // nominal dsts per src (used by dense/clustered)
    bool   has_hub;        // if true, append exactly one hub
    size_t hub_degree;     // degree of the hub (only when has_hub)
};

Plan pick_plan(std::mt19937_64& rng, Distribution dist)
{
    Plan p{};
    const uint32_t bucket = static_cast<uint32_t>(rng() % 100u);
    if (dist == Distribution::HubHeavy) {
        // All hub-heavy iterations build a hub — but keep the degree
        // small (200-600) for 80% of them, medium (600-1200) for 15%,
        // large (1200-2000) for 5%, to cap cost.
        p.num_srcs = 1 + static_cast<size_t>(rng() % 10u);
        p.dsts_per_src = 1 + static_cast<size_t>(rng() % 5u);
        p.has_hub = true;
        const uint32_t hb = static_cast<uint32_t>(rng() % 100u);
        if (hb < 80u) {
            p.hub_degree = 200 + static_cast<size_t>(rng() % 401u);  // [200,600]
        } else if (hb < 95u) {
            p.hub_degree = 600 + static_cast<size_t>(rng() % 601u);  // [600,1200]
        } else {
            p.hub_degree = 1200 + static_cast<size_t>(rng() % 801u); // [1200,2000]
        }
        return p;
    }

    if (bucket < 85u) {
        // Small page.
        p.num_srcs     = 1 + static_cast<size_t>(rng() % 30u);
        p.dsts_per_src = 1 + static_cast<size_t>(rng() % 5u);
    } else if (bucket < 98u) {
        // Medium.
        p.num_srcs     = 1 + static_cast<size_t>(rng() % 200u);
        p.dsts_per_src = 1 + static_cast<size_t>(rng() % 8u);
    } else {
        // Heavier, but no hub: exercise many-srcs-per-page.
        p.num_srcs     = 1 + static_cast<size_t>(rng() % 500u);
        p.dsts_per_src = 1 + static_cast<size_t>(rng() % 3u);
    }
    p.has_hub = false;
    p.hub_degree = 0;
    return p;
}

// Fill the groups[] vector with sorted src groups according to the
// distribution. The caller may add a hub tail via append_hub_group.
void generate_groups(std::mt19937_64&      rng,
                     Distribution          dist,
                     const Plan&           plan,
                     std::vector<SrcGroup>& groups)
{
    groups.clear();
    groups.reserve(plan.num_srcs + (plan.has_hub ? 1 : 0));

    switch (dist) {
    case Distribution::DenseMonotonic: {
        // src increments by 1..10 each group; dsts uniform in a small range.
        uint64_t cur_src = static_cast<uint64_t>(rng() % 10'000u);
        for (size_t i = 0; i < plan.num_srcs; ++i) {
            SrcGroup g;
            g.src = cur_src;
            cur_src += 1u + static_cast<uint64_t>(rng() % 10u);
            const size_t d = 1u + static_cast<size_t>(rng() % plan.dsts_per_src);
            g.dsts.reserve(d);
            uint64_t d_cur = static_cast<uint64_t>(rng() % 1'000'000u);
            for (size_t j = 0; j < d; ++j) {
                g.dsts.push_back(d_cur);
                d_cur += 1u + static_cast<uint64_t>(rng() % 100u);
            }
            groups.push_back(std::move(g));
        }
        break;
    }
    case Distribution::SparseRandom: {
        // Random srcs (must still be non-decreasing across groups, so
        // sample then sort). Random dsts per group.
        std::vector<uint64_t> srcs;
        srcs.reserve(plan.num_srcs);
        for (size_t i = 0; i < plan.num_srcs; ++i) {
            srcs.push_back(rng() & 0xFFFFFFFFu);  // 32-bit range
        }
        std::sort(srcs.begin(), srcs.end());
        // Strict monotonic: dedupe. BPTLeafCSRWriter groups-by-src so
        // duplicates are fine, but the reader returns ONE entry per
        // unique src — to keep the post-decode comparison simple,
        // dedupe here.
        srcs.erase(std::unique(srcs.begin(), srcs.end()), srcs.end());
        for (uint64_t src : srcs) {
            SrcGroup g;
            g.src = src;
            const size_t d = 1u + static_cast<size_t>(rng() % plan.dsts_per_src);
            g.dsts.reserve(d);
            for (size_t j = 0; j < d; ++j) {
                g.dsts.push_back(rng() & 0xFFFFFFFFFFFFull);  // 48-bit dst
            }
            groups.push_back(std::move(g));
        }
        break;
    }
    case Distribution::Clustered: {
        // 10 srcs × ~50 dsts each (nominal) — clip num_srcs to ≤ 10,
        // inflate dsts_per_src.
        const size_t srcs_to_use = std::min<size_t>(plan.num_srcs, 10u);
        const size_t dsts_nominal = std::max<size_t>(plan.dsts_per_src * 10, 20u);
        uint64_t cur_src = static_cast<uint64_t>(rng() % 1'000u);
        for (size_t i = 0; i < srcs_to_use; ++i) {
            SrcGroup g;
            g.src = cur_src;
            cur_src += 1u + static_cast<uint64_t>(rng() % 100u);
            const size_t d = 1u + static_cast<size_t>(rng() % dsts_nominal);
            g.dsts.reserve(d);
            uint64_t base = static_cast<uint64_t>(rng() % 1'000'000u);
            for (size_t j = 0; j < d; ++j) {
                g.dsts.push_back(base + (rng() % 10'000u));
            }
            std::sort(g.dsts.begin(), g.dsts.end());
            // Dedupe — the writer does not dedupe and the reader returns
            // exactly what was written. Duplicates would confuse the
            // "matching input" assertion only if we compared sets; we
            // compare vectors, so duplicates ARE permitted. But to keep
            // generator behavior predictable and avoid zero-length dsts
            // after dedup, only dedupe if the caller wants it — here we
            // skip dedupe to exercise delta-of-zero encodings.
            groups.push_back(std::move(g));
        }
        break;
    }
    case Distribution::HubHeavy: {
        // 1-2 small leading srcs + plan.has_hub tail.
        uint64_t cur_src = 1ULL + static_cast<uint64_t>(rng() % 100u);
        const size_t leading = std::min<size_t>(plan.num_srcs, 2u);
        for (size_t i = 0; i < leading; ++i) {
            SrcGroup g;
            g.src = cur_src;
            cur_src += 1u;
            const size_t d = 1u + static_cast<size_t>(rng() % 5u);
            g.dsts.reserve(d);
            uint64_t d_cur = static_cast<uint64_t>(rng() % 1000u);
            for (size_t j = 0; j < d; ++j) {
                g.dsts.push_back(d_cur);
                d_cur += 1u + static_cast<uint64_t>(rng() % 50u);
            }
            groups.push_back(std::move(g));
        }
        // Hub tail (appended below by caller).
        break;
    }
    }
}

// Append a hub src group with a specified degree. Hub dst stride is
// varied so some hubs fit inside 1-byte varints and others stretch to
// multi-byte zigzag deltas.
void append_hub_group(std::mt19937_64& rng,
                      size_t           degree,
                      uint64_t         after_src,
                      std::vector<SrcGroup>& groups)
{
    SrcGroup hub;
    hub.src = after_src + 1u + (rng() % 10u);
    hub.dsts.reserve(degree);
    // Pick a stride regime: 50% small stride (1..8), 30% medium (100..10k),
    // 20% large (≥ 1M, forcing 3+-byte deltas).
    const uint32_t rs = static_cast<uint32_t>(rng() % 100u);
    uint64_t d_cur = static_cast<uint64_t>(rng() % 1'000'000u);
    for (size_t j = 0; j < degree; ++j) {
        hub.dsts.push_back(d_cur);
        uint64_t step;
        if (rs < 50u)       step = 1u + (rng() % 8u);
        else if (rs < 80u)  step = 100u + (rng() % 10'000u);
        else                step = 1'000'000u + (rng() % 10'000'000u);
        d_cur += step;
    }
    groups.push_back(std::move(hub));
}

// ----------------------------------------------------------------------------
// Write groups via BPTLeafCSRWriter<N> to a temp file, then read the file
// back and decode via the T8.4 BPTLeafCSR<N> reader + chain walk. Compare
// (src, degree, dsts[]) tuples with the original groups. Returns true on
// mismatch (caller logs context).
// ----------------------------------------------------------------------------

std::vector<char> read_file_bytes(const std::string& path)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> out(static_cast<std::size_t>(sz));
    if (sz > 0) f.read(out.data(), static_cast<std::streamsize>(sz));
    return out;
}

inline const char* page_at(const std::vector<char>& bytes, std::size_t page_num)
{
    const std::size_t off = page_num * Page::SIZE;
    if (off + Page::SIZE > bytes.size()) return nullptr;
    return bytes.data() + off;
}

// Walk the file page by page; return a vector of decoded (src, dsts) in
// write order. Throws BPTLeafCSRDecodeException on reader-side failure.
template <std::size_t N>
std::vector<SrcGroup>
decode_all_entries(const std::vector<char>& bytes)
{
    std::vector<SrcGroup> out;
    const std::size_t num_pages = bytes.size() / Page::SIZE;

    std::size_t page = 0;
    while (page < num_pages) {
        const char* p = page_at(bytes, page);
        if (p == nullptr) break;
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(p);

        if (raw[0] != 3) { ++page; continue; }
        if (BPT::is_csr_continuation(raw)) { ++page; continue; }

        BPTLeafCSR<N> leaf(p, typename BPTLeafCSR<N>::ReadTag{});
        const uint32_t vc = leaf.src_entry_count();

        // For each src entry on this chain-head page, decode the
        // (src, degree, dsts[]) tuple. Hub dsts may spill onto one or
        // more continuation pages following this chain head.
        for (uint32_t i = 0; i < vc; ++i) {
            const uint8_t* off_ptr = raw + 16 + 2 * i;
            const uint32_t off = static_cast<uint32_t>(off_ptr[0])
                               | (static_cast<uint32_t>(off_ptr[1]) << 8);
            const uint8_t* end = raw + Page::SIZE;

            uint64_t src_id = 0;
            uint64_t degree = 0;
            const uint8_t* in = raw + off;
            in += BPT::varint_decode(in, end, src_id);
            in += BPT::varint_decode(in, end, degree);

            uint32_t start_off = 0;
            uint32_t deg_out   = 0;
            if (!leaf.find_src_entry(src_id, start_off, deg_out)
                || deg_out != static_cast<uint32_t>(degree))
            {
                SrcGroup bogus;
                bogus.src = src_id;
                out.push_back(std::move(bogus));
                continue;
            }

            SrcGroup g;
            g.src = src_id;
            g.dsts.reserve(deg_out);

            // Determine how many dsts live on the chain head vs on
            // continuations: total_continuation_count = sum of chunk_counts.
            uint64_t total_continuation_count = 0;
            {
                uint32_t probe_pg = leaf.next_leaf();
                while (probe_pg != 0 && probe_pg < num_pages) {
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

            const uint8_t* dst_in = raw + start_off;
            uint64_t running = 0;
            for (uint32_t k = 0; k < k_head_target; ++k) {
                uint64_t v = 0;
                dst_in += BPT::varint_decode(dst_in, end, v);
                if (k == 0) {
                    running = v;
                } else {
                    const int64_t delta = BPT::zigzag_decode_u64(v);
                    running += static_cast<uint64_t>(delta);
                }
                g.dsts.push_back(running);
            }
            uint32_t k_decoded = k_head_target;

            if (k_decoded < degree) {
                uint32_t next_pg = leaf.next_leaf();
                while (k_decoded < degree && next_pg != 0 && next_pg < num_pages) {
                    const char* cp = page_at(bytes, next_pg);
                    if (cp == nullptr) break;
                    const uint8_t* craw = reinterpret_cast<const uint8_t*>(cp);
                    uint8_t hdrbuf[16];
                    std::memcpy(hdrbuf, craw, 16);
                    if (!BPT::is_csr_continuation(hdrbuf)) break;
                    auto chdr = BPT::deserialize_csr_continuation_header(hdrbuf);

                    const uint8_t* cin  = craw + 16;
                    const uint8_t* cend = craw + Page::SIZE;
                    for (uint32_t j = 0; j < chdr.chunk_count && k_decoded < degree; ++j) {
                        uint64_t v = 0;
                        cin += BPT::varint_decode(cin, cend, v);
                        const int64_t delta = BPT::zigzag_decode_u64(v);
                        running += static_cast<uint64_t>(delta);
                        g.dsts.push_back(running);
                        ++k_decoded;
                    }
                    next_pg = chdr.next_leaf;
                }
            }

            out.push_back(std::move(g));
        }

        // Skip over any continuation pages that follow this chain head.
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

// Returns true on mismatch.
template <std::size_t N>
bool run_one_iteration(std::mt19937_64& rng,
                       Distribution     dist,
                       const Plan&      plan)
{
    static_assert(N == 2 || N == 3, "CSR writer supports N in {2, 3}");

    std::vector<SrcGroup> groups;
    generate_groups(rng, dist, plan, groups);
    if (plan.has_hub) {
        const uint64_t last_src = groups.empty() ? 0ULL : groups.back().src;
        append_hub_group(rng, plan.hub_degree, last_src, groups);
    }
    if (groups.empty()) return false;  // no-op

    TempFile tf;
    {
        BPTLeafCSRWriter<N> w(tf.path);
        for (const auto& g : groups) {
            for (uint64_t dst : g.dsts) {
                std::array<uint64_t, N> rec{};
                rec[0] = g.src;
                rec[1] = dst;
                // Higher dimensions are unused by writer; leave zero.
                w.append(rec);
            }
        }
    }

    auto bytes = read_file_bytes(tf.path);
    if (bytes.empty()) return true;

    std::vector<SrcGroup> decoded;
    try {
        decoded = decode_all_entries<N>(bytes);
    } catch (const std::exception&) {
        return true;
    }

    if (decoded.size() != groups.size()) return true;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (decoded[i].src != groups[i].src) return true;
        if (decoded[i].dsts.size() != groups[i].dsts.size()) return true;
        for (std::size_t k = 0; k < groups[i].dsts.size(); ++k) {
            if (decoded[i].dsts[k] != groups[i].dsts[k]) return true;
        }
    }
    return false;
}

// Single-iteration dispatcher: picks N ∈ {2, 3} and a distribution.
bool run_one(std::mt19937_64& rng)
{
    const unsigned n  = 2u + static_cast<unsigned>(rng() % 2u);      // 2 or 3
    const auto dist   = static_cast<Distribution>(rng() % 4u);
    const Plan plan   = pick_plan(rng, dist);
    if (n == 2) return run_one_iteration<2>(rng, dist, plan);
    else        return run_one_iteration<3>(rng, dist, plan);
}

}  // namespace

// ============================================================================
// Main 500k-iteration fuzz run.
// ============================================================================
TEST(BPTLeafCSRFuzzTest, HalfMillionRoundtrips_MainSeed) {
    std::mt19937_64 rng(MAIN_SEED);
    size_t mismatches = 0;
    for (size_t i = 0; i < MAIN_ITERATIONS; ++i) {
        if (run_one(rng)) {
            ++mismatches;
            ADD_FAILURE() << "fuzz mismatch at iteration " << i
                          << " seed=0x" << std::hex << MAIN_SEED << std::dec;
            if (mismatches >= 5) {
                FAIL() << "too many mismatches; aborting";
            }
        }
        if ((i + 1) % 50'000 == 0) {
            std::cerr << "  csr fuzz progress: " << (i + 1) << " / "
                      << MAIN_ITERATIONS << '\n';
        }
    }
    ASSERT_EQ(mismatches, 0u);
}

// ============================================================================
// Smoke run with distinct seed — runs in Debug/ASan quickly.
// ============================================================================
TEST(BPTLeafCSRFuzzTest, SmokeRun_SmokeSeed) {
    std::mt19937_64 rng(SMOKE_SEED);
    for (size_t i = 0; i < SMOKE_ITERATIONS; ++i) {
        const bool mismatch = run_one(rng);
        ASSERT_FALSE(mismatch) << "smoke run mismatch at iter " << i;
    }
}

// ============================================================================
// Boundary cases — explicit adversarial inputs.
// ============================================================================
namespace {

// Verify a single group against its writer output.
template <std::size_t N>
void verify_groups(const std::vector<SrcGroup>& groups)
{
    if (groups.empty()) return;
    TempFile tf;
    {
        BPTLeafCSRWriter<N> w(tf.path);
        for (const auto& g : groups) {
            for (uint64_t dst : g.dsts) {
                std::array<uint64_t, N> rec{};
                rec[0] = g.src;
                rec[1] = dst;
                w.append(rec);
            }
        }
    }
    auto bytes = read_file_bytes(tf.path);
    ASSERT_FALSE(bytes.empty());
    std::vector<SrcGroup> decoded = decode_all_entries<N>(bytes);
    ASSERT_EQ(decoded.size(), groups.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ASSERT_EQ(decoded[i].src, groups[i].src) << "boundary src mismatch at i=" << i;
        ASSERT_EQ(decoded[i].dsts.size(), groups[i].dsts.size())
            << "boundary degree mismatch at i=" << i
            << " src=" << groups[i].src;
        for (std::size_t k = 0; k < groups[i].dsts.size(); ++k) {
            ASSERT_EQ(decoded[i].dsts[k], groups[i].dsts[k])
                << "boundary dst mismatch at i=" << i
                << " src=" << groups[i].src << " k=" << k;
        }
    }
}

}  // namespace

TEST(BPTLeafCSRFuzzTest, BoundaryCases) {
    // (1) Single src, single dst.
    {
        std::vector<SrcGroup> g = { {42, {7}} };
        verify_groups<3>(g);
        verify_groups<2>(g);
    }

    // (2) 1000-dst hub (spans continuation pages).
    {
        SrcGroup hub;
        hub.src = 100;
        hub.dsts.reserve(1000);
        uint64_t d = 1;
        for (size_t i = 0; i < 1000; ++i) { hub.dsts.push_back(d); d += 1000; }
        std::vector<SrcGroup> g = { hub };
        verify_groups<3>(g);
    }

    // (3) 500 srcs × 1 dst each (dense offset table).
    {
        std::vector<SrcGroup> g;
        g.reserve(500);
        for (uint64_t s = 1; s <= 500; ++s) {
            g.push_back({ s, { s * 2 } });
        }
        verify_groups<3>(g);
    }

    // (4) All srcs same degree — highest compressibility.
    {
        std::vector<SrcGroup> g;
        for (uint64_t s = 10; s < 30; ++s) {
            SrcGroup gg;
            gg.src = s;
            gg.dsts = {1, 2, 3, 4, 5};
            g.push_back(std::move(gg));
        }
        verify_groups<3>(g);
    }

    // (5) Two hubs + many small interleaved.
    {
        std::vector<SrcGroup> g;
        g.push_back({ 1, {10, 20, 30} });
        {
            SrcGroup hub1;
            hub1.src = 2;
            hub1.dsts.reserve(400);
            uint64_t d = 100;
            for (size_t i = 0; i < 400; ++i) { hub1.dsts.push_back(d); d += 50; }
            g.push_back(std::move(hub1));
        }
        for (uint64_t s = 3; s <= 10; ++s) g.push_back({ s, { s * 100, s * 100 + 1 } });
        {
            SrcGroup hub2;
            hub2.src = 11;
            hub2.dsts.reserve(800);
            uint64_t d = 1'000'000;
            for (size_t i = 0; i < 800; ++i) { hub2.dsts.push_back(d); d += 31; }
            g.push_back(std::move(hub2));
        }
        for (uint64_t s = 12; s <= 20; ++s) g.push_back({ s, { s, s + 1 } });
        verify_groups<3>(g);
    }

    // (6) Max-range dsts (near UINT64_MAX).
    {
        constexpr uint64_t MAX = std::numeric_limits<uint64_t>::max();
        std::vector<SrcGroup> g;
        g.push_back({ 1, {0, MAX / 4, MAX / 2, (MAX / 4) * 3, MAX - 1} });
        g.push_back({ 2, {1, 2, MAX - 1, MAX} });
        verify_groups<3>(g);
    }

    // (7) Randomized boundary iteration — reuse the main dispatcher with a
    //     distinct seed, biased toward adversarial plans.
    std::mt19937_64 rng(MAIN_SEED ^ 0xB0DACAFE01234567ULL);
    size_t mismatches = 0;
    for (size_t i = 0; i < BOUNDARY_ITERATIONS; ++i) {
        if (run_one(rng)) {
            ++mismatches;
            ADD_FAILURE() << "boundary-subset mismatch at iter " << i;
            if (mismatches >= 5) {
                FAIL() << "too many boundary mismatches; aborting";
            }
        }
    }
    ASSERT_EQ(mismatches, 0u);
}

// ============================================================================
// Tamper-injection — every bit flip in a chain-head page must be caught.
// ============================================================================
namespace {

// Build a valid ~50-record chain-head page and return (file bytes,
// ground-truth groups). The caller flips bits within the first
// `bytes_in_payload` bytes of the first page and decodes.
struct TamperSubject {
    std::vector<char>      page_bytes;   // exactly 4096 bytes
    std::vector<SrcGroup>  truth;
};

TamperSubject build_tamper_subject()
{
    // Dense-monotonic packing: 50 srcs × 1 dst each. Stays on a single
    // chain-head page, maximizing the payload area exposed to corruption.
    std::vector<SrcGroup> truth;
    truth.reserve(50);
    for (uint64_t s = 1000; s < 1050; ++s) {
        truth.push_back({ s, { s * 7 } });
    }

    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path);
        for (const auto& g : truth) {
            for (uint64_t dst : g.dsts) {
                w.append({g.src, dst, 0});
            }
        }
    }
    auto all = read_file_bytes(tf.path);
    // Should be exactly one page.
    TamperSubject ts;
    ts.page_bytes.assign(all.begin(), all.begin() + Page::SIZE);
    ts.truth = std::move(truth);
    return ts;
}

// Decode a single-page buffer directly via the reader + synchronous
// walk (no continuation chain to follow because the subject is one
// page). Returns true if the decode produced ground-truth groups; false
// if it differed (materially detected corruption). Throws are caught by
// the caller and counted as detection.
bool decode_single_page_exact(const char* page,
                              const std::vector<SrcGroup>& truth)
{
    // The BPTLeafCSR reader ctor validates the header and offset table.
    // If the flip corrupts any of those invariants it will throw, which
    // the caller counts as detected_exception.
    BPTLeafCSR<3> leaf(page, BPTLeafCSR<3>::ReadTag{});
    const uint32_t vc = leaf.src_entry_count();
    if (vc != static_cast<uint32_t>(truth.size())) return false;

    const uint8_t* raw = reinterpret_cast<const uint8_t*>(page);
    const uint8_t* end = raw + Page::SIZE;

    for (uint32_t i = 0; i < vc; ++i) {
        const uint8_t* off_ptr = raw + 16 + 2 * i;
        const uint32_t off = static_cast<uint32_t>(off_ptr[0])
                           | (static_cast<uint32_t>(off_ptr[1]) << 8);
        if (off >= Page::SIZE) return false;

        uint64_t src_id = 0;
        uint64_t degree = 0;
        const uint8_t* in = raw + off;
        in += BPT::varint_decode(in, end, src_id);
        in += BPT::varint_decode(in, end, degree);

        if (src_id != truth[i].src) return false;
        if (degree != truth[i].dsts.size()) return false;

        uint32_t start_off = 0;
        uint32_t deg_out   = 0;
        if (!leaf.find_src_entry(src_id, start_off, deg_out)) return false;
        if (deg_out != degree) return false;

        // No continuations — all dsts live on the head.
        const uint8_t* dst_in = raw + start_off;
        uint64_t running = 0;
        for (uint32_t k = 0; k < deg_out; ++k) {
            uint64_t v = 0;
            dst_in += BPT::varint_decode(dst_in, end, v);
            if (k == 0) {
                running = v;
            } else {
                const int64_t delta = BPT::zigzag_decode_u64(v);
                running += static_cast<uint64_t>(delta);
            }
            if (running != truth[i].dsts[k]) return false;
        }
    }
    return true;
}

}  // namespace

TEST(BPTLeafCSRFuzzTest, TamperInjection_AllDetected) {
    auto subject = build_tamper_subject();
    ASSERT_EQ(subject.page_bytes.size(), static_cast<std::size_t>(Page::SIZE));

    // Determine the tamper region: [0, used_bytes). Past used_bytes is
    // zero padding that the reader does not inspect for the on-head
    // dst stream, so flips there cannot (in general) be detected.
    //
    // Compute used_bytes as (header 16 + offset table 2*vc + sum of
    // entry body lengths). We derive it by decoding the ground truth
    // layout: each entry body = varint(src) + varint(degree) + varint(dst[0])
    // (+ zigzag deltas for i>=1).
    std::size_t used_bytes = 16 + 2 * subject.truth.size();
    for (const auto& g : subject.truth) {
        used_bytes += BPT::varint_size(g.src);
        used_bytes += BPT::varint_size(static_cast<uint64_t>(g.dsts.size()));
        if (!g.dsts.empty()) {
            used_bytes += BPT::varint_size(g.dsts[0]);
            for (std::size_t k = 1; k < g.dsts.size(); ++k) {
                const uint64_t delta_u = g.dsts[k] - g.dsts[k - 1];
                const int64_t  delta_i = static_cast<int64_t>(delta_u);
                used_bytes += BPT::varint_size(BPT::zigzag_encode_i64(delta_i));
            }
        }
    }
    ASSERT_LE(used_bytes, static_cast<std::size_t>(Page::SIZE));

    // Header fields that the reader cannot detect corruption in (by
    // design, per bpt_leaf_csr_format.h + bplus_tree_leaf_csr.cc):
    //
    //   - flags bit 1 (kHasEdgeIds, byte 2 bit 1): writer emits 0; reader
    //     tolerates 1 without using the bit in decode (T8.6+ scope, per
    //     design §3.4). Flipping this bit is semantically harmless.
    //   - next_leaf (bytes 8..11): cross-page pointer. Reader cannot
    //     cross-check without opening the neighbour page. Matches T5.14's
    //     documented exemption for BPTLeafV2's next_leaf.
    //   - min_src_id_low (bytes 12..15): fsck-only cross-check against the
    //     directory routing key (design §3.4). Reader ctor does not
    //     validate this field because the directory side is what routes to
    //     a page, not the header side — corruption here does not affect
    //     record decoding on this page. Matches the "next_leaf"-like
    //     exemption documented in T5.14 for pointer-style header fields.
    //
    // The tamper loop below skips these bytes so detection is measured
    // only against bytes that participate in on-head decoding.
    constexpr std::size_t kNextLeafLo     =  8;
    constexpr std::size_t kNextLeafHi     = 12;  // exclusive
    constexpr std::size_t kMinSrcLowLo    = 12;
    constexpr std::size_t kMinSrcLowHi    = 16;  // exclusive
    constexpr uint8_t     kFlagsSkipMask  = BPT::CSRHybridFlags::kHasEdgeIds;
    constexpr std::size_t kFlagsByte      =  2;

    // Save original for restore between flips.
    std::vector<char> original = subject.page_bytes;

    std::size_t flipped            = 0;
    std::size_t detected_exception = 0;
    std::size_t detected_mismatch  = 0;
    std::size_t silent             = 0;

    for (std::size_t off = 0; off < used_bytes; ++off) {
        if (off >= kNextLeafLo && off < kNextLeafHi) continue;
        if (off >= kMinSrcLowLo && off < kMinSrcLowHi) continue;
        for (int bit = 0; bit < 8; ++bit) {
            const uint8_t mask = static_cast<uint8_t>(1u << bit);
            // The flags byte has an allowed-but-unused bit (kHasEdgeIds).
            // Flipping it does not corrupt the record stream and the
            // reader tolerates it by design — skip this specific bit.
            if (off == kFlagsByte && (mask & kFlagsSkipMask) != 0) {
                continue;
            }
            subject.page_bytes[off] = static_cast<char>(
                static_cast<uint8_t>(subject.page_bytes[off]) ^ mask);
            ++flipped;

            bool exception_thrown = false;
            bool records_match    = false;
            try {
                records_match = decode_single_page_exact(subject.page_bytes.data(),
                                                         subject.truth);
            } catch (const std::exception&) {
                exception_thrown = true;
            }

            if (exception_thrown) {
                ++detected_exception;
            } else if (!records_match) {
                ++detected_mismatch;
            } else {
                // Silent corruption — bit flip survived decode AND produced
                // ground-truth records. This is a correctness bug. The
                // exceptions are already handled upstream (next_leaf field).
                ++silent;
                ADD_FAILURE() << "silent corruption: byte=" << off
                              << " bit=" << bit;
            }

            // Restore.
            subject.page_bytes[off] = original[off];
        }
    }

    EXPECT_EQ(silent, 0u) << silent << " bit flips went undetected";
    EXPECT_EQ(flipped, detected_exception + detected_mismatch + silent);
    EXPECT_GE(flipped, 1000u) << "not enough bits flipped to be meaningful";

    std::cerr << "  csr tamper stats: flipped=" << flipped
              << " via_exception=" << detected_exception
              << " via_mismatch=" << detected_mismatch
              << " silent=" << silent << '\n';
}

// ============================================================================
// Spec #8-B task #1 (hub completion) — focused fuzz over hubs with eids.
//
// Generates random hub adjacency lists with parallel edge_ids, runs them
// through BPTLeafCSRWriter<3>(emit_edge_ids=true), and walks the chain via
// the ReadTag + ContinuationTag readers, verifying every (src, dst, eid)
// triple matches the input. Distinct from the main fuzz harness because:
//   - the main harness uses the default writer (no eids), so its tamper
//     coverage already pins the legacy contract;
//   - here we exercise the writer's k_on_head packing + chain-head's
//     extra varint + continuation eid stream end-to-end on randomized
//     inputs.
// ============================================================================
namespace {

struct HubFixture {
    uint64_t              src;
    std::vector<uint64_t> dsts;
    std::vector<uint64_t> eids;
};

HubFixture generate_hub_with_eids(std::mt19937_64& rng)
{
    HubFixture h;
    h.src = static_cast<uint64_t>(rng() % 1'000'000ull);

    // Hub degree in [200, 4000]: small enough for fast iteration, large
    // enough to force at least one continuation page in most cases.
    const std::size_t degree = 200 + (rng() % 3801ull);
    h.dsts.reserve(degree);
    h.eids.reserve(degree);

    // dst stride regime mirrors the main fuzz harness's hub generator.
    const uint32_t rs = static_cast<uint32_t>(rng() % 100u);
    uint64_t d_cur = static_cast<uint64_t>(rng() % 1'000'000ull);
    uint64_t e_cur = static_cast<uint64_t>(rng() % 1'000'000ull);
    for (std::size_t i = 0; i < degree; ++i) {
        h.dsts.push_back(d_cur);
        h.eids.push_back(e_cur);
        uint64_t dstep, estep;
        if (rs < 50u) {
            dstep = 1ull + (rng() % 8ull);
            estep = 1ull + (rng() % 5ull);
        } else if (rs < 80u) {
            dstep = 100ull + (rng() % 10'000ull);
            estep = 50ull + (rng() % 1000ull);
        } else {
            dstep = 1'000'000ull + (rng() % 10'000'000ull);
            estep = 100ull + (rng() % 100'000ull);
        }
        d_cur += dstep;
        e_cur += estep;
    }
    return h;
}

bool fuzz_one_hub_with_eids(std::mt19937_64& rng)
{
    HubFixture h = generate_hub_with_eids(rng);

    TempFile tf;
    {
        BPTLeafCSRWriter<3> w(tf.path, /*emit_edge_ids=*/true);
        for (std::size_t i = 0; i < h.dsts.size(); ++i) {
            w.append({h.src, h.dsts[i], h.eids[i]});
        }
    }
    auto bytes = read_file_bytes(tf.path);
    if (bytes.empty()) return true;

    // Walk pages: chain-head first, then continuations.
    BPTLeafCSR<3> head(bytes.data(), BPTLeafCSR<3>::ReadTag{});
    std::vector<std::array<uint64_t, 3>> got;
    got.reserve(h.dsts.size());
    uint64_t carry_dst = 0;
    uint64_t carry_eid = 0;
    for (uint32_t i = 0; i < head.get_value_count(); ++i) {
        Record<3> r = head.get_record(i);
        got.push_back({r[0], r[1], r[2]});
        carry_dst = r[1];
        carry_eid = r[2];
    }
    uint32_t pg = head.next_leaf();
    while (pg != 0) {
        const char* p = page_at(bytes, pg);
        if (p == nullptr) return true;
        BPTLeafCSR<3> cont(p, BPTLeafCSR<3>::ContinuationTag{
            h.src, carry_dst, carry_eid});
        for (uint32_t i = 0; i < cont.get_value_count(); ++i) {
            Record<3> r = cont.get_record(i);
            got.push_back({r[0], r[1], r[2]});
            carry_dst = r[1];
            carry_eid = r[2];
        }
        pg = cont.next_leaf();
    }

    if (got.size() != h.dsts.size()) return true;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i][0] != h.src) return true;
        if (got[i][1] != h.dsts[i]) return true;
        if (got[i][2] != h.eids[i]) return true;
    }
    return false;
}

}  // namespace

TEST(BPTLeafCSRFuzzTest, HubEdgeIds_RoundTripFuzz) {
#if MDB_FUZZ_UNDER_ASAN
    constexpr std::size_t kIters = 200;
#else
    constexpr std::size_t kIters = 2000;
#endif
    constexpr uint64_t HUB_EID_SEED = 0xABCD1234EFF00DULL;
    std::mt19937_64 rng(HUB_EID_SEED);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < kIters; ++i) {
        if (fuzz_one_hub_with_eids(rng)) {
            ++mismatches;
            ADD_FAILURE() << "hub-eid fuzz mismatch at iter " << i;
            if (mismatches >= 3) break;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}
