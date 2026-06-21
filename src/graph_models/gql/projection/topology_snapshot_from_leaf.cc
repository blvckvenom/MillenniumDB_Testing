#include "graph_models/gql/projection/topology_snapshot_from_leaf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef HAS_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

#include "graph_models/object_id.h"
#include "storage/page/page.h"

namespace GQL::Projection {

namespace {

// .leaf source basename for each CSR direction (mirrors
// topology_snapshot_writer.cc's source_basename_for).
const char* source_basename_for(TopologySnapshotWriter::Direction d) {
    switch (d) {
    case TopologySnapshotWriter::Direction::FORWARD: return "from_to_edge.leaf";
    case TopologySnapshotWriter::Direction::REVERSE: return "to_from_edge.leaf";
    }
    return "from_to_edge.leaf";
}

// Env var resolution for the parallel snapshot build.
// Mirrors the resolve_bool_env / resolve_partition_count_env pattern from
// native_scanner.cc so the operator surface is consistent across the other
// parallel projection-build stages.
bool resolve_bool_env(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (env == nullptr) {
        return default_value;
    }
    std::string v(env);
    if (v == "0" || v == "false" || v == "off" ||
        v == "FALSE" || v == "OFF" || v == "False" || v == "Off")
    {
        return false;
    }
    return true;
}

std::size_t resolve_snapshot_partition_count() {
    constexpr std::size_t kMin = 2;
    constexpr std::size_t kMaxCap = 64;
    constexpr std::size_t kDefaultCap = 16;

    std::size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 4;
    }
    std::size_t k = std::min<std::size_t>(hw, kDefaultCap);

    if (const char* env = std::getenv("MDB_PROJECTION_SNAPSHOT_PARTITIONS")) {
        try {
            long parsed = std::stol(env);
            if (parsed > 0) {
                k = static_cast<std::size_t>(parsed);
            }
        } catch (...) {
            // ignore malformed values
        }
    }
    return std::clamp(k, kMin, kMaxCap);
}

bool resolve_parallel_snapshot_enabled() {
    return resolve_bool_env("MDB_PROJECTION_PARALLEL_SNAPSHOT", true);
}

// RAII wrapper over an mmap region. Constructed mapped-and-owning; the
// destructor munmaps and closes. Move-only.
class MappedFile {
public:
    static MappedFile open_readonly(const std::filesystem::path& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            int e = errno;
            throw std::runtime_error(
                "topology_snapshot_from_leaf: open failed for "
                + path.string()
                + " (errno=" + std::to_string(e) + ")");
        }

        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            int e = errno;
            ::close(fd);
            throw std::runtime_error(
                "topology_snapshot_from_leaf: fstat failed for "
                + path.string()
                + " (errno=" + std::to_string(e) + ")");
        }

        const std::size_t len = static_cast<std::size_t>(st.st_size);
        if (len == 0) {
            ::close(fd);
            return MappedFile(nullptr, 0, -1);
        }

        void* addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            int e = errno;
            ::close(fd);
            throw std::runtime_error(
                "topology_snapshot_from_leaf: mmap failed for "
                + path.string()
                + " (errno=" + std::to_string(e) + ")");
        }

        // Hint the kernel that we will scan sequentially. This keeps the
        // readahead window aggressive for the two sequential passes and
        // evicts each page once we've moved past it, so the mmap of a
        // 37 GB .leaf on papers100M doesn't bloat RSS.
        ::madvise(addr, len, MADV_SEQUENTIAL);

        return MappedFile(addr, len, fd);
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept
        : base_(other.base_), len_(other.len_), fd_(other.fd_) {
        other.base_ = nullptr;
        other.len_  = 0;
        other.fd_   = -1;
    }

    ~MappedFile() {
        if (base_ != nullptr && len_ > 0) {
            ::munmap(base_, len_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    const uint8_t* data() const noexcept {
        return static_cast<const uint8_t*>(base_);
    }
    std::size_t size() const noexcept { return len_; }
    bool        empty() const noexcept { return base_ == nullptr || len_ == 0; }

private:
    MappedFile(void* base, std::size_t len, int fd)
        : base_(base), len_(len), fd_(fd) {}

    void*       base_ = nullptr;
    std::size_t len_  = 0;
    int         fd_   = -1;
};

// Leaf page layout constants. See bpt_mem_import.h:31-44 and
// projection_storage.cc::build_index_streaming::write_leaf_page.
constexpr std::size_t kLeafPageSize    = Page::SIZE;
constexpr std::size_t kLeafRecordArity = 3;
constexpr std::size_t kLeafRecordBytes = kLeafRecordArity * sizeof(uint64_t);
constexpr std::size_t kLeafBitsetBytes = kLeafRecordArity;   // N bytes (N==3)
constexpr std::size_t kLeafHeaderBytes = 2 * sizeof(uint32_t);

// Per-page redundant-byte view. BPTLeafWriter<3>::process_block may emit
// COMPRESSED leaves (default since 2026-06-16): bytes that are constant across
// every record on the page are stored ONCE in a "redundant" section right
// after the N-byte bitset, and only the non-constant bytes are stored per
// record. This direct-byte walker must decode that exactly the way BPTLeafV1
// does (bplus_tree_leaf.cc:79 set_record). When the bitset is all-zero
// (MDB_PROJECTION_NO_LEAF_COMPRESSION=1 or no redundancy) this degrades to the
// legacy fixed-stride layout with zero overhead.
struct LeafPageDecoder {
    const uint8_t* redundant_bytes; // start of the shared redundant section
    const uint8_t* records;         // start of the per-record non-redundant data
    std::size_t    redundant_count; // number of set bits = redundant byte count
    std::size_t    unique_bytes;    // bytes per record on disk = 24 - redundant_count
    std::array<uint8_t, kLeafRecordBytes> redundant_bit; // 1 if byte pos is redundant
    std::array<std::size_t, kLeafRecordBytes> redundant_idx; // pos -> index into redundant section

    explicit LeafPageDecoder(const uint8_t* page) {
        const uint8_t* bitset_ptr = page + kLeafHeaderBytes;
        redundant_count = 0;
        for (std::size_t i = 0; i < kLeafRecordBytes; ++i) {
            const uint8_t byte_pos_bit =
                (bitset_ptr[i / 8] >> (i % 8)) & 1u;
            redundant_bit[i] = byte_pos_bit;
            if (byte_pos_bit) {
                redundant_idx[i] = redundant_count;
                ++redundant_count;
            } else {
                redundant_idx[i] = 0;
            }
        }
        unique_bytes    = kLeafRecordBytes - redundant_count;
        redundant_bytes = bitset_ptr + kLeafBitsetBytes;
        records         = redundant_bytes + redundant_count;
    }

    // Total on-disk bytes from the start of the page through `count` records.
    std::size_t page_bytes(uint32_t count) const {
        return kLeafHeaderBytes + kLeafBitsetBytes + redundant_count
             + static_cast<std::size_t>(count) * unique_bytes;
    }

    // Reconstruct record `idx` into out[0..2] (3 uint64), honoring the bitset.
    void decode_record(uint32_t idx, uint64_t out[kLeafRecordArity]) const {
        unsigned char* oc = reinterpret_cast<unsigned char*>(out);
        const uint8_t* cur = records + static_cast<std::size_t>(idx) * unique_bytes;
        std::size_t upos = 0;
        for (std::size_t i = 0; i < kLeafRecordBytes; ++i) {
            if (redundant_bit[i]) {
                oc[i] = redundant_bytes[redundant_idx[i]];
            } else {
                oc[i] = cur[upos++];
            }
        }
    }
};

// Walk leaf pages in `[page_lo, page_hi)` and invoke
// `visit(src_u64_raw, dst_u64, eid_u64)` once per record, in the exact
// order they were written by BPTLeafWriter<3>::process_block.
//
// `[page_lo, page_hi)` is a half-open page index range. The legacy
// whole-file walker corresponds to `page_lo=0, page_hi=num_pages`.
// Empty leaf pages (value_count == 0 — including the make_empty
// sentinel and trailing zero-padding) are skipped silently.
template<typename Visitor>
void visit_leaf_records_in_range(const MappedFile& mm,
                                 std::size_t page_lo,
                                 std::size_t page_hi,
                                 Visitor&& visit) {
    if (mm.empty() || page_lo >= page_hi) {
        return;
    }
    if (mm.size() % kLeafPageSize != 0) {
        throw std::runtime_error(
            "topology_snapshot_from_leaf: .leaf file size ("
            + std::to_string(mm.size())
            + ") is not a multiple of Page::SIZE="
            + std::to_string(kLeafPageSize));
    }

    const uint8_t* base = mm.data();
    const std::size_t num_pages = mm.size() / kLeafPageSize;
    const std::size_t hi = std::min(page_hi, num_pages);

    for (std::size_t page = page_lo; page < hi; ++page) {
        const uint8_t* p = base + page * kLeafPageSize;

        uint32_t value_count = 0;
        std::memcpy(&value_count, p, sizeof(uint32_t));

        if (value_count == 0) {
            continue;
        }

        // Decode the redundant-byte bitset so compressed pages reconstruct
        // correctly (see LeafPageDecoder). On uncompressed pages this is the
        // legacy fixed-stride layout.
        const LeafPageDecoder dec(p);

        const std::size_t needed = dec.page_bytes(value_count);
        if (needed > kLeafPageSize) {
            throw std::runtime_error(
                "topology_snapshot_from_leaf: corrupt leaf page "
                + std::to_string(page) + " (value_count="
                + std::to_string(value_count) + " implies "
                + std::to_string(needed) + " B > "
                + std::to_string(kLeafPageSize) + " B Page::SIZE)");
        }

        for (uint32_t i = 0; i < value_count; ++i) {
            uint64_t r[kLeafRecordArity] = {0, 0, 0};
            dec.decode_record(i, r);
            visit(r[0], r[1], r[2]);
        }
    }
}

template<typename Visitor>
void visit_leaf_records(const MappedFile& mm, Visitor&& visit) {
    if (mm.empty()) {
        return;
    }
    if (mm.size() % kLeafPageSize != 0) {
        throw std::runtime_error(
            "topology_snapshot_from_leaf: .leaf file size ("
            + std::to_string(mm.size())
            + ") is not a multiple of Page::SIZE="
            + std::to_string(kLeafPageSize));
    }
    visit_leaf_records_in_range(mm, 0, mm.size() / kLeafPageSize,
                                std::forward<Visitor>(visit));
}

// Read the first record's `key[0]` (i.e. the page-leading src ObjectId,
// before tag-stripping) for every non-empty page, plus the last record's
// `key[0]`. Used by the parallel snapshot build to assign each worker
// its page range.
//
// Returns a vector of (first_src, last_src) pairs of length num_pages;
// empty pages get sentinel (UINT64_MAX, 0) so the "find first page with
// first_src >= lo" search ignores them by construction.
//
// Cost: O(num_pages) random reads of Page-aligned regions; for
// papers100M's 37 GB .leaf this is ~9.5 M pages × 32 B touched per
// page = ~300 MB of streamed reads. The OS prefetches sequentially so
// after this primer pass both Pass 1 and Pass 2 hit the page cache.
struct LeafPageBoundary {
    uint64_t first_src;  // tag-stripped src of the first record on the page
    uint64_t last_src;   // tag-stripped src of the last record on the page
};

std::vector<LeafPageBoundary> build_page_boundaries(const MappedFile& mm) {
    if (mm.empty()) {
        return {};
    }
    if (mm.size() % kLeafPageSize != 0) {
        throw std::runtime_error(
            "topology_snapshot_from_leaf: .leaf file size ("
            + std::to_string(mm.size())
            + ") is not a multiple of Page::SIZE="
            + std::to_string(kLeafPageSize));
    }
    const uint8_t* base = mm.data();
    const std::size_t num_pages = mm.size() / kLeafPageSize;

    std::vector<LeafPageBoundary> out(num_pages,
                                      LeafPageBoundary{UINT64_MAX, 0});
    for (std::size_t page = 0; page < num_pages; ++page) {
        const uint8_t* p = base + page * kLeafPageSize;
        uint32_t value_count = 0;
        std::memcpy(&value_count, p, sizeof(uint32_t));
        if (value_count == 0) {
            continue;
        }
        // Honor the redundant-byte bitset (compressed leaves). key[0] (src)
        // may itself be partly redundant, so reconstruct the full record.
        const LeafPageDecoder dec(p);
        if (dec.page_bytes(value_count) > kLeafPageSize) {
            throw std::runtime_error(
                "topology_snapshot_from_leaf: corrupt leaf page "
                + std::to_string(page) + " (value_count="
                + std::to_string(value_count) + ")");
        }

        uint64_t first_r[kLeafRecordArity] = {0, 0, 0};
        dec.decode_record(0, first_r);
        uint64_t last_r[kLeafRecordArity] = {0, 0, 0};
        dec.decode_record(value_count - 1, last_r);

        out[page].first_src = first_r[0] & ObjectId::VALUE_MASK;
        out[page].last_src  = last_r[0] & ObjectId::VALUE_MASK;
    }
    return out;
}

// Find the first page index that can contain a record with src >= lo_src.
// A record landing on page P has src in [first_src[P], last_src[P]] (the
// .leaf is sorted by key[0]), so the predicate "page P could contain a
// record >= lo_src" is exactly "last_src[P] >= lo_src".
//
// Empty pages carry the sentinel (first_src=UINT64_MAX, last_src=0),
// which makes their last_src=0 < every real lo_src>=1, so the predicate
// correctly skips them on lo_src>=1. For lo_src=0 the very first page
// is always selected, which is what we want.
//
// Returns boundaries.size() if no such page exists (the entire .leaf
// stops before lo_src — the worker has no records to scan).
std::size_t find_first_page_for_src(
    const std::vector<LeafPageBoundary>& boundaries, uint64_t lo_src)
{
    if (lo_src == 0) {
        // Any record is >= 0 by definition, but we still want to skip
        // leading all-empty pages (none in practice for the projection
        // builder, but defend against the make_empty sentinel).
        for (std::size_t p = 0; p < boundaries.size(); ++p) {
            if (boundaries[p].first_src != UINT64_MAX) {
                return p;
            }
        }
        return boundaries.size();
    }
    // Linear scan is fine: the boundaries array length equals
    // num_pages = .leaf size / 4096 B. On papers100M (37 GB) that's
    // ~9.5 M pages × K workers = 76 M comparisons per build, well
    // under 1 second on a single core. The whole "build_parallel"
    // function only runs once per direction per build. A binary search
    // would speed this up to log(num_pages) per worker but requires a
    // monotone predicate, and the trailing empty-page tail breaks
    // monotonicity of last_src; the linear walk is simpler and
    // correct for both small and large fixtures.
    for (std::size_t p = 0; p < boundaries.size(); ++p) {
        if (boundaries[p].first_src == UINT64_MAX) continue;  // skip empty
        if (boundaries[p].last_src >= lo_src) {
            return p;
        }
    }
    return boundaries.size();
}

// Sequential implementation — the original single-threaded code path that
// predates the parallel build. Used as the fallback when
// MDB_PROJECTION_PARALLEL_SNAPSHOT=0, when partitions < 2, or when
// HAS_TBB is not defined.
void build_sequential(const std::filesystem::path& projection_dir,
                      TopologySnapshotWriter::Direction dir,
                      uint64_t num_nodes,
                      bool include_edge_ids,
                      const MappedFile& mm)
{
    // ---- Pass 1: degrees ---------------------------------------------------
    std::vector<uint64_t> degrees(num_nodes, 0);
    visit_leaf_records(mm,
        [&](uint64_t k0, uint64_t /*k1*/, uint64_t /*k2*/) {
            const uint64_t src_idx = k0 & ObjectId::VALUE_MASK;
            if (src_idx < num_nodes) {
                ++degrees[src_idx];
            }
        });

    // ---- Pass 2: stream edges through the writer --------------------------
    TopologySnapshotWriter writer(projection_dir, dir, num_nodes,
                                  std::move(degrees), include_edge_ids);

    visit_leaf_records(mm,
        [&](uint64_t k0, uint64_t k1, uint64_t k2) {
            const uint64_t src_idx = k0 & ObjectId::VALUE_MASK;
            if (src_idx >= num_nodes) {
                return;
            }
            writer.append_edge(ObjectId{src_idx}, ObjectId{k1}, ObjectId{k2});
        });

    writer.finalize();
}

#ifdef HAS_TBB
// Parallel (multi-worker) implementation.
//
// Strategy: assign each worker a disjoint half-open src range
// [lo_src, hi_src) covering the dense node id space. Workers walk the
// shared page-boundary index to find their starting page; they then
// scan pages forward, filtering records in [lo_src, hi_src). Boundary
// pages may be visited by two workers but each only emits records for
// its own src range, so output bytes never collide.
//
// Pass 1 (degrees): each worker writes into its own slice of `degrees`
// (shared array, disjoint slot index ranges). No atomic ops, no
// per-worker N-array allocation.
//
// Pass 2 (edge stream): each worker accumulates dst (and optionally
// edge_id) into per-worker stack vectors of length
// `row_ptr[hi_src] - row_ptr[lo_src]` (= edges in its src range) ordered
// to match the legacy sequential traversal exactly. The writer's
// `append_subrange` then pwrites each worker's buffer to its disjoint
// COL_IDX / EDGE_IDS region. pwrite is thread-safe and the regions don't
// overlap, so byte-for-byte output equals the sequential path.
//
// QueryContext note: unlike the other parallel projection-build stages
// (which dispatched B+Tree iterators that decode leaf pages and consult
// the thread_local QueryContext::_query_ctx pointer), this builder reads
// raw mmap'd bytes directly via std::memcpy. No BPT leaf decode happens
// on the worker thread, so the QueryContext propagation pattern is not
// required here. The post-hoc BPT-iterator path in
// native_projection_builder.cc:build_one_topology_snapshot_ — used as
// a fallback when the integrated emission failed — remains sequential
// for that reason; parallelizing it would need those other stages'
// `parent_ctx` capture pattern.
void build_parallel(const std::filesystem::path& projection_dir,
                    TopologySnapshotWriter::Direction dir,
                    uint64_t num_nodes,
                    bool include_edge_ids,
                    const MappedFile& mm,
                    std::size_t num_partitions)
{
    const std::vector<LeafPageBoundary> boundaries = build_page_boundaries(mm);
    const std::size_t num_pages = boundaries.size();

    // Build K disjoint src ranges spanning [0, num_nodes). Uniform
    // partitioning of the dense node id space — node ObjectIds in a
    // projection are a contiguous [0, N) range after tag stripping, so
    // uniform splits balance roughly without a histogram prepass.
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.reserve(num_partitions);
    if (num_nodes == 0) {
        // No nodes means no edges — emit empty CSR via the sequential
        // path (which handles writer construction with zero-degrees).
        std::vector<uint64_t> degrees(num_nodes, 0);
        TopologySnapshotWriter writer(projection_dir, dir, num_nodes,
                                      std::move(degrees), include_edge_ids);
        writer.finalize();
        return;
    }
    {
        const uint64_t step =
            (num_nodes + num_partitions - 1) / num_partitions;
        uint64_t lo = 0;
        for (std::size_t p = 0; p < num_partitions; ++p) {
            const uint64_t hi = std::min<uint64_t>(lo + step, num_nodes);
            ranges.emplace_back(lo, hi);
            lo = hi;
            if (lo >= num_nodes) {
                // Remaining partitions get empty ranges; they no-op.
                for (std::size_t q = p + 1; q < num_partitions; ++q) {
                    ranges.emplace_back(num_nodes, num_nodes);
                }
                break;
            }
        }
    }

    // ---- Pass 1: per-worker degree increments to disjoint slot ranges ----
    std::vector<uint64_t> degrees(num_nodes, 0);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_partitions, 1),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t p = r.begin(); p < r.end(); ++p) {
                const uint64_t lo_src = ranges[p].first;
                const uint64_t hi_src = ranges[p].second;
                if (lo_src >= hi_src) continue;

                const std::size_t start_page =
                    find_first_page_for_src(boundaries, lo_src);
                if (start_page >= num_pages) continue;

                visit_leaf_records_in_range(mm, start_page, num_pages,
                    [&, hi_src](uint64_t k0,
                                uint64_t /*k1*/,
                                uint64_t /*k2*/) {
                        const uint64_t src_idx = k0 & ObjectId::VALUE_MASK;
                        if (src_idx >= hi_src) {
                            // Records past hi_src belong to the next
                            // worker; we cannot break the visitor mid-page
                            // cleanly without a control flow rewrite, so
                            // fall through and let the bounds check
                            // filter them out. The "stop early" win is
                            // approximated by the page-range start
                            // computed via find_first_page_for_src above.
                            return;
                        }
                        if (src_idx < lo_src) {
                            // Records before lo_src belong to the
                            // previous worker; same control-flow note.
                            return;
                        }
                        if (src_idx < num_nodes) {
                            ++degrees[src_idx];
                        }
                    });
            }
        });

    // ---- Build the writer (computes row_ptr) ----
    TopologySnapshotWriter writer(projection_dir, dir, num_nodes,
                                  std::move(degrees), include_edge_ids);

    const std::vector<uint64_t>& row_ptr = writer.row_ptr();

    // ---- Pass 2: per-worker dst/edge_id buffers, then pwrite via writer ----
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_partitions, 1),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t p = r.begin(); p < r.end(); ++p) {
                const uint64_t lo_src = ranges[p].first;
                const uint64_t hi_src = ranges[p].second;
                if (lo_src >= hi_src) continue;

                const uint64_t base = row_ptr[static_cast<std::size_t>(lo_src)];
                const uint64_t end  = row_ptr[static_cast<std::size_t>(hi_src)];
                const uint64_t expected = end - base;

                if (expected == 0) continue;

                std::vector<uint64_t> dst_buf;
                dst_buf.reserve(static_cast<std::size_t>(expected));
                std::vector<uint64_t> eid_buf;
                if (include_edge_ids) {
                    eid_buf.reserve(static_cast<std::size_t>(expected));
                }

                const std::size_t start_page =
                    find_first_page_for_src(boundaries, lo_src);
                if (start_page >= num_pages) continue;

                visit_leaf_records_in_range(mm, start_page, num_pages,
                    [&, lo_src, hi_src](uint64_t k0,
                                        uint64_t k1,
                                        uint64_t k2) {
                        const uint64_t src_idx = k0 & ObjectId::VALUE_MASK;
                        if (src_idx >= hi_src) return;
                        if (src_idx < lo_src) return;
                        if (src_idx >= num_nodes) return;
                        dst_buf.push_back(k1);
                        if (include_edge_ids) {
                            eid_buf.push_back(k2);
                        }
                    });

                // Defensive: dst_buf must equal `expected` for the
                // writer's append_subrange to accept the call. Mismatch
                // would indicate a partition-boundary bug; surface it as
                // an exception rather than letting the writer reject and
                // leave the .csr.tmp orphaned.
                if (dst_buf.size() != expected) {
                    throw std::runtime_error(
                        "topology_snapshot_from_leaf: parallel pass 2 "
                        "collected " + std::to_string(dst_buf.size())
                        + " edges in src range ["
                        + std::to_string(lo_src) + ", "
                        + std::to_string(hi_src) + ") but row_ptr expected "
                        + std::to_string(expected));
                }

                writer.append_subrange(lo_src, hi_src, dst_buf, eid_buf);
            }
        });

    writer.finalize();
}
#endif  // HAS_TBB

}  // namespace

void build_topology_snapshot_from_leaf(
    const std::filesystem::path&      projection_dir,
    TopologySnapshotWriter::Direction dir,
    uint64_t                          num_nodes,
    bool                              include_edge_ids)
{
    const std::filesystem::path leaf_path =
        projection_dir / source_basename_for(dir);

    // ---- mmap the .leaf file (both passes share the kernel page cache) ----
    // The file was just fsync'd + closed by BPTLeafWriter at the end of
    // build_index_streaming, so it is guaranteed present and complete.
    // An absent file is treated as a zero-edge projection (matches the
    // legacy post-hoc path's behavior when num_edges_ == 0).
    if (!std::filesystem::exists(leaf_path)) {
        std::vector<uint64_t> degrees(num_nodes, 0);
        TopologySnapshotWriter writer(projection_dir, dir, num_nodes,
                                      std::move(degrees), include_edge_ids);
        writer.finalize();
        return;
    }

    MappedFile mm = MappedFile::open_readonly(leaf_path);

    // Parallel-vs-sequential snapshot-build dispatch — opt-in via
    // MDB_PROJECTION_PARALLEL_SNAPSHOT (default true). Falls back to the
    // sequential path when:
    //   - the env var is set to 0/false/off
    //   - HAS_TBB is not defined at compile time
    //   - num_partitions < 2
    //   - num_nodes is 0 (handled inside build_parallel)
    //   - the .leaf is empty (mm.empty()) — the parallel and sequential
    //     paths produce the same all-zero CSR; sequential is simpler.
    bool parallel_enabled = resolve_parallel_snapshot_enabled();
    std::size_t num_partitions = resolve_snapshot_partition_count();

#ifndef HAS_TBB
    parallel_enabled = false;
#endif

    if (!parallel_enabled || num_partitions < 2 || mm.empty()) {
        build_sequential(projection_dir, dir, num_nodes,
                         include_edge_ids, mm);
        return;
    }

#ifdef HAS_TBB
    build_parallel(projection_dir, dir, num_nodes,
                   include_edge_ids, mm, num_partitions);
#else
    build_sequential(projection_dir, dir, num_nodes,
                     include_edge_ids, mm);
#endif
}

}  // namespace GQL::Projection
