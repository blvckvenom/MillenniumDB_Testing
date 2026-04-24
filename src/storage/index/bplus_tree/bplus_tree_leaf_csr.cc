// BPTLeafCSR read-path implementation (Spec #8 T8.4).
//
// The reader validates the v3 header at construction (per design §5.5),
// caches a pointer to the in-page offset table, and exposes:
//   - find_src_entry(src_id) — O(log value_count) binary search over the
//     offset table.
//   - get_dst_at(start_offset, degree, i) — O(1) amortized sequential
//     decode, backed by a cursor cache mirroring Spec #5 T5.13b.
//   - The BPTLeafBase<N> contract surface (get_record / search_index /
//     etc.) iterates the full (src, dst, edge_id) tuple stream for
//     BptIter range-scan compatibility.
//
// Mutation paths (insert / delete_record / update_to_next_leaf) raise
// std::logic_error per Spec #8 I6 — v3 pages are immutable post-build.
//
// Design reference: docs/superpowers/specs/2026-04-25-csr-hybrid-design.md

#include "storage/index/bplus_tree/bplus_tree_leaf_csr.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "storage/index/bplus_tree/bplus_tree_split.h"
#include "storage/index/bplus_tree/varint.h"
#include "storage/page/page.h"

namespace {

constexpr std::size_t kPageSize = Page::SIZE;         // 4096
constexpr std::size_t kHeaderBytes = sizeof(BPT::BPTLeafCSRHeader); // 16

// Upper bound on src entries per page. Each entry needs at minimum 2 bytes
// in the offset table plus 3 bytes in the entry body (1-byte src_id varint,
// 1-byte degree=0 varint, 1-byte sentinel — the true minimum is 2 bytes
// for a zero-degree entry, but we use 3 for defense-in-depth). The page
// budget minus the header must accommodate (offset_table + bodies).
constexpr std::size_t kMinEntryBodyBytes = 2;
constexpr std::size_t kMaxSrcsPerPage =
    (kPageSize - kHeaderBytes) / (2 /* offset slot */ + kMinEntryBodyBytes);

// Range of valid offsets that an offset_table entry may hold: the entry must
// start after the offset table itself and before the end of the page.
inline std::size_t payload_start_offset(uint32_t value_count) noexcept {
    return kHeaderBytes + 2 * static_cast<std::size_t>(value_count);
}

}  // namespace


// ============================================================================
// Constructor — validate header and offset table
// ============================================================================

template <std::size_t N>
BPTLeafCSR<N>::BPTLeafCSR(const char* page_bytes, ReadTag) :
    page_bytes_(page_bytes)
{
    if (page_bytes_ == nullptr) {
        throw BPT::BPTLeafCSRDecodeException(
            "BPTLeafCSR: null page_bytes passed to reader ctor");
    }

    // Deserialize the 16-byte header.
    uint8_t raw[kHeaderBytes];
    std::memcpy(raw, page_bytes_, sizeof(raw));
    header_ = BPT::deserialize_csr_header(raw);

    // --- Per-byte header validation (design §5.5) ---

    if (header_.format_version != 3) {
        throw BPT::BPTLeafCSRDecodeException(
            "invalid format_version at page offset 0 (expected 3, got "
            + std::to_string(header_.format_version) + ")");
    }
    if (header_.record_width != static_cast<uint8_t>(N)) {
        throw BPT::BPTLeafCSRDecodeException(
            "record_width mismatch at page offset 1 (expected "
            + std::to_string(N) + ", got "
            + std::to_string(header_.record_width) + ")");
    }
    // Reserved flag bits (bits 2..7) must be zero.
    if ((header_.flags & BPT::CSRHybridFlags::kReservedMask) != 0) {
        throw BPT::BPTLeafCSRDecodeException(
            "reserved flag bits non-zero at page offset 2 (flags=0x"
            + std::to_string(header_.flags) + ")");
    }
    // A continuation page cannot be opened directly as a BPTLeafCSR root;
    // continuations are reached by chain traversal from their chain head.
    if ((header_.flags & BPT::CSRHybridFlags::kIsContinuation) != 0) {
        throw BPT::BPTLeafCSRDecodeException(
            "cannot open continuation page as BPTLeafCSR root "
            "(flags bit 0 is set; reach via chain-head traversal)");
    }
    if (header_.reserved != 0) {
        throw BPT::BPTLeafCSRDecodeException(
            "non-zero reserved byte at offset 3 (got "
            + std::to_string(header_.reserved) + ")");
    }

    // Upper bound sanity on value_count. value_count == 0 is permitted —
    // corresponds to an empty projection whose .leaf files contain a
    // single "no src entries" page. Such pages still have a well-formed
    // 16-byte header.
    if (header_.value_count > kMaxSrcsPerPage) {
        throw BPT::BPTLeafCSRDecodeException(
            "value_count " + std::to_string(header_.value_count)
            + " exceeds per-page cap " + std::to_string(kMaxSrcsPerPage));
    }

    // --- Offset table bounds + monotonicity (design I8) ---

    offset_table_ = reinterpret_cast<const uint8_t*>(page_bytes_) + kHeaderBytes;

    const uint32_t vc = header_.value_count;
    if (vc > 0) {
        const std::size_t first_entry_start = payload_start_offset(vc);

        uint32_t prev_off = 0;
        for (uint_fast32_t i = 0; i < vc; ++i) {
            const uint32_t off = offset_at_(i);
            if (off < first_entry_start) {
                throw BPT::BPTLeafCSRDecodeException(
                    "offset_table[" + std::to_string(i) + "]=" + std::to_string(off)
                    + " is below the payload start "
                    + std::to_string(first_entry_start));
            }
            if (off >= kPageSize) {
                throw BPT::BPTLeafCSRDecodeException(
                    "offset_table[" + std::to_string(i) + "]=" + std::to_string(off)
                    + " is beyond the page end " + std::to_string(kPageSize));
            }
            if (i > 0 && off <= prev_off) {
                throw BPT::BPTLeafCSRDecodeException(
                    "offset_table non-monotonic at i=" + std::to_string(i)
                    + " (prev=" + std::to_string(prev_off)
                    + ", curr=" + std::to_string(off) + ")");
            }
            prev_off = off;
        }
    }

    // --- Compute physical tuple counts per entry (T8-B.1 Bug-A fix) ---
    //
    // Pre-fix, this loop accumulated stored `degree` values directly. But
    // hub entries' stored degree is the TOTAL chain size (design §3.9), and
    // only a subset of those dsts is physically serialized on the chain-head
    // page — the rest spills onto continuation pages. Summing stored degrees
    // inflated total_tuples_, causing BptIter to walk past the physical end
    // of the varint stream and crash in decode_tuple_.
    //
    // The fix: count the actual varints present on THIS page per entry by
    // decoding the col_idx stream until one of:
    //   (a) stored `degree` varints consumed (non-hub or chain-head whose
    //       dsts all fit on the page),
    //   (b) the next varint would start at or beyond the next entry's
    //       offset_table[i+1] (hit the neighbor boundary),
    //   (c) the next varint would start at or beyond kPageSize (last entry
    //       on page, hit page end).
    //
    // Also cache entry_col_idx_start_[i] so decode_tuple_ / find_src_entry
    // don't have to re-decode the (src_id, degree) header on every call.

    const uint8_t* const page_start = reinterpret_cast<const uint8_t*>(page_bytes_);
    const uint8_t* const page_end   = page_start + kPageSize;

    physical_degrees_.assign(vc, 0);
    entry_col_idx_start_.assign(vc, 0);
    entry_edge_id_start_.assign(vc, 0);

    // Spec #8-B task #1: cache whether this page carries a parallel
    // edge_id stream per entry. Checked once at construction so every
    // subsequent decode_tuple_ / get_dst_at call is branch-predictable.
    page_has_edge_ids_ =
        (header_.flags & BPT::CSRHybridFlags::kHasEdgeIds) != 0;

    uint64_t running_total = 0;
    for (uint_fast32_t i = 0; i < vc; ++i) {
        const uint32_t off = offset_at_(i);
        const uint8_t* in  = page_start + off;

        uint64_t src_id = 0;
        uint64_t degree = 0;
        try {
            in += BPT::varint_decode(in, page_end, src_id);
            in += BPT::varint_decode(in, page_end, degree);
        } catch (const BPT::BPTLeafV2DecodeException& e) {
            throw BPT::BPTLeafCSRDecodeException(
                std::string("malformed entry header at src index ")
                + std::to_string(i) + ": " + e.what());
        }
        (void)src_id; // decoded only for bounds validation

        if (degree > static_cast<uint64_t>(UINT32_MAX)) {
            throw BPT::BPTLeafCSRDecodeException(
                "implausible degree " + std::to_string(degree)
                + " at src index " + std::to_string(i));
        }

        entry_col_idx_start_[i] = static_cast<uint32_t>(in - page_start);

        // Boundary for this entry's col_idx stream: either the next entry's
        // offset_table slot, or the page end for the last entry.
        const uint8_t* boundary = page_end;
        if (i + 1 < vc) {
            const uint32_t next_off = offset_at_(i + 1);
            boundary = page_start + next_off;
        }

        // Walk varints until stored degree is exhausted OR next varint
        // would cross the boundary OR we hit the zero-padding sentinel
        // (see below). Physical count is min(stored, actual).
        //
        // Zero-padding detection for hub chain-heads:
        // the chain-head of a hub carries a stored `degree` that reflects
        // the TOTAL chain size (including dsts spilled onto continuation
        // pages); only `k_on_head <= degree` dsts are physically serialized
        // here and the tail is zero-padded. A zigzag-delta of 0 (encoded
        // as the single byte 0x00) means dst[k] == dst[k-1] — a duplicate
        // edge, which violates the per-src distinct-dst invariant enforced
        // by the GQL projection builder (see `native_projection_builder.cc`
        // dedup + sort). Therefore, seeing a 0x00 delta byte at position
        // k >= 1 marks the start of zero-padding and we stop counting.
        //
        // This heuristic does NOT apply to position 0 (full varint, not a
        // delta — the first dst can legitimately be 0 for node id 0), nor
        // to continuation pages (chunk_count is authoritative and
        // pre-decoded by the ContinuationTag ctor).
        uint64_t phys = 0;
        const uint8_t* cursor = in;
        while (phys < degree) {
            if (cursor >= boundary) {
                break;
            }
            // Zero-padding sentinel: a single 0x00 byte at position phys>=1
            // is a zigzag-delta=0, meaning a duplicate-dst that writer
            // output never produces.
            if (phys >= 1 && *cursor == 0x00) {
                break;
            }
            uint64_t v = 0;
            std::size_t consumed = 0;
            try {
                consumed = BPT::varint_decode(cursor, boundary, v);
            } catch (...) {
                // Next varint would span the boundary — stop here.
                break;
            }
            cursor += consumed;
            ++phys;
        }

        physical_degrees_[i] = static_cast<uint32_t>(phys);

        // Spec #8-B: if the page advertises edge_ids, the parallel eid
        // stream starts at the byte immediately after the dst stream's
        // last-consumed varint. Capture that offset now so decode_tuple_
        // can resolve eids in O(1) amortized alongside dsts.
        if (page_has_edge_ids_) {
            entry_edge_id_start_[i] =
                static_cast<uint32_t>(cursor - page_start);
        }

        running_total += phys;
    }

    if (running_total > static_cast<uint64_t>(UINT32_MAX)) {
        throw BPT::BPTLeafCSRDecodeException(
            "total tuples on page exceeds uint32 range");
    }
    total_tuples_ = static_cast<uint32_t>(running_total);
}


// ============================================================================
// Continuation-mode ctor (T8-B.1 Bug-B fix)
// ============================================================================
//
// Opens a v3 continuation page as a BPTLeafBase<N> view. Validates header,
// pre-decodes all `chunk_count` zigzag-delta varints starting from
// `tag.prev_dst_carry` into cont_dsts_, and sets total_tuples_ = chunk_count
// so BptIter can iterate the dsts via get_record / update_record.
//
// This is the ONLY legal path to open a continuation page. The ReadTag
// ctor still rejects continuation pages per I6 — directory-routed opens
// should never land on a continuation.

template <std::size_t N>
BPTLeafCSR<N>::BPTLeafCSR(const char* page_bytes, ContinuationTag tag) :
    mode_(Mode::Continuation),
    page_bytes_(page_bytes)
{
    if (page_bytes_ == nullptr) {
        throw BPT::BPTLeafCSRDecodeException(
            "BPTLeafCSR: null page_bytes passed to continuation ctor");
    }

    // Validate header bytes 0..3 before trusting anything else.
    const uint8_t* const page_start = reinterpret_cast<const uint8_t*>(page_bytes_);
    if (page_start[0] != 3) {
        throw BPT::BPTLeafCSRDecodeException(
            "continuation ctor: format_version != 3 (got "
            + std::to_string(page_start[0]) + ")");
    }
    if (page_start[1] != static_cast<uint8_t>(N)) {
        throw BPT::BPTLeafCSRDecodeException(
            "continuation ctor: record_width mismatch (expected "
            + std::to_string(N) + ", got "
            + std::to_string(page_start[1]) + ")");
    }
    if ((page_start[2] & BPT::CSRHybridFlags::kReservedMask) != 0) {
        throw BPT::BPTLeafCSRDecodeException(
            "continuation ctor: reserved flag bits non-zero");
    }
    if ((page_start[2] & BPT::CSRHybridFlags::kIsContinuation) == 0) {
        throw BPT::BPTLeafCSRDecodeException(
            "continuation ctor: kIsContinuation flag is clear "
            "(expected a continuation page)");
    }
    if (page_start[3] != 0) {
        throw BPT::BPTLeafCSRDecodeException(
            "continuation ctor: reserved byte at offset 3 non-zero");
    }

    // Deserialize header in the continuation view (same 16 bytes, but the
    // last uint32 has different semantics — chain_head_page_id). We route
    // it through the chain-head struct anyway for storage-compat reasons
    // (header_ is shared across modes); only fields actually used per-mode
    // are read.
    uint8_t raw[kHeaderBytes];
    std::memcpy(raw, page_bytes_, sizeof(raw));
    header_ = BPT::deserialize_csr_header(raw);

    cont_owning_src_id_ = tag.owning_src_id;

    // Pre-decode all chunk_count zigzag-delta varints starting from
    // prev_dst_carry.
    const uint32_t chunk_count = header_.value_count;  // chunk_count in this mode

    // Upper bound: a continuation payload is at most kPageSize - 16 bytes
    // at 1 byte per varint = 4080 varints. Refuse anything larger than that
    // as grossly corrupt.
    const uint32_t max_possible = static_cast<uint32_t>(kPageSize - kHeaderBytes);
    if (chunk_count > max_possible) {
        throw BPT::BPTLeafCSRDecodeException(
            "continuation chunk_count " + std::to_string(chunk_count)
            + " exceeds plausible bound " + std::to_string(max_possible));
    }

    cont_dsts_.reserve(chunk_count);

    const uint8_t* const page_end = page_start + kPageSize;
    const uint8_t* in             = page_start + kHeaderBytes;
    uint64_t running              = tag.prev_dst_carry;

    for (uint32_t i = 0; i < chunk_count; ++i) {
        uint64_t v = 0;
        try {
            in += BPT::varint_decode(in, page_end, v);
        } catch (const std::exception& e) {
            throw BPT::BPTLeafCSRDecodeException(
                std::string("continuation payload decode failure at i=")
                + std::to_string(i) + ": " + e.what());
        }
        const int64_t delta = BPT::zigzag_decode_u64(v);
        running += static_cast<uint64_t>(delta);
        cont_dsts_.push_back(running);
    }

    total_tuples_ = chunk_count;
}


// ============================================================================
// Helpers
// ============================================================================

template <std::size_t N>
uint32_t BPTLeafCSR<N>::offset_at_(uint_fast32_t i) const noexcept
{
    // Little-endian uint16 at offset_table_[2*i].
    const uint8_t* p = offset_table_ + 2 * i;
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8);
}

template <std::size_t N>
uint64_t BPTLeafCSR<N>::decode_src_id_at_(uint_fast32_t i) const noexcept
{
    const uint8_t* const page_start = reinterpret_cast<const uint8_t*>(page_bytes_);
    const uint8_t* const page_end   = page_start + kPageSize;
    const uint32_t off = offset_at_(i);

    uint64_t src_id = 0;
    try {
        BPT::varint_decode(page_start + off, page_end, src_id);
    } catch (...) {
        // Validated at construction; re-raising here would be a surprise
        // on the noexcept binary-search path. Return UINT64_MAX so the
        // comparison returns "greater than any real id" and the search
        // skews away — caller's find_src_entry handles no-match cleanly.
        return ~0ULL;
    }
    return src_id;
}

template <std::size_t N>
bool BPTLeafCSR<N>::binary_search_src_(uint64_t src_id,
                                       uint_fast32_t& out_index) const noexcept
{
    uint_fast32_t lo = 0;
    uint_fast32_t hi = header_.value_count;
    while (lo < hi) {
        const uint_fast32_t mid = lo + (hi - lo) / 2;
        const uint64_t mid_src = decode_src_id_at_(mid);
        if (mid_src == src_id) {
            out_index = mid;
            return true;
        }
        if (mid_src < src_id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return false;
}

template <std::size_t N>
void BPTLeafCSR<N>::invalidate_cache_() const noexcept
{
    cache_valid_     = false;
    cache_start_off_ = 0;
    cache_i_         = 0;
    cache_value_     = 0;
    cache_in_        = nullptr;
}


// ============================================================================
// BPTLeafBase<N> contract: metadata
// ============================================================================

template <std::size_t N>
uint32_t BPTLeafCSR<N>::get_value_count() const
{
    // Return the total tuple count (sum of degrees), not the number of
    // src entries — BptIter and range scans iterate over tuples.
    return total_tuples_;
}

template <std::size_t N>
bool BPTLeafCSR<N>::has_next() const
{
    return header_.next_leaf != 0;
}

template <std::size_t N>
uint32_t BPTLeafCSR<N>::next_leaf() const noexcept
{
    return header_.next_leaf;
}

template <std::size_t N>
uint32_t BPTLeafCSR<N>::src_entry_count() const noexcept
{
    return header_.value_count;
}

template <std::size_t N>
bool BPTLeafCSR<N>::is_chain_head() const noexcept
{
    // Continuation-mode instances (T8-B.1) report false; ReadTag-opened
    // instances report true.
    return mode_ == Mode::ChainHead;
}


// ============================================================================
// CSR-specific API
// ============================================================================

template <std::size_t N>
bool BPTLeafCSR<N>::find_src_entry(uint64_t src_id,
                                   uint32_t& out_start_offset,
                                   uint32_t& out_degree) const noexcept
{
    uint_fast32_t idx = 0;
    if (!binary_search_src_(src_id, idx)) {
        return false;
    }

    const uint8_t* const page_start = reinterpret_cast<const uint8_t*>(page_bytes_);
    const uint8_t* const page_end   = page_start + kPageSize;
    const uint32_t off              = offset_at_(idx);

    uint64_t sid_check = 0;
    uint64_t degree    = 0;
    const uint8_t* in = page_start + off;
    try {
        in += BPT::varint_decode(in, page_end, sid_check);
        in += BPT::varint_decode(in, page_end, degree);
    } catch (...) {
        return false;
    }
    (void)sid_check; // already validated equality via the binary search path

    out_start_offset = static_cast<uint32_t>(in - page_start);
    out_degree       = static_cast<uint32_t>(degree);

    // Probing a new entry invalidates the cache (a cursor belonging to a
    // previous src must not be reused by a get_dst_at() on this entry).
    if (!cache_valid_ || cache_start_off_ != out_start_offset) {
        invalidate_cache_();
    }
    return true;
}

template <std::size_t N>
bool BPTLeafCSR<N>::get_dst_at(uint32_t start_offset,
                               uint32_t degree,
                               uint_fast32_t i,
                               uint64_t& out_dst) const noexcept
{
    if (i >= degree) {
        return false;
    }

    const uint8_t* const page_start = reinterpret_cast<const uint8_t*>(page_bytes_);
    const uint8_t* const page_end   = page_start + kPageSize;

    // Fast path: the cache's last-decoded index is exactly one less than
    // the requested index, belonging to the same start_offset. Resume
    // from cache_in_.
    if (cache_valid_
        && cache_start_off_ == start_offset
        && i == cache_i_ + 1)
    {
        const uint8_t* in = cache_in_;
        uint64_t v = 0;
        try {
            in += BPT::varint_decode(in, page_end, v);
        } catch (...) {
            return false;
        }
        // Subsequent dsts are zigzag(delta) varints against the previous.
        const int64_t delta = BPT::zigzag_decode_u64(v);
        cache_value_     += static_cast<uint64_t>(delta);
        cache_i_          = i;
        cache_in_         = in;
        out_dst           = cache_value_;
        return true;
    }

    // Idempotent hit: same index as last time.
    if (cache_valid_
        && cache_start_off_ == start_offset
        && i == cache_i_)
    {
        out_dst = cache_value_;
        return true;
    }

    // Cache miss or non-sequential: restart from start_offset. First dst is
    // a FULL varint; subsequent dsts are zigzag(delta) varints.
    const uint8_t* in = page_start + start_offset;
    uint64_t running = 0;

    for (uint_fast32_t k = 0; k <= i; ++k) {
        uint64_t v = 0;
        try {
            in += BPT::varint_decode(in, page_end, v);
        } catch (...) {
            return false;
        }
        if (k == 0) {
            running = v;
        } else {
            const int64_t delta = BPT::zigzag_decode_u64(v);
            running += static_cast<uint64_t>(delta);
        }
    }

    cache_valid_     = true;
    cache_start_off_ = start_offset;
    cache_i_         = i;
    cache_value_     = running;
    cache_in_        = in;

    out_dst = running;
    return true;
}


// ============================================================================
// Spec #8-B task #1: parallel edge_id stream decoder
// ============================================================================
//
// Standalone O(i) walk over the eid varint chain (no cache). Used only
// from decode_tuple_'s fallback path; the sequential tuple cache above
// collapses successive eid lookups into back-to-back O(i) calls whose
// cost stays bounded by the degree of one src entry.

template <std::size_t N>
bool BPTLeafCSR<N>::get_eid_at(uint32_t eid_start_offset,
                               uint32_t degree,
                               uint_fast32_t i,
                               uint64_t& out_eid) const noexcept
{
    if (i >= degree) {
        return false;
    }

    const uint8_t* const page_start = reinterpret_cast<const uint8_t*>(page_bytes_);
    const uint8_t* const page_end   = page_start + kPageSize;

    const uint8_t* in = page_start + eid_start_offset;
    uint64_t running = 0;

    for (uint_fast32_t k = 0; k <= i; ++k) {
        uint64_t v = 0;
        try {
            in += BPT::varint_decode(in, page_end, v);
        } catch (...) {
            return false;
        }
        if (k == 0) {
            running = v;
        } else {
            const int64_t delta = BPT::zigzag_decode_u64(v);
            running += static_cast<uint64_t>(delta);
        }
    }

    out_eid = running;
    return true;
}

// ============================================================================
// BPTLeafBase<N> contract: tuple iteration
// ============================================================================
//
// For BptIter range scans, callers iterate `get_record(0..total_tuples_-1)`.
// We walk the page in offset-table order, advancing through each entry's
// col_idx stream. The cursor cache accelerates sequential access; random
// access restarts from the beginning of the entry containing `pos`.

template <std::size_t N>
Record<N> BPTLeafCSR<N>::decode_tuple_(uint_fast32_t pos) const
{
    if (pos >= total_tuples_) {
        throw std::out_of_range(
            "BPTLeafCSR::get_record position " + std::to_string(pos)
            + " >= total_tuples " + std::to_string(total_tuples_));
    }

    // Continuation mode: dsts are pre-decoded into cont_dsts_ during ctor,
    // all with the same owning src_id. Short-circuit ahead of the chain-head
    // offset-table walk.
    if (mode_ == Mode::Continuation) {
        Record<N> rec{};
        if constexpr (N >= 1) rec[0] = cont_owning_src_id_;
        if constexpr (N >= 2) rec[1] = cont_dsts_[pos];
        if constexpr (N >= 3) rec[2] = 0;
        return rec;
    }

    const uint8_t* const page_start = reinterpret_cast<const uint8_t*>(page_bytes_);
    const uint8_t* const page_end   = page_start + kPageSize;

    // ---- Fast path (T8.12b): sequential forward access ---------------------
    //
    // If the previous decode landed on pos = seq_tuple_pos_, and the caller
    // is now asking for pos + 1, we can advance in O(1) amortized without
    // walking the offset table from entry 0:
    //   (a) If within-entry room remains, stay in the same src entry and
    //       increment seq_tuple_within_idx_. The per-dst varint advance is
    //       serviced by the existing dst-level cache via get_dst_at.
    //   (b) If the current entry is exhausted, cross to the next src entry:
    //       bump seq_tuple_entry_idx_, decode its (src_id, degree) header
    //       exactly once, and emit its first dst.
    //
    // Correctness: the cache memoizes a derived state of the immutable v3
    // page; any path that reaches the same (entry_idx, within_idx) observes
    // the same tuple values.
    if (seq_tuple_pos_ != UINT_FAST32_MAX
        && pos == seq_tuple_pos_ + 1
        && seq_tuple_entry_idx_ < header_.value_count)
    {
        // (a) Stay in the same src entry, advance within it. Note that
        // seq_tuple_entry_degree_ is the PHYSICAL degree (T8-B.1 Bug-A fix),
        // not the stored degree — so this bound reflects actual varints
        // present on this page, not the hub-total stored in the entry header.
        if (seq_tuple_within_idx_ + 1 < seq_tuple_entry_degree_) {
            const uint_fast32_t next_within = seq_tuple_within_idx_ + 1;
            uint64_t dst_value = 0;
            if (get_dst_at(seq_tuple_dst_start_off_,
                           seq_tuple_entry_degree_,
                           next_within,
                           dst_value))
            {
                seq_tuple_within_idx_ = next_within;
                seq_tuple_pos_        = pos;
                // Spec #8-B: when the page advertises edge_ids, decode the
                // parallel eid stream at the same within-entry index so
                // downstream consumers (count(e), edge-id lookups) see
                // real values instead of the ADR 008 zero fallback.
                uint64_t eid_value = 0;
                if (page_has_edge_ids_) {
                    get_eid_at(
                        entry_edge_id_start_[seq_tuple_entry_idx_],
                        seq_tuple_entry_degree_,
                        next_within,
                        eid_value);
                }
                Record<N> rec{};
                if constexpr (N >= 1) rec[0] = seq_tuple_src_id_;
                if constexpr (N >= 2) rec[1] = dst_value;
                if constexpr (N >= 3) rec[2] = eid_value;
                return rec;
            }
            // get_dst_at failure falls through to the linear restart below.
        } else {
            // (b) Cross into the next src entry. Use cached start offset and
            // physical degree (T8-B.1 Bug-A fix) — no header re-decode needed.
            const uint_fast32_t next_entry = seq_tuple_entry_idx_ + 1;
            if (next_entry < header_.value_count) {
                const uint32_t off = offset_at_(next_entry);
                const uint8_t* in  = page_start + off;

                uint64_t src_id = 0;
                bool hdr_ok = true;
                try {
                    // Only src_id needs re-decode for the record; physical
                    // degree + dst stream start are pre-cached.
                    in += BPT::varint_decode(in, page_end, src_id);
                } catch (...) {
                    hdr_ok = false;
                }

                const uint32_t phys_deg = physical_degrees_[next_entry];
                if (hdr_ok && phys_deg > 0) {
                    const uint32_t dst_start_off =
                        entry_col_idx_start_[next_entry];
                    uint64_t dst_value = 0;
                    if (get_dst_at(dst_start_off,
                                   phys_deg,
                                   0,
                                   dst_value))
                    {
                        seq_tuple_entry_idx_ = next_entry;
                        seq_tuple_entry_cumulative_ =
                            seq_tuple_entry_cumulative_ + seq_tuple_entry_degree_;
                        seq_tuple_entry_degree_  = phys_deg;
                        seq_tuple_dst_start_off_ = dst_start_off;
                        seq_tuple_src_id_        = src_id;
                        seq_tuple_within_idx_    = 0;
                        seq_tuple_pos_           = pos;

                        // Spec #8-B: decode parallel eid for tuple 0 of
                        // the newly-entered entry.
                        uint64_t eid_value = 0;
                        if (page_has_edge_ids_) {
                            get_eid_at(
                                entry_edge_id_start_[next_entry],
                                phys_deg,
                                0,
                                eid_value);
                        }

                        Record<N> rec{};
                        if constexpr (N >= 1) rec[0] = seq_tuple_src_id_;
                        if constexpr (N >= 2) rec[1] = dst_value;
                        if constexpr (N >= 3) rec[2] = eid_value;
                        return rec;
                    }
                }
                // Any failure falls through to the linear restart below.
            }
            // next_entry out of range means total_tuples_ already covered —
            // but the top-of-function bound check guarantees pos <
            // total_tuples_, so this branch is unreachable under a
            // well-formed page. Fall through for safety.
        }
    }

    // ---- Fallback: linear walk over the offset table ----------------------
    //
    // Used for random access, backwards access, the first call, or any
    // fast-path failure. Walk physical degrees (T8-B.1 Bug-A fix) until
    // the entry containing `pos` is located; then decode its within-entry
    // dst. Populate the sequential cache so subsequent pos+1 calls take the
    // fast path above.

    uint_fast32_t cumulative = 0;
    for (uint_fast32_t i = 0; i < header_.value_count; ++i) {
        const uint32_t phys_deg = physical_degrees_[i];

        if (pos < cumulative + phys_deg) {
            // The tuple falls in this entry. Compute within-entry index.
            const uint_fast32_t within = pos - cumulative;

            // Decode only src_id (degree + stream start are pre-cached).
            const uint32_t off = offset_at_(i);
            const uint8_t* in  = page_start + off;
            uint64_t src_id = 0;
            in += BPT::varint_decode(in, page_end, src_id);

            const uint32_t dst_start_off = entry_col_idx_start_[i];
            uint64_t dst_value = 0;
            if (!get_dst_at(dst_start_off, phys_deg, within, dst_value))
            {
                throw BPT::BPTLeafCSRDecodeException(
                    "get_dst_at failed inside decode_tuple_ at pos "
                    + std::to_string(pos));
            }

            seq_tuple_pos_              = pos;
            seq_tuple_entry_idx_        = i;
            seq_tuple_within_idx_       = within;
            seq_tuple_entry_cumulative_ = cumulative;
            seq_tuple_entry_degree_     = phys_deg;
            seq_tuple_dst_start_off_    = dst_start_off;
            seq_tuple_src_id_           = src_id;

            // Spec #8-B: resolve parallel eid at the same within-entry
            // index. Skipped (left as zero-sentinel) on pages that do not
            // advertise edge_ids — preserves pre-Spec-#8-B behavior.
            uint64_t eid_value = 0;
            if (page_has_edge_ids_) {
                get_eid_at(entry_edge_id_start_[i],
                           phys_deg,
                           within,
                           eid_value);
            }

            Record<N> rec{};
            if constexpr (N >= 1) rec[0] = src_id;
            if constexpr (N >= 2) rec[1] = dst_value;
            if constexpr (N >= 3) rec[2] = eid_value;
            return rec;
        }

        cumulative += phys_deg;
    }

    // Unreachable: pos < total_tuples_ means some entry must contain it,
    // and total_tuples_ was validated at construction. If reached, the
    // on-disk state was tampered with between ctor and this call.
    throw BPT::BPTLeafCSRDecodeException(
        "decode_tuple_ internal error at pos " + std::to_string(pos));
}

template <std::size_t N>
Record<N> BPTLeafCSR<N>::get_record(uint_fast32_t pos) const
{
    return decode_tuple_(pos);
}

template <std::size_t N>
void BPTLeafCSR<N>::set_record(uint_fast32_t pos, Record<N>& out) const
{
    out = get_record(pos);
}

template <std::size_t N>
void BPTLeafCSR<N>::set_redundant_record(Record<N>& out) const
{
    // v3 has no redundant-bitset concept. Zero-initialize so the caller
    // (which typically treats this as a scratch buffer before set_record)
    // sees a defined state.
    for (std::size_t j = 0; j < N; ++j) {
        out[j] = 0;
    }
}

template <std::size_t N>
void BPTLeafCSR<N>::update_record(uint_fast32_t pos, Record<N>& out) const
{
    // CSR has no per-field redundancy; overwrite fully.
    out = get_record(pos);
}

template <std::size_t N>
uint_fast32_t BPTLeafCSR<N>::search_index(const Record<N>& target) const noexcept
{
    // Design §3.4 analogue: linear scan over the tuple sequence, return the
    // index of the first tuple >= target (lex). For N=3 edge indexes the
    // primary key is src; since src is stored once per entry (not per
    // tuple), we could short-circuit on src alone when target.get_key()
    // doesn't match. For simplicity and uniformity with the BPTLeafV2
    // semantics (noexcept; return value_count on malformed varints), we
    // reuse decode_tuple_ inside a try/catch guard.

    for (uint_fast32_t pos = 0; pos < total_tuples_; ++pos) {
        Record<N> r;
        try {
            r = decode_tuple_(pos);
        } catch (...) {
            return total_tuples_;
        }
        bool lt = false;
        for (std::size_t j = 0; j < N; ++j) {
            if (r[j] < target[j]) { lt = true; break; }
            if (r[j] > target[j]) { break; }
        }
        if (!lt) {
            return pos;
        }
    }
    return total_tuples_;
}

template <std::size_t N>
bool BPTLeafCSR<N>::check_range(const Record<N>& r) const
{
    if (total_tuples_ == 0) {
        return false;
    }
    const auto min = get_record(0);
    const auto max = get_record(total_tuples_ - 1);
    return min <= r && r <= max;
}


// ============================================================================
// Mutation: v3 pages are immutable (Spec #8 I6)
// ============================================================================

template <std::size_t N>
std::unique_ptr<BPlusTreeSplit<N>>
BPTLeafCSR<N>::insert(const Record<N>&, bool&)
{
    throw std::logic_error(
        "CSR_HYBRID leaves are immutable; rebuild projection");
}

template <std::size_t N>
bool BPTLeafCSR<N>::delete_record(const Record<N>&)
{
    throw std::logic_error(
        "CSR_HYBRID leaves are immutable; rebuild projection");
}

template <std::size_t N>
void BPTLeafCSR<N>::update_to_next_leaf()
{
    // Caller should construct a new BPTLeafCSR(page_bytes, ReadTag) on the
    // follow-up page rather than mutating this instance.
    throw std::logic_error(
        "CSR_HYBRID leaves are immutable; rebuild projection");
}


// ============================================================================
// Diagnostics
// ============================================================================

template <std::size_t N>
bool BPTLeafCSR<N>::check(std::ostream& os) const
{
    if (page_bytes_ == nullptr) {
        os << "  ERROR: BPTLeafCSR::check on nullptr page\n";
        return false;
    }

    if (total_tuples_ == 0) {
        os << "  WARNING: empty v3 leaf (value_count=" << header_.value_count
           << ", total_tuples=0).\n";
        return true;
    }

    Record<N> prev{};
    for (uint_fast32_t i = 0; i < total_tuples_; ++i) {
        Record<N> cur;
        try {
            cur = get_record(i);
        } catch (const std::exception& e) {
            os << "  ERROR: BPTLeafCSR decode failure at tuple "
               << i << ": " << e.what() << "\n";
            return false;
        }
        if (i > 0 && !(prev < cur || prev == cur)) {
            os << "  ERROR: bad tuple order at BPTLeafCSR at tuple " << i << "\n";
            return false;
        }
        prev = cur;
    }
    return true;
}

template <std::size_t N>
void BPTLeafCSR<N>::print(std::ostream& os) const
{
    os << "Printing v3 CSR Leaf (src_entries=" << header_.value_count
       << ", total_tuples=" << total_tuples_ << "):\n";
    for (uint_fast32_t pos = 0; pos < total_tuples_; ++pos) {
        Record<N> r;
        try {
            r = get_record(pos);
        } catch (const std::exception& e) {
            os << "  <decode error at tuple " << pos << ": " << e.what() << ">\n";
            return;
        }
        os << "  (";
        for (std::size_t j = 0; j < N; ++j) {
            if (j != 0) os << ", ";
            os << r[j];
        }
        os << ")\n";
    }
}


// Explicit template instantiations.
template class BPTLeafCSR<1>;
template class BPTLeafCSR<2>;
template class BPTLeafCSR<3>;
