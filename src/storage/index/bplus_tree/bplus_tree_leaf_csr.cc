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

    // --- Compute total_tuples_ (sum of degrees) by a single pass ---
    //
    // We decode each src entry's header (src_id varint + degree varint)
    // to accumulate total tuples. This also cross-checks that the entry
    // headers are well-formed varints at construction time, so the
    // BPTLeafBase contract methods can assume valid headers later.

    const uint8_t* const page_start = reinterpret_cast<const uint8_t*>(page_bytes_);
    const uint8_t* const page_end   = page_start + kPageSize;

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

        // Sanity: degree cannot exceed the number of uint64 that could fit
        // in a single page even if every varint were 1 byte. For a
        // non-chained entry this is conservative; chained hubs store
        // `degree` that covers the combined chain, which can exceed
        // single-page capacity — the cross-check below is thus only a
        // ceiling against gross corruption, not a tight bound.
        if (degree > static_cast<uint64_t>(UINT32_MAX)) {
            throw BPT::BPTLeafCSRDecodeException(
                "implausible degree " + std::to_string(degree)
                + " at src index " + std::to_string(i));
        }

        running_total += degree;
    }

    if (running_total > static_cast<uint64_t>(UINT32_MAX)) {
        throw BPT::BPTLeafCSRDecodeException(
            "total tuples on page exceeds uint32 range");
    }
    total_tuples_ = static_cast<uint32_t>(running_total);
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
    // Construction rejected continuation pages, so every successfully-
    // constructed instance is a chain head (possibly a trivial one-page
    // chain). Expose the bit explicitly so future T8.9 code that opens
    // a continuation via a lower-level path can share the accessor.
    return (header_.flags & BPT::CSRHybridFlags::kIsContinuation) == 0;
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
        // (a) Stay in the same src entry, advance within it.
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
                Record<N> rec{};
                if constexpr (N >= 1) rec[0] = seq_tuple_src_id_;
                if constexpr (N >= 2) rec[1] = dst_value;
                if constexpr (N >= 3) rec[2] = 0;
                return rec;
            }
            // get_dst_at failure falls through to the linear restart below.
        } else {
            // (b) Cross into the next src entry. Decode its header once.
            const uint_fast32_t next_entry = seq_tuple_entry_idx_ + 1;
            if (next_entry < header_.value_count) {
                const uint32_t off = offset_at_(next_entry);
                const uint8_t* in  = page_start + off;

                uint64_t src_id = 0;
                uint64_t degree = 0;
                bool hdr_ok = true;
                try {
                    in += BPT::varint_decode(in, page_end, src_id);
                    in += BPT::varint_decode(in, page_end, degree);
                } catch (...) {
                    hdr_ok = false;
                }

                if (hdr_ok && degree > 0) {
                    const uint32_t dst_start_off =
                        static_cast<uint32_t>(in - page_start);
                    uint64_t dst_value = 0;
                    // Fresh entry: the dst-level cache must not be reused
                    // across different start_offsets. get_dst_at with i=0
                    // against a new start_offset misses the cache and
                    // restarts from the first varint, which is the desired
                    // behavior.
                    if (get_dst_at(dst_start_off,
                                   static_cast<uint32_t>(degree),
                                   0,
                                   dst_value))
                    {
                        seq_tuple_entry_idx_ = next_entry;
                        seq_tuple_entry_cumulative_ =
                            seq_tuple_entry_cumulative_ + seq_tuple_entry_degree_;
                        seq_tuple_entry_degree_  = static_cast<uint32_t>(degree);
                        seq_tuple_dst_start_off_ = dst_start_off;
                        seq_tuple_src_id_        = src_id;
                        seq_tuple_within_idx_    = 0;
                        seq_tuple_pos_           = pos;

                        Record<N> rec{};
                        if constexpr (N >= 1) rec[0] = seq_tuple_src_id_;
                        if constexpr (N >= 2) rec[1] = dst_value;
                        if constexpr (N >= 3) rec[2] = 0;
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
    // fast-path failure. Walk the offset table accumulating degrees until
    // the entry containing `pos` is located; then decode its within-entry
    // dst. Populate the sequential cache so subsequent pos+1 calls take the
    // fast path above.

    uint_fast32_t cumulative = 0;
    for (uint_fast32_t i = 0; i < header_.value_count; ++i) {
        const uint32_t off = offset_at_(i);
        const uint8_t* in  = page_start + off;

        uint64_t src_id = 0;
        uint64_t degree = 0;
        in += BPT::varint_decode(in, page_end, src_id);
        in += BPT::varint_decode(in, page_end, degree);

        if (pos < cumulative + degree) {
            // The tuple falls in this entry. Compute its within-entry index.
            const uint_fast32_t within = pos - cumulative;

            const uint32_t dst_start_off = static_cast<uint32_t>(in - page_start);
            uint64_t dst_value = 0;
            if (!get_dst_at(dst_start_off, static_cast<uint32_t>(degree),
                            within, dst_value))
            {
                throw BPT::BPTLeafCSRDecodeException(
                    "get_dst_at failed inside decode_tuple_ at pos "
                    + std::to_string(pos));
            }

            // Populate the sequential cursor so the next call with pos+1
            // can advance in O(1) amortized.
            seq_tuple_pos_              = pos;
            seq_tuple_entry_idx_        = i;
            seq_tuple_within_idx_       = within;
            seq_tuple_entry_cumulative_ = cumulative;
            seq_tuple_entry_degree_     = static_cast<uint32_t>(degree);
            seq_tuple_dst_start_off_    = dst_start_off;
            seq_tuple_src_id_           = src_id;

            Record<N> rec{};
            if constexpr (N >= 1) rec[0] = src_id;
            if constexpr (N >= 2) rec[1] = dst_value;
            // For N >= 3 we do NOT reconstruct edge_id in this base
            // contract path. Edge_id recovery for v3 pages is the scope
            // of T8.5/T8.6 — the flags-bit-1 has_edge_ids encoding adds
            // a parallel varint stream that the writer emits; T8.4's
            // read surface focuses on (src, dst) adjacency lookup. The
            // BPTLeafBase path here fills edge_id with 0 and logs no
            // error; the field will be populated when the 3-way
            // dispatch and hub-aware iteration land in T8.6-T8.9.
            if constexpr (N >= 3) rec[2] = 0;
            return rec;
        }

        cumulative += static_cast<uint_fast32_t>(degree);
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
