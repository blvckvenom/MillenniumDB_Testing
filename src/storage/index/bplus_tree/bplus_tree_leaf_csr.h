#pragma once

// BPTLeafCSR — read-mode view over a v3 CSR-hybrid leaf page. Third
// subclass of BPTLeafBase<N>, alongside BPTLeafV1<N> (bitset) and
// BPTLeafV2<N> (delta+varint).
//
// Spec #8: each v3 leaf page holds MULTIPLE source nodes' adjacency
// lists, packed CSR-style. An in-page offset table (uint16 per source
// node entry) enables O(log srcs) lookup within a page; each entry
// contains (src_id varint, degree varint, col_idx[degree] list) where
// col_idx is delta+varint encoded (first dst full, subsequent zigzag
// deltas — composes with Spec #5 T5.4/T5.5 codec).
//
// Hub nodes whose adjacency exceeds 4 KB span multiple pages via the
// continuation-header variant (see bpt_leaf_csr_format.h): the
// chain-head page carries value_count > 0 src entries with degrees
// that describe the total (including bytes spilled onto continuation
// pages), and continuation pages are marked via flags bit 0 and point
// back via chain_head_page_id (stored in the repurposed
// min_src_id_low slot).
//
// This class is READ-ONLY. v3 pages are immutable post-build — the
// mutation methods on BPTLeafBase<N> all throw std::logic_error (I6).
//
// Design reference: docs/superpowers/specs/2026-04-25-csr-hybrid-design.md
//   §3.1 (D1 multi-source layout), §3.3 (D3 offset table),
//   §3.5 (D5 composition with varint), §3.10 (D10 read path),
//   §5 (on-disk format).

#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "storage/index/bplus_tree/bplus_tree_leaf_base.h"
#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
#include "storage/index/record.h"

// Forward declarations
template <std::size_t N> class BPlusTreeSplit;

template <std::size_t N>
class BPTLeafCSR : public BPTLeafBase<N> {
public:
    /// Reader-mode tag. Disambiguates from a future writer-mode ctor (T8.5).
    struct ReadTag {};

    /// Continuation-mode tag (Spec #8-B T8-B.1). Enables BptIter to open a
    /// hub continuation page as a BPTLeafBase<N> view during cross-page
    /// transitions, so BptIter can iterate the hub's remaining dsts that
    /// spilled onto continuations. The chain-head reader cannot do this —
    /// its ctor rejects continuations per I6.
    ///
    /// `owning_src_id` is the hub's src_id decoded from its chain-head entry;
    /// continuation pages do not store it.
    /// `prev_dst_carry` is the running dst cursor at the end of the previous
    /// chunk (chain-head's last dst, or the previous continuation's last dst).
    /// The first varint on this continuation decodes as a zigzag-delta
    /// against this carry per the writer's §5.4 convention.
    struct ContinuationTag {
        uint64_t owning_src_id;
        uint64_t prev_dst_carry;
    };

    /// Read-mode construction. Validates the v3 header at the start of
    /// `page_bytes` (design §5.5 + T8.3 checks) and caches metadata,
    /// including a pointer to the in-page uint16 offset table that
    /// begins immediately after the 16-byte header.
    ///
    /// Raises BPT::BPTLeafCSRDecodeException on any invariant violation:
    ///   - byte 0 (format_version) != 3
    ///   - byte 1 (record_width) != N
    ///   - flags byte has any reserved bit set (bits 2..7)
    ///   - flags byte has bit 0 set (opening a continuation page as root
    ///     is not supported; continuations are reached by chain traversal
    ///     through the chain head, not opened directly)
    ///   - byte 3 (reserved) != 0
    ///   - value_count out of range for a single 4 KB page
    ///   - offset table not monotonically increasing or out of page bounds
    BPTLeafCSR(const char* page_bytes, ReadTag);

    /// Continuation-mode construction (Spec #8-B T8-B.1). Validates the
    /// v3 continuation header (format_version=3, record_width=N,
    /// flags & kIsContinuation != 0, reserved=0), pre-decodes the chunk's
    /// zigzag-delta varint stream against `prev_dst_carry` into an internal
    /// buffer, and exposes them via the BPTLeafBase<N> contract so BptIter
    /// can iterate the hub's remaining dsts.
    ///
    /// Raises BPT::BPTLeafCSRDecodeException on header / payload corruption.
    BPTLeafCSR(const char* page_bytes, ContinuationTag tag);

    ~BPTLeafCSR() override = default;

    BPTLeafCSR(const BPTLeafCSR&)            = delete;
    BPTLeafCSR& operator=(const BPTLeafCSR&) = delete;
    BPTLeafCSR(BPTLeafCSR&&)                 = default;
    BPTLeafCSR& operator=(BPTLeafCSR&&)      = delete;

    // ======= BPTLeafBase<N> contract ========================================
    //
    // get_value_count() returns the TOTAL number of (src, dst, edge_id)
    // tuples on this page — i.e. the sum of degrees across all src entries
    // — NOT the number of src entries. This matches BptIter / range-scan
    // semantics for which a "record" is one triple.

    uint32_t      get_value_count() const override;
    bool          has_next() const override;
    Record<N>     get_record(uint_fast32_t pos) const override;
    void          set_record(uint_fast32_t pos, Record<N>& out) const override;
    void          set_redundant_record(Record<N>& out) const override;
    void          update_record(uint_fast32_t pos, Record<N>& out) const override;
    uint_fast32_t search_index(const Record<N>& record) const noexcept override;
    bool          check_range(const Record<N>& r) const override;

    std::unique_ptr<BPlusTreeSplit<N>> insert(const Record<N>& record, bool& error) override;
    bool delete_record(const Record<N>& record) override;
    void update_to_next_leaf() override;

    bool check(std::ostream& os) const override;
    void print(std::ostream& os) const override;

    // ======= CSR-specific API (not on BPTLeafBase<N>) =======================

    /// Look up the adjacency list for source node `src_id` within THIS page.
    /// Returns true if found, populating:
    ///   - `out_start_offset`: byte offset into the page where the col_idx
    ///     varint stream begins (i.e. immediately past the src_id and degree
    ///     varints at the start of the matched entry).
    ///   - `out_degree`: number of destinations in that entry.
    ///
    /// Returns false if `src_id` is not present on this page.
    ///
    /// O(log value_count) via binary search over the in-page offset table
    /// (design §3.3 R-A).
    bool find_src_entry(uint64_t src_id,
                        uint32_t& out_start_offset,
                        uint32_t& out_degree) const noexcept;

    /// Decode the i-th destination of an adjacency list starting at
    /// `start_offset` (the col_idx stream position returned by
    /// find_src_entry) with a known `degree`. `i` must be in [0, degree).
    ///
    /// Uses a sequential-access cache (mirroring the Spec #5 T5.13b cursor
    /// pattern) to amortize decoding cost. Non-sequential or cross-entry
    /// access restarts the cursor from `start_offset`.
    ///
    /// Returns true on success with `out_dst` populated; false if `i >= degree`.
    bool get_dst_at(uint32_t start_offset,
                    uint32_t degree,
                    uint_fast32_t i,
                    uint64_t& out_dst) const noexcept;

    /// Chain support: returns true if this is a chain-head page (always
    /// the case when the reader-mode ctor accepts the page, since
    /// continuation pages are rejected at construction per I6 —
    /// kept as an explicit accessor for future-proofing T8.9's
    /// TopologyAccessor shortcut).
    bool is_chain_head() const noexcept;

    /// Page id of the next leaf in the leaf chain, used by the reader of a
    /// hub's multi-page adjacency (T8.9) to follow continuation pages. A
    /// value of 0 means "this is the last leaf in the B+Tree".
    uint32_t next_leaf() const noexcept;

    /// Number of src entries on this page (NOT the total tuple count —
    /// that's get_value_count()).
    uint32_t src_entry_count() const noexcept;

private:
    // Payload starts at the offset table right after the 16-byte header.
    static constexpr std::size_t kHeaderBytes = sizeof(BPT::BPTLeafCSRHeader); // 16

    // Two distinct decode modes share one class. ChainHead is the original
    // reader opened by ReadTag — has offset table, multiple src entries.
    // Continuation (T8-B.1) is the hub-chunk reader opened by ContinuationTag
    // — no offset table, dsts pre-decoded into cont_dsts_, single owning_src_id.
    enum class Mode : uint8_t { ChainHead, Continuation };
    Mode                    mode_         = Mode::ChainHead;

    const char*             page_bytes_   = nullptr;
    BPT::BPTLeafCSRHeader   header_{};

    // Pointer to the in-page offset table (uint16 LE per src entry), just
    // past the 16-byte header. Length = header_.value_count. Stored as a
    // uint8_t* and decoded lazily via offset_at(i) so we don't rely on
    // alignment of the buffer (the buffer comes from BufferManager which
    // aligns to 4 KB, but unit tests may pass stack buffers).
    //
    // ChainHead mode only; null in Continuation mode.
    const uint8_t*          offset_table_ = nullptr;

    // Cached total tuple count: sum of PHYSICAL degrees across all src
    // entries on this page (post T8-B.1 Bug-A fix). For a chain-head page
    // carrying a hub whose on-disk `degree` header describes the TOTAL chain
    // size (§3.9), physical_degrees_[i] holds only the dsts actually
    // serialized on THIS page — varints past the entry boundary / page end
    // are not counted. For a non-hub entry physical_degrees_[i] == stored
    // degree.
    //
    // In Continuation mode, this equals chunk_count (the pre-decoded dst
    // count) and is used by get_value_count() uniformly.
    uint32_t                total_tuples_ = 0;

    // Per-src-entry physical tuple count (ChainHead mode only, length =
    // header_.value_count). For non-hub entries: equal to the entry's stored
    // `degree`. For hub chain-head entries where stored degree > physical
    // varints, this is the count of varints actually present on THIS page.
    std::vector<uint32_t>   physical_degrees_;

    // Byte offset within the page where each src entry's col_idx varint
    // stream begins (ChainHead mode only, length = header_.value_count).
    // Pre-computed at construction to avoid re-decoding the (src_id, degree)
    // header on every decode_tuple_ / find_src_entry call.
    std::vector<uint32_t>   entry_col_idx_start_;

    // Continuation mode state. Empty in ChainHead mode.
    uint64_t                cont_owning_src_id_ = 0;
    std::vector<uint64_t>   cont_dsts_;

    // Sequential-decode cache. When get_dst_at(start_offset, degree, i+1)
    // follows a get_dst_at(..., i) call with the same (start_offset, degree),
    // we resume from cache_dst_in_ using cache_dst_value_ as the running
    // accumulator. Non-sequential access restarts from start_offset.
    //
    // Mutable so const methods (get_dst_at / get_record / search_index) can
    // update the cache — purely a memoization of the immutable page state.
    //
    // cache_valid_ == false means "cache empty".
    mutable bool            cache_valid_     = false;
    mutable uint32_t        cache_start_off_ = 0;      // start_offset of the cached entry
    mutable uint_fast32_t   cache_i_         = 0;      // last decoded index within that entry
    mutable uint64_t        cache_value_     = 0;      // its decoded dst value
    mutable const uint8_t*  cache_in_        = nullptr; // byte ptr just past the last varint

    // Sequential tuple-cursor cache for decode_tuple_ / get_record (T8.12b).
    // Pre-fix, decode_tuple_(pos) walked the offset table from src entry 0
    // on every call, linearly accumulating degrees until it found the src
    // entry containing tuple `pos`. BptIter<N>::next() drives pos=0,1,...,
    // total_tuples_-1 sequentially, so without a cross-entry cursor the cost
    // was O(total_tuples_ * value_count) per leaf scan.
    //
    // This cache captures the (src_entry, within-entry) position of the
    // most recently decoded tuple. Calls with pos == seq_tuple_pos_ + 1
    // advance O(1) amortized: either the next within-entry dst (decoded via
    // the dst cache above), or — when crossing an entry boundary — the next
    // src entry decoded in O(1) via seq_tuple_next_entry_idx_. Random or
    // backwards access falls back to the linear walk.
    //
    // seq_tuple_pos_ == UINT_FAST32_MAX sentinel means "cache empty".
    mutable uint_fast32_t   seq_tuple_pos_              = UINT_FAST32_MAX;
    mutable uint_fast32_t   seq_tuple_entry_idx_        = 0;  // src entry index for this tuple
    mutable uint_fast32_t   seq_tuple_within_idx_       = 0;  // within-entry dst index
    mutable uint_fast32_t   seq_tuple_entry_cumulative_ = 0;  // sum of degrees BEFORE this entry
    mutable uint32_t        seq_tuple_entry_degree_     = 0;  // degree of this entry
    mutable uint32_t        seq_tuple_dst_start_off_    = 0;  // byte offset where col_idx stream starts
    mutable uint64_t        seq_tuple_src_id_           = 0;  // decoded src_id of this entry

    // ------- internal helpers -------

    /// Read offset_table[i] with explicit LE byte layout. i in [0, value_count).
    uint32_t offset_at_(uint_fast32_t i) const noexcept;

    /// Binary search over the offset table. Returns true and sets `out_index`
    /// to the index of the entry whose src_id equals `src_id`; else false.
    /// O(log value_count) varint decodes.
    bool binary_search_src_(uint64_t src_id,
                            uint_fast32_t& out_index) const noexcept;

    /// Decode the src_id stored at `offset_table_[i]`. Callers must have
    /// validated i < value_count and the offset itself.
    uint64_t decode_src_id_at_(uint_fast32_t i) const noexcept;

    /// Decode the raw tuple (src_id, dst, edge_id) at logical position `pos`
    /// (0 <= pos < total_tuples_). For the BPTLeafBase<N> contract.
    /// Used only when N == 3 for edge indexes; for N != 3 it returns
    /// a record with (src_id, dst, 0) (N==2) or (src_id,) (N==1) — but
    /// Spec #8 scope limits CSR_HYBRID to edge indexes (N=3), so those
    /// paths are defensive only.
    Record<N> decode_tuple_(uint_fast32_t pos) const;

    /// Invalidate the sequential cache. Called when a new entry is probed
    /// so stale cursor state doesn't serve a wrong dst.
    void invalidate_cache_() const noexcept;
};

extern template class BPTLeafCSR<1>;
extern template class BPTLeafCSR<2>;
extern template class BPTLeafCSR<3>;
