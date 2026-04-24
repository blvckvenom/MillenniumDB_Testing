#pragma once

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "misc/fatal_error.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"
#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/bplus_tree/varint.h"
#include "storage/index/record.h"
#include "storage/page/page.h"

template<std::size_t N>
class BPTLeafWriter {
public:
    static constexpr auto max_records = (Page::SIZE - 2 * sizeof(int32_t)) / (sizeof(uint64_t) * N);

    BPTLeafWriter(const std::string& filename)
    {
        file.open(filename, std::ios::out | std::ios::binary);
        buffer = new char[Page::SIZE];
    }

    ~BPTLeafWriter()
    {
        file.close();
        delete[] buffer;
    }

    void process_block(char* bytes, uint32_t size, std::bitset<N * 8> bitset, uint32_t next_block)
    {
        auto value_count = reinterpret_cast<uint32_t*>(buffer);
        auto next_leaf = reinterpret_cast<uint32_t*>(buffer + sizeof(uint32_t));

        memset(buffer, 0, Page::SIZE);
        *value_count = size;
        *next_leaf = next_block;
        std::memcpy(
            buffer + 2 * sizeof(uint32_t),
            bytes,
            N + bitset.count() + size * (sizeof(uint64_t) * N - bitset.count())
        );
        file.write(buffer, Page::SIZE);
    }

    void make_empty()
    {
        memset(buffer, 0, Page::SIZE);
        file.write(buffer, Page::SIZE);
    }

private:
    std::fstream file;

    char* buffer;
};

// ---------------------------------------------------------------------------
// BPTLeafV2Writer — bulk-load sibling of BPTLeafWriter for the delta +
// LEB128-varint (v2) leaf encoding. v2 pages replace the v1 redundant-byte
// bitset format with a compact representation: record 0 is stored as N full
// LEB128 varints; records 1..k-1 are stored as N zigzag-delta LEB128
// varints that exploit the sorted-record invariant to shrink high-bit values.
//
// Unlike BPTLeafWriter — which the caller drives page-at-a-time with a
// pre-packed byte buffer + redundant-byte bitset — BPTLeafV2Writer exposes
// a streaming record-at-a-time API. The writer buffers a single 4 KB page
// internally, drives a BPTLeafV2<N> to encode each record as zigzag-delta
// varints, and transparently flushes the current page + chains a new one
// whenever the next record would overflow the v2 budget.
//
// Rationale: the v2 payload size per record is variable (1..N*10 bytes) and
// depends on the magnitude of the inter-record delta. We cannot statically
// pre-partition the record stream into "one v1-sized block per page" and
// guarantee it fits, so the writer owns the overflow decision and signals
// page boundaries back to the caller via a boolean return from
// append_record. Callers use that signal to emit a matching directory entry
// for the record that started the new page (B+Tree dir convention: the first
// leaf does NOT get a dir entry; every subsequent first-record does).
//
// Records MUST arrive in non-decreasing lexicographic order — the delta
// encoding depends on the sorted invariant. Callers also handle dedup
// upstream (BPTLeafV2Writer does not dedup).
//
// Pages are fully zero-padded to Page::SIZE = 4096 before write, matching
// BPTLeafWriter's deterministic-padding behavior.
template<std::size_t N>
class BPTLeafV2Writer {
public:
    // Open the output file and initialize the first page.
    explicit BPTLeafV2Writer(const std::string& filename)
        : file_(filename, std::ios::out | std::ios::binary)
        , page_buffer_(std::make_unique<char[]>(Page::SIZE))
        , leaf_(nullptr)
        , pages_written_(0)
        , finalized_(false)
    {
        start_new_page_();
    }

    ~BPTLeafV2Writer()
    {
        // Defensive: flush any pending page if finalize() was not called
        // (e.g., when an exception unwinds past the caller). We do not
        // rethrow from a dtor.
        try {
            if (!finalized_) {
                finalize();
            }
        } catch (...) { /* swallow */ }
        file_.close();
    }

    BPTLeafV2Writer(const BPTLeafV2Writer&)            = delete;
    BPTLeafV2Writer& operator=(const BPTLeafV2Writer&) = delete;
    BPTLeafV2Writer(BPTLeafV2Writer&&)                 = delete;
    BPTLeafV2Writer& operator=(BPTLeafV2Writer&&)      = delete;

    // Append one record.
    //
    // Returns TRUE when this call triggered a page boundary — the just-
    // appended record is the first record of a new page. The caller should
    // emit a B+Tree directory entry for this record pointing at the page
    // index returned by current_page_index(), IF the previous page existed
    // (i.e., pages_written_ > 1 after the boundary crossing — see the v1
    // convention in BPTDirWriter usage: the first leaf never gets a dir
    // entry).
    //
    // Returns FALSE when the record fit in the current page (no boundary).
    //
    // Throws std::runtime_error if a single record cannot fit on an empty
    // page (pathological N * 10-byte encoding > 4080 budget — structurally
    // impossible for N in [1..3]).
    bool append_record(const Record<N>& rec)
    {
        if (finalized_) {
            throw std::runtime_error("BPTLeafV2Writer: append_record after finalize");
        }
        if (leaf_->append_record(rec)) {
            return false;  // fit in current page
        }

        // Overflow: flush current page with a forward link to the next
        // page, then start a new page and append the record there.
        //
        // Next-page number for the ABOUT-TO-FLUSH page is "current page
        // index + 1" in the normal (non-final) case — we are opening a
        // successor right after.
        const uint32_t next_page =
            static_cast<uint32_t>(pages_written_ + 1);
        flush_current_page_(next_page);
        start_new_page_();

        if (!leaf_->append_record(rec)) {
            // Structural guarantee: one record on an empty page CANNOT
            // overflow for N in [1..3] (N * VARINT_MAX_BYTES = 30 bytes
            // max << 4080-byte page budget). This path is defensive.
            throw std::runtime_error(
                "BPTLeafV2Writer: record too large for an empty v2 page "
                "(N=" + std::to_string(N) + ")");
        }
        return true;  // caller should emit a dir entry for this record
    }

    // Flush the final page (with next_leaf = 0) and mark the writer done.
    // Must be called exactly once after the last append_record. Safe to
    // call on an empty writer (no records) — in that case it produces a
    // single empty v2 leaf page, mirroring BPTLeafWriter::make_empty's
    // "always at least one leaf" invariant.
    void finalize()
    {
        if (finalized_) return;
        // Final page: next_leaf = 0 marks the end of the leaf chain.
        flush_current_page_(/*next_block=*/0);
        finalized_ = true;
    }

    // Emit a single empty v2 leaf page and close. Equivalent to finalize()
    // on a fresh writer, provided for API symmetry with
    // BPTLeafWriter::make_empty.
    void make_empty()
    {
        if (finalized_) return;
        finalize();
    }

    // Page index assigned to the page currently being filled (0-based).
    // Useful when the caller needs to emit a dir entry pointing at the
    // just-started page after append_record returns true.
    uint32_t current_page_index() const noexcept { return pages_written_; }

    // Total number of leaf pages written so far (post-flush count).
    uint32_t num_pages() const noexcept
    {
        return finalized_ ? pages_written_ : pages_written_;
    }

private:
    void start_new_page_()
    {
        std::memset(page_buffer_.get(), 0, Page::SIZE);
        // next_leaf is patched at flush time; start with 0, overwritten by
        // flush_current_page_ before write.
        leaf_ = std::make_unique<BPTLeafV2<N>>(page_buffer_.get(),
                                               /*next_leaf=*/ 0);
    }

    void flush_current_page_(uint32_t next_block)
    {
        // The BPTLeafV2<N> writer was constructed with its next_leaf at 0.
        // We cannot reach that field except by rebuilding the header bytes
        // after flush(), so let BPTLeafV2::flush() emit its 16-byte header
        // with next_leaf=0, then overwrite bytes 8..11 with the correct
        // value. Layout (from bpt_leaf_format.h):
        //   offset 0..3  : format_version | record_width | flags | reserved
        //   offset 4..7  : value_count   (uint32 LE)
        //   offset 8..11 : next_leaf     (uint32 LE)
        //   offset 12..15: reserved2     (uint32 LE = 0)
        leaf_->flush();
        const uint32_t nb_le = next_block;  // x86_64 is LE; matches serialize_header.
        std::memcpy(page_buffer_.get() + 8, &nb_le, sizeof(uint32_t));
        file_.write(page_buffer_.get(), Page::SIZE);
        ++pages_written_;
    }

    std::fstream                     file_;
    std::unique_ptr<char[]>          page_buffer_;
    std::unique_ptr<BPTLeafV2<N>>    leaf_;
    uint32_t                         pages_written_;
    bool                             finalized_;
};

// ---------------------------------------------------------------------------
// BPTLeafCSRWriter — bulk-load sibling of BPTLeafWriter (v1 bitset) and
// BPTLeafV2Writer (v2 delta+varint). Emits CSR-hybrid (v3) leaf pages in
// which the edge-index B+Tree leaves ARE the CSR layout: each page stores
// sorted (src, dst) adjacency data directly in a compressed CSR encoding
// so GNN samplers can perform O(1) neighbor slice reads rather than
// O(log N) B+Tree directory walks.
//
// Consumption contract: the caller streams sorted (src, dst, edge_id) tuples
// one at a time via append(Record<N>). Records MUST arrive in non-decreasing
// lexicographic order on the first field (src) — this matches the output of
// sorter_dispatch (classic/radix). Violation of the order invariant is UB.
//
// Output format:
//   - Chain-head page: 16-byte BPTLeafCSRHeader with format_version=3,
//     flags=0, value_count=number of src entries; uint16 offset_table of
//     value_count slots; packed src entries [varint(src), varint(degree),
//     varint(dst[0]), zigzag-varint(delta[i]) for i>=1]; zero padding
//     to Page::SIZE.
//   - Continuation page (for hubs whose entry exceeds one page): 16-byte
//     header with flags bit 0 (kIsContinuation) set, chunk_count=number of
//     raw dst varints in this chunk, chain_head_page_id back-pointer;
//     continuation payload is raw varints only — the first dst of the chunk
//     is zigzag-varint(delta) against the last dst of the previous chunk
//     (the running cursor carries across chain pages).
//
// Hub overflow handling: when a single src's entry would exceed the
// per-page budget, the writer packs as many dsts as fit into the chain-head
// page, then emits 1+ continuation pages for the remainder. The chain-head's
// next_leaf field points to the first continuation; each continuation's
// next_leaf points to the next in chain, or 0 if last. The chain-head's
// degree field always carries the total adjacency size (including the
// portion spilled onto continuations). The chain-head's value_count remains
// 1 (just the hub src entry on that page).
//
// Edge-id encoding: not emitted in this writer by default (flags bit 1
// always 0 when emit_edge_ids is false). When the reader encounters a v3
// page without the kHasEdgeIds flag, it returns edge_id=0 for every tuple.
// The parallel-stream edge-id encoding (persisting a second zigzag-varint
// stream alongside dsts) is opt-in via the emit_edge_ids constructor
// argument; see the emit_edge_ids_ member comment for details. This keeps
// the basic writer focused on the structural novelty (CSR layout + hub
// chaining) without requiring callers that only need neighbor access to
// also carry edge_ids.
//
// next_leaf chain: continuation pages chain via next_leaf just like regular
// leaf pages. When a chain-head's hub run finishes, its LAST continuation's
// next_leaf points forward to the next chain-head page (the src following
// the hub) — the intent is that a single leaf chain walks the entire
// B+Tree in src-ascending order, whether or not individual srcs are hubs.
template<std::size_t N>
class BPTLeafCSRWriter {
public:
    static_assert(N >= 2, "BPTLeafCSRWriter requires Record<N>.src + dst at minimum");

    // `emit_edge_ids` controls whether the v3 leaf page writes a parallel
    // edge_id varint stream alongside the dst stream. When true and N == 3,
    // each entry body layout becomes:
    //     varint(src_id) | varint(degree) | dst_varint_chain | eid_varint_chain
    // (dst_varint_chain and eid_varint_chain each hold `degree` varints in
    // the same DELTA-zigzag format used for dsts); the header flag
    // `kHasEdgeIds` is set per page. Persisting edge_ids in the v3 pages
    // allows count(e) queries over CSR-hybrid projections to match the B+Tree
    // result instead of returning zero. When false (default) the writer
    // preserves the dst-stream-only layout — eid=0 at read time — which
    // keeps every legacy projection byte-identical.
    explicit BPTLeafCSRWriter(const std::string& filename,
                              bool emit_edge_ids = false)
        : file_(filename, std::ios::out | std::ios::in | std::ios::binary | std::ios::trunc)
        , buffer_(new char[Page::SIZE])
        , pages_written_(0)
        , finalized_(false)
        , current_page_entry_count_(0)
        , current_page_bytes_used_(0)
        , staging_has_(false)
        , staging_src_(0)
        , pending_cont_patch_page_(UINT32_MAX)
        , emit_edge_ids_(emit_edge_ids && N >= 3)
    {
        if (!file_.is_open()) {
            // Retry without the `in` flag — some std::fstream impls refuse
            // to truncate-open in r/w mode if the file does not pre-exist.
            file_.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        }
        reset_current_page_();
    }

    ~BPTLeafCSRWriter()
    {
        try {
            if (!finalized_) {
                flush_finalize();
            }
        } catch (...) { /* dtor must not throw */ }
        if (file_.is_open()) {
            file_.close();
        }
        delete[] buffer_;
    }

    BPTLeafCSRWriter(const BPTLeafCSRWriter&)            = delete;
    BPTLeafCSRWriter& operator=(const BPTLeafCSRWriter&) = delete;
    BPTLeafCSRWriter(BPTLeafCSRWriter&&)                 = delete;
    BPTLeafCSRWriter& operator=(BPTLeafCSRWriter&&)      = delete;

    // Append one sorted record. Groups by record[0] (src). The dst field is
    // record[1]. N>=3 fields beyond dst (edge_id) are not encoded in the
    // current v3 payload — see class comment.
    //
    // Returns true on success; false only if the underlying file I/O failed.
    // Throws nothing directly (I/O errors surface via file_'s state which
    // flush_finalize() propagates).
    bool append(const std::array<uint64_t, N>& record) noexcept
    {
        const uint64_t src = record[0];
        const uint64_t dst = record[1];
        // Third field (edge_id) only present when N >= 3; reading it at
        // compile-time-index 2 is guarded by the static_assert above, but
        // we only consult it when emit_edge_ids_ is true (which itself
        // requires N >= 3 by ctor guard). Using `if constexpr (N >= 3)`
        // keeps the template valid for N == 2 pure edge lists.
        uint64_t eid = 0;
        if constexpr (N >= 3) {
            eid = record[2];
        }

        if (!staging_has_) {
            staging_src_  = src;
            staging_has_  = true;
            staging_dsts_.clear();
            staging_eids_.clear();
            staging_dsts_.push_back(dst);
            if (emit_edge_ids_) {
                staging_eids_.push_back(eid);
            }
            return true;
        }
        if (src == staging_src_) {
            staging_dsts_.push_back(dst);
            if (emit_edge_ids_) {
                staging_eids_.push_back(eid);
            }
            return true;
        }
        // New src: emit the previously-buffered src, then start staging.
        emit_current_src_to_page_();
        staging_src_  = src;
        staging_dsts_.clear();
        staging_eids_.clear();
        staging_dsts_.push_back(dst);
        if (emit_edge_ids_) {
            staging_eids_.push_back(eid);
        }
        return true;
    }

    // Emit any partially-staged src and the current page. Idempotent.
    void flush_finalize() noexcept
    {
        if (finalized_) return;
        if (staging_has_) {
            emit_current_src_to_page_();
            staging_has_ = false;
            staging_dsts_.clear();
            staging_eids_.clear();
        }
        if (current_page_entry_count_ > 0) {
            // Final chain-head page has no successor. When edge_ids are
            // in play, advertise them via the per-page flag so the reader
            // knows each entry carries a parallel eid stream.
            const uint8_t final_flags = emit_edge_ids_
                ? BPT::CSRHybridFlags::kHasEdgeIds
                : 0u;
            finalize_current_page_(/*next_leaf=*/0, final_flags,
                                   /*value_or_chunk_count=*/current_page_entry_count_,
                                   /*min_src_or_head=*/
                                       static_cast<uint32_t>(
                                           current_page_entries_.empty() ? 0u
                                           : (current_page_entries_.front() & 0xFFFFFFFFu)));
        }
        finalized_ = true;
    }

    // Emit one zero-initialised page with a valid v3 chain-head header. Use
    // when the B+Tree must have at least one empty leaf (projection with zero
    // edges).
    void make_empty() noexcept
    {
        if (finalized_) return;
        std::memset(buffer_, 0, Page::SIZE);
        BPT::BPTLeafCSRHeader h{};
        h.format_version = 3;
        h.record_width   = static_cast<uint8_t>(N);
        h.flags          = 0;
        h.reserved       = 0;
        h.value_count    = 0;
        h.next_leaf      = 0;
        h.min_src_id_low = 0;
        uint8_t raw[16];
        BPT::serialize_csr_header(h, raw);
        std::memcpy(buffer_, raw, 16);
        file_.write(buffer_, Page::SIZE);
        ++pages_written_;
        finalized_ = true;
    }

    uint32_t pages_written() const noexcept { return pages_written_; }

private:
    // Reset per-page bookkeeping to "starting a fresh chain-head page".
    void reset_current_page_() noexcept
    {
        std::memset(buffer_, 0, Page::SIZE);
        current_page_entry_count_ = 0;
        current_page_bytes_used_  = 0;
        current_page_offsets_.clear();
        current_page_entries_.clear();
        current_page_entry_bodies_.clear();
    }

    // Pre-size a src entry's encoded byte length, given its src_id and the
    // full buffered dst list (plus an optional parallel edge_id list when
    // emit_eids is true). Used to decide whether the entry fits on the
    // current page or needs to trigger a page flush / hub chain.
    //
    // When `emit_eids` is true, `eids` must be the same length as `dsts`
    // and is encoded using the same DELTA-zigzag convention (first entry
    // full varint, subsequent entries zigzag(delta) against the previous
    // eid). The total includes BOTH streams so the chain-head packing
    // decision reflects the true on-page cost.
    static std::size_t estimate_entry_bytes_(uint64_t src_id,
                                             const std::vector<uint64_t>& dsts,
                                             const std::vector<uint64_t>& eids,
                                             bool emit_eids) noexcept
    {
        std::size_t total = BPT::varint_size(src_id)
                          + BPT::varint_size(static_cast<uint64_t>(dsts.size()));
        if (!dsts.empty()) {
            total += BPT::varint_size(dsts[0]);
            uint64_t prev = dsts[0];
            for (std::size_t i = 1; i < dsts.size(); ++i) {
                const uint64_t cur = dsts[i];
                // Caller contract: dsts non-decreasing OR at least signed
                // delta encodable via zigzag. Use unsigned subtraction to
                // obtain the two's-complement int64 delta, matching the
                // reader's sequential cursor-cache resume formula (the reader
                // accumulates deltas from the first dst of each page via the
                // same zigzag-varint scheme).
                const uint64_t delta_u = cur - prev;
                const int64_t  delta_i = static_cast<int64_t>(delta_u);
                total += BPT::varint_size(BPT::zigzag_encode_i64(delta_i));
                prev = cur;
            }
        }
        if (emit_eids && !eids.empty()) {
            total += BPT::varint_size(eids[0]);
            uint64_t prev = eids[0];
            for (std::size_t i = 1; i < eids.size(); ++i) {
                const uint64_t cur = eids[i];
                const uint64_t delta_u = cur - prev;
                const int64_t  delta_i = static_cast<int64_t>(delta_u);
                total += BPT::varint_size(BPT::zigzag_encode_i64(delta_i));
                prev = cur;
            }
        }
        return total;
    }

    // Serialize a src entry into an out-buffer. Returns bytes written.
    // Mirrors estimate_entry_bytes_ byte-for-byte.
    //
    // When `emit_eids` is true and `eids` is non-empty, a second DELTA-
    // zigzag varint stream is appended immediately after the dst stream.
    // The two streams share the same degree (entry-local count) read from
    // the entry's degree header field; no intermediate length prefix is
    // needed because the reader knows both streams have exactly `degree`
    // varints apiece.
    static std::size_t encode_entry_(uint64_t src_id,
                                     const std::vector<uint64_t>& dsts,
                                     const std::vector<uint64_t>& eids,
                                     bool emit_eids,
                                     uint8_t* out) noexcept
    {
        std::size_t off = 0;
        off += BPT::varint_encode(src_id, out + off, BPT::VARINT_MAX_BYTES);
        off += BPT::varint_encode(static_cast<uint64_t>(dsts.size()),
                                  out + off, BPT::VARINT_MAX_BYTES);
        if (!dsts.empty()) {
            off += BPT::varint_encode(dsts[0], out + off, BPT::VARINT_MAX_BYTES);
            uint64_t prev = dsts[0];
            for (std::size_t i = 1; i < dsts.size(); ++i) {
                const uint64_t cur = dsts[i];
                const uint64_t delta_u = cur - prev;
                const int64_t  delta_i = static_cast<int64_t>(delta_u);
                off += BPT::varint_encode(BPT::zigzag_encode_i64(delta_i),
                                          out + off, BPT::VARINT_MAX_BYTES);
                prev = cur;
            }
        }
        if (emit_eids && !eids.empty()) {
            off += BPT::varint_encode(eids[0], out + off, BPT::VARINT_MAX_BYTES);
            uint64_t prev = eids[0];
            for (std::size_t i = 1; i < eids.size(); ++i) {
                const uint64_t cur = eids[i];
                const uint64_t delta_u = cur - prev;
                const int64_t  delta_i = static_cast<int64_t>(delta_u);
                off += BPT::varint_encode(BPT::zigzag_encode_i64(delta_i),
                                          out + off, BPT::VARINT_MAX_BYTES);
                prev = cur;
            }
        }
        return off;
    }

    // Byte budget for packing src entry bodies on a chain-head page. Subtract
    // the 16-byte header and the current offset-table footprint (2 bytes per
    // entry already in the page). Appending a new entry costs an extra 2
    // bytes in the offset table.
    std::size_t remaining_body_budget_with_new_entry_() const noexcept
    {
        const std::size_t header_bytes = 16;
        const std::size_t offset_table_after = 2 * (current_page_entry_count_ + 1);
        const std::size_t used_body = current_page_bytes_used_;
        const std::size_t total_used = header_bytes + offset_table_after + used_body;
        if (total_used >= Page::SIZE) return 0;
        return Page::SIZE - total_used;
    }

    // Patch the last-emitted continuation's next_leaf to point at the page
    // about to be written. emit_hub_continuation_ writes the tail
    // continuation with next_leaf=0 because it does not yet know whether a
    // subsequent src will open a new chain-head page. This helper, called
    // immediately before every new page write that follows a hub chain,
    // resolves the forward pointer by seeking back to the continuation's
    // next_leaf field (bytes 8..11) and overwriting it with the target page
    // number, so the leaf chain stays walkable across hub boundaries.
    void patch_pending_continuation_next_leaf_(uint32_t target_page_num) noexcept
    {
        if (pending_cont_patch_page_ == UINT32_MAX) return;

        const std::streamoff patch_byte_off =
            static_cast<std::streamoff>(pending_cont_patch_page_)
          * static_cast<std::streamoff>(Page::SIZE)
          + 8;  // next_leaf is at bytes 8..11
        const std::streamoff resume_pos =
            static_cast<std::streamoff>(pages_written_)
          * static_cast<std::streamoff>(Page::SIZE);

        char patch_bytes[4];
        patch_bytes[0] = static_cast<char>( target_page_num        & 0xFFu);
        patch_bytes[1] = static_cast<char>((target_page_num >>  8) & 0xFFu);
        patch_bytes[2] = static_cast<char>((target_page_num >> 16) & 0xFFu);
        patch_bytes[3] = static_cast<char>((target_page_num >> 24) & 0xFFu);
        file_.seekp(patch_byte_off, std::ios::beg);
        file_.write(patch_bytes, 4);
        file_.seekp(resume_pos, std::ios::beg);

        pending_cont_patch_page_ = UINT32_MAX;
    }

    // Write the current buffered page (with the given header fields) to
    // disk. After writing, resets per-page state for the next chain-head
    // page. Does NOT touch staging_ fields.
    void finalize_current_page_(uint32_t next_leaf,
                                uint8_t  flags,
                                uint32_t value_or_chunk_count,
                                uint32_t min_src_or_head) noexcept
    {
        // Before writing this page, patch any dangling continuation next_leaf
        // so the leaf chain walks into this page. pages_written_ is the
        // page number about to be assigned.
        patch_pending_continuation_next_leaf_(pages_written_);

        // Assemble header.
        BPT::BPTLeafCSRHeader h{};
        h.format_version = 3;
        h.record_width   = static_cast<uint8_t>(N);
        h.flags          = flags;
        h.reserved       = 0;
        h.value_count    = value_or_chunk_count;
        h.next_leaf      = next_leaf;
        h.min_src_id_low = min_src_or_head;

        uint8_t raw[16];
        BPT::serialize_csr_header(h, raw);
        std::memcpy(buffer_, raw, 16);

        // Chain-head: write offset_table then entry bodies. Continuation
        // pages route through a different path (emit_hub_continuation_) so
        // here we assume flags bit 0 == 0.
        if ((flags & BPT::CSRHybridFlags::kIsContinuation) == 0) {
            const std::size_t payload_start = 16 + 2 * static_cast<std::size_t>(value_or_chunk_count);
            // offset table
            for (std::size_t i = 0; i < current_page_offsets_.size(); ++i) {
                const uint16_t o = current_page_offsets_[i];
                buffer_[16 + 2 * i]     = static_cast<char>(o & 0xFF);
                buffer_[16 + 2 * i + 1] = static_cast<char>((o >> 8) & 0xFF);
            }
            // entry bodies
            std::size_t cursor = payload_start;
            for (std::size_t i = 0; i < current_page_entry_bodies_.size(); ++i) {
                const auto& body = current_page_entry_bodies_[i];
                std::memcpy(buffer_ + cursor, body.data(), body.size());
                cursor += body.size();
            }
        }

        file_.write(buffer_, Page::SIZE);
        ++pages_written_;

        reset_current_page_();
    }

    // Emit one continuation page for a hub's overflow. Returns the page
    // number that was written (0-based, pre-increment semantics: i.e. the
    // page number OF this continuation is `pages_written_` before this call
    // runs, equivalently `pages_written_ - 1` after).
    //
    // `dsts_slice` is the portion of the hub's dsts to serialize on this
    // chunk. The FIRST dst in the slice is encoded as zigzag(delta against
    // prev_dst_carry) — the running cursor crosses chain boundaries.
    // Subsequent dsts within the slice are zigzag(delta) against
    // the previous dst in the same slice.
    //
    // `next_leaf` is the forward pointer in this continuation's header.
    // `chain_head_page_id` is the back-pointer.
    // `prev_dst_carry` is the dst value at the end of the previous chunk
    // (so the first dst in this chunk encodes as zigzag-delta against it).
    //
    // When `eids_slice` is non-null and `eids_count` equals `dsts_count`,
    // a parallel edge_id varint stream is appended after the dst stream.
    // Each eid is encoded as zigzag-delta against the previous eid, with
    // `prev_eid_carry` as the seed value so cross-chunk decoding is uniform
    // with the chain-head convention. The writer sets the kHasEdgeIds flag
    // on the page when eids are present, enabling count(e) correctness over
    // hub-bearing CSR-hybrid projections.
    void emit_hub_continuation_(const uint64_t* dsts_slice,
                                std::size_t     dsts_count,
                                uint32_t        next_leaf,
                                uint32_t        chain_head_page_id,
                                uint64_t        prev_dst_carry,
                                const uint64_t* eids_slice = nullptr,
                                std::size_t     eids_count = 0,
                                uint64_t        prev_eid_carry = 0) noexcept
    {
        // A continuation page that is itself NOT the first continuation in
        // its own chain may need to patch a prior-chain dangling next_leaf
        // (e.g. back-to-back hubs). The forward pointer we're writing here
        // is authoritative; patch any earlier dangling ptr to point at us.
        patch_pending_continuation_next_leaf_(pages_written_);

        std::memset(buffer_, 0, Page::SIZE);

        const bool has_eids =
            (eids_slice != nullptr) && (eids_count == dsts_count);

        // Header as continuation variant.
        BPT::BPTLeafCSRContinuationHeader h{};
        h.format_version     = 3;
        h.record_width       = static_cast<uint8_t>(N);
        h.flags              = static_cast<uint8_t>(
            BPT::CSRHybridFlags::kIsContinuation
          | (has_eids ? BPT::CSRHybridFlags::kHasEdgeIds : 0u));
        h.reserved           = 0;
        h.chunk_count        = static_cast<uint32_t>(dsts_count);
        h.next_leaf          = next_leaf;
        h.chain_head_page_id = chain_head_page_id;

        uint8_t raw[16];
        BPT::serialize_csr_continuation_header(h, raw);
        std::memcpy(buffer_, raw, 16);

        // Payload: sequence of zigzag-varint deltas. Running cursor starts
        // at prev_dst_carry.
        std::size_t cursor = 16;
        uint64_t prev = prev_dst_carry;
        for (std::size_t i = 0; i < dsts_count; ++i) {
            const uint64_t cur = dsts_slice[i];
            const uint64_t delta_u = cur - prev;
            const int64_t  delta_i = static_cast<int64_t>(delta_u);
            uint8_t scratch[BPT::VARINT_MAX_BYTES];
            const std::size_t n = BPT::varint_encode(
                BPT::zigzag_encode_i64(delta_i), scratch, sizeof(scratch));
            std::memcpy(buffer_ + cursor, scratch, n);
            cursor += n;
            prev = cur;
        }

        // Parallel edge_id stream for CSR-hybrid pages that carry edge_ids:
        // encoded analogously to the dst stream (zigzag-delta against
        // prev_eid_carry). Counts match dsts_count so the reader can pair
        // them 1:1 in cont_dsts_ / cont_eids_.
        if (has_eids) {
            uint64_t prev_eid = prev_eid_carry;
            for (std::size_t i = 0; i < eids_count; ++i) {
                const uint64_t cur = eids_slice[i];
                const uint64_t delta_u = cur - prev_eid;
                const int64_t  delta_i = static_cast<int64_t>(delta_u);
                uint8_t scratch[BPT::VARINT_MAX_BYTES];
                const std::size_t n = BPT::varint_encode(
                    BPT::zigzag_encode_i64(delta_i), scratch, sizeof(scratch));
                std::memcpy(buffer_ + cursor, scratch, n);
                cursor += n;
                prev_eid = cur;
            }
        }

        file_.write(buffer_, Page::SIZE);
        ++pages_written_;
    }

    // Overflow routine for a hub src. Called from emit_current_src_to_page_
    // when (src_id, dsts) cannot fit even on an empty chain-head page. The
    // writer first flushes any pending chain-head page in progress (so the
    // hub starts on a clean page), then packs as many dsts as fit on the
    // new chain-head, and finally spills remaining dsts onto continuation
    // pages. Patches the chain-head's next_leaf AFTER chain emission so the
    // header reflects the true forward pointer.
    //
    // When emit_edge_ids_ is true, the hub chain ALSO persists parallel
    // edge_ids alongside dsts so that count(e) queries over CSR-hybrid
    // projections return the correct count instead of zero. The chain-head
    // carries an extra `k_on_head` varint after `(src_id, total_degree)` so
    // the reader can determine where the dst stream ends and the eid stream
    // begins on that page; that varint is signalled by setting
    // kIsHubChainHead in the page flags. Each continuation page also carries
    // `chunk_count` eid varints after its dst chunk, with the running eid
    // cursor crossing chunk boundaries the same way prev_dst_carry does.
    void emit_hub_chain_(uint64_t src_id,
                         const std::vector<uint64_t>& dsts,
                         const std::vector<uint64_t>& eids) noexcept
    {
        // If the current page already holds at least one entry, flush it
        // first so the hub starts with a fresh chain-head page. This
        // preserves the invariant that chain-head pages for hubs carry a
        // single src entry (matches the reader's expectation that chunk_count
        // on continuations belongs to exactly one hub).
        if (current_page_entry_count_ > 0) {
            const uint32_t min_src_low =
                static_cast<uint32_t>(current_page_entries_.front() & 0xFFFFFFFFu);
            // next_leaf for the just-flushed page = the upcoming chain-head
            // page (which will be this hub's chain head). Carry the per-page
            // eid flag so post-hub scans see the same signal as entries
            // emitted before it.
            const uint8_t page_flags = emit_edge_ids_
                ? BPT::CSRHybridFlags::kHasEdgeIds
                : 0u;
            finalize_current_page_(/*next_leaf=*/pages_written_ + 1,
                                   page_flags,
                                   /*value_or_chunk_count=*/current_page_entry_count_,
                                   /*min_src_or_head=*/min_src_low);
        }

        // When emit_edge_ids_ is on, eids must be a length-matched parallel
        // stream to dsts. Defensive only — the append() entry point enforces
        // this invariant; if the caller somehow passes a mismatched eids
        // vector we fall back to the dst-stream-only path for this hub.
        const bool hub_emit_eids =
            emit_edge_ids_ && (eids.size() == dsts.size());

        // --- Determine how many dsts fit on the fresh chain-head page.
        //
        // Layout on chain-head (single entry): 16 header + 2 offset slot =
        // 18 bytes fixed. Entry body: varint(src) + varint(total_degree) +
        // (hub_emit_eids ? varint(k_on_head) : 0) + varint(dst[0]) +
        // sum_i>=1 zigzag-varint-delta + (hub_emit_eids ? eid stream).
        // The chain-head's degree field is the TOTAL degree (including
        // dsts on continuations).
        //
        // Greedy K-packing: we walk dsts, tracking the running on-page
        // byte cost INCLUDING the parallel eid stream that would be
        // emitted if we picked this K. The k_on_head varint is variable-
        // length itself, so we conservatively reserve VARINT_MAX_BYTES (10)
        // up front and trim later. (The exact varint size depends on the
        // final K, which depends on the budget — chicken-and-egg. Using a
        // 10-byte upper bound costs at most 9 bytes of unused budget on a
        // ~4 KB page, which is negligible.)
        const std::size_t fixed_overhead = 16 + 2;
        const std::size_t budget = Page::SIZE - fixed_overhead;
        const std::size_t k_on_head_reserved =
            hub_emit_eids ? BPT::VARINT_MAX_BYTES : 0u;
        const std::size_t header_entry_prefix =
            BPT::varint_size(src_id)
          + BPT::varint_size(static_cast<uint64_t>(dsts.size()))
          + k_on_head_reserved;
        // Find K (number of dsts to include on chain-head) by greedy packing.
        // Start with dst[0] (full varint) and append dst[i] zigzag-deltas
        // until the next would exceed budget. When hub_emit_eids is true,
        // each new dst also costs an eid varint at the same K position.
        std::size_t k_on_head = 0;
        std::size_t running = header_entry_prefix;
        if (!dsts.empty()) {
            const std::size_t d0_sz = BPT::varint_size(dsts[0]);
            const std::size_t e0_sz = hub_emit_eids
                ? BPT::varint_size(eids[0]) : 0u;
            if (running + d0_sz + e0_sz <= budget) {
                running += d0_sz + e0_sz;
                k_on_head = 1;
                uint64_t prev = dsts[0];
                uint64_t prev_eid = hub_emit_eids ? eids[0] : 0;
                for (std::size_t i = 1; i < dsts.size(); ++i) {
                    const uint64_t cur = dsts[i];
                    const uint64_t delta_u = cur - prev;
                    const int64_t  delta_i = static_cast<int64_t>(delta_u);
                    const std::size_t dst_sz =
                        BPT::varint_size(BPT::zigzag_encode_i64(delta_i));
                    std::size_t eid_sz = 0;
                    if (hub_emit_eids) {
                        const uint64_t eid_cur = eids[i];
                        const uint64_t eid_delta_u = eid_cur - prev_eid;
                        const int64_t  eid_delta_i =
                            static_cast<int64_t>(eid_delta_u);
                        eid_sz = BPT::varint_size(
                            BPT::zigzag_encode_i64(eid_delta_i));
                    }
                    if (running + dst_sz + eid_sz > budget) break;
                    running += dst_sz + eid_sz;
                    ++k_on_head;
                    prev = cur;
                    if (hub_emit_eids) prev_eid = eids[i];
                }
            }
        }
        // Defensive: if even dst[0] didn't fit (pathological huge src_id),
        // fall back to k_on_head == 0 — the chain head carries only the
        // (src, degree) prefix and ALL dsts spill onto continuations. The
        // reader's cursor-start formula handles this (prev_dst_carry == 0
        // for the first continuation, which decodes the first varint as
        // zigzag(delta) against 0 == the absolute first dst, provided we
        // ensure the writer emits it that way). To keep things simple and
        // matching the reader's current decode path, we require k_on_head
        // >= 1 whenever dsts is non-empty. The estimate_entry_bytes_ budget
        // check upstream guarantees this since a single src_id + degree +
        // dst[0] is at most 30 bytes << 4078.
        // (No assertion; if the above ever fires in tests we will see K==0
        // continuation runs which the reader's cache path does not
        // currently model. Safeguarded by the upstream budget check.)

        // Remember the chain-head page number; we'll rewrite its next_leaf
        // field (bytes 8..11) after continuations are emitted so it points
        // to the first continuation page.
        const uint32_t chain_head_page_num = pages_written_;

        // Pack the chain-head body: just one entry.
        // Chain-head body layout when hub_emit_eids is true (parallel edge_id
        // stream persisted for count(e) correctness):
        //     varint(src_id) | varint(total_degree) | varint(k_on_head)
        //   | dst[0] full varint | dst[1..k-1] zigzag-deltas
        //   | eid[0] full varint | eid[1..k-1] zigzag-deltas
        // Chain-head body layout when hub_emit_eids is false (dst stream only):
        //     varint(src_id) | varint(total_degree)
        //   | dst[0] full varint | dst[1..k-1] zigzag-deltas
        std::vector<uint8_t> body;
        body.resize(running);  // upper bound from the packing walk above
        {
            std::size_t off = 0;
            off += BPT::varint_encode(src_id, body.data() + off, BPT::VARINT_MAX_BYTES);
            off += BPT::varint_encode(static_cast<uint64_t>(dsts.size()),
                                      body.data() + off, BPT::VARINT_MAX_BYTES);
            if (hub_emit_eids) {
                off += BPT::varint_encode(static_cast<uint64_t>(k_on_head),
                                          body.data() + off, BPT::VARINT_MAX_BYTES);
            }
            if (k_on_head > 0) {
                off += BPT::varint_encode(dsts[0], body.data() + off, BPT::VARINT_MAX_BYTES);
                uint64_t prev = dsts[0];
                for (std::size_t i = 1; i < k_on_head; ++i) {
                    const uint64_t cur = dsts[i];
                    const uint64_t delta_u = cur - prev;
                    const int64_t  delta_i = static_cast<int64_t>(delta_u);
                    off += BPT::varint_encode(BPT::zigzag_encode_i64(delta_i),
                                              body.data() + off, BPT::VARINT_MAX_BYTES);
                    prev = cur;
                }
                if (hub_emit_eids) {
                    off += BPT::varint_encode(eids[0], body.data() + off,
                                              BPT::VARINT_MAX_BYTES);
                    uint64_t prev_eid = eids[0];
                    for (std::size_t i = 1; i < k_on_head; ++i) {
                        const uint64_t cur = eids[i];
                        const uint64_t delta_u = cur - prev_eid;
                        const int64_t  delta_i = static_cast<int64_t>(delta_u);
                        off += BPT::varint_encode(
                            BPT::zigzag_encode_i64(delta_i),
                            body.data() + off, BPT::VARINT_MAX_BYTES);
                        prev_eid = cur;
                    }
                }
            }
            body.resize(off);  // trim to actual encoded length
        }

        // Stage the single-entry chain head in the per-page vectors so the
        // finalize_current_page_ path reuses its offset-table serialization.
        current_page_entry_count_ = 1;
        current_page_offsets_.push_back(static_cast<uint16_t>(16 + 2));
        current_page_entries_.push_back(src_id);
        current_page_entry_bodies_.push_back(std::move(body));
        current_page_bytes_used_  = current_page_entry_bodies_.front().size();

        // Finalize the chain head with next_leaf tentatively = 0; we patch
        // it to point to the first continuation below.
        const uint32_t chain_head_min_src_low =
            static_cast<uint32_t>(src_id & 0xFFFFFFFFu);
        const uint8_t chain_head_flags = hub_emit_eids
            ? static_cast<uint8_t>(BPT::CSRHybridFlags::kHasEdgeIds
                                 | BPT::CSRHybridFlags::kIsHubChainHead)
            : 0u;
        finalize_current_page_(/*next_leaf=*/0,
                               /*flags=*/chain_head_flags,
                               /*value_or_chunk_count=*/1,
                               /*min_src_or_head=*/chain_head_min_src_low);

        // --- Emit continuation pages for dsts [k_on_head, end).
        //
        // We need to decide, for each continuation, how many dsts fit in
        // one page (4080 byte payload budget). Greedy fill: starting from
        // the running cursor (previous dst), pack dsts via zigzag-varint
        // until the next varint would exceed budget. When hub_emit_eids is
        // true (parallel edge_ids are being persisted alongside dsts), each
        // dst slot also costs an eid varint at the same K position, so the
        // budget walk must account for both streams together.
        const std::size_t continuation_budget = Page::SIZE - 16;
        uint64_t prev_dst_carry = (k_on_head > 0) ? dsts[k_on_head - 1] : 0;
        uint64_t prev_eid_carry = (hub_emit_eids && k_on_head > 0)
            ? eids[k_on_head - 1] : 0;

        std::size_t i = k_on_head;
        const std::size_t total = dsts.size();

        // Collect continuation-page metadata BEFORE emitting, so each
        // continuation's next_leaf can be set correctly: forward to the
        // next continuation, or 0 for the last-in-chain. The 0 is
        // provisional — the last continuation's page number is remembered
        // in pending_cont_patch_page_, and when a subsequent page is
        // written, patch_pending_continuation_next_leaf_ seeks back and
        // rewrites the 0 to point at it, keeping the leaf chain walkable
        // across hub boundaries. If nothing follows, the 0 correctly marks
        // end-of-chain.
        //
        // Page-slicing walk: for each chunk, compute how many dsts fit.
        // When edge_ids are being persisted, each chunk also carries an eid
        // carry-in / carry-out so emit_hub_continuation_ knows how to
        // delta-encode eid[0] of the chunk against the previous chunk's
        // last eid.
        struct ChunkSlice {
            std::size_t start;
            std::size_t count;
            uint64_t    prev_dst_carry;
            uint64_t    prev_eid_carry;
        };
        std::vector<ChunkSlice> slices;
        {
            uint64_t carry_dst = prev_dst_carry;
            uint64_t carry_eid = prev_eid_carry;
            while (i < total) {
                std::size_t start = i;
                std::size_t used  = 0;
                uint64_t local_prev_dst = carry_dst;
                uint64_t local_prev_eid = carry_eid;
                while (i < total) {
                    const uint64_t cur = dsts[i];
                    const uint64_t delta_u = cur - local_prev_dst;
                    const int64_t  delta_i = static_cast<int64_t>(delta_u);
                    const std::size_t dst_sz =
                        BPT::varint_size(BPT::zigzag_encode_i64(delta_i));
                    std::size_t eid_sz = 0;
                    if (hub_emit_eids) {
                        const uint64_t eid_cur = eids[i];
                        const uint64_t eid_delta_u = eid_cur - local_prev_eid;
                        const int64_t  eid_delta_i =
                            static_cast<int64_t>(eid_delta_u);
                        eid_sz = BPT::varint_size(
                            BPT::zigzag_encode_i64(eid_delta_i));
                    }
                    if (used + dst_sz + eid_sz > continuation_budget) {
                        break;
                    }
                    used += dst_sz + eid_sz;
                    local_prev_dst = cur;
                    if (hub_emit_eids) local_prev_eid = eids[i];
                    ++i;
                }
                if (i == start) {
                    // Pathological: even one dst would overflow a continuation
                    // page. Mathematically impossible for varint <= 10 bytes.
                    // Break to avoid an infinite loop under a corrupt budget.
                    break;
                }
                slices.push_back(ChunkSlice{start, i - start, carry_dst,
                                            carry_eid});
                carry_dst = local_prev_dst;
                carry_eid = local_prev_eid;
            }
            prev_dst_carry = carry_dst;
            prev_eid_carry = carry_eid;
        }

        // Emit continuations. The LAST continuation is emitted with
        // next_leaf=0 initially; if a subsequent chain-head page follows
        // (another src), patch_pending_continuation_next_leaf_ will rewrite
        // it at the next page-write callsite to point at the new chain-head,
        // keeping the leaf chain walkable across hub boundaries. If no src
        // follows, the 0 correctly marks end-of-chain.
        const uint32_t first_continuation_page_num = pages_written_;
        uint32_t last_continuation_page_num = UINT32_MAX;
        for (std::size_t s = 0; s < slices.size(); ++s) {
            const auto& sl = slices[s];
            const bool is_last = (s + 1 == slices.size());
            const uint32_t this_page_num = pages_written_;
            const uint32_t next = is_last ? 0u : (this_page_num + 1);
            const uint64_t* eid_ptr = hub_emit_eids
                ? (eids.data() + sl.start) : nullptr;
            const std::size_t eid_count = hub_emit_eids ? sl.count : 0u;
            emit_hub_continuation_(dsts.data() + sl.start,
                                   sl.count,
                                   next,
                                   chain_head_page_num,
                                   sl.prev_dst_carry,
                                   eid_ptr,
                                   eid_count,
                                   sl.prev_eid_carry);
            if (is_last) {
                last_continuation_page_num = this_page_num;
            }
        }
        if (last_continuation_page_num != UINT32_MAX) {
            pending_cont_patch_page_ = last_continuation_page_num;
        }

        // Patch the chain-head's next_leaf to point at the first
        // continuation. Header layout:
        //   offset 8..11 : next_leaf (uint32 LE)
        // We seek back to the chain-head's byte offset within the file,
        // rewrite 4 bytes, and resume append at end of file.
        if (!slices.empty()) {
            const std::streamoff chain_head_byte_off =
                static_cast<std::streamoff>(chain_head_page_num)
              * static_cast<std::streamoff>(Page::SIZE);
            const std::streamoff resume_pos =
                static_cast<std::streamoff>(pages_written_)
              * static_cast<std::streamoff>(Page::SIZE);
            uint32_t patch_value = first_continuation_page_num;
            char patch_bytes[4];
            patch_bytes[0] = static_cast<char>( patch_value        & 0xFFu);
            patch_bytes[1] = static_cast<char>((patch_value >> 8)  & 0xFFu);
            patch_bytes[2] = static_cast<char>((patch_value >> 16) & 0xFFu);
            patch_bytes[3] = static_cast<char>((patch_value >> 24) & 0xFFu);
            file_.seekp(chain_head_byte_off + 8, std::ios::beg);
            file_.write(patch_bytes, 4);
            file_.seekp(resume_pos, std::ios::beg);
        }
    }

    // Push the staged src's (staging_src_, staging_dsts_) entry onto the
    // current page, triggering flush(es) and/or hub chain(s) as needed.
    void emit_current_src_to_page_() noexcept
    {
        const std::size_t entry_sz =
            estimate_entry_bytes_(staging_src_, staging_dsts_,
                                  staging_eids_, emit_edge_ids_);

        // Does a fresh empty page have room for this entry plus the one
        // offset-table slot? Fresh budget = Page::SIZE - 16 - 2 = 4078.
        const std::size_t fresh_budget = Page::SIZE - 16 - 2;
        if (entry_sz > fresh_budget) {
            // Hub. Let emit_hub_chain_ flush the current page (if any),
            // open a fresh chain-head, fill chain-head + continuations.
            emit_hub_chain_(staging_src_, staging_dsts_, staging_eids_);
            return;
        }

        // Fits in a fresh page. Does it fit in the CURRENT page?
        const std::size_t need = entry_sz + 2;  // body + new offset-table slot
        const std::size_t remaining = remaining_body_budget_with_new_entry_();
        if (need > remaining) {
            // Flush current page, open a fresh one, retry. Propagate the
            // per-page edge_id flag so cross-page scans see a consistent
            // signal about which pages carry the parallel eid stream.
            if (current_page_entry_count_ > 0) {
                const uint32_t min_src_low =
                    static_cast<uint32_t>(current_page_entries_.front() & 0xFFFFFFFFu);
                const uint8_t page_flags = emit_edge_ids_
                    ? BPT::CSRHybridFlags::kHasEdgeIds
                    : 0u;
                finalize_current_page_(/*next_leaf=*/pages_written_ + 1,
                                       page_flags,
                                       /*value_or_chunk_count=*/current_page_entry_count_,
                                       /*min_src_or_head=*/min_src_low);
            }
            // After finalize, current page is reset.
        }

        // Pack the entry on the current page.
        std::vector<uint8_t> body;
        body.resize(entry_sz);
        const std::size_t actual = encode_entry_(staging_src_, staging_dsts_,
                                                 staging_eids_, emit_edge_ids_,
                                                 body.data());
        body.resize(actual);

        // Offset of this entry within the page: 16 + 2*(new_count) + bytes_used.
        // Note: the 2*new_count depends on the FINAL offset-table size on
        // this page, which we don't know yet. However, since the offset
        // table is laid out at the very start (after the 16B header) and
        // entries follow contiguously, offset_table[i] = 16 + 2*V + sum_{j<i} body_j.size()
        // where V is the final value_count for this page. That means we
        // cannot know the absolute offset of any entry until we know V.
        //
        // Workaround: store "raw offset from start of bodies" (i.e.
        // sum_{j<i} body_j.size()) and patch it into the final absolute
        // offset at finalize time, using the known V at that moment.
        //
        // We store the cumulative relative offset here and finalize_current_page_
        // recomputes the absolute offsets during page serialization.
        current_page_offsets_.push_back(0);  // placeholder; patched below
        current_page_entries_.push_back(staging_src_);
        current_page_entry_bodies_.push_back(std::move(body));
        current_page_bytes_used_ += current_page_entry_bodies_.back().size();
        ++current_page_entry_count_;

        // Recompute offsets (O(value_count), but value_count is bounded by
        // the ~hundreds of srcs per page at worst — cheap).
        const std::size_t payload_start = 16 + 2 * current_page_entry_count_;
        std::size_t cursor = payload_start;
        for (std::size_t i = 0; i < current_page_offsets_.size(); ++i) {
            current_page_offsets_[i] = static_cast<uint16_t>(cursor);
            cursor += current_page_entry_bodies_[i].size();
        }
    }

    // --- members ---
    std::fstream file_;
    char*        buffer_;
    uint32_t     pages_written_;
    bool         finalized_;

    // Current chain-head page staging.
    uint32_t                         current_page_entry_count_;
    std::size_t                      current_page_bytes_used_;
    std::vector<uint16_t>            current_page_offsets_;
    std::vector<uint64_t>            current_page_entries_;
    std::vector<std::vector<uint8_t>> current_page_entry_bodies_;

    // Running src staging (buffers the full adjacency list for one src).
    bool                  staging_has_;
    uint64_t              staging_src_;
    std::vector<uint64_t> staging_dsts_;
    // Parallel edge-id stream. Populated only when emit_edge_ids_ is true
    // and N >= 3. Always has the same length as staging_dsts_.
    std::vector<uint64_t> staging_eids_;

    // Deferred forward-pointer patch state. When emit_hub_chain_ writes a
    // continuation chain whose last page has next_leaf=0 (because the
    // next chain-head page number is not yet known), we record the
    // continuation's page number here. The next new-page write callsite
    // calls patch_pending_continuation_next_leaf_ to seek back and rewrite
    // bytes 8..11 with the correct target page before appending the new page.
    // UINT32_MAX == no pending patch.
    uint32_t pending_cont_patch_page_;

    // Whether the page-level flag kHasEdgeIds must be set and a parallel
    // eid varint chain must be appended after every dst varint chain.
    // Enabling this persists edge_ids in v3 CSR-hybrid pages so that
    // count(e) queries return the correct count rather than zero.
    // Set once at construction from the ctor argument and never mutated.
    // Implies N >= 3 (guarded in the ctor initializer).
    bool emit_edge_ids_;
};

template<std::size_t N>
struct SplitData {
    const std::array<uint64_t, N>* record;
    int32_t encoded_page_number;
    bool need_split;

    SplitData(const std::array<uint64_t, N>* record, int32_t encoded_page_number, bool need_split) :
        record(record),
        encoded_page_number(encoded_page_number),
        need_split(need_split)
    { }

    SplitData() = default;
};

template<std::size_t N>
class BPTDirWriter {
private:
    std::fstream file;
    std::vector<char*> pages;

public:
    static constexpr auto max_records = (Page::SIZE - 2 * sizeof(int32_t))
                                      / (sizeof(uint64_t) * N + sizeof(int32_t));

    BPTDirWriter(const std::string& filename)
    {
        file.open(filename, std::ios::out | std::ios::binary);
        if (file.fail()) {
            WARN("Error opening file ", filename);
        }
        auto root = new char[Page::SIZE];
        memset(root, 0, Page::SIZE);
        pages.push_back(root);
    }

    ~BPTDirWriter()
    {
        for (auto page : pages) {
            file.write(page, Page::SIZE);
            delete[] page;
        }
        file.close();
    }

    uint64_t* get_keys(int32_t dir_page_number)
    {
        return reinterpret_cast<uint64_t*>(pages[dir_page_number]);
    }

    uint32_t* get_key_count(int32_t dir_page_number)
    {
        return reinterpret_cast<uint32_t*>(pages[dir_page_number] + (sizeof(uint64_t) * max_records * N));
    }

    int32_t* get_children(int32_t dir_page_number)
    {
        return reinterpret_cast<int32_t*>(
            pages[dir_page_number] + (sizeof(uint64_t) * max_records * N) + sizeof(uint32_t)
        );
    }

    SplitData<N>
        bulk_insert(const std::array<uint64_t, N>* record, int32_t dir_page_number, int32_t leaf_page_number)
    {
        uint64_t* keys = get_keys(dir_page_number);
        uint32_t* key_count = get_key_count(dir_page_number);
        int32_t* children = get_children(dir_page_number);

        SplitData<N> split_data;

        if (children[*key_count] < 0) {
            // negative number: pointer to dir
            split_data = bulk_insert(record, children[*key_count] * -1, leaf_page_number);
        } else {
            // positive number: pointer to leaf
            split_data = SplitData(record, leaf_page_number, true);
        }

        if (split_data.need_split) {
            // Case 1: no need to split this node
            if (*key_count < max_records) {
                // update key
                std::memcpy(&keys[(*key_count) * N], split_data.record->data(), N * sizeof(uint64_t));
                ++(*key_count);
                // update child
                children[*key_count] = split_data.encoded_page_number;
                return SplitData<N>(nullptr, 0, false);
            }
            // Case 2: non-root split
            else if (dir_page_number != 0)
            {
                // create new dir page
                int32_t new_page_number = pages.size();
                auto new_page = new char[Page::SIZE];
                memset(new_page, 0, Page::SIZE);
                pages.push_back(new_page);

                auto new_dir_children = get_children(new_page_number);
                auto new_dir_key_count = get_key_count(new_page_number);

                new_dir_children[0] = split_data.encoded_page_number;
                *new_dir_key_count = 0;
                return SplitData<N>(split_data.record, new_page_number * -1, true);
            }
            // Case 3: root split
            else {
                // create 2 new pages (new_lhs, new_rhs)
                int32_t lhs_page_number = pages.size();
                auto lhs_page = new char[Page::SIZE];
                memset(lhs_page, 0, Page::SIZE);
                pages.push_back(lhs_page);

                int32_t rhs_page_number = pages.size();
                auto rhs_page = new char[Page::SIZE];
                memset(rhs_page, 0, Page::SIZE);
                pages.push_back(rhs_page);

                // new_lhs has everything previous
                std::memcpy(lhs_page, pages[dir_page_number], Page::SIZE);

                // new_rhs has 0 keys and 1 record (the splitted record)
                auto rhs_key_count = get_key_count(rhs_page_number);
                auto rhs_children = get_children(rhs_page_number);
                *rhs_key_count = 0;
                rhs_children[0] = split_data.encoded_page_number;

                // new root will have the new pages as children
                std::memcpy(keys, split_data.record->data(), N * sizeof(uint64_t));
                *key_count = 1;
                children[0] = lhs_page_number * -1;
                children[1] = rhs_page_number * -1;

                return SplitData<N>(nullptr, 0, false);
            }
        } else {
            return SplitData<N>(nullptr, 0, false);
        }
    }
};
