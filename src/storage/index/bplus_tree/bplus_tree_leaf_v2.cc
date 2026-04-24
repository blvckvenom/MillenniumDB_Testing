// BPTLeafV2 write-path implementation (Spec #5 T5.7).
//
// This file implements the encoder: append_record() serialises a Record<N>
// as delta + LEB128 varint bytes against a running cursor, and flush()
// commits the 16-byte v2 header plus the accumulated payload to the backing
// 4 KB page buffer, zero-padding to Page::SIZE.
//
// The reader side (get_record, search_index, check_range, print, mutating
// methods) is T5.8; those overrides here throw std::logic_error so the class
// remains instantiable for the write path without committing to a
// half-finished read-path contract.
//
// Spec reference: docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md
//                 (§5.2 layout, §5.3 worked example, §5.4 padding)

#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"

#include <cstring>
#include <stdexcept>

namespace {

// Scratch buffer size per record: every one of N fields consumes up to
// VARINT_MAX_BYTES (10) bytes. For N=3 this is 30 bytes on the stack —
// no heap allocation per append_record().
constexpr size_t kScratchBytesFor(std::size_t N) noexcept
{
    return N * BPT::VARINT_MAX_BYTES;
}

// Convenience: the byte offset at which record payload begins on-disk.
constexpr size_t kPayloadOffset = sizeof(BPT::BPTLeafV2Header); // 16

// Budget for record bytes on a single page.
constexpr size_t kPayloadBudget = Page::SIZE - kPayloadOffset;  // 4080

}  // namespace


template <std::size_t N>
BPTLeafV2<N>::BPTLeafV2(char* page_bytes, uint32_t next_leaf) noexcept :
    page_bytes_ (page_bytes),
    next_leaf_  (next_leaf)
{
    payload_.reserve(kPayloadBudget);
}


template <std::size_t N>
size_t BPTLeafV2<N>::encode_one_record_(const Record<N>& rec,
                                        const Record<N>* prev_or_null,
                                        uint8_t*         out_buf,
                                        size_t           max_bytes) noexcept
{
    size_t written = 0;
    for (std::size_t j = 0; j < N; ++j) {
        // Remaining budget for this field's varint. The codec writes at most
        // VARINT_MAX_BYTES; the caller sizes the scratch at N * max so this
        // check is a safety net rather than a functional constraint.
        const size_t remaining = (written < max_bytes) ? (max_bytes - written) : 0;
        if (remaining < BPT::VARINT_MAX_BYTES) {
            // Scratch too small — this indicates a caller contract violation.
            // Return whatever was written so far; the overflow-check path in
            // append_record() will treat it uniformly.
            return written;
        }

        uint64_t to_encode;
        if (prev_or_null == nullptr) {
            to_encode = rec[j];
        } else {
            const int64_t delta =
                static_cast<int64_t>(rec[j]) - static_cast<int64_t>((*prev_or_null)[j]);
            to_encode = BPT::zigzag_encode_i64(delta);
        }

        written += BPT::varint_encode(to_encode, out_buf + written, remaining);
    }
    return written;
}


template <std::size_t N>
bool BPTLeafV2<N>::append_record(const Record<N>& rec) noexcept
{
    if (flushed_) {
        return false;
    }

    // Stack-local scratch buffer sized for the worst case. For N=3 this is
    // 30 bytes; N=1 gives 10 bytes; N=2 gives 20 bytes. No heap allocation.
    uint8_t scratch[kScratchBytesFor(N)];

    const size_t encoded_size = encode_one_record_(
        rec,
        has_prev_ ? &prev_record_ : nullptr,
        scratch,
        sizeof(scratch));

    if (encoded_size == 0) {
        // Scratch-too-small signal — unreachable for sane N in [1,3].
        return false;
    }

    if (payload_.size() + encoded_size > kPayloadBudget) {
        // Overflow: caller must flush() + open a new page.
        return false;
    }

    payload_.insert(payload_.end(), scratch, scratch + encoded_size);
    prev_record_ = rec;
    has_prev_    = true;
    ++value_count_;
    return true;
}


template <std::size_t N>
void BPTLeafV2<N>::flush() noexcept
{
    if (flushed_) {
        return;
    }

    // Build the 16-byte v2 header.
    BPT::BPTLeafV2Header h{};
    h.format_version = 2;
    h.record_width   = static_cast<uint8_t>(N);
    h.flags          = 0;
    h.reserved       = 0;
    h.value_count    = value_count_;
    h.next_leaf      = next_leaf_;
    h.reserved2      = 0;

    // Serialize header to page offset 0..15 (explicit little-endian).
    BPT::serialize_header(h, reinterpret_cast<uint8_t*>(page_bytes_));

    // Copy payload to page offset 16..16+payload_.size().
    if (!payload_.empty()) {
        std::memcpy(page_bytes_ + kPayloadOffset,
                    payload_.data(),
                    payload_.size());
    }

    // Zero-fill the remaining bytes so the on-disk page is deterministic.
    const size_t end = kPayloadOffset + payload_.size();
    if (end < Page::SIZE) {
        std::memset(page_bytes_ + end, 0, Page::SIZE - end);
    }

    flushed_ = true;
}


template <std::size_t N>
size_t BPTLeafV2<N>::bytes_used() const noexcept
{
    return kPayloadOffset + payload_.size();
}


// ===== Read-side stubs (replaced by T5.8). ===================================

template <std::size_t N>
Record<N> BPTLeafV2<N>::get_record(uint_fast32_t) const
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
void BPTLeafV2<N>::set_record(uint_fast32_t, Record<N>&) const
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
void BPTLeafV2<N>::set_redundant_record(Record<N>&) const
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
void BPTLeafV2<N>::update_record(uint_fast32_t, Record<N>&) const
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
uint_fast32_t BPTLeafV2<N>::search_index(const Record<N>&) const noexcept
{
    // noexcept contract prevents throwing here. Return a sentinel that any
    // caller invoking this on a T5.7 writer will treat as "no match found".
    // T5.8 replaces this with the real linear-scan implementation.
    return 0;
}

template <std::size_t N>
bool BPTLeafV2<N>::check_range(const Record<N>&) const
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
std::unique_ptr<BPlusTreeSplit<N>>
BPTLeafV2<N>::insert(const Record<N>&, bool&)
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
bool BPTLeafV2<N>::delete_record(const Record<N>&)
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
void BPTLeafV2<N>::update_to_next_leaf()
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
bool BPTLeafV2<N>::check(std::ostream&) const
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}

template <std::size_t N>
void BPTLeafV2<N>::print(std::ostream&) const
{
    throw std::logic_error("BPTLeafV2 read path not yet implemented - T5.8");
}


// Explicit template instantiations (mirror bplus_tree_leaf.cc).
template class BPTLeafV2<1>;
template class BPTLeafV2<2>;
template class BPTLeafV2<3>;
