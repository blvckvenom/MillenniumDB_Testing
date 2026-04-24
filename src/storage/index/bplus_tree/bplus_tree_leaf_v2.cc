// BPTLeafV2 write-path (writer) and read-path (reader) implementation.
//
// The writer (append_record + flush) serialises a Record<N> as delta +
// LEB128 varint bytes against a running cursor, and flush() commits the
// 16-byte v2 header plus the accumulated payload to the backing 4 KB page
// buffer, zero-padding to Page::SIZE.
//
// The reader (ReadTag ctor + get_record + search_index): the ctor
// validates the header, get_record linear-decodes through the varint
// stream accumulating the running cursor, and search_index scans the
// records in-stream. There is deliberately no binary search and no
// in-page offset index: each record is delta-encoded against the
// previous one, so decoding is inherently sequential and an offset table
// would spend page bytes without changing the access pattern.
//
// Mutation paths (insert, delete_record, update_to_next_leaf) throw
// std::logic_error: v2 pages are immutable post-build. Changing an index
// stored in this format requires rebuilding the projection that owns it.

#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"

#include <cstring>
#include <stdexcept>
#include <string>

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
            // Unsigned subtraction is well-defined modular arithmetic; signed
            // subtraction of two int64 casts would be UB for record values
            // near UINT64_MAX. Reinterp the uint64 wrap-around as int64 for
            // the zigzag step — two's-complement makes this bit-equivalent
            // to the naive signed path for every input pair.
            const uint64_t delta_u = rec[j] - (*prev_or_null)[j];
            const int64_t delta    = static_cast<int64_t>(delta_u);
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


// ===== Read-side implementation. =============================================

template <std::size_t N>
BPTLeafV2<N>::BPTLeafV2(const char* page_bytes, ReadTag) :
    page_bytes_ (nullptr),   // writer-side buffer unused in reader mode
    next_leaf_  (0),
    read_page_bytes_ (page_bytes)
{
    // Deserialize the 16-byte header from page_bytes[0..15].
    uint8_t raw[sizeof(BPT::BPTLeafV2Header)];
    std::memcpy(raw, page_bytes, sizeof(raw));
    read_header_ = BPT::deserialize_header(raw);

    // Header validation: reject anything that is not a well-formed v2
    // page of width N.
    if (read_header_.format_version != 2) {
        throw BPT::BPTLeafV2DecodeException(
            "invalid format_version at page offset 0 (expected 2, got "
            + std::to_string(read_header_.format_version) + ")");
    }
    if (read_header_.record_width != N) {
        throw BPT::BPTLeafV2DecodeException(
            "record_width mismatch at page offset 1 (expected "
            + std::to_string(N) + ", got "
            + std::to_string(read_header_.record_width) + ")");
    }
    if (read_header_.flags != 0) {
        throw BPT::BPTLeafV2DecodeException(
            "non-zero flags byte at offset 2 (reserved in the v2 leaf format)");
    }
    if (read_header_.reserved != 0) {
        throw BPT::BPTLeafV2DecodeException(
            "non-zero reserved byte at offset 3");
    }
    if (read_header_.reserved2 != 0) {
        throw BPT::BPTLeafV2DecodeException(
            "non-zero reserved2 field at offset 12");
    }
    if (read_header_.value_count > leaf_max_records_v2()) {
        throw BPT::BPTLeafV2DecodeException(
            "value_count " + std::to_string(read_header_.value_count)
            + " exceeds leaf_max_records_v2 "
            + std::to_string(leaf_max_records_v2()));
    }

    // Mirror the writer-facing fields so get_value_count() / has_next() /
    // search_index() can read value_count_ and next_leaf_ uniformly.
    value_count_ = read_header_.value_count;
    next_leaf_   = read_header_.next_leaf;
    // prev_record_ / has_prev_ / payload_ remain default-constructed —
    // writer-only state.
}


template <std::size_t N>
Record<N> BPTLeafV2<N>::get_record(uint_fast32_t pos) const
{
    if (read_page_bytes_ == nullptr) {
        throw std::logic_error(
            "BPTLeafV2::get_record called on a writer-mode instance");
    }
    if (pos >= value_count_) {
        throw std::out_of_range(
            "BPTLeafV2::get_record position " + std::to_string(pos)
            + " >= value_count " + std::to_string(value_count_));
    }

    const uint8_t* const start = reinterpret_cast<const uint8_t*>(read_page_bytes_) + kPayloadOffset;
    const uint8_t* const end   = reinterpret_cast<const uint8_t*>(read_page_bytes_) + Page::SIZE;

    // Fast-path: if pos == cache_pos_, return cached record directly.
    if (pos == cache_pos_) {
        Record<N> rec;
        for (std::size_t j = 0; j < N; ++j) {
            rec[j] = cache_cursor_[j];
        }
        return rec;
    }

    // Decide: resume from cache (pos > cache_pos_) or restart (pos < cache_pos_).
    uint_fast32_t resume_from;
    uint64_t cursor[N];
    const uint8_t* in;

    if (cache_pos_ != UINT_FAST32_MAX && pos > cache_pos_) {
        // Resume from cache: cache_in_ is the byte pointer AFTER the varints
        // of record cache_pos_, so decoding of record cache_pos_ + 1 starts
        // exactly there with cache_cursor_ as the running accumulator.
        in = cache_in_;
        for (std::size_t j = 0; j < N; ++j) {
            cursor[j] = cache_cursor_[j];
        }
        resume_from = cache_pos_ + 1;
    } else {
        // Restart from record 0.
        in = start;
        for (std::size_t j = 0; j < N; ++j) {
            cursor[j] = 0;
        }
        resume_from = 0;
    }

    for (uint_fast32_t i = resume_from; i <= pos; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            uint64_t v = 0;
            const size_t consumed = BPT::varint_decode(in, end, v);
            in += consumed;
            if (i == 0) {
                cursor[j] = v;
            } else {
                // Zigzag-decoded delta is int64; adding to a uint64 is
                // well-defined modular arithmetic via the static_cast. This
                // mirrors the writer's `int64 delta = rec[j] - prev[j]` on
                // the encode side.
                const int64_t delta = BPT::zigzag_decode_u64(v);
                cursor[j] += static_cast<uint64_t>(delta);
            }
        }
    }

    // Update cache with our final state. `in` now points just past the
    // varints of record `pos`, ready for the next resume.
    cache_pos_ = pos;
    for (std::size_t j = 0; j < N; ++j) {
        cache_cursor_[j] = cursor[j];
    }
    cache_in_ = in;

    Record<N> rec;
    for (std::size_t j = 0; j < N; ++j) {
        rec[j] = cursor[j];
    }
    return rec;
}


template <std::size_t N>
void BPTLeafV2<N>::set_record(uint_fast32_t pos, Record<N>& out) const
{
    // Write-through semantics: out receives the decoded record.
    out = get_record(pos);
}


template <std::size_t N>
void BPTLeafV2<N>::set_redundant_record(Record<N>& out) const
{
    // V2 has no redundant-bitset concept: delta encoding replaces the v1
    // shared-prefix bitset. Zero out `out` so callers that use this as an initial
    // scratch value see a well-defined state. V1 callers that relied on
    // set_redundant_record populating bitset-redundant bytes are not valid
    // on a V2 page.
    for (std::size_t j = 0; j < N; ++j) {
        out[j] = 0;
    }
}


template <std::size_t N>
void BPTLeafV2<N>::update_record(uint_fast32_t pos, Record<N>& out) const
{
    // V1 semantics: overwrite only the non-redundant fields. V2 has no
    // redundant fields, so every field is overwritten — equivalent to
    // set_record on V2.
    out = get_record(pos);
}


template <std::size_t N>
uint_fast32_t BPTLeafV2<N>::search_index(const Record<N>& target) const noexcept
{
    // Linear scan; the delta encoding makes decoding sequential, so there
    // is no binary search and no offset index to consult.
    // `noexcept` because the caller (BptIter) cannot handle exceptions
    // here; a malformed page should have failed at ctor time. On
    // per-record varint corruption we return value_count_ (no match) and
    // the caller falls through to the next leaf.
    if (read_page_bytes_ == nullptr) {
        // Writer-mode instance; caller contract violation.
        return 0;
    }

    const uint8_t* in  = reinterpret_cast<const uint8_t*>(read_page_bytes_) + kPayloadOffset;
    const uint8_t* end = reinterpret_cast<const uint8_t*>(read_page_bytes_) + Page::SIZE;

    uint64_t cursor[N] = {};

    for (uint_fast32_t i = 0; i < value_count_; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            uint64_t v = 0;
            size_t consumed = 0;
            try {
                consumed = BPT::varint_decode(in, end, v);
            } catch (...) {
                // In-page varint corruption on a noexcept path. Stop and
                // return "not found"; caller descends to the next leaf.
                return value_count_;
            }
            in += consumed;
            if (i == 0) {
                cursor[j] = v;
            } else {
                cursor[j] += static_cast<uint64_t>(BPT::zigzag_decode_u64(v));
            }
        }

        // Compare the just-decoded record to target: find first cursor >= target.
        bool lt = false;
        for (std::size_t j = 0; j < N; ++j) {
            if (cursor[j] < target[j]) { lt = true; break; }
            if (cursor[j] > target[j]) { break; }
        }
        if (!lt) {
            // cursor >= target; this is the first such record.
            return i;
        }
    }
    return value_count_;
}


template <std::size_t N>
bool BPTLeafV2<N>::check_range(const Record<N>& r) const
{
    if (value_count_ == 0) {
        return false;
    }
    const auto min = get_record(0);
    const auto max = get_record(value_count_ - 1);
    return min <= r && r <= max;
}


// ===== Mutation path: V2 pages are immutable post-build. =====================

template <std::size_t N>
std::unique_ptr<BPlusTreeSplit<N>>
BPTLeafV2<N>::insert(const Record<N>&, bool&)
{
    // V2 pages are immutable post-build.
    throw std::logic_error("BPTLeafV2 is immutable; insert() is not supported");
}

template <std::size_t N>
bool BPTLeafV2<N>::delete_record(const Record<N>&)
{
    // V2 pages are immutable post-build.
    throw std::logic_error("BPTLeafV2 is immutable; delete_record() is not supported");
}

template <std::size_t N>
void BPTLeafV2<N>::update_to_next_leaf()
{
    // V2 pages are immutable post-build.
    // Caller should construct a new BPTLeafV2(page_bytes, ReadTag) on the
    // next page rather than mutating this instance.
    throw std::logic_error("BPTLeafV2 is immutable; update_to_next_leaf() is not supported");
}


// ===== Diagnostics ===========================================================

template <std::size_t N>
bool BPTLeafV2<N>::check(std::ostream& os) const
{
    if (read_page_bytes_ == nullptr) {
        // Writer-mode: nothing to check on the page (flush has not happened
        // or bytes are not readable through this instance's reader state).
        os << "  WARNING: BPTLeafV2::check called on writer-mode instance\n";
        return true;
    }

    if (value_count_ == 0) {
        os << "  WARNING: empty v2 leaf. Ok only if the b+tree is empty.\n";
        return true;
    }

    Record<N> x;
    try {
        x = get_record(0);
    } catch (const BPT::BPTLeafV2DecodeException& e) {
        os << "  ERROR: BPTLeafV2 decode failure at record 0: " << e.what() << "\n";
        return false;
    }

    for (std::size_t i = 0; i < N; ++i) {
        if (x[i] == 0xFFFF'FFFF'FFFF'FFFFULL) {
            os << "  ERROR: record not_found(0xFFFF'FFFF'FFFF'FFFF) at BPTLeafV2\n";
            return false;
        }
    }

    Record<N> y;
    for (uint32_t k = 1; k < value_count_; ++k) {
        try {
            y = get_record(k);
        } catch (const BPT::BPTLeafV2DecodeException& e) {
            os << "  ERROR: BPTLeafV2 decode failure at record " << k
               << ": " << e.what() << "\n";
            return false;
        }
        if (y <= x) {
            os << "  ERROR: bad record order at BPTLeafV2\n";
            for (std::size_t n = 0; n < N; ++n) {
                os << "\t" << x[n];
            }
            os << "\n";
            for (std::size_t n = 0; n < N; ++n) {
                os << "\t" << y[n];
            }
            os << "\n";
            return false;
        }
        x = y;
    }
    return true;
}


template <std::size_t N>
void BPTLeafV2<N>::print(std::ostream& os) const
{
    os << "Printing Leaf:\n";
    if (read_page_bytes_ == nullptr) {
        os << "  (writer-mode instance; no page bytes to print)\n";
        return;
    }
    for (uint_fast32_t i = 0; i < value_count_; ++i) {
        Record<N> r;
        try {
            r = get_record(i);
        } catch (const BPT::BPTLeafV2DecodeException& e) {
            os << "  <decode error at record " << i << ": " << e.what() << ">\n";
            return;
        }
        os << "  (";
        for (std::size_t j = 0; j < N; ++j) {
            if (j != 0) {
                os << ", ";
            }
            os << r[j];
        }
        os << ")\n";
    }
}


// Explicit template instantiations (mirror bplus_tree_leaf.cc).
template class BPTLeafV2<1>;
template class BPTLeafV2<2>;
template class BPTLeafV2<3>;
