#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "storage/index/bplus_tree/bplus_tree_leaf_base.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/bplus_tree/varint.h"
#include "storage/index/record.h"
#include "storage/page/page.h"

// BPTLeafV2 — delta + LEB128 varint encoding (Spec #5 v2 leaf format).
//
// Write path (T5.7): append_record(r) encodes r[0] as N full varints on the
// first call, and subsequent records as N zigzag(delta) varints against the
// previously-appended record. flush() commits the 16-byte header + encoded
// bytes to the page and zero-pads to Page::SIZE = 4096. Overflow detection
// is against Page::SIZE - sizeof(BPTLeafV2Header).
//
// Read path (T5.8): get_record / search_index via linear scan. Stubbed here.
//
// Page-boundary invariant: no record spans two pages — if the next record
// would push bytes_used past Page::SIZE, the writer signals "page full" and
// the caller flushes + opens a new page.
//
// The constructor takes the raw 4 KB backing buffer (`char*`). Production
// callers pass `page.get_bytes()`; the value is the same `bytes` pointer
// BPTLeafV1 resolves via its `Page*`. Unit tests allocate an aligned 4096-byte
// buffer directly, sidestepping BufferManager.
//
// Virtual destructor is inherited from BPTLeafBase<N>.
//
// Spec reference: design §5.2 (layout), §5.3 (worked example), §5.4 (padding).

template <std::size_t N>
class BPTLeafV2 : public BPTLeafBase<N> {
public:
    /// Construct a writer view over a 4 KB buffer. The buffer is the page
    /// bytes (either from Page::get_bytes() in production or a heap/stack
    /// buffer in unit tests). The first sizeof(BPTLeafV2Header) = 16 bytes
    /// are reserved for the header; append_record() writes payload bytes
    /// starting at offset 16.
    ///
    /// `next_leaf` is the page id of the leaf that follows this one in the
    /// leaf chain, or 0 if this is the last leaf. Stored in the header at
    /// flush time.
    explicit BPTLeafV2(char* page_bytes, uint32_t next_leaf = 0) noexcept;

    BPTLeafV2(const BPTLeafV2&)            = delete;
    BPTLeafV2& operator=(const BPTLeafV2&) = delete;
    BPTLeafV2(BPTLeafV2&&)                 = default;
    BPTLeafV2& operator=(BPTLeafV2&&)      = delete;

    ~BPTLeafV2() override = default;

    /// Append one record to the in-memory buffer.
    ///
    /// Returns true on success, false if the record would overflow the page
    /// (in which case the caller must flush() + open a new page before
    /// retrying).
    ///
    /// Must be called in strictly non-decreasing lexicographic order on the
    /// Record<N> primary sort key. Secondary field deltas are encoded via
    /// zigzag and may be negative when the primary field advances.
    bool append_record(const Record<N>& rec) noexcept;

    /// Commit the encoded bytes + 16-byte header to the backing buffer. Safe
    /// to call at most once meaningfully; additional calls are no-ops. After
    /// flush, further append_record() calls return false.
    ///
    /// Zero-fills all bytes between the last-written record end and
    /// Page::SIZE = 4096.
    void flush() noexcept;

    /// Number of bytes currently used on the page (header + payload), for
    /// the caller's overflow-planning loop before commit.
    size_t bytes_used() const noexcept;

    /// Number of records appended so far.
    uint32_t value_count() const noexcept { return value_count_; }

    /// True after flush() has been called.
    bool is_flushed() const noexcept { return flushed_; }

    // ========== BPTLeafBase<N> contract =====================================
    //
    // T5.7 implements only the write path. get_value_count() and has_next()
    // are trivially answerable from writer state and are implemented inline.
    // The remaining read-side methods throw std::logic_error until T5.8
    // delivers the linear-scan reader.

    uint32_t      get_value_count() const override { return value_count_; }
    bool          has_next() const override { return next_leaf_ != 0; }
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

private:
    char*                page_bytes_;
    uint32_t             next_leaf_;
    uint32_t             value_count_ = 0;
    bool                 flushed_     = false;

    // Running cursor: most recently appended record. For the first record,
    // each field is encoded as a full varint; for subsequent records, fields
    // are zigzag(delta) against this cursor.
    Record<N>            prev_record_{};
    bool                 has_prev_    = false;

    // Encoded record bytes. Buffered in memory until flush() to support the
    // overflow-check-before-commit protocol. Capacity is at most
    // Page::SIZE - sizeof(BPTLeafV2Header) = 4080 bytes.
    std::vector<uint8_t> payload_;

    // Helper: encode one record into a scratch buffer. Returns the number of
    // bytes written. `prev_or_null == nullptr` for the first record (full
    // varints); otherwise zigzag(delta) against *prev_or_null for each field.
    static size_t encode_one_record_(const Record<N>& rec,
                                     const Record<N>* prev_or_null,
                                     uint8_t*         out_buf,
                                     size_t           max_bytes) noexcept;
};

// Explicit template instantiations for the supported record widths.
extern template class BPTLeafV2<1>;
extern template class BPTLeafV2<2>;
extern template class BPTLeafV2<3>;
