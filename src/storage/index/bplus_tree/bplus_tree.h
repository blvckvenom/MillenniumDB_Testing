#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>

#include "storage/file_id.h"
#include "storage/index/bplus_tree/bplus_tree_dir.h"
#include "storage/index/bplus_tree/bplus_tree_leaf.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_base.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_csr.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"
#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/record.h"

template <std::size_t N> class BptIter {
public:
    // shouldn't use a BptIter constructed like this.
    // This exists only to allow reserving space and then reassign to a valid BptIter
    BptIter() noexcept :
        interruption_requested (nullptr),
        leaf_format_           (BPT::LeafFormat::BITSET) { }

    // Constructs a polymorphic BptIter. `leaf_format` mirrors the owning
    // BPlusTree<N>'s format constant and is paid through the iterator so
    // next-leaf transitions can dispatch without reaching back into the
    // tree. The SearchLeafResult always carries a BPTLeafV1<N> because the
    // directory-layer search path is V1-only in Spec #5 T5.9; the V1 is
    // moved into the polymorphic unique_ptr here. Wiring DELTA_VARINT
    // through the directory is T5.10 / T5.11.
    BptIter(bool* interruption_requested,
            SearchLeafResult<N>&& leaf_and_pos,
            const Record<N>& max,
            BPT::LeafFormat leaf_format = BPT::LeafFormat::BITSET,
            const Record<N>* min = nullptr) noexcept;

    BptIter(BptIter&& other) noexcept;

    void operator=(BptIter&& other) noexcept;

    const Record<N>* next();

    inline bool is_null() const {
        return interruption_requested == nullptr;
    }

    inline void set_null() {
        this->interruption_requested = nullptr;
        current_leaf_.reset();
    }

private:
    bool* interruption_requested;
    uint_fast32_t current_pos;
    Record<N> current_record;
    Record<N> max;
    // Primary leaf view. In BITSET mode this is the only reader. In
    // DELTA_VARINT mode this is still a BPTLeafV1 purely to hold the
    // BufferManager page pin through RAII — see v2_reader_ below for the
    // actual record decoder.
    std::unique_ptr<BPTLeafBase<N>> current_leaf_;
    // DELTA_VARINT auxiliary reader. Populated only when
    // leaf_format_ == DELTA_VARINT. Views the same page bytes as
    // current_leaf_ under a BPTLeafV2<N> ReadTag decoder; all record
    // reads (get_value_count, update_record, has_next) dispatch here
    // in v2 mode. See Spec #5 §5.2 for the on-disk format.
    std::unique_ptr<BPTLeafBase<N>> v2_reader_;
    // CSR_HYBRID auxiliary reader. Populated only when
    // leaf_format_ == CSR_HYBRID. Views the same page bytes as
    // current_leaf_ under a BPTLeafCSR<N> ReadTag decoder (Spec #8
    // T8.4). The v1 held in current_leaf_ remains purely as a pin
    // holder; all record reads dispatch to v3_reader_ when this slot
    // is populated. See Spec #8 §5 for the on-disk format.
    std::unique_ptr<BPTLeafBase<N>> v3_reader_;
    BPT::LeafFormat leaf_format_;
};


template <std::size_t N> class BPlusTree {
public:
    // (MDB_PAGE_SIZE - SIZE_OF(value_count) - SIZE_OF(next_leaf)) / (SIZE_OF(UINT64) * N)
    static constexpr auto leaf_max_records = (Page::SIZE - 2*sizeof(int32_t) ) / (sizeof(uint64_t)*N);
    static constexpr auto dir_max_records  = (Page::SIZE - 2*sizeof(int32_t) ) / (sizeof(uint64_t)*N + sizeof(int32_t));

    // Optional leaf_format selects between the pre-Spec-#5 BITSET encoding
    // (default, byte-identical to pre-T5.9 behavior) and the Spec-#5
    // DELTA_VARINT encoding. Every call site in the current codebase relies
    // on the default, so pre-Spec-#5 trees are unaffected. The catalog
    // plumbing that populates this parameter from on-disk metadata lands in
    // T5.10; the GQL config plumbing in T5.11.
    BPlusTree(const std::string& name,
              BPT::LeafFormat leaf_format = BPT::LeafFormat::BITSET);

    const FileId dir_file_id;
    const FileId leaf_file_id;

    // Read-only accessor for the tree's leaf encoding format. Constant over
    // the BPlusTree's lifetime.
    BPT::LeafFormat get_leaf_format() const noexcept { return leaf_format_; }

    // Factory helper: construct a polymorphic leaf view over `page` based on
    // `fmt`. When fmt == DELTA_VARINT the helper cross-checks that the page's
    // first byte is 2 (format_version); a mismatch raises
    // BPT::BPTLeafV2DecodeException. The BITSET branch does not cross-check
    // byte 0 because pre-Spec-#5 pages legitimately have value_count=2 at
    // byte 0 (design §6.1 edge case). The returned unique_ptr owns the
    // Page* pin through BPTLeafV1's destructor in the BITSET path; the
    // DELTA_VARINT path does NOT own the pin (V2 holds raw bytes only) —
    // T5.10 will layer a pin-holding wrapper when threading format through
    // the directory.
    static std::unique_ptr<BPTLeafBase<N>> open_leaf_page(Page& page,
                                                          BPT::LeafFormat fmt);

    // returns true if record was inserted, false if record was already there
    bool insert(const Record<N>& record);

    // returns true if record was deleted, false if record did not exists
    bool delete_record(const Record<N>& record);


    // returns false if an error in the BPT is found
    bool check(std::ostream& os) const;

    BptIter<N> get_range(bool* interruption_requested,
                         const Record<N>& min,
                         const Record<N>& max) const noexcept;

    double estimate_records(const Record<N>& min,
                            const Record<N>& max) const;

    static double estimate_records(const BPlusTreeDir<N>& root,
                                   const Record<N>& min,
                                   const Record<N>& max);

    // It doesn't simply return the root, it is an unique_ptr so it pins the page
    std::unique_ptr<BPlusTreeDir<N>> get_root() const noexcept;

private:
    // Leaf-encoding format discriminator for this tree. Set at construction
    // time from the optional ctor parameter; used by get_range() to initialize
    // BptIter<N> and by open_leaf_page() to select the concrete reader.
    const BPT::LeafFormat leaf_format_;
};
