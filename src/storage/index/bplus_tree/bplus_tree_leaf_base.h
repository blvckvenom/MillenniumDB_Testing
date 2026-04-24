#pragma once

// Abstract base for B+Tree leaf page views. Concrete subclasses:
//   BPTLeafV1<N> — legacy redundant-bitset encoding (original leaf format,
//                  see bplus_tree_leaf.h). Produces byte-identical output
//                  to the original leaf format.
//   BPTLeafV2<N> — delta + LEB128 varint leaf encoding (B+Tree leaf
//                  compression: record 0 stored as full LEB128 varints,
//                  subsequent records as zigzag-delta LEB128 varints;
//                  see bplus_tree_leaf_v2.h).
//
// Virtual dispatch cost is paid once per page-open (via BptIter), not per
// record access; the inner get_record / search_index loops are non-virtual
// within each subclass.
//
// The base class owns NO data members. Subclasses hold their own Page*
// references and format-specific metadata.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>

#include "storage/index/record.h"

// forward declarations
template <std::size_t N> class BPlusTreeSplit;

template <std::size_t N>
class BPTLeafBase {
public:
    virtual ~BPTLeafBase() = default;

    // --- Read-only page-view contract ---
    virtual uint32_t      get_value_count() const = 0;
    virtual bool          has_next() const = 0;
    virtual Record<N>     get_record(uint_fast32_t pos) const = 0;
    virtual void          set_record(uint_fast32_t pos, Record<N>& out) const = 0;
    virtual void          set_redundant_record(Record<N>& out) const = 0;
    virtual void          update_record(uint_fast32_t pos, Record<N>& out) const = 0;
    virtual uint_fast32_t search_index(const Record<N>& record) const noexcept = 0;
    virtual bool          check_range(const Record<N>& r) const = 0;

    // --- Mutating contract ---
    virtual std::unique_ptr<BPlusTreeSplit<N>> insert(const Record<N>& record, bool& error) = 0;
    virtual bool delete_record(const Record<N>& record) = 0;
    virtual void update_to_next_leaf() = 0;

    // --- Diagnostics ---
    virtual bool check(std::ostream& os) const = 0;
    virtual void print(std::ostream& os) const = 0;

    // NOTE: clone(), get_page(), and the constructor-set Page* are V1-only
    // implementation details. The clone() return type is BPTLeafV1<N> by
    // value (not the abstract base), so it cannot be expressed as a pure
    // virtual on this base. Polymorphic page cloning is introduced together
    // with BptIter polymorphism when the iterator layer was extended to support
    // both V1 and V2 leaf formats; until then, clone() is V1-only and
    // reachable through the BPlusTreeLeaf<N> = BPTLeafV1<N> alias.

protected:
    BPTLeafBase() = default;
    BPTLeafBase(const BPTLeafBase&) = delete;
    BPTLeafBase& operator=(const BPTLeafBase&) = delete;
    // Move is permitted so subclasses (which carry the actual data) may
    // be moved without slicing concerns; the base itself has no state to
    // move, so this is a no-op at the base level.
    BPTLeafBase(BPTLeafBase&&) = default;
    BPTLeafBase& operator=(BPTLeafBase&&) = default;
};
