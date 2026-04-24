#include "bplus_tree.h"

#include <cassert>
#include <stdexcept>
#include <string>

#include "macros/likely.h"
#include "query/exceptions.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_base.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/bplus_tree/varint.h"
#include "storage/index/record.h"
#include "system/buffer_manager.h"
#include "system/file_manager.h"

template <std::size_t N>
BPlusTree<N>::BPlusTree(const std::string& name, BPT::LeafFormat leaf_format) :
    dir_file_id  (file_manager.get_file_id(name + ".dir")),
    leaf_file_id (file_manager.get_file_id(name + ".leaf")),
    leaf_format_ (leaf_format) { }


template <std::size_t N>
std::unique_ptr<BPTLeafBase<N>> BPlusTree<N>::open_leaf_page(Page& page,
                                                             BPT::LeafFormat fmt)
{
    switch (fmt) {
        case BPT::LeafFormat::BITSET:
            // V1 owns the Page* pin through its destructor. No byte-0
            // cross-check here: a V1 page with value_count=2 legitimately
            // has byte 0 == 2 (design §6.1 edge case).
            return std::make_unique<BPTLeafV1<N>>(&page);

        case BPT::LeafFormat::DELTA_VARINT: {
            // DELTA_VARINT is only supported for the record widths that have
            // BPTLeafV2<N> explicit template instantiations (N = 1, 2, 3).
            // BPlusTree<4> — the quad-model edge table — never carries
            // DELTA_VARINT in Spec #5 and hitting this branch for N=4 would
            // be a catalog bit-flip; fail loudly instead of link-failing.
            if constexpr (N >= 1 && N <= 3) {
                // Defense-in-depth: catalog says DELTA_VARINT so the page
                // must carry format_version=2 in byte 0. Anything else
                // indicates a cross-index mix-up or mid-migration corruption.
                const auto* bytes = reinterpret_cast<const uint8_t*>(page.get_bytes());
                if (bytes[0] != 2) {
                    throw BPT::BPTLeafV2DecodeException(
                        std::string("leaf-format mismatch: catalog says DELTA_VARINT "
                                    "but page byte 0 is ")
                        + std::to_string(static_cast<unsigned>(bytes[0])));
                }
                return std::make_unique<BPTLeafV2<N>>(
                    page.get_bytes(),
                    typename BPTLeafV2<N>::ReadTag{});
            } else {
                throw std::logic_error(
                    "BPT::LeafFormat::DELTA_VARINT is not supported for this "
                    "record width; only N in {1,2,3} has a BPTLeafV2 reader");
            }
        }
    }
    // Unreachable for valid enum values; guards against bit-pattern
    // corruption of a LeafFormat loaded from disk.
    throw std::logic_error("unknown BPT::LeafFormat enum value");
}


template <std::size_t N>
std::unique_ptr<BPlusTreeDir<N>> BPlusTree<N>::get_root() const noexcept {
    return std::make_unique<BPlusTreeDir<N>>(
        leaf_file_id,
        &buffer_manager.get_page_readonly(dir_file_id, 0)
    );
}


template <std::size_t N>
BptIter<N> BPlusTree<N>::get_range(bool* interruption_requested,
                                   const Record<N>& min,
                                   const Record<N>& max) const noexcept {
    BPlusTreeDir<N> root(
        leaf_file_id,
        &buffer_manager.get_page_readonly(dir_file_id, 0)
    );
    auto leaf_and_pos = root.search_leaf(min);
    return BptIter<N>(interruption_requested, std::move(leaf_and_pos), max, leaf_format_);
}


template <std::size_t N>
bool BPlusTree<N>::insert(const Record<N>& record) {
    // although root can be modified in insert, we start as readonly, and
    // it will create a new version in the insert method if needed
    BPlusTreeDir<N> root(
        leaf_file_id,
        &buffer_manager.get_page_readonly(dir_file_id, 0)
    );
    bool error;
    root.insert(record, error);
    return !error;
}


template <std::size_t N>
bool BPlusTree<N>::delete_record(const Record<N>& record) {
    // although root can be modified in insert, we start as readonly, and
    // it will create a new version in the insert method if needed
    BPlusTreeDir<N> root(
        leaf_file_id,
        &buffer_manager.get_page_readonly(dir_file_id, 0)
    );
    return root.delete_record(record);
}


template <std::size_t N>
bool BPlusTree<N>::check(std::ostream& os) const {
    BPlusTreeDir<N> root(
        leaf_file_id,
        &buffer_manager.get_page_readonly(dir_file_id, 0)
    );
    return root.check(os);
}


uint64_t powi(uint64_t base, size_t exp) {
    uint64_t res = 1;
    while (exp) {
        if (exp & 1)
            res *= base;
        exp >>= 1;
        base *= base;
    }
    return res;
}


template <std::size_t N>
double BPlusTree<N>::estimate_records(const Record<N>& min,
                                      const Record<N>& max) const
{
    BPlusTreeDir<N> root(
        leaf_file_id,
        &buffer_manager.get_page_readonly(dir_file_id, 0)
    );
    return BPlusTree<N>::estimate_records(root, min, max);
}


template <std::size_t N>
double BPlusTree<N>::estimate_records(const BPlusTreeDir<N>& root,
                                      const Record<N>& min,
                                      const Record<N>& max)
{
    std::vector<int> min_idxs;
    std::vector<int> max_idxs;

    root.get_branch_indexes(min, min_idxs);
    root.get_branch_indexes(max, max_idxs);

    assert(min_idxs.size() == max_idxs.size());
    assert(min_idxs.size() > 0);

    double estimated_min = 0;
    double estimated_max = 0;
    size_t current_exponent = max_idxs.size() - 1;
    for (size_t i = 0; i < min_idxs.size(); i++) {
        estimated_min += min_idxs[i] * powi(dir_max_records+1, current_exponent);
        estimated_max += max_idxs[i] * powi(dir_max_records+1, current_exponent);
        current_exponent--;
    }
    return 1 + (estimated_max - estimated_min) * leaf_max_records;
}


/******************************* BptIter ********************************/
template<std::size_t N>
BptIter<N>::BptIter(bool* interruption_requested,
                    SearchLeafResult<N>&& leaf_and_pos,
                    const Record<N>& max,
                    BPT::LeafFormat leaf_format) noexcept :
    interruption_requested(interruption_requested),
    current_pos(leaf_and_pos.result_index),
    max(max),
    // SearchLeafResult<N> is V1-only by design (the directory-layer
    // search_leaf currently constructs BPTLeafV1<N> directly; threading
    // DELTA_VARINT through the directory is T5.10's work). We always move
    // the V1 into the polymorphic unique_ptr here; once the tree's format
    // is DELTA_VARINT, T5.10 will re-open the same page under BPTLeafV2
    // via BPlusTree::open_leaf_page(). Until then, holding a V1 under the
    // base pointer preserves the pre-Spec-#5 behavior byte-for-byte.
    current_leaf_(std::make_unique<BPTLeafV1<N>>(std::move(leaf_and_pos.leaf))),
    leaf_format_(leaf_format)
{
    current_leaf_->set_redundant_record(current_record);
}

template<std::size_t N>
BptIter<N>::BptIter(BptIter&& other) noexcept :
    interruption_requested(other.interruption_requested),
    current_pos(other.current_pos),
    max(std::move(other.max)),
    current_leaf_(std::move(other.current_leaf_)),
    leaf_format_(other.leaf_format_)
{
    if (current_leaf_) {
        current_leaf_->set_redundant_record(current_record);
    }
}

template<std::size_t N>
void BptIter<N>::operator=(BptIter&& other) noexcept {
    interruption_requested = other.interruption_requested;
    current_pos            = other.current_pos;
    max                    = std::move(other.max);
    current_leaf_          = std::move(other.current_leaf_);
    leaf_format_           = other.leaf_format_;
    if (current_leaf_) {
        current_leaf_->set_redundant_record(current_record);
    }
}

template <std::size_t N>
const Record<N>* BptIter<N>::next() {
    while (true) {
        if (MDB_unlikely(*interruption_requested)) {
            throw InterruptedException();
        }
        if (current_pos < current_leaf_->get_value_count()) {
            current_leaf_->update_record(current_pos, current_record);
            // check if res is less than max
            for (size_t i = 0; i < N; ++i) {
                if (current_record[i] < max[i]) {
                    ++current_pos;
                    return &current_record;
                }
                else if (current_record[i] > max[i]) {
                    return nullptr;
                }
                // continue iterating if current_record[i] == max[i]
            }
            ++current_pos;
            return &current_record; // res == max
        }
        else if (current_leaf_->has_next()) {
            // Cross-page transition. For BITSET (V1) we reuse the existing
            // in-place update_to_next_leaf() pathway, which preserves the
            // pre-Spec-#5 page-walking behavior byte-for-byte. V2 pages are
            // immutable post-build (design §2.2 non-goal 6), so V2's
            // update_to_next_leaf throws — T5.10 will replace current_leaf_
            // with a fresh BPTLeafV2 view over the next page via
            // BPlusTree::open_leaf_page(). Until T5.10 wires DELTA_VARINT
            // through the directory, no caller passes that format, so this
            // branch is V1-only in practice.
            current_leaf_->update_to_next_leaf();
            current_pos = 0;
            current_leaf_->set_redundant_record(current_record);
            // continue while
        }
        else {
            return nullptr;
        }
    }
}


template class BPlusTree<1>;
template class BPlusTree<2>;
template class BPlusTree<3>;
template class BPlusTree<4>;

template class BptIter<1>;
template class BptIter<2>;
template class BptIter<3>;
template class BptIter<4>;
