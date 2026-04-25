#include "bplus_tree.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

#include "macros/likely.h"
#include "query/exceptions.h"
#include "storage/index/bplus_tree/bplus_tree_leaf.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_base.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_csr.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"
#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
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

        case BPT::LeafFormat::CSR_HYBRID: {
            // CSR_HYBRID (Spec #8) is an edge-index-only encoding
            // instantiated for N in {2, 3}. Per design §3.6 D6 the hybrid
            // format is selected only for FROM_TO_EDGE / TO_FROM_EDGE
            // topology indexes; BPlusTree<1> / BPlusTree<4> never carries
            // it, so hitting this branch outside [2..3] is a catalog
            // bit-flip and we fail loudly rather than link-failing.
            if constexpr (N >= 2 && N <= 3) {
                const auto* bytes = reinterpret_cast<const uint8_t*>(page.get_bytes());
                // Cross-check 1: catalog says CSR_HYBRID so byte 0 must be 3.
                if (bytes[0] != 3) {
                    throw BPT::BPTLeafV2DecodeException(
                        std::string("leaf-format mismatch: catalog says CSR_HYBRID "
                                    "but page byte 0 is ")
                        + std::to_string(static_cast<unsigned>(bytes[0])));
                }
                // Cross-check 2: a directory-routed open must land on a
                // chain-head page (flags bit 0 clear). Continuation pages
                // are reached only via chain-head traversal; if the
                // directory points at a continuation, the catalog / builder
                // state is inconsistent.
                if ((bytes[2] & BPT::CSRHybridFlags::kIsContinuation) != 0) {
                    throw BPT::BPTLeafV2DecodeException(
                        "leaf-format mismatch: catalog says CSR_HYBRID but "
                        "page is a continuation (flags bit 0 set), not a "
                        "chain-head — directory points to wrong page");
                }
                return std::make_unique<BPTLeafCSR<N>>(
                    page.get_bytes(),
                    typename BPTLeafCSR<N>::ReadTag{});
            } else {
                throw std::logic_error(
                    "BPT::LeafFormat::CSR_HYBRID is not supported for this "
                    "record width; only N in {2,3} has a BPTLeafCSR reader");
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
    return BptIter<N>(
        interruption_requested, std::move(leaf_and_pos), max, leaf_format_,
        /*min=*/ &min);
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

// Rebuild `v2_reader_` over the page currently pinned by `current_leaf_`
// (a BPTLeafV1<N> held only for its Page* pin). Used at ctor time and on
// every `update_to_next_leaf` transition when leaf_format_ == DELTA_VARINT.
// The V1 instance's `get_page()` accessor is public and non-mutating; we
// view the same buffer under BPTLeafV2<N>'s ReadTag decoder so the
// on-disk bytes are interpreted per Spec #5 §5.2 (16-byte header +
// zigzag-delta varint payload). The V2 reader owns nothing about the
// page pin — it holds a raw `const char*` into the pinned page bytes,
// valid only as long as `current_leaf_` stays alive.
template <std::size_t N>
static std::unique_ptr<BPTLeafBase<N>> open_v2_reader_over_pinned_page_(
    BPTLeafBase<N>* current_leaf_v1)
{
    if constexpr (N >= 1 && N <= 3) {
        auto* v1 = dynamic_cast<BPTLeafV1<N>*>(current_leaf_v1);
        if (v1 == nullptr) {
            // The pin-holder slot is not a V1. This is a programming error
            // — the DELTA_VARINT path always populates current_leaf_ with
            // a V1 at ctor / update_to_next_leaf time. Return nullptr; the
            // caller will surface a NO_MATCH path to avoid corruption.
            return nullptr;
        }
        return std::make_unique<BPTLeafV2<N>>(
            v1->get_page().get_bytes(),
            typename BPTLeafV2<N>::ReadTag{});
    } else {
        // N outside [1..3] has no BPTLeafV2<N> instantiation (quad-model
        // Record<4> never opts into DELTA_VARINT).
        return nullptr;
    }
}

// Rebuild `v3_reader_` over the page currently pinned by `current_leaf_`
// (a BPTLeafV1<N> held only for its Page* pin). Used at ctor time and on
// every cross-page transition when leaf_format_ == CSR_HYBRID. The V3
// reader validates the 16-byte v3 header at construction per Spec #8
// §5.5; any failure here raises BPTLeafCSRDecodeException (propagated
// to the caller). The v3 reader owns nothing about the page pin — it
// holds a raw `const char*` into the pinned page bytes, valid only as
// long as `current_leaf_` stays alive.
template <std::size_t N>
static std::unique_ptr<BPTLeafBase<N>> open_v3_reader_over_pinned_page_(
    BPTLeafBase<N>* current_leaf_v1)
{
    if constexpr (N >= 2 && N <= 3) {
        auto* v1 = dynamic_cast<BPTLeafV1<N>*>(current_leaf_v1);
        if (v1 == nullptr) {
            // The pin-holder slot is not a V1. The CSR_HYBRID path always
            // populates current_leaf_ with a V1 at ctor / cross-page
            // transition time; reaching here indicates a programming
            // error. Return nullptr; the caller surfaces a NO_MATCH path
            // to avoid corruption.
            return nullptr;
        }
        return std::make_unique<BPTLeafCSR<N>>(
            v1->get_page().get_bytes(),
            typename BPTLeafCSR<N>::ReadTag{});
    } else {
        // N outside [2..3] has no BPTLeafCSR<N> instantiation per design
        // §3.6 D6 (CSR_HYBRID is edge-index-only).
        return nullptr;
    }
}

template<std::size_t N>
BptIter<N>::BptIter(bool* interruption_requested,
                    SearchLeafResult<N>&& leaf_and_pos,
                    const Record<N>& max,
                    BPT::LeafFormat leaf_format,
                    const Record<N>* min) noexcept :
    interruption_requested(interruption_requested),
    current_pos(leaf_and_pos.result_index),
    max(max),
    // SearchLeafResult<N> always carries a BPTLeafV1<N>. The directory's
    // search_leaf is V1-oriented because dir pages store raw uint64 keys
    // and compare against V1-decoded leaf keys; the selected leaf page
    // itself is then interpreted per leaf_format_ (V1 bytes for BITSET,
    // V2 bytes under a re-view for DELTA_VARINT). We always move the V1
    // in here — it doubles as the BufferManager pin holder under v2.
    current_leaf_(std::make_unique<BPTLeafV1<N>>(std::move(leaf_and_pos.leaf))),
    v2_reader_(nullptr),
    v3_reader_(nullptr),
    leaf_format_(leaf_format)
{
    if (leaf_format_ == BPT::LeafFormat::DELTA_VARINT) {
        // Re-interpret the pinned page under a v2 decoder. The V1 held in
        // `current_leaf_` stays alive purely as a pin; every get_*/update_*/
        // has_next call in next() dispatches to v2_reader_ below.
        v2_reader_ = open_v2_reader_over_pinned_page_<N>(current_leaf_.get());
        // Position the cursor inside the v2 page via v2-aware search.
        // The V1 `result_index` we inherited from the directory layer
        // was computed against V1 byte offsets and is meaningless for V2
        // positional addressing, so we recompute with V2's search_index
        // against the caller-supplied `min`. If `min` is absent (callers
        // that don't pass one — currently none under DELTA_VARINT, but
        // belt-and-suspenders), we fall back to 0 and let the `max`
        // filter in next() do the work.
        if (v2_reader_ && min != nullptr) {
            current_pos = v2_reader_->search_index(*min);
        } else {
            current_pos = 0;
        }
    } else if (leaf_format_ == BPT::LeafFormat::CSR_HYBRID) {
        // Re-interpret the pinned page under a v3 decoder. The V1 held
        // in `current_leaf_` stays alive purely as a pin; all reads
        // dispatch to v3_reader_ below. Construction validates the v3
        // header per Spec #8 §5.5 and rejects continuation pages
        // (flags bit 0 set) — a directory that routed us to a
        // continuation page is a catalog inconsistency, surfaced here
        // via BPTLeafCSRDecodeException.
        v3_reader_ = open_v3_reader_over_pinned_page_<N>(current_leaf_.get());
        // V3's search_index walks the logical tuple stream; same
        // reasoning as V2 applies for why the directory's V1 result_index
        // cannot be reused.
        if (v3_reader_ && min != nullptr) {
            current_pos = v3_reader_->search_index(*min);
        } else {
            current_pos = 0;
        }
    } else {
        // BITSET mode: set_redundant_record populates the running record
        // with the V1 redundant-byte prefix. V2 / V3 have no redundant
        // concept.
        current_leaf_->set_redundant_record(current_record);
    }
}

template<std::size_t N>
BptIter<N>::BptIter(BptIter&& other) noexcept :
    interruption_requested(other.interruption_requested),
    current_pos(other.current_pos),
    max(std::move(other.max)),
    current_leaf_(std::move(other.current_leaf_)),
    v2_reader_(std::move(other.v2_reader_)),
    v3_reader_(std::move(other.v3_reader_)),
    leaf_format_(other.leaf_format_)
{
    if (current_leaf_ && leaf_format_ == BPT::LeafFormat::BITSET) {
        current_leaf_->set_redundant_record(current_record);
    }
}

template<std::size_t N>
void BptIter<N>::operator=(BptIter&& other) noexcept {
    interruption_requested = other.interruption_requested;
    current_pos            = other.current_pos;
    max                    = std::move(other.max);
    current_leaf_          = std::move(other.current_leaf_);
    v2_reader_             = std::move(other.v2_reader_);
    v3_reader_             = std::move(other.v3_reader_);
    leaf_format_           = other.leaf_format_;
    if (current_leaf_ && leaf_format_ == BPT::LeafFormat::BITSET) {
        current_leaf_->set_redundant_record(current_record);
    }
}

template <std::size_t N>
const Record<N>* BptIter<N>::next() {
    // Pick the effective reader for this format. v2_reader_ is populated
    // only in DELTA_VARINT mode; v3_reader_ only in CSR_HYBRID mode.
    // current_leaf_ stays alive under V2/V3 as the BufferManager pin
    // holder.
    BPTLeafBase<N>* reader;
    if (leaf_format_ == BPT::LeafFormat::DELTA_VARINT && v2_reader_) {
        reader = v2_reader_.get();
    } else if (leaf_format_ == BPT::LeafFormat::CSR_HYBRID && v3_reader_) {
        reader = v3_reader_.get();
    } else {
        reader = current_leaf_.get();
    }

    while (true) {
        if (MDB_unlikely(*interruption_requested)) {
            throw InterruptedException();
        }
        if (current_pos < reader->get_value_count()) {
            reader->update_record(current_pos, current_record);
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
        else if (reader->has_next()) {
            // Cross-page transition. For BITSET (V1) we reuse the existing
            // in-place update_to_next_leaf() pathway on current_leaf_,
            // which preserves the pre-Spec-#5 page-walking behavior
            // byte-for-byte. For DELTA_VARINT we advance current_leaf_
            // (the V1 pin-holder) to the next page — V1 reads the v2
            // header's `next_leaf` field from offset 8..11, which happens
            // to land on bytes V1 treats as `next_leaf` too (V1's
            // next_leaf is at offset 4..7, so a V1 view of a v2 page
            // sees the v2 value_count as its next_leaf; *) — wait, that
            // would be wrong. Instead, read next_leaf from v2_reader_
            // and transition manually via the buffer_manager. See the
            // DELTA_VARINT branch below.
            if (leaf_format_ == BPT::LeafFormat::DELTA_VARINT) {
                // Cast current_leaf_ to V1 to access its page + file id,
                // unpin the current page, pin the next page, rebuild
                // current_leaf_ as a fresh V1 view (pin holder), and
                // rebuild v2_reader_ over the new page bytes.
                if constexpr (N >= 1 && N <= 3) {
                    auto* v1 = dynamic_cast<BPTLeafV1<N>*>(current_leaf_.get());
                    if (v1 == nullptr) return nullptr;  // defensive
                    // Read next-leaf page number from the v2 header
                    // (bytes 8..11, little-endian). We have a ready-made
                    // v2_reader_ whose has_next returned true, so the
                    // underlying header's next_leaf is non-zero.
                    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(
                        v1->get_page().get_bytes());
                    uint32_t next_page_number = 0;
                    std::memcpy(&next_page_number, bytes + 8, sizeof(uint32_t));
                    const FileId leaf_file_id = v1->get_page().page_id.file_id;
                    // Drop the v2 view first, then the v1 pin holder.
                    v2_reader_.reset();
                    current_leaf_.reset();  // V1 dtor unpins the old page
                    Page& new_page = buffer_manager.get_page_readonly(
                        leaf_file_id, next_page_number);
                    current_leaf_ = std::make_unique<BPTLeafV1<N>>(&new_page);
                    v2_reader_ = open_v2_reader_over_pinned_page_<N>(
                        current_leaf_.get());
                    reader = v2_reader_ ? v2_reader_.get() : current_leaf_.get();
                    current_pos = 0;
                    continue;
                } else {
                    return nullptr;  // no V2 for N outside [1..3]
                }
            }
            if (leaf_format_ == BPT::LeafFormat::CSR_HYBRID) {
                // Cross-page transition for v3 pages (Spec #8 / T8-B.1).
                //
                // Two sub-cases depending on the new page's flags byte:
                //   (a) Chain-head page (flags bit 0 clear): open via
                //       BPTLeafCSR<N>::ReadTag, as before.
                //   (b) Continuation page (flags bit 0 set): open via
                //       BPTLeafCSR<N>::ContinuationTag, carrying over the
                //       owning src_id and last dst of the CURRENT page so
                //       the hub's spilled adjacency is iterable end-to-end.
                //
                // Before T8-B.1 this branch unconditionally used ReadTag
                // which rejected continuation pages, causing hub sampling
                // on arxiv-scale projections to fail mid-iteration.
                if constexpr (N >= 2 && N <= 3) {
                    auto* v1 = dynamic_cast<BPTLeafV1<N>*>(current_leaf_.get());
                    if (v1 == nullptr) return nullptr;  // defensive

                    // Carry state from the CURRENT page before we drop it.
                    // If the current page returned at least one tuple, the
                    // BptIter has already seen a record — use the last
                    // emitted record's src/dst as the carry-over for a
                    // continuation page. If the current page was empty
                    // (value_count=0), no continuation can legitimately
                    // follow; fall back to (0, 0) — writer invariant
                    // guarantees an empty chain-head has no continuations.
                    //
                    // Spec #8-B task #1: also carry the previous eid so a
                    // continuation page that advertises kHasEdgeIds can
                    // delta-decode its eid stream. current_record[2]
                    // already holds the last emitted eid (real value when
                    // the page advertised eids, 0 otherwise — the carry
                    // is consulted by the continuation reader only when
                    // its own header bit is set).
                    uint64_t carry_src_id   = 0;
                    uint64_t carry_prev_dst = 0;
                    uint64_t carry_prev_eid = 0;
                    if (reader->get_value_count() > 0) {
                        carry_src_id   = current_record[0];
                        carry_prev_dst = current_record[1];
                        if constexpr (N >= 3) {
                            carry_prev_eid = current_record[2];
                        }
                    }

                    // Read next-leaf page number from the v3 header
                    // (bytes 8..11, little-endian) — same slot across
                    // chain-head and continuation variants.
                    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(
                        v1->get_page().get_bytes());
                    uint32_t next_page_number = 0;
                    std::memcpy(&next_page_number, bytes + 8, sizeof(uint32_t));
                    const FileId leaf_file_id = v1->get_page().page_id.file_id;

                    // Drop the v3 view first, then the v1 pin holder.
                    v3_reader_.reset();
                    current_leaf_.reset();  // V1 dtor unpins the old page
                    Page& new_page = buffer_manager.get_page_readonly(
                        leaf_file_id, next_page_number);
                    current_leaf_ = std::make_unique<BPTLeafV1<N>>(&new_page);

                    // Sniff the new page's continuation flag. The 16-byte
                    // header layout guarantees flags is at byte 2 regardless
                    // of chain-head vs continuation variant.
                    const uint8_t* new_bytes = reinterpret_cast<const uint8_t*>(
                        new_page.get_bytes());
                    const bool is_continuation =
                        (new_bytes[2] & BPT::CSRHybridFlags::kIsContinuation) != 0;

                    if (is_continuation) {
                        try {
                            v3_reader_ = std::make_unique<BPTLeafCSR<N>>(
                                new_page.get_bytes(),
                                typename BPTLeafCSR<N>::ContinuationTag{
                                    carry_src_id, carry_prev_dst,
                                    carry_prev_eid});
                        } catch (const BPT::BPTLeafCSRDecodeException&) {
                            return nullptr;  // corrupt continuation; stop
                        }
                    } else {
                        v3_reader_ = open_v3_reader_over_pinned_page_<N>(
                            current_leaf_.get());
                    }
                    reader = v3_reader_ ? v3_reader_.get() : current_leaf_.get();
                    current_pos = 0;
                    continue;
                } else {
                    return nullptr;  // no V3 for N outside [2..3]
                }
            }
            // BITSET path preserved byte-for-byte below.
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
