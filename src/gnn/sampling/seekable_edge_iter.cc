#include "gnn/sampling/seekable_edge_iter.h"

#include <optional>

namespace mdb::gnn {

/**
 * @brief Implementation of SeekableEdgeIter using B+Tree range iteration.
 *
 * The implementation uses BPlusTree<3>::get_range() for seeks and maintains
 * a BptIter<3> for sequential access. Each seek operation creates a new
 * range iterator positioned at the target.
 *
 * ## Memory Model
 *
 * - Holds reference to BPlusTree (does not own)
 * - BptIter holds leaf page references (managed by buffer pool)
 * - Current record cached in EdgeRecord struct
 *
 * ## Thread Safety
 *
 * - NOT thread-safe (holds mutable cursor state)
 * - Multiple iterators can safely share same BPlusTree (read-only access)
 */
struct SeekableEdgeIter::Impl {
    BPlusTree<3>& index;
    bool* interruption;

    /// Current range iterator (recreated on each seek)
    std::optional<BptIter<3>> iter;

    /// Cached current record (valid when !is_exhausted)
    EdgeRecord current_record;

    /// Flag indicating no more records
    bool is_exhausted = true;

    /// Statistics
    uint64_t seek_count = 0;
    uint64_t edge_count = 0;

    Impl(BPlusTree<3>& index_, bool* interruption_)
        : index(index_)
        , interruption(interruption_)
        , current_record{0, 0, 0}
    {}

    /**
     * @brief Seek to first edge with from_id >= target.
     *
     * Creates a range [target, 0, 0] to [MAX, MAX, MAX] and gets first match.
     *
     * @param target_from Minimum source node ID
     * @return true if found an edge, false if exhausted
     */
    bool do_seek_from(uint64_t target_from) {
        seek_count++;

        // Range: [target_from, 0, 0] to [MAX, MAX, MAX]
        Record<3> min_key = { target_from, 0, 0 };
        Record<3> max_key = { UINT64_MAX, UINT64_MAX, UINT64_MAX };

        iter = index.get_range(interruption, min_key, max_key);

        // Try to get first record
        if (const Record<3>* record = iter->next()) {
            current_record.from_id = (*record)[0];
            current_record.to_id = (*record)[1];
            current_record.edge_id = (*record)[2];
            is_exhausted = false;
            edge_count++;
            return true;
        }

        is_exhausted = true;
        return false;
    }

    /**
     * @brief Seek to edge with (from_id, to_id) >= (target_from, target_to).
     */
    bool do_seek_from_to(uint64_t target_from, uint64_t target_to) {
        seek_count++;

        Record<3> min_key = { target_from, target_to, 0 };
        Record<3> max_key = { UINT64_MAX, UINT64_MAX, UINT64_MAX };

        iter = index.get_range(interruption, min_key, max_key);

        if (const Record<3>* record = iter->next()) {
            current_record.from_id = (*record)[0];
            current_record.to_id = (*record)[1];
            current_record.edge_id = (*record)[2];
            is_exhausted = false;
            edge_count++;
            return true;
        }

        is_exhausted = true;
        return false;
    }

    /**
     * @brief Move to next edge in current range.
     *
     * @return true if advanced, false if exhausted
     */
    bool do_next() {
        if (is_exhausted || !iter.has_value()) {
            return false;
        }

        if (const Record<3>* record = iter->next()) {
            current_record.from_id = (*record)[0];
            current_record.to_id = (*record)[1];
            current_record.edge_id = (*record)[2];
            edge_count++;
            return true;
        }

        is_exhausted = true;
        return false;
    }

    /**
     * @brief Move to next edge from same source node.
     *
     * Advances only if next edge has same from_id as current.
     *
     * @return true if advanced to edge from same source
     * @return false if next edge has different source or exhausted
     */
    bool do_next_from_current() {
        if (is_exhausted || !iter.has_value()) {
            return false;
        }

        uint64_t expected_from = current_record.from_id;

        if (const Record<3>* record = iter->next()) {
            uint64_t from_id = (*record)[0];

            if (from_id == expected_from) {
                current_record.from_id = from_id;
                current_record.to_id = (*record)[1];
                current_record.edge_id = (*record)[2];
                edge_count++;
                return true;
            }

            // Different source - "unread" this record by caching it
            // but mark as having moved past current source
            current_record.from_id = from_id;
            current_record.to_id = (*record)[1];
            current_record.edge_id = (*record)[2];
            return false;
        }

        is_exhausted = true;
        return false;
    }

    void do_reset() {
        iter.reset();
        is_exhausted = true;
        current_record = { 0, 0, 0 };
        seek_count = 0;
        edge_count = 0;
    }
};

// =============================================================================
// Public Interface
// =============================================================================

SeekableEdgeIter::SeekableEdgeIter(BPlusTree<3>& index, bool* interruption)
    : impl_(std::make_unique<Impl>(index, interruption))
{}

SeekableEdgeIter::~SeekableEdgeIter() = default;

SeekableEdgeIter::SeekableEdgeIter(SeekableEdgeIter&&) noexcept = default;
SeekableEdgeIter& SeekableEdgeIter::operator=(SeekableEdgeIter&&) noexcept = default;

bool SeekableEdgeIter::seek_from(uint64_t target_from) {
    return impl_->do_seek_from(target_from);
}

bool SeekableEdgeIter::seek_from_to(uint64_t target_from, uint64_t target_to) {
    return impl_->do_seek_from_to(target_from, target_to);
}

bool SeekableEdgeIter::next_from_current() {
    return impl_->do_next_from_current();
}

bool SeekableEdgeIter::next() {
    return impl_->do_next();
}

EdgeRecord SeekableEdgeIter::current() const {
    return impl_->current_record;
}

bool SeekableEdgeIter::exhausted() const {
    return impl_->is_exhausted;
}

void SeekableEdgeIter::reset() {
    impl_->do_reset();
}

uint64_t SeekableEdgeIter::seeks_performed() const {
    return impl_->seek_count;
}

uint64_t SeekableEdgeIter::edges_iterated() const {
    return impl_->edge_count;
}

} // namespace mdb::gnn
