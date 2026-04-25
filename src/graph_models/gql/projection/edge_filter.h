// src/graph_models/gql/projection/edge_filter.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "graph_models/gql/projection/edge_keep_bitmap.h"
#include "graph_models/object_id.h"

namespace GQL {

/**
 * @brief Write-once / read-many bit-set keyed by the EDGE COUNTER (not the
 *        full ObjectId). Holds two independent EdgeKeepBitmap instances
 *        (directed / undirected) because GQL edge counters are per-orientation
 *        dense starting from 0 — so a single combined bitmap would collide.
 *
 * Used by the projection's serialized edge-scan pipeline (Phase B producer,
 * Phase C consumers). The caller passes raw ObjectIds; EdgeFilter strips
 * the top 8 type bits (via ObjectId::VALUE_MASK) and routes by the top-byte
 * type tag (MASK_UNDIRECTED_EDGE vs MASK_DIRECTED_EDGE from object_id.h).
 *
 * Memory: ~1 bit per kept edge counter, summed across orientations.
 * Papers100M (1.6B CITES directed): ~200 MB in the directed bitmap, zero
 * in the undirected bitmap.
 *
 * Thread-safety: after finalize(), concurrent is_kept() calls from multiple
 * threads are safe. Pre-finalize the class is not thread-safe.
 */
class EdgeFilter {
public:
    EdgeFilter() = default;

    /// Reserve capacity for directed-edge counters up to max_counter.
    void reserve_directed(std::size_t max_counter) {
        directed_.reserve(max_counter);
    }

    /// Reserve capacity for undirected-edge counters up to max_counter.
    void reserve_undirected(std::size_t max_counter) {
        undirected_.reserve(max_counter);
    }

    /// Marks an edge as kept. Auto-grows the appropriate bitmap.
    /// Routes by the ObjectId's top-byte type tag. Throws std::logic_error
    /// if called after finalize().
    void set_kept(ObjectId edge_id) {
        const uint64_t counter = counter_of_(edge_id);
        if (is_undirected_(edge_id)) {
            undirected_.set_kept(counter);
        } else {
            directed_.set_kept(counter);
        }
    }

    void finalize() noexcept {
        directed_.finalize();
        undirected_.finalize();
    }

    bool is_kept(ObjectId edge_id) const noexcept {
        const uint64_t counter = counter_of_(edge_id);
        return is_undirected_(edge_id)
            ? undirected_.is_kept(counter)
            : directed_.is_kept(counter);
    }

    std::size_t bytes_allocated() const noexcept {
        return directed_.bytes_allocated() + undirected_.bytes_allocated();
    }

private:
    EdgeKeepBitmap directed_;
    EdgeKeepBitmap undirected_;

    // Strip the 8-bit type prefix, keep the 56-bit counter.
    // ObjectId::VALUE_MASK = 0x00'FFFFFFFFFFFFFFUL (from object_id.h).
    static constexpr uint64_t counter_of_(ObjectId id) noexcept {
        return id.id & ObjectId::VALUE_MASK;
    }

    // Edge ObjectId top-byte: MASK_UNDIRECTED_EDGE = 0xE4'00000000000000UL
    //                         MASK_DIRECTED_EDGE   = 0xE0'00000000000000UL
    // Both constants are defined in object_id.h. We compare the full
    // top-byte (via SUB_TYPE_MASK) against the undirected constant; anything
    // else is treated as directed.
    static constexpr bool is_undirected_(ObjectId id) noexcept {
        return (id.id & ObjectId::SUB_TYPE_MASK) == ObjectId::MASK_UNDIRECTED_EDGE;
    }
};

}  // namespace GQL
