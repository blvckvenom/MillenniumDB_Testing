// src/graph_models/gql/projection/edge_keep_bitmap.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace GQL {

/**
 * @brief Write-once / read-many bit vector keyed by edge_id.
 *
 * Used by the serialized scan pipeline (Spec #2) as the output of Phase B
 * (edge filter pre-computation). After `finalize()`, the bitmap is
 * immutable; `set_kept()` throws std::logic_error. Phase C's 9 edge-index
 * passes consume the bitmap via `is_kept()` without mutating it.
 *
 * Memory: 1 bit per edge_id. For papers100M (1.6B edges): ~200 MB.
 * Grows automatically on out-of-range set_kept.
 *
 * Thread-safety: after finalize(), concurrent is_kept() calls from
 * multiple threads are safe. Pre-finalize, the class is not
 * thread-safe — writers must synchronize externally.
 */
class EdgeKeepBitmap {
public:
    EdgeKeepBitmap() = default;

    /**
     * @brief Ensures capacity for at least @p max_edge_id elements.
     *
     * @note Unlike std::vector::reserve, this grows size() (not just
     *       capacity), zero-initializing new bits. Equivalent to
     *       std::vector::resize(max_edge_id, false). Safe to call
     *       before any set_kept() to avoid repeated resizes when the
     *       max edge_id is known.
     */
    void reserve(std::size_t max_edge_id) {
        if (max_edge_id > kept_.size()) kept_.resize(max_edge_id);
    }

    void set_kept(std::uint64_t edge_id) {
        if (finalized_) {
            throw std::logic_error("EdgeKeepBitmap: set_kept after finalize()");
        }
        if (edge_id >= kept_.size()) {
            kept_.resize(edge_id + 1, false);
        }
        kept_[edge_id] = true;
    }

    void finalize() noexcept { finalized_ = true; }

    bool is_kept(std::uint64_t edge_id) const noexcept {
        return edge_id < kept_.size() && kept_[edge_id];
    }

    std::size_t bytes_allocated() const noexcept {
        return (kept_.size() + 7) / 8;  // std::vector<bool> packs at ~1 bit per entry
    }

    std::size_t size() const noexcept { return kept_.size(); }

private:
    std::vector<bool> kept_;
    bool finalized_ = false;
};

}  // namespace GQL
