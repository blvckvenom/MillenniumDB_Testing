#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gnn/projection/adj_entry.h"

namespace mdb::gnn {

/**
 * @brief Tier-1 (RAM hot) adjacency cache for the Four-Level Topology
 *        Store (Spec #13 Phase 2 / T13.4).
 *
 * Lifts the in-memory `unordered_map<uint64_t, vector<AdjEntry>>` that
 * Spec #11 (`TopologyAccessor::Impl::fwd_cache_` / `rev_cache_`) builds
 * unconditionally for ALL nodes into a free-standing class that only
 * accepts inserts for nodes flagged tier 1 by the
 * `TopologyFrequencyProfiler`'s `compute_tier_assignment()` output.
 * Nodes flagged tier 2/3/4 are silently no-op'd at insert time so that
 * the upper-tier dispatch in `FourLevelTopologyStore` stays uniform
 * (every cache "tries" to insert; only the right one keeps the data).
 *
 * The cache is **build-time mutable, runtime read-only** (Spec #13
 * design D7). Callers populate it once during `build()`, then issue
 * `get()` queries from any number of sampler threads. There is no
 * concurrent `insert()` after the build completes — enforcing that is
 * the orchestrator's responsibility.
 *
 * Memory accounting consumes the `kL1*` constants from
 * `topology_frequency_profiler.h` (Spec #13 Phase 1 contract) so the
 * profiler's tier-sizing math and the runtime's `total_bytes()`
 * diagnostic stay in lockstep.
 */
class L1HashCache {
public:
    /**
     * @brief Lookup result: pointer + length view into the cache's
     *        per-node neighbor vector.
     *
     * Mirrors the `ConstU64Span`-style return shape used by
     * `TopologySnapshotReader::neighbors()` (Spec #4-B), but typed for
     * `AdjEntry`. The `data` pointer is owned by the cache and remains
     * valid until the cache is destroyed.
     */
    struct Span {
        const AdjEntry* data = nullptr;
        std::size_t     size = 0;

        bool empty() const noexcept { return size == 0; }
    };

    /**
     * @brief Construct the cache bound to a tier-assignment vector.
     *
     * @param tier_assignment Per-row tier IDs as produced by
     *                        `compute_tier_assignment()`. Indexed by row
     *                        index in the underlying topology
     *                        (`NodeIterator` order). Held by const
     *                        reference; caller must keep it alive at
     *                        least until the last `insert()`.
     */
    explicit L1HashCache(const std::vector<uint8_t>& tier_assignment);

    L1HashCache(const L1HashCache&)            = delete;
    L1HashCache& operator=(const L1HashCache&) = delete;
    L1HashCache(L1HashCache&&) noexcept        = default;
    L1HashCache& operator=(L1HashCache&&) noexcept = default;

    /**
     * @brief Insert a node's adjacency list into the cache.
     *
     * Silently no-ops when `tier_assignment[row_idx] != 1` (or when
     * `row_idx` is past the end of the assignment vector). The
     * orchestrator may therefore call `insert()` for every node it
     * scans without first filtering by tier.
     *
     * @param src_node_id Raw `ObjectId.id` of the source node.
     * @param neighbors   Adjacency list (taken by value so the caller
     *                    can move into the cache).
     * @param row_idx     Row index of `src_node_id` in the underlying
     *                    topology iteration order. Used to consult
     *                    `tier_assignment` only.
     */
    void insert(uint64_t                 src_node_id,
                std::vector<AdjEntry>    neighbors,
                std::size_t              row_idx);

    /**
     * @brief Look up the cached neighbors for a source node.
     *
     * @return A `Span` over the cache-owned vector. `Span::empty()`
     *         is true on miss (node was not inserted, e.g. wrong tier
     *         or never seen).
     */
    Span get(uint64_t src_node_id) const;

    /// Convenience predicate for tests / diagnostics.
    bool contains(uint64_t src_node_id) const;

    /// Number of source nodes currently stored.
    std::size_t node_count() const noexcept { return entries_.size(); }

    /**
     * @brief Approximate resident-byte cost using the Phase 1 contract.
     *
     * Computed on-the-fly (not memoized) because Phase 2 builds the
     * cache once and freezes it; the call is rare and the cost is
     * O(node_count). Sums `kL1NodeFixedOverhead + kL1PerEdgeBytes *
     * degree` per inserted node.
     */
    std::size_t total_bytes() const;

    /// Total directed edge entries across all stored nodes.
    std::size_t total_edges() const;

private:
    const std::vector<uint8_t>& tier_assignment_;
    std::unordered_map<uint64_t, std::vector<AdjEntry>> entries_;
};

}  // namespace mdb::gnn
