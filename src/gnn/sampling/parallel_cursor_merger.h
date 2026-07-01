#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "graph_models/object_id.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace mdb::gnn {

/**
 * @brief Merged edge from undirected traversal.
 *
 * Contains both the neighbor and edge ID, plus direction flag for debugging.
 */
struct MergedEdge {
    uint64_t neighbor_id;  ///< Connected node
    uint64_t edge_id;      ///< Original edge ID
    bool is_outgoing;      ///< true if original direction is from→to

    bool operator==(const MergedEdge& other) const {
        return neighbor_id == other.neighbor_id && edge_id == other.edge_id;
    }
};

/**
 * @brief Hash for MergedEdge to enable deduplication in unordered_set.
 */
struct MergedEdgeHash {
    size_t operator()(const MergedEdge& e) const {
        // Combine neighbor_id and edge_id for unique identification
        return std::hash<uint64_t>()(e.neighbor_id) ^
               (std::hash<uint64_t>()(e.edge_id) << 1);
    }
};

/**
 * @brief Coordinates dual-index seeks for efficient undirected edge traversal.
 *
 * This class optimizes undirected neighbor collection by:
 * 1. Interleaving seeks on from_to and to_from indexes
 * 2. Processing both directions per node before moving to next
 * 3. Inline deduplication of edges appearing in both directions
 *
 * ## Performance Improvement
 *
 * | Approach | I/O Pattern | Cache Behavior |
 * |----------|-------------|----------------|
 * | 2× Sequential Passes | OOOOO-IIIII | Cold on 2nd pass |
 * | **Parallel Cursors** | OI-OI-OI-OI | Hot (interleaved) |
 *
 * Where O = outgoing edge access, I = incoming edge access.
 *
 * ## Memory Model
 *
 * - Two SeekableEdgeIter instances (one per index)
 * - Deduplication set per node (reset between nodes)
 * - O(max_degree) temporary memory per node
 *
 * ## Usage Example
 *
 * @code
 *   ParallelCursorMerger merger(from_to_index, to_from_index, &interruption);
 *
 *   for (uint64_t node_id : sorted_nodes) {
 *       std::vector<MergedEdge> edges = merger.get_all_edges(node_id);
 *       for (const auto& edge : edges) {
 *           process(node_id, edge.neighbor_id, edge.edge_id);
 *       }
 *   }
 *
 *   // Check statistics
 *   auto [seeks, edges] = merger.get_statistics();
 * @endcode
 *
 * @see SeekBasedGnnSampler for seek-based directed sampling
 * @see LeapfrogGnnSampler::sample_undirected for sweep-based approach
 */
class ParallelCursorMerger {
public:
    /**
     * @brief Construct merger for dual-index undirected traversal.
     *
     * @param from_to_index Reference to from→to edge index
     * @param to_from_index Reference to to→from edge index
     * @param interruption Pointer to interruption flag
     *
     * @note Both indexes must have the same edges (different key order).
     */
    ParallelCursorMerger(
        BPlusTree<3>& from_to_index,
        BPlusTree<3>& to_from_index,
        bool* interruption
    );

    ~ParallelCursorMerger();

    // Non-copyable, moveable
    ParallelCursorMerger(const ParallelCursorMerger&) = delete;
    ParallelCursorMerger& operator=(const ParallelCursorMerger&) = delete;
    ParallelCursorMerger(ParallelCursorMerger&&) noexcept;
    ParallelCursorMerger& operator=(ParallelCursorMerger&&) noexcept;

    /**
     * @brief Get all edges (both directions) for a single node.
     *
     * Performs two seeks (one per index) and merges results with deduplication.
     * Edges are deduplicated by (neighbor_id, edge_id) pair.
     *
     * @param node_id Target node to get edges for
     * @return Vector of unique MergedEdge (outgoing + incoming)
     *
     * @note The same edge may appear in both indexes with opposite directions.
     *       This method deduplicates such edges.
     *
     * ## Complexity
     *
     * - Seeks: 2 × O(log E)
     * - Iteration: O(out_degree + in_degree)
     * - Dedup: O(degree) with hash set
     */
    std::vector<MergedEdge> get_all_edges(uint64_t node_id);

    /**
     * @brief Get all edges for a node with limit (for reservoir sampling integration).
     *
     * If total edges exceed limit, returns all edges for caller to apply
     * reservoir sampling (since we don't know the exact count beforehand).
     *
     * @param node_id Target node
     * @param limit Maximum edges to return (0 = unlimited)
     * @return Vector of unique MergedEdge up to limit
     *
     * @note For reservoir sampling, pass limit=0 and sample externally.
     */
    std::vector<MergedEdge> get_all_edges(uint64_t node_id, size_t limit);

    /**
     * @brief Get statistics for cost model validation.
     *
     * @return Pair of (total_seeks, total_edges_accessed)
     */
    std::pair<uint64_t, uint64_t> get_statistics() const;

    /**
     * @brief Reset statistics counters.
     */
    void reset_statistics();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
