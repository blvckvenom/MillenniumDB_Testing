#pragma once

#include <cstdint>
#include <memory>

#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"

namespace mdb::gnn {

/**
 * @brief Edge record from B+Tree index.
 *
 * Represents a single edge with its source, destination, and unique ID.
 * The record format matches BPlusTree<3> key structure.
 */
struct EdgeRecord {
    uint64_t from_id;   ///< Source node ID
    uint64_t to_id;     ///< Destination node ID
    uint64_t edge_id;   ///< Unique edge identifier

    bool operator==(const EdgeRecord& other) const {
        return from_id == other.from_id &&
               to_id == other.to_id &&
               edge_id == other.edge_id;
    }
};

/**
 * @brief Simplified seek interface for B+Tree edge index traversal.
 *
 * This wrapper provides an O(log E) seek capability for GNN neighbor sampling,
 * specifically designed for batch processing where sorted node IDs can be
 * processed with logarithmic jumps through the index.
 *
 * ## Design Rationale
 *
 * `LeapfrogBptIter` is designed for multi-way joins (Leapfrog Triejoin) requiring
 * ScanRange/VarId vectors for intersection operations. GNN sampling only needs
 * single-relation seeks, so this wrapper provides a simplified API:
 *
 * | LeapfrogBptIter | SeekableEdgeIter |
 * |-----------------|------------------|
 * | Multi-way joins | Single relation  |
 * | Level management | No levels       |
 * | ScanRange vectors | Direct seeks   |
 * | Intersection vars | No bindings    |
 *
 * ## Performance
 *
 * - `seek_from(target)`: O(log E) B+Tree traversal
 * - `next_from_current()`: O(1) sequential iteration
 * - `next()`: O(1) sequential iteration (any source)
 *
 * Where E = total edges in index.
 *
 * ## Usage Example
 *
 * @code
 *   // Seek-based batch processing
 *   SeekableEdgeIter iter(from_to_index, &interruption);
 *   std::vector<uint64_t> sorted_nodes = {100, 500, 1000, 2000};
 *
 *   for (uint64_t target_node : sorted_nodes) {
 *       if (!iter.seek_from(target_node)) {
 *           continue;  // No edges from this node
 *       }
 *
 *       // Collect all edges from target_node
 *       do {
 *           EdgeRecord edge = iter.current();
 *           if (edge.from_id != target_node) break;  // Moved to next source
 *           process_edge(edge);
 *       } while (iter.next_from_current());
 *   }
 * @endcode
 *
 * ## Comparison with Coordinated Sweep
 *
 * | Approach | Sparse Batches | Dense Batches |
 * |----------|----------------|---------------|
 * | Sweep    | O(E_range)     | O(E_range)    |
 * | **Seek** | O(B × log E)   | O(B × log E)  |
 *
 * Seek is better when batch density (B / E_range) is low.
 *
 * @see LeapfrogGnnSampler for the coordinated sweep approach
 * @see SeekBasedGnnSampler for seek-based batch sampling
 */
class SeekableEdgeIter {
public:
    /**
     * @brief Construct seekable iterator over edge index.
     *
     * @param index Reference to B+Tree<3> edge index (must outlive iterator)
     *              Index format: [from_id, to_id, edge_id]
     * @param interruption Pointer to interruption flag (for query cancellation)
     *
     * @note The index is NOT modified - this is a read-only iterator.
     */
    SeekableEdgeIter(BPlusTree<3>& index, bool* interruption);

    ~SeekableEdgeIter();

    // Disable copy (holds cursor state)
    SeekableEdgeIter(const SeekableEdgeIter&) = delete;
    SeekableEdgeIter& operator=(const SeekableEdgeIter&) = delete;

    // Allow move
    SeekableEdgeIter(SeekableEdgeIter&&) noexcept;
    SeekableEdgeIter& operator=(SeekableEdgeIter&&) noexcept;

    // =========================================================================
    // Seek Operations - O(log E)
    // =========================================================================

    /**
     * @brief Seek to first edge with from_id >= target_from.
     *
     * This is the core O(log E) operation that enables efficient sparse batch
     * processing. After seek, current() returns the found edge (if any).
     *
     * @param target_from Target source node ID to seek to
     * @return true if found an edge with from_id >= target_from
     * @return false if no such edge exists (iterator exhausted)
     *
     * @note If found edge has from_id > target_from, the target node has no
     *       outgoing edges. Caller should check current().from_id.
     *
     * ## Example: Processing sparse batch
     * @code
     *   for (uint64_t node : sorted_nodes) {
     *       if (!iter.seek_from(node)) break;  // Index exhausted
     *       if (iter.current().from_id != node) continue;  // No edges
     *       // Process edges from this node...
     *   }
     * @endcode
     */
    bool seek_from(uint64_t target_from);

    /**
     * @brief Seek to specific edge position.
     *
     * Finds edge with (from_id, to_id) >= (target_from, target_to).
     * Useful for resuming iteration or seeking within a source node's edges.
     *
     * @param target_from Source node ID
     * @param target_to Minimum destination node ID
     * @return true if found matching edge, false if exhausted
     */
    bool seek_from_to(uint64_t target_from, uint64_t target_to);

    // =========================================================================
    // Sequential Operations - O(1)
    // =========================================================================

    /**
     * @brief Move to next edge from the SAME source node.
     *
     * Advances to the next edge where from_id == current().from_id.
     * Returns false if the next edge has a different source (or index exhausted).
     *
     * @return true if advanced to next edge from same source
     * @return false if no more edges from current source
     *
     * @note More efficient than next() when you only want edges from one source,
     *       as it avoids checking source node membership.
     *
     * ## Example: Collect all neighbors
     * @code
     *   if (iter.seek_from(node_id)) {
     *       do {
     *           if (iter.current().from_id != node_id) break;
     *           neighbors.push_back(iter.current().to_id);
     *       } while (iter.next_from_current());
     *   }
     * @endcode
     */
    bool next_from_current();

    /**
     * @brief Move to next edge (any source).
     *
     * Simple sequential advancement through the index.
     *
     * @return true if advanced to next edge
     * @return false if index exhausted
     */
    bool next();

    // =========================================================================
    // State Access
    // =========================================================================

    /**
     * @brief Get current edge record.
     *
     * @return EdgeRecord with from_id, to_id, edge_id
     *
     * @pre !exhausted() - behavior undefined if iterator is exhausted
     *
     * @note Returns by value for thread safety and simplicity.
     *       The underlying B+Tree cursor may be repositioned by seek operations.
     */
    EdgeRecord current() const;

    /**
     * @brief Check if iterator has no more edges.
     *
     * @return true if seek/next returned false (no more edges)
     * @return false if current() is valid
     */
    bool exhausted() const;

    /**
     * @brief Reset iterator to beginning of index.
     *
     * After reset, must call seek_from() or next() to position.
     */
    void reset();

    // =========================================================================
    // Statistics (for adaptive strategy selection)
    // =========================================================================

    /**
     * @brief Get total seeks performed since construction/reset.
     */
    uint64_t seeks_performed() const;

    /**
     * @brief Get total edges iterated since construction/reset.
     */
    uint64_t edges_iterated() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
