#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "graph_models/object_id.h"
#include "gnn/projection/topology_accessor.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

/**
 * @brief Result of batch neighbor collection.
 *
 * Maps each source node to its sampled neighbors with edge IDs.
 */
struct BatchNeighbors {
    /// node_id -> [(neighbor_id, edge_id), ...]
    std::unordered_map<uint64_t, std::vector<std::pair<ObjectId, ObjectId>>> neighbors;

    /**
     * @brief Get neighbors for a specific node.
     * @return Empty vector if node not found
     */
    const std::vector<std::pair<ObjectId, ObjectId>>& get(ObjectId node_id) const {
        static const std::vector<std::pair<ObjectId, ObjectId>> empty;
        auto it = neighbors.find(node_id.id);
        return it != neighbors.end() ? it->second : empty;
    }

    /**
     * @brief Check if node has any neighbors.
     */
    bool has_neighbors(ObjectId node_id) const {
        auto it = neighbors.find(node_id.id);
        return it != neighbors.end() && !it->second.empty();
    }

    /**
     * @brief Total number of nodes with neighbors.
     */
    size_t num_nodes() const {
        return neighbors.size();
    }

    /**
     * @brief Total number of edges across all nodes.
     */
    size_t total_edges() const {
        size_t count = 0;
        for (const auto& [_, edges] : neighbors) {
            count += edges.size();
        }
        return count;
    }
};

/**
 * @brief Optimized batch neighbor sampler using sorted traversal.
 *
 * This class provides significant I/O optimization over naive per-node sampling
 * by processing nodes in sorted order, enabling a single B+tree traversal.
 *
 * ## Performance Comparison
 *
 * | Approach | Tree Traversals | I/O Pattern |
 * |----------|-----------------|-------------|
 * | Naive (per-node) | N × O(log E) | Random seeks |
 * | Sorted Batch | 1 × O(log E + E_batch) | Sequential scan |
 *
 * Where N = batch size, E = total edges, E_batch = edges among batch nodes.
 *
 * ## Algorithm
 *
 * 1. Sort input nodes by ObjectId
 * 2. Open single range iterator on B+tree [min_node, max_node]
 * 3. Scan sequentially, collecting neighbors for each node
 * 4. Apply sampling (uniform/reservoir) during collection
 *
 * ## Usage
 *
 * @code
 *   SortedBatchSampler sampler(storage, EdgeOrientation::REVERSE);
 *   sampler.set_fanout(15);
 *
 *   std::vector<ObjectId> nodes = {...};
 *   BatchNeighbors result = sampler.sample_batch(nodes);
 *
 *   for (const auto& node : nodes) {
 *       auto neighbors = result.get(node);
 *       // Process neighbors...
 *   }
 * @endcode
 *
 * @see BasicKHopSampler for full k-hop sampling
 * @see ReservoirSampler for high-degree node optimization
 */
class SortedBatchSampler {
public:
    /**
     * @brief Construct sampler for a projection.
     *
     * @param storage Reference to projection storage (must outlive sampler)
     * @param orientation Edge traversal direction
     */
    SortedBatchSampler(
        GQL::ProjectionStorage& storage,
        EdgeOrientation orientation = EdgeOrientation::REVERSE
    );

    ~SortedBatchSampler();

    // Disable copy
    SortedBatchSampler(const SortedBatchSampler&) = delete;
    SortedBatchSampler& operator=(const SortedBatchSampler&) = delete;

    // Allow move
    SortedBatchSampler(SortedBatchSampler&&) noexcept;
    SortedBatchSampler& operator=(SortedBatchSampler&&) noexcept;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set maximum neighbors per node (fanout).
     *
     * @param fanout Maximum neighbors to sample. 0 = unlimited (all neighbors)
     */
    void set_fanout(uint64_t fanout);

    /**
     * @brief Get current fanout setting.
     */
    uint64_t get_fanout() const;

    /**
     * @brief Set random seed for reproducible sampling.
     */
    void set_random_seed(uint64_t seed);

    /**
     * @brief Set edge orientation for traversal.
     */
    void set_orientation(EdgeOrientation orientation);

    /**
     * @brief Get current orientation.
     */
    EdgeOrientation get_orientation() const;

    /**
     * @brief Set threshold for switching to reservoir sampling.
     *
     * Nodes with degree > threshold use streaming reservoir sampling
     * instead of loading all neighbors.
     *
     * @param threshold Degree threshold (default: 10000)
     */
    void set_reservoir_threshold(uint64_t threshold);

    // =========================================================================
    // Sampling Interface
    // =========================================================================

    /**
     * @brief Sample neighbors for a batch of nodes using sorted traversal.
     *
     * This is the main optimized method. Nodes are internally sorted
     * for optimal B+tree access pattern.
     *
     * @param nodes Nodes to sample neighbors for (will be sorted internally)
     * @return BatchNeighbors mapping each node to its sampled neighbors
     */
    BatchNeighbors sample_batch(const std::vector<ObjectId>& nodes);

    /**
     * @brief Collect ALL neighbors for a batch (no sampling).
     *
     * Useful for small graphs or when you need complete neighborhood.
     *
     * @param nodes Nodes to collect neighbors for
     * @return BatchNeighbors with all neighbors (no sampling applied)
     */
    BatchNeighbors collect_all_batch(const std::vector<ObjectId>& nodes);

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get number of B+tree pages accessed in last operation.
     *
     * Useful for benchmarking I/O efficiency.
     */
    uint64_t last_pages_accessed() const;

    /**
     * @brief Get number of edges scanned in last operation.
     */
    uint64_t last_edges_scanned() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
