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
 * @brief Result of batch neighbor sampling.
 *
 * Maps each seed node to its sampled neighbors and edge IDs.
 */
struct BatchNeighbors {
    /// Map: node_id -> vector of (neighbor_id, edge_id) pairs
    std::unordered_map<uint64_t, std::vector<std::pair<ObjectId, ObjectId>>> neighbors;

    /// Total neighbors collected across all seeds
    size_t total_neighbors() const {
        size_t count = 0;
        for (const auto& [_, vec] : neighbors) {
            count += vec.size();
        }
        return count;
    }

    /// Check if a node has any neighbors
    bool has_neighbors(uint64_t node_id) const {
        auto it = neighbors.find(node_id);
        return it != neighbors.end() && !it->second.empty();
    }
};

/**
 * @brief Optimized batch neighbor sampler using coordinated B+Tree iteration.
 *
 * This class implements efficient batch neighbor sampling by:
 * 1. Sorting batch node IDs for sequential B+Tree access
 * 2. Using a single coordinated sweep through the edge index
 * 3. Employing reservoir sampling for high-degree nodes
 *
 * ## Performance
 *
 * | Approach | Seeks | I/O Pattern |
 * |----------|-------|-------------|
 * | Per-node lookup | B × O(log E) | Random |
 * | **Coordinated sweep** | O(log E) + O(E_batch) | **Sequential** |
 *
 * Where B = batch size, E = total edges, E_batch = edges touching batch nodes.
 *
 * ## Algorithm
 *
 * For a sorted batch [n1, n2, ..., nB]:
 * 1. Seek to the first edge with from_id >= n1
 * 2. For each edge encountered:
 *    - If edge.from_id == current_batch_node: collect neighbor
 *    - If edge.from_id > current_batch_node: advance to next batch node
 *    - Apply reservoir sampling if fanout exceeded
 * 3. Stop when all batch nodes processed or index exhausted
 *
 * ## Usage
 *
 * @code
 *   LeapfrogGnnSampler sampler(projection_storage);
 *   sampler.set_fanout(15);
 *   sampler.set_random_seed(42);
 *
 *   std::vector<ObjectId> seeds = {node1, node2, node3};
 *   BatchNeighbors result = sampler.sample_batch(seeds, EdgeOrientation::REVERSE);
 *
 *   for (const auto& [seed_id, neighbors] : result.neighbors) {
 *       for (const auto& [neighbor, edge] : neighbors) {
 *           // Process neighbor
 *       }
 *   }
 * @endcode
 *
 * @see BasicKHopSampler for multi-layer sampling that uses this internally
 * @see TopologyAccessor for per-node neighbor access
 */
class LeapfrogGnnSampler {
public:
    /**
     * @brief Construct sampler for a projection.
     *
     * @param storage Reference to projection storage (must outlive sampler)
     */
    explicit LeapfrogGnnSampler(GQL::ProjectionStorage& storage);

    ~LeapfrogGnnSampler();

    // Disable copy
    LeapfrogGnnSampler(const LeapfrogGnnSampler&) = delete;
    LeapfrogGnnSampler& operator=(const LeapfrogGnnSampler&) = delete;

    // Allow move
    LeapfrogGnnSampler(LeapfrogGnnSampler&&) noexcept;
    LeapfrogGnnSampler& operator=(LeapfrogGnnSampler&&) noexcept;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the maximum neighbors to sample per node.
     * @param fanout Maximum neighbors (0 = unlimited)
     */
    void set_fanout(uint64_t fanout);

    /**
     * @brief Get the current fanout setting.
     */
    uint64_t get_fanout() const;

    /**
     * @brief Set random seed for reproducible sampling.
     * @param seed Random seed value
     */
    void set_random_seed(uint64_t seed);

    /**
     * @brief Get the current random seed.
     */
    uint64_t get_random_seed() const;

    // =========================================================================
    // Batch Sampling
    // =========================================================================

    /**
     * @brief Sample neighbors for a batch of nodes using coordinated sweep.
     *
     * This is the main optimization over per-node sampling. For large batches,
     * this can be 5-10x faster than calling get_neighbors() for each node.
     *
     * @param nodes Batch of seed nodes (will be sorted internally)
     * @param orientation Edge direction: NATURAL, REVERSE, or UNDIRECTED
     * @return BatchNeighbors with sampled neighbors for each seed
     */
    BatchNeighbors sample_batch(
        const std::vector<ObjectId>& nodes,
        EdgeOrientation orientation = EdgeOrientation::REVERSE
    );

    /**
     * @brief Sample neighbors with explicit fanout (overrides set_fanout).
     *
     * @param nodes Batch of seed nodes
     * @param fanout Maximum neighbors per node
     * @param orientation Edge direction
     * @return BatchNeighbors with sampled neighbors for each seed
     */
    BatchNeighbors sample_batch(
        const std::vector<ObjectId>& nodes,
        uint64_t fanout,
        EdgeOrientation orientation = EdgeOrientation::REVERSE
    );

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get total edges scanned in last sample_batch call.
     */
    uint64_t last_edges_scanned() const;

    /**
     * @brief Get total neighbors collected in last sample_batch call.
     */
    uint64_t last_neighbors_collected() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Threshold for using optimized batch sampling.
 *
 * For batches smaller than this, per-node lookup may be faster
 * due to reduced overhead.
 */
constexpr size_t LEAPFROG_BATCH_THRESHOLD = 50;

} // namespace mdb::gnn
