#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "graph_models/object_id.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/sampling/leapfrog_gnn_sampler.h"  // For BatchNeighbors

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

/**
 * @brief Seek-based batch neighbor sampler using O(log E) seeks.
 *
 * This class implements efficient batch neighbor sampling for **sparse batches**
 * where the batch nodes are sparsely distributed across the node ID space.
 * It complements LeapfrogGnnSampler which is optimal for dense batches.
 *
 * ## Algorithm
 *
 * For a sorted batch [n1, n2, ..., nB] (B nodes):
 * 1. seek_from(n1)     → O(log E) B+Tree traversal
 * 2. Collect edges while from_id == n1 with reservoir sampling
 * 3. seek_from(n2)     → O(log E) jump (forward in tree)
 * 4. Repeat until all B nodes processed
 *
 * Total complexity: O(B × log E) seeks + O(Σ degree(ni)) iterations
 *
 * ## Performance Comparison
 *
 * | Scenario              | Sweep (Leapfrog) | Seek (This class) | Winner |
 * |-----------------------|------------------|-------------------|--------|
 * | 1K nodes in 5M edges  | O(5M)            | O(16K)            | **Seek** |
 * | 1K nodes in 2K edges  | O(2K)            | O(16K)            | Sweep  |
 * | Dense batch (>50%)    | O(E_range)       | O(B × log E)      | Sweep  |
 *
 * ## When to Use
 *
 * - Batch density < 10% (nodes cover small fraction of ID range)
 * - Large gaps between consecutive node IDs
 * - Graph has millions of edges but batch is small
 *
 * ## Usage
 *
 * @code
 *   SeekBasedGnnSampler sampler(projection_storage);
 *   sampler.set_random_seed(42);
 *
 *   std::vector<ObjectId> seeds = {node1, node2, node3};  // Sparse IDs
 *   BatchNeighbors result = sampler.sample_batch(seeds, 15, EdgeOrientation::REVERSE);
 *
 *   for (const auto& [seed_id, neighbors] : result.neighbors) {
 *       for (const auto& [neighbor, edge] : neighbors) {
 *           // Process neighbor
 *       }
 *   }
 * @endcode
 *
 * @see LeapfrogGnnSampler for sweep-based approach (better for dense batches)
 * @see AdaptiveBatchSampler for automatic strategy selection
 */
class SeekBasedGnnSampler {
public:
    /**
     * @brief Construct sampler for a projection.
     *
     * @param storage Reference to projection storage (must outlive sampler)
     */
    explicit SeekBasedGnnSampler(GQL::ProjectionStorage& storage);

    ~SeekBasedGnnSampler();

    // Disable copy
    SeekBasedGnnSampler(const SeekBasedGnnSampler&) = delete;
    SeekBasedGnnSampler& operator=(const SeekBasedGnnSampler&) = delete;

    // Allow move
    SeekBasedGnnSampler(SeekBasedGnnSampler&&) noexcept;
    SeekBasedGnnSampler& operator=(SeekBasedGnnSampler&&) noexcept;

    // =========================================================================
    // Configuration
    // =========================================================================

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
     * @brief Sample neighbors for a batch of nodes using seek-based iteration.
     *
     * Uses O(log E) seeks per node instead of linear sweep. Optimal for sparse
     * batches where nodes are distributed across wide ID ranges.
     *
     * @param nodes Batch of seed nodes (will be sorted internally)
     * @param fanout Maximum neighbors per node (0 = unlimited)
     * @param orientation Edge direction: NATURAL, REVERSE, or UNDIRECTED
     * @return BatchNeighbors with sampled neighbors for each seed
     *
     * @note For small batches (<50 nodes), per-node lookup may be faster.
     *       For dense batches, LeapfrogGnnSampler's sweep is better.
     */
    BatchNeighbors sample_batch(
        const std::vector<ObjectId>& nodes,
        uint64_t fanout,
        EdgeOrientation orientation = EdgeOrientation::REVERSE
    );

    // =========================================================================
    // Statistics (for adaptive strategy decisions)
    // =========================================================================

    /**
     * @brief Get total seeks performed in last sample_batch call.
     *
     * Useful for comparing with sweep operations to validate strategy choice.
     */
    uint64_t last_seeks_performed() const;

    /**
     * @brief Get total edges iterated in last sample_batch call.
     */
    uint64_t last_edges_iterated() const;

    /**
     * @brief Get total neighbors collected in last sample_batch call.
     */
    uint64_t last_neighbors_collected() const;

    /**
     * @brief Estimate cost of seek-based sampling for given batch.
     *
     * Used by adaptive strategy selector to compare with sweep cost.
     *
     * @param batch_size Number of nodes in batch
     * @param total_edges Total edges in projection
     * @return Estimated operations (seeks × log factor)
     */
    static double estimate_seek_cost(
        size_t batch_size,
        uint64_t total_edges,
        double overhead_factor = 2.0
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
