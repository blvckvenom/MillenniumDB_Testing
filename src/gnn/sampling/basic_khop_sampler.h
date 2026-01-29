#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "graph_models/object_id.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sampling_config.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

class TopologyAccessor;

/**
 * @brief Basic k-hop neighborhood sampler for GNN mini-batches.
 *
 * Implements the standard GraphSAGE-style sampling algorithm:
 * 1. Start with seed nodes (layer 0)
 * 2. For each layer k, sample up to fanout[k] neighbors for each node
 * 3. Build computational graph with local indices
 *
 * ## Algorithm
 *
 * @code
 *   nodes[0] = seeds
 *   for k in 0..K-1:
 *       for each node in nodes[k]:
 *           neighbors = sample_neighbors(node, fanout[k])
 *           nodes[k+1] = union(nodes[k+1], neighbors)
 *       edges[k] = {(src, dst) | src in nodes[k+1], dst in nodes[k], edge exists}
 * @endcode
 *
 * ## Message Passing Direction
 *
 * By default (EdgeOrientation::REVERSE), messages flow FROM neighbors TO seeds:
 * - Layer K (input features) → Layer K-1 → ... → Layer 0 (output embeddings)
 * - This matches the standard GNN convention where node embeddings are computed
 *   by aggregating neighbor information.
 *
 * ## Limitations (Basic Implementation)
 *
 * - Samples each node independently (N tree traversals for N nodes)
 * - Loads all neighbors into memory before sampling
 * - No reservoir sampling for high-degree nodes
 *
 * For optimized sampling, see Phase 2B: SortedBatchSampler, ReservoirSampler.
 *
 * @see GraphSample for output format
 * @see SamplingConfig for configuration
 * @see TopologyAccessor for neighbor retrieval
 */
class BasicKHopSampler {
public:
    /**
     * @brief Construct sampler for a projection.
     *
     * @param storage Reference to projection storage (must outlive sampler)
     * @param config Sampling configuration with fanouts and orientation
     */
    BasicKHopSampler(GQL::ProjectionStorage& storage, const SamplingConfig& config);

    ~BasicKHopSampler();

    // Disable copy
    BasicKHopSampler(const BasicKHopSampler&) = delete;
    BasicKHopSampler& operator=(const BasicKHopSampler&) = delete;

    // Allow move
    BasicKHopSampler(BasicKHopSampler&&) noexcept;
    BasicKHopSampler& operator=(BasicKHopSampler&&) noexcept;

    // =========================================================================
    // Sampling Interface
    // =========================================================================

    /**
     * @brief Sample k-hop neighborhood for a batch of seeds.
     *
     * This is the main sampling method. Given seed nodes, it:
     * 1. Samples neighbors layer by layer according to fanouts
     * 2. Builds the computational graph with local indices
     * 3. Computes all unique nodes for feature fetching
     *
     * Following DiskGNN architecture, epoch information is NOT stored
     * in samples - epochs belong to the training layer.
     *
     * @param seeds Seed nodes (will become layer 0 in output)
     * @param batch_id Identifier for this batch
     * @param split Train/validation/test split type
     * @return GraphSample containing the sampled subgraph
     */
    GraphSample sample(
        const std::vector<ObjectId>& seeds,
        uint64_t batch_id,
        SplitType split
    );

    /**
     * @brief Sample with default metadata (batch_id=0, TRAIN).
     *
     * Convenience method for testing and simple use cases.
     */
    GraphSample sample(const std::vector<ObjectId>& seeds);

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Get number of layers (K = fanouts.size()).
     */
    size_t num_layers() const;

    /**
     * @brief Get fanout for a specific layer.
     */
    uint64_t fanout(size_t layer) const;

    /**
     * @brief Set random seed for reproducible sampling.
     */
    void set_random_seed(uint64_t seed);

    /**
     * @brief Get current random seed.
     */
    uint64_t get_random_seed() const;

    // =========================================================================
    // Optimization Options
    // =========================================================================

    /**
     * @brief Enable/disable Leapfrog optimization for batch sampling.
     *
     * When enabled, layers with more than LEAPFROG_BATCH_THRESHOLD nodes
     * use coordinated B+Tree iteration instead of per-node lookups.
     *
     * @param enable true to enable (default), false to use basic per-node sampling
     */
    void set_use_leapfrog(bool enable);

    /**
     * @brief Check if Leapfrog optimization is enabled.
     */
    bool get_use_leapfrog() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
