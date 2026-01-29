#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gnn/projection/topology_accessor.h"  // For EdgeOrientation, SamplingStrategy

namespace mdb::gnn {

/**
 * @brief Configuration for offline sampling.
 *
 * Defines all parameters for the offline sampling process:
 * - Seed selection (which nodes to train on)
 * - Train/val/test splitting
 * - K-hop sampling (layers, fanouts)
 * - Reproducibility (random seed)
 *
 * Usage:
 * @code
 *   SamplingConfig config;
 *   config.projection_name = "my_graph";
 *   config.fanouts = {15, 10};  // 2-hop with fanout 15 then 10
 *   config.batch_size = 1024;
 *   config.validate();  // Throws if invalid
 * @endcode
 *
 * @see OfflineSamplingEngine for usage
 */
struct SamplingConfig {
    // =========================================================================
    // Projection Selection
    // =========================================================================

    std::string projection_name;  ///< Name of the projection to sample from

    // =========================================================================
    // Seed Selection
    // =========================================================================

    /**
     * @brief GQL query to select seed nodes.
     *
     * If empty, uses all nodes with the label_property defined.
     * Example: "MATCH (n:User) WHERE n.label IS NOT NULL RETURN id(n)"
     */
    std::optional<std::string> seed_query;

    /**
     * @brief Property name containing node labels for classification.
     *
     * Nodes without this property are excluded from training seeds.
     */
    std::string label_property = "label";

    // =========================================================================
    // Train/Val/Test Split
    // =========================================================================

    double train_ratio = 0.7;   ///< Fraction of seeds for training (default: 70%)
    double val_ratio = 0.15;    ///< Fraction of seeds for validation (default: 15%)
    double test_ratio = 0.15;   ///< Fraction of seeds for testing (default: 15%)

    // =========================================================================
    // Batching
    // =========================================================================

    uint64_t batch_size = 1024;  ///< Number of seed nodes per batch

    // =========================================================================
    // K-Hop Sampling
    // =========================================================================

    /**
     * @brief Number of neighbors to sample per layer.
     *
     * fanouts[0] = neighbors for 1-hop, fanouts[1] = neighbors for 2-hop, etc.
     * Number of layers K = fanouts.size()
     *
     * Example: {15, 10} means:
     *   - Sample 15 neighbors at 1-hop distance
     *   - Sample 10 neighbors at 2-hop distance
     */
    std::vector<uint64_t> fanouts = {15, 10};

    /**
     * @brief Edge orientation for neighbor traversal.
     *
     * - REVERSE: Messages flow from neighbors to seed (standard GNN convention)
     * - NATURAL: Messages flow from seed to neighbors
     * - UNDIRECTED: Traverse both directions
     */
    EdgeOrientation orientation = EdgeOrientation::REVERSE;

    /**
     * @brief Sampling strategy for neighbor selection.
     */
    SamplingStrategy strategy = SamplingStrategy::UNIFORM;

    /**
     * @brief Whether to sample with replacement.
     *
     * If true, same neighbor can be sampled multiple times.
     * If false (default), each neighbor appears at most once.
     */
    bool sample_with_replacement = false;

    // =========================================================================
    // Batch Strategy (B+Tree Traversal Optimization)
    // =========================================================================

    /**
     * @brief Strategy for batch neighbor sampling I/O pattern.
     *
     * - AUTO (default): Automatically choose between SWEEP and SEEK based on
     *   batch density and graph size
     * - SWEEP: Use coordinated B+Tree sweep (LeapfrogGnnSampler)
     * - SEEK: Use O(log E) seeks per node (SeekBasedGnnSampler)
     * - PER_NODE: Use TopologyAccessor per-node lookups
     *
     * @see BatchStrategy enum for performance comparison
     */
    BatchStrategy batch_strategy = BatchStrategy::AUTO;

    /**
     * @brief Overhead factor for seek cost estimation.
     *
     * Used in AUTO mode to decide between SWEEP and SEEK strategies.
     * Higher values favor SWEEP (more conservative about seek overhead).
     *
     * The cost model is: seek_cost = B × log2(E) × seek_overhead_factor
     *
     * Typical values:
     * - 1.5: Aggressive (prefers SEEK more often)
     * - 2.0: Balanced (default)
     * - 3.0: Conservative (prefers SWEEP more often)
     *
     * Tune based on your hardware characteristics:
     * - SSD storage: Use lower values (1.5-2.0)
     * - HDD storage: Use higher values (2.5-3.0)
     * - Large buffer pool: Use lower values
     */
    double seek_overhead_factor = 2.0;

    // =========================================================================
    // Reproducibility
    // =========================================================================

    uint64_t random_seed = 42;  ///< Random seed for deterministic sampling

    // =========================================================================
    // Memory Optimization
    // =========================================================================

    /**
     * @brief Degree threshold for reservoir sampling.
     *
     * Nodes with degree > this value use streaming reservoir sampling
     * instead of loading all neighbors into memory.
     */
    uint64_t reservoir_threshold = 10000;

    // =========================================================================
    // Output
    // =========================================================================

    /**
     * @brief Name for the sample storage.
     *
     * Samples are stored at: <db_folder>/samples/<sample_name>/
     */
    std::string sample_name;

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate configuration parameters.
     *
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (projection_name.empty()) {
            throw std::invalid_argument("projection_name cannot be empty");
        }

        if (sample_name.empty()) {
            throw std::invalid_argument("sample_name cannot be empty");
        }

        if (fanouts.empty()) {
            throw std::invalid_argument("fanouts cannot be empty (need at least 1 layer)");
        }

        for (size_t i = 0; i < fanouts.size(); ++i) {
            if (fanouts[i] == 0) {
                throw std::invalid_argument("fanout[" + std::to_string(i) + "] cannot be 0");
            }
        }

        if (batch_size == 0) {
            throw std::invalid_argument("batch_size must be > 0");
        }

        double total_ratio = train_ratio + val_ratio + test_ratio;
        if (total_ratio < 0.999 || total_ratio > 1.001) {
            throw std::invalid_argument(
                "train_ratio + val_ratio + test_ratio must equal 1.0 (got " +
                std::to_string(total_ratio) + ")"
            );
        }

        if (train_ratio < 0.0 || val_ratio < 0.0 || test_ratio < 0.0) {
            throw std::invalid_argument("split ratios cannot be negative");
        }
    }

    /**
     * @brief Get the number of GNN layers (K).
     */
    size_t num_layers() const {
        return fanouts.size();
    }

    /**
     * @brief Estimate maximum nodes per batch (upper bound).
     *
     * Actual count is typically lower due to overlap and degree variance.
     */
    uint64_t estimated_max_nodes_per_batch() const {
        uint64_t total = batch_size;  // Layer 0 (seeds)
        uint64_t layer_size = batch_size;

        for (uint64_t fanout : fanouts) {
            layer_size *= fanout;
            total += layer_size;
        }

        return total;
    }
};

} // namespace mdb::gnn
