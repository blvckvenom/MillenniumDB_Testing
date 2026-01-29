#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph_models/object_id.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sampling_config.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

/**
 * @brief Result of seed selection and splitting.
 *
 * Contains all seeds partitioned into train/validation/test sets,
 * ready for batch generation across multiple epochs.
 */
struct SeedSplit {
    std::vector<ObjectId> train_seeds;      ///< Training seed nodes
    std::vector<ObjectId> validation_seeds; ///< Validation seed nodes
    std::vector<ObjectId> test_seeds;       ///< Test seed nodes

    /**
     * @brief Total number of seeds across all splits.
     */
    size_t total_size() const {
        return train_seeds.size() + validation_seeds.size() + test_seeds.size();
    }

    /**
     * @brief Check if splits are non-empty and valid.
     */
    bool is_valid() const {
        // At minimum, we need training seeds
        return !train_seeds.empty();
    }
};

/**
 * @brief Batches organized by split type (train/validation/test).
 *
 * Following DiskGNN architecture: these batches are generated ONCE and
 * reused across training epochs. The training layer handles epoch iteration
 * and batch order shuffling.
 */
struct SplitBatches {
    std::vector<std::vector<ObjectId>> train_batches;      ///< Training batches
    std::vector<std::vector<ObjectId>> validation_batches; ///< Validation batches
    std::vector<std::vector<ObjectId>> test_batches;       ///< Test batches

    /**
     * @brief Total number of batches across all splits.
     */
    size_t total_batches() const {
        return train_batches.size() + validation_batches.size() + test_batches.size();
    }
};

// Backward compatibility alias
using EpochBatches = SplitBatches;

/**
 * @brief Selects and prepares seed nodes for GNN training.
 *
 * The SeedSelector is responsible for:
 * 1. **Seed Collection**: Gather nodes eligible for training (labeled nodes or query result)
 * 2. **Train/Val/Test Split**: Partition seeds according to configured ratios
 * 3. **Epoch Shuffling**: Deterministically shuffle seeds for each epoch
 * 4. **Batch Generation**: Create fixed-size batches from shuffled seeds
 *
 * ## Usage
 *
 * @code
 *   SamplingConfig config;
 *   config.projection_name = "my_graph";
 *   config.train_ratio = 0.7;
 *   config.batch_size = 1024;
 *
 *   SeedSelector selector(projection_storage, config);
 *
 *   // Get the train/val/test split
 *   SeedSplit split = selector.get_seed_split();
 *
 *   // Generate batches ONCE (following DiskGNN architecture)
 *   SplitBatches batches = selector.generate_batches();
 *
 *   for (const auto& batch : batches.train_batches) {
 *       // Process batch...
 *   }
 * @endcode
 *
 * ## DiskGNN Architecture
 *
 * Following DiskGNN (SIGMOD 2025), batches are generated ONCE and reused
 * across training epochs. The training layer handles:
 * - Epoch iteration
 * - Batch order shuffling per epoch (via `torch.randperm(num_batches)`)
 *
 * ## Deterministic Shuffling
 *
 * For reproducibility, initial batch shuffling uses config.random_seed.
 * This ensures identical batches across runs with the same configuration.
 *
 * @see SamplingConfig for configuration options
 * @see OfflineSamplingEngine for full sampling pipeline
 */
class SeedSelector {
public:
    /**
     * @brief Construct a seed selector for a projection.
     *
     * @param storage Reference to projection storage (must outlive selector)
     * @param config Sampling configuration with split ratios and batch size
     * @throws std::invalid_argument if config is invalid
     */
    SeedSelector(GQL::ProjectionStorage& storage, const SamplingConfig& config);

    ~SeedSelector();

    // Disable copy (holds internal state)
    SeedSelector(const SeedSelector&) = delete;
    SeedSelector& operator=(const SeedSelector&) = delete;

    // Allow move
    SeedSelector(SeedSelector&&) noexcept;
    SeedSelector& operator=(SeedSelector&&) noexcept;

    // =========================================================================
    // Seed Selection
    // =========================================================================

    /**
     * @brief Collect all seed nodes from the projection.
     *
     * If config.seed_query is set, executes the GQL query to get seeds.
     * Otherwise, collects all nodes that have the label_property defined.
     *
     * Results are cached after first call.
     *
     * @return Vector of all seed node IDs
     * @throws std::runtime_error if seed collection fails
     */
    std::vector<ObjectId> collect_seeds();

    /**
     * @brief Get the train/validation/test split.
     *
     * Performs seed collection (if not already done) and splits
     * according to config ratios. Results are cached.
     *
     * @return SeedSplit containing partitioned seeds
     * @throws std::runtime_error if splitting fails
     */
    SeedSplit get_seed_split();

    // =========================================================================
    // Batch Generation
    // =========================================================================

    /**
     * @brief Generate batches for all splits (single pass).
     *
     * This is the primary method following DiskGNN architecture:
     * batches are generated ONCE and reused across training epochs.
     *
     * @return SplitBatches containing batches for train/val/test splits
     */
    SplitBatches generate_batches();

    /**
     * @brief Generate batches with specific shuffle seed.
     *
     * Used internally and for training-layer batch order shuffling.
     * Each split is independently shuffled using:
     *   `rng_seed = config.random_seed + shuffle_seed * 3 + split_offset`
     *
     * where split_offset is 0 for train, 1 for val, 2 for test.
     *
     * @param shuffle_seed Seed for shuffling (typically epoch number in training)
     * @return SplitBatches with shuffled batches
     * @deprecated Use generate_batches() for offline sampling;
     *             batch ORDER shuffling should happen in training layer
     */
    SplitBatches generate_epoch_batches(uint64_t shuffle_seed);

    /**
     * @brief Generate batches for a specific split with shuffle seed.
     *
     * @param split_type Which split to generate batches for
     * @param shuffle_seed Seed for shuffling
     * @return Vector of batches, each containing up to batch_size seeds
     */
    std::vector<std::vector<ObjectId>> generate_split_batches(
        SplitType split_type,
        uint64_t shuffle_seed
    );

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get total number of seeds (all splits).
     */
    size_t total_seed_count();

    /**
     * @brief Get seed count for a specific split.
     */
    size_t seed_count(SplitType split_type);

    /**
     * @brief Get number of batches per epoch for a split.
     */
    size_t batches_per_epoch(SplitType split_type);

    /**
     * @brief Get total batches per epoch (all splits).
     */
    size_t total_batches_per_epoch();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Shuffle a vector deterministically with given seed.
 *
 * Uses Fisher-Yates shuffle with std::mt19937_64.
 *
 * @tparam T Element type
 * @param vec Vector to shuffle (modified in place)
 * @param seed Random seed for reproducibility
 */
template<typename T>
void deterministic_shuffle(std::vector<T>& vec, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(vec.begin(), vec.end(), rng);
}

/**
 * @brief Split a vector into fixed-size batches.
 *
 * @tparam T Element type
 * @param vec Source vector
 * @param batch_size Maximum elements per batch
 * @return Vector of batches (last batch may be smaller)
 */
template<typename T>
std::vector<std::vector<T>> create_batches(
    const std::vector<T>& vec,
    size_t batch_size
) {
    if (batch_size == 0) {
        throw std::invalid_argument("batch_size must be > 0");
    }

    std::vector<std::vector<T>> batches;
    size_t num_batches = (vec.size() + batch_size - 1) / batch_size;
    batches.reserve(num_batches);

    for (size_t i = 0; i < vec.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, vec.size());
        batches.emplace_back(vec.begin() + i, vec.begin() + end);
    }

    return batches;
}

} // namespace mdb::gnn
