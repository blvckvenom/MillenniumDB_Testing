#include "gnn/sampling/seed_selector.h"

#include <algorithm>
#include <stdexcept>

#include "gnn/projection/topology_accessor.h"
#include "graph_models/gql/projection/projection_storage.h"

namespace mdb::gnn {

// =============================================================================
// Implementation Details
// =============================================================================

struct SeedSelector::Impl {
    GQL::ProjectionStorage& storage;
    SamplingConfig config;

    // Cached data
    std::vector<ObjectId> all_seeds;
    SeedSplit split;
    bool seeds_collected = false;
    bool split_computed = false;

    Impl(GQL::ProjectionStorage& storage_, const SamplingConfig& config_)
        : storage(storage_)
        , config(config_)
    {
        // Validate configuration
        config.validate();
    }

    /**
     * @brief Collect seeds using NodeIterator (all nodes in projection).
     *
     * For MVP, we collect ALL nodes. Future enhancement: filter by label_property
     * or execute seed_query via GQL engine.
     */
    void collect_all_seeds() {
        if (seeds_collected) {
            return;
        }

        all_seeds.clear();

        // Use NodeIterator for memory-efficient streaming
        NodeIterator iter(storage);
        uint64_t total = iter.total_count();
        all_seeds.reserve(total);

        // Batch iteration is more efficient than single-node iteration
        constexpr size_t BATCH_SIZE = 10000;
        while (auto batch = iter.next_batch(BATCH_SIZE)) {
            for (const auto& node_id : *batch) {
                all_seeds.push_back(node_id);
            }
        }

        seeds_collected = true;
    }

    /**
     * @brief Split seeds into train/validation/test sets.
     *
     * Uses deterministic shuffle with config.random_seed for reproducibility.
     */
    void compute_split() {
        if (split_computed) {
            return;
        }

        // Ensure seeds are collected
        collect_all_seeds();

        if (all_seeds.empty()) {
            split_computed = true;
            return;
        }

        // Create a shuffled copy for splitting
        std::vector<ObjectId> shuffled = all_seeds;
        deterministic_shuffle(shuffled, config.random_seed);

        // Calculate split indices
        size_t total = shuffled.size();
        size_t train_end = static_cast<size_t>(total * config.train_ratio);
        size_t val_end = train_end + static_cast<size_t>(total * config.val_ratio);

        // Ensure at least 1 sample per split if ratio > 0
        if (config.train_ratio > 0 && train_end == 0 && total > 0) {
            train_end = 1;
        }
        if (config.val_ratio > 0 && val_end == train_end && total > train_end) {
            val_end = train_end + 1;
        }

        // Partition seeds
        split.train_seeds.assign(shuffled.begin(), shuffled.begin() + train_end);
        split.validation_seeds.assign(shuffled.begin() + train_end, shuffled.begin() + val_end);
        split.test_seeds.assign(shuffled.begin() + val_end, shuffled.end());

        split_computed = true;
    }

    /**
     * @brief Generate shuffled batches for a specific split and epoch.
     */
    std::vector<std::vector<ObjectId>> generate_split_batches(
        const std::vector<ObjectId>& seeds,
        uint64_t epoch,
        uint64_t split_offset
    ) {
        if (seeds.empty()) {
            return {};
        }

        // Create a copy and shuffle for this epoch
        std::vector<ObjectId> shuffled = seeds;

        // Deterministic seed: base_seed + epoch * 3 + split_offset
        // This ensures different shuffles per epoch AND per split
        uint64_t rng_seed = config.random_seed + epoch * 3 + split_offset;
        deterministic_shuffle(shuffled, rng_seed);

        // Create batches
        return create_batches(shuffled, config.batch_size);
    }
};

// =============================================================================
// SeedSelector Public Interface
// =============================================================================

SeedSelector::SeedSelector(GQL::ProjectionStorage& storage, const SamplingConfig& config)
    : impl_(std::make_unique<Impl>(storage, config))
{
}

SeedSelector::~SeedSelector() = default;

SeedSelector::SeedSelector(SeedSelector&&) noexcept = default;
SeedSelector& SeedSelector::operator=(SeedSelector&&) noexcept = default;

std::vector<ObjectId> SeedSelector::collect_seeds() {
    impl_->collect_all_seeds();
    return impl_->all_seeds;
}

SeedSplit SeedSelector::get_seed_split() {
    impl_->compute_split();
    return impl_->split;
}

SplitBatches SeedSelector::generate_batches() {
    // Generate batches with initial shuffle (using config.random_seed)
    return generate_epoch_batches(0);
}

SplitBatches SeedSelector::generate_epoch_batches(uint64_t shuffle_seed) {
    impl_->compute_split();

    SplitBatches batches;
    batches.train_batches = impl_->generate_split_batches(
        impl_->split.train_seeds, shuffle_seed, 0
    );
    batches.validation_batches = impl_->generate_split_batches(
        impl_->split.validation_seeds, shuffle_seed, 1
    );
    batches.test_batches = impl_->generate_split_batches(
        impl_->split.test_seeds, shuffle_seed, 2
    );

    return batches;
}

std::vector<std::vector<ObjectId>> SeedSelector::generate_split_batches(
    SplitType split_type,
    uint64_t shuffle_seed
) {
    impl_->compute_split();

    switch (split_type) {
        case SplitType::TRAIN:
            return impl_->generate_split_batches(impl_->split.train_seeds, shuffle_seed, 0);
        case SplitType::VALIDATION:
            return impl_->generate_split_batches(impl_->split.validation_seeds, shuffle_seed, 1);
        case SplitType::TEST:
            return impl_->generate_split_batches(impl_->split.test_seeds, shuffle_seed, 2);
    }

    // Unreachable, but satisfy compiler
    return {};
}

size_t SeedSelector::total_seed_count() {
    impl_->collect_all_seeds();
    return impl_->all_seeds.size();
}

size_t SeedSelector::seed_count(SplitType split_type) {
    impl_->compute_split();

    switch (split_type) {
        case SplitType::TRAIN:
            return impl_->split.train_seeds.size();
        case SplitType::VALIDATION:
            return impl_->split.validation_seeds.size();
        case SplitType::TEST:
            return impl_->split.test_seeds.size();
    }

    return 0;
}

size_t SeedSelector::batches_per_epoch(SplitType split_type) {
    size_t seeds = seed_count(split_type);
    if (seeds == 0) return 0;
    return (seeds + impl_->config.batch_size - 1) / impl_->config.batch_size;
}

size_t SeedSelector::total_batches_per_epoch() {
    return batches_per_epoch(SplitType::TRAIN) +
           batches_per_epoch(SplitType::VALIDATION) +
           batches_per_epoch(SplitType::TEST);
}

} // namespace mdb::gnn
