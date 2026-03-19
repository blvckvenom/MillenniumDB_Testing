#include "gnn/sampling/seed_selector.h"

#include <algorithm>
#include <climits>
#include <numeric>
#include <stdexcept>

#include "gnn/projection/topology_accessor.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/gql/projection/projection_storage.h"

namespace mdb::gnn {

// =============================================================================
// Implementation Details
// =============================================================================

struct SeedSelector::Impl {
    GQL::ProjectionStorage& storage;
    SamplingConfig config;

    // Cached data — one of two paths is used:
    //  (a) ObjectId path: all_seeds stores full ObjectIds (8 bytes each)
    //  (b) Index path:    seed_indices stores uint32_t row indices (4 bytes each)
    //      ObjectIds are recovered on demand via row_mapping_->get()
    const RowMapping* row_mapping_ = nullptr;
    bool use_index_path_ = false;             // true when RowMapping available and size <= UINT32_MAX
    std::vector<ObjectId>  all_seeds;         // (a) ObjectId path
    std::vector<uint32_t>  seed_indices;      // (b) Index path
    SeedSplit split;
    bool seeds_collected = false;
    bool split_computed = false;

    Impl(GQL::ProjectionStorage& storage_, const SamplingConfig& config_,
         const RowMapping* row_mapping)
        : storage(storage_)
        , config(config_)
        , row_mapping_(row_mapping)
    {
        // Validate configuration
        config.validate();

        // Determine if we can use the index path (uint32_t row indices).
        // Falls back to ObjectId path when RowMapping is null or too large.
        if (row_mapping_ && row_mapping_->size() <= static_cast<uint64_t>(UINT32_MAX)) {
            use_index_path_ = true;
        }
    }

    /**
     * @brief Collect seeds using NodeIterator or RowMapping.
     *
     * Two paths:
     *  (a) Index path (use_index_path_): generate [0, 1, ..., N-1] as uint32_t
     *      indices into RowMapping. Half the memory of ObjectId storage.
     *  (b) ObjectId path (fallback): load all ObjectIds via NodeIterator.
     */
    void collect_all_seeds() {
        if (seeds_collected) {
            return;
        }

        if (use_index_path_) {
            // Index-based: generate [0, 1, 2, ..., N-1] as uint32_t
            uint64_t N = row_mapping_->size();
            seed_indices.resize(N);
            std::iota(seed_indices.begin(), seed_indices.end(), uint32_t(0));
            // ObjectIds recovered on demand via row_mapping_->get(seed_indices[i])
        } else {
            // Original path: load all ObjectIds via NodeIterator
            all_seeds.clear();
            NodeIterator iter(storage);
            uint64_t total = iter.total_count();
            all_seeds.reserve(total);

            constexpr size_t BATCH_SIZE = 10000;
            while (auto batch = iter.next_batch(BATCH_SIZE)) {
                for (const auto& node_id : *batch) {
                    all_seeds.push_back(node_id);
                }
            }
        }

        seeds_collected = true;
    }

    /**
     * @brief Convert a range of seed_indices to ObjectIds via RowMapping.
     *
     * Helper for compute_split() on the index path.
     */
    std::vector<ObjectId> indices_to_object_ids(
        const std::vector<uint32_t>& indices,
        size_t begin,
        size_t end
    ) const {
        std::vector<ObjectId> result;
        result.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            result.push_back(row_mapping_->get(indices[i]));
        }
        return result;
    }

    /**
     * @brief Split seeds into train/validation/test sets.
     *
     * Uses deterministic shuffle with config.random_seed for reproducibility.
     * Operates on either seed_indices (index path) or all_seeds (ObjectId path).
     */
    void compute_split() {
        if (split_computed) {
            return;
        }

        // Ensure seeds are collected
        collect_all_seeds();

        if (use_index_path_) {
            // Index path: shuffle uint32_t indices, then convert ranges to ObjectIds
            if (seed_indices.empty()) {
                split_computed = true;
                return;
            }

            std::vector<uint32_t> shuffled = seed_indices;
            deterministic_shuffle(shuffled, config.random_seed);

            size_t total = shuffled.size();
            size_t train_end = static_cast<size_t>(total * config.train_ratio);
            size_t val_end = train_end + static_cast<size_t>(total * config.val_ratio);

            if (config.train_ratio > 0 && train_end == 0 && total > 0) {
                train_end = 1;
            }
            if (config.val_ratio > 0 && val_end == train_end && total > train_end) {
                val_end = train_end + 1;
            }

            split.train_seeds      = indices_to_object_ids(shuffled, 0, train_end);
            split.validation_seeds = indices_to_object_ids(shuffled, train_end, val_end);
            split.test_seeds       = indices_to_object_ids(shuffled, val_end, total);
        } else {
            // ObjectId path (original)
            if (all_seeds.empty()) {
                split_computed = true;
                return;
            }

            std::vector<ObjectId> shuffled = all_seeds;
            deterministic_shuffle(shuffled, config.random_seed);

            size_t total = shuffled.size();
            size_t train_end = static_cast<size_t>(total * config.train_ratio);
            size_t val_end = train_end + static_cast<size_t>(total * config.val_ratio);

            if (config.train_ratio > 0 && train_end == 0 && total > 0) {
                train_end = 1;
            }
            if (config.val_ratio > 0 && val_end == train_end && total > train_end) {
                val_end = train_end + 1;
            }

            split.train_seeds.assign(shuffled.begin(), shuffled.begin() + train_end);
            split.validation_seeds.assign(shuffled.begin() + train_end, shuffled.begin() + val_end);
            split.test_seeds.assign(shuffled.begin() + val_end, shuffled.end());
        }

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

SeedSelector::SeedSelector(GQL::ProjectionStorage& storage,
                           const SamplingConfig& config,
                           const RowMapping* row_mapping)
    : impl_(std::make_unique<Impl>(storage, config, row_mapping))
{
}

SeedSelector::~SeedSelector() = default;

SeedSelector::SeedSelector(SeedSelector&&) noexcept = default;
SeedSelector& SeedSelector::operator=(SeedSelector&&) noexcept = default;

std::vector<ObjectId> SeedSelector::collect_seeds() {
    impl_->collect_all_seeds();

    if (impl_->use_index_path_) {
        // Materialize ObjectIds from indices for callers that need the full vector
        return impl_->indices_to_object_ids(
            impl_->seed_indices, 0, impl_->seed_indices.size());
    }
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
    return impl_->use_index_path_ ? impl_->seed_indices.size() : impl_->all_seeds.size();
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
