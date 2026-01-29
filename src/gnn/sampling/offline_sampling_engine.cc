#include "gnn/sampling/offline_sampling_engine.h"

#include <chrono>
#include <stdexcept>

#include "gnn/sampling/basic_khop_sampler.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/seed_selector.h"
#include "gnn/sampling/sorted_batch_sampler.h"
#include "graph_models/gql/projection/projection_storage.h"

namespace mdb::gnn {

// =============================================================================
// Implementation
// =============================================================================

struct OfflineSamplingEngine::Impl {
    GQL::ProjectionStorage& storage;
    SamplingConfig config;

    // Components
    std::unique_ptr<SeedSelector> seed_selector;
    std::unique_ptr<BasicKHopSampler> khop_sampler;

    // Callbacks and settings
    ProgressCallback progress_callback;
    uint64_t progress_interval = 10;
    bool use_optimized_sampling = true;

    // State
    std::atomic<bool> cancel_requested{false};

    Impl(GQL::ProjectionStorage& storage_, const SamplingConfig& config_)
        : storage(storage_)
        , config(config_)
    {
        config.validate();
    }

    /**
     * @brief Initialize components lazily.
     */
    void init_components() {
        if (!seed_selector) {
            seed_selector = std::make_unique<SeedSelector>(storage, config);
        }
        if (!khop_sampler) {
            khop_sampler = std::make_unique<BasicKHopSampler>(storage, config);
        }
    }

    /**
     * @brief Report progress if callback is set.
     * @return false if cancellation requested
     */
    bool report_progress(const SamplingProgress& progress) {
        if (cancel_requested.load()) {
            return false;
        }
        if (progress_callback) {
            return progress_callback(progress);
        }
        return true;
    }

    /**
     * @brief Main sampling loop.
     */
    SamplingResult do_run(const std::filesystem::path& db_folder) {
        SamplingResult result;
        result.success = false;
        result.cancelled = false;
        result.total_samples = 0;
        result.total_seconds = 0;

        auto start_time = std::chrono::steady_clock::now();

        try {
            // Initialize components
            init_components();

            // Check if storage already exists
            if (SampleStorage::exists(db_folder, config.sample_name)) {
                result.error_message = "Sample storage already exists: " + config.sample_name;
                return result;
            }

            // Create storage
            SampleStorage sample_storage = SampleStorage::create(db_folder, config);

            // Get seed split
            SeedSplit split = seed_selector->get_seed_split();

            if (split.train_seeds.empty()) {
                result.error_message = "No training seeds found";
                return result;
            }

            // Calculate totals - single pass (no epochs in sampling layer)
            uint64_t total_batches = seed_selector->total_batches_per_epoch();
            uint64_t batch_id = 0;

            // Progress tracking
            SamplingProgress progress;
            progress.total_batches = total_batches;
            progress.current_batch = 0;
            progress.current_split = SplitType::TRAIN;
            progress.samples_written = 0;

            // Generate batches ONCE (following DiskGNN architecture)
            // Using epoch=0 for initial shuffle; training will shuffle batch ORDER
            EpochBatches batches = seed_selector->generate_epoch_batches(0);

            // Lambda to process batches for a split
            auto process_batches = [&](
                const std::vector<std::vector<ObjectId>>& split_batches,
                SplitType split_type
            ) -> bool {
                progress.current_split = split_type;

                for (size_t i = 0; i < split_batches.size(); ++i) {
                    if (cancel_requested.load()) {
                        result.cancelled = true;
                        return false;
                    }

                    const auto& batch_seeds = split_batches[i];

                    // Sample k-hop neighborhood (no epoch parameter)
                    GraphSample sample = khop_sampler->sample(
                        batch_seeds,
                        batch_id,
                        split_type
                    );

                    // Write to storage
                    sample_storage.write_sample(sample);

                    batch_id++;
                    progress.current_batch = batch_id;
                    progress.samples_written++;

                    // Report progress periodically
                    if (progress.samples_written % progress_interval == 0) {
                        auto now = std::chrono::steady_clock::now();
                        progress.elapsed_seconds = std::chrono::duration<double>(
                            now - start_time
                        ).count();

                        double rate = progress.throughput();
                        uint64_t remaining = total_batches - progress.samples_written;
                        progress.estimated_remaining = rate > 0 ? remaining / rate : 0;

                        if (!report_progress(progress)) {
                            result.cancelled = true;
                            return false;
                        }
                    }
                }
                return true;
            };

            // Process all splits (single pass, no epoch loop)
            if (process_batches(batches.train_batches, SplitType::TRAIN) &&
                process_batches(batches.validation_batches, SplitType::VALIDATION)) {
                process_batches(batches.test_batches, SplitType::TEST);
            }

            // Finalize storage
            sample_storage.finalize();

            // Calculate final timing
            auto end_time = std::chrono::steady_clock::now();
            result.total_seconds = std::chrono::duration<double>(end_time - start_time).count();
            result.total_samples = batch_id;
            result.catalog = sample_storage.get_catalog();

            if (!result.cancelled) {
                result.success = true;
            }

        } catch (const std::exception& e) {
            result.error_message = e.what();
            result.success = false;
        }

        return result;
    }

    /**
     * @brief Validate configuration.
     */
    std::string do_validate() {
        try {
            config.validate();
        } catch (const std::exception& e) {
            return e.what();
        }

        // Check projection has nodes
        init_components();
        if (seed_selector->total_seed_count() == 0) {
            return "Projection has no nodes";
        }

        return "";
    }

    /**
     * @brief Estimate total samples (single pass - no epochs).
     */
    uint64_t do_estimate_total_samples() {
        init_components();
        return seed_selector->total_batches_per_epoch();
    }

    /**
     * @brief Estimate storage size.
     *
     * Rough estimate based on:
     * - Each sample has ~batch_size * product(fanouts) nodes
     * - Each node is 8 bytes (ObjectId)
     * - Edges add ~50% overhead
     */
    uint64_t do_estimate_storage_bytes() {
        uint64_t nodes_per_sample = config.estimated_max_nodes_per_batch();
        uint64_t bytes_per_sample = nodes_per_sample * 8 * 2;  // nodes + edges overhead

        return do_estimate_total_samples() * bytes_per_sample;
    }
};

// =============================================================================
// OfflineSamplingEngine Public Interface
// =============================================================================

OfflineSamplingEngine::OfflineSamplingEngine(
    GQL::ProjectionStorage& storage,
    const SamplingConfig& config
)
    : impl_(std::make_unique<Impl>(storage, config))
{
}

OfflineSamplingEngine::~OfflineSamplingEngine() = default;

OfflineSamplingEngine::OfflineSamplingEngine(OfflineSamplingEngine&&) noexcept = default;
OfflineSamplingEngine& OfflineSamplingEngine::operator=(OfflineSamplingEngine&&) noexcept = default;

void OfflineSamplingEngine::set_progress_callback(ProgressCallback callback) {
    impl_->progress_callback = std::move(callback);
}

void OfflineSamplingEngine::set_progress_interval(uint64_t batches) {
    impl_->progress_interval = batches > 0 ? batches : 1;
}

void OfflineSamplingEngine::set_use_optimized_sampling(bool enabled) {
    impl_->use_optimized_sampling = enabled;
}

const SamplingConfig& OfflineSamplingEngine::get_config() const {
    return impl_->config;
}

SamplingResult OfflineSamplingEngine::run(const std::filesystem::path& db_folder) {
    impl_->cancel_requested.store(false);
    return impl_->do_run(db_folder);
}

void OfflineSamplingEngine::request_cancel() {
    impl_->cancel_requested.store(true);
}

bool OfflineSamplingEngine::is_cancelled() const {
    return impl_->cancel_requested.load();
}

std::string OfflineSamplingEngine::validate() {
    return impl_->do_validate();
}

uint64_t OfflineSamplingEngine::estimate_total_samples() const {
    return impl_->do_estimate_total_samples();
}

uint64_t OfflineSamplingEngine::estimate_storage_bytes() const {
    return impl_->do_estimate_storage_bytes();
}

} // namespace mdb::gnn
