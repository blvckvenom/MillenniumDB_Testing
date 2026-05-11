#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sampling_config.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

/**
 * @brief Progress information during offline sampling.
 *
 * Following DiskGNN architecture: sampling produces ONE set of batches
 * (no epochs in sampling layer - epochs belong to training).
 */
struct SamplingProgress {
    uint64_t current_batch;       ///< Current batch being processed (0-indexed)
    uint64_t total_batches;       ///< Total batches to generate
    SplitType current_split;      ///< Current split being processed
    uint64_t samples_written;     ///< Total samples written so far
    double elapsed_seconds;       ///< Time elapsed since start
    double estimated_remaining;   ///< Estimated time remaining

    /**
     * @brief Overall progress as fraction [0.0, 1.0].
     */
    double progress() const {
        return total_batches > 0
            ? static_cast<double>(samples_written) / total_batches
            : 0.0;
    }

    /**
     * @brief Samples per second throughput.
     */
    double throughput() const {
        return elapsed_seconds > 0 ? samples_written / elapsed_seconds : 0.0;
    }
};

/**
 * @brief Callback type for progress updates.
 *
 * Return false to request cancellation.
 */
using ProgressCallback = std::function<bool(const SamplingProgress&)>;

/**
 * @brief Result of offline sampling operation.
 */
struct SamplingResult {
    bool success;                 ///< True if sampling completed successfully
    std::string error_message;    ///< Error description if !success
    SampleCatalog catalog;        ///< Final catalog with statistics
    double total_seconds;         ///< Total time taken
    uint64_t total_samples;       ///< Total samples generated
    bool cancelled;               ///< True if cancelled via callback

    // Plan E Phase 0 telemetry (2026-05-11) — populated only when
    // `auto_profile_on_cold_start=true` AND the profiler actually ran.
    // See `BasicKHopSampler::Impl::Phase0Telemetry` for definitions.
    bool        phase0_triggered     = false;
    bool        phase0_succeeded     = false;
    uint64_t    phase0_walks_done    = 0;
    uint64_t    phase0_lookups_done  = 0;
    double      phase0_elapsed_seconds = 0.0;
};

/**
 * @brief Main orchestrator for offline GNN mini-batch pre-computation.
 *
 * The OfflineSamplingEngine coordinates the entire offline sampling workflow:
 * 1. Seed selection and train/val/test splitting
 * 2. K-hop neighborhood sampling for each batch
 * 3. Persistent storage of all samples
 * 4. Frequency analysis for cache optimization
 *
 * ## DiskGNN Architecture
 *
 * This follows the DiskGNN (SIGMOD 2025) approach of pre-computing ALL
 * mini-batches before training. Benefits:
 * - Optimal I/O patterns (no random access during training)
 * - Reproducible training (deterministic sample order)
 * - Enables out-of-core training on graphs larger than memory
 *
 * ## Usage
 *
 * @code
 *   SamplingConfig config;
 *   config.projection_name = "my_graph";
 *   config.sample_name = "training_v1";
 *   config.fanouts = {15, 10};
 *   config.batch_size = 1024;
 *
 *   OfflineSamplingEngine engine(projection_storage, config);
 *
 *   // With progress callback
 *   engine.set_progress_callback([](const SamplingProgress& p) {
 *       std::cout << "Progress: " << (p.progress() * 100) << "%" << std::endl;
 *       return true;  // Continue
 *   });
 *
 *   SamplingResult result = engine.run(db_folder);
 *
 *   if (result.success) {
 *       std::cout << "Generated " << result.total_samples << " samples" << std::endl;
 *   }
 * @endcode
 *
 * ## Thread Safety
 *
 * The engine is NOT thread-safe. Run from a single thread.
 * Cancellation via callback is safe from any thread.
 *
 * @see SamplingConfig for configuration options
 * @see SampleStorage for accessing generated samples
 */
class OfflineSamplingEngine {
public:
    /**
     * @brief Construct engine for a projection.
     *
     * @param storage Reference to projection storage (must outlive engine)
     * @param config Sampling configuration
     * @param db_folder Database root folder. Required when
     *        config.use_predefined_splits=true so the engine can locate the
     *        RowMapping at <db_folder>/gnn_features/<feature_name>.rmap.
     *        Empty string disables predefined-splits support.
     * @throws std::invalid_argument if config is invalid
     */
    OfflineSamplingEngine(
        GQL::ProjectionStorage& storage,
        const SamplingConfig& config,
        const std::filesystem::path& db_folder = {}
    );

    ~OfflineSamplingEngine();

    // Disable copy
    OfflineSamplingEngine(const OfflineSamplingEngine&) = delete;
    OfflineSamplingEngine& operator=(const OfflineSamplingEngine&) = delete;

    // Allow move
    OfflineSamplingEngine(OfflineSamplingEngine&&) noexcept;
    OfflineSamplingEngine& operator=(OfflineSamplingEngine&&) noexcept;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set progress callback.
     *
     * Called periodically during sampling. Return false to cancel.
     *
     * @param callback Progress callback function
     */
    void set_progress_callback(ProgressCallback callback);

    /**
     * @brief Set progress report interval.
     *
     * @param batches Report progress every N batches (default: 10)
     */
    void set_progress_interval(uint64_t batches);

    /**
     * @brief Enable/disable optimized batch sampling.
     *
     * When enabled, uses SortedBatchSampler for better I/O.
     * Default: true
     */
    void set_use_optimized_sampling(bool enabled);

    /**
     * @brief Get the sampling configuration.
     */
    const SamplingConfig& get_config() const;

    // =========================================================================
    // Execution
    // =========================================================================

    /**
     * @brief Run the complete offline sampling workflow.
     *
     * This is the main entry point. Following DiskGNN architecture, it:
     * 1. Validates configuration
     * 2. Collects and splits seed nodes
     * 3. Generates batches ONCE (no epoch loop - epochs belong to training)
     * 4. Samples k-hop neighborhoods for each batch
     * 5. Writes samples to storage
     * 6. Computes frequency statistics
     * 7. Finalizes storage
     *
     * Note: The same pre-sampled batches can be reused for multiple training
     * epochs - the training layer decides how many epochs and batch order.
     *
     * @param db_folder Database root folder (samples stored in <db>/samples/<name>/)
     * @return SamplingResult with success status and statistics
     */
    SamplingResult run(const std::filesystem::path& db_folder);

    /**
     * @brief Request cancellation of running operation.
     *
     * Safe to call from any thread.
     */
    void request_cancel();

    /**
     * @brief Check if cancellation was requested.
     */
    bool is_cancelled() const;

    // =========================================================================
    // Pre-flight Checks
    // =========================================================================

    /**
     * @brief Validate configuration and estimate resource usage.
     *
     * Call before run() to check for potential issues.
     *
     * @return Error message if invalid, empty string if OK
     */
    std::string validate();

    /**
     * @brief Estimate total samples to be generated.
     */
    uint64_t estimate_total_samples() const;

    /**
     * @brief Estimate storage size in bytes.
     *
     * Based on batch_size and fanouts.
     */
    uint64_t estimate_storage_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
