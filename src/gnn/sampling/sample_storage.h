#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "graph_models/object_id.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sampling_config.h"

namespace mdb::gnn {

/**
 * @brief Persistent storage for pre-computed GNN samples.
 *
 * Stores GraphSamples to disk using flat binary files with an index for
 * efficient random access by batch_id. This enables:
 * - Reuse of samples across training runs (no re-sampling)
 * - Out-of-core training (samples larger than memory)
 * - Reproducible training (deterministic sample order)
 *
 * ## Storage Layout
 *
 * ```
 * <db_folder>/samples/<sample_name>/
 * ├── catalog.dat          # Metadata (SampleCatalog)
 * ├── batches.dat          # Serialized GraphSample objects (binary)
 * ├── batches.idx          # Index: [offset, size] pairs per batch_id
 * └── frequency.dat        # Node frequency data
 * ```
 *
 * ## File Formats
 *
 * **batches.dat**: Magic (`BTCH`), version (uint32), then concatenated
 * serialized GraphSample blobs.
 *
 * **batches.idx**: Magic (`INDX`), version (uint32), entry count (uint64),
 * then `[offset, size]` pairs (each uint64) — one per batch_id.
 *
 * **frequency.dat**: Magic (`FREQ`), version (uint32), entry count (uint64),
 * then `[node_id, count]` pairs (each uint64).
 *
 * ## Usage
 *
 * @code
 *   // Writing samples
 *   SampleStorage storage = SampleStorage::create(db_path, config);
 *   storage.write_sample(sample);
 *   storage.finalize();  // Must call before reading
 *
 *   // Reading samples
 *   SampleStorage storage = SampleStorage::open(db_path / "samples" / name);
 *   GraphSample sample = storage.read_sample(batch_id);
 *   auto catalog = storage.get_catalog();
 * @endcode
 *
 * @see SampleCatalog for metadata format
 * @see GraphSample for sample data structure
 */
class SampleStorage {
public:
    ~SampleStorage();

    // Disable copy
    SampleStorage(const SampleStorage&) = delete;
    SampleStorage& operator=(const SampleStorage&) = delete;

    // Allow move
    SampleStorage(SampleStorage&&) noexcept;
    SampleStorage& operator=(SampleStorage&&) noexcept;

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * @brief Create new sample storage.
     *
     * Creates directory structure and initializes storage files.
     *
     * @param db_folder Database root folder
     * @param config Sampling configuration (determines sample_name)
     * @return New SampleStorage in write mode
     * @throws std::runtime_error if storage already exists or creation fails
     */
    static SampleStorage create(
        const std::filesystem::path& db_folder,
        const SamplingConfig& config
    );

    /**
     * @brief Open existing sample storage for reading.
     *
     * @param storage_path Path to sample storage directory
     * @return SampleStorage in read mode
     * @throws std::runtime_error if storage doesn't exist or is invalid
     */
    static SampleStorage open(const std::filesystem::path& storage_path);

    /**
     * @brief Check if sample storage exists.
     *
     * @param db_folder Database root folder
     * @param sample_name Name of sample set
     * @return true if storage exists and is valid
     */
    static bool exists(
        const std::filesystem::path& db_folder,
        const std::string& sample_name
    );

    /**
     * @brief Get path to sample storage.
     *
     * @param db_folder Database root folder
     * @param sample_name Name of sample set
     * @return Path to storage directory
     */
    static std::filesystem::path get_storage_path(
        const std::filesystem::path& db_folder,
        const std::string& sample_name
    );

    // =========================================================================
    // Write Interface
    // =========================================================================

    /**
     * @brief Write a single sample to storage.
     *
     * Samples must be written in batch_id order for optimal index performance.
     *
     * @param sample Sample to write
     * @throws std::runtime_error if not in write mode or write fails
     */
    void write_sample(const GraphSample& sample);

    /**
     * @brief Finalize storage after writing all samples.
     *
     * Must be called after all samples are written and before reading.
     * Updates catalog with final statistics.
     *
     * @throws std::runtime_error if finalization fails
     */
    void finalize();

    // =========================================================================
    // Read Interface
    // =========================================================================

    /**
     * @brief Read a sample by batch ID.
     *
     * @param batch_id Batch to retrieve
     * @return Reconstructed GraphSample
     * @throws std::runtime_error if batch doesn't exist or read fails
     */
    GraphSample read_sample(uint64_t batch_id);

    /**
     * @brief Read multiple samples.
     *
     * More efficient than multiple read_sample calls due to iterator reuse.
     *
     * @param batch_ids Batches to retrieve
     * @return Vector of GraphSamples in same order as batch_ids
     */
    std::vector<GraphSample> read_samples(const std::vector<uint64_t>& batch_ids);

    /**
     * @brief Get batch IDs for a specific split.
     *
     * Following DiskGNN architecture, batches are generated ONCE and don't
     * have epoch information. The training layer decides epoch iteration.
     *
     * @param split Train/validation/test
     * @return Vector of batch IDs for the requested split
     */
    std::vector<uint64_t> get_batch_ids(SplitType split);

    /**
     * @brief Get all batch IDs.
     */
    std::vector<uint64_t> get_all_batch_ids();

    // =========================================================================
    // Metadata
    // =========================================================================

    /**
     * @brief Get the sample catalog.
     */
    const SampleCatalog& get_catalog() const;

    /**
     * @brief Get storage directory path.
     */
    const std::filesystem::path& get_path() const;

    /**
     * @brief Check if storage is in write mode.
     */
    bool is_write_mode() const;

    // =========================================================================
    // Frequency Data (for cache optimization)
    // =========================================================================

    /**
     * @brief Get node access frequency.
     *
     * Returns how many times each node appears across all samples.
     * Used for cache priority in Phase 3.
     *
     * @return Map of node_id -> access_count
     */
    std::unordered_map<uint64_t, uint64_t> get_node_frequencies();

    /**
     * @brief Get top-K most frequent nodes.
     *
     * @param k Number of nodes to return
     * @return Vector of (node_id, frequency) pairs, sorted by frequency descending
     */
    std::vector<std::pair<ObjectId, uint64_t>> get_top_frequent_nodes(size_t k);

private:
    SampleStorage();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
