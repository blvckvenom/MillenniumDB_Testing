#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

#include "graph_models/object_id.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sampling_config.h"

namespace mdb::gnn {

// Forward declaration — full definition in gnn/storage/row_mapping.h.
// Used by the dense frequency path; avoids adding a heavy include to this header.
class RowMapping;

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
 * **frequency.dat v1**: Magic (`FREQ`), version=1 (uint32), entry count (uint64),
 * then `[node_id, count]` pairs (each uint64).
 *
 * **frequency.dat v2** (dense): Magic (`FREQ`), version=2 (uint32),
 * count=N (uint64), then N consecutive uint64 frequency values indexed
 * by RowMapping row_index. Requires a RowMapping for interpretation.
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
     * @brief Create new sample storage with RowMapping for dense tracking.
     *
     * When a RowMapping is provided, frequency counting uses a dense
     * vector<uint64_t>[row_index] (8 bytes/node) instead of an
     * unordered_map (approximately 50 bytes/node), and unique-node tracking uses a
     * vector<bool> bitset (1 bit/node) instead of unordered_set
     * (approximately 50 bytes/node). At 100M nodes this reduces RAM from 9.5 GB to under 1 GB.
     *
     * The RowMapping must outlive the SampleStorage write phase (until
     * finalize() is called).
     *
     * @param db_folder Database root folder
     * @param config Sampling configuration
     * @param row_mapping RowMapping for ObjectId to row_index translation
     * @return New SampleStorage in write mode with dense tracking enabled
     * @throws std::runtime_error if storage already exists or creation fails
     */
    static SampleStorage create(
        const std::filesystem::path& db_folder,
        const SamplingConfig& config,
        const RowMapping& row_mapping
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

    // =========================================================================
    // Deserialized-sample cache (train-time hot path)
    // =========================================================================

    /**
     * @brief Cache deserialized GraphSamples in RAM, bounded by a byte budget.
     *
     * The cost model showed gnn_train re-reads + re-deserializes the entire
     * batches.dat from disk on EVERY epoch (the per-batch read_sample on the
     * assemble hot path), and Fix #22's fadvise/madvise DONTNEED evicts those
     * pages so even a sample set that fits in RAM is re-fetched each epoch.
     *
     * With a budget set, read_sample serves a previously-read batch from an
     * in-RAM LRU of deserialized samples — skipping both the disk read and the
     * deserialize. When batches.dat fits in the budget (cora/arxiv), epochs
     * 2..N do zero sample-structure I/O; when it does not (papers100M), the LRU
     * holds the hot subset and the DONTNEED hints still relieve the cold tail.
     *
     * @param budget_bytes Max bytes of cached samples. 0 disables (default).
     */
    void set_sample_cache_budget_bytes(size_t budget_bytes);

    /**
     * @brief Skip the per-layer edge_ids blocks when deserializing samples.
     *
     * edge_ids are ~half the serialized bytes of a deep-fanout sample and are
     * NOT consumed by the training / embedding read path (only the src/dst
     * indices are). With this enabled, read_sample seeks past the edge_ids
     * blocks — roughly halving the per-batch sample read I/O — and returns
     * samples whose edges_per_layer[k].edge_ids are empty. Off by default;
     * the training setup turns it on. Affects only reads from THIS storage.
     */
    void set_skip_edge_ids_on_read(bool enable);

    /**
     * @brief Skip the per-layer src/dst edge index blocks when deserializing.
     *
     * When set, read_sample deserializes WITHOUT the per-layer src/dst edge
     * indices — used when baked blocks supply the edge structure. Off by
     * default. Affects only reads from THIS storage. Caller MUST supply the
     * edge structure from another source (a baked computation-graph block) when
     * enabled; otherwise the assembled graph is edgeless (src/dst left empty)
     * and training silently produces wrong results.
     */
    void set_skip_edges_on_read(bool enable);

    struct SampleCacheStats {
        uint64_t hits = 0;        ///< read_sample served from the RAM cache
        uint64_t misses = 0;      ///< read_sample fell through to disk
        uint64_t evictions = 0;   ///< LRU evictions due to budget pressure
        size_t   bytes = 0;       ///< current cached bytes (estimated)
        size_t   budget = 0;      ///< configured budget
        size_t   entries = 0;     ///< cached sample count
    };

    /// Snapshot of the deserialized-sample cache counters.
    SampleCacheStats sample_cache_stats() const;

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
     * @brief Get dense frequency vector indexed by RowMapping row_index.
     *
     * Returns a vector of size rm.size() where result[row_index] is the
     * number of times that node appeared across all samples.
     *
     * Works with both file formats:
     * - v2 (dense): Returns stored vector directly (zero-copy path).
     * - v1 (sparse oid/count pairs): Converts on the fly via RowMapping.
     *
     * Primary consumer: FourLevelStore::build() for cache-tier assignment.
     *
     * @param rm RowMapping for ObjectId to row_index translation
     * @return Dense frequency vector, empty if no frequency data available
     */
    std::vector<uint64_t> get_dense_frequencies(const RowMapping& rm);

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
