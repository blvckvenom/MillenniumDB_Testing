#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gnn/sampling/graph_sample.h"      // For SplitType
#include "gnn/sampling/sampling_config.h"

namespace mdb::gnn {

/**
 * @brief Metadata catalog for a pre-computed sample set.
 *
 * Stores all information needed to understand and use a sample set:
 * - Configuration used for sampling
 * - Statistics (batch counts, node counts)
 * - Creation timestamp
 * - Version information
 *
 * ## File Format
 *
 * Binary format with magic number for validation:
 * ```
 * Offset  Size     Field
 * 0       4        Magic ("GNNS" = 0x534E4E47)
 * 4       4        Version (currently 2)
 * 8       8        Created timestamp (Unix epoch)
 * 16      8        Total batches
 * 24      8        Train batches
 * 32      8        Validation batches
 * 40      8        Test batches
 * 48      8        Unique nodes
 * 56      8        Total edges
 * 64      8        Batch size
 * 72      8        Random seed
 * 80      8        Num layers (K)
 * 88      K*8      Fanouts array
 * ...     var      Projection name (length-prefixed string)
 * ...     var      Sample name (length-prefixed string)
 * ...     8        Sample content fingerprint (v3+ only; 0 = absent/UNKNOWN)
 * ```
 *
 * V1 compatibility: between total_edges and batch_size, v1 has an
 * extra uint64 legacy_num_epochs field (read and discarded).
 *
 * V2 compatibility: v1/v2 catalogs have no trailing sample_content_fp; it
 * reads as 0 (UNKNOWN), which routes the feature store to recompute (the safe
 * direction — never silently reuse a store whose sample provenance is unknown).
 *
 * @see SampleStorage for the actual sample data
 */
struct SampleCatalog {
    // =========================================================================
    // Format Constants
    // =========================================================================

    static constexpr uint32_t MAGIC = 0x534E4E47;  // "GNNS" in little-endian
    static constexpr uint32_t VERSION = 3;
    static constexpr const char* CATALOG_FILENAME = "catalog.dat";

    // =========================================================================
    // Identification
    // =========================================================================

    std::string projection_name;  ///< Source projection
    std::string sample_name;      ///< Name of this sample set

    // =========================================================================
    // Configuration (from SamplingConfig)
    // =========================================================================

    uint64_t batch_size;          ///< Seeds per batch
    uint64_t random_seed;         ///< Random seed used
    std::vector<uint64_t> fanouts; ///< Fanout per layer

    // =========================================================================
    // Statistics
    // =========================================================================

    uint64_t total_batches;       ///< Total batches across all epochs and splits
    uint64_t train_batches;       ///< Batches for training
    uint64_t validation_batches;  ///< Batches for validation
    uint64_t test_batches;        ///< Batches for testing
    uint64_t unique_nodes;        ///< Unique nodes across all samples
    uint64_t total_edges;         ///< Total edges across all samples

    // =========================================================================
    // Content fingerprint (STEP 8)
    // =========================================================================

    /// Layout-independent XOR fold of per-batch content hashes (see
    /// sample_fingerprint.h). 0 = absent/UNKNOWN (v1/v2 catalogs, or a sample
    /// written before STEP 8). Consumed by FourLevelStore to decide reuse vs
    /// recompute of the feature store. Persisted only in catalog v3+.
    uint64_t sample_content_fp = 0;

    // =========================================================================
    // Timestamps
    // =========================================================================

    std::chrono::system_clock::time_point created_at;

    // =========================================================================
    // Constructors
    // =========================================================================

    SampleCatalog() = default;

    /**
     * @brief Create catalog from sampling configuration.
     */
    explicit SampleCatalog(const SamplingConfig& config)
        : projection_name(config.projection_name)
        , sample_name(config.sample_name)
        , batch_size(config.batch_size)
        , random_seed(config.random_seed)
        , fanouts(config.fanouts)
        , total_batches(0)
        , train_batches(0)
        , validation_batches(0)
        , test_batches(0)
        , unique_nodes(0)
        , total_edges(0)
        , sample_content_fp(0)
        , created_at(std::chrono::system_clock::now())
    {
    }

    // =========================================================================
    // Persistence
    // =========================================================================

    /**
     * @brief Save catalog to file.
     *
     * @param directory Directory to save in (catalog.dat will be created)
     * @throws std::runtime_error on write failure
     */
    void save(const std::filesystem::path& directory) const {
        std::filesystem::create_directories(directory);
        std::filesystem::path filepath = directory / CATALOG_FILENAME;

        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open catalog file for writing: " + filepath.string());
        }

        write_binary(out);

        if (!out) {
            throw std::runtime_error("Failed to write catalog file: " + filepath.string());
        }
    }

    /**
     * @brief Load catalog from file.
     *
     * @param directory Directory containing catalog.dat
     * @return Loaded catalog
     * @throws std::runtime_error on read failure or invalid format
     */
    static SampleCatalog load(const std::filesystem::path& directory) {
        std::filesystem::path filepath = directory / CATALOG_FILENAME;

        std::ifstream in(filepath, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open catalog file: " + filepath.string());
        }

        SampleCatalog catalog;
        catalog.read_binary(in);

        return catalog;
    }

    /**
     * @brief Check if a catalog exists in directory.
     */
    static bool exists(const std::filesystem::path& directory) {
        return std::filesystem::exists(directory / CATALOG_FILENAME);
    }

    // =========================================================================
    // Utility
    // =========================================================================

    /**
     * @brief Get number of GNN layers.
     */
    size_t num_layers() const {
        return fanouts.size();
    }

    /**
     * @brief Get human-readable summary.
     */
    std::string summary() const {
        std::string s;
        s += "SampleCatalog: " + sample_name + "\n";
        s += "  Projection: " + projection_name + "\n";
        s += "  Batch size: " + std::to_string(batch_size) + "\n";
        s += "  Layers: " + std::to_string(fanouts.size()) + " (fanouts: ";
        for (size_t i = 0; i < fanouts.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(fanouts[i]);
        }
        s += ")\n";
        s += "  Total batches: " + std::to_string(total_batches) + "\n";
        s += "  Train/Val/Test: " + std::to_string(train_batches) + "/" +
             std::to_string(validation_batches) + "/" + std::to_string(test_batches) + "\n";
        s += "  Unique nodes: " + std::to_string(unique_nodes) + "\n";
        s += "  Total edges: " + std::to_string(total_edges) + "\n";
        return s;
    }

private:
    // =========================================================================
    // Binary I/O Helpers
    // =========================================================================

    template<typename T>
    static void write_value(std::ostream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template<typename T>
    static T read_value(std::istream& in) {
        T value;
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    static void write_string(std::ostream& out, const std::string& str) {
        uint64_t len = str.size();
        write_value(out, len);
        out.write(str.data(), len);
    }

    static std::string read_string(std::istream& in) {
        uint64_t len = read_value<uint64_t>(in);
        if (len > 1'000'000) {
            throw std::runtime_error(
                "SampleCatalog: string length " + std::to_string(len) +
                " exceeds 1MB limit (file likely corrupted)");
        }
        std::string str(len, '\0');
        in.read(str.data(), len);
        return str;
    }

    void write_binary(std::ostream& out) const {
        // Header
        write_value(out, MAGIC);
        write_value(out, VERSION);

        // Timestamp
        auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
            created_at.time_since_epoch()
        ).count();
        write_value(out, static_cast<int64_t>(epoch));

        // Statistics
        write_value(out, total_batches);
        write_value(out, train_batches);
        write_value(out, validation_batches);
        write_value(out, test_batches);
        write_value(out, unique_nodes);
        write_value(out, total_edges);

        // Configuration (v2: no num_epochs field)
        write_value(out, batch_size);
        write_value(out, random_seed);

        // Fanouts
        write_value(out, static_cast<uint64_t>(fanouts.size()));
        for (uint64_t f : fanouts) {
            write_value(out, f);
        }

        // Strings
        write_string(out, projection_name);
        write_string(out, sample_name);

        // Content fingerprint (v3+, append-only at the tail).
        write_value(out, sample_content_fp);
    }

    void read_binary(std::istream& in) {
        // Header validation
        uint32_t magic = read_value<uint32_t>(in);
        if (magic != MAGIC) {
            throw std::runtime_error("Invalid catalog magic number");
        }

        uint32_t version = read_value<uint32_t>(in);
        if (version != VERSION && version != 1) {
            throw std::runtime_error("Unsupported catalog version: " + std::to_string(version));
        }

        // Timestamp (Unix epoch, not training epoch)
        int64_t unix_epoch = read_value<int64_t>(in);
        created_at = std::chrono::system_clock::time_point(std::chrono::seconds(unix_epoch));

        // Statistics
        total_batches = read_value<uint64_t>(in);
        train_batches = read_value<uint64_t>(in);
        validation_batches = read_value<uint64_t>(in);
        test_batches = read_value<uint64_t>(in);
        unique_nodes = read_value<uint64_t>(in);
        total_edges = read_value<uint64_t>(in);

        // Configuration
        // v1 compatibility: read and discard legacy num_epochs field
        if (version == 1) {
            [[maybe_unused]] uint64_t legacy_num_epochs = read_value<uint64_t>(in);
        }
        batch_size = read_value<uint64_t>(in);
        random_seed = read_value<uint64_t>(in);

        // Fanouts
        uint64_t num_fanouts = read_value<uint64_t>(in);
        if (num_fanouts > 100) {
            throw std::runtime_error(
                "SampleCatalog: fanout count " + std::to_string(num_fanouts) +
                " exceeds limit of 100 (file likely corrupted)");
        }
        fanouts.resize(num_fanouts);
        for (uint64_t i = 0; i < num_fanouts; ++i) {
            fanouts[i] = read_value<uint64_t>(in);
        }

        // Strings
        projection_name = read_string(in);
        sample_name = read_string(in);

        // Content fingerprint (v3+ only). v1/v2 catalogs have no trailing
        // field; leave it 0 = UNKNOWN so the feature store recomputes.
        if (version >= 3) {
            sample_content_fp = read_value<uint64_t>(in);
        } else {
            sample_content_fp = 0;
        }

        if (!in) {
            throw std::runtime_error("Error reading catalog file");
        }
    }
};

} // namespace mdb::gnn
