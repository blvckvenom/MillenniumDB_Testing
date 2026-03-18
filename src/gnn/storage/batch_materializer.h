#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "gnn/sampling/minhash_reorderer.h"

namespace mdb::gnn {

// Forward declarations — avoid heavy includes in header
class FeatureMatrix;
class RowMapping;
class SampleStorage;

/**
 * @brief Orchestrates L3 (MinHash reorder) + L4 (packed batch) generation.
 *
 * Bridges offline sampling output (GraphSamples with node IDs) to
 * training-ready packed feature batches. Implements DiskGNN Section 5:
 * disk cache reordering via MinHash, then per-batch feature packing.
 *
 * All building blocks are existing tested components:
 * - FeatureMatrix::extract_rows() for feature lookup
 * - FeatureMatrix::create_reordered() for L3
 * - RowMapping::find() for ObjectId → row translation
 * - MinHashReorderer for access pattern clustering
 * - PackedBatchWriter for L4 file generation
 *
 * Usage:
 *   auto result = BatchMaterializer::materialize(fm, rm, samples, config, db, "node_features");
 */
class BatchMaterializer {
public:
    struct Config {
        bool reorder = true;
        bool force   = false;
        MinHashReorderer::Config minhash;
    };

    struct Result {
        uint64_t total_batches   = 0;
        bool     reordered       = false;
        int64_t  reorder_time_ms = 0;
        int64_t  pack_time_ms    = 0;
        int64_t  total_time_ms   = 0;
        std::string packed_dir;
    };

    /// Run the full materialization pipeline (L3 + L4).
    /// Reads batches from SampleStorage, optionally reorders FeatureMatrix,
    /// then generates per-batch packed files.
    /// Throws on error. Cleans up partial outputs on failure.
    static Result materialize(
        const FeatureMatrix& features,
        const RowMapping&    row_mapping,
        SampleStorage&       samples,
        const Config&        config,
        const std::filesystem::path& db_folder,
        const std::string&   feature_name
    );

private:
    BatchMaterializer() = delete; // Static-only class
};

} // namespace mdb::gnn
