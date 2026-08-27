#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gnn/sampling/minhash_reorderer.h"
#include "graph_models/object_id.h"

namespace mdb::gnn {

// Forward declarations — avoid heavy includes in header
class FeatureMatrix;
class RowMapping;
class SampleStorage;

/**
 * @brief Orchestrates L3 (MinHash reorder) + L4 (packed batch) generation.
 *
 * Bridges offline sampling output (GraphSamples with node IDs) to
 * training-ready packed feature batches. Follows DiskGNN (SIGMOD'25):
 * disk-cache reordering via MinHash (§5.1), then per-batch feature
 * packing (§5.2).
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

    /// Translate ObjectIds to FeatureMatrix row indices.
    /// If inverse is non-null, applies reorder mapping: old_row → inverse[old_row].
    /// Throws with descriptive error if any ObjectId is missing from RowMapping
    /// or if a row index exceeds inverse bounds.
    /// Exposed for direct unit testing.
    static std::vector<uint64_t> translate_to_rows(
        const std::vector<ObjectId>&    oids,
        const RowMapping&               rm,
        const std::vector<uint64_t>*    inverse,
        uint64_t                        batch_id
    );

private:
    BatchMaterializer() = delete; // Static-only class
};

} // namespace mdb::gnn
