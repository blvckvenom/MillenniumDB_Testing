#include "gnn/storage/batch_materializer.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/packed_batch_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/graph_sample.h"
#include "graph_models/object_id.h"

namespace fs = std::filesystem;

namespace mdb::gnn {

/// Translate ObjectIds from a GraphSample to row indices in a FeatureMatrix.
/// If inverse is non-null, applies reorder mapping: old_row → inverse[old_row].
static std::vector<uint64_t> translate_to_rows(
    const std::vector<ObjectId>&    oids,
    const RowMapping&               rm,
    const std::vector<uint64_t>*    inverse,
    uint64_t                        batch_id)
{
    std::vector<uint64_t> rows;
    rows.reserve(oids.size());

    for (const auto& oid : oids) {
        auto old_row = rm.find(oid);
        if (!old_row.has_value()) {
            throw std::runtime_error(
                "Node ObjectId " + std::to_string(oid.id) +
                " in batch " + std::to_string(batch_id) +
                " has no corresponding feature row.\n"
                "The sample and FeatureMatrix were likely created from different node sets.");
        }
        if (inverse) {
            if (*old_row >= inverse->size()) {
                throw std::runtime_error(
                    "Row index " + std::to_string(*old_row) +
                    " out of bounds for inverse mapping (size " +
                    std::to_string(inverse->size()) + ") in batch " +
                    std::to_string(batch_id));
            }
            rows.push_back((*inverse)[*old_row]);
        } else {
            rows.push_back(*old_row);
        }
    }
    return rows;
}

BatchMaterializer::Result BatchMaterializer::materialize(
    const FeatureMatrix& features,
    const RowMapping&    row_mapping,
    SampleStorage&       samples,
    const Config&        config,
    const fs::path&      db_folder,
    const std::string&   feature_name)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    Result result;

    const auto& catalog   = samples.get_catalog();
    uint64_t num_batches  = catalog.total_batches;
    uint64_t N            = features.num_rows();

    // --- Output paths ---
    auto gnn_dir        = db_folder / "gnn_features";
    auto reordered_fmat = gnn_dir / (feature_name + "_reordered.fmat");
    auto reordered_rmap = gnn_dir / (feature_name + "_reordered.rmap");
    auto sample_dir     = SampleStorage::get_storage_path(db_folder.string(), catalog.sample_name);
    auto packed_dir     = fs::path(sample_dir) / "packed";
    result.packed_dir   = packed_dir.string();

    // --- Force cleanup ---
    if (config.force) {
        std::error_code ec;
        fs::remove_all(packed_dir, ec);
        fs::remove(reordered_fmat, ec);
        fs::remove(reordered_rmap, ec);
    }

    // --- Pre-condition checks ---
    if (fs::exists(packed_dir)) {
        throw std::runtime_error(
            "Packed batches already exist at: " + packed_dir.string() + "\n\n"
            "Solutions:\n"
            "  1. Re-run with force option: {force: 1}\n"
            "  2. Delete the sample and re-create:\n"
            "     CALL gnn_sample_drop('" + catalog.sample_name + "')");
    }
    if (config.reorder && fs::exists(reordered_fmat)) {
        throw std::runtime_error(
            "Reordered matrix already exists at: " + reordered_fmat.string() + "\n\n"
            "Solutions:\n"
            "  1. Re-run with force option: {force: 1}\n"
            "  2. Skip reordering: {reorder: 0}");
    }

    // --- Cleanup guard ---
    try {

    // =========================================================================
    // L3: MinHash Reordering (optional)
    // =========================================================================
    const FeatureMatrix* active_fm = &features;
    std::unique_ptr<FeatureMatrix> reordered_fm_holder;
    std::vector<uint64_t> inverse;

    if (config.reorder) {
        auto l3_start = std::chrono::high_resolution_clock::now();

        // Step 1: Build access graph from all batches
        MinHashReorderer reorderer(config.minhash);
        reorderer.build_access_graph(num_batches,
            [&](uint64_t batch_id) -> std::vector<uint64_t> {
                auto sample = samples.read_sample(batch_id);
                std::vector<uint64_t> row_ids;
                row_ids.reserve(sample.all_unique_nodes.size());
                for (const auto& oid : sample.all_unique_nodes) {
                    auto row = row_mapping.find(oid);
                    if (row.has_value()) {
                        row_ids.push_back(*row);
                    }
                    // Silently skip unmapped nodes in access graph (safe for MinHash)
                }
                return row_ids;
            });

        // Step 2: Compute permutation + inverse
        auto permutation = reorderer.compute_permutation(N);
        inverse = MinHashReorderer::compute_inverse(permutation);

        // Step 3a: Create reordered FeatureMatrix
        auto reordered = FeatureMatrix::create_reordered(features, permutation, reordered_fmat);

        // Step 3b: Create reordered RowMapping (maintain rm[i] ↔ fm[i] invariant)
        std::vector<ObjectId> reordered_ids(N);
        for (uint64_t i = 0; i < N; ++i) {
            reordered_ids[i] = row_mapping.get(permutation[i]);
        }
        RowMapping::create(reordered_rmap, reordered_ids);

        // Use reordered matrix for packing
        reordered_fm_holder = std::make_unique<FeatureMatrix>(std::move(reordered));
        active_fm = reordered_fm_holder.get();
        result.reordered = true;

        auto l3_end = std::chrono::high_resolution_clock::now();
        result.reorder_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            l3_end - l3_start).count();
    }

    // =========================================================================
    // L4: Pack Batches
    // =========================================================================
    auto l4_start = std::chrono::high_resolution_clock::now();

    const std::vector<uint64_t>* inv_ptr = config.reorder ? &inverse : nullptr;

    generate_packed_batches(
        *active_fm,
        num_batches,
        [&](uint64_t batch_id) -> std::vector<uint64_t> {
            auto sample = samples.read_sample(batch_id);
            return translate_to_rows(sample.all_unique_nodes, row_mapping, inv_ptr, batch_id);
        },
        packed_dir);

    result.total_batches = num_batches;

    auto l4_end = std::chrono::high_resolution_clock::now();
    result.pack_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        l4_end - l4_start).count();

    } catch (...) {
        // Best-effort cleanup of partial outputs
        std::error_code ec;
        fs::remove_all(packed_dir, ec);
        if (config.reorder) {
            fs::remove(reordered_fmat, ec);
            fs::remove(reordered_rmap, ec);
        }
        throw;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start).count();

    return result;
}

} // namespace mdb::gnn
