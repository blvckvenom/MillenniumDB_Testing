#include "gnn/storage/batch_materializer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/packed_batch_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/graph_sample.h"
#include "graph_models/object_id.h"
#include "misc/ablation_registry.h"

namespace fs = std::filesystem;

namespace mdb::gnn {

std::vector<uint64_t> BatchMaterializer::translate_to_rows(
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

    std::cout << "\n"
              << "========================================================\n"
              << "[Materialize] STARTING\n"
              << "========================================================\n"
              << "  sample_name:  " << catalog.sample_name      << "\n"
              << "  feature_name: " << feature_name             << "\n"
              << "  num_batches:  " << num_batches              << "\n"
              << "  feature_rows: " << N                        << "\n"
              << "  feature_cols: " << features.num_cols()      << "\n"
              << "  reorder:      " << (config.reorder ? "yes" : "no") << "\n"
              << "  numHashes:    " << config.minhash.num_hashes << "\n"
              << "========================================================\n"
              << std::flush;

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

        std::cout << "[Materialize] L3 reorder phase: building MinHash access graph from "
                  << num_batches << " batches...\n" << std::flush;

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

        std::cout << "[Materialize] computing MinHash permutation over "
                  << N << " rows..." << std::flush;
        auto perm_t0 = std::chrono::steady_clock::now();
        // Step 2: Compute permutation + inverse
        auto permutation = reorderer.compute_permutation(N);
        inverse = MinHashReorderer::compute_inverse(permutation);
        auto perm_t1 = std::chrono::steady_clock::now();
        std::cout << " done in "
                  << std::chrono::duration_cast<std::chrono::seconds>(perm_t1 - perm_t0).count()
                  << "s\n[Materialize] writing reordered feature matrix ("
                  << (features.num_rows() * features.num_cols() * sizeof(float) / (1024ULL * 1024ULL))
                  << " MB) → " << reordered_fmat.string() << "\n" << std::flush;

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

    auto batch_provider_lambda = [&](uint64_t batch_id) -> std::vector<uint64_t> {
        auto sample = samples.read_sample(batch_id);
        return translate_to_rows(sample.all_unique_nodes, row_mapping, inv_ptr, batch_id);
    };

    // Spec B1: dispatch between classic and partitioned packer via env var.
    // MDB_BATCH_PACKER=partitioned activates the DiskGNN-style inverted loop
    // (1× sequential .fmat scan + scatter pwrites). Default classic preserves
    // pre-2026-04-27 behavior; flip to partitioned only after measuring on
    // your dataset (mirrors MDB_PROJECTION_SORTER pattern from ADR-004).
    // Resolved through the registry because the two arms differ only in write
    // pattern: a run that silently took the default is otherwise
    // indistinguishable from one that asked for it. "partitioned" is the only
    // value the comparison below ever honoured, so anything else still means
    // classic, now with the rejected spelling on the record.
    static const std::string packer =
        Ablation::choice("MDB_BATCH_PACKER", "classic", {"classic", "partitioned"});
    const bool use_partitioned = (packer == "partitioned");

    if (use_partitioned) {
        std::cout << "[Materialize] L4 packer mode: partitioned (Spec B1)\n"
                  << std::flush;
        // Resolved on the partitioned path only, so the log records the knob
        // where it actually applies. A value that does not parse is reported by
        // the registry instead of vanishing into the default.
        static const long partition_mb = Ablation::number("MDB_BATCH_PARTITION_MB", 256);
        size_t partition_bytes = 256ULL * 1024 * 1024;
        if (partition_mb > 0) {
            partition_bytes = static_cast<size_t>(partition_mb) * 1024 * 1024;
        }
        // OID provider: feed sample.all_unique_nodes so the partitioned
        // packer can write a v2 OID table alongside the data section. The
        // OID table is required for downstream consumers to map physical
        // file row → ObjectId, since the partitioned packer writes rows in
        // partition iteration order (not sample-input order).
        auto oid_provider_lambda = [&](uint64_t batch_id) -> std::vector<ObjectId> {
            auto sample = samples.read_sample(batch_id);
            return sample.all_unique_nodes;
        };
        generate_packed_batches_partitioned(
            *active_fm, num_batches, batch_provider_lambda,
            packed_dir, partition_bytes, oid_provider_lambda);
    } else {
        std::cout << "[Materialize] L4 packer mode: classic\n" << std::flush;
        generate_packed_batches(
            *active_fm, num_batches, batch_provider_lambda, packed_dir);
    }

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

    std::cout << "[Materialize] DONE — total "
              << (result.total_time_ms / 1000) << "s ("
              << "reorder=" << (result.reorder_time_ms / 1000) << "s, "
              << "pack=" << (result.pack_time_ms / 1000) << "s)\n"
              << std::flush;

    return result;
}

} // namespace mdb::gnn
