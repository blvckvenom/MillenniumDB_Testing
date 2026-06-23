#include "query/procedure/builtin/gnn_materialize_batches_procedure.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

#include "query/procedure/procedure_context.h"

#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/batch_materializer.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/gql/gql_model.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

void GnnMaterializeBatchesProcedure::execute(ProcedureContext& ctx) {
    using namespace mdb::gnn;

    // Deprecation notice (audit 2026-06-04): gnn_materialize_batches is redundant.
    // gnn_build_feature_store is self-sufficient — it reorders + packs directly
    // from the sample. This procedure's packed/ output is not consumed by the
    // downstream stages (it is deleted by the next stage). Prefer calling
    // gnn_build_feature_store directly; this procedure may be removed in a future
    // release. (Kept for now to avoid breaking existing user scripts.)
    std::cerr << "[gnn_materialize_batches] DEPRECATED/REDUNDANT: prefer "
                 "gnn_build_feature_store (self-sufficient); this procedure's "
                 "packed/ output is unused downstream and may be removed.\n";

    // =========================================================================
    // Step 1: Parse arguments
    // =========================================================================
    if (ctx.arguments.size() < 2 || ctx.arguments.size() > 3) {
        throw std::runtime_error(
            "gnn_materialize_batches requires 2-3 arguments.\n\n"
            "Usage: CALL gnn_materialize_batches(sampleName, featureName [, options])\n"
            "Example: CALL gnn_materialize_batches('my_sample', 'node_features', {reorder: 1})");
    }

    // Parse sampleName
    std::string sample_name;
    try {
        sample_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid sampleName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING.\n"
            "Example: CALL gnn_materialize_batches('my_sample', 'node_features')");
    }
    if (sample_name.empty()) {
        throw std::runtime_error("sampleName cannot be empty.");
    }
    validate_safe_name(sample_name, "sampleName");

    // Parse featureName
    std::string feature_name;
    try {
        feature_name = ctx.get_string_argument(1);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid featureName parameter: " + std::string(e.what()) + "\n\n"
            "The second parameter must be a STRING.\n"
            "Example: CALL gnn_materialize_batches('my_sample', 'node_features')");
    }
    if (feature_name.empty()) {
        throw std::runtime_error("featureName cannot be empty.");
    }
    validate_safe_name(feature_name, "featureName");

    // =========================================================================
    // Step 2: Parse options
    // =========================================================================
    BatchMaterializer::Config config;

    if (ctx.arguments.size() == 3) {
        DictOptions opts(ctx.get_argument(2));

        if (auto v = opts.get_bool("reorder")) {
            config.reorder = *v;
        }
        if (auto v = opts.get_bool("force")) {
            config.force = *v;
        }
        if (auto v = opts.get_string("strategy")) {
            std::string s = *v;
            std::transform(s.begin(), s.end(), s.begin(), ::toupper);
            if (s == "SEGMENTED") {
                config.minhash.strategy = MinHashReorderer::Strategy::SEGMENTED;
            } else if (s == "MULTIPASS_BOUNDED") {
                config.minhash.strategy = MinHashReorderer::Strategy::MULTIPASS_BOUNDED;
            } else {
                throw std::runtime_error(
                    "Invalid strategy: '" + *v + "'. Must be 'SEGMENTED' or 'MULTIPASS_BOUNDED'.");
            }
        }
        if (auto v = opts.get_int("numHashes")) {
            if (*v <= 0) throw std::runtime_error("numHashes must be positive, got: " + std::to_string(*v));
            config.minhash.num_hashes = static_cast<uint32_t>(*v);
        }
        if (auto v = opts.get_int("segmentSize")) {
            if (*v <= 0) throw std::runtime_error("segmentSize must be positive, got: " + std::to_string(*v));
            config.minhash.segment_size = static_cast<uint32_t>(*v);
        }
    }

    // =========================================================================
    // Step 3: Validate inputs exist
    // =========================================================================
    std::string db_folder = get_db_folder();

    // Validate feature name is registered in catalog
    const auto& names = gql_model.catalog.gnn_feature_names;
    if (std::find(names.begin(), names.end(), feature_name) == names.end()) {
        throw std::runtime_error(
            format_not_found_error("feature", feature_name, names,
                "Import with: mdb import data.gql <db> --with-tensors features.npy"));
    }

    // Validate FeatureMatrix and RowMapping files exist
    auto fmat_path = fs::path(db_folder) / "gnn_features" / (feature_name + ".fmat");
    auto rmap_path = fs::path(db_folder) / "gnn_features" / (feature_name + ".rmap");
    if (!fs::exists(fmat_path)) {
        throw std::runtime_error("FeatureMatrix not found at: " + fmat_path.string());
    }
    if (!fs::exists(rmap_path)) {
        throw std::runtime_error("RowMapping not found at: " + rmap_path.string());
    }

    // Validate sample exists
    auto storage_path = SampleStorage::get_storage_path(db_folder, sample_name);
    if (!fs::is_directory(storage_path)) {
        std::vector<std::string> available;
        auto samples_root = fs::path(db_folder) / "samples";
        if (fs::exists(samples_root)) {
            for (const auto& entry : fs::directory_iterator(samples_root)) {
                if (entry.is_directory()) {
                    available.push_back(entry.path().filename().string());
                }
            }
        }
        throw std::runtime_error(
            format_not_found_error("sample", sample_name, available,
                "CALL gnn_offline_sample('projection', 'name', [fanouts])"));
    }

    // =========================================================================
    // Step 4 (DEPRECATED no-op): gnn_build_feature_store packs directly from the
    // sample; this procedure's packed/ output was never read downstream and is
    // deleted by the next stage (four_level_store.cc Step 6). We keep the call
    // invocable and the YIELD contract intact, but skip the redundant materialize.
    // =========================================================================
    (void) config;  // parsed for the legacy materialize path; now unused.

    ctx.yield("sampleName",    ctx.create_string(sample_name));
    ctx.yield("featureName",   ctx.create_string(feature_name));
    ctx.yield("totalBatches",  ctx.create_int(0));
    ctx.yield("reordered",     ctx.create_bool(false));
    ctx.yield("reorderTimeMs", ctx.create_int(0));
    ctx.yield("packTimeMs",    ctx.create_int(0));
    ctx.yield("totalTimeMs",   ctx.create_int(0));
    ctx.yield("packedDir",     ctx.create_string(""));
    ctx.yield_row();
}

} // namespace GQL::Procedures
