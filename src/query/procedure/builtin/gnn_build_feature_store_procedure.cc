#include "query/procedure/builtin/gnn_build_feature_store_procedure.h"

#include <algorithm>
#include <filesystem>
#include <string>

#include "query/procedure/procedure_context.h"

#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/four_level_store.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/gql/gql_model.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

void GnnBuildFeatureStoreProcedure::execute(ProcedureContext& ctx) {
    using namespace mdb::gnn;

    // =========================================================================
    // Step 1: Parse arguments
    // =========================================================================
    if (ctx.arguments.size() < 2 || ctx.arguments.size() > 3) {
        throw std::runtime_error(
            "gnn_build_feature_store requires 2-3 arguments.\n\n"
            "Usage: CALL gnn_build_feature_store(sampleName, featureName [, options])\n"
            "Example: CALL gnn_build_feature_store('my_sample', 'node_features', {gpu_budget_mb: 2048})");
    }

    // Parse sampleName
    std::string sample_name;
    try {
        sample_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid sampleName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING.\n"
            "Example: CALL gnn_build_feature_store('my_sample', 'node_features')");
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
            "Example: CALL gnn_build_feature_store('my_sample', 'node_features')");
    }
    if (feature_name.empty()) {
        throw std::runtime_error("featureName cannot be empty.");
    }
    validate_safe_name(feature_name, "featureName");

    // =========================================================================
    // Step 2: Parse options
    // =========================================================================
    FourLevelStore::Config config;

    if (ctx.arguments.size() == 3) {
        DictOptions opts(ctx.get_argument(2));

        if (auto v = opts.get_int("gpu_budget_mb")) {
            if (*v < 0) throw std::runtime_error("gpu_budget_mb must be non-negative, got: " + std::to_string(*v));
            config.gpu.budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
        }
        if (auto v = opts.get_int("cpu_budget_mb")) {
            if (*v <= 0) throw std::runtime_error("cpu_budget_mb must be positive, got: " + std::to_string(*v));
            config.cpu.budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
        }
        if (auto v = opts.get_bool("reorder")) {
            config.reorder = *v;
        }
        if (auto v = opts.get_bool("force")) {
            config.force = *v;
        }
        // Fix #15: granular force flags. Each defaults to true, so passing
        // `force: true` alone preserves legacy behaviour. Set to false to
        // preserve a specific output across a force rebuild. Examples:
        //   {force: true, force_reorder: false}
        //       Rebuild L1/L2 caches + packed_slim, KEEP reordered.fmat.
        //       Skips the MinHash recompute (typically the L3 wall-clock
        //       leader on papers100M-scale graphs).
        //   {force: true, force_caches: false, force_reorder: false}
        //       Rebuild only packed_slim + meta. Useful to re-bench Fix #1-4.
        if (auto v = opts.get_bool("force_caches"))      config.force_caches = *v;
        if (auto v = opts.get_bool("force_reorder"))     config.force_reorder = *v;
        if (auto v = opts.get_bool("force_packed_slim")) config.force_packed_slim = *v;
        if (auto v = opts.get_bool("force_meta"))        config.force_meta = *v;
        // Path 4 (2026-05-19): pre-resolve per-batch classification offline.
        // Default true — enables the fast runtime path in gnn_train via
        // addr_tables/batch_NNNNNN.addr sidecars. Set false to skip Phase 5.
        if (auto v = opts.get_bool("buildAddrTables")) config.build_addr_tables = *v;
        // cleanupIntermediate: delete the non-slim packed/ from materialize_batches
        // after build succeeds. Default true. Set false only for debugging.
        if (auto v = opts.get_bool("cleanupIntermediate")) {
            config.cleanup_materialize_scratch = *v;
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
        // Spec D telemetry (2026-05-07): expose disk_budget as a procedure
        // parameter. 0 (default) = unlimited; otherwise a soft constraint
        // that triggers a warning when actual footprint exceeds it.
        // Future Spec C2 will use this value to drive heuristic search of
        // segment_size that satisfies the constraint without warning.
        if (auto v = opts.get_int("diskBudgetMb")) {
            if (*v < 0) throw std::runtime_error("diskBudgetMb must be non-negative, got: " + std::to_string(*v));
            config.disk_budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
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
    // Step 4: Open inputs and build feature store
    // =========================================================================
    auto fm = FeatureMatrix::open(fmat_path);
    auto rm = RowMapping::open(rmap_path);
    auto samples = SampleStorage::open(storage_path);

    auto result = FourLevelStore::build(
        fm, rm, samples, config, db_folder, feature_name);

    // =========================================================================
    // Step 5: Yield results
    // =========================================================================
    auto bytes_to_mb = [](uint64_t b) -> int64_t {
        return static_cast<int64_t>(b / (1024ULL * 1024));
    };

    ctx.yield("sampleName",   ctx.create_string(sample_name));
    ctx.yield("featureName",  ctx.create_string(feature_name));
    ctx.yield("l1Nodes",      ctx.create_int(static_cast<int64_t>(result.l1_nodes)));
    ctx.yield("l2Nodes",      ctx.create_int(static_cast<int64_t>(result.l2_nodes)));
    ctx.yield("l3Nodes",      ctx.create_int(static_cast<int64_t>(result.l3_nodes)));
    ctx.yield("l4Nodes",      ctx.create_int(static_cast<int64_t>(result.l4_nodes)));
    ctx.yield("gpuAvailable", ctx.create_bool(result.gpu_available));
    ctx.yield("buildTimeMs",  ctx.create_int(result.build_time_ms));
    // Spec D telemetry yields (post-build, measured from filesystem)
    ctx.yield("slimMb",       ctx.create_int(bytes_to_mb(result.slim_bytes)));
    ctx.yield("reorderedMb",  ctx.create_int(bytes_to_mb(result.reordered_bytes)));
    ctx.yield("gpuCacheMb",   ctx.create_int(bytes_to_mb(result.gpu_cache_bytes)));
    ctx.yield("cpuCacheMb",   ctx.create_int(bytes_to_mb(result.cpu_cache_bytes)));
    ctx.yield("totalDiskMb",  ctx.create_int(bytes_to_mb(result.total_disk_bytes)));
    ctx.yield("overBudget",   ctx.create_bool(result.over_budget));
    // Path 4 (2026-05-19): Phase 5 telemetry.
    ctx.yield("addrTablesMb",
              ctx.create_int(bytes_to_mb(result.addr_tables_bytes)));
    ctx.yield("addrTablesBuiltOk",
              ctx.create_bool(result.addr_tables_built_ok));
    ctx.yield_row();
}

} // namespace GQL::Procedures
