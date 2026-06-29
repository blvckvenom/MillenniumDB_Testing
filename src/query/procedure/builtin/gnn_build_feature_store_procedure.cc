#include "query/procedure/builtin/gnn_build_feature_store_procedure.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "query/procedure/procedure_context.h"

#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/block_store.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/four_level_store.h"
#include "gnn/storage/row_mapping.h"
#include "gpu/gpu_device.h"
#include "graph_models/gql/gql_model.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"
#include "query/procedure/builtin/gnn_feature_store_config.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

namespace {

// The first options (gpu_budget_mb, cpu_budget_mb, force_*) predate the
// camelCase convention used by every later key in this map; both spellings
// are accepted for those. Everything else is camelCase-only.
constexpr const char* KNOWN_OPTION_KEYS[] = {
    "gpu_budget_mb",     "gpuBudgetMb",
    "cpu_budget_mb",     "cpuBudgetMb",
    "reorder",
    "force",
    "force_caches",      "forceCaches",
    "force_reorder",     "forceReorder",
    "force_packed_slim", "forcePackedSlim",
    "force_meta",        "forceMeta",
    "buildAddrTables",
    "bakeBlocks",
    "noCacheBin",
    "packFullFeatures",
    "writeConsolidatedSlim",
    "cleanupIntermediate",
    "strategy",
    "numHashes",
    "segmentSize",
    "diskBudgetMb",
};

// A mistyped option key must fail loudly instead of silently taking the
// default (e.g. 'gpuBudget_mb' would otherwise leave the auto budget on with
// zero feedback).
void assert_known_option_keys(ObjectId arg) {
    auto dict = Common::Conversions::unpack_dictionary(arg);
    auto* dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) return;  // DictOptions already rejects non-dictionaries
    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string key = Conversions::unpack_string(key_oid);
        bool known = false;
        for (const char* k : KNOWN_OPTION_KEYS) {
            if (key == k) { known = true; break; }
        }
        if (!known) {
            std::string msg = "Unknown option '" + key +
                "' for gnn_build_feature_store. Accepted options: [";
            bool first = true;
            for (const char* k : KNOWN_OPTION_KEYS) {
                if (!first) msg += ", ";
                first = false;
                msg += "'";
                msg += k;
                msg += "'";
            }
            msg += "]";
            throw std::runtime_error(msg);
        }
    }
}

} // namespace

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
    // Step 2: Parse options (shared parser; single source of truth with the
    // unified gnn_offline_sample(buildFeatureStore) path)
    // =========================================================================
    FourLevelStore::Config config;
    if (ctx.arguments.size() == 3) {
        assert_known_option_keys(ctx.get_argument(2));
        DictOptions opts(ctx.get_argument(2));
        config = build_feature_store_config(&opts);
    } else {
        config = build_feature_store_config(nullptr);
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

    // Address-tables-only mode (added 2026-05-20) — when the caller asks for ONLY
    // buildAddrTables=true with all force flags off AND the prior feature store
    // is fully on disk (store.meta exists), we can rebuild addr_tables/
    // sidecars without re-opening the source FeatureMatrix. This unblocks
    // re-running the address-table build when node_features.fmat has been
    // cleaned up but the gpu_cache/cpu_cache/reordered files remain. The runtime
    // ctor will load those caches directly; this avoids the FeatureMatrix::open's
    // strict header-size check on the source fmat.
    bool phase5_only_mode =
        config.build_addr_tables && !config.pack_full &&
        !config.force && !config.force_caches && !config.force_reorder &&
        !config.force_packed_slim && !config.force_meta;
    auto store_meta_path = fs::path(db_folder) / "gnn_features" /
                            (feature_name + "_store.meta");
    bool store_already_built = fs::exists(store_meta_path);

    if (!phase5_only_mode || !store_already_built) {
        if (!fs::exists(fmat_path)) {
            throw std::runtime_error("FeatureMatrix not found at: " + fmat_path.string());
        }
        if (!fs::exists(rmap_path)) {
            throw std::runtime_error("RowMapping not found at: " + rmap_path.string());
        }
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
    auto samples = SampleStorage::open(storage_path);

    FourLevelStore::BuildResult result;

    if (phase5_only_mode && store_already_built) {
        // Refuse to reuse addr_tables built from a DIFFERENT sample.
        // The fast path skips the source FeatureMatrix, so a content mismatch
        // here cannot be repaired in-place — surface a clear error instead of
        // silently serving stale L1/L2/L3 membership.
        if (!FourLevelStore::store_matches_sample_fp(
                db_folder, feature_name, samples.get_catalog().sample_content_fp)) {
            throw std::runtime_error(
                "gnn_build_feature_store: the existing feature store for '" +
                feature_name + "' was built from a different sample "
                "(content fingerprint mismatch). Re-run with force:1 to rebuild "
                "it (requires " + feature_name + ".fmat).");
        }
        // Address-tables-only fast path: rebuild only addr_tables/ via the
        // runtime ctor. No source FeatureMatrix needed. Tier counts / disk sizes
        // in the BuildResult stay at their default zero values — only the
        // addrTablesMb / addrTablesBuiltOk yields are meaningful here.
        FourLevelStore store(db_folder, feature_name, samples);
        uint64_t blocks_bytes = 0;
        uint64_t addr_bytes = store.rebuild_addr_tables(
            db_folder, config.bake_blocks, &blocks_bytes);
        result.addr_tables_bytes    = addr_bytes;
        result.addr_tables_built_ok = true;
        result.blocks_bytes         = blocks_bytes;
        if (config.bake_blocks) {
            // Report the bake outcome from the on-disk artifacts, not from the
            // request flag: a bake that skipped over stale-format blocks must
            // not claim success. Mirrors the train-time eligibility gate —
            // every batch needs a block stamped with the catalog's store
            // fingerprint (header-only probes, no body reads).
            const auto& catalog = samples.get_catalog();
            auto blocks_dir = storage_path / "blocks";
            bool blocks_ok = catalog.total_batches > 0;
            for (uint64_t b = 0; blocks_ok && b < catalog.total_batches; ++b) {
                auto blk = block_filename(blocks_dir, b);
                blocks_ok = catalog.sample_content_fp != 0
                    ? BlockReader::read_store_fp(blk) == catalog.sample_content_fp
                    : fs::exists(blk);
            }
            result.blocks_built_ok = blocks_ok;
        }
    } else {
        auto fm = FeatureMatrix::open(fmat_path);
        auto rm = RowMapping::open(rmap_path);
        result = FourLevelStore::build(
            fm, rm, samples, config, db_folder, feature_name);
    }

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
    // Four-Level Feature Store telemetry yields (post-build, measured from filesystem)
    ctx.yield("slimMb",       ctx.create_int(bytes_to_mb(result.slim_bytes)));
    ctx.yield("reorderedMb",  ctx.create_int(bytes_to_mb(result.reordered_bytes)));
    ctx.yield("gpuCacheMb",   ctx.create_int(bytes_to_mb(result.gpu_cache_bytes)));
    ctx.yield("cpuCacheMb",   ctx.create_int(bytes_to_mb(result.cpu_cache_bytes)));
    ctx.yield("totalDiskMb",  ctx.create_int(bytes_to_mb(result.total_disk_bytes)));
    ctx.yield("overBudget",   ctx.create_bool(result.over_budget));
    // Offline address-table build telemetry (added 2026-05-19).
    ctx.yield("addrTablesMb",
              ctx.create_int(bytes_to_mb(result.addr_tables_bytes)));
    ctx.yield("addrTablesBuiltOk",
              ctx.create_bool(result.addr_tables_built_ok));
    // Baked computation-graph block telemetry.
    ctx.yield("blocksMb",
              ctx.create_int(bytes_to_mb(result.blocks_bytes)));
    ctx.yield("blocksBuiltOk",
              ctx.create_bool(result.blocks_built_ok));
    // Packed-full feature pack telemetry.
    ctx.yield("packedFullMb",
              ctx.create_int(bytes_to_mb(result.packed_full_bytes)));
    ctx.yield_row();
}

} // namespace GQL::Procedures
