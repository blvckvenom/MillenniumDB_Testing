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
    // Step 2: Parse options
    // =========================================================================
    FourLevelStore::Config config;

    // Dynamic per-hardware budgets: track whether the caller passed explicit
    // budgets. When MDB_GNN_AUTO_RESOURCES=1 (default OFF) and a budget is NOT
    // explicit, it is derived from the detected machine below. Explicit always wins.
    bool gpu_budget_explicit = false;
    bool cpu_budget_explicit = false;

    if (ctx.arguments.size() == 3) {
        DictOptions opts(ctx.get_argument(2));
        assert_known_option_keys(ctx.get_argument(2));

        auto get_int_opt = [&](const char* key, const char* alias) {
            auto v = opts.get_int(key);
            return v ? v : opts.get_int(alias);
        };
        auto get_bool_opt = [&](const char* key, const char* alias) {
            auto v = opts.get_bool(key);
            return v ? v : opts.get_bool(alias);
        };

        if (auto v = get_int_opt("gpu_budget_mb", "gpuBudgetMb")) {
            if (*v < 0) throw std::runtime_error("gpu_budget_mb must be non-negative, got: " + std::to_string(*v));
            config.gpu.budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
            gpu_budget_explicit = true;
        }
        if (auto v = get_int_opt("cpu_budget_mb", "cpuBudgetMb")) {
            if (*v <= 0) throw std::runtime_error("cpu_budget_mb must be positive, got: " + std::to_string(*v));
            config.cpu.budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
            cpu_budget_explicit = true;
        }
        if (auto v = opts.get_bool("reorder")) {
            config.reorder = *v;
        }
        if (auto v = opts.get_bool("force")) {
            config.force = *v;
        }
        // Granular force flags, each defaulting to true when the top-level
        // `force: true` is set, preserving legacy all-or-nothing behaviour.
        // Setting individual flags to false lets a caller preserve a specific
        // build artifact while forcing the rest. Examples:
        //   {force: true, force_reorder: false}
        //       Rebuild L1/L2 caches + packed_slim, KEEP reordered.fmat.
        //       Skips the MinHash recompute (typically the L3 wall-clock
        //       leader on papers100M-scale graphs).
        //   {force: true, force_caches: false, force_reorder: false}
        //       Rebuild only packed_slim + meta. Useful to isolate L4 timing.
        if (auto v = get_bool_opt("force_caches", "forceCaches"))           config.force_caches = *v;
        if (auto v = get_bool_opt("force_reorder", "forceReorder"))         config.force_reorder = *v;
        if (auto v = get_bool_opt("force_packed_slim", "forcePackedSlim"))  config.force_packed_slim = *v;
        if (auto v = get_bool_opt("force_meta", "forceMeta"))               config.force_meta = *v;
        // Pre-resolve per-batch node classification offline (added 2026-05-19).
        // Default true — enables the fast runtime path in gnn_train via
        // addr_tables/batch_NNNNNN.addr sidecars. Set false to skip building
        // the offline address tables.
        if (auto v = opts.get_bool("buildAddrTables")) config.build_addr_tables = *v;
        // Bake per-batch computation-graph blocks (blocks/block_NNNNNN.blk)
        // keyed by sample content hash, idempotent across re-runs. Default OFF —
        // when off the build is byte-identical to before. NOTE: bakeBlocks on a
        // reused store requires buildAddrTables (the default) to be on; with
        // buildAddrTables:false + reuse, pass force to bake.
        if (auto v = opts.get_bool("bakeBlocks")) config.bake_blocks = *v;
        // packFullFeatures is DEPRECATED AND REMOVED. The packed-full pack stores
        // every batch's full receptive field with no cross-batch dedup (~18x the
        // feature matrix, ~1 TB on papers100M) and is infeasible. The default
        // four-level feature store (which dedups across batches) is the supported
        // path. The option is hard-refused so it can never be requested.
        if (opts.get_bool("packFullFeatures")) {
            throw std::runtime_error(
                "packFullFeatures is deprecated and removed: the packed-full feature "
                "store is infeasible (~18x the feature matrix) and is no longer "
                "supported; use the default four-level feature store.");
        }
        // Also emit a single consolidated packed_slim/consolidated.slim file
        // during the partitioned L4 pack (+ v2 addr_tables). Opt-in, default OFF.
        // The runtime reads it only when MDB_GNN_CONSOLIDATED_SLIM is set.
        if (auto v = opts.get_bool("writeConsolidatedSlim")) config.write_consolidated_slim = *v;
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
        // Expose disk_budget as a procedure parameter. 0 (default) = unlimited;
        // otherwise a soft constraint that triggers a warning when the actual
        // on-disk footprint of the Four-Level Feature Store (L1 GPU cache +
        // L2 CPU cache + L3 reordered fmat + L4 packed-slim) exceeds it. A
        // future heuristic search over MinHash segment_size could use this
        // value to find the smallest segment_size that keeps total disk usage
        // within budget without warning.
        if (auto v = opts.get_int("diskBudgetMb")) {
            if (*v < 0) throw std::runtime_error("diskBudgetMb must be non-negative, got: " + std::to_string(*v));
            config.disk_budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
        }
    }

    // =========================================================================
    // Step 2b: dynamic per-hardware — auto-detect cache budgets.
    // =========================================================================
    // Opt-in via env MDB_GNN_AUTO_RESOURCES=1 (default OFF). When a budget was
    // NOT passed explicitly, derive it from the actual machine via the same
    // probe the GPU sort planner uses (mdb::gpu::detect_resources()). This is
    // the missing wire identified in HARDWARE_ADAPTIVITY_AUDIT.md: the GNN
    // feature store otherwise uses fixed 2 GB GPU / 4 GB CPU defaults regardless
    // of the host's VRAM/RAM. Budgets choose cache TIER placement only (which
    // nodes live in L1 GPU / L2 CPU vs L3/L4 disk) — never the gathered feature
    // values — so the trained result is INVARIANT to the budget (validate via
    // cora 0.8574939 + a papers100M feature checksum). Explicit gpu_budget_mb /
    // cpu_budget_mb always win. First-cut heuristic; tune the reserves via A/B.
    if (const char* e = std::getenv("MDB_GNN_AUTO_RESOURCES"); e && std::string(e) == "1") {
        auto res = mdb::gpu::detect_resources();
        if (!gpu_budget_explicit && res.has_gpu) {
            // CONSERVATIVE by design: the L1 GPU cache lives in VRAM DURING TRAINING,
            // competing with the model + activations + the assembler's per-batch
            // device tensors. Giving L1 most of the VRAM would OOM the train step.
            // free_vram is already 20%-margined (gpu_device VRAM_SAFETY_FACTOR); take
            // a quarter of it for L1 (~0.2x total VRAM), matching the empirically-safe
            // hand-tuned ratio (e.g. 2 GB L1 on a ~12 GB free card) while adapting to
            // bigger/smaller GPUs. A smaller L1 only shifts reads to L3/L4 disk (slower,
            // never OOM). Tune the fraction via A/B (see HARDWARE_ADAPTIVITY_AUDIT.md).
            config.gpu.budget_bytes = res.gpu.free_vram / 4;
        }
        if (!cpu_budget_explicit && res.ram_available > 0) {
            // A quarter of available RAM for the L2 CPU cache, leaving room for the
            // build working set (mmap'd fmat + reorder) + page cache.
            config.cpu.budget_bytes = res.ram_available / 4;
        }
        std::cerr << "[gnn_build_feature_store] MDB_GNN_AUTO_RESOURCES: "
                  << "gpu_budget=" << (config.gpu.budget_bytes >> 20) << "MB "
                  << "cpu_budget=" << (config.cpu.budget_bytes >> 20) << "MB "
                  << "(has_gpu=" << (res.has_gpu ? 1 : 0)
                  << " free_vram=" << (res.gpu.free_vram >> 20) << "MB"
                  << " ram_avail=" << (res.ram_available >> 20) << "MB"
                  << " gpu_explicit=" << (gpu_budget_explicit ? 1 : 0)
                  << " cpu_explicit=" << (cpu_budget_explicit ? 1 : 0) << ")\n";
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
