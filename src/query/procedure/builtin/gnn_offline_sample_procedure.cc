#include "gnn_offline_sample_procedure.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "gnn/sampling/offline_sampling_engine.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_config.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/storage/four_level_store.h"
#include "query/procedure/builtin/gnn_feature_store_config.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_object_id.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "gnn_procedure_utils.h"

using namespace GQL;
using namespace GQL::Procedures;
using namespace mdb::gnn;

void GnnOfflineSampleProcedure::execute(ProcedureContext& ctx) {
    // Step 1: Validate argument count
    if (ctx.arguments.size() < 3 || ctx.arguments.size() > 4) {
        throw std::runtime_error(
            "gnn_offline_sample() requires 3-4 arguments, got " +
            std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL gnn_offline_sample(projectionName, sampleName, fanouts [, options])\n"
            "  YIELD sampleName, totalBatches, trainBatches, ...\n\n"
            "Parameters:\n"
            "  - projectionName (STRING): Source graph projection\n"
            "  - sampleName (STRING): Name for the sample set\n"
            "  - fanouts (LIST<INT>): Fanouts per layer in DGL order (last "
            "element = hop adjacent to seeds), e.g., [10, 15]\n"
            "  - options (MAP, optional): batchSize, trainRatio, randomSeed, etc.\n\n"
            "Examples:\n"
            "  CALL gnn_offline_sample('social', 'samples_v1', [15, 10])\n"
            "  CALL gnn_offline_sample('social', 'samples_v1', [15, 10], {batchSize: 512})"
        );
    }

    // Step 2: Parse projectionName
    std::string projection_name;
    try {
        projection_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid projectionName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING containing the projection name.\n"
            "Example: CALL gnn_offline_sample('myProjection', ...)"
        );
    }

    if (projection_name.empty()) {
        throw std::runtime_error(
            "Invalid projection name: name cannot be empty.\n"
            "Provide a non-empty string as the first argument.\n"
            "Example: CALL gnn_offline_sample('myProjection', ...)"
        );
    }

    // Step 3: Parse sampleName
    std::string sample_name;
    try {
        sample_name = ctx.get_string_argument(1);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid sampleName parameter: " + std::string(e.what()) + "\n\n"
            "The second parameter must be a STRING containing the sample set name.\n"
            "Example: CALL gnn_offline_sample('projection', 'mySamples', ...)"
        );
    }

    if (sample_name.empty()) {
        throw std::runtime_error(
            "Invalid sample name: name cannot be empty.\n"
            "Provide a non-empty string as the second argument.\n"
            "Example: CALL gnn_offline_sample('projection', 'mySamples', ...)"
        );
    }

    validate_safe_name(sample_name, "sampleName");

    // Step 4: Parse fanouts list
    std::vector<uint64_t> fanouts;
    try {
        fanouts = parse_fanouts(ctx, 2);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid fanouts parameter: " + std::string(e.what()) + "\n\n"
            "The third parameter must be a LIST of positive integers, read in\n"
            "DGL/GraphBolt order: the LAST element samples the hop adjacent to\n"
            "the seeds (pass fanoutsNearestFirst:true for the legacy hop order).\n\n"
            "Examples:\n"
            "  [10, 15]       - 2-hop: 15 direct neighbors, then 10 at 2-hop\n"
            "  [5, 10, 15]    - 3-hop: 15 direct, 10 at 2-hop, 5 at 3-hop\n"
            "  [25]           - 1-hop: 25 neighbors"
        );
    }

    // Step 5: Parse optional options map
    uint64_t batch_size = 1024;
    double train_ratio = 0.7;
    double val_ratio = 0.15;
    double test_ratio = 0.15;
    uint64_t random_seed = SamplingConfig::DEFAULT_RANDOM_SEED;
    std::string orientation_str = "UNDIRECTED";
    bool use_predefined_splits = false;
    bool use_predefined_splits_explicit = false;
    bool use_adjacency_cache = true;
    bool use_four_level_topology_store = true;
    uint64_t l1_cache_mb = 0;
    uint64_t l2_cache_mb = 0;
    bool use_l3_mmap_sidecar = true;
    bool auto_profile_on_cold_start = true;
    uint64_t profile_num_walks = 0;
    uint64_t profile_walk_length = 0;
    uint64_t num_workers = std::numeric_limits<uint64_t>::max();  // sentinel: max = unset (auto-detect below)
    bool force = false;
    std::string sampling_backend_str = "auto";  // dynamic GPU/CPU backend choice
    std::string symmetric_topology_str = "auto";  // auto|on|off single merged slice
    bool fanouts_nearest_first = false;  // legacy hop-order reading of the fanouts list

    if (ctx.arguments.size() >= 4) {
        try {
            parse_options(ctx, 3, batch_size, train_ratio, val_ratio,
                          test_ratio, random_seed, orientation_str,
                          use_predefined_splits,
                          use_predefined_splits_explicit,
                          use_adjacency_cache,
                          use_four_level_topology_store,
                          l1_cache_mb, l2_cache_mb,
                          use_l3_mmap_sidecar,
                          auto_profile_on_cold_start,
                          profile_num_walks, profile_walk_length,
                          num_workers, force, sampling_backend_str,
                          symmetric_topology_str,
                          fanouts_nearest_first);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Invalid options parameter: " + std::string(e.what()) + "\n\n"
                "The fourth parameter must be a MAP with optional keys:\n"
                "  - batchSize (INT): Seeds per batch (default: 1024)\n"
                "  - trainRatio (FLOAT): Training fraction (default: 0.7)\n"
                "  - validationRatio (FLOAT): Validation fraction (default: 0.15)\n"
                "  - testRatio (FLOAT): Test fraction (default: 0.15)\n"
                "  - randomSeed (INT): For reproducibility (default: 42)\n"
                "  - orientation (STRING): NATURAL, REVERSE, or UNDIRECTED\n"
                "  - usePredefinedSplits (BOOL): Use splits.bin (default: false)\n"
                "  - useAdjacencyCache (BOOL): in-memory adjacency cache (default: true)\n"
                "  - useFourLevelTopologyStore (BOOL): frequency-tiered topology store (default: true)\n"
                "  - l1CacheMb (INT): L1 budget in MiB (0 = auto-detect)\n"
                "  - l2CacheMb (INT): L2 budget in MiB (0 = auto-detect)\n"
                "  - useL3MmapSidecar (BOOL): mmap'd topology CSR sidecar as L3 (default: true)\n"
                "  - fanoutsNearestFirst (BOOL): read fanouts in hop order, first element = hop adjacent to seeds (default: false = DGL order)\n"
                "  - useSymmetricTopology (STRING): 'auto'|'on'|'off' single pre-merged undirected slice (default: 'auto')\n"
                "  - force (BOOL): Drop + re-create an existing sample set (default: false)\n\n"
                "Example:\n"
                "  CALL gnn_offline_sample('proj', 'samples', [15, 10], {\n"
                "      batchSize: 512,\n"
                "      trainRatio: 0.8,\n"
                "      validationRatio: 0.1,\n"
                "      testRatio: 0.1,\n"
                "      randomSeed: 12345\n"
                "  })"
            );
        }
    }

    // Fanout convention: the list is read in DGL/GraphBolt order by default —
    // the LAST element samples the hop adjacent to the seeds — so a config is
    // comparable string-for-string with DGL-based systems. Internally (and in
    // the sample catalog) fanouts are stored in hop order: fanouts[0] = hop
    // adjacent to the seeds. fanoutsNearestFirst:true skips the reversal and
    // reads the list directly in hop order (the pre-2026-07 behavior).
    if (!fanouts_nearest_first) {
        std::reverse(fanouts.begin(), fanouts.end());
    }

    // Step 6: Verify projection exists
    auto& manager = ProjectionManager::get_instance();
    if (!manager.projection_exists(projection_name)) {
        // Provide helpful error with available projections
        auto projections = manager.list_projections();
        std::vector<std::string> proj_names;
        proj_names.reserve(projections.size());
        for (const auto& p : projections) {
            proj_names.push_back(p.name);
        }
        throw std::runtime_error(
            format_not_found_error("projection", projection_name, proj_names,
                                   "CALL graph_project('name', 'NodeLabel', 'EdgeType')")
        );
    }

    // Step 7: Check if sample already exists
    std::string db_folder = get_db_folder();

    // `force` option — when the sample already exists, drop it
    // and re-sample (matches gnn_materialize_batches / gnn_build_feature_store
    // force semantics) instead of hard-failing. The sample dir is self-contained,
    // so remove_all is an atomic-enough drop. Default false preserves the
    // fail-loud behavior. The drop itself is deferred until just before the
    // engine runs (Step 10b) so that orientation parsing / config validation
    // failures leave the existing sample untouched.
    const bool sample_exists = SampleStorage::exists(db_folder, sample_name);
    if (sample_exists && !force) {
        throw std::runtime_error(
            "Sample set '" + sample_name + "' already exists.\n\n"
            "Solutions:\n"
            "  1. Pass force:true to overwrite it in place\n"
            "  2. Use a different name for the new sample set\n"
            "  3. Delete the existing one first:\n"
            "     CALL gnn_sample_drop('" + sample_name + "')\n"
            "  4. List existing samples:\n"
            "     CALL gnn_sample_list() YIELD sampleName"
        );
    }

    // Step 8: Parse orientation
    EdgeOrientation orientation = EdgeOrientation::UNDIRECTED;
    std::string upper_orientation = orientation_str;
    std::transform(upper_orientation.begin(), upper_orientation.end(),
                   upper_orientation.begin(), ::toupper);

    if (upper_orientation == "NATURAL") {
        orientation = EdgeOrientation::NATURAL;
    } else if (upper_orientation == "REVERSE") {
        orientation = EdgeOrientation::REVERSE;
    } else if (upper_orientation == "UNDIRECTED") {
        orientation = EdgeOrientation::UNDIRECTED;
    } else {
        throw std::runtime_error(
            "Invalid orientation: '" + orientation_str + "'.\n"
            "Must be 'NATURAL', 'REVERSE', or 'UNDIRECTED' (case-insensitive)."
        );
    }

    // =========================================================================
    // Environment-adaptive defaults. Both respect explicit user values; they
    // only fill in a better default when the caller was silent.
    // =========================================================================
    // Auto-parallelize the sampler when numWorkers was not specified. The
    // per-batch re-seeding (each batch uses random_seed XOR batch_id) makes
    // the sampled output deterministic regardless of how many workers pick
    // up batches, so using multiple workers is a free speedup over sequential.
    // Explicit numWorkers (including 0 = legacy sequential) always wins.
    if (num_workers == std::numeric_limits<uint64_t>::max()) {
        unsigned hc = std::thread::hardware_concurrency();
        num_workers = (hc == 0) ? 4u : std::min<unsigned>(hc, 8u);
        std::cerr << "[gnn_offline_sample] notice: numWorkers not set — defaulting "
                     "to " << num_workers << " (min(cores,8)). Pass numWorkers "
                     "explicitly to override (0 = legacy sequential).\n";
    }

    // Auto-use predefined splits when the projection has splits.bin and the
    // caller did not set usePredefinedSplits. Otherwise a random ratio split
    // silently ignores splits.bin -> validation collapses on labeled-subset
    // datasets (e.g. OGB). Explicit usePredefinedSplits always wins.
    if (!use_predefined_splits_explicit && !use_predefined_splits) {
        auto splits_path = std::filesystem::path(db_folder) /
                           "projections" / projection_name / "splits.bin";
        if (std::filesystem::exists(splits_path)) {
            use_predefined_splits = true;
            std::cerr << "[gnn_offline_sample] notice: projection has splits.bin "
                         "and usePredefinedSplits was not set — defaulting to "
                         "true (using the predefined train/val/test split). Pass "
                         "usePredefinedSplits:false to force a ratio split.\n";
        }
    }

    // Step 9: Build SamplingConfig
    SamplingConfig config;
    config.projection_name = projection_name;
    config.sample_name = sample_name;
    config.fanouts = fanouts;
    config.batch_size = batch_size;
    config.train_ratio = train_ratio;
    config.val_ratio = val_ratio;
    config.test_ratio = test_ratio;
    config.random_seed = random_seed;
    config.orientation = orientation;
    {
        std::string sb_lower = sampling_backend_str;
        std::transform(sb_lower.begin(), sb_lower.end(), sb_lower.begin(), ::tolower);
        if (sb_lower == "cpu") {
            config.sampling_backend = SamplingBackendChoice::FORCE_CPU;
        } else if (sb_lower == "gpu") {
            config.sampling_backend = SamplingBackendChoice::FORCE_GPU;
        } else {
            config.sampling_backend = SamplingBackendChoice::AUTO;  // "auto" or unknown
        }
    }
    {
        std::string st = symmetric_topology_str;
        std::transform(st.begin(), st.end(), st.begin(), ::tolower);
        if (st == "on") {
            config.use_symmetric_topology = SamplingConfig::SymmetricTopologyMode::ON;
        } else if (st == "off") {
            config.use_symmetric_topology = SamplingConfig::SymmetricTopologyMode::OFF;
        } else {
            config.use_symmetric_topology = SamplingConfig::SymmetricTopologyMode::AUTO;
        }
    }
    config.use_predefined_splits = use_predefined_splits;
    config.use_adjacency_cache = use_adjacency_cache;
    config.use_four_level_topology_store = use_four_level_topology_store;
    config.l1_cache_mb = static_cast<std::size_t>(l1_cache_mb);
    config.l2_cache_mb = static_cast<std::size_t>(l2_cache_mb);
    config.use_l3_mmap_sidecar = use_l3_mmap_sidecar;
    config.auto_profile_on_cold_start = auto_profile_on_cold_start;
    config.profile_num_walks   = static_cast<std::size_t>(profile_num_walks);
    config.profile_walk_length = static_cast<std::size_t>(profile_walk_length);
    config.num_workers = static_cast<std::uint32_t>(num_workers);

    // Validate config
    try {
        config.validate();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid sampling configuration: " + std::string(e.what())
        );
    }

    // Step 10: Open projection storage
    std::string proj_dir = manager.get_projection_dir(projection_name);
    ProjectionStorage storage(proj_dir, db_folder);
    storage.open();

    // Step 10b: force-drop the existing sample. Done only now — after every
    // option is parsed, the config validated, and the projection opened — so
    // none of those failure modes can destroy a sample that may have taken
    // hours to build.
    if (sample_exists && force) {
        std::error_code ec;
        std::filesystem::remove_all(
            SampleStorage::get_storage_path(db_folder, sample_name), ec);
        if (ec) {
            throw std::runtime_error(
                "gnn_offline_sample: force=true could not remove existing "
                "sample '" + sample_name + "': " + ec.message());
        }
    }

    // Step 11: Create and run engine
    // Pass db_folder so the engine can locate the global RowMapping when
    // config.use_predefined_splits=true (needed to index labels.bin/splits.bin).
    OfflineSamplingEngine engine(storage, config, db_folder);
    SamplingResult result = engine.run(db_folder);

    // Step 12: Handle result
    if (!result.success) {
        if (result.cancelled) {
            throw std::runtime_error("Sampling was cancelled.");
        }
        throw std::runtime_error(
            "Sampling failed: " + result.error_message
        );
    }

    // -----------------------------------------------------------------------
    // Folded feature-store build (opt-in, default OFF). With
    // buildFeatureStore:true the four-level feature store is built right after
    // the sample is written, so the data-dependent reorder / address-tables /
    // blocks are produced in one call instead of requiring a separate
    // gnn_build_feature_store invocation. Default off => byte-identical to
    // prior behavior; the standalone gnn_build_feature_store verb remains.
    // (This reuses the existing FourLevelStore::build, which re-reads the
    // just-written sample from disk; a build fed from the in-RAM sample,
    // eliminating that re-read, would be a further increment.)
    // -----------------------------------------------------------------------
    bool build_feature_store_done = false;
    std::string bfs_feature_name;
    if (ctx.arguments.size() == 4) {
        DictOptions bfs_opts(ctx.get_argument(3));
        const bool build_feature_store =
            bfs_opts.get_bool("buildFeatureStore").value_or(false);
        if (build_feature_store) {
            bfs_feature_name =
                bfs_opts.get_string("featureName").value_or(std::string("node_features"));
            auto sp = SampleStorage::get_storage_path(db_folder, sample_name);
            auto fmat_path = std::filesystem::path(db_folder) / "gnn_features" /
                             (bfs_feature_name + ".fmat");
            auto rmap_path = std::filesystem::path(db_folder) / "gnn_features" /
                             (bfs_feature_name + ".rmap");
            if (!std::filesystem::exists(fmat_path) ||
                !std::filesystem::exists(rmap_path)) {
                throw std::runtime_error(
                    "buildFeatureStore: '" + bfs_feature_name +
                    ".fmat'/'.rmap' not found under gnn_features/ — import the "
                    "tensors first (mdb import ... --with-tensors features.npy).");
            }
            auto bfs_samples = SampleStorage::open(sp);
            auto bfs_fm = FeatureMatrix::open(fmat_path);
            auto bfs_rm = RowMapping::open(rmap_path);
            // Honor the same build options the standalone gnn_build_feature_store
            // accepts (budgets, force flags, reorder strategy, bakeBlocks, numHashes,
            // ...), parsed from the SAME options map (sampling keys are ignored).
            FourLevelStore::Config bfs_config = build_feature_store_config(&bfs_opts);
            // autoCache: size L1 to a prior gnn_train's measured recommendation
            // (no-op without autoCache:true / when explicit gpu_budget_mb wins).
            apply_auto_cache_budget(bfs_config, &bfs_opts, db_folder, bfs_feature_name);
            FourLevelStore::build(bfs_fm, bfs_rm, bfs_samples, bfs_config,
                                  db_folder, bfs_feature_name);
            build_feature_store_done = true;
        }
    }

    // Step 13: Yield results
    auto storage_path = SampleStorage::get_storage_path(db_folder, sample_name);
    int64_t compute_millis = static_cast<int64_t>(result.total_seconds * 1000);

    ctx.yield("sampleName", ctx.create_string(sample_name));
    ctx.yield("projectionName", ctx.create_string(projection_name));
    ctx.yield("totalBatches", ctx.create_int(static_cast<int64_t>(result.catalog.total_batches)));
    ctx.yield("trainBatches", ctx.create_int(static_cast<int64_t>(result.catalog.train_batches)));
    ctx.yield("validationBatches", ctx.create_int(static_cast<int64_t>(result.catalog.validation_batches)));
    ctx.yield("testBatches", ctx.create_int(static_cast<int64_t>(result.catalog.test_batches)));
    ctx.yield("uniqueNodes", ctx.create_int(static_cast<int64_t>(result.catalog.unique_nodes)));
    ctx.yield("storagePath", ctx.create_string(storage_path.string()));
    ctx.yield("computeMillis", ctx.create_int(compute_millis));

    // Cold-start topology profiler telemetry: if node_counts.bin was absent,
    // a degree-weighted random-walk pass (Vose alias seeds) ran before sampling
    // to produce access-frequency estimates for Four-Level Topology Store tier
    // assignment. These yields report whether that profiler was triggered,
    // whether it succeeded, and the number of walks and neighbor lookups done.
    ctx.yield("phase0Triggered", ctx.create_bool(result.phase0_triggered));
    ctx.yield("phase0Succeeded", ctx.create_bool(result.phase0_succeeded));
    ctx.yield("phase0WalksDone",   ctx.create_int(static_cast<int64_t>(result.phase0_walks_done)));
    ctx.yield("phase0LookupsDone", ctx.create_int(static_cast<int64_t>(result.phase0_lookups_done)));
    ctx.yield("phase0Millis",      ctx.create_int(
        static_cast<int64_t>(result.phase0_elapsed_seconds * 1000.0)));

    // Resolved parallel sampler worker count. Reports 1 for the legacy
    // single-thread path (numWorkers=0); matches the actual pool size otherwise.
    ctx.yield("numWorkersUsed", ctx.create_int(
        static_cast<int64_t>(result.num_workers_used)));

    // Sample content fingerprint (the staleness/reproducibility check),
    // surfaced as a 16-char hex STRING (avoids the
    // signed-int64 high-bit wrap a numeric yield would suffer). Order/worker-
    // invariant (sort-then-XOR-fold) — the O(1) semantic-equality gate for
    // parallelism work (numWorkers, single-vs-parallel populate).
    {
        char fp_hex[17];
        std::snprintf(fp_hex, sizeof(fp_hex), "%016llx",
                      static_cast<unsigned long long>(result.catalog.sample_content_fp));
        ctx.yield("sampleContentFp", ctx.create_string(std::string(fp_hex)));
    }

    // Sampling-backend telemetry: the backend the hardware-based planner
    // chose, the directions the GPU path serves, and the planner's reason.
    // The plan was applied by the engine — a GPU backend whose pinned
    // topology view registered ran the GPU kernel; otherwise the CPU
    // out-of-core path ran and the plan is still reported here.
    ctx.yield("samplingBackend",    ctx.create_string(result.sampling_backend));
    ctx.yield("samplingDirections", ctx.create_string(result.sampling_directions));
    ctx.yield("samplingPlanReason", ctx.create_string(result.sampling_plan_reason));

    // Symmetric single-slice topology telemetry. symmetricUsed reflects the
    // resolved decision (AUTO => UNDIRECTED); symmetricBuiltOk + symmetricMs +
    // symmetricRamBytes report the in-RAM merge of the two directional CSRs into
    // one pre-merged undirected slice consumed by the GPU-UVA pin / sym tier.
    ctx.yield("symmetricUsed",     ctx.create_bool(result.symmetric_used));
    ctx.yield("symmetricBuiltOk",  ctx.create_bool(result.symmetric_built_ok));
    ctx.yield("symmetricMs",       ctx.create_int(
        static_cast<int64_t>(result.symmetric_ms)));
    ctx.yield("symmetricRamBytes", ctx.create_int(
        static_cast<int64_t>(result.symmetric_ram_bytes)));

    // RAM relief telemetry. directionalRamFreedBytes is what releasing the
    // directional fwd/rev tiers reclaimed once the merged slice was pinned (the
    // symmetric GPU path); rssPeakBeforeFreeBytes is the process high-water mark
    // at the tightest moment (merged slice + directional tiers both resident),
    // rssPeakAfterFreeBytes the post-free sampling-phase high-water mark, and
    // rssCurrentEndBytes the resident size at end. 0 when unavailable / no free.
    ctx.yield("directionalRamFreedBytes", ctx.create_int(
        static_cast<int64_t>(result.directional_ram_freed_bytes)));
    ctx.yield("rssPeakBeforeFreeBytes", ctx.create_int(
        static_cast<int64_t>(result.rss_peak_before_free_bytes)));
    ctx.yield("rssPeakAfterFreeBytes", ctx.create_int(
        static_cast<int64_t>(result.rss_peak_after_free_bytes)));
    ctx.yield("rssCurrentEndBytes", ctx.create_int(
        static_cast<int64_t>(result.rss_current_end_bytes)));

    // Folded feature-store build telemetry (only present when buildFeatureStore:true).
    if (build_feature_store_done) {
        ctx.yield("buildFeatureStoreOk", ctx.create_bool(true));
        ctx.yield("featureName", ctx.create_string(bfs_feature_name));
    }

    ctx.yield_row();
}

std::vector<uint64_t> GnnOfflineSampleProcedure::parse_fanouts(
    ProcedureContext& ctx,
    size_t arg_index
) {
    ObjectId arg = ctx.get_argument(arg_index);
    auto type = GQL_OID::get_type(arg);

    if (type != GQL_OID::Type::LIST) {
        throw std::runtime_error(
            "fanouts must be a LIST, got type: " +
            std::to_string(static_cast<int>(type))
        );
    }

    std::vector<ObjectId> list_items = Conversions::unpack_list(arg);

    if (list_items.empty()) {
        throw std::runtime_error(
            "fanouts list cannot be empty. "
            "Provide at least one fanout value (e.g., [15] for 1-hop sampling)."
        );
    }

    std::vector<uint64_t> fanouts;
    fanouts.reserve(list_items.size());

    for (size_t i = 0; i < list_items.size(); i++) {
        const auto& item_oid = list_items[i];
        auto item_type = GQL_OID::get_type(item_oid);

        // Accept integers
        if (item_type == GQL_OID::Type::INT56_INLINE ||
            item_type == GQL_OID::Type::INT64_EXTERN ||
            item_type == GQL_OID::Type::INT64_TMP)
        {
            int64_t value = Conversions::unpack_int(item_oid);
            if (value <= 0) {
                throw std::runtime_error(
                    "fanouts[" + std::to_string(i) + "] must be positive, got: " +
                    std::to_string(value)
                );
            }
            fanouts.push_back(static_cast<uint64_t>(value));
        } else {
            throw std::runtime_error(
                "fanouts[" + std::to_string(i) + "] must be an integer, got type: " +
                std::to_string(static_cast<int>(item_type))
            );
        }
    }

    return fanouts;
}

void GnnOfflineSampleProcedure::parse_options(
    ProcedureContext& ctx,
    size_t arg_index,
    uint64_t& batch_size,
    double& train_ratio,
    double& val_ratio,
    double& test_ratio,
    uint64_t& random_seed,
    std::string& orientation,
    bool& use_predefined_splits,
    bool& use_predefined_splits_explicit,
    bool& use_adjacency_cache,
    bool& use_four_level_topology_store,
    uint64_t& l1_cache_mb,
    uint64_t& l2_cache_mb,
    bool& use_l3_mmap_sidecar,
    bool& auto_profile_on_cold_start,
    uint64_t& profile_num_walks,
    uint64_t& profile_walk_length,
    uint64_t& num_workers,
    bool& force,
    std::string& sampling_backend,
    std::string& symmetric_topology,
    bool& fanouts_nearest_first
) {
    DictOptions opts(ctx.get_argument(arg_index));

    // Parse batchSize
    if (auto v = opts.get_int("batchSize")) {
        if (*v <= 0) {
            throw std::runtime_error("batchSize must be positive, got: " + std::to_string(*v));
        }
        batch_size = static_cast<uint64_t>(*v);
    }

    // Parse trainRatio
    if (auto v = opts.get_double("trainRatio")) {
        train_ratio = *v;
        if (train_ratio < 0.0 || train_ratio > 1.0) {
            throw std::runtime_error("trainRatio must be between 0.0 and 1.0");
        }
    }

    // Parse validationRatio
    if (auto v = opts.get_double("validationRatio")) {
        val_ratio = *v;
        if (val_ratio < 0.0 || val_ratio > 1.0) {
            throw std::runtime_error("validationRatio must be between 0.0 and 1.0");
        }
    }

    // Parse testRatio
    if (auto v = opts.get_double("testRatio")) {
        test_ratio = *v;
        if (test_ratio < 0.0 || test_ratio > 1.0) {
            throw std::runtime_error("testRatio must be between 0.0 and 1.0");
        }
    }

    // Parse usePredefinedSplits (before ratio validation). The explicitness
    // flag lets the splits.bin auto-default logic above defer to the caller's
    // choice when the user sets this option explicitly.
    if (auto v = opts.get_bool("usePredefinedSplits")) {
        use_predefined_splits = *v;
        use_predefined_splits_explicit = true;
    }

    // Validate ratios sum to 1.0 (only when not using predefined splits)
    if (!use_predefined_splits) {
        double total = train_ratio + val_ratio + test_ratio;
        if (total < 0.999 || total > 1.001) {
            throw std::runtime_error(
                "trainRatio + validationRatio + testRatio must equal 1.0, got: " +
                std::to_string(total)
            );
        }
    }

    // Parse randomSeed
    if (auto v = opts.get_int("randomSeed")) {
        random_seed = static_cast<uint64_t>(*v);
    }

    // Parse orientation
    if (auto v = opts.get_string("orientation")) {
        orientation = *v;
    }

    // Parse samplingBackend: 'auto' (default; decide GPU vs CPU by hardware),
    // 'cpu' (force the CPU out-of-core sampler, bit-reproducible reference), or
    // 'gpu' (force the GPU path; requires the Four-Level Topology Store).
    if (auto v = opts.get_string("samplingBackend")) {
        sampling_backend = *v;
    }

    // Symmetric topology: 'auto' (on for UNDIRECTED), 'on' (force the merged
    // undirected slice), or 'off' (keep the runtime out+in+merge path).
    if (auto v = opts.get_string("useSymmetricTopology")) {
        symmetric_topology = *v;
    }

    // Parse useAdjacencyCache: when true, a single full B+Tree scan over
    // from_to_edge and to_from_edge is performed at sampler startup and the
    // results are stored in an in-memory unordered_map<src, vector<AdjEntry>>,
    // turning each subsequent get_neighbors() call from O(log N) B+Tree lookup
    // into an O(1) hash lookup.
    if (auto v = opts.get_bool("useAdjacencyCache")) {
        use_adjacency_cache = *v;
    }

    // Four-Level Topology Store opt-ins. When enabled, the sampler assigns
    // nodes to four tiers based on access frequency: L1 (RAM hash for the
    // hottest hubs), L2 (compact uint32 CSR for warm nodes), L3 (mmap-backed
    // CSR sidecar files topology_{fwd,rev}.csr for the cold tail), and L4
    // (direct B+Tree fallback). l1CacheMb and l2CacheMb set the RAM budget
    // for each tier (0 = auto-detect from /proc/meminfo).
    if (auto v = opts.get_bool("useFourLevelTopologyStore")) {
        use_four_level_topology_store = *v;
    }
    if (auto v = opts.get_int("l1CacheMb")) {
        if (*v < 0) {
            throw std::runtime_error(
                "l1CacheMb must be non-negative, got: " + std::to_string(*v));
        }
        l1_cache_mb = static_cast<uint64_t>(*v);
    }
    if (auto v = opts.get_int("l2CacheMb")) {
        if (*v < 0) {
            throw std::runtime_error(
                "l2CacheMb must be non-negative, got: " + std::to_string(*v));
        }
        l2_cache_mb = static_cast<uint64_t>(*v);
    }
    if (auto v = opts.get_bool("useL3MmapSidecar")) {
        use_l3_mmap_sidecar = *v;
    }

    // Cold-start topology profiler opt-out and tuning. When node_counts.bin
    // is absent, the sampler can run a lightweight degree-weighted random-walk
    // pass (using a Vose alias table for seed selection) to estimate per-node
    // access frequencies before building the Four-Level Topology Store tier
    // assignments. autoProfileOnColdStart=false disables this pass entirely.
    // profileNumWalks and profileWalkLength tune the walk count and step depth.
    if (auto v = opts.get_bool("autoProfileOnColdStart")) {
        auto_profile_on_cold_start = *v;
    }
    if (auto v = opts.get_int("profileNumWalks")) {
        if (*v < 0) {
            throw std::runtime_error(
                "profileNumWalks must be non-negative, got: " + std::to_string(*v));
        }
        profile_num_walks = static_cast<uint64_t>(*v);
    }
    if (auto v = opts.get_int("profileWalkLength")) {
        if (*v < 0) {
            throw std::runtime_error(
                "profileWalkLength must be non-negative, got: " + std::to_string(*v));
        }
        profile_walk_length = static_cast<uint64_t>(*v);
    }

    // Parallel sampling worker count. Workers pull batches from a shared
    // atomic counter; each worker owns a private RNG + sampler state and
    // re-seeds deterministically per batch (random_seed XOR batch_id) so
    // output is reproducible regardless of which thread processes which batch.
    // 0 selects the legacy single-thread path; >= 1 activates the worker pool.
    if (auto v = opts.get_int("numWorkers")) {
        if (*v < 0) {
            throw std::runtime_error(
                "numWorkers must be non-negative, got: " + std::to_string(*v));
        }
        num_workers = static_cast<uint64_t>(*v);
    }

    // Parse force — drop + re-create the sample set when it already exists.
    if (auto v = opts.get_bool("force")) {
        force = *v;
    }

    // Parse fanoutsNearestFirst (legacy hop-order reading of the fanouts list)
    if (auto v = opts.get_bool("fanoutsNearestFirst")) {
        fanouts_nearest_first = *v;
    }
}
