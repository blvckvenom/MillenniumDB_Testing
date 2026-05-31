#include "query/procedure/builtin/gnn_train_procedure.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "query/procedure/procedure_context.h"

#include "gnn/models/graphsage_model.h"
#include "gnn/output/auto_checkpointer.h"
#include "gnn/output/embedding_writer.h"
#include "gnn/output/model_checkpoint.h"
#include "gnn/projection/gnn_meta.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/cache_stats_snapshot.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/four_level_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/batch_assembler.h"
#include "gnn/training/label_store.h"
#include "gnn/training/mini_batch.h"
#include "gnn/training/npy_writer.h"
#include "gnn/training/split_store.h"
#include "gnn/training/training_loop.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

// =============================================================================
// Helper: write training_log.json
// =============================================================================

// Spec B2 (2026-04-27): CacheStatsSnapshot moved to a public header so
// TrainingLoop can take it as a callback return type. Re-export under the
// GQL::Procedures namespace via using-alias to keep this file's internal
// references unqualified.
using CacheStatsSnapshot = mdb::gnn::CacheStatsSnapshot;

static void write_training_log(
    const fs::path&                          output_dir,
    const std::string&                       model_name,
    const std::string&                       sample_name,
    const std::string&                       feature_name,
    const std::string&                       projection_name,
    const mdb::gnn::TrainingLoop::Config&    loop_config,
    const mdb::gnn::GraphSAGEConfig&         gnn_config,
    const mdb::gnn::TrainingLoop::Result&    result,
    double                                   test_accuracy,
    bool                                     exported_embeddings,
    const CacheStatsSnapshot&                cache_stats)
{
    std::ofstream f(output_dir / "training_log.json");
    if (!f.is_open()) {
        throw std::runtime_error(
            "gnn_train: cannot write training_log.json to: " + output_dir.string());
    }

    f << std::fixed << std::setprecision(6);
    f << "{\n";
    f << "  \"version\": 1,\n";
    f << "  \"model\": \"" << model_name << "\",\n";
    f << "  \"sample_name\": \"" << sample_name << "\",\n";
    f << "  \"feature_name\": \"" << feature_name << "\",\n";
    f << "  \"projection_name\": \"" << projection_name << "\",\n";

    // Hyperparameters
    f << "  \"hyperparameters\": {\n";
    f << "    \"input_dim\": " << gnn_config.input_dim << ",\n";
    f << "    \"hidden_dim\": " << gnn_config.hidden_dim << ",\n";
    f << "    \"num_classes\": " << gnn_config.num_classes << ",\n";
    f << "    \"num_layers\": " << gnn_config.num_layers << ",\n";
    f << "    \"dropout\": " << gnn_config.dropout << ",\n";
    f << "    \"normalize\": " << (gnn_config.normalize ? "true" : "false") << ",\n";
    f << "    \"learning_rate\": " << loop_config.learning_rate << ",\n";
    f << "    \"weight_decay\": " << loop_config.weight_decay << ",\n";
    f << "    \"epochs\": " << loop_config.epochs << ",\n";
    f << "    \"patience\": " << loop_config.patience << ",\n";
    f << "    \"tolerance\": " << loop_config.tolerance << ",\n";
    f << "    \"random_seed\": " << loop_config.random_seed << "\n";
    f << "  },\n";

    // Results
    f << "  \"results\": {\n";
    f << "    \"ran_epochs\": " << result.ran_epochs << ",\n";
    f << "    \"converged\": " << (result.converged ? "true" : "false") << ",\n";
    f << "    \"best_val_accuracy\": " << result.best_val_accuracy << ",\n";
    f << "    \"test_accuracy\": " << test_accuracy << ",\n";
    f << "    \"train_seconds\": " << result.train_seconds << ",\n";
    // Spec C3 stage 0: per-stage timing breakdown (sum across all train
    // batches across all epochs). Per-session, NOT accumulated across
    // resume_from continuations.
    f << "    \"assemble_seconds\": " << result.assemble_seconds << ",\n";
    f << "    \"forward_seconds\": "  << result.forward_seconds  << ",\n";
    f << "    \"backward_seconds\": " << result.backward_seconds << "\n";
    f << "  },\n";

    // Per-epoch losses
    f << "  \"epoch_losses\": [";
    for (size_t i = 0; i < result.epoch_losses.size(); ++i) {
        if (i > 0) f << ", ";
        f << result.epoch_losses[i];
    }
    f << "],\n";

    f << "  \"exported_embeddings\": " << (exported_embeddings ? "true" : "false") << ",\n";

    // FourLevelStore cache statistics — DiskGNN cache hierarchy diagnostics.
    // l1_hit_ratio + l2_hit_ratio + (l3_reads + l4_reads) / total_requests = 1.0
    // when every requested node was resolved (note that l3 also accumulates
    // misses where the node was outside the projection — see four_level_store.cc).
    //
    // Spec A1 (2026-04-27): byte-level fields added for paper-comparable
    // disk-traffic accounting (DiskGNN SIGMOD'25 Table 1 column
    // "Disk access volume (GB)" = total_bytes_disk).
    f << "  \"cache_stats\": {\n";
    f << "    \"l1_hits\": "              << cache_stats.l1_hits              << ",\n";
    f << "    \"l2_hits\": "              << cache_stats.l2_hits              << ",\n";
    f << "    \"l3_reads\": "             << cache_stats.l3_reads             << ",\n";
    f << "    \"l4_reads\": "             << cache_stats.l4_reads             << ",\n";
    f << "    \"total_requests\": "       << cache_stats.total_requests       << ",\n";
    f << "    \"l1_hit_ratio\": "         << cache_stats.l1_hit_ratio()       << ",\n";
    f << "    \"l2_hit_ratio\": "         << cache_stats.l2_hit_ratio()       << ",\n";
    f << "    \"l1_bytes_served\": "      << cache_stats.l1_bytes_served      << ",\n";
    f << "    \"l2_bytes_served\": "      << cache_stats.l2_bytes_served      << ",\n";
    f << "    \"l3_bytes_wanted\": "      << cache_stats.l3_bytes_wanted      << ",\n";
    f << "    \"l3_bytes_disk\": "        << cache_stats.l3_bytes_disk        << ",\n";
    f << "    \"l4_bytes_wanted\": "      << cache_stats.l4_bytes_wanted      << ",\n";
    f << "    \"l4_bytes_disk\": "        << cache_stats.l4_bytes_disk        << ",\n";
    f << "    \"total_bytes_disk\": "     << cache_stats.total_bytes_disk()   << ",\n";
    f << "    \"l3_read_amplification\": "<< cache_stats.l3_read_amplification() << "\n";
    f << "  }\n";
    f << "}\n";

    if (!f) {
        throw std::runtime_error(
            "gnn_train: I/O error writing training_log.json to: " + output_dir.string());
    }
}

// =============================================================================
// Helper: export embeddings (hidden representations before classifier)
// =============================================================================

static void export_embeddings(
    mdb::gnn::GraphSAGEModel&      model,
    mdb::gnn::BatchAssembler&      assembler,
    const mdb::gnn::SampleCatalog& catalog,
    const fs::path&                output_dir)
{
    torch::NoGradGuard no_grad;
    model.eval();

    std::vector<torch::Tensor> all_embeddings;
    auto model_device = model.parameters().begin()->device();

    // Single pass: assemble each batch, compute the seed embeddings, accumulate.
    // (A previous version ran this loop twice — plus a discarded second
    // assemble() per batch, ~3x the inference work — then cleared the first
    // pass and recomputed. Output is unchanged: embeddings are laid out
    // batch 0 seeds, batch 1 seeds, ... batch N seeds; node IDs are
    // recoverable from the sample catalog.)
    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        auto mini = assembler.assemble(bid);
        if (!model_device.is_cpu()) {
            mini.features = mini.features.to(model_device);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(model_device);
            }
            for (auto& ai : mini.active_indices_per_layer) {
                ai = ai.to(model_device);
            }
        }
        auto emb = model.get_embeddings(
            mini.features,
            mini.edge_indices,
            mini.active_sizes_per_layer
        );
        all_embeddings.push_back(emb.cpu());
    }

    if (!all_embeddings.empty()) {
        auto combined = torch::cat(all_embeddings, /*dim=*/0);
        mdb::gnn::NpyWriter::write_float32(output_dir / "embeddings.npy", combined);
    }

    model.train();
}

// =============================================================================
// Procedure execution
// =============================================================================

void GnnTrainProcedure::execute(ProcedureContext& ctx) {
    using namespace mdb::gnn;

    // =========================================================================
    // Step 1: Parse arguments
    // =========================================================================
    if (ctx.arguments.size() < 2 || ctx.arguments.size() > 3) {
        throw std::runtime_error(
            "gnn_train requires 2-3 arguments.\n\n"
            "Usage: CALL gnn_train(sampleName, featureName [, options])\n"
            "Example: CALL gnn_train('my_sample', 'node_features', "
            "{model: 'graphsage', epochs: 50, lr: 0.01})");
    }

    // Parse sampleName
    std::string sample_name;
    try {
        sample_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid sampleName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING.\n"
            "Example: CALL gnn_train('my_sample', 'node_features')");
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
            "Example: CALL gnn_train('my_sample', 'node_features')");
    }
    if (feature_name.empty()) {
        throw std::runtime_error("featureName cannot be empty.");
    }
    validate_safe_name(feature_name, "featureName");

    // =========================================================================
    // Step 2: Parse options
    // =========================================================================
    std::string model_type       = "graphsage";
    int64_t     hidden_dim       = 256;
    double      dropout          = 0.5;
    uint64_t    epochs           = 50;
    double      lr               = 0.01;
    double      weight_decay     = 0.0;
    uint64_t    patience         = 5;
    double      tolerance        = 1e-4;
    bool        normalize        = false;
    int64_t     random_seed      = -1;
    std::string output_dir_name  = "default";
    bool        export_emb       = true;
    std::string write_property;              // writeProperty option (empty = no write-back)
    uint64_t    inference_batch_size = 0;    // 0 = use EmbeddingWriter default (Phase B chunk size)
    std::string resume_from;       // empty = fresh training
    bool        save_on_best_val = true;
    bool        save_final       = true;
    std::string profile_log_path = "";  // Phase 0 per-batch timing CSV (empty = disabled)
    // Spec C3 stage 1 (default true since 2026-05-07): async prefetcher.
    // 1.609× speedup measured on papers100M, bit-identical accuracy.
    bool        use_async_prefetcher = true;
    // Spec C3 stage 3 (started 2026-05-08): split assemble_kernel and
    // model.forward+backward onto separate CUDA streams. Default false
    // until empirical validation in Module 6.
    bool        use_cuda_streams     = false;
    // Round 3B (2026-05-15): number of worker threads in AsyncBatchPrefetcher.
    // Default 1 = byte-identical to Stage 1 behavior. >1 has effect only
    // when the BatchAssembler runs in FeatureMatrix-fallback mode (see
    // TrainingLoop::Config::prefetch_num_workers for thread-safety notes
    // and the FourLevelStore clamp).
    uint64_t    prefetch_num_workers = 1;

    if (ctx.arguments.size() == 3) {
        DictOptions opts(ctx.get_argument(2));

        if (auto v = opts.get_string("model")) {
            std::string m = *v;
            std::transform(m.begin(), m.end(), m.begin(), ::tolower);
            if (m != "graphsage") {
                throw std::runtime_error(
                    "Unsupported model: '" + *v + "'. Currently only 'graphsage' is supported.");
            }
            model_type = m;
        }
        if (auto v = opts.get_int("hiddenDim")) {
            if (*v <= 0) throw std::runtime_error("hiddenDim must be positive, got: " + std::to_string(*v));
            hidden_dim = *v;
        }
        if (auto v = opts.get_double("dropout")) {
            if (*v < 0.0 || *v >= 1.0) throw std::runtime_error("dropout must be in [0, 1), got: " + std::to_string(*v));
            dropout = *v;
        }
        if (auto v = opts.get_int("epochs")) {
            if (*v <= 0) throw std::runtime_error("epochs must be positive, got: " + std::to_string(*v));
            epochs = static_cast<uint64_t>(*v);
        }
        if (auto v = opts.get_double("lr")) {
            if (*v <= 0.0) throw std::runtime_error("lr must be positive, got: " + std::to_string(*v));
            lr = *v;
        }
        if (auto v = opts.get_double("weightDecay")) {
            if (*v < 0.0) throw std::runtime_error("weightDecay must be non-negative, got: " + std::to_string(*v));
            weight_decay = *v;
        }
        if (auto v = opts.get_int("patience")) {
            if (*v <= 0) throw std::runtime_error("patience must be positive, got: " + std::to_string(*v));
            patience = static_cast<uint64_t>(*v);
        }
        if (auto v = opts.get_double("tolerance")) {
            if (*v < 0.0) throw std::runtime_error("tolerance must be non-negative, got: " + std::to_string(*v));
            tolerance = *v;
        }
        // Spec C3 stage 1.B: opt-in async batch prefetcher.
        if (auto v = opts.get_bool("useAsyncPrefetcher")) {
            use_async_prefetcher = *v;
        }
        // Spec C3 stage 3: opt-in CUDA streams for assemble vs train overlap.
        if (auto v = opts.get_bool("useCudaStreams")) {
            use_cuda_streams = *v;
        }
        // Round 3B (2026-05-15): multi-worker AsyncBatchPrefetcher.
        if (auto v = opts.get_int("prefetchNumWorkers")) {
            if (*v < 1) {
                throw std::runtime_error(
                    "prefetchNumWorkers must be >= 1, got: " +
                    std::to_string(*v));
            }
            prefetch_num_workers = static_cast<uint64_t>(*v);
        }
        if (auto v = opts.get_bool("normalize")) {
            normalize = *v;
        }
        if (auto v = opts.get_int("randomSeed")) {
            random_seed = *v;
        }
        if (auto v = opts.get_string("outputDir")) {
            if (v->empty()) throw std::runtime_error("outputDir cannot be empty.");
            validate_safe_name(*v, "outputDir");
            output_dir_name = *v;
        }
        if (auto v = opts.get_bool("exportEmbeddings")) {
            export_emb = *v;
        }
        if (auto v = opts.get_string("writeProperty")) {
            write_property = *v;
            if (write_property.empty()) {
                throw std::runtime_error("writeProperty must be a non-empty string");
            }
            validate_safe_name(write_property, "writeProperty");
        }
        if (auto v = opts.get_int("inferenceBatchSize")) {
            if (*v <= 0) {
                throw std::runtime_error(
                    "inferenceBatchSize must be positive, got: "
                    + std::to_string(*v));
            }
            inference_batch_size = static_cast<uint64_t>(*v);
        }
        if (auto v = opts.get_string("resumeFrom")) {
            resume_from = *v;
            // Empty string is allowed (= fresh training); non-empty paths validated at load time
        }
        if (auto v = opts.get_bool("saveOnBestVal")) {
            save_on_best_val = *v;
        }
        if (auto v = opts.get_bool("saveFinal")) {
            save_final = *v;
        }
        // Phase 0 (2026-05-17): per-batch profile CSV path. Empty disables.
        if (auto v = opts.get_string("profileLog")) {
            profile_log_path = *v;
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

    // Validate the FourLevelStore metadata exists. gnn_train is DiskGNN-faithful
    // and requires the hierarchical feature store to be pre-built. There is no
    // silent fallback to a plain mmap: if the store is missing, tell the user
    // how to build it.
    auto store_meta_path =
        fs::path(db_folder) / "gnn_features" / (feature_name + "_store.meta");
    if (!fs::exists(store_meta_path)) {
        throw std::runtime_error(
            "gnn_train requires a pre-built FourLevelStore (DiskGNN feature store).\n"
            "Store metadata not found at: " + store_meta_path.string() + "\n"
            "Run: CALL gnn_build_feature_store('" + sample_name + "', '"
            + feature_name + "', {gpu_budget_mb: <N>, cpu_budget_mb: <M>})\n"
            "before calling gnn_train.");
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
    // Step 4: Open data sources
    //
    // The original FeatureMatrix is intentionally NOT opened here: features
    // are served exclusively through FourLevelStore below, which reads from
    // its own reordered .fmat / _cpu_cache.bin / packed_slim files. The
    // RowMapping is still required because BatchAssembler uses it for label
    // lookups (labels.bin is indexed by the ORIGINAL row indices).
    // =========================================================================
    auto rm = RowMapping::open(rmap_path);
    auto samples = SampleStorage::open(storage_path);
    const auto& catalog = samples.get_catalog();
    std::string projection_name = catalog.projection_name;

    // Projection directory (for GnnMeta, labels, splits)
    auto& proj_manager = GQL::ProjectionManager::get_instance();
    if (!proj_manager.projection_exists(projection_name)) {
        throw std::runtime_error(
            "Projection '" + projection_name + "' referenced by sample '"
            + sample_name + "' no longer exists.");
    }
    std::string proj_dir = proj_manager.get_projection_dir(projection_name);

    // GnnMeta (written by graph_project with GNN config)
    auto meta_path = fs::path(proj_dir) / "gnn_meta.bin";
    if (!fs::exists(meta_path)) {
        throw std::runtime_error(
            "GNN metadata not found at: " + meta_path.string() + "\n"
            "The projection '" + projection_name + "' was not created with GNN config.\n"
            "Recreate with: CALL graph_project(name, nodes, rels, {gnn: {featureName: ...}})");
    }
    auto meta = GnnMeta::read(meta_path);

    // Labels (conditional — written by graph_project when labelProperty is set)
    std::unique_ptr<LabelStore> labels;
    auto labels_path = fs::path(proj_dir) / "labels.bin";
    if (fs::exists(labels_path)) {
        labels = std::make_unique<LabelStore>(LabelStore::open(labels_path));
    }

    // Splits (conditional — written by graph_project when splitProperty is set)
    std::unique_ptr<SplitStore> splits;
    auto splits_path = fs::path(proj_dir) / "splits.bin";
    if (fs::exists(splits_path)) {
        splits = std::make_unique<SplitStore>(SplitStore::open(splits_path));
    }

    // Consistency guard: labels.bin and splits.bin must agree on row indexing
    // (both indexed by rmap row). If labels.bin was built against an older
    // rmap and the rmap was later rewritten, TRAIN rows in splits.bin will
    // map to -1 labels — model trains on noise. Scan first N rmap rows and
    // verify the invariant; abort with remediation hint if violated.
    if (labels && splits) {
        constexpr uint64_t SCAN_LIMIT = 100000;
        const uint64_t scan_n = std::min<uint64_t>(SCAN_LIMIT, splits->num_nodes());
        uint64_t train_seen = 0;
        uint64_t train_with_label = 0;
        for (uint64_t r = 0; r < scan_n; ++r) {
            if (splits->get(r) == SplitStore::TRAIN) {
                train_seen++;
                if (labels->get(r) >= 0) train_with_label++;
            }
        }
        if (train_seen >= 16 && train_with_label * 4 < train_seen) {
            throw std::runtime_error(
                "gnn_train: labels.bin appears mis-indexed against rmap. "
                "Scanned " + std::to_string(scan_n) + " rmap rows, found " +
                std::to_string(train_seen) + " marked TRAIN but only " +
                std::to_string(train_with_label) + " have a valid (non -1) label. "
                "Likely cause: rmap was rewritten after labels.bin was built. "
                "Fix: regenerate labels.bin + splits.bin against the current rmap "
                "(scripts/regenerate_labels_splits.py) or rebuild the projection "
                "with graph_project."
            );
        }
    }

    // =========================================================================
    // Step 5: Configure model and training
    // =========================================================================

    // Infer num_layers from sample catalog fanouts
    int64_t num_layers = static_cast<int64_t>(catalog.num_layers());
    if (num_layers < 1) {
        throw std::runtime_error(
            "Sample catalog has 0 layers (fanouts empty). Cannot train a GNN model.");
    }

    GraphSAGEConfig gnn_config;
    gnn_config.input_dim   = static_cast<int64_t>(meta.feature_dim);
    gnn_config.hidden_dim  = hidden_dim;
    gnn_config.num_classes = static_cast<int64_t>(meta.num_classes);
    gnn_config.num_layers  = num_layers;
    gnn_config.dropout     = dropout;
    gnn_config.normalize   = normalize;

    if (gnn_config.input_dim <= 0) {
        throw std::runtime_error(
            "Invalid feature_dim=" + std::to_string(meta.feature_dim)
            + " in GNN metadata. Must be positive.");
    }
    if (gnn_config.num_classes <= 0) {
        throw std::runtime_error(
            "Invalid num_classes=" + std::to_string(meta.num_classes)
            + " in GNN metadata. Must be positive.");
    }

    // Reproducibility seed
    if (random_seed >= 0) {
        torch::manual_seed(static_cast<uint64_t>(random_seed));
    }

    GraphSAGEModel model(gnn_config);

    // =========================================================================
    // Step 6: Create output directory
    // =========================================================================
    auto output_dir = fs::path(proj_dir) / "gnn_output" / output_dir_name;
    fs::create_directories(output_dir);

    // =========================================================================
    // Step 7: Open FourLevelStore and create BatchAssembler in full mode
    //
    // FourLevelStore reads the hierarchical feature store built by
    // gnn_build_feature_store: L1 (GPU) + L2 (CPU pinned) + L3 (disk reordered)
    // + L4 (packed slim per-batch). Its stats counters let us verify that all
    // four tiers are exercised during training — this is the DiskGNN contract.
    // =========================================================================
    mdb::gnn::FourLevelStore feature_store(db_folder, feature_name, samples);

    // STEP 2 contract guard: the projection's gnn_meta.bin and the feature
    // store's store.meta are two independent sources of feature_dim. If they
    // disagree (projection re-imported / rebuilt with a different feature width
    // but the feature store not rebuilt), the model is sized to meta.feature_dim
    // while assemble() emits store.feature_dim()-wide rows -> an opaque torch
    // shape error mid-forward. Fail early with a clear remediation message.
    if (feature_store.feature_dim() != static_cast<uint64_t>(meta.feature_dim)) {
        throw std::runtime_error(
            "gnn_train: feature_dim mismatch — projection gnn_meta.bin reports "
            + std::to_string(meta.feature_dim) + " but the feature store reports "
            + std::to_string(feature_store.feature_dim())
            + ". The feature store is stale for this projection. Rebuild it: "
              "CALL gnn_build_feature_store('" + sample_name + "', '"
            + feature_name + "', {force:true}).");
    }
    BatchAssembler assembler(feature_store, samples, labels.get(), splits.get(), rm);

    // =========================================================================
    // Step 8: Run training
    // =========================================================================
    TrainingLoop::Config loop_config;
    loop_config.epochs        = epochs;
    loop_config.learning_rate = lr;
    loop_config.weight_decay  = weight_decay;
    loop_config.tolerance     = tolerance;
    loop_config.patience      = patience;
    loop_config.random_seed   = random_seed;
    loop_config.use_async_prefetcher = use_async_prefetcher;
    loop_config.use_cuda_streams     = use_cuda_streams;
    loop_config.prefetch_num_workers = static_cast<unsigned>(prefetch_num_workers);
    loop_config.output_dir    = output_dir.string();
    loop_config.profile_log_path = profile_log_path;  // Phase 0 instrumentation

    // =========================================================================
    // Step 8a: Build base TrainingState (identifying fields) for checkpoints
    // =========================================================================
    mdb::gnn::TrainingState base_state;
    base_state.input_dim          = gnn_config.input_dim;
    base_state.hidden_dim         = gnn_config.hidden_dim;
    base_state.num_classes        = gnn_config.num_classes;
    base_state.num_layers         = gnn_config.num_layers;
    base_state.dropout            = gnn_config.dropout;
    base_state.normalize          = gnn_config.normalize;
    base_state.model_type         = model_type;   // "graphsage"
    base_state.projection_name    = projection_name;
    base_state.gnn_meta_hash      = mdb::gnn::ModelCheckpoint::compute_gnn_meta_hash(meta_path);
    base_state.creation_time_unix =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

    torch::optim::Adam optimizer(
        model.parameters(),
        torch::optim::AdamOptions(lr).weight_decay(weight_decay)
    );

    // =========================================================================
    // Step 8b: Resume from checkpoint if requested
    // =========================================================================
    uint64_t resumed_from_epoch = 0;
    if (!resume_from.empty()) {
        fs::path resume_basename;
        fs::path resume_input(resume_from);
        if (resume_input.is_absolute()) {
            resume_basename = resume_input;
        } else {
            resume_basename = output_dir / "checkpoints" / resume_from;
        }

        auto loaded = mdb::gnn::ModelCheckpoint::load_full(
            model, optimizer, resume_basename);
        mdb::gnn::ModelCheckpoint::validate_compat(
            loaded, meta_path, projection_name);

        loop_config.start_epoch     = loaded.epoch;
        loop_config.start_patience  = loaded.patience_counter;
        loop_config.start_best_val  = loaded.best_val_accuracy;
        loop_config.seed_losses     = loaded.epoch_losses;
        resumed_from_epoch          = loaded.epoch;

        // Preserve original creation time + accumulate total time
        base_state.creation_time_unix      = loaded.creation_time_unix;
        base_state.total_training_time_sec = loaded.total_training_time_sec;
    }

    // =========================================================================
    // Step 8c: Construct AutoCheckpointer and wire its callback
    // =========================================================================
    mdb::gnn::AutoCheckpointer::Policy ac_policy;
    ac_policy.save_on_best_val = save_on_best_val;
    ac_policy.save_final       = save_final;

    auto ckpt_dir = output_dir / "checkpoints";
    mdb::gnn::AutoCheckpointer autockpt(
        model, optimizer, ckpt_dir, base_state, ac_policy);

    // Spec B2 (2026-04-27): wire cumulative L3+L4 disk-bytes provider so
    // the training loop can compute per-epoch deltas inline. Returns a
    // single uint64_t (not a struct) to keep TrainingLoop::Config ABI
    // stable across translation units. Captured by reference —
    // feature_store outlives the lambda.
    loop_config.cumulative_disk_bytes_provider =
        [&feature_store]() -> uint64_t {
            const auto& s = feature_store.get_stats();
            return s.l3_bytes_disk.load() + s.l4_bytes_disk.load();
        };

    loop_config.on_epoch_end = [&autockpt](const mdb::gnn::TrainingLoop::EpochEvent& e) {
        autockpt.on_epoch_end(e);
    };

    TrainingLoop loop(model, assembler, catalog, optimizer, loop_config);
    auto result = loop.train();

    // =========================================================================
    // Step 8d: Finalize checkpoint
    // =========================================================================
    mdb::gnn::TrainingState final_state  = base_state;
    final_state.epoch                    = loop_config.start_epoch + result.ran_epochs;
    final_state.patience_counter         = 0;  // exhausted / reset at end
    final_state.best_val_accuracy        = static_cast<float>(result.best_val_accuracy);
    final_state.epoch_losses             = result.epoch_losses;
    final_state.total_training_time_sec  = base_state.total_training_time_sec + result.train_seconds;

    autockpt.save_final(final_state);

    std::string best_checkpoint_str;
    if (save_on_best_val && autockpt.best_val_seen() > 0.0) {
        best_checkpoint_str =
            std::filesystem::absolute(ckpt_dir / ac_policy.best_basename).string();
    }
    std::string final_checkpoint_str;
    if (save_final) {
        final_checkpoint_str =
            std::filesystem::absolute(ckpt_dir / ac_policy.final_basename).string();
    }

    // =========================================================================
    // Step 9: Evaluate on test set
    // =========================================================================
    double test_accuracy = -1.0;
    if (labels && catalog.test_batches > 0) {
        test_accuracy = loop.evaluate(
            catalog.train_batches + catalog.validation_batches,
            catalog.test_batches);
    }

    // =========================================================================
    // Step 10: Save final model checkpoint
    // =========================================================================
    {
        torch::serialize::OutputArchive archive;
        model.save(archive);
        archive.save_to((output_dir / "model.pt").string());
    }

    // =========================================================================
    // Step 11: Export embeddings
    // =========================================================================
    bool did_export_emb = false;
    if (export_emb) {
        export_embeddings(model, assembler, catalog, output_dir);
        did_export_emb = true;
    }

    // =========================================================================
    // Step 11.5: Write embeddings to projection (if writeProperty is set)
    //
    // Opens ProjectionStorage for the EmbeddingWriter which needs topology
    // access (Phase B: on-the-fly k-hop inference for non-seed nodes) and
    // property write access (Phase C: persist embeddings as tensor properties).
    // Fanouts and orientation are taken from the SampleCatalog so that
    // Phase B sampling is consistent with the original offline sampling.
    // =========================================================================
    uint64_t nodes_written  = 0;
    uint64_t nodes_inferred = 0;
    double   inference_ms   = 0.0;
    double   write_ms       = 0.0;

    if (!write_property.empty()) {
        EmbeddingWriter::Config wconfig;
        wconfig.property_name      = write_property;
        wconfig.fanouts            = catalog.fanouts;
        wconfig.orientation        = EdgeOrientation::UNDIRECTED;
        wconfig.feature_matrix_path = fmat_path;
        if (inference_batch_size > 0) {
            wconfig.batch_size = inference_batch_size;
        }

        GQL::ProjectionStorage proj_storage(proj_dir, db_folder);
        proj_storage.open();

        EmbeddingWriter writer(
            model,
            assembler,
            samples,
            rm,
            catalog,
            proj_storage,
            wconfig
        );
        auto wresult = writer.write_all();

        nodes_written  = wresult.nodes_written;
        nodes_inferred = wresult.nodes_inferred;
        inference_ms   = wresult.inference_ms;
        write_ms       = wresult.write_ms;
    }

    // =========================================================================
    // Step 11.6: Snapshot FourLevelStore cache statistics
    //
    // Snapshot the atomic counters once after training/eval/export are done
    // so the JSON log and the YIELD row report consistent values. This is
    // the DiskGNN diagnostic surface — without it, the four-level cache
    // hierarchy is invisible to the user.
    // =========================================================================
    auto cache_stats = CacheStatsSnapshot::from(feature_store.get_stats());

    // Spec A1: headline disk-traffic print so the user sees the
    // paper-comparable number directly in their server terminal
    // (alongside the per-epoch lines from training_loop.cc).
    {
        constexpr double GB = 1024.0 * 1024.0 * 1024.0;
        double total_gb = static_cast<double>(cache_stats.total_bytes_disk()) / GB;
        double l3_gb    = static_cast<double>(cache_stats.l3_bytes_disk)      / GB;
        double l4_gb    = static_cast<double>(cache_stats.l4_bytes_disk)      / GB;
        std::cout << "[gnn_train] disk traffic total="
                  << std::fixed << std::setprecision(2) << total_gb << " GB"
                  << " (l3=" << l3_gb << ", l4=" << l4_gb << ")"
                  << "  l3_amplification=" << std::setprecision(3)
                  << cache_stats.l3_read_amplification() << "x\n";
    }

    // =========================================================================
    // Step 12: Write training log
    // =========================================================================
    write_training_log(
        output_dir, model_type, sample_name, feature_name,
        projection_name, loop_config, gnn_config, result,
        test_accuracy, did_export_emb, cache_stats);

    // =========================================================================
    // Step 13: Yield results
    // =========================================================================
    ctx.yield("modelName",       ctx.create_string(model_type));
    ctx.yield("ranEpochs",       ctx.create_int(static_cast<int64_t>(result.ran_epochs)));
    ctx.yield("didConverge",     ctx.create_bool(result.converged));
    ctx.yield("bestValAccuracy", ctx.create_float(static_cast<float>(result.best_val_accuracy)));
    ctx.yield("testAccuracy",    ctx.create_float(static_cast<float>(test_accuracy)));
    ctx.yield("trainSeconds",    ctx.create_float(static_cast<float>(result.train_seconds)));
    // Spec C3 stage 0 (2026-05-07): per-stage cumulative timings for
    // baselining the upcoming async-prefetcher / pipeline-overlap work.
    // assemble + forward + backward should sum to ≈ train phase wall time
    // (excluding eval). On a typical run, asssembleSeconds is the upper
    // bound on what stage 1 prefetcher can hide behind compute.
    ctx.yield("assembleSeconds", ctx.create_float(static_cast<float>(result.assemble_seconds)));
    ctx.yield("forwardSeconds",  ctx.create_float(static_cast<float>(result.forward_seconds)));
    ctx.yield("backwardSeconds", ctx.create_float(static_cast<float>(result.backward_seconds)));
    ctx.yield("l1HitRatio",      ctx.create_float(static_cast<float>(cache_stats.l1_hit_ratio())));
    ctx.yield("l2HitRatio",      ctx.create_float(static_cast<float>(cache_stats.l2_hit_ratio())));
    ctx.yield("l3Reads",         ctx.create_int(static_cast<int64_t>(cache_stats.l3_reads)));
    ctx.yield("l4Reads",         ctx.create_int(static_cast<int64_t>(cache_stats.l4_reads)));
    // Spec A1: byte-level disk-traffic surface — paper comparable.
    ctx.yield("l3BytesDisk",
              ctx.create_int(static_cast<int64_t>(cache_stats.l3_bytes_disk)));
    ctx.yield("l4BytesDisk",
              ctx.create_int(static_cast<int64_t>(cache_stats.l4_bytes_disk)));
    ctx.yield("totalBytesDisk",
              ctx.create_int(static_cast<int64_t>(cache_stats.total_bytes_disk())));
    ctx.yield("l3ReadAmplification",
              ctx.create_float(static_cast<float>(cache_stats.l3_read_amplification())));
    ctx.yield("nodesWritten",    ctx.create_int(static_cast<int64_t>(nodes_written)));
    ctx.yield("nodesInferred",   ctx.create_int(static_cast<int64_t>(nodes_inferred)));
    ctx.yield("inferenceMillis", ctx.create_float(static_cast<float>(inference_ms)));
    ctx.yield("writeMillis",     ctx.create_float(static_cast<float>(write_ms)));
    ctx.yield("bestCheckpointPath",  ctx.create_string(best_checkpoint_str));
    ctx.yield("finalCheckpointPath", ctx.create_string(final_checkpoint_str));
    ctx.yield("resumedFromEpoch",    ctx.create_int(static_cast<int64_t>(resumed_from_epoch)));
    ctx.yield("effectivePrefetchWorkers",
              ctx.create_int(static_cast<int64_t>(result.effective_prefetch_workers)));
    // Path 4 (2026-05-19): v2 addr_table fast-path telemetry from TrainingLoop::Result.
    ctx.yield("useAddrTablesEffective",
              ctx.create_bool(result.addr_tables_used_ever));
    ctx.yield("addrTableLoadUs",
              ctx.create_double(result.addr_table_load_us_mean));
    ctx.yield_row();
}

} // namespace GQL::Procedures
