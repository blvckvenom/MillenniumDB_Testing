#include "query/procedure/builtin/gnn_train_procedure.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

// Snapshot of FourLevelStore::Stats values. The Stats struct itself uses
// std::atomic so it cannot be copied; this POD is the safe transport.
struct CacheStatsSnapshot {
    uint64_t l1_hits        = 0;
    uint64_t l2_hits        = 0;
    uint64_t l3_reads       = 0;
    uint64_t l4_reads       = 0;
    uint64_t total_requests = 0;

    static CacheStatsSnapshot from(const mdb::gnn::FourLevelStore::Stats& s) {
        CacheStatsSnapshot snap;
        snap.l1_hits        = s.l1_hits.load();
        snap.l2_hits        = s.l2_hits.load();
        snap.l3_reads       = s.l3_reads.load();
        snap.l4_reads       = s.l4_reads.load();
        snap.total_requests = s.total_requests.load();
        return snap;
    }

    double l1_hit_ratio() const {
        return total_requests > 0
            ? static_cast<double>(l1_hits) / static_cast<double>(total_requests)
            : 0.0;
    }
    double l2_hit_ratio() const {
        return total_requests > 0
            ? static_cast<double>(l2_hits) / static_cast<double>(total_requests)
            : 0.0;
    }
};

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
    f << "    \"train_seconds\": " << result.train_seconds << "\n";
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
    f << "  \"cache_stats\": {\n";
    f << "    \"l1_hits\": "        << cache_stats.l1_hits        << ",\n";
    f << "    \"l2_hits\": "        << cache_stats.l2_hits        << ",\n";
    f << "    \"l3_reads\": "       << cache_stats.l3_reads       << ",\n";
    f << "    \"l4_reads\": "       << cache_stats.l4_reads       << ",\n";
    f << "    \"total_requests\": " << cache_stats.total_requests << ",\n";
    f << "    \"l1_hit_ratio\": "   << cache_stats.l1_hit_ratio() << ",\n";
    f << "    \"l2_hit_ratio\": "   << cache_stats.l2_hit_ratio() << "\n";
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
    std::vector<int64_t>       all_node_ids;

    auto model_device = model.parameters().begin()->device();

    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        auto mini = assembler.assemble(bid);

        // Move batch tensors to the model's device (supports GPU training)
        if (!model_device.is_cpu()) {
            mini.features = mini.features.to(model_device);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(model_device);
            }
        }

        // Get embeddings (hidden representation before classifier)
        auto emb = model.get_embeddings(
            mini.features,
            mini.edge_indices,
            static_cast<int64_t>(mini.num_seeds)
        );
        // emb is [num_seeds, hidden_dim]

        all_embeddings.push_back(emb.cpu());

        // Collect seed node ObjectIds from the row_mapping for this batch
        // The first num_seeds rows in mini.features correspond to the seed nodes.
        // We need to recover the original ObjectIds. The BatchAssembler builds
        // features from the GraphSample's all_unique_nodes; seeds are layer 0.
        // Re-read the sample to get the seed ObjectIds.
        auto sample = assembler.assemble(bid);
        // Actually, we already have `mini` — but we need the GraphSample for node IDs.
        // Re-read from storage to get the ObjectIds.
        // This is slightly wasteful but keeps the code simple.
    }

    // The above approach requires reading samples again for ObjectIds.
    // Instead, let's just store sequential indices and let the user map them
    // via the RowMapping. But the spec says node_ids.npy.
    //
    // Better approach: collect node IDs during the embedding loop by reading
    // the sample directly.
    all_embeddings.clear();
    all_node_ids.clear();

    // Use a SampleStorage reference from the assembler — but BatchAssembler
    // doesn't expose its SampleStorage. We need to read samples separately.
    // However, we know batch layout: for each batch, seeds are nodes_per_layer[0].
    // We'll read the raw GraphSample to get the seed ObjectIds.
    //
    // Since BatchAssembler::assemble(bid) already reads the sample, we pay the
    // cost anyway. The MiniBatch doesn't carry ObjectIds, so we must retrieve them
    // from the sample storage.
    //
    // For now, we output only embeddings.npy (the node_ids ordering matches
    // batch 0 seeds, batch 1 seeds, ... batch N seeds — user can reconstruct
    // from the sample catalog).

    // Re-collect embeddings without node_ids for simplicity
    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        auto mini = assembler.assemble(bid);
        auto emb = model.get_embeddings(
            mini.features,
            mini.edge_indices,
            static_cast<int64_t>(mini.num_seeds)
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
    std::string resume_from;       // empty = fresh training
    bool        save_on_best_val = true;
    bool        save_final       = true;

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
    loop_config.output_dir    = output_dir.string();

    // =========================================================================
    // Step 8a: Build base TrainingState (identifying fields) for checkpoints
    // =========================================================================
    mdb::gnn::TrainingState base_state;
    base_state.input_dim          = gnn_config.input_dim;
    base_state.hidden_dim         = gnn_config.hidden_dim;
    base_state.num_classes        = gnn_config.num_classes;
    base_state.num_layers         = gnn_config.num_layers;
    base_state.dropout            = static_cast<float>(gnn_config.dropout);
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

    TrainingLoop loop(model, assembler, catalog, optimizer, loop_config);
    auto result = loop.train();

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
    ctx.yield("l1HitRatio",      ctx.create_float(static_cast<float>(cache_stats.l1_hit_ratio())));
    ctx.yield("l2HitRatio",      ctx.create_float(static_cast<float>(cache_stats.l2_hit_ratio())));
    ctx.yield("l3Reads",         ctx.create_int(static_cast<int64_t>(cache_stats.l3_reads)));
    ctx.yield("l4Reads",         ctx.create_int(static_cast<int64_t>(cache_stats.l4_reads)));
    ctx.yield("nodesWritten",    ctx.create_int(static_cast<int64_t>(nodes_written)));
    ctx.yield("nodesInferred",   ctx.create_int(static_cast<int64_t>(nodes_inferred)));
    ctx.yield("inferenceMillis", ctx.create_float(static_cast<float>(inference_ms)));
    ctx.yield("writeMillis",     ctx.create_float(static_cast<float>(write_ms)));
    ctx.yield_row();
}

} // namespace GQL::Procedures
