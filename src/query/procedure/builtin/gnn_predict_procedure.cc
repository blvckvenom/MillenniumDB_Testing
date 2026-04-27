#include "query/procedure/builtin/gnn_predict_procedure.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "query/procedure/procedure_context.h"

#include "gnn/models/graphsage_model.h"
#include "gnn/output/embedding_writer.h"
#include "gnn/output/model_checkpoint.h"
#include "gnn/projection/gnn_meta.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/four_level_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/batch_assembler.h"
#include "gnn/training/npy_writer.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

void GnnPredictProcedure::execute(ProcedureContext& ctx) {
    using namespace mdb::gnn;

    // ----- Argument parsing -----
    if (ctx.arguments.size() < 3 || ctx.arguments.size() > 4) {
        throw std::runtime_error(
            "gnn_predict requires 3-4 arguments.\n"
            "Usage: CALL gnn_predict(sampleName, featureName, checkpointPath [, options])");
    }

    auto sample_name    = ctx.get_string_argument(0);
    auto feature_name   = ctx.get_string_argument(1);
    auto checkpoint_arg = ctx.get_string_argument(2);
    validate_safe_name(sample_name,  "sampleName");
    validate_safe_name(feature_name, "featureName");

    std::string write_property;
    bool        export_emb       = true;
    std::string output_dir_name;

    if (ctx.arguments.size() == 4) {
        DictOptions opts(ctx.get_argument(3));
        if (auto v = opts.get_string("writeProperty")) {
            write_property = *v;
            if (!write_property.empty()) validate_safe_name(write_property, "writeProperty");
        }
        if (auto v = opts.get_bool("exportEmbeddings")) {
            export_emb = *v;
        }
        if (auto v = opts.get_string("outputDir")) {
            validate_safe_name(*v, "outputDir");
            output_dir_name = *v;
        }
    }

    if (output_dir_name.empty()) {
        // predict_<unix_seconds>
        auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        output_dir_name = "predict_" + std::to_string(now);
    }

    // ----- Validate sample + feature + gnn_meta + FourLevelStore -----
    auto db_folder      = get_db_folder();
    auto storage_path   = SampleStorage::get_storage_path(db_folder, sample_name);
    if (!fs::is_directory(storage_path)) {
        throw std::runtime_error("gnn_predict: sample '" + sample_name + "' not found");
    }
    auto samples        = SampleStorage::open(storage_path);
    const auto& catalog = samples.get_catalog();
    auto projection_name = catalog.projection_name;

    auto& proj_manager  = GQL::ProjectionManager::get_instance();
    if (!proj_manager.projection_exists(projection_name)) {
        throw std::runtime_error(
            "gnn_predict: projection '" + projection_name + "' no longer exists");
    }
    auto proj_dir = proj_manager.get_projection_dir(projection_name);
    auto meta_path = fs::path(proj_dir) / "gnn_meta.bin";
    if (!fs::exists(meta_path)) {
        throw std::runtime_error(
            "gnn_predict: gnn_meta.bin not found at " + meta_path.string());
    }
    auto meta = GnnMeta::read(meta_path);

    auto store_meta = fs::path(db_folder) / "gnn_features" / (feature_name + "_store.meta");
    if (!fs::exists(store_meta)) {
        throw std::runtime_error(
            "gnn_predict: FourLevelStore not found. "
            "Run CALL gnn_build_feature_store('" + sample_name + "', '" + feature_name + "', ...) first.");
    }

    // ----- Resolve checkpoint basename (relative or absolute) -----
    fs::path ckpt_basename;
    if (fs::path(checkpoint_arg).is_absolute()) {
        ckpt_basename = checkpoint_arg;
    } else {
        // Search all gnn_output subdirs for a matching checkpoint (first match wins)
        auto out_root = fs::path(proj_dir) / "gnn_output";
        if (fs::exists(out_root)) {
            for (const auto& sub : fs::directory_iterator(out_root)) {
                if (!sub.is_directory()) continue;
                auto cand = sub.path() / "checkpoints" / checkpoint_arg;
                if (ModelCheckpoint::exists(cand)) {
                    ckpt_basename = cand;
                    break;
                }
            }
        }
        if (ckpt_basename.empty()) {
            throw std::runtime_error(
                "gnn_predict: checkpoint '" + checkpoint_arg + "' not found in any "
                "output_dir under " + (fs::path(proj_dir) / "gnn_output").string());
        }
    }

    // ----- Peek checkpoint metadata, construct matching model -----
    auto peek_state = ModelCheckpoint::read_ckptmeta(
        fs::path(ckpt_basename.string() + ".ckptmeta"));

    GraphSAGEConfig gnn_config;
    gnn_config.input_dim   = peek_state.input_dim;
    gnn_config.hidden_dim  = peek_state.hidden_dim;
    gnn_config.num_classes = peek_state.num_classes;
    gnn_config.num_layers  = peek_state.num_layers;
    gnn_config.dropout     = peek_state.dropout;
    gnn_config.normalize   = peek_state.normalize;

    GraphSAGEModel model(gnn_config);
    auto loaded = ModelCheckpoint::load_weights(model, ckpt_basename);
    ModelCheckpoint::validate_compat(loaded, meta_path, projection_name);

    // ----- Open FourLevelStore + BatchAssembler (labels/splits optional) -----
    auto rmap_path = fs::path(db_folder) / "gnn_features" / (feature_name + ".rmap");
    auto rm        = RowMapping::open(rmap_path);
    FourLevelStore feature_store(db_folder, feature_name, samples);
    BatchAssembler assembler(feature_store, samples,
                             /*labels=*/nullptr, /*splits=*/nullptr, rm);

    // ----- Inference loop -----
    torch::NoGradGuard no_grad;
    model.eval();

    auto model_device = model.parameters().begin()->device();
    std::vector<torch::Tensor> all_embeddings;
    uint64_t total_seeds = 0;

    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        auto mini = assembler.assemble(bid);
        if (!model_device.is_cpu()) {
            mini.features = mini.features.to(model_device);
            for (auto& ei : mini.edge_indices) ei = ei.to(model_device);
        }
        auto emb = model.get_embeddings(
            mini.features, mini.edge_indices, static_cast<int64_t>(mini.num_seeds));
        all_embeddings.push_back(emb.cpu());
        total_seeds += mini.num_seeds;
    }

    // ----- Output dir for .npy export -----
    auto output_dir = fs::path(proj_dir) / "gnn_output" / output_dir_name;
    fs::create_directories(output_dir);

    if (export_emb && !all_embeddings.empty()) {
        auto combined = torch::cat(all_embeddings, /*dim=*/0);
        NpyWriter::write_float32(output_dir / "embeddings.npy", combined);
    }

    // ----- Optional write-back to projection -----
    uint64_t nodes_written  = 0;
    uint64_t nodes_inferred = 0;
    double   inference_ms   = 0.0;
    double   write_ms       = 0.0;

    if (!write_property.empty()) {
        EmbeddingWriter::Config wconfig;
        wconfig.property_name       = write_property;
        wconfig.fanouts             = catalog.fanouts;
        wconfig.orientation         = EdgeOrientation::UNDIRECTED;
        wconfig.feature_matrix_path = fs::path(db_folder) / "gnn_features" / (feature_name + ".fmat");

        GQL::ProjectionStorage proj_storage(proj_dir, db_folder);
        proj_storage.open();

        EmbeddingWriter writer(
            model, assembler, samples, rm, catalog, proj_storage, wconfig);
        auto wr = writer.write_all();
        nodes_written  = wr.nodes_written;
        nodes_inferred = wr.nodes_inferred;
        inference_ms   = wr.inference_ms;
        write_ms       = wr.write_ms;
    }

    // ----- Snapshot cache stats -----
    // Stats contains atomics (non-copyable); bind by reference and snapshot
    // each counter once for a consistent view in the YIELD row.
    const auto& stats = feature_store.get_stats();
    uint64_t l1h = stats.l1_hits.load();
    uint64_t l2h = stats.l2_hits.load();
    uint64_t l3r_count = stats.l3_reads.load();
    uint64_t l4r_count = stats.l4_reads.load();
    uint64_t tot = stats.total_requests.load();
    double l1r = tot > 0 ? double(l1h) / double(tot) : 0.0;
    double l2r = tot > 0 ? double(l2h) / double(tot) : 0.0;

    // Spec A1 (2026-04-27): byte-level disk-traffic accounting.
    uint64_t l3_bytes_disk   = stats.l3_bytes_disk.load();
    uint64_t l4_bytes_disk   = stats.l4_bytes_disk.load();
    uint64_t l3_bytes_wanted = stats.l3_bytes_wanted.load();
    uint64_t total_bytes_disk = l3_bytes_disk + l4_bytes_disk;
    double l3_amp = l3_bytes_wanted > 0
        ? double(l3_bytes_disk) / double(l3_bytes_wanted)
        : 0.0;

    // ----- Yields -----
    ctx.yield("checkpointPath",        ctx.create_string(std::filesystem::absolute(ckpt_basename).string()));
    ctx.yield("checkpointEpoch",       ctx.create_int(static_cast<int64_t>(loaded.epoch)));
    ctx.yield("checkpointValAccuracy", ctx.create_float(loaded.best_val_accuracy));
    ctx.yield("numBatches",            ctx.create_int(static_cast<int64_t>(catalog.total_batches)));
    ctx.yield("numSeedNodes",          ctx.create_int(static_cast<int64_t>(total_seeds)));
    ctx.yield("embeddingDim",          ctx.create_int(static_cast<int64_t>(gnn_config.hidden_dim)));
    ctx.yield("nodesWritten",          ctx.create_int(static_cast<int64_t>(nodes_written)));
    ctx.yield("nodesInferred",         ctx.create_int(static_cast<int64_t>(nodes_inferred)));
    ctx.yield("inferenceMillis",       ctx.create_float(static_cast<float>(inference_ms)));
    ctx.yield("writeMillis",           ctx.create_float(static_cast<float>(write_ms)));
    ctx.yield("l1HitRatio",            ctx.create_float(static_cast<float>(l1r)));
    ctx.yield("l2HitRatio",            ctx.create_float(static_cast<float>(l2r)));
    ctx.yield("l3Reads",               ctx.create_int(static_cast<int64_t>(l3r_count)));
    ctx.yield("l4Reads",               ctx.create_int(static_cast<int64_t>(l4r_count)));
    // Spec A1: byte-level disk-traffic surface — paper comparable.
    ctx.yield("l3BytesDisk",           ctx.create_int(static_cast<int64_t>(l3_bytes_disk)));
    ctx.yield("l4BytesDisk",           ctx.create_int(static_cast<int64_t>(l4_bytes_disk)));
    ctx.yield("totalBytesDisk",        ctx.create_int(static_cast<int64_t>(total_bytes_disk)));
    ctx.yield("l3ReadAmplification",   ctx.create_float(static_cast<float>(l3_amp)));
    ctx.yield_row();
}

} // namespace GQL::Procedures
