#include "query/procedure/builtin/gnn_list_checkpoints_procedure.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "query/procedure/procedure_context.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"
#include "gnn/output/model_checkpoint.h"
#include "graph_models/gql/projection/projection_manager.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

void GnnListCheckpointsProcedure::execute(ProcedureContext& ctx) {
    using namespace mdb::gnn;

    if (ctx.arguments.size() < 1 || ctx.arguments.size() > 3) {
        throw std::runtime_error(
            "gnn_list_checkpoints requires 1-3 arguments.\n"
            "Usage: CALL gnn_list_checkpoints(projectionName [, outputDir [, name]])");
    }

    auto projection_name = ctx.get_string_argument(0);
    std::optional<std::string> output_dir_name;
    std::optional<std::string> name_filter;
    if (ctx.arguments.size() >= 2) output_dir_name = ctx.get_string_argument(1);
    if (ctx.arguments.size() >= 3) name_filter     = ctx.get_string_argument(2);

    auto& pm = GQL::ProjectionManager::get_instance();
    if (!pm.projection_exists(projection_name)) {
        throw std::runtime_error(
            "gnn_list_checkpoints: projection '" + projection_name + "' not found");
    }

    std::vector<fs::path> scan_dirs;
    if (output_dir_name) {
        validate_safe_name(*output_dir_name, "outputDir");
        scan_dirs.push_back(
            fs::path(pm.get_projection_dir(projection_name))
            / "gnn_output" / *output_dir_name / "checkpoints");
    } else {
        auto root = fs::path(pm.get_projection_dir(projection_name)) / "gnn_output";
        if (fs::exists(root)) {
            for (const auto& sub : fs::directory_iterator(root)) {
                if (sub.is_directory()) {
                    scan_dirs.push_back(sub.path() / "checkpoints");
                }
            }
        }
    }

    std::vector<CheckpointInfo> all;
    for (const auto& d : scan_dirs) {
        auto cps = ModelCheckpoint::list_checkpoints(d, name_filter);
        all.insert(all.end(), cps.begin(), cps.end());
    }

    std::sort(all.begin(), all.end(),
              [](const CheckpointInfo& a, const CheckpointInfo& b) {
                  return a.creation_time_unix > b.creation_time_unix;
              });

    for (const auto& info : all) {
        auto out_dir_comp = info.basename.parent_path().parent_path().filename().string();
        std::string kind_str =
            (info.save_kind == SaveKind::Full) ? "full" : "weights_only";

        ctx.yield("basename",        ctx.create_string(info.basename.string()));
        ctx.yield("outputDir",       ctx.create_string(out_dir_comp));
        ctx.yield("saveKind",        ctx.create_string(kind_str));
        ctx.yield("epoch",           ctx.create_int(static_cast<int64_t>(info.epoch)));
        ctx.yield("bestValAccuracy", ctx.create_float(info.best_val_accuracy));
        ctx.yield("modelType",       ctx.create_string(info.model_type));
        ctx.yield("projectionName",  ctx.create_string(info.projection_name));
        ctx.yield("creationTime",    ctx.create_int(static_cast<int64_t>(info.creation_time_unix)));
        ctx.yield("ptBytes",         ctx.create_int(static_cast<int64_t>(info.pt_bytes)));
        ctx.yield("metaBytes",       ctx.create_int(static_cast<int64_t>(info.meta_bytes)));
        ctx.yield_row();
    }
}

} // namespace GQL::Procedures
