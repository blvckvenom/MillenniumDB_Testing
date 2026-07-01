#include "query/procedure/builtin/gnn_checkpoint_delete_procedure.h"

#include <filesystem>

#include "query/procedure/procedure_context.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"
#include "gnn/output/model_checkpoint.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

void GnnCheckpointDeleteProcedure::execute(ProcedureContext& ctx) {
    if (ctx.arguments.size() != 3) {
        throw std::runtime_error(
            "gnn_checkpoint_delete requires 3 arguments: "
            "(projectionName, outputDir, name)");
    }
    auto projection_name = ctx.get_string_argument(0);
    auto output_dir_name = ctx.get_string_argument(1);
    auto name            = ctx.get_string_argument(2);
    validate_safe_name(name, "name");

    auto ckpt_dir = resolve_checkpoint_dir(projection_name, output_dir_name);
    auto basename = fs::absolute(ckpt_dir / name);
    auto pt_path   = basename.string() + ".pt";
    auto meta_path = basename.string() + ".ckptmeta";

    bool pt_existed   = fs::exists(pt_path);
    bool meta_existed = fs::exists(meta_path);

    mdb::gnn::ModelCheckpoint::delete_checkpoint(basename);

    ctx.yield("deleted",     ctx.create_bool(true));
    ctx.yield("basename",    ctx.create_string(basename.string()));
    ctx.yield("ptDeleted",   ctx.create_bool(pt_existed));
    ctx.yield("metaDeleted", ctx.create_bool(meta_existed));
    ctx.yield_row();
}

} // namespace GQL::Procedures
