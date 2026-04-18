#include "query/procedure/builtin/gnn_checkpoint_exists_procedure.h"

#include <filesystem>

#include "query/procedure/procedure_context.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"
#include "gnn/output/model_checkpoint.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

void GnnCheckpointExistsProcedure::execute(ProcedureContext& ctx) {
    if (ctx.arguments.size() != 3) {
        throw std::runtime_error(
            "gnn_checkpoint_exists requires 3 arguments: "
            "(projectionName, outputDir, name)");
    }
    auto projection_name = ctx.get_string_argument(0);
    auto output_dir_name = ctx.get_string_argument(1);
    auto name            = ctx.get_string_argument(2);

    auto ckpt_dir = resolve_checkpoint_dir(projection_name, output_dir_name);
    auto basename = fs::absolute(ckpt_dir / name);
    bool e = mdb::gnn::ModelCheckpoint::exists(basename);

    ctx.yield("exists",   ctx.create_bool(e));
    ctx.yield("basename", ctx.create_string(basename.string()));
    ctx.yield_row();
}

} // namespace GQL::Procedures
