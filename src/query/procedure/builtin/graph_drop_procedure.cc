#include "graph_drop_procedure.h"

#include <stdexcept>

#include "graph_models/gql/projection/projection_manager.h"

namespace GQL {
namespace Procedures {

void GraphDropProcedure::execute(ProcedureContext& ctx) {
    // Validate argument count
    if (ctx.arguments.size() < 1) {
        throw std::runtime_error(
            "graph_drop() requires 1 argument: graphName\n"
            "Usage: CALL graph_drop('projectionName') YIELD success, message"
        );
    }

    // Get the projection name
    std::string graph_name = ctx.get_string_argument(0);

    // Validate name is not empty
    if (graph_name.empty()) {
        throw std::runtime_error(
            "Invalid projection name: name cannot be empty.\n"
            "Provide a non-empty string as the argument.\n"
            "Example: CALL graph_drop('myProjection') YIELD success, message"
        );
    }

    // Get projection manager
    auto& manager = ProjectionManager::get_instance();

    // Check if projection exists first
    if (!manager.projection_exists(graph_name)) {
        // Projection doesn't exist - return success=false with descriptive message
        ctx.yield("success", ctx.create_bool(false));
        ctx.yield("message", ctx.create_string(
            "Projection '" + graph_name + "' does not exist"
        ));
        ctx.yield_row();
        return;
    }

    // Attempt to drop the projection
    bool dropped = manager.drop_projection(graph_name);

    if (dropped) {
        ctx.yield("success", ctx.create_bool(true));
        ctx.yield("message", ctx.create_string(
            "Projection '" + graph_name + "' has been deleted"
        ));
    } else {
        // This case shouldn't happen since we checked existence,
        // but handle it gracefully
        ctx.yield("success", ctx.create_bool(false));
        ctx.yield("message", ctx.create_string(
            "Failed to delete projection '" + graph_name + "'"
        ));
    }
    ctx.yield_row();
}

} // namespace Procedures
} // namespace GQL
