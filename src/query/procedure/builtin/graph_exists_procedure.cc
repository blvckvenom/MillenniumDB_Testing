#include "graph_exists_procedure.h"

#include <stdexcept>

#include "graph_models/gql/projection/projection_manager.h"

namespace GQL {
namespace Procedures {

void GraphExistsProcedure::execute(ProcedureContext& ctx) {
    // Validate argument count
    if (ctx.arguments.size() < 1) {
        throw std::runtime_error(
            "graph_exists() requires 1 argument: graphName\n"
            "Usage: CALL graph_exists('projectionName') YIELD exists"
        );
    }

    // Get the projection name
    std::string graph_name = ctx.get_string_argument(0);

    // Validate name is not empty
    if (graph_name.empty()) {
        throw std::runtime_error(
            "Invalid projection name: name cannot be empty.\n"
            "Provide a non-empty string as the argument.\n"
            "Example: CALL graph_exists('myProjection') YIELD exists"
        );
    }

    // Check if projection exists using ProjectionManager
    auto& manager = ProjectionManager::get_instance();
    bool exists = manager.projection_exists(graph_name);

    // Yield the result
    ctx.yield("exists", ctx.create_bool(exists));
    ctx.yield_row();
}

} // namespace Procedures
} // namespace GQL
