#include "graph_list_procedure.h"

#include "graph_models/gql/projection/projection_manager.h"

namespace GQL {
namespace Procedures {

void GraphListProcedure::execute(ProcedureContext& ctx) {
    // Get projection manager
    auto& manager = ProjectionManager::get_instance();

    // Get all projections
    std::vector<ProjectionInfo> projections = manager.list_projections();

    // Yield one row per projection
    for (const auto& info : projections) {
        ctx.yield("graphName", ctx.create_string(info.name));
        ctx.yield("nodeCount", ctx.create_int(static_cast<int64_t>(info.node_count)));
        ctx.yield("relationshipCount", ctx.create_int(static_cast<int64_t>(info.edge_count)));
        ctx.yield("createdAt", ctx.create_int(static_cast<int64_t>(info.creation_timestamp)));
        ctx.yield_row();
    }

    // If no projections exist, the result is simply empty (zero rows)
    // This is the expected behavior - no special handling needed
}

} // namespace Procedures
} // namespace GQL
