#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Checks if a graph projection exists.
 *
 * Returns a boolean indicating whether a projection with the given name
 * is currently registered in the projection catalog.
 *
 * Syntax:
 * @code{.gql}
 *   CALL graph_exists('projectionName')
 *   YIELD exists
 * @endcode
 *
 * Examples:
 * @code{.gql}
 *   // Check if a projection exists before creating
 *   CALL graph_exists('my_projection') YIELD exists
 *   RETURN exists
 *
 *   // Use in conditional logic
 *   CALL graph_exists('social_graph') YIELD exists
 *   WHERE exists = TRUE
 *   RETURN 'Projection is ready'
 * @endcode
 *
 * @see ProjectionManager::projection_exists() for underlying implementation
 * @see graph_drop for deleting projections
 * @see graph_list for listing all projections
 */
class GraphExistsProcedure : public Procedure {
public:
    /**
     * @brief Returns the simple name of the procedure.
     * @return "graph_exists"
     */
    std::string name() const override {
        return "graph_exists";
    }

    /**
     * @brief Returns the fully qualified name of the procedure.
     * @return "graph_exists"
     */
    std::string qualified_name() const override {
        return "graph_exists";
    }

    /**
     * @brief Returns a description of the procedure.
     * @return Human-readable description.
     */
    std::string description() const override {
        return "Checks if a graph projection with the given name exists.";
    }

    /**
     * @brief Returns the parameter specification.
     *
     * Parameters:
     * 1. graphName (STRING, required): Name of the projection to check
     *
     * @return Vector of parameter metadata.
     */
    std::vector<Parameter> parameters() const override {
        return {
            Parameter("graphName", ParamType::STRING, true,
                "Name of the projection to check for existence")
        };
    }

    /**
     * @brief Returns the yield field specification.
     *
     * YIELD fields:
     * - exists: Boolean indicating if projection exists
     *
     * @return Vector of yield field metadata.
     */
    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"exists", YieldType::BOOL,
                "TRUE if the projection exists, FALSE otherwise"}
        };
    }

    /**
     * @brief Executes the existence check.
     *
     * @param ctx Procedure context with arguments and yield interface.
     * @throws std::runtime_error for invalid parameters.
     */
    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
