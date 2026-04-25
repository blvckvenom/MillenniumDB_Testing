#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Deletes a graph projection and all its data.
 *
 * Removes a projection from the catalog and deletes its directory
 * including all index files and metadata.
 *
 * Syntax:
 * @code{.gql}
 *   CALL graph_drop('projectionName')
 *   YIELD success, message
 * @endcode
 *
 * Examples:
 * @code{.gql}
 *   // Drop a projection and check result
 *   CALL graph_drop('old_projection') YIELD success, message
 *   RETURN success, message
 *
 *   // Drop with confirmation
 *   CALL graph_drop('temp_graph') YIELD success
 *   WHERE success = TRUE
 *   RETURN 'Projection deleted successfully'
 * @endcode
 *
 * @note This operation is **irreversible**. All projection data is permanently deleted.
 * @warning The projection must not be in active use by other queries.
 *
 * @see ProjectionManager::drop_projection() for underlying implementation
 * @see graph_exists for checking existence before drop
 * @see graph_list for listing all projections
 */
class GraphDropProcedure : public Procedure {
public:
    /**
     * @brief Returns the simple name of the procedure.
     * @return "graph_drop"
     */
    std::string name() const override {
        return "graph_drop";
    }

    /**
     * @brief Returns the fully qualified name of the procedure.
     * @return "graph_drop"
     */
    std::string qualified_name() const override {
        return "graph_drop";
    }

    /**
     * @brief Returns a description of the procedure.
     * @return Human-readable description.
     */
    std::string description() const override {
        return "Deletes a graph projection and all its data. This operation is irreversible.";
    }

    /**
     * @brief Returns the parameter specification.
     *
     * Parameters:
     * 1. graphName (STRING, required): Name of the projection to delete
     *
     * @return Vector of parameter metadata.
     */
    std::vector<Parameter> parameters() const override {
        return {
            Parameter("graphName", ParamType::STRING, true,
                "Name of the projection to delete")
        };
    }

    /**
     * @brief Returns the yield field specification.
     *
     * YIELD fields:
     * - success: Boolean indicating if deletion succeeded
     * - message: Human-readable status message
     *
     * @return Vector of yield field metadata.
     */
    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"success", YieldType::BOOL,
                "TRUE if the projection was successfully deleted"},
            YieldField{"message", YieldType::STRING,
                "Human-readable status message describing the result"}
        };
    }

    /**
     * @brief Executes the projection deletion.
     *
     * @param ctx Procedure context with arguments and yield interface.
     * @throws std::runtime_error for invalid parameters.
     */
    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
