#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Lists all available graph projections.
 *
 * Returns metadata for all projections currently registered in the database,
 * including name, node count, relationship count, and creation time.
 *
 * Syntax:
 * @code{.gql}
 *   CALL graph_list()
 *   YIELD graphName, nodeCount, relationshipCount, createdAt
 * @endcode
 *
 * Examples:
 * @code{.gql}
 *   // List all projections
 *   CALL graph_list()
 *   YIELD graphName, nodeCount, relationshipCount
 *   RETURN graphName, nodeCount, relationshipCount
 *
 *   // Filter to large projections
 *   CALL graph_list() YIELD graphName, nodeCount
 *   WHERE nodeCount > 1000
 *   RETURN graphName, nodeCount
 *   ORDER BY nodeCount DESC
 *
 *   // Count projections
 *   CALL graph_list() YIELD graphName
 *   RETURN COUNT(*) AS projectionCount
 * @endcode
 *
 * @note If no projections exist, the procedure yields zero rows.
 *
 * @see ProjectionManager::list_projections() for underlying implementation
 * @see graph_exists for checking specific projection
 * @see graph_drop for deleting projections
 */
class GraphListProcedure : public Procedure {
public:
    /**
     * @brief Returns the simple name of the procedure.
     * @return "graph_list"
     */
    std::string name() const override {
        return "graph_list";
    }

    /**
     * @brief Returns the fully qualified name of the procedure.
     * @return "graph_list"
     */
    std::string qualified_name() const override {
        return "graph_list";
    }

    /**
     * @brief Returns a description of the procedure.
     * @return Human-readable description.
     */
    std::string description() const override {
        return "Lists all graph projections with their metadata (name, node count, relationship count, creation time).";
    }

    /**
     * @brief Returns the parameter specification.
     *
     * This procedure takes no parameters.
     *
     * @return Empty vector.
     */
    std::vector<Parameter> parameters() const override {
        return {};
    }

    /**
     * @brief Returns the yield field specification.
     *
     * YIELD fields:
     * - graphName: Name of the projection
     * - nodeCount: Number of nodes in projection
     * - relationshipCount: Number of relationships in projection
     * - createdAt: Unix timestamp of creation time
     *
     * @return Vector of yield field metadata.
     */
    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"graphName", YieldType::STRING,
                "Name of the projection"},
            YieldField{"nodeCount", YieldType::INT,
                "Number of nodes in the projection"},
            YieldField{"relationshipCount", YieldType::INT,
                "Number of relationships in the projection"},
            YieldField{"createdAt", YieldType::INT,
                "Unix timestamp when the projection was created"}
        };
    }

    /**
     * @brief Executes the projection listing.
     *
     * @param ctx Procedure context with yield interface.
     */
    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
