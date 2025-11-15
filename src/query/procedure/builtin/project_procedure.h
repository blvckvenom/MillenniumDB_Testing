#pragma once

#include <memory>
#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Native graph projection procedure for MillenniumDB.
 *
 * Creates disk-based graph projections by scanning label_node and label_edge
 * B+Tree indexes directly, achieving O(n+m) complexity versus Cypher's O(n²+m²).
 *
 * This procedure implements the CALL PROJECT(...) YIELD ... syntax for creating
 * native graph projections without pattern matching overhead.
 *
 * Syntax:
 * @code{.gql}
 *   CALL PROJECT(graphName, nodeProjection, relationshipProjection [, config])
 *   YIELD graphName, nodeCount, relationshipCount, projectMillis
 * @endcode
 *
 * Examples:
 * @code{.gql}
 *   // String variant - single label/type
 *   CALL PROJECT('myGraph', 'User', 'KNOWS')
 *   YIELD graphName, nodeCount, relationshipCount
 *
 *   // List variant - multiple labels/types
 *   CALL PROJECT('social', ['User', 'Post'], ['KNOWS', 'LIKES'])
 *   YIELD graphName, nodeCount, relationshipCount, projectMillis
 * @endcode
 *
 * @see ARCHITECTURE_DESIGN.md Section 3.1 for complete specification
 * @see ISO/IEC 39075:2024 Section 15 for CALL/YIELD semantics
 */
class ProjectProcedure : public Procedure {
public:
    /**
     * @brief Returns the simple name of the procedure.
     * @return "graph_project" (avoiding reserved keyword PROJECT)
     */
    std::string name() const override {
        return "graph_project";
    }

    /**
     * @brief Returns the fully qualified name of the procedure.
     * @return "graph_project" (simplified - GQL doesn't support dotted names in CALL yet)
     */
    std::string qualified_name() const override {
        return "graph_project";
    }

    /**
     * @brief Returns a description of the procedure.
     * @return Human-readable description of what the procedure does.
     */
    std::string description() const override {
        return "Creates a native graph projection by scanning node labels and edge types directly. "
               "Achieves O(n+m) performance by bypassing pattern matching and scanning B+Tree indexes.";
    }

    /**
     * @brief Returns the parameter specification for this procedure.
     *
     * Parameters:
     * 1. graphName (STRING, required): Name of the projection to create
     * 2. nodeProjection (ANY, required): String label or list of labels to include
     * 3. relationshipProjection (ANY, required): String type or list of types to include
     * 4. configuration (ANY, optional): Configuration map (not implemented in M1.2)
     *
     * @return Vector of parameter metadata.
     */
    std::vector<Parameter> parameters() const override {
        return {
            Parameter("graphName", ParamType::STRING, true,
                "Name of the projection to create"),
            Parameter("nodeProjection", ParamType::ANY, true,
                "String label (e.g., 'User') or list of labels (e.g., ['User', 'Post'])"),
            Parameter("relationshipProjection", ParamType::ANY, true,
                "String type (e.g., 'KNOWS') or list of types (e.g., ['KNOWS', 'LIKES'])"),
            Parameter("configuration", ParamType::ANY, false,
                "Optional configuration map (reserved for future use)")
        };
    }

    /**
     * @brief Returns the yield field specification for this procedure.
     *
     * YIELD fields:
     * - graphName: Name of created projection
     * - nodeCount: Total nodes in projection
     * - relationshipCount: Total relationships in projection
     * - projectMillis: Time taken to create projection (milliseconds)
     *
     * @return Vector of yield field metadata.
     */
    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"graphName", YieldType::STRING,
                "Name of the created projection"},
            YieldField{"nodeCount", YieldType::INT,
                "Total number of nodes in the projection"},
            YieldField{"relationshipCount", YieldType::INT,
                "Total number of relationships in the projection"},
            YieldField{"projectMillis", YieldType::INT,
                "Time taken to create the projection (milliseconds)"}
        };
    }

    /**
     * @brief Executes the native projection creation.
     *
     * Execution steps:
     * 1. Validate parameter count (3-4 parameters required)
     * 2. Extract graphName (string)
     * 3. Parse nodeProjection (string or list variant)
     * 4. Parse relationshipProjection (string or list variant)
     * 5. Validate that all labels/types exist in the database
     * 6. Create NativeProjectionBuilder
     * 7. Build projection by scanning B+Trees
     * 8. Yield results (graphName, counts, time)
     *
     * Error conditions:
     * - Invalid parameter count → std::runtime_error
     * - Invalid parameter types → std::runtime_error
     * - Empty lists → std::runtime_error
     * - Nonexistent labels/types → std::runtime_error
     * - Duplicate projection name → std::runtime_error
     * - Disk/permission errors → std::runtime_error
     *
     * @param ctx Procedure context with arguments and yield interface.
     * @throws std::runtime_error for invalid parameters or projection errors.
     * @throws std::out_of_range if argument access is out of bounds.
     */
    void execute(ProcedureContext& ctx) override;

private:
    /**
     * @brief Parses the graphName parameter (argument 0).
     *
     * @param ctx Procedure context.
     * @return The projection name as a string.
     * @throws std::runtime_error if parameter is not a string.
     */
    std::string parse_graph_name(ProcedureContext& ctx);

    /**
     * @brief Parses the nodeProjection parameter (argument 1).
     *
     * Supports two variants:
     * - String: 'User' → ["User"]
     * - List: ['User', 'Post'] → ["User", "Post"]
     *
     * @param ctx Procedure context.
     * @return Vector of node label strings.
     * @throws std::runtime_error if parameter is invalid or list is empty.
     */
    std::vector<std::string> parse_node_projection(ProcedureContext& ctx);

    /**
     * @brief Parses the relationshipProjection parameter (argument 2).
     *
     * Supports two variants:
     * - String: 'KNOWS' → ["KNOWS"]
     * - List: ['KNOWS', 'LIKES'] → ["KNOWS", "LIKES"]
     *
     * @param ctx Procedure context.
     * @return Vector of relationship type strings.
     * @throws std::runtime_error if parameter is invalid or list is empty.
     */
    std::vector<std::string> parse_relationship_projection(ProcedureContext& ctx);

    /**
     * @brief Validates that a node label exists in the database.
     *
     * @param label The label string to validate.
     * @throws std::runtime_error if the label does not exist.
     */
    void validate_label_exists(const std::string& label);

    /**
     * @brief Validates that a relationship type exists in the database.
     *
     * @param type The type string to validate.
     * @throws std::runtime_error if the type does not exist.
     */
    void validate_type_exists(const std::string& type);

    /**
     * @brief Parses property list from config map.
     *
     * Extracts a list of property names from config map by key.
     * Supports: ['age', 'city'] or 'age' (single property)
     *
     * @param ctx Procedure context.
     * @param config_map The configuration map.
     * @param key The key to look up ('nodeProperties' or 'relationshipProperties').
     * @return Vector of property names (empty if key not present).
     * @throws std::runtime_error if value is invalid type.
     */
    std::vector<std::string> parse_property_list_from_config(
        ProcedureContext& ctx,
        ObjectId config_map,
        const std::string& key
    );
};

} // namespace Procedures
} // namespace GQL
