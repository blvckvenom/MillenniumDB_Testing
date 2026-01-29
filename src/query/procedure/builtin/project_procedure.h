#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Edge orientation for graph projections.
 *
 * Controls the directionality of edges in the projection:
 * - NATURAL: Edges stored as specified (from → to) - default
 * - REVERSE: Edge direction reversed (to → from)
 * - UNDIRECTED: Edges stored bidirectionally (from ↔ to) - critical for GNN algorithms
 *
 * UNDIRECTED behavior depends on source edge type:
 * - **Directed source**: Stores two edges (A→B and B→A) with same edge_id
 * - **Undirected source**: No-op (already symmetric per ISO 39075 §3.4.13)
 *
 * This differs from Neo4j GDS, which always doubles (Neo4j has no native undirected edges).
 * MillenniumDB's approach is semantically correct: undirected edges are already symmetric.
 *
 * Storage overhead:
 * - NATURAL: 1.0× baseline
 * - REVERSE: 1.0× baseline
 * - UNDIRECTED on directed: ~1.75× baseline
 * - UNDIRECTED on undirected: 1.0× baseline (no-op)
 */
enum class Orientation {
    NATURAL,     ///< Default: edges as specified (from → to)
    REVERSE,     ///< Reversed: edges reversed (to → from)
    UNDIRECTED   ///< Undirected: bidirectional for directed edges, no-op for undirected
};

/**
 * @brief Aggregation strategy for parallel edges (multigraph support).
 *
 * Handles multiple edges between the same pair of nodes:
 * - SINGLE: Fail if parallel edges detected (default, strict validation)
 * - MIN: Keep edge with minimum property value
 * - MAX: Keep edge with maximum property value
 * - SUM: Sum property values across parallel edges
 * - COUNT: Count parallel edges and add as property
 *
 * Based on Neo4j GDS aggregation patterns for multigraph analysis.
 * See: NEO4J_AGGREGATION_PATTERNS.md for detailed semantics.
 *
 * Memory overhead: ~132 KB constant (independent of graph size)
 * Performance impact: 40% CPU overhead per edge (hash table lookup)
 *
 * @see Phase 2 implementation (M1.5_Implementation_Roadmap)
 */
enum class Aggregation {
    SINGLE,  ///< Default: fail on duplicate edges (strict validation)
    MIN,     ///< Minimum: keep edge with minimum property value
    MAX,     ///< Maximum: keep edge with maximum property value
    SUM,     ///< Sum: sum property values across parallel edges
    COUNT    ///< Count: count parallel edges (adds count property)
};

/**
 * @brief Property configuration for Neo4j GDS-compatible map syntax.
 *
 * Supports:
 * - Property renaming: `score: { property: 'rating' }` → read 'rating', store as 'score'
 * - Default values: `age: { defaultValue: 0 }` → use 0 if property missing
 * - Per-property aggregation: `weight: { aggregation: 'SUM' }` → aggregate this property
 *
 * Based on Neo4j GDS property configuration:
 * @code{.cypher}
 * properties: {
 *   age: { property: 'age', defaultValue: 0 },
 *   score: { property: 'rating', defaultValue: 1.0, aggregation: 'SUM' }
 * }
 * @endcode
 */
struct PropertyConfig {
    std::string source_property;     ///< Original property name (for renaming, empty = same as key)
    std::optional<double> default_value;  ///< Default if property missing (nullopt = NULL)
    Aggregation aggregation = Aggregation::SINGLE;  ///< Per-property aggregation (for edges)
};

/**
 * @brief Node projection configuration for Neo4j GDS-compatible map syntax.
 *
 * Supports per-label configuration in the map syntax:
 * @code{.gql}
 * CALL graph_project('graph', {
 *   Person: {
 *     label: 'Person',  // Optional: source label (defaults to key)
 *     properties: {
 *       age: { property: 'age', defaultValue: 0 },
 *       score: { property: 'rating', defaultValue: 1.0 }
 *     }
 *   },
 *   Book: { properties: ['title', 'price'] }  // Simple property list
 * }, ...)
 * @endcode
 */
struct NodeProjectionConfig {
    std::string label;               ///< Source label (may differ from key for aliasing)
    std::vector<std::string> simple_properties;  ///< Properties without extra config
    std::unordered_map<std::string, PropertyConfig> property_configs;  ///< Properties with config
};

/**
 * @brief Relationship projection configuration for Neo4j GDS-compatible map syntax.
 *
 * Supports per-type configuration with orientation and aggregation:
 * @code{.gql}
 * CALL graph_project('graph', 'Person', {
 *   KNOWS: {
 *     type: 'KNOWS',  // Optional: source type (defaults to key)
 *     orientation: 'UNDIRECTED',
 *     aggregation: 'NONE',
 *     properties: ['weight']
 *   },
 *   FOLLOWS: {
 *     orientation: 'NATURAL',
 *     aggregation: 'COUNT',
 *     properties: {
 *       strength: { defaultValue: 1.0, aggregation: 'SUM' }
 *     }
 *   }
 * }, { orientation: 'NATURAL' })  // Global defaults
 * @endcode
 */
struct RelationshipProjectionConfig {
    std::string type;                ///< Source type (may differ from key for aliasing)
    Orientation orientation = Orientation::NATURAL;   ///< Per-type orientation
    Aggregation aggregation = Aggregation::SINGLE;    ///< Per-type aggregation
    std::string aggregation_property;  ///< Property for MIN/MAX/SUM aggregation
    std::vector<std::string> simple_properties;  ///< Properties without extra config
    std::unordered_map<std::string, PropertyConfig> property_configs;  ///< Properties with config
};

/**
 * @brief Type alias for node projection map (key = projected label, value = config).
 */
using NodeProjectionMap = std::unordered_map<std::string, NodeProjectionConfig>;

/**
 * @brief Type alias for relationship projection map (key = projected type, value = config).
 */
using RelationshipProjectionMap = std::unordered_map<std::string, RelationshipProjectionConfig>;

/**
 * @brief Variant for node projection: either simple list or full map configuration.
 *
 * - `std::vector<std::string>`: Backward compatible string/list syntax
 * - `NodeProjectionMap`: Neo4j GDS-compatible map syntax
 */
using NodeProjectionVariant = std::variant<std::vector<std::string>, NodeProjectionMap>;

/**
 * @brief Variant for relationship projection: either simple list or full map configuration.
 *
 * - `std::vector<std::string>`: Backward compatible string/list syntax
 * - `RelationshipProjectionMap`: Neo4j GDS-compatible map syntax
 */
using RelationshipProjectionVariant = std::variant<std::vector<std::string>, RelationshipProjectionMap>;

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
     * Supports three variants:
     * - String: 'User' → ["User"]
     * - List: ['User', 'Post'] → ["User", "Post"]
     * - Map: {Person: {label: 'Person', properties: {...}}} → NodeProjectionMap
     *
     * @param ctx Procedure context.
     * @return Either vector of labels (backward compat) or NodeProjectionMap.
     * @throws std::runtime_error if parameter is invalid or list is empty.
     */
    NodeProjectionVariant parse_node_projection(ProcedureContext& ctx);

    /**
     * @brief Parses the relationshipProjection parameter (argument 2).
     *
     * Supports three variants:
     * - String: 'KNOWS' → ["KNOWS"]
     * - List: ['KNOWS', 'LIKES'] → ["KNOWS", "LIKES"]
     * - Map: {KNOWS: {orientation: 'UNDIRECTED', ...}} → RelationshipProjectionMap
     *
     * @param ctx Procedure context.
     * @return Either vector of types (backward compat) or RelationshipProjectionMap.
     * @throws std::runtime_error if parameter is invalid or list is empty.
     */
    RelationshipProjectionVariant parse_relationship_projection(ProcedureContext& ctx);

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

    /**
     * @brief Parses orientation parameter from config map.
     *
     * Extracts orientation value from config map.
     * Valid values: 'NATURAL', 'REVERSE', 'UNDIRECTED' (case-insensitive)
     *
     * @param ctx Procedure context.
     * @param config_map The configuration map.
     * @return Orientation enum value (defaults to NATURAL if key not present).
     * @throws std::runtime_error if value is invalid.
     */
    Orientation parse_orientation_from_config(
        ProcedureContext& ctx,
        ObjectId config_map
    );

    /**
     * @brief Parses aggregation parameter from config map.
     *
     * Extracts aggregation value from config map.
     * Valid values: 'SINGLE', 'MIN', 'MAX', 'SUM', 'COUNT' (case-insensitive)
     *
     * @param ctx Procedure context.
     * @param config_map The configuration map.
     * @return Aggregation enum value (defaults to SINGLE if key not present).
     * @throws std::runtime_error if value is invalid.
     */
    Aggregation parse_aggregation_from_config(
        ProcedureContext& ctx,
        ObjectId config_map
    );

    /**
     * @brief Parses aggregationProperty parameter from config map.
     *
     * Extracts which property to use for MIN/MAX/SUM aggregation.
     * If not specified, uses first property in relationshipProperties list.
     * For COUNT or SINGLE, returns empty string (no property needed).
     *
     * @param ctx Procedure context.
     * @param config_map The configuration map.
     * @param edge_properties List of edge properties from config.
     * @param aggregation The aggregation strategy.
     * @return Property name to use for aggregation (empty string if not needed).
     * @throws std::runtime_error if MIN/MAX/SUM requires property but none specified.
     */
    std::string parse_aggregation_property_from_config(
        ProcedureContext& ctx,
        ObjectId config_map,
        const std::vector<std::string>& edge_properties,
        Aggregation aggregation
    );

    // =========================================================================
    // Neo4j GDS Map Syntax Parsing Helpers (Phase 1)
    // =========================================================================

    /**
     * @brief Parses a node projection map from a DICTIONARY ObjectId.
     *
     * Handles the Neo4j GDS map syntax:
     * @code{.gql}
     * {
     *   Person: {
     *     label: 'Person',
     *     properties: { age: { defaultValue: 0 } }
     *   },
     *   Book: { properties: ['title', 'price'] }
     * }
     * @endcode
     *
     * @param ctx Procedure context.
     * @param dict_oid The DICTIONARY ObjectId to parse.
     * @return NodeProjectionMap with per-label configurations.
     * @throws std::runtime_error if the map structure is invalid.
     */
    NodeProjectionMap parse_node_projection_map(ProcedureContext& ctx, ObjectId dict_oid);

    /**
     * @brief Parses a relationship projection map from a DICTIONARY ObjectId.
     *
     * Handles the Neo4j GDS map syntax:
     * @code{.gql}
     * {
     *   KNOWS: {
     *     type: 'KNOWS',
     *     orientation: 'UNDIRECTED',
     *     aggregation: 'SUM',
     *     properties: ['weight']
     *   },
     *   FOLLOWS: { orientation: 'NATURAL' }
     * }
     * @endcode
     *
     * @param ctx Procedure context.
     * @param dict_oid The DICTIONARY ObjectId to parse.
     * @param global_orientation Global default orientation.
     * @param global_aggregation Global default aggregation.
     * @return RelationshipProjectionMap with per-type configurations.
     * @throws std::runtime_error if the map structure is invalid.
     */
    RelationshipProjectionMap parse_relationship_projection_map(
        ProcedureContext& ctx,
        ObjectId dict_oid,
        Orientation global_orientation,
        Aggregation global_aggregation
    );

    /**
     * @brief Parses a single node projection config from a nested dictionary.
     *
     * Handles:
     * @code{.gql}
     * Person: {
     *   label: 'Person',  // Optional, defaults to key
     *   properties: { age: { defaultValue: 0 } }  // or ['age', 'name']
     * }
     * @endcode
     *
     * @param ctx Procedure context.
     * @param config_oid The DICTIONARY ObjectId for this label's config.
     * @param projected_label The key name (used as default label).
     * @return NodeProjectionConfig for this label.
     */
    NodeProjectionConfig parse_single_node_config(
        ProcedureContext& ctx,
        ObjectId config_oid,
        const std::string& projected_label
    );

    /**
     * @brief Parses a single relationship projection config from a nested dictionary.
     *
     * Handles:
     * @code{.gql}
     * KNOWS: {
     *   type: 'KNOWS',  // Optional, defaults to key
     *   orientation: 'UNDIRECTED',
     *   aggregation: 'SUM',
     *   properties: ['weight']
     * }
     * @endcode
     *
     * @param ctx Procedure context.
     * @param config_oid The DICTIONARY ObjectId for this type's config.
     * @param projected_type The key name (used as default type).
     * @param global_orientation Default orientation if not specified.
     * @param global_aggregation Default aggregation if not specified.
     * @return RelationshipProjectionConfig for this type.
     */
    RelationshipProjectionConfig parse_single_relationship_config(
        ProcedureContext& ctx,
        ObjectId config_oid,
        const std::string& projected_type,
        Orientation global_orientation,
        Aggregation global_aggregation
    );

    /**
     * @brief Parses a property configuration from a nested dictionary.
     *
     * Handles:
     * @code{.gql}
     * age: {
     *   property: 'age',        // Source property (for renaming)
     *   defaultValue: 0,        // Default if missing
     *   aggregation: 'SUM'      // Per-property aggregation
     * }
     * @endcode
     *
     * @param ctx Procedure context.
     * @param config_oid The DICTIONARY ObjectId for this property's config.
     * @param property_key The key name (used as default source property).
     * @return PropertyConfig for this property.
     */
    PropertyConfig parse_property_config(
        ProcedureContext& ctx,
        ObjectId config_oid,
        const std::string& property_key
    );

    /**
     * @brief Parses properties from config, handling both list and map formats.
     *
     * Handles two formats:
     * - Simple list: `['age', 'name']` → simple_properties
     * - Map format: `{ age: { defaultValue: 0 } }` → property_configs
     *
     * @param ctx Procedure context.
     * @param properties_oid The ObjectId (LIST or DICTIONARY) for properties.
     * @param[out] simple_properties Vector for simple property names.
     * @param[out] property_configs Map for properties with configuration.
     */
    void parse_properties_value(
        ProcedureContext& ctx,
        ObjectId properties_oid,
        std::vector<std::string>& simple_properties,
        std::unordered_map<std::string, PropertyConfig>& property_configs
    );

    /**
     * @brief Extracts a string value from a dictionary by key.
     *
     * @param ctx Procedure context.
     * @param dict_obj The DictionaryObject to search.
     * @param key The key to look for.
     * @param default_value Value to return if key not found.
     * @return The string value or default_value.
     */
    std::string get_string_from_dict(
        ProcedureContext& ctx,
        void* dict_obj,
        const std::string& key,
        const std::string& default_value
    );

    /**
     * @brief Extracts an orientation value from a dictionary by key.
     *
     * @param ctx Procedure context.
     * @param dict_obj The DictionaryObject to search.
     * @param key The key to look for (usually "orientation").
     * @param default_value Value to return if key not found.
     * @return The Orientation enum value.
     */
    Orientation get_orientation_from_dict(
        ProcedureContext& ctx,
        void* dict_obj,
        const std::string& key,
        Orientation default_value
    );

    /**
     * @brief Extracts an aggregation value from a dictionary by key.
     *
     * @param ctx Procedure context.
     * @param dict_obj The DictionaryObject to search.
     * @param key The key to look for (usually "aggregation").
     * @param default_value Value to return if key not found.
     * @return The Aggregation enum value.
     */
    Aggregation get_aggregation_from_dict(
        ProcedureContext& ctx,
        void* dict_obj,
        const std::string& key,
        Aggregation default_value
    );

    /**
     * @brief Extracts an optional double value from a dictionary by key.
     *
     * @param ctx Procedure context.
     * @param dict_obj The DictionaryObject to search.
     * @param key The key to look for (usually "defaultValue").
     * @return The double value if present, nullopt otherwise.
     */
    std::optional<double> get_optional_double_from_dict(
        ProcedureContext& ctx,
        void* dict_obj,
        const std::string& key
    );

    /**
     * @brief Extracts an ObjectId value from a dictionary by key.
     *
     * @param dict_obj The DictionaryObject to search.
     * @param key The key to look for.
     * @param[out] found Set to true if key was found.
     * @return The ObjectId value if found, or NULL_OID.
     */
    ObjectId get_value_from_dict(void* dict_obj, const std::string& key, bool& found);

    /**
     * @brief Parses an orientation string to enum value.
     *
     * @param orientation_str The string (case-insensitive).
     * @return Orientation enum value.
     * @throws std::runtime_error if string is invalid.
     */
    Orientation parse_orientation_string(const std::string& orientation_str);

    /**
     * @brief Parses an aggregation string to enum value.
     *
     * @param aggregation_str The string (case-insensitive).
     * @return Aggregation enum value.
     * @throws std::runtime_error if string is invalid.
     */
    Aggregation parse_aggregation_string(const std::string& aggregation_str);
};

} // namespace Procedures
} // namespace GQL
