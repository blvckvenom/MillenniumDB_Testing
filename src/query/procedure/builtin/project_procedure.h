#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

class DictionaryObject;

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

/// Type alias for node projection map (key = projected label, value = config).
using NodeProjectionMap = std::unordered_map<std::string, NodeProjectionConfig>;

/// Type alias for relationship projection map (key = projected type, value = config).
using RelationshipProjectionMap = std::unordered_map<std::string, RelationshipProjectionConfig>;

/// Variant for node projection: either simple list or full map configuration.
using NodeProjectionVariant = std::variant<std::vector<std::string>, NodeProjectionMap>;

/// Variant for relationship projection: either simple list or full map configuration.
using RelationshipProjectionVariant = std::variant<std::vector<std::string>, RelationshipProjectionMap>;

/**
 * @brief Native graph projection procedure for MillenniumDB.
 *
 * Creates disk-based graph projections by scanning label_node and label_edge
 * B+Tree indexes directly, achieving O(n+m) complexity versus Cypher's O(n²+m²).
 *
 * Syntax:
 * @code{.gql}
 *   CALL PROJECT(graphName, nodeProjection, relationshipProjection [, config])
 *   YIELD graphName, nodeCount, relationshipCount, projectMillis
 * @endcode
 *
 * Examples:
 * @code{.gql}
 *   CALL PROJECT('myGraph', 'User', 'KNOWS')
 *   CALL PROJECT('social', ['User', 'Post'], ['KNOWS', 'LIKES'])
 * @endcode
 *
 * @see ARCHITECTURE_DESIGN.md Section 3.1 for complete specification
 * @see ISO/IEC 39075:2024 Section 15 for CALL/YIELD semantics
 */
class ProjectProcedure : public Procedure {
public:
    std::string name() const override {
        return "graph_project";
    }

    std::string qualified_name() const override {
        return "graph_project";
    }

    std::string description() const override {
        return "Creates a native graph projection by scanning node labels and edge types directly. "
               "Achieves O(n+m) performance by bypassing pattern matching and scanning B+Tree indexes.";
    }

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

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"graphName", YieldType::STRING,
                "Name of the created projection"},
            YieldField{"nodeCount", YieldType::INT,
                "Total number of nodes in the projection"},
            YieldField{"relationshipCount", YieldType::INT,
                "Total number of relationships in the projection"},
            YieldField{"projectMillis", YieldType::INT,
                "Time taken to create the projection (milliseconds)"},
            YieldField{"featureDim", YieldType::INT,
                "Feature vector dimension (0 if includeFeatures not set)"},
            YieldField{"numClasses", YieldType::INT,
                "Number of distinct classification classes (0 if labelProperty not set)"},
            YieldField{"topologySnapshotBytes", YieldType::INT,
                "Total bytes of CSR topology sidecar files (0 if buildTopologySnapshot not set)"}
        };
    }

    void execute(ProcedureContext& ctx) override;

private:
    // ── Argument parsing ──────────────────────────────────────────────────

    /// Parse graphName from argument 0.
    std::string parse_graph_name(ProcedureContext& ctx);

    /// Parse nodeProjection from argument 1 (STRING, LIST, or MAP).
    NodeProjectionVariant parse_node_projection(ProcedureContext& ctx);

    /// Parse relationshipProjection from argument 2 (STRING or LIST only).
    RelationshipProjectionVariant parse_relationship_projection(ProcedureContext& ctx);

    // ── Validation (non-blocking warnings) ────────────────────────────────

    /// Emit warning if node label does not exist in catalog. Does NOT throw.
    void warn_missing_label(const std::string& label);

    /// Emit warning if edge type does not exist in catalog. Does NOT throw.
    void warn_missing_type(const std::string& type);

    /// Validate projection name is safe for filesystem paths. Throws on invalid.
    static void validate_projection_name(const std::string& name);

    // ── Config dict parsing (nullptr-safe: returns defaults if dict is null) ──

    /// Extract a property list from a config dict by key.
    std::vector<std::string> parse_property_list_from_dict(
        DictionaryObject* dict, const std::string& key);

    /// Determine which property to use for MIN/MAX/SUM aggregation.
    std::string resolve_aggregation_property(
        DictionaryObject* dict,
        const std::vector<std::string>& edge_properties,
        Aggregation aggregation);

    // ── Neo4j GDS map syntax parsing ──────────────────────────────────────

    /// Parse a node projection map from a DICTIONARY ObjectId.
    NodeProjectionMap parse_node_projection_map(ObjectId dict_oid);

    /// Parse a relationship projection map from a DICTIONARY ObjectId.
    RelationshipProjectionMap parse_relationship_projection_map(
        ObjectId dict_oid, Orientation global_orientation, Aggregation global_aggregation);

    /// Parse a single node label config from a nested dictionary.
    NodeProjectionConfig parse_single_node_config(
        ObjectId config_oid, const std::string& projected_label);

    /// Parse a single relationship type config from a nested dictionary.
    RelationshipProjectionConfig parse_single_relationship_config(
        ObjectId config_oid, const std::string& projected_type,
        Orientation global_orientation, Aggregation global_aggregation);

    /// Parse a single property config from a dictionary or string ObjectId.
    PropertyConfig parse_property_config(ObjectId config_oid, const std::string& property_key);

    /// Parse properties value (LIST, STRING, or MAP) into simple_properties and property_configs.
    void parse_properties_value(
        ObjectId properties_oid,
        std::vector<std::string>& simple_properties,
        std::unordered_map<std::string, PropertyConfig>& property_configs);

    /// Extract properties from a "properties" key in a DictionaryObject.
    /// Shared helper used by parse_single_node_config and parse_single_relationship_config.
    void extract_nested_properties(
        DictionaryObject* parent_dict,
        std::vector<std::string>& simple_properties,
        std::unordered_map<std::string, PropertyConfig>& property_configs);

    // ── Dictionary value extraction (nullptr-safe) ────────────────────────

    /// Get a raw ObjectId value from a dictionary by key.
    ObjectId get_value_from_dict(DictionaryObject* dict, const std::string& key, bool& found);

    /// Get a string value from a dictionary, returning default_value if absent.
    std::string get_string_from_dict(
        DictionaryObject* dict, const std::string& key, const std::string& default_value);

    /// Get an orientation value from a dictionary, returning default_value if absent.
    Orientation get_orientation_from_dict(
        DictionaryObject* dict, const std::string& key, Orientation default_value);

    /// Get an aggregation value from a dictionary, returning default_value if absent.
    Aggregation get_aggregation_from_dict(
        DictionaryObject* dict, const std::string& key, Aggregation default_value);

    /// Get an optional double value from a dictionary.
    std::optional<double> get_optional_double_from_dict(
        DictionaryObject* dict, const std::string& key);

    // ── String-to-enum converters ─────────────────────────────────────────

    Orientation parse_orientation_string(const std::string& orientation_str);
    Aggregation parse_aggregation_string(const std::string& aggregation_str);
};

} // namespace Procedures
} // namespace GQL
