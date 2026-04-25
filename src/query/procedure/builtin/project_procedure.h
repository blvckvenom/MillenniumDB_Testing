#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

class DictionaryObject;

// Opaque declarations for the storage-format enums held by
// ProjectProcedure::ParsedConfig. Their definitions live in
// storage/index/bplus_tree/bpt_leaf_format.h and
// graph_models/gql/projection/index_set.h; the latter transitively includes
// native_projection_builder.h, which includes this header back, so a full
// include here would be circular.
namespace BPT {
enum class LeafFormat : uint8_t;
enum class GraphStorage : uint8_t;
} // namespace BPT

namespace GQL {
enum class IndexSet : uint8_t;
} // namespace GQL

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
 * - **Undirected source**: No-op. ISO/IEC 39075:2024 §3.4.13 defines the
 *   undirected edge and notes it "expresses a relationship that is
 *   necessarily symmetric", so both directions are already implied.
 *
 * This differs from Neo4j GDS, which always doubles (Neo4j has no native undirected edges).
 * MillenniumDB's approach is semantically correct: undirected edges are already symmetric.
 *
 * Storage cost: NATURAL and REVERSE store one record per source edge.
 * UNDIRECTED on a directed source stores each edge in both directions, so
 * the edge indexes roughly double while node and property data are
 * unchanged; on an undirected source it stores nothing extra.
 */
enum class Orientation {
    NATURAL,     ///< Default: edges as specified (from → to)
    REVERSE,     ///< Reversed: edges reversed (to → from)
    UNDIRECTED   ///< Undirected: bidirectional for directed edges, no-op for undirected
};

/**
 * @brief Aggregation strategy for parallel edges (multigraph support).
 *
 * Two edges are "parallel" when they share the same (from, to, type) triple
 * after orientation is applied. The strategy decides what the projection
 * stores for each such group:
 * - SINGLE (default): refuse — parallel edges raise a QueryException that
 *   asks the caller to pick an explicit collapse strategy, instead of
 *   dropping data silently. Detection is windowed, not exact: the detector's
 *   seen-edge set is cleared periodically to bound memory, so parallels far
 *   apart in scan order can escape detection and both be stored.
 * - MIN / MAX: keep the edge whose aggregation-property value is smallest /
 *   largest; that edge's id becomes the group's representative. Edges
 *   missing the property do not compete.
 * - SUM: keep the first-seen edge, with the aggregation property replaced
 *   by the sum over the whole group.
 * - COUNT: keep the first-seen edge and attach the group size as a
 *   synthetic `_count` property; no aggregation property is required.
 *
 * The value names follow Neo4j Graph Data Science: its Graph Data Science
 * manual page "Native projection" lists the allowed aggregation values
 * (NONE default, SINGLE, COUNT, MIN, MAX, SUM) under "Relationship
 * projection", but does not define what SINGLE does
 * (https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/).
 * The behaviors above — SINGLE as strict fail-on-parallel and as our
 * default, and NONE accepted as an alias for SINGLE — are this
 * implementation's own definitions.
 *
 * Cost: aggregation state is bounded by construction (the duplicate
 * detector holds one bounded batch of group keys; the streaming aggregator
 * holds a single group at a time), so memory use is independent of graph
 * size. The per-edge cost is one hash-table lookup on the group key.
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
 * Creates disk-based graph projections by scanning the label_node and
 * label_edge B+Tree indexes directly — one pass over the projected nodes
 * and one over the projected edges, with no pattern-matching executor
 * involved.
 *
 * Syntax:
 * @code{.gql}
 *   CALL graph_project(graphName, nodeProjection, relationshipProjection [, config])
 *   YIELD graphName, nodeCount, relationshipCount, projectMillis
 * @endcode
 *
 * Examples:
 * @code{.gql}
 *   CALL graph_project('myGraph', 'User', 'KNOWS')
 *   CALL graph_project('social', ['User', 'Post'], ['KNOWS', 'LIKES'])
 * @endcode
 *
 * @see ISO/IEC 39075:2024, Clause 15 "Procedure calling" (15.1 <call
 *      procedure statement>, 15.3 <named procedure call>) for CALL/YIELD
 *      semantics
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
                "Optional configuration map: orientation, aggregation, "
                "aggregationProperty, nodeProperties, relationshipProperties, "
                "indexSet, leafFormat, graphStorage, includeLabelIndexes, "
                "buildTopologySnapshot, includeFeatures, labelProperty, "
                "splitProperty")
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
    // ── Config-map parsing ────────────────────────────────────────────────

    /// Every option parsed from the optional 4th `configuration` map argument
    /// of graph_project(). Produced exclusively by parse_config_options(),
    /// which assigns ALL fields (documented defaults included) — hence no
    /// member initializers: the three enum types are only opaque-declared at
    /// this point in the header.
    struct ParsedConfig {
        std::vector<std::string> global_node_properties;  ///< `nodeProperties`
        std::vector<std::string> global_edge_properties;  ///< `relationshipProperties`
        Orientation global_orientation;                   ///< `orientation` (default NATURAL)
        Aggregation global_aggregation;                   ///< `aggregation` (default SINGLE)
        std::string global_aggregation_property;          ///< resolved `aggregationProperty`
        std::string include_features;                     ///< GNN: `includeFeatures` ("" = disabled)
        std::string label_property;                       ///< GNN: `labelProperty` ("" = disabled)
        std::string split_property;                       ///< GNN: `splitProperty` ("" = disabled)
        bool include_label_indexes;                       ///< `includeLabelIndexes` (default true)
        GQL::IndexSet index_set;                          ///< `indexSet` (default ALL)
        BPT::LeafFormat leaf_format;                      ///< `leafFormat` (default BITSET)
        BPT::GraphStorage graph_storage;                  ///< `graphStorage` (default BTREE)
        bool build_topology_snapshot;                     ///< `buildTopologySnapshot` (default false)
    };

    /// Parse the optional configuration map (argument index 3) into a
    /// ParsedConfig. A missing or non-map 4th argument yields the documented
    /// defaults; invalid values throw. May print notices to stderr (the
    /// CSR_HYBRID sidecar supersedence and the GNN-intent auto-defaults).
    ParsedConfig parse_config_options(ProcedureContext& ctx);

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
