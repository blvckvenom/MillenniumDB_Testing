#pragma once

#include <optional>
#include <string>

namespace GQL {

/**
 * @brief Represents a parsed graph reference from GQL syntax.
 *
 * ISO GQL (ISO/IEC 39075:2024) Section 17.2 defines <graph reference> as:
 *   <graph reference> ::=
 *       <catalog object parent reference> <graph name>
 *     | <delimited graph name>
 *     | <home graph>
 *     | <reference parameter specification>
 *
 * This struct captures all components of a qualified graph reference
 * (e.g., "catalog.schema.graph_name") for future extensibility.
 *
 * CURRENT LIMITATION (2025-12):
 * MillenniumDB uses a flat projection namespace (like Neo4j GDS).
 * Only the graph_name field is used for resolution; catalog and schema
 * prefixes are parsed but ignored. This is intentional:
 *
 * - Full catalog support requires Feature GT03 (multi-graph transactions)
 * - Neo4j GDS also uses flat namespace for projections
 * - GNN training workflows don't require multi-catalog support
 *
 * FUTURE EXTENSIBILITY:
 * When catalog/schema support is implemented, the resolution logic
 * can use all fields without changing the parsing code.
 *
 * @see visitGraphReference() in query_visitor.cc
 * @see ISO/IEC 39075:2024 Section 17.2
 */
struct GraphReference {
    /// The catalog name (optional, from qualified reference like "catalog.schema.graph")
    std::optional<std::string> catalog;

    /// The schema name (optional, from qualified reference like "schema.graph")
    std::optional<std::string> schema;

    /// The graph/projection name (required, always present)
    std::string graph_name;

    /// Whether this reference was HOME_GRAPH or HOME_PROPERTY_GRAPH
    bool is_home_graph = false;

    /// Whether this reference was CURRENT_GRAPH
    bool is_current_graph = false;

    /// Original full text of the reference (for error messages and debugging)
    std::string original_text;

    /**
     * @brief Check if this is a qualified reference (has catalog or schema prefix).
     * @return true if catalog or schema prefix was specified
     */
    bool is_qualified() const {
        return catalog.has_value() || schema.has_value();
    }

    /**
     * @brief Get the effective graph name for resolution.
     *
     * Currently returns just graph_name (flat namespace).
     * Future: may return fully qualified name when catalog support is added.
     *
     * @return The graph name to use for projection lookup
     */
    std::string get_effective_name() const {
        return graph_name;
    }

    /**
     * @brief Get the full qualified name as a string.
     *
     * Returns "catalog.schema.graph", "schema.graph", or just "graph"
     * depending on what was parsed.
     *
     * @return Fully qualified name string
     */
    std::string get_qualified_name() const {
        std::string result;
        if (catalog.has_value()) {
            result += catalog.value() + ".";
        }
        if (schema.has_value()) {
            result += schema.value() + ".";
        }
        result += graph_name;
        return result;
    }
};

} // namespace GQL
