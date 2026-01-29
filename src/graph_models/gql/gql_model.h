#pragma once

#include "graph_models/gql/gql_catalog.h"
#include "graph_models/model_destroyer.h"

template<std::size_t N>
class BPlusTree;

/**
 * @brief Core storage model for GQL (Graph Query Language) databases.
 *
 * Implements a disk-based property graph model with persistent B+Tree indexes.
 * Supports the ISO GQL standard (ISO/IEC 39075:2024) including:
 * - Nodes with multiple labels
 * - Directed and undirected edges with multiple labels
 * - Properties on nodes and edges
 *
 * ## Index Architecture
 *
 * The model maintains multiple B+Tree indexes for efficient graph traversal:
 *
 * ### Label Indexes (2-key B+Trees)
 * | Index | Keys | Purpose |
 * |-------|------|---------|
 * | node_label | (node_id, label_id) | Find labels of a node |
 * | label_node | (label_id, node_id) | Find nodes with label |
 * | edge_label | (edge_id, label_id) | Find labels of an edge |
 * | label_edge | (label_id, edge_id) | Find edges with label |
 *
 * ### Property Indexes (3-key B+Trees)
 * | Index | Keys | Purpose |
 * |-------|------|---------|
 * | node_key_value | (node_id, key_id, value) | Get node properties |
 * | key_value_node | (key_id, value, node_id) | Property-based lookup |
 * | edge_key_value | (edge_id, key_id, value) | Get edge properties |
 * | key_value_edge | (key_id, value, edge_id) | Property-based lookup |
 *
 * ### Connectivity Indexes (3-key B+Trees)
 * | Index | Keys | Purpose |
 * |-------|------|---------|
 * | from_to_edge | (from_id, to_id, edge_id) | Outgoing edge traversal |
 * | to_from_edge | (to_id, from_id, edge_id) | Incoming edge traversal |
 * | edge_from_to | (edge_id, from_id, to_id) | Edge endpoint lookup |
 * | n1_n2_edge | (node1, node2, edge_id) | Undirected traversal |
 * | edge_n1_n2 | (edge_id, node1, node2) | Undirected endpoint lookup |
 *
 * ## Usage with Projections
 *
 * Methods like get_from_to_edge() automatically check if a graph projection
 * is active (via USE GRAPH) and return the appropriate index.
 *
 * @note This is a singleton - use the global `gql_model` reference
 * @see GQLCatalog for metadata and statistics
 * @see docs/native_projection_review/ for projection architecture
 */
class GQLModel {
public:
    /// @name Label Indexes
    /// @{
    std::unique_ptr<BPlusTree<2>> node_label;   ///< (node_id, label_id) - labels for each node
    std::unique_ptr<BPlusTree<2>> label_node;   ///< (label_id, node_id) - nodes with each label
    std::unique_ptr<BPlusTree<2>> edge_label;   ///< (edge_id, label_id) - labels for each edge
    std::unique_ptr<BPlusTree<2>> label_edge;   ///< (label_id, edge_id) - edges with each label
    /// @}

    /// @name Property Indexes
    /// @{
    std::unique_ptr<BPlusTree<3>> node_key_value;  ///< (node_id, key_id, value) - node properties
    std::unique_ptr<BPlusTree<3>> key_value_node;  ///< (key_id, value, node_id) - property lookup
    std::unique_ptr<BPlusTree<3>> edge_key_value;  ///< (edge_id, key_id, value) - edge properties
    std::unique_ptr<BPlusTree<3>> key_value_edge;  ///< (key_id, value, edge_id) - property lookup
    /// @}

    /// @name Directed Edge Connectivity Indexes
    /// @{
    std::unique_ptr<BPlusTree<3>> from_to_edge;  ///< (from_id, to_id, edge_id) - outgoing traversal
    std::unique_ptr<BPlusTree<3>> to_from_edge;  ///< (to_id, from_id, edge_id) - incoming traversal
    std::unique_ptr<BPlusTree<3>> edge_from_to;  ///< (edge_id, from_id, to_id) - endpoint lookup
    /// @}

    /// @name Undirected Edge Connectivity Indexes
    /// @{
    std::unique_ptr<BPlusTree<3>> n1_n2_edge;   ///< (node1, node2, edge_id) - undirected traversal
    std::unique_ptr<BPlusTree<3>> edge_n1_n2;   ///< (edge_id, node1, node2) - undirected lookup
    /// @}

    /// @name Self-loop Indexes
    /// @{
    std::unique_ptr<BPlusTree<2>> equal_u_edge;  ///< (node_id, edge_id) - undirected self-loops
    std::unique_ptr<BPlusTree<2>> equal_d_edge;  ///< (node_id, edge_id) - directed self-loops
    /// @}

    GQLCatalog catalog;  ///< Database metadata and statistics

    /**
     * @brief Initializes the GQL model and opens/creates the database.
     *
     * Must be called before any database operations. Opens existing database
     * or creates a new one at the specified folder.
     *
     * @param db_folder Path to database folder (empty for in-memory)
     * @return ModelDestroyer that cleans up on destruction
     */
    static std::unique_ptr<ModelDestroyer> init(const std::string& db_folder = "");

    /// @name Dynamic Index Selection (Projection-Aware)
    /// @brief These methods check if a graph projection is active and return the appropriate index.
    /// @{

    /// @brief Gets outgoing edge index (projection-aware)
    BPlusTree<3>& get_from_to_edge();
    /// @brief Gets incoming edge index (projection-aware)
    BPlusTree<3>& get_to_from_edge();

    /// @brief Gets edge endpoint lookup index (main graph only)
    BPlusTree<3>& get_edge_from_to();
    /// @brief Gets undirected traversal index (main graph only)
    BPlusTree<3>& get_n1_n2_edge();
    /// @brief Gets undirected endpoint lookup index (main graph only)
    BPlusTree<3>& get_edge_n1_n2();

    /// @brief Gets directed self-loop index (main graph only)
    BPlusTree<2>& get_equal_d_edge();
    /// @brief Gets undirected self-loop index (main graph only)
    BPlusTree<2>& get_equal_u_edge();

    /// @brief Gets node-to-labels index (projection-aware if labels included)
    BPlusTree<2>& get_node_label();
    /// @brief Gets label-to-nodes index (projection-aware if labels included)
    BPlusTree<2>& get_label_node();
    /// @brief Gets edge-to-labels index (projection-aware if labels included)
    BPlusTree<2>& get_edge_label();
    /// @brief Gets label-to-edges index (projection-aware if labels included)
    BPlusTree<2>& get_label_edge();

    /// @brief Gets node property index (projection-aware if properties included)
    BPlusTree<3>& get_node_key_value();
    /// @brief Gets property-to-nodes index (projection-aware if properties included)
    BPlusTree<3>& get_key_value_node();
    /// @brief Gets edge property index (projection-aware if properties included)
    BPlusTree<3>& get_edge_key_value();
    /// @brief Gets property-to-edges index (projection-aware if properties included)
    BPlusTree<3>& get_key_value_edge();
    /// @}

private:
    GQLModel();
};

/// Global GQL model instance (singleton pattern)
extern GQLModel& gql_model;
