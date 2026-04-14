#pragma once

#include "graph_models/gql/gql_catalog.h"
#include "graph_models/model_destroyer.h"

template<std::size_t N>
class BPlusTree;

// This is the worst model, only for completeness.
// The idea is in the future to either have a model with a single label per edge
// or another redundant structure with labels inside the edge
//( like {label, from, to, edge} and {label, to, from, edge})
class GQLModel {
public:
    // TODO: deleted_nodes?

    std::unique_ptr<BPlusTree<2>> node_label;
    std::unique_ptr<BPlusTree<2>> label_node;

    std::unique_ptr<BPlusTree<2>> edge_label;
    std::unique_ptr<BPlusTree<2>> label_edge;

    std::unique_ptr<BPlusTree<3>> node_key_value;
    std::unique_ptr<BPlusTree<3>> key_value_node;

    std::unique_ptr<BPlusTree<3>> edge_key_value;
    std::unique_ptr<BPlusTree<3>> key_value_edge;

    std::unique_ptr<BPlusTree<3>> from_to_edge;
    std::unique_ptr<BPlusTree<3>> to_from_edge;
    std::unique_ptr<BPlusTree<3>> edge_from_to;

    std::unique_ptr<BPlusTree<3>> n1_n2_edge;
    std::unique_ptr<BPlusTree<3>> edge_n1_n2;

    // {node, edge}
    std::unique_ptr<BPlusTree<2>> equal_u_edge;
    std::unique_ptr<BPlusTree<2>> equal_d_edge;

    GQLCatalog catalog;

    // necessary to be called before first usage
    static std::unique_ptr<ModelDestroyer> init(const std::string& db_folder = "");

    // Dynamic index selection helpers for USE GRAPH projection support
    // These methods check if a projection is active and return the appropriate index

    // Edge connectivity indexes (exist in both main graph and projections)
    BPlusTree<3>& get_from_to_edge();
    BPlusTree<3>& get_to_from_edge();

    // Edge indexes with different orderings (NOT in projections)
    BPlusTree<3>& get_edge_from_to();
    BPlusTree<3>& get_n1_n2_edge();
    BPlusTree<3>& get_edge_n1_n2();

    // Self-loop indexes (NOT in projections)
    BPlusTree<2>& get_equal_d_edge();
    BPlusTree<2>& get_equal_u_edge();

    // Label indexes (optional in projections if INCLUDE LABELS was used)
    BPlusTree<2>& get_node_label();  // {node_id, label_id}
    BPlusTree<2>& get_label_node();  // {label_id, node_id}
    BPlusTree<2>& get_edge_label();  // {edge_id, label_id}
    BPlusTree<2>& get_label_edge();  // {label_id, edge_id}

    // Property indexes (optional in projections if INCLUDE PROPERTIES was used)
    BPlusTree<3>& get_node_key_value();  // {node_id, key_id, value_id}
    BPlusTree<3>& get_key_value_node();  // {key_id, value_id, node_id}
    BPlusTree<3>& get_edge_key_value();  // {edge_id, key_id, value_id}
    BPlusTree<3>& get_key_value_edge();  // {key_id, value_id, edge_id}

private:
    GQLModel();
};

extern GQLModel& gql_model; // global object
