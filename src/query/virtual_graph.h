#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "graph_models/object_id.h"

class VirtualGraph {
public:
    struct Edge {
        ObjectId id;
        ObjectId from;
        ObjectId to;
        bool undirected;
    };

    void clear()
    {
        node_ids.clear();
        node_list.clear();
        edge_ids.clear();
        edge_list.clear();
    }

    void add_node(ObjectId node)
    {
        if (node.is_null()) {
            return;
        }
        if (node_ids.insert(node.id).second) {
            node_list.push_back(node);
        }
    }

    void add_edge(ObjectId edge, ObjectId from, ObjectId to, bool undirected)
    {
        if (edge.is_null()) {
            return;
        }
        if (edge_ids.insert(edge.id).second) {
            edge_list.push_back(Edge{ edge, from, to, undirected });
        }
    }

    bool contains_node(ObjectId node) const
    {
        if (node.is_null()) {
            return false;
        }
        return node_ids.find(node.id) != node_ids.end();
    }

    bool contains_edge(ObjectId edge) const
    {
        if (edge.is_null()) {
            return false;
        }
        return edge_ids.find(edge.id) != edge_ids.end();
    }

    const std::vector<ObjectId>& nodes() const
    {
        return node_list;
    }

    const std::vector<Edge>& edges() const
    {
        return edge_list;
    }

private:
    std::unordered_set<uint64_t> node_ids;
    std::vector<ObjectId> node_list;

    std::unordered_set<uint64_t> edge_ids;
    std::vector<Edge> edge_list;
};
