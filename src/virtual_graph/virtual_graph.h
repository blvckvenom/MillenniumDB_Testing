#pragma once

#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

struct VirtualGraph {
    struct Node {
        std::string id;
        std::unordered_map<std::string, std::string> properties;
    };

    struct Edge {
        std::string id;
        std::string var;
        std::string from;
        std::string to;
        std::string type;
        std::unordered_map<std::string, std::string> properties;
    };

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::unordered_map<std::string, std::size_t> node_index;
};

inline std::ostream& operator<<(std::ostream& os, const VirtualGraph::Edge& e)
{
    if (!e.type.empty()) {
        os << e.type;
    } else if (!e.var.empty()) {
        os << e.var;
    } else {
        os << e.id;
    }
    return os;
}
