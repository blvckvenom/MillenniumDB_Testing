#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct VirtualGraph {
    struct Node {
        std::string id;
        std::unordered_map<std::string, std::string> properties;
    };

    struct Edge {
        std::string from;
        std::string to;
        std::unordered_map<std::string, std::string> properties;
    };

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::unordered_map<std::string, std::size_t> node_index;
};

