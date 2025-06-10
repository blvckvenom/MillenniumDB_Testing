#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct PropertyGraphNode {
    std::string id;
    std::vector<std::string> labels;
    std::unordered_map<std::string, std::string> properties;
};

struct PropertyGraphEdge {
    std::string from;
    std::string to;
    std::string type;
    std::unordered_map<std::string, std::string> properties;
};

struct PropertyGraph {
    std::unordered_map<std::string, PropertyGraphNode> nodes;
    std::vector<PropertyGraphEdge> edges;
};

extern std::unordered_map<std::string, PropertyGraph> projectedGraphs;

std::string project(const std::string& name, const std::string& filePath);
