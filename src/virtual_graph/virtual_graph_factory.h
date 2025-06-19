#pragma once

#include <memory>
#include <string>

#include "virtual_graph.h"

class VirtualGraphFactory {
public:
    static std::shared_ptr<VirtualGraph> project(
        const std::string& name,
        const std::string& node_query,
        const std::string& edge_query);

    static std::shared_ptr<VirtualGraph> get(const std::string& name);
};

