#pragma once

#include <memory>
#include <vector>

#include "query/executor/binding_iter.h"
#include "query/virtual_graph.h"

class VirtualGraphFilter : public BindingIter {
public:
    VirtualGraphFilter(
        std::unique_ptr<BindingIter> child,
        std::shared_ptr<VirtualGraph> graph,
        std::vector<VarId> node_vars,
        std::vector<VarId> edge_vars
    );

    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;
    void assign_nulls() override;
    void print(std::ostream& os, int indent, bool stats) const override;

private:
    std::unique_ptr<BindingIter> child;
    std::shared_ptr<VirtualGraph> graph;
    std::vector<VarId> node_vars;
    std::vector<VarId> edge_vars;
    Binding* parent_binding = nullptr;
};
