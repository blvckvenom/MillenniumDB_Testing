#pragma once

#include "query/executor/binding_iter.h"
#include "virtual_graph/virtual_graph_factory.h"
#include "graph_models/quad_model/quad_object_id.h"
#include "query/var_id.h"

class VirtualGraphNodeScan : public BindingIter {
public:
    VirtualGraphNodeScan(std::shared_ptr<VirtualGraph> g, VarId var)
        : graph(std::move(g)), var(var) {}

    void _begin(Binding& parent) override {
        parent_binding = &parent;
        idx = 0;
    }

    bool _next() override {
        if (!graph || idx >= graph->nodes.size())
            return false;
        const auto& node = graph->nodes[idx++];
        parent_binding->add(var, QuadObjectId::get_fixed_node_inside(node.id));
        return true;
    }

    void _reset() override { idx = 0; }

    void assign_nulls() override {
        parent_binding->add(var, ObjectId::get_null());
    }

    void accept_visitor(BindingIterVisitor&) override;

private:
    std::shared_ptr<VirtualGraph> graph;
    VarId var;
    size_t idx = 0;
    Binding* parent_binding;
};

