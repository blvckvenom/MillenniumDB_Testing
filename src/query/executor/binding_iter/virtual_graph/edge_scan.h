#pragma once

#include "graph_models/quad_model/quad_object_id.h"
#include "query/executor/binding_iter.h"
#include "query/var_id.h"
#include "virtual_graph/virtual_graph_factory.h"

class VirtualGraphEdgeScan : public BindingIter {
public:
    VirtualGraphEdgeScan(std::shared_ptr<VirtualGraph> g, VarId from, VarId to, VarId edge) :
        graph(std::move(g)),
        from_var(from),
        to_var(to),
        edge_var(edge)
    { }

    void _begin(Binding& parent) override
    {
        parent_binding = &parent;
        idx = 0;
    }

    bool _next() override
    {
        if (!graph || idx >= graph->edges.size())
            return false;
        const auto& e = graph->edges[idx];
        parent_binding->add(from_var, QuadObjectId::get_fixed_node_inside(e.from));
        parent_binding->add(to_var, QuadObjectId::get_fixed_node_inside(e.to));
        if (!e.type.empty()) {
            parent_binding->add(edge_var, QuadObjectId::get_edge(e.type));
        } else {
            parent_binding->add(edge_var, QuadObjectId::get_edge("_e" + std::to_string(idx)));
        }
        idx++;
        return true;
    }

    void _reset() override
    {
        idx = 0;
    }

    void assign_nulls() override
    {
        parent_binding->add(from_var, ObjectId::get_null());
        parent_binding->add(to_var, ObjectId::get_null());
        parent_binding->add(edge_var, ObjectId::get_null());
    }

    void accept_visitor(BindingIterVisitor&) override;

private:
    std::shared_ptr<VirtualGraph> graph;
    VarId from_var;
    VarId to_var;
    VarId edge_var;
    size_t idx = 0;
    Binding* parent_binding;
};
