#pragma once

#include "graph_models/quad_model/quad_object_id.h"
#include "query/executor/binding_iter.h"
#include "query/var_id.h"
#include "virtual_graph/virtual_graph_factory.h"

class VirtualGraphEdgeScan : public BindingIter {
public:
    VirtualGraphEdgeScan(std::shared_ptr<VirtualGraph> g,
                         VarId from,
                         VarId to,
                         VarId edge,
                         VarId type = VarId(0),
                         bool has_type = false) :
        graph(std::move(g)),
        from_var(from),
        to_var(to),
        edge_var(edge),
        type_var(type),
        type_assigned(has_type)
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
        parent_binding->add(from_var, get_node_oid(e.from));
        parent_binding->add(to_var, get_node_oid(e.to));

        if (!e.var.empty()) {
            parent_binding->add(edge_var, QuadObjectId::get_edge(e.var));
        } else {
            parent_binding->add(edge_var, QuadObjectId::get_edge("_e" + std::to_string(idx)));
        }

        if (type_assigned) {
            if (!e.type.empty()) {
                parent_binding->add(type_var, QuadObjectId::get_string(e.type));
            } else {
                parent_binding->add(type_var, ObjectId::get_null());
            }
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
        if (type_assigned) {
            parent_binding->add(type_var, ObjectId::get_null());
        }
    }

    void accept_visitor(BindingIterVisitor&) override { }

private:
    std::shared_ptr<VirtualGraph> graph;
    VarId from_var;
    VarId to_var;
    VarId edge_var;
    VarId type_var;
    bool type_assigned;
    size_t idx = 0;
    Binding* parent_binding;

    ObjectId get_node_oid(const std::string& id)
    {
        auto it = graph->node_index.find(id);
        if (it != graph->node_index.end()) {
            const auto& n = graph->nodes[it->second];
            if (n.is_literal) {
                switch (n.lit_type) {
                case VirtualGraph::Node::LitType::STRING:
                    return QuadObjectId::get_string(n.lit_string);
                case VirtualGraph::Node::LitType::INT: {
                    uint64_t oid = ObjectId::MASK_POSITIVE_INT;
                    int64_t v = n.lit_int;
                    if (v < 0) {
                        oid = ObjectId::MASK_NEGATIVE_INT;
                        v = -v;
                        oid |= (~static_cast<uint64_t>(v)) & ObjectId::VALUE_MASK;
                        return ObjectId(oid);
                    }
                    return ObjectId(oid | static_cast<uint64_t>(v));
                }
                case VirtualGraph::Node::LitType::DOUBLE:
                    return QuadObjectId::get_string(std::to_string(n.lit_double));
                case VirtualGraph::Node::LitType::BOOL:
                    return ObjectId(ObjectId::MASK_BOOL | (n.lit_bool ? 1ULL : 0ULL));
                case VirtualGraph::Node::LitType::DATE:
                    return QuadObjectId::get_string(n.lit_string);
                default:
                    return QuadObjectId::get_string(n.lit_string);
                }
            }
        }
        return QuadObjectId::get_fixed_node_inside(id);
    }
};
