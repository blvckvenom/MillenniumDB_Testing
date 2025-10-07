#pragma once

#include <map>
#include <memory>
#include <vector>

#include "query/executor/binding_iter.h"
#include "query/parser/op/gql/op_project.h"
#include "query/virtual_graph.h"

class ProjectGraph : public BindingIter {
public:
    ProjectGraph(
        VarId alias,
        std::unique_ptr<BindingIter> subquery_iter,
        std::vector<VarId> projected_items,
        std::map<VarId, GQL::VarType> projected_types
    );

    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;
    void assign_nulls() override;
    void print(std::ostream& os, int indent, bool stats) const override;

private:
    void materialize(Binding& parent_binding);

    std::pair<ObjectId, ObjectId> resolve_directed_endpoints(ObjectId edge) const;
    std::pair<ObjectId, ObjectId> resolve_undirected_endpoints(ObjectId edge) const;

    VarId alias;
    std::unique_ptr<BindingIter> child;
    std::vector<VarId> projected_items;
    std::map<VarId, GQL::VarType> projected_types;

    std::shared_ptr<VirtualGraph> graph;

    bool materialized = false;
    bool produced = false;
    Binding* parent_binding = nullptr;
};
