#include "project_graph.h"

#include <limits>
#include <utility>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/gql_object_id.h"
#include "query/exceptions.h"
#include "query/query_context.h"
#include "storage/index/record.h"

ProjectGraph::ProjectGraph(
    VarId alias,
    std::unique_ptr<BindingIter> subquery_iter,
    std::vector<VarId> projected_items,
    std::map<VarId, GQL::VarType> projected_types
) :
    alias(alias),
    child(std::move(subquery_iter)),
    projected_items(std::move(projected_items)),
    projected_types(std::move(projected_types))
{
}

void ProjectGraph::_begin(Binding& parent)
{
    parent_binding = &parent;
    produced = false;

    if (!materialized) {
        materialize(parent);
    }
}

bool ProjectGraph::_next()
{
    if (produced) {
        return false;
    }
    produced = true;
    return true;
}

void ProjectGraph::_reset()
{
    produced = false;
}

void ProjectGraph::assign_nulls()
{
    produced = true;
}

void ProjectGraph::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "ProjectGraph(alias=" << alias
       << ", nodes=" << (graph ? graph->nodes().size() : 0)
       << ", edges=" << (graph ? graph->edges().size() : 0) << ")\n";
    if (child) {
        child->print(os, indent + 2, stats);
    }
}

void ProjectGraph::materialize(Binding& parent)
{
    graph = std::make_shared<VirtualGraph>();

    if (!child) {
        get_query_ctx().register_virtual_graph(alias, graph);
        materialized = true;
        return;
    }

    Binding subquery_binding(parent.size);
    subquery_binding.add_all(parent);

    child->begin(subquery_binding);

    while (child->next()) {
        for (auto var : projected_items) {
            auto value = subquery_binding[var];
            auto type_it = projected_types.find(var);

            GQL::VarType::Type var_type;
            if (type_it != projected_types.end()) {
                var_type = type_it->second.type;
            } else {
                switch (GQL_OID::get_type(value)) {
                case GQL_OID::Type::NODE:
                    var_type = GQL::VarType::Node;
                    break;
                case GQL_OID::Type::DIRECTED_EDGE:
                case GQL_OID::Type::UNDIRECTED_EDGE:
                    var_type = GQL::VarType::Edge;
                    break;
                default:
                    continue;
                }
            }

            if (var_type == GQL::VarType::Node) {
                graph->add_node(value);
            } else if (var_type == GQL::VarType::Edge) {
                auto gql_type = GQL_OID::get_type(value);
                auto endpoints = gql_type == GQL_OID::Type::UNDIRECTED_EDGE
                    ? resolve_undirected_endpoints(value)
                    : resolve_directed_endpoints(value);

                graph->add_edge(value, endpoints.first, endpoints.second, gql_type == GQL_OID::Type::UNDIRECTED_EDGE);
                graph->add_node(endpoints.first);
                graph->add_node(endpoints.second);
            }
        }
    }

    get_query_ctx().register_virtual_graph(alias, graph);
    materialized = true;
}

std::pair<ObjectId, ObjectId> ProjectGraph::resolve_directed_endpoints(ObjectId edge) const
{
    Record<3> min{ edge.id, 0, 0 };
    Record<3> max{ edge.id, std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() };

    auto iter = gql_model.edge_from_to->get_range(&get_query_ctx().thread_info.interruption_requested, min, max);
    const auto* record = iter.next();
    if (record == nullptr) {
        throw QueryException("Failed to resolve endpoints for directed edge in PROJECT");
    }
    return { ObjectId((*record)[1]), ObjectId((*record)[2]) };
}

std::pair<ObjectId, ObjectId> ProjectGraph::resolve_undirected_endpoints(ObjectId edge) const
{
    Record<3> min{ edge.id, 0, 0 };
    Record<3> max{ edge.id, std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() };

    auto iter = gql_model.edge_n1_n2->get_range(&get_query_ctx().thread_info.interruption_requested, min, max);
    const auto* record = iter.next();
    if (record == nullptr) {
        throw QueryException("Failed to resolve endpoints for undirected edge in PROJECT");
    }
    return { ObjectId((*record)[1]), ObjectId((*record)[2]) };
}
