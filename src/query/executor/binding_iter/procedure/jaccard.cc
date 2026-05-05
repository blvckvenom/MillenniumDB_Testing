#include "jaccard.h"

#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <set>

#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "query/exceptions.h"

using namespace Procedure;

Jaccard::Jaccard(
    std::vector<std::unique_ptr<BindingExpr>>&& argument_binding_exprs_,
    std::vector<VarId>&& yield_vars_
) :
    argument_binding_exprs { std::move(argument_binding_exprs_) },
    yield_vars { std::move(yield_vars_) }
{
    assert(argument_binding_exprs.size() <= 1);
    assert(yield_vars.size() == 3);
}

void Jaccard::_begin(Binding& parent_binding_)
{
    parent_binding = &parent_binding_;
    _reset();
}

void Jaccard::_reset()
{
    results.clear();
    cursor = 0;
    eval_arguments();

    std::array<uint64_t, 3> min_ids = { 0, 0, 0 };
    std::array<uint64_t, 3> max_ids = { UINT64_MAX, UINT64_MAX, UINT64_MAX };

    std::map<uint64_t, std::set<uint64_t>> adjacency;

    auto undirected_edge_iter = gql_model.get_n1_n2_edge().get_range(
        &get_query_ctx().thread_info.interruption_requested,
        min_ids,
        max_ids
    );
    while (const auto* current_record = undirected_edge_iter.next()) {
        const auto node1 = (*current_record)[0];
        const auto node2 = (*current_record)[1];

        // Undirected edge contributes both ways: node1~node2 => neighbors(node1)+=node2 and neighbors(node2)+=node1
        adjacency[node1].insert(node2);
        adjacency[node2].insert(node1);
    }

    auto directed_edge_iter = gql_model.get_from_to_edge().get_range(
        &get_query_ctx().thread_info.interruption_requested,
        min_ids,
        max_ids
    );
    while (const auto* current_record = directed_edge_iter.next()) {
        const auto from = (*current_record)[0];
        const auto to   = (*current_record)[1];

        // Directed edge contributes only in outgoing direction: from->to => neighbors(from)+=to
        adjacency[from].insert(to);
    }
    // Note: Self-loop indexes (equal_u_edge/equal_d_edge) are intentionally not scanned here yet.
    // They are not projection-aware, and calling their getters under USE projection throws.

    if (adjacency.size() < 2) {
        return;
    }

    std::vector<uint64_t> nodes;
    nodes.reserve(adjacency.size());
    for (const auto& [node, _] : adjacency) {
        nodes.push_back(node);
    }

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& neighbors_i = adjacency.at(nodes[i]);
        for (std::size_t j = i + 1; j < nodes.size(); ++j) {
            const auto& neighbors_j = adjacency.at(nodes[j]);

            const auto& smaller = (neighbors_i.size() <= neighbors_j.size()) ? neighbors_i : neighbors_j;
            const auto& larger = (neighbors_i.size() <= neighbors_j.size()) ? neighbors_j : neighbors_i;

            std::size_t intersection_size = 0;
            for (const auto& neighbor : smaller) {
                if (larger.count(neighbor) == 1) {
                    ++intersection_size;
                }
            }

            const auto union_size = neighbors_i.size() + neighbors_j.size() - intersection_size;
            const auto similarity = (union_size == 0)
                                        ? 0.0
                                        : static_cast<double>(intersection_size) / static_cast<double>(union_size);

            if (similarity >= similarity_cutoff) {
                results.emplace_back(
                    ObjectId(nodes[i]),
                    ObjectId(nodes[j]),
                    GQL::Conversions::pack_double(similarity)
                );
            }
        }
    }
}

void Jaccard::eval_arguments()
{
    similarity_cutoff = 0.0;
    if (argument_binding_exprs.empty()) {
        return;
    }

    const ObjectId cutoff_oid = argument_binding_exprs[0]->eval(*parent_binding);
    switch (cutoff_oid.get_sub_type()) {
    case ObjectId::MASK_INT:
    case ObjectId::MASK_DECIMAL:
    case ObjectId::MASK_FLOAT:
    case ObjectId::MASK_DOUBLE:
        similarity_cutoff = GQL::Conversions::to_double(cutoff_oid);
        break;
    default:
        throw QueryExecutionException("CALL jaccard(similarityCutoff): similarityCutoff must be numeric");
    }

    if (!std::isfinite(similarity_cutoff) || similarity_cutoff < 0.0 || similarity_cutoff > 1.0) {
        throw QueryExecutionException("CALL jaccard(similarityCutoff): similarityCutoff must be in range [0, 1]");
    }
}

bool Jaccard::_next()
{
    if (cursor >= results.size()) {
        return false;
    }

    const auto& [node1, node2, similarity] = results[cursor];
    ++cursor;

    parent_binding->add(yield_vars[0], node1);
    parent_binding->add(yield_vars[1], node2);
    parent_binding->add(yield_vars[2], similarity);
    return true;
}

void Jaccard::assign_nulls()
{
    for (const auto& var : yield_vars) {
        parent_binding->add(var, ObjectId::get_null());
    }
}

void Jaccard::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "Jaccard() -> (";
    if (!yield_vars.empty()) {
        os << yield_vars[0];
        for (std::size_t i = 1; i < yield_vars.size(); ++i) {
            os << ", " << yield_vars[i];
        }
    }
    os << ")\n";
}
