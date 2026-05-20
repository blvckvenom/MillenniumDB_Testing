#include "jaccard.h"

#include <algorithm>
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
    assert(argument_binding_exprs.size() <= 5);
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

    std::vector<uint64_t> nodes;
    nodes.reserve(adjacency.size());
    for (const auto& [node, neighbors] : adjacency) {
        const auto degree = static_cast<uint64_t>(neighbors.size());
        if (degree >= degree_cutoff && degree <= upper_degree_cutoff) {
            nodes.push_back(node);
        }
    }

    if (nodes.size() < 2) {
        return;
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

    if (top_n.has_value()) {
        std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
            const auto& [lhs_node1, lhs_node2, lhs_similarity_oid] = lhs;
            const auto& [rhs_node1, rhs_node2, rhs_similarity_oid] = rhs;

            const double lhs_similarity = GQL::Conversions::to_double(lhs_similarity_oid);
            const double rhs_similarity = GQL::Conversions::to_double(rhs_similarity_oid);

            if (lhs_similarity != rhs_similarity) {
                return lhs_similarity > rhs_similarity;
            }
            if (lhs_node1.id != rhs_node1.id) {
                return lhs_node1.id < rhs_node1.id;
            }
            return lhs_node2.id < rhs_node2.id;
        });

        if (*top_n < results.size()) {
            results.resize(static_cast<std::size_t>(*top_n));
        }
    } else if (bottom_n.has_value()) {
        std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
            const auto& [lhs_node1, lhs_node2, lhs_similarity_oid] = lhs;
            const auto& [rhs_node1, rhs_node2, rhs_similarity_oid] = rhs;

            const double lhs_similarity = GQL::Conversions::to_double(lhs_similarity_oid);
            const double rhs_similarity = GQL::Conversions::to_double(rhs_similarity_oid);

            if (lhs_similarity != rhs_similarity) {
                return lhs_similarity < rhs_similarity;
            }
            if (lhs_node1.id != rhs_node1.id) {
                return lhs_node1.id < rhs_node1.id;
            }
            return lhs_node2.id < rhs_node2.id;
        });

        if (*bottom_n < results.size()) {
            results.resize(static_cast<std::size_t>(*bottom_n));
        }
    }
}

void Jaccard::eval_arguments()
{
    similarity_cutoff = 0.0;
    degree_cutoff = 1;
    upper_degree_cutoff = UINT64_MAX;
    top_n.reset();
    bottom_n.reset();

    auto eval_numeric = [&](std::size_t arg_pos, const char* arg_name) -> ObjectId {
        const ObjectId oid = argument_binding_exprs[arg_pos]->eval(*parent_binding);
        switch (oid.get_sub_type()) {
        case ObjectId::MASK_INT:
        case ObjectId::MASK_DECIMAL:
        case ObjectId::MASK_FLOAT:
        case ObjectId::MASK_DOUBLE:
            return oid;
        default:
            throw QueryExecutionException(
                std::string("CALL jaccard(...): ") + arg_name + " must be numeric"
            );
        }
    };

    auto eval_non_negative_integer = [&](std::size_t arg_pos, const char* arg_name) -> uint64_t {
        const ObjectId oid = argument_binding_exprs[arg_pos]->eval(*parent_binding);
        if (oid.get_sub_type() != ObjectId::MASK_INT) {
            throw QueryExecutionException(
                std::string("CALL jaccard(...): ") + arg_name + " must be an integer >= 0"
            );
        }

        const int64_t value = GQL::Conversions::to_integer(oid);
        if (value < 0) {
            throw QueryExecutionException(
                std::string("CALL jaccard(...): ") + arg_name + " must be an integer >= 0"
            );
        }
        return static_cast<uint64_t>(value);
    };

    if (argument_binding_exprs.size() >= 1) {
        const ObjectId cutoff_oid = eval_numeric(0, "similarityCutoff");
        similarity_cutoff = GQL::Conversions::to_double(cutoff_oid);
    }

    if (argument_binding_exprs.size() >= 2) {
        degree_cutoff = eval_non_negative_integer(1, "degreeCutoff");
    }

    if (argument_binding_exprs.size() >= 3) {
        upper_degree_cutoff = eval_non_negative_integer(2, "upperDegreeCutoff");
    }

    if (argument_binding_exprs.size() >= 4) {
        top_n = eval_non_negative_integer(3, "topN");
    }

    if (argument_binding_exprs.size() >= 5) {
        bottom_n = eval_non_negative_integer(4, "bottomN");
    }

    if (!std::isfinite(similarity_cutoff) || similarity_cutoff < 0.0 || similarity_cutoff > 1.0) {
        throw QueryExecutionException("CALL jaccard(...): similarityCutoff must be in range [0, 1]");
    }

    if (degree_cutoff > upper_degree_cutoff) {
        throw QueryExecutionException(
            "CALL jaccard(...): degreeCutoff must be <= upperDegreeCutoff"
        );
    }

    if (top_n.has_value() && bottom_n.has_value()) {
        throw QueryExecutionException(
            "CALL jaccard(...): topN and bottomN cannot be used together"
        );
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
