#include "node_similarity.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <utility>

#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "query/exceptions.h"

using namespace Procedure;

namespace {

using ProfileClock = std::chrono::steady_clock;
using ProfileDurationMS = std::chrono::duration<double, std::milli>;

struct CsrAdjacency {
    std::vector<uint64_t> node_ids;
    std::vector<std::size_t> offsets { 0 };
    std::vector<uint64_t> neighbors;

    std::size_t degree(std::size_t row) const noexcept
    {
        assert(offsets.size() == node_ids.size() + 1);
        assert(row < node_ids.size());
        return offsets[row + 1] - offsets[row];
    }
};

static constexpr const char* NODE_SIMILARITY_PROFILE_PATH =
    "/Users/andres/Documents/beauchef/memoria/node_similarity_benchmarks/results/node_similarity_profile.csv";

double profile_ms(ProfileClock::time_point start, ProfileClock::time_point end)
{
    return ProfileDurationMS(end - start).count();
}

struct NodeSimilarityProfile {
    const char* similarity_metric = "JACCARD";
    double similarity_cutoff = 0.0;
    uint64_t degree_cutoff = 1;
    uint64_t upper_degree_cutoff = UINT64_MAX;
    bool has_top_k = false;
    uint64_t top_k = 0;
    bool has_bottom_k = false;
    uint64_t bottom_k = 0;
    bool has_top_n = false;
    uint64_t top_n = 0;
    bool has_bottom_n = false;
    uint64_t bottom_n = 0;

    double eval_arguments_ms = 0.0;
    double scan_undirected_ms = 0.0;
    double scan_directed_ms = 0.0;
    double degree_filter_ms = 0.0;
    double pair_scoring_ms = 0.0;
    double per_node_ranking_ms = 0.0;
    double global_ranking_ms = 0.0;
    double total_reset_ms = 0.0;
    double total_next_ms = 0.0;
    double total_operator_ms = 0.0;

    uint64_t undirected_records = 0;
    uint64_t directed_records = 0;
    uint64_t adjacency_nodes = 0;
    uint64_t neighbor_entries = 0;
    uint64_t nodes_after_degree_filter = 0;
    uint64_t pairs_checked = 0;
    uint64_t pairs_after_similarity_cutoff = 0;
    uint64_t results_before_global_limit = 0;
    uint64_t k_candidates_total = 0;
    uint64_t max_k_candidates_for_node = 0;
    uint64_t k_candidates_capacity_total = 0;
    uint64_t max_k_candidates_capacity_for_node = 0;
    uint64_t results_size = 0;
    uint64_t results_capacity = 0;
    uint64_t result_tuple_size = sizeof(std::tuple<ObjectId, ObjectId, ObjectId>);
    uint64_t k_candidate_tuple_size = sizeof(std::tuple<ObjectId, ObjectId, ObjectId>);
    uint64_t next_calls = 0;
    uint64_t next_true_calls = 0;
    uint64_t next_false_calls = 0;
};

void write_profile_csv(const NodeSimilarityProfile& profile)
{
    const bool write_header = []() {
        std::ifstream input(NODE_SIMILARITY_PROFILE_PATH);
        return !input.good() || input.peek() == std::ifstream::traits_type::eof();
    }();

    std::ofstream output(NODE_SIMILARITY_PROFILE_PATH, std::ios::app);
    if (!output.good()) {
        return;
    }

    if (write_header) {
        output << "similarity_metric,"
               << "similarity_cutoff,"
               << "degree_cutoff,"
               << "upper_degree_cutoff,"
               << "top_k,"
               << "bottom_k,"
               << "top_n,"
               << "bottom_n,"
               << "eval_arguments_ms,"
               << "scan_undirected_ms,"
               << "scan_directed_ms,"
               << "degree_filter_ms,"
               << "pair_scoring_ms,"
               << "per_node_ranking_ms,"
               << "global_ranking_ms,"
               << "total_reset_ms,"
               << "total_next_ms,"
               << "total_operator_ms,"
               << "undirected_records,"
               << "directed_records,"
               << "adjacency_nodes,"
               << "neighbor_entries,"
               << "nodes_after_degree_filter,"
               << "pairs_checked,"
               << "pairs_after_similarity_cutoff,"
               << "results_before_global_limit,"
               << "k_candidates_total,"
               << "max_k_candidates_for_node,"
               << "k_candidates_capacity_total,"
               << "max_k_candidates_capacity_for_node,"
               << "results_size,"
               << "results_capacity,"
               << "result_tuple_size,"
               << "k_candidate_tuple_size,"
               << "next_calls,"
               << "next_true_calls,"
               << "next_false_calls\n";
    }

    output << profile.similarity_metric << ','
           << profile.similarity_cutoff << ','
           << profile.degree_cutoff << ','
           << profile.upper_degree_cutoff << ',';
    if (profile.has_top_k) {
        output << profile.top_k;
    }
    output << ',';
    if (profile.has_bottom_k) {
        output << profile.bottom_k;
    }
    output << ',';
    if (profile.has_top_n) {
        output << profile.top_n;
    }
    output << ',';
    if (profile.has_bottom_n) {
        output << profile.bottom_n;
    }
    output << ','
           << profile.eval_arguments_ms << ','
           << profile.scan_undirected_ms << ','
           << profile.scan_directed_ms << ','
           << profile.degree_filter_ms << ','
           << profile.pair_scoring_ms << ','
           << profile.per_node_ranking_ms << ','
           << profile.global_ranking_ms << ','
           << profile.total_reset_ms << ','
           << profile.total_next_ms << ','
           << profile.total_operator_ms << ','
           << profile.undirected_records << ','
           << profile.directed_records << ','
           << profile.adjacency_nodes << ','
           << profile.neighbor_entries << ','
           << profile.nodes_after_degree_filter << ','
           << profile.pairs_checked << ','
           << profile.pairs_after_similarity_cutoff << ','
           << profile.results_before_global_limit << ','
           << profile.k_candidates_total << ','
           << profile.max_k_candidates_for_node << ','
           << profile.k_candidates_capacity_total << ','
           << profile.max_k_candidates_capacity_for_node << ','
           << profile.results_size << ','
           << profile.results_capacity << ','
           << profile.result_tuple_size << ','
           << profile.k_candidate_tuple_size << ','
           << profile.next_calls << ','
           << profile.next_true_calls << ','
           << profile.next_false_calls << '\n';
}

} // namespace

NodeSimilarity::NodeSimilarity(
    std::vector<std::unique_ptr<BindingExpr>>&& argument_binding_exprs_,
    std::vector<VarId>&& yield_vars_
) :
    argument_binding_exprs { std::move(argument_binding_exprs_) },
    yield_vars { std::move(yield_vars_) }
{
    assert(argument_binding_exprs.size() <= 8);
    assert(yield_vars.size() == 3);
}

void NodeSimilarity::_begin(Binding& parent_binding_)
{
    parent_binding = &parent_binding_;
    _reset();
}

void NodeSimilarity::_reset()
{
    const auto total_reset_start = ProfileClock::now();
    NodeSimilarityProfile profile;

    results.clear();
    cursor = 0;

    const auto eval_arguments_start = ProfileClock::now();
    eval_arguments();
    profile.eval_arguments_ms = profile_ms(eval_arguments_start, ProfileClock::now());
    switch (similarity_metric) {
    case SimilarityMetric::JACCARD:
        profile.similarity_metric = "JACCARD";
        break;
    case SimilarityMetric::OVERLAP:
        profile.similarity_metric = "OVERLAP";
        break;
    case SimilarityMetric::COSINE:
        profile.similarity_metric = "COSINE";
        break;
    }
    profile.similarity_cutoff = similarity_cutoff;
    profile.degree_cutoff = degree_cutoff;
    profile.upper_degree_cutoff = upper_degree_cutoff;
    if (top_k.has_value()) {
        profile.has_top_k = true;
        profile.top_k = *top_k;
    }
    if (bottom_k.has_value()) {
        profile.has_bottom_k = true;
        profile.bottom_k = *bottom_k;
    }
    if (top_n.has_value()) {
        profile.has_top_n = true;
        profile.top_n = *top_n;
    }
    if (bottom_n.has_value()) {
        profile.has_bottom_n = true;
        profile.bottom_n = *bottom_n;
    }

    std::array<uint64_t, 3> min_ids = { 0, 0, 0 };
    std::array<uint64_t, 3> max_ids = { UINT64_MAX, UINT64_MAX, UINT64_MAX };

    std::vector<std::pair<uint64_t, uint64_t>> adjacency_contributions;
    std::map<uint64_t, std::set<uint64_t>> adjacency;

    const auto scan_undirected_start = ProfileClock::now();
    auto undirected_edge_iter = gql_model.get_n1_n2_edge().get_range(
        &get_query_ctx().thread_info.interruption_requested,
        min_ids,
        max_ids
    );
    while (const auto* current_record = undirected_edge_iter.next()) {
        const auto node1 = (*current_record)[0];
        const auto node2 = (*current_record)[1];

        adjacency_contributions.emplace_back(node1, node2);
        adjacency_contributions.emplace_back(node2, node1);

        // Undirected edge contributes both ways: node1~node2 => neighbors(node1)+=node2 and neighbors(node2)+=node1
        adjacency[node1].insert(node2);
        adjacency[node2].insert(node1);
        ++profile.undirected_records;
    }
    profile.scan_undirected_ms = profile_ms(scan_undirected_start, ProfileClock::now());

    const auto scan_directed_start = ProfileClock::now();
    auto directed_edge_iter = gql_model.get_from_to_edge().get_range(
        &get_query_ctx().thread_info.interruption_requested,
        min_ids,
        max_ids
    );
    while (const auto* current_record = directed_edge_iter.next()) {
        const auto from = (*current_record)[0];
        const auto to   = (*current_record)[1];

        adjacency_contributions.emplace_back(from, to);

        // Directed edge contributes only in outgoing direction: from->to => neighbors(from)+=to
        adjacency[from].insert(to);
        ++profile.directed_records;
    }
    profile.scan_directed_ms = profile_ms(scan_directed_start, ProfileClock::now());
    // Self-loops are included through the main edge indexes above. The equal_u_edge/equal_d_edge
    // indexes are only specialized access paths for explicit self-loop patterns.

    std::sort(adjacency_contributions.begin(), adjacency_contributions.end());
    adjacency_contributions.erase(
        std::unique(adjacency_contributions.begin(), adjacency_contributions.end()),
        adjacency_contributions.end()
    );

    CsrAdjacency csr_adjacency;
    for (const auto& [source, neighbor] : adjacency_contributions) {
        if (csr_adjacency.node_ids.empty() || csr_adjacency.node_ids.back() != source) {
            if (!csr_adjacency.node_ids.empty()) {
                csr_adjacency.offsets.push_back(csr_adjacency.neighbors.size());
            }
            csr_adjacency.node_ids.push_back(source);
        }
        csr_adjacency.neighbors.push_back(neighbor);
    }
    if (!csr_adjacency.node_ids.empty()) {
        csr_adjacency.offsets.push_back(csr_adjacency.neighbors.size());
    }

#ifndef NDEBUG
    assert(csr_adjacency.offsets.size() == csr_adjacency.node_ids.size() + 1);
    assert(csr_adjacency.offsets.front() == 0);
    assert(csr_adjacency.offsets.back() == csr_adjacency.neighbors.size());
    assert(csr_adjacency.node_ids.size() == adjacency.size());

    auto adjacency_it = adjacency.cbegin();
    for (std::size_t row = 0; row < csr_adjacency.node_ids.size(); ++row, ++adjacency_it) {
        assert(adjacency_it != adjacency.cend());
        assert(csr_adjacency.node_ids[row] == adjacency_it->first);
        assert(csr_adjacency.offsets[row] <= csr_adjacency.offsets[row + 1]);
        assert(csr_adjacency.offsets[row + 1] <= csr_adjacency.neighbors.size());
        assert(csr_adjacency.degree(row) == adjacency_it->second.size());
        assert(std::equal(
            csr_adjacency.neighbors.cbegin() + csr_adjacency.offsets[row],
            csr_adjacency.neighbors.cbegin() + csr_adjacency.offsets[row + 1],
            adjacency_it->second.cbegin(),
            adjacency_it->second.cend()
        ));
    }
    assert(adjacency_it == adjacency.cend());
#endif

    profile.adjacency_nodes = static_cast<uint64_t>(adjacency.size());
    for (const auto& [node, neighbors] : adjacency) {
        profile.neighbor_entries += static_cast<uint64_t>(neighbors.size());
    }

    const auto degree_filter_start = ProfileClock::now();
    std::vector<uint64_t> nodes;
    nodes.reserve(adjacency.size());
    for (const auto& [node, neighbors] : adjacency) {
        const auto degree = static_cast<uint64_t>(neighbors.size());
        if (degree >= degree_cutoff && degree <= upper_degree_cutoff) {
            nodes.push_back(node);
        }
    }
    profile.nodes_after_degree_filter = static_cast<uint64_t>(nodes.size());
    profile.degree_filter_ms = profile_ms(degree_filter_start, ProfileClock::now());

    if (nodes.size() < 2) {
        profile.results_before_global_limit = static_cast<uint64_t>(results.size());
        profile.results_size = static_cast<uint64_t>(results.size());
        profile.results_capacity = static_cast<uint64_t>(results.capacity());
        profile.total_reset_ms = profile_ms(total_reset_start, ProfileClock::now());
        profile.total_operator_ms = profile.total_reset_ms;
        write_profile_csv(profile);
        return;
    }

    std::map<uint64_t, std::vector<std::tuple<ObjectId, ObjectId, ObjectId>>> k_candidates;

    const auto pair_scoring_start = ProfileClock::now();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& neighbors_i = adjacency.at(nodes[i]);
        for (std::size_t j = i + 1; j < nodes.size(); ++j) {
            const auto& neighbors_j = adjacency.at(nodes[j]);
            ++profile.pairs_checked;

            const auto& smaller = (neighbors_i.size() <= neighbors_j.size()) ? neighbors_i : neighbors_j;
            const auto& larger = (neighbors_i.size() <= neighbors_j.size()) ? neighbors_j : neighbors_i;

            std::size_t intersection_size = 0;
            for (const auto& neighbor : smaller) {
                if (larger.count(neighbor) == 1) {
                    ++intersection_size;
                }
            }

            double similarity = 0.0;
            switch (similarity_metric) {
            case SimilarityMetric::JACCARD: {
                const auto union_size = neighbors_i.size() + neighbors_j.size() - intersection_size;
                similarity = (union_size == 0)
                                 ? 0.0
                                 : static_cast<double>(intersection_size) / static_cast<double>(union_size);
                break;
            }
            case SimilarityMetric::OVERLAP: {
                const auto min_degree = std::min(neighbors_i.size(), neighbors_j.size());
                similarity = (min_degree == 0)
                                 ? 0.0
                                 : static_cast<double>(intersection_size) / static_cast<double>(min_degree);
                break;
            }
            case SimilarityMetric::COSINE: {
                const auto denominator = std::sqrt(
                    static_cast<double>(neighbors_i.size()) * static_cast<double>(neighbors_j.size())
                );
                similarity = (denominator == 0.0)
                                 ? 0.0
                                 : static_cast<double>(intersection_size) / denominator;
                break;
            }
            }

            if (similarity >= similarity_cutoff) {
                ++profile.pairs_after_similarity_cutoff;
                const auto node_i = ObjectId(nodes[i]);
                const auto node_j = ObjectId(nodes[j]);
                const auto similarity_oid = GQL::Conversions::pack_double(similarity);

                if (top_k.has_value() || bottom_k.has_value()) {
                    k_candidates[nodes[i]].emplace_back(node_i, node_j, similarity_oid);
                    k_candidates[nodes[j]].emplace_back(node_j, node_i, similarity_oid);
                } else {
                    results.emplace_back(node_i, node_j, similarity_oid);
                }
            }
        }
    }
    profile.pair_scoring_ms = profile_ms(pair_scoring_start, ProfileClock::now());

    const auto per_node_ranking_start = ProfileClock::now();
    if (top_k.has_value() || bottom_k.has_value()) {
        for (auto& [node, candidates] : k_candidates) {
            profile.k_candidates_total += static_cast<uint64_t>(candidates.size());
            profile.max_k_candidates_for_node = std::max<uint64_t>(
                profile.max_k_candidates_for_node,
                static_cast<uint64_t>(candidates.size())
            );
            profile.k_candidates_capacity_total += static_cast<uint64_t>(candidates.capacity());
            profile.max_k_candidates_capacity_for_node = std::max<uint64_t>(
                profile.max_k_candidates_capacity_for_node,
                static_cast<uint64_t>(candidates.capacity())
            );

            std::sort(candidates.begin(), candidates.end(), [&](const auto& lhs, const auto& rhs) {
                const auto& [lhs_node1, lhs_node2, lhs_similarity_oid] = lhs;
                const auto& [rhs_node1, rhs_node2, rhs_similarity_oid] = rhs;

                const double lhs_similarity = GQL::Conversions::to_double(lhs_similarity_oid);
                const double rhs_similarity = GQL::Conversions::to_double(rhs_similarity_oid);

                if (lhs_similarity != rhs_similarity) {
                    return top_k.has_value()
                               ? lhs_similarity > rhs_similarity
                               : lhs_similarity < rhs_similarity;
                }
                return lhs_node2.id < rhs_node2.id;
            });

            const auto k = top_k.has_value() ? *top_k : *bottom_k;
            const auto result_count = std::min<std::size_t>(
                static_cast<std::size_t>(k),
                candidates.size()
            );
            results.insert(results.end(), candidates.begin(), candidates.begin() + result_count);
        }
    }
    profile.per_node_ranking_ms = profile_ms(per_node_ranking_start, ProfileClock::now());
    profile.results_before_global_limit = static_cast<uint64_t>(results.size());

    const auto global_ranking_start = ProfileClock::now();
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
    profile.global_ranking_ms = profile_ms(global_ranking_start, ProfileClock::now());
    profile.results_size = static_cast<uint64_t>(results.size());
    profile.results_capacity = static_cast<uint64_t>(results.capacity());
    profile.total_reset_ms = profile_ms(total_reset_start, ProfileClock::now());
    profile.total_operator_ms = profile.total_reset_ms;

    write_profile_csv(profile);
}

void NodeSimilarity::eval_arguments()
{
    similarity_metric = SimilarityMetric::JACCARD;
    similarity_cutoff = 0.0;
    degree_cutoff = 1;
    upper_degree_cutoff = UINT64_MAX;
    top_k.reset();
    bottom_k.reset();
    top_n.reset();
    bottom_n.reset();

    auto eval_optional_oid = [&](std::size_t arg_pos) -> std::optional<ObjectId> {
        if (arg_pos >= argument_binding_exprs.size()) {
            return std::nullopt;
        }
        const ObjectId oid = argument_binding_exprs[arg_pos]->eval(*parent_binding);
        if (oid.is_null()) {
            return std::nullopt;
        }
        return oid;
    };

    auto eval_numeric = [&](const ObjectId oid, const char* arg_name) -> ObjectId {
        switch (oid.get_sub_type()) {
        case ObjectId::MASK_INT:
        case ObjectId::MASK_DECIMAL:
        case ObjectId::MASK_FLOAT:
        case ObjectId::MASK_DOUBLE:
            return oid;
        default:
            throw QueryExecutionException(
                std::string("CALL nodeSimilarity(...): ") + arg_name + " must be numeric"
            );
        }
    };

    auto eval_non_negative_integer = [&](const ObjectId oid, const char* arg_name) -> uint64_t {
        if (oid.get_sub_type() != ObjectId::MASK_INT) {
            throw QueryExecutionException(
                std::string("CALL nodeSimilarity(...): ") + arg_name + " must be an integer >= 0"
            );
        }

        const int64_t value = GQL::Conversions::to_integer(oid);
        if (value < 0) {
            throw QueryExecutionException(
                std::string("CALL nodeSimilarity(...): ") + arg_name + " must be an integer >= 0"
            );
        }
        return static_cast<uint64_t>(value);
    };

    auto eval_positive_integer = [&](const ObjectId oid, const char* arg_name) -> uint64_t {
        const uint64_t value = eval_non_negative_integer(oid, arg_name);
        if (value == 0) {
            throw QueryExecutionException(
                std::string("CALL nodeSimilarity(...): ") + arg_name + " must be an integer > 0"
            );
        }
        return value;
    };

    auto eval_similarity_metric = [&](const ObjectId oid) -> SimilarityMetric {
        if ((oid.id & ObjectId::GENERIC_TYPE_MASK) != ObjectId::MASK_STRING) {
            throw QueryExecutionException(
                "CALL nodeSimilarity(...): similarityMetric must be a string"
            );
        }

        const std::string metric_name = GQL::Conversions::unpack_string(oid);
        if (metric_name == "JACCARD") {
            return SimilarityMetric::JACCARD;
        } else if (metric_name == "OVERLAP") {
            return SimilarityMetric::OVERLAP;
        } else if (metric_name == "COSINE") {
            return SimilarityMetric::COSINE;
        }

        throw QueryExecutionException(
            "CALL nodeSimilarity(...): unsupported similarityMetric \"" + metric_name
            + "\". Supported values are: JACCARD, OVERLAP, COSINE"
        );
    };

    if (auto maybe_similarity_metric_oid = eval_optional_oid(0); maybe_similarity_metric_oid.has_value()) {
        similarity_metric = eval_similarity_metric(*maybe_similarity_metric_oid);
    }

    if (auto maybe_cutoff_oid = eval_optional_oid(1); maybe_cutoff_oid.has_value()) {
        const ObjectId cutoff_oid = eval_numeric(*maybe_cutoff_oid, "similarityCutoff");
        similarity_cutoff = GQL::Conversions::to_double(cutoff_oid);
    }

    if (auto maybe_degree_cutoff_oid = eval_optional_oid(2); maybe_degree_cutoff_oid.has_value()) {
        degree_cutoff = eval_non_negative_integer(*maybe_degree_cutoff_oid, "degreeCutoff");
    }

    if (auto maybe_upper_degree_cutoff_oid = eval_optional_oid(3); maybe_upper_degree_cutoff_oid.has_value()) {
        upper_degree_cutoff = eval_non_negative_integer(*maybe_upper_degree_cutoff_oid, "upperDegreeCutoff");
    }

    if (auto maybe_top_k_oid = eval_optional_oid(4); maybe_top_k_oid.has_value()) {
        top_k = eval_positive_integer(*maybe_top_k_oid, "topK");
    }

    if (auto maybe_bottom_k_oid = eval_optional_oid(5); maybe_bottom_k_oid.has_value()) {
        bottom_k = eval_positive_integer(*maybe_bottom_k_oid, "bottomK");
    }

    if (auto maybe_top_n_oid = eval_optional_oid(6); maybe_top_n_oid.has_value()) {
        top_n = eval_non_negative_integer(*maybe_top_n_oid, "topN");
    }

    if (auto maybe_bottom_n_oid = eval_optional_oid(7); maybe_bottom_n_oid.has_value()) {
        bottom_n = eval_non_negative_integer(*maybe_bottom_n_oid, "bottomN");
    }

    if (!std::isfinite(similarity_cutoff) || similarity_cutoff < 0.0 || similarity_cutoff > 1.0) {
        throw QueryExecutionException("CALL nodeSimilarity(...): similarityCutoff must be in range [0, 1]");
    }

    if (degree_cutoff > upper_degree_cutoff) {
        throw QueryExecutionException(
            "CALL nodeSimilarity(...): degreeCutoff must be <= upperDegreeCutoff"
        );
    }

    if (top_n.has_value() && bottom_n.has_value()) {
        throw QueryExecutionException(
            "CALL nodeSimilarity(...): topN and bottomN cannot be used together"
        );
    }

    if (top_k.has_value() && bottom_k.has_value()) {
        throw QueryExecutionException(
            "CALL nodeSimilarity(...): topK and bottomK cannot be used together"
        );
    }

    if (top_k.has_value() && bottom_n.has_value()) {
        throw QueryExecutionException(
            "CALL nodeSimilarity(...): topK and bottomN cannot be used together"
        );
    }

    if (bottom_k.has_value() && top_n.has_value()) {
        throw QueryExecutionException(
            "CALL nodeSimilarity(...): bottomK and topN cannot be used together"
        );
    }

}

bool NodeSimilarity::_next()
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

void NodeSimilarity::assign_nulls()
{
    for (const auto& var : yield_vars) {
        parent_binding->add(var, ObjectId::get_null());
    }
}

void NodeSimilarity::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "NodeSimilarity() -> (";
    if (!yield_vars.empty()) {
        os << yield_vars[0];
        for (std::size_t i = 1; i < yield_vars.size(); ++i) {
            os << ", " << yield_vars[i];
        }
    }
    os << ")\n";
}
