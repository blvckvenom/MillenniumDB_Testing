#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "query/executor/binding_iter.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"
#include "query/var_id.h"

namespace Procedure {

class NodeSimilarity : public BindingIter {
public:
    NodeSimilarity(
        std::vector<std::unique_ptr<BindingExpr>>&& argument_binding_exprs_,
        std::vector<VarId>&& yield_vars_
    );

    void print(std::ostream& os, int indent, bool stats) const override;

    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;
    void assign_nulls() override;

private:
    enum class SimilarityMetric {
        JACCARD,
        OVERLAP
    };

    const std::vector<std::unique_ptr<BindingExpr>> argument_binding_exprs;
    const std::vector<VarId> yield_vars;

    Binding* parent_binding = nullptr;
    std::vector<std::tuple<ObjectId, ObjectId, ObjectId>> results;
    std::size_t cursor = 0;
    SimilarityMetric similarity_metric = SimilarityMetric::JACCARD;
    double similarity_cutoff = 0.0;
    uint64_t degree_cutoff = 1;
    uint64_t upper_degree_cutoff = UINT64_MAX;
    std::optional<uint64_t> top_k;
    std::optional<uint64_t> bottom_k;
    std::optional<uint64_t> top_n;
    std::optional<uint64_t> bottom_n;

    void eval_arguments();
};

} // namespace Procedure
