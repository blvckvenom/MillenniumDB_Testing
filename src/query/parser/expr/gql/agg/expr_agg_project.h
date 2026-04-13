#pragma once

#include <optional>
#include <string>
#include <vector>

#include "query/parser/expr/gql/expr.h"

namespace GQL {

// Options for projection creation (parsed from INCLUDE clauses)
struct ProjectionOptions {
    bool include_labels = false;      // INCLUDE LABELS (both node and edge labels)
    bool include_properties = false;  // INCLUDE PROPERTIES (both node and edge properties)

    ProjectionOptions() = default;

    ProjectionOptions(bool labels, bool properties) :
        include_labels(labels),
        include_properties(properties)
    { }
};

// Configuration for selective properties (parsed from dataConfig Map parameter)
struct DataConfig {
    std::optional<std::vector<std::string>> source_node_properties;
    std::optional<std::vector<std::string>> target_node_properties;
    std::optional<std::vector<std::string>> relationship_properties;

    DataConfig() = default;

    bool is_empty() const {
        return !source_node_properties &&
               !target_node_properties &&
               !relationship_properties;
    }
};

class ExprAggProject : public Expr {
public:
    std::unique_ptr<Expr> projection_name_expr;
    std::unique_ptr<Expr> source_node_expr;  // Optional: source node expression
    std::unique_ptr<Expr> target_node_expr;  // Optional: target node expression
    VarId var;
    ProjectionOptions options;
    DataConfig data_config;  // Optional: selective properties configuration

    ExprAggProject(
        std::unique_ptr<Expr> projection_name_expr,
        VarId var,
        ProjectionOptions options = ProjectionOptions(),
        std::unique_ptr<Expr> source_node_expr = nullptr,
        std::unique_ptr<Expr> target_node_expr = nullptr,
        DataConfig data_config = DataConfig()
    ) :
        projection_name_expr(std::move(projection_name_expr)),
        source_node_expr(std::move(source_node_expr)),
        target_node_expr(std::move(target_node_expr)),
        var(var),
        options(options),
        data_config(std::move(data_config))
    { }

    virtual std::unique_ptr<Expr> clone() const override
    {
        return std::make_unique<ExprAggProject>(
            projection_name_expr->clone(),
            var,
            options,
            source_node_expr ? source_node_expr->clone() : nullptr,
            target_node_expr ? target_node_expr->clone() : nullptr,
            data_config
        );
    }

    void accept_visitor(ExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    std::set<VarId> get_all_vars() const override
    {
        // Collect variables from all expressions
        std::set<VarId> result = projection_name_expr->get_all_vars();

        if (source_node_expr) {
            auto source_vars = source_node_expr->get_all_vars();
            result.insert(source_vars.begin(), source_vars.end());
        }

        if (target_node_expr) {
            auto target_vars = target_node_expr->get_all_vars();
            result.insert(target_vars.begin(), target_vars.end());
        }

        return result;
    }

    bool has_aggregation() const override
    {
        return true;
    }
};
} // namespace GQL
