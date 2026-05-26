#pragma once

#include "query/parser/expr/gql/expr.h"

namespace GQL {
class ExprListComprehension : public Expr {
public:
    VarId local_var;
    std::unique_ptr<Expr> list_expr;
    std::unique_ptr<Expr> where_expr;
    std::unique_ptr<Expr> projection_expr;

    ExprListComprehension(
        VarId local_var,
        std::unique_ptr<Expr> list_expr,
        std::unique_ptr<Expr> where_expr,
        std::unique_ptr<Expr> projection_expr
    ) :
        local_var(local_var),
        list_expr(std::move(list_expr)),
        where_expr(std::move(where_expr)),
        projection_expr(std::move(projection_expr))
    { }

    virtual std::unique_ptr<Expr> clone() const override
    {
        return std::make_unique<ExprListComprehension>(
            local_var,
            list_expr->clone(),
            where_expr == nullptr ? nullptr : where_expr->clone(),
            projection_expr->clone()
        );
    }

    void accept_visitor(ExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    bool has_aggregation() const override
    {
        return list_expr->has_aggregation() ||
               (where_expr != nullptr && where_expr->has_aggregation()) ||
               projection_expr->has_aggregation();
    }

    std::set<VarId> get_all_vars() const override
    {
        auto vars = list_expr->get_all_vars();
        if (where_expr != nullptr) {
            auto where_vars = where_expr->get_all_vars();
            vars.insert(where_vars.begin(), where_vars.end());
        }
        auto projection_vars = projection_expr->get_all_vars();
        vars.insert(projection_vars.begin(), projection_vars.end());

        vars.erase(local_var);
        return vars;
    }
};
} // namespace GQL
