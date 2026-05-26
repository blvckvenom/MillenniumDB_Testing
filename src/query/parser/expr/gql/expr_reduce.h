#pragma once

#include "query/parser/expr/gql/expr.h"

namespace GQL {
class ExprReduce : public Expr {
public:
    VarId accumulator_var;
    VarId loop_var;
    std::unique_ptr<Expr> initial_expr;
    std::unique_ptr<Expr> list_expr;
    std::unique_ptr<Expr> projection_expr;

    ExprReduce(
        VarId accumulator_var,
        VarId loop_var,
        std::unique_ptr<Expr> initial_expr,
        std::unique_ptr<Expr> list_expr,
        std::unique_ptr<Expr> projection_expr
    ) :
        accumulator_var(accumulator_var),
        loop_var(loop_var),
        initial_expr(std::move(initial_expr)),
        list_expr(std::move(list_expr)),
        projection_expr(std::move(projection_expr))
    { }

    virtual std::unique_ptr<Expr> clone() const override
    {
        return std::make_unique<ExprReduce>(
            accumulator_var,
            loop_var,
            initial_expr->clone(),
            list_expr->clone(),
            projection_expr->clone()
        );
    }

    void accept_visitor(ExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    bool has_aggregation() const override
    {
        return initial_expr->has_aggregation() ||
               list_expr->has_aggregation() ||
               projection_expr->has_aggregation();
    }

    std::set<VarId> get_all_vars() const override
    {
        auto vars = initial_expr->get_all_vars();

        auto list_vars = list_expr->get_all_vars();
        vars.insert(list_vars.begin(), list_vars.end());

        auto projection_vars = projection_expr->get_all_vars();
        vars.insert(projection_vars.begin(), projection_vars.end());

        vars.erase(accumulator_var);
        vars.erase(loop_var);
        return vars;
    }
};
} // namespace GQL
