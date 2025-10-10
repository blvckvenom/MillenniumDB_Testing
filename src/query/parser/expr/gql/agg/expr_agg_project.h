#pragma once

#include "query/parser/expr/gql/expr.h"

namespace GQL {
class ExprAggProject : public Expr {
public:
    std::unique_ptr<Expr> projection_name_expr;
    VarId var;

    ExprAggProject(
        std::unique_ptr<Expr> projection_name_expr,
        VarId var
    ) :
        projection_name_expr(std::move(projection_name_expr)),
        var(var)
    { }

    virtual std::unique_ptr<Expr> clone() const override
    {
        return std::make_unique<ExprAggProject>(projection_name_expr->clone(), var);
    }

    void accept_visitor(ExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    std::set<VarId> get_all_vars() const override
    {
        // PROJECT doesn't use any regular variables, only the projection name (a literal)
        // But we still return any variables that might be in the projection name expression
        return projection_name_expr->get_all_vars();
    }

    bool has_aggregation() const override
    {
        return true;
    }
};
} // namespace GQL
