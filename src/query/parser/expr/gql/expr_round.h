#pragma once

#include "query/parser/expr/gql/expr.h"

namespace GQL {
class ExprRound : public Expr {
public:
    std::unique_ptr<Expr> expr;
    std::unique_ptr<Expr> precision;

    ExprRound(std::unique_ptr<Expr> expr, std::unique_ptr<Expr> precision = nullptr) :
        expr(std::move(expr)),
        precision(std::move(precision))
    { }

    virtual std::unique_ptr<Expr> clone() const override
    {
        return std::make_unique<ExprRound>(
            expr->clone(),
            precision == nullptr ? nullptr : precision->clone()
        );
    }

    void accept_visitor(ExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    bool has_aggregation() const override
    {
        return expr->has_aggregation() || (precision != nullptr && precision->has_aggregation());
    }

    std::set<VarId> get_all_vars() const override
    {
        auto vars = expr->get_all_vars();
        if (precision != nullptr) {
            auto precision_vars = precision->get_all_vars();
            vars.insert(precision_vars.begin(), precision_vars.end());
        }
        return vars;
    }
};
} // namespace GQL
