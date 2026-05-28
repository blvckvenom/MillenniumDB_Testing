#pragma once

#include <optional>

#include "graph_models/gql/conversions.h"
#include "query/parser/expr/gql/expr.h"

namespace GQL {
class ExprCast : public Expr {
public:
    std::unique_ptr<Expr> expr;
    GQL_OID::GenericType targetType;
    std::optional<GQL_OID::GenericSubType> targetNumericSubType;

    ExprCast(
        std::unique_ptr<Expr> operand,
        GQL_OID::GenericType targetType,
        std::optional<GQL_OID::GenericSubType> targetNumericSubType = std::nullopt
    ) :
        expr(std::move(operand)),
        targetType(targetType),
        targetNumericSubType(targetNumericSubType)
    { }

    virtual std::unique_ptr<Expr> clone() const override
    {
        return std::make_unique<ExprCast>(expr->clone(), targetType, targetNumericSubType);
    }

    void accept_visitor(ExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    bool has_aggregation() const override
    {
        return expr->has_aggregation();
    }

    std::set<VarId> get_all_vars() const override
    {
        return expr->get_all_vars();
    }
};
} // namespace GQL
