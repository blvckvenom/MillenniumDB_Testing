#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>

#include "graph_models/gql/conversions.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"

namespace GQL {
class BindingExprRound : public BindingExpr {
public:
    std::unique_ptr<BindingExpr> expr;
    std::unique_ptr<BindingExpr> precision;

    BindingExprRound(std::unique_ptr<BindingExpr> expr, std::unique_ptr<BindingExpr> precision = nullptr) :
        expr(std::move(expr)),
        precision(std::move(precision))
    { }

    ObjectId eval(const Binding& binding) override
    {
        auto expr_oid = expr->eval(binding);

        auto expr_subtype = GQL_OID::get_generic_sub_type(expr_oid);
        auto expr_generic_type = GQL_OID::get_generic_type(expr_oid);
        if (expr_generic_type != GQL_OID::GenericType::NUMERIC) {
            return ObjectId::get_null();
        }

        // ROUND(x): nearest integer
        if (precision == nullptr) {
            switch (expr_subtype) {
            case GQL_OID::GenericSubType::INTEGER: {
                auto value = GQL::Conversions::to_integer(expr_oid);
                return GQL::Conversions::pack_int(value);
            }
            case GQL_OID::GenericSubType::DECIMAL: {
                auto value = GQL::Conversions::to_decimal(expr_oid);
                return GQL::Conversions::pack_decimal(value.round());
            }
            case GQL_OID::GenericSubType::FLOAT: {
                auto value = GQL::Conversions::to_float(expr_oid);
                return GQL::Conversions::pack_int(static_cast<int64_t>(std::round(value)));
            }
            case GQL_OID::GenericSubType::DOUBLE: {
                auto value = GQL::Conversions::to_double(expr_oid);
                return GQL::Conversions::pack_int(static_cast<int64_t>(std::round(value)));
            }
            default: {
                assert(false);
                return ObjectId::get_null();
            }
            }
        }

        // ROUND(x, n): nearest value with n decimal places
        auto precision_oid = precision->eval(binding);
        if (GQL_OID::get_generic_type(precision_oid) != GQL_OID::GenericType::NUMERIC) {
            return ObjectId::get_null();
        }

        auto decimal_places = GQL::Conversions::to_integer(precision_oid);
        auto factor = std::pow(10.0, static_cast<double>(decimal_places));
        auto value = GQL::Conversions::to_double(expr_oid);
        auto rounded = std::round(value * factor) / factor;
        return GQL::Conversions::pack_double(rounded);
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    void print(std::ostream& os, std::vector<BindingIter*>& ops) const override
    {
        os << "ROUND(";
        expr->print(os, ops);
        if (precision != nullptr) {
            os << ", ";
            precision->print(os, ops);
        }
        os << ")";
    }
};
} // namespace GQL
