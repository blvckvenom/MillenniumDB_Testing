#pragma once

#include <memory>

#include "graph_models/gql/conversions.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"

namespace GQL {
class BindingExprListSize : public BindingExpr {
public:
    std::unique_ptr<BindingExpr> expr;

    BindingExprListSize(std::unique_ptr<BindingExpr> expr) :
        expr(std::move(expr))
    { }

    ObjectId eval(const Binding& binding) override
    {
        auto expr_oid = expr->eval(binding);

        if (GQL_OID::get_type(expr_oid) == GQL_OID::Type::LIST) {
            auto list = GQL::Conversions::unpack_list(expr_oid);
            return GQL::Conversions::pack_int(static_cast<int64_t>(list.size()));
        }

        return ObjectId::get_null();
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    void print(std::ostream& os, std::vector<BindingIter*>& ops) const override
    {
        os << "SIZE(";
        expr->print(os, ops);
        os << ")";
    }
};
} // namespace GQL
