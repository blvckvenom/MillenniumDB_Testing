#pragma once

#include <memory>
#include <vector>

#include "graph_models/gql/conversions.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"

namespace GQL {
class BindingExprListComprehension : public BindingExpr {
public:
    VarId local_var;
    std::unique_ptr<BindingExpr> list_expr;
    std::unique_ptr<BindingExpr> where_expr;
    std::unique_ptr<BindingExpr> projection_expr;

    BindingExprListComprehension(
        VarId local_var,
        std::unique_ptr<BindingExpr> list_expr,
        std::unique_ptr<BindingExpr> where_expr,
        std::unique_ptr<BindingExpr> projection_expr
    ) :
        local_var(local_var),
        list_expr(std::move(list_expr)),
        where_expr(std::move(where_expr)),
        projection_expr(std::move(projection_expr))
    { }

    ObjectId eval(const Binding& binding) override
    {
        auto source_oid = list_expr->eval(binding);
        if (GQL_OID::get_type(source_oid) != GQL_OID::Type::LIST) {
            return ObjectId::get_null();
        }

        auto source_list = GQL::Conversions::unpack_list(source_oid);
        std::vector<ObjectId> out;
        out.reserve(source_list.size());

        Binding local_binding(binding.size);
        local_binding.add_all(binding);

        for (auto& element : source_list) {
            local_binding.add(local_var, element);

            if (where_expr != nullptr) {
                auto where_oid = where_expr->eval(local_binding);
                auto where_bool = GQL::Conversions::to_boolean(where_oid);
                if (where_bool != GQL::Conversions::pack_bool(true)) {
                    continue;
                }
            }

            out.push_back(projection_expr->eval(local_binding));
        }

        return GQL::Conversions::pack_list(out);
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    void print(std::ostream& os, std::vector<BindingIter*>& ops) const override
    {
        os << "[" << local_var << " IN ";
        list_expr->print(os, ops);

        if (where_expr != nullptr) {
            os << " WHERE ";
            where_expr->print(os, ops);
        }

        os << " | ";
        projection_expr->print(os, ops);
        os << "]";
    }
};
} // namespace GQL
