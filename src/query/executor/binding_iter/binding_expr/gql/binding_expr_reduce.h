#pragma once

#include <memory>
#include <vector>

#include "graph_models/gql/conversions.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"

namespace GQL {
class BindingExprReduce : public BindingExpr {
public:
    VarId accumulator_var;
    VarId loop_var;
    std::unique_ptr<BindingExpr> initial_expr;
    std::unique_ptr<BindingExpr> list_expr;
    std::unique_ptr<BindingExpr> projection_expr;

    BindingExprReduce(
        VarId accumulator_var,
        VarId loop_var,
        std::unique_ptr<BindingExpr> initial_expr,
        std::unique_ptr<BindingExpr> list_expr,
        std::unique_ptr<BindingExpr> projection_expr
    ) :
        accumulator_var(accumulator_var),
        loop_var(loop_var),
        initial_expr(std::move(initial_expr)),
        list_expr(std::move(list_expr)),
        projection_expr(std::move(projection_expr))
    { }

    ObjectId eval(const Binding& binding) override
    {
        ObjectId accumulator = initial_expr->eval(binding);

        auto source_oid = list_expr->eval(binding);
        if (GQL_OID::get_type(source_oid) != GQL_OID::Type::LIST) {
            return ObjectId::get_null();
        }

        auto source_list = GQL::Conversions::unpack_list(source_oid);

        Binding local_binding(binding.size);
        local_binding.add_all(binding);

        for (auto& element : source_list) {
            local_binding.add(loop_var, element);
            local_binding.add(accumulator_var, accumulator);
            accumulator = projection_expr->eval(local_binding);
        }

        return accumulator;
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    void print(std::ostream& os, std::vector<BindingIter*>& ops) const override
    {
        os << "REDUCE(" << accumulator_var << " = ";
        initial_expr->print(os, ops);
        os << ", " << loop_var << " IN ";
        list_expr->print(os, ops);
        os << " | ";
        projection_expr->print(os, ops);
        os << ")";
    }
};
} // namespace GQL
