#pragma once

#include <memory>
#include <vector>

#include "query/executor/binding_iter/binding_expr/binding_expr.h"

namespace MQL {
class BindingExprNotIn : public BindingExpr {
public:
    std::unique_ptr<BindingExpr> lhs;
    std::vector<std::unique_ptr<BindingExpr>> rhs;

    BindingExprNotIn(std::unique_ptr<BindingExpr> lhs, std::vector<std::unique_ptr<BindingExpr>> rhs) :
        lhs(std::move(lhs)),
        rhs(std::move(rhs))
    { }

    ObjectId eval(const Binding& binding) override
    {
        auto lhs_oid = lhs->eval(binding);
        auto lhs_type = lhs_oid.id & ObjectId::TYPE_MASK;

        // Only real nodes should be considered. Any other value (relations,
        // labels, internal edge identifiers, strings, etc.) must be ignored and
        // therefore filtered out from the result set.
        if (lhs_type != ObjectId::MASK_NODE) {
            return ObjectId(ObjectId::BOOL_FALSE);
        }

        for (auto& expr : rhs) {
            auto rhs_oid = expr->eval(binding);
            auto rhs_type = rhs_oid.id & ObjectId::TYPE_MASK;

            if (rhs_type == ObjectId::MASK_NODE && lhs_oid == rhs_oid) {
                return ObjectId(ObjectId::BOOL_FALSE);
            }
        }

        return ObjectId(ObjectId::BOOL_TRUE);
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }
};
} // namespace MQL
