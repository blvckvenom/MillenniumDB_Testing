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
        auto lhs_generic = lhs_oid.id & ObjectId::GENERIC_TYPE_MASK;

        // Only evaluate NOT IN for real node identifiers. Any other value
        // (relations, labels, internal edge identifiers, strings, etc.) is
        // ignored by returning FALSE so that the binding is filtered out.
        if (lhs_type != ObjectId::MASK_NODE && lhs_generic != ObjectId::MASK_NAMED_NODE
            && lhs_generic != ObjectId::MASK_ANON)
        {
            return ObjectId(ObjectId::BOOL_FALSE);
        }

        for (auto& expr : rhs) {
            auto rhs_oid = expr->eval(binding);
            auto rhs_type = rhs_oid.id & ObjectId::TYPE_MASK;
            auto rhs_generic = rhs_oid.id & ObjectId::GENERIC_TYPE_MASK;

            if ((rhs_type == ObjectId::MASK_NODE || rhs_generic == ObjectId::MASK_NAMED_NODE
                 || rhs_generic == ObjectId::MASK_ANON)
                && lhs_oid == rhs_oid)
            {
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
