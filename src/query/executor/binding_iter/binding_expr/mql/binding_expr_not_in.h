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
        auto lhs_generic = lhs_oid.id & ObjectId::GENERIC_TYPE_MASK;
        auto lhs_type = lhs_oid.id & ObjectId::TYPE_MASK;

        // ignore relations, labels and internal edge identifiers (_eX)
        if (lhs_generic == ObjectId::MASK_EDGE || lhs_type == ObjectId::MASK_EDGE_LABEL
            || lhs_type == ObjectId::MASK_NODE_LABEL || lhs_type == ObjectId::MASK_EDGE_KEY
            || lhs_type == ObjectId::MASK_NODE_KEY)
        {
            return ObjectId::get_null();
        }

        bool compatible = false;
        for (auto& expr : rhs) {
            auto rhs_oid = expr->eval(binding);
            auto rhs_generic = rhs_oid.id & ObjectId::GENERIC_TYPE_MASK;
            auto rhs_type = rhs_oid.id & ObjectId::TYPE_MASK;

            if (rhs_generic == lhs_generic && rhs_type == lhs_type) {
                compatible = true;
                if (lhs_oid == rhs_oid) {
                    return ObjectId(ObjectId::BOOL_FALSE);
                }
            }
        }

        if (!compatible) {
            return ObjectId::get_null();
        }

        return ObjectId(ObjectId::BOOL_TRUE);
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }
};
} // namespace MQL
