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

        // ignore relations, labels and internal edge identifiers (_eX)
        if (lhs_type == ObjectId::MASK_DIRECTED_EDGE || lhs_type == ObjectId::MASK_UNDIRECTED_EDGE
            || lhs_type == ObjectId::MASK_EDGE_LABEL || lhs_type == ObjectId::MASK_NODE_LABEL
            || lhs_type == ObjectId::MASK_EDGE_KEY || lhs_type == ObjectId::MASK_NODE_KEY)
        {
            return ObjectId(ObjectId::BOOL_FALSE);
        }

        for (auto& expr : rhs) {
            if (lhs_oid == expr->eval(binding)) {
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
