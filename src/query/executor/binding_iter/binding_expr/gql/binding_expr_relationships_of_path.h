#pragma once

#include <memory>
#include <vector>

#include "graph_models/gql/conversions.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"

namespace GQL {
class BindingExprRelationshipsOfPath : public BindingExpr {
public:
    std::unique_ptr<BindingExpr> expr;

    BindingExprRelationshipsOfPath(std::unique_ptr<BindingExpr> expr) :
        expr(std::move(expr))
    { }

    ObjectId eval(const Binding& binding) override
    {
        auto path_oid = expr->eval(binding);

        if (GQL_OID::get_type(path_oid) != GQL_OID::Type::PATH) {
            return ObjectId::get_null();
        }

        std::vector<ObjectId> path;
        GQL::Conversions::unpack_path(path_oid, path);
        std::vector<ObjectId> relationships;
        relationships.reserve(path.size() / 3);

        for (size_t i = 1; i < path.size(); i += 3) {
            relationships.push_back(path[i]);
        }

        return GQL::Conversions::pack_list(relationships);
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    void print(std::ostream& os, std::vector<BindingIter*>& ops) const override
    {
        os << "RELATIONSHIPS(";
        expr->print(os, ops);
        os << ")";
    }
};
} // namespace GQL
