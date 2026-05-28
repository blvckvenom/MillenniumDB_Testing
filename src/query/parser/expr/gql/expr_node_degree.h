#pragma once

#include "query/parser/expr/gql/expr.h"

namespace GQL {
enum class NodeDegreeType {
    OUT,
    IN,
    BOTH
};

class ExprNodeDegree : public Expr {
public:
    VarId var;
    NodeDegreeType degree_type;

    ExprNodeDegree(VarId var, NodeDegreeType degree_type) :
        var(var),
        degree_type(degree_type)
    { }

    std::unique_ptr<Expr> clone() const override
    {
        return std::make_unique<ExprNodeDegree>(var, degree_type);
    }

    void accept_visitor(ExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    bool has_aggregation() const override
    {
        return false;
    }

    std::set<VarId> get_all_vars() const override
    {
        return { var };
    }
};
} // namespace GQL
