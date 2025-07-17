#pragma once

#include <memory>
#include <vector>

#include "query/parser/op/gql/op.h"
#include "query/parser/op/gql/op_query_statements.h"

namespace GQL {

class OpCallSubquery : public Op {
public:
    std::unique_ptr<Op> subquery;
    std::vector<VarId> imported_vars;
    bool optional;

    OpCallSubquery(
        std::unique_ptr<Op> subquery,
        std::vector<VarId> imported_vars,
        bool optional
    ) :
        subquery(std::move(subquery)),
        imported_vars(std::move(imported_vars)),
        optional(optional)
    { }

    std::unique_ptr<Op> clone() const override
    {
        std::vector<VarId> vars_copy = imported_vars;
        return std::make_unique<OpCallSubquery>(
            subquery->clone(),
            std::move(vars_copy),
            optional
        );
    }

    void accept_visitor(OpVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    std::set<VarId> get_all_vars() const override
    {
        auto res = subquery->get_all_vars();
        res.insert(imported_vars.begin(), imported_vars.end());
        return res;
    }

    std::map<VarId, VarType> get_var_types() const override
    {
        return subquery->get_var_types();
    }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override
    {
        os << std::string(indent, ' ') << "OpCallSubquery(";
        if (optional)
            os << "OPTIONAL";
        if (!imported_vars.empty()) {
            os << " imports:";
            bool first = true;
            for (auto v : imported_vars) {
                if (!first) os << ',';
                first = false;
                os << ' ' << v;
            }
        }
        os << ")\n";
        subquery->print_to_ostream(os, indent + 2);
        return os;
    }
};

} // namespace GQL

