#pragma once

#include "query/parser/op/gql/op.h"

namespace GQL {

class OpOptional : public Op {
public:
    std::unique_ptr<Op> op;

    OpOptional(std::unique_ptr<Op> op) :
        op(std::move(op))
    { }

    std::unique_ptr<Op> clone() const override
    {
        return std::make_unique<OpOptional>(op->clone());
    }

    void accept_visitor(OpVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    std::set<VarId> get_all_vars() const override
    {
        return op->get_all_vars();
    }

    std::map<VarId, GQL::VarType> get_var_types() const override
    {
        return op->get_var_types();
    }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override
    {
        os << std::string(indent, ' ') << "OpOptional()\n";
        op->print_to_ostream(os, indent + 2);
        return os;
    }
};
} // namespace GQL
