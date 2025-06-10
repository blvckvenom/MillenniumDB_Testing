#pragma once

#include <memory>

#include "query/parser/op/op.h"

namespace MQL {

class OpProject : public Op {
public:
    std::unique_ptr<Op> op;

    explicit OpProject(std::unique_ptr<Op> op) : op(std::move(op)) {}

    std::unique_ptr<Op> clone() const override {
        return std::make_unique<OpProject>(op->clone());
    }

    void accept_visitor(OpVisitor& visitor) override { visitor.visit(*this); }

    std::set<VarId> get_all_vars() const override { return op->get_all_vars(); }

    std::set<VarId> get_scope_vars() const override { return op->get_scope_vars(); }

    std::set<VarId> get_safe_vars() const override { return op->get_safe_vars(); }

    std::set<VarId> get_fixable_vars() const override { return op->get_fixable_vars(); }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ') << "OpProject()\n";
        return op->print_to_ostream(os, indent + 2);
    }
};

} // namespace MQL
