#pragma once

#include <string>
#include "query/parser/op/op.h"

namespace MQL {

class OpProject : public Op {
public:
    std::string name;
    std::string node_query;
    std::string edge_query;

    OpProject(std::string name_, std::string node_q, std::string edge_q)
        : name(std::move(name_)), node_query(std::move(node_q)), edge_query(std::move(edge_q)) {}

    std::unique_ptr<Op> clone() const override {
        return std::make_unique<OpProject>(name, node_query, edge_query);
    }

    void accept_visitor(OpVisitor& visitor) override { visitor.visit(*this); }

    std::set<VarId> get_all_vars() const override { return {}; }
    std::set<VarId> get_scope_vars() const override { return {}; }
    std::set<VarId> get_safe_vars() const override { return {}; }
    std::set<VarId> get_fixable_vars() const override { return {}; }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ');
        os << "OpProject(name: \"" << name << "\")\n";
        return os;
    }
};

} // namespace MQL
