#pragma once

#include "query/parser/op/op.h"
#include <vector>
#include <string>
#include "graph_models/object_id.h"

namespace MQL {
class OpProject : public Op {
public:
    std::string graph_name;
    std::vector<ObjectId> node_labels;
    std::vector<ObjectId> edge_types;

    OpProject(std::string&& graph_name_, std::vector<ObjectId>&& node_labels_, std::vector<ObjectId>&& edge_types_)
        : graph_name(std::move(graph_name_)), node_labels(std::move(node_labels_)), edge_types(std::move(edge_types_)) {}

    std::unique_ptr<Op> clone() const override {
        return std::make_unique<OpProject>(graph_name, node_labels, edge_types);
    }

    bool read_only() const override { return false; }

    void accept_visitor(OpVisitor& visitor) override { visitor.visit(*this); }

    std::set<VarId> get_all_vars() const override { return {}; }
    std::set<VarId> get_scope_vars() const override { return {}; }
    std::set<VarId> get_safe_vars() const override { return {}; }
    std::set<VarId> get_fixable_vars() const override { return {}; }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ')
           << "OpProject(graph_name:" << graph_name
           << ", node_labels:" << node_labels.size()
           << ", edge_types:" << edge_types.size() << ")\n";
        return os;
    }
};
} // namespace MQL
