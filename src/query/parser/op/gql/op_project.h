#pragma once

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "query/parser/op/gql/op.h"

namespace GQL {

class OpProject : public Op {
public:
    VarId alias;
    std::unique_ptr<Op> subquery;
    std::vector<VarId> projected_items;
    std::map<VarId, VarType> projected_types;

    OpProject(VarId alias, std::unique_ptr<Op> subquery)
        : alias(alias),
          subquery(std::move(subquery))
    { }

    std::unique_ptr<Op> clone() const override
    {
        std::vector<VarId> items_clone = projected_items;
        auto result = std::make_unique<OpProject>(
            alias,
            subquery == nullptr ? nullptr : subquery->clone()
        );
        result->projected_items = std::move(items_clone);
        result->projected_types = projected_types;
        return result;
    }

    void accept_visitor(OpVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    std::set<VarId> get_all_vars() const override
    {
        return std::set<VarId>(projected_items.begin(), projected_items.end());
    }

    std::map<VarId, VarType> get_var_types() const override
    {
        return projected_types;
    }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override
    {
        os << std::string(indent, ' ');
        os << "OpProject(alias=" << alias << ", columns=[";
        bool first = true;
        for (auto& var : projected_items) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << var;
        }
        os << "])\n";
        if (subquery != nullptr) {
            subquery->print_to_ostream(os, indent + 2);
        }
        return os;
    }
};

} // namespace GQL

