#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "query/parser/op/gql/op.h"

namespace GQL {

class OpCall : public Op {
public:
    std::unique_ptr<Op> subquery;
    std::vector<VarId> yield_items;
    bool has_explicit_yield;
    std::map<VarId, VarType> yield_types;

    OpCall(std::unique_ptr<Op> subquery, std::vector<VarId> yield_items, bool has_explicit_yield)
        : subquery(std::move(subquery)),
          yield_items(std::move(yield_items)),
          has_explicit_yield(has_explicit_yield)
    { }

    std::unique_ptr<Op> clone() const override
    {
        std::vector<VarId> yield_clone = yield_items;
        auto result = std::make_unique<OpCall>(
            subquery == nullptr ? nullptr : subquery->clone(),
            std::move(yield_clone),
            has_explicit_yield
        );
        result->yield_types = yield_types;
        return result;
    }

    void accept_visitor(OpVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    std::set<VarId> get_all_vars() const override
    {
        return std::set<VarId>(yield_items.begin(), yield_items.end());
    }

    std::map<VarId, VarType> get_var_types() const override
    {
        return yield_types;
    }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override
    {
        os << std::string(indent, ' ');
        os << "OpCall(yield=[";
        bool first = true;
        for (auto& var : yield_items) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << var;
        }
        os << "], explicit=" << (has_explicit_yield ? "true" : "false") << ")\n";
        if (subquery != nullptr) {
            subquery->print_to_ostream(os, indent + 2);
        }
        return os;
    }
};

} // namespace GQL
