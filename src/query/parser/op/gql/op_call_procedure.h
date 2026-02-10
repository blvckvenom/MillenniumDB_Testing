#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_printer.h"
#include "query/parser/op/gql/op.h"
#include "query/procedure/procedure.h"
#include "query/var_id.h"

namespace GQL {

/**
 * Operator representing a CALL procedure statement.
 * Stores the procedure reference, arguments, yield items, and optional flag.
 */
class OpCallProcedure : public Op {
public:
    /**
     * Yield item specifying which procedure output field to bind to which variable.
     */
    struct YieldItem {
        std::string field_name;  // Output field name from procedure
        VarId var;               // Variable to bind the value to

        YieldItem(std::string field_name_, VarId var_) :
            field_name(std::move(field_name_)),
            var(var_)
        { }
    };

    Procedure* procedure;                          // Pointer to registered procedure
    std::vector<std::unique_ptr<Expr>> arguments;  // Argument expressions
    std::vector<YieldItem> yield_items;            // YIELD field → variable mappings
    bool optional;                                 // OPTIONAL CALL flag

    OpCallProcedure(
        Procedure* procedure_,
        std::vector<std::unique_ptr<Expr>>&& arguments_,
        std::vector<YieldItem>&& yield_items_,
        bool optional_ = false
    ) :
        procedure(procedure_),
        arguments(std::move(arguments_)),
        yield_items(std::move(yield_items_)),
        optional(optional_)
    { }

    std::unique_ptr<Op> clone() const override
    {
        std::vector<std::unique_ptr<Expr>> arguments_clone;
        arguments_clone.reserve(arguments.size());
        for (auto& arg : arguments) {
            arguments_clone.push_back(arg->clone());
        }

        std::vector<YieldItem> yield_items_clone;
        yield_items_clone.reserve(yield_items.size());
        for (auto& item : yield_items) {
            yield_items_clone.emplace_back(item.field_name, item.var);
        }

        return std::make_unique<OpCallProcedure>(
            procedure,
            std::move(arguments_clone),
            std::move(yield_items_clone),
            optional
        );
    }

    void accept_visitor(OpVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    std::set<VarId> get_all_vars() const override
    {
        std::set<VarId> res;

        // Add variables from argument expressions
        for (auto& arg : arguments) {
            for (auto& var : arg->get_all_vars()) {
                res.insert(var);
            }
        }

        // Add yield variables
        for (auto& item : yield_items) {
            res.insert(item.var);
        }

        return res;
    }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override
    {
        os << std::string(indent, ' ');
        os << "OpCallProcedure(\n";

        os << std::string(indent + 2, ' ');
        os << "procedure: " << procedure->qualified_name();
        if (optional) {
            os << " [OPTIONAL]";
        }
        os << "\n";

        if (!arguments.empty()) {
            os << std::string(indent + 2, ' ');
            os << "arguments: [";
            ExprPrinter printer(os);
            for (size_t i = 0; i < arguments.size(); i++) {
                if (i > 0) os << ", ";
                arguments[i]->accept_visitor(printer);
            }
            os << "]\n";
        }

        if (!yield_items.empty()) {
            os << std::string(indent + 2, ' ');
            os << "yield: [";
            for (size_t i = 0; i < yield_items.size(); i++) {
                if (i > 0) os << ", ";
                os << yield_items[i].field_name << " AS " << yield_items[i].var;
            }
            os << "]\n";
        }

        os << std::string(indent, ' ');
        os << ")";

        return os;
    }
};

} // namespace GQL
