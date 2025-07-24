#pragma once

#include <memory>
#include <string>
#include <vector>

#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_printer.h"
#include "query/parser/op/gql/op.h"
#include "query/var_id.h"

namespace GQL {

class OpCall : public Op {
public:
    struct YieldItem {
        std::string field_name;
        VarId var_id;  // Variable to bind the result to
        
        YieldItem(std::string field_name, VarId var_id) 
            : field_name(std::move(field_name)), var_id(var_id) {}
    };

    virtual ~OpCall() = default;
    
    virtual std::unique_ptr<Op> clone() const override = 0;
    virtual void accept_visitor(OpVisitor& visitor) override = 0;
    virtual std::set<VarId> get_all_vars() const override = 0;
    virtual std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override = 0;

protected:
    std::vector<YieldItem> yield_items;
    bool is_optional;

public:
    OpCall(std::vector<YieldItem> yield_items, bool is_optional = false)
        : yield_items(std::move(yield_items)), is_optional(is_optional) {}

    const std::vector<YieldItem>& get_yield_items() const { return yield_items; }
    bool get_is_optional() const { return is_optional; }
    
    friend class QueryVisitor;
};

// Named procedure call: CALL procedure.name(args) YIELD ...
class OpCallNamed : public OpCall {
public:
    OpCallNamed(
        std::string procedure_name,
        std::vector<std::unique_ptr<Expr>> arguments,
        std::vector<YieldItem> yield_items,
        bool is_optional = false
    ) : OpCall(std::move(yield_items), is_optional),
        procedure_name(std::move(procedure_name)),
        arguments(std::move(arguments)) {}

    std::unique_ptr<Op> clone() const override {
        std::vector<std::unique_ptr<Expr>> cloned_args;
        for (const auto& arg : arguments) {
            cloned_args.push_back(arg->clone());
        }
        return std::make_unique<OpCallNamed>(
            procedure_name,
            std::move(cloned_args),
            yield_items,
            is_optional
        );
    }

    void accept_visitor(OpVisitor& visitor) override;

    std::set<VarId> get_all_vars() const override {
        std::set<VarId> vars;
        
        // Add variables from arguments
        for (const auto& arg : arguments) {
            auto arg_vars = arg->get_all_vars();
            vars.insert(arg_vars.begin(), arg_vars.end());
        }
        
        // Add yield variables
        for (const auto& yield_item : yield_items) {
            vars.insert(yield_item.var_id);
        }
        
        return vars;
    }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ') << "OpCallNamed(\n";
        os << std::string(indent + 2, ' ') << "procedure: " << procedure_name << "\n";
        os << std::string(indent + 2, ' ') << "optional: " << (is_optional ? "true" : "false") << "\n";
        
        if (!arguments.empty()) {
            os << std::string(indent + 2, ' ') << "arguments: [\n";
            ExprPrinter expr_printer(os);
            for (const auto& arg : arguments) {
                os << std::string(indent + 4, ' ');
                arg->accept_visitor(expr_printer);
                os << "\n";
            }
            os << std::string(indent + 2, ' ') << "]\n";
        }
        
        if (!yield_items.empty()) {
            os << std::string(indent + 2, ' ') << "yield: [\n";
            for (const auto& item : yield_items) {
                os << std::string(indent + 4, ' ') << item.field_name << " AS " << item.var_id << "\n";
            }
            os << std::string(indent + 2, ' ') << "]\n";
        }
        
        os << std::string(indent, ' ') << ")";
        return os;
    }

    const std::string& get_procedure_name() const { return procedure_name; }
    const std::vector<std::unique_ptr<Expr>>& get_arguments() const { return arguments; }

    friend class QueryVisitor;
private:
    std::string procedure_name;
    std::vector<std::unique_ptr<Expr>> arguments;
};

// Inline procedure call: CALL { subquery } YIELD ...
class OpCallInline : public OpCall {
public:
    OpCallInline(
        std::unique_ptr<Op> subquery,
        std::vector<YieldItem> yield_items,
        std::vector<VarId> scope_vars = {},
        bool is_optional = false
    ) : OpCall(std::move(yield_items), is_optional),
        subquery(std::move(subquery)),
        scope_vars(std::move(scope_vars)) {}

    std::unique_ptr<Op> clone() const override {
        return std::make_unique<OpCallInline>(
            subquery->clone(),
            yield_items,
            scope_vars,
            is_optional
        );
    }

    void accept_visitor(OpVisitor& visitor) override;

    std::set<VarId> get_all_vars() const override {
        auto vars = subquery->get_all_vars();
        
        // Add scope variables
        for (const auto& scope_var : scope_vars) {
            vars.insert(scope_var);
        }
        
        // Add yield variables
        for (const auto& yield_item : yield_items) {
            vars.insert(yield_item.var_id);
        }
        
        return vars;
    }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ') << "OpCallInline(\n";
        os << std::string(indent + 2, ' ') << "optional: " << (is_optional ? "true" : "false") << "\n";
        
        if (!scope_vars.empty()) {
            os << std::string(indent + 2, ' ') << "scope_vars: [";
            for (size_t i = 0; i < scope_vars.size(); ++i) {
                if (i > 0) os << ", ";
                os << scope_vars[i];
            }
            os << "]\n";
        }
        
        os << std::string(indent + 2, ' ') << "subquery:\n";
        subquery->print_to_ostream(os, indent + 4);
        os << "\n";
        
        if (!yield_items.empty()) {
            os << std::string(indent + 2, ' ') << "yield: [\n";
            for (const auto& item : yield_items) {
                os << std::string(indent + 4, ' ') << item.field_name << " AS " << item.var_id << "\n";
            }
            os << std::string(indent + 2, ' ') << "]\n";
        }
        
        os << std::string(indent, ' ') << ")";
        return os;
    }

    const std::unique_ptr<Op>& get_subquery() const { return subquery; }
    const std::vector<VarId>& get_scope_vars() const { return scope_vars; }

    friend class QueryVisitor;
private:
    std::unique_ptr<Op> subquery;
    std::vector<VarId> scope_vars;  // Variables from outer scope accessible in subquery
};

} // namespace GQL
