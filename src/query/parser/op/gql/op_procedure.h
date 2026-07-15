#pragma once

#include "query/exceptions.h"
#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_printer.h"
#include "query/parser/op/gql/op.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace GQL {

class OpProcedure : public Op {
public:
    enum class ProcedureType : uint8_t {
        HELLO_WORLD, // primera función analítica de juguete
        NEIGHBORS, // devolver todos los vecinos de todos los nodos
        NODE_SIMILARITY // devolver similaridad NodeSimilarity entre pares de nodos
    };
    static std::string get_procedure_string(ProcedureType procedure_type)
    {
        switch (procedure_type) {
        case ProcedureType::HELLO_WORLD:
            return "HELLO_WORLD";
        case ProcedureType::NEIGHBORS:
            return "NEIGHBORS";
        case ProcedureType::NODE_SIMILARITY:
            return "NODE_SIMILARITY";
        default:
            throw NotSupportedException(
                "OpProcedure::get_procedure_string: Unhandled procedure type: "
                + std::to_string(static_cast<uint8_t>(procedure_type))
            );
        }
    }
    static std::vector<std::string> get_procedure_available_yield_variable_names(ProcedureType procedure_type)
    {
        switch (procedure_type) {
        case ProcedureType::HELLO_WORLD:
            return { "message" };
        case ProcedureType::NEIGHBORS:
            return { "node", "neighbor" };
        case ProcedureType::NODE_SIMILARITY:
            return { "node1", "node2", "similarity" };
        default:
            throw NotSupportedException(
                "OpProcedure::get_procedure_available_yield_variable_names: Unhandled procedure type: "
                + std::to_string(static_cast<uint8_t>(procedure_type))
            );
        }
    }

    ProcedureType procedure_type;
    std::vector<std::unique_ptr<Expr>> argument_exprs;
    std::vector<VarId> yield_vars;

    OpProcedure(
        ProcedureType procedure_type_,
        std::vector<std::unique_ptr<Expr>>&& argument_exprs_,
        std::vector<VarId>&& yield_vars_
    ) :
        procedure_type { procedure_type_ },
        argument_exprs { std::move(argument_exprs_) },
        yield_vars { std::move(yield_vars_) }
    {
        assert(!yield_vars.empty());
    }

    std::unique_ptr<Op> clone() const override
    {
        std::vector<std::unique_ptr<Expr>> argument_exprs_clone;
        argument_exprs_clone.reserve(argument_exprs.size());
        for (const auto& expr : argument_exprs) {
            argument_exprs_clone.emplace_back(expr->clone());
        }

        auto yield_vars_clone = yield_vars;
        return std::make_unique<OpProcedure>(
            procedure_type,
            std::move(argument_exprs_clone),
            std::move(yield_vars_clone)
        );
    }
    void accept_visitor(OpVisitor& visitor) override
    {
        visitor.visit(*this);
    }
    std::set<VarId> get_all_vars() const override
    {
        std::set<VarId> vars(yield_vars.begin(), yield_vars.end());
        for (const auto& expr : argument_exprs) {
            auto expr_vars = expr->get_all_vars();
            vars.insert(expr_vars.begin(), expr_vars.end());
        }
        return vars;
    }
    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override
    {
        ExprPrinter expr_printer(os);
        os << std::string(indent, ' ');
        os << "OpProcedure(";
        os << get_procedure_string(procedure_type) << ", ";

        if (!argument_exprs.empty()) {
            argument_exprs[0]->accept_visitor(expr_printer);
            for (std::size_t i = 1; i < argument_exprs.size(); ++i) {
                os << ", ";
                argument_exprs[i]->accept_visitor(expr_printer);
            }
        }

        os << ") -> (";
        if (!yield_vars.empty()) {
            os << yield_vars[0];
            for (std::size_t i = 1; i < yield_vars.size(); ++i) {
                os << ", " << yield_vars[i];
            }
        }
        return os << ")\n";
    }
};
} // namespace GQL
