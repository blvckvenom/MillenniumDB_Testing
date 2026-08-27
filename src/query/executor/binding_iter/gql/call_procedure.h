#pragma once

#include <memory>
#include <vector>

#include "query/executor/binding_iter.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"
#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {

/**
 * Binding iterator for executing CALL procedure statements.
 *
 * - Evaluates procedure arguments and executes the procedure once per
 *   begin/reset cycle.
 * - Without YIELD: returns a single row indicating success.
 * - With YIELD: iterates the procedure's result rows, binding each YIELD
 *   field to its variable (fields missing from a row bind to null).
 * - OPTIONAL flag: a failing procedure produces one row with every YIELD
 *   variable set to null instead of propagating the error.
 */
class CallProcedure : public BindingIter {
public:
    /**
     * Constructor for CallProcedure iterator.
     *
     * @param procedure Pointer to the registered procedure to call
     * @param arguments Vector of binding expressions for procedure arguments
     * @param yield_items Vector of (field_name, var_id) pairs for YIELD clause
     * @param optional If true, suppresses errors during procedure execution
     */
    CallProcedure(
        Procedure* procedure,
        std::vector<std::unique_ptr<BindingExpr>>&& arguments,
        std::vector<std::pair<std::string, VarId>>&& yield_items,
        bool optional = false
    );

    void _begin(Binding& parent_binding) override;

    void _reset() override;

    bool _next() override;

    void assign_nulls() override;

    void print(std::ostream& os, int indent, bool stats) const override;

private:
    Procedure* procedure;                                         // Procedure to execute
    std::vector<std::unique_ptr<BindingExpr>> arguments;          // Argument expressions
    std::vector<std::pair<std::string, VarId>> yield_items;       // YIELD field → variable mappings
    bool optional;                                                // OPTIONAL CALL flag

    Binding* parent_binding;                                      // Parent binding (for reading/writing vars)
    std::unique_ptr<ProcedureContext> context;                    // Execution context for procedure

    bool executed;                                                // Has procedure been executed?
    size_t current_result_row;                                    // Current result row index (for YIELD)
};

} // namespace GQL
