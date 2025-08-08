//===----------------------------------------------------------------------===//
//
// This file is part of MillenniumDB
//
// MillenniumDB is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MillenniumDB is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with MillenniumDB.  If not, see <https://www.gnu.org/licenses/>.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <cassert>

#include "query/parser/op/gql/op.h"
#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_printer.h"
#include "query/var_id.h"

namespace GQL {

class OpProcedure : public Op {
public:
    /**
     * Enumeration of supported procedures.  Additional procedures can be
     * appended here as the GDS catalog grows (e.g. graph export or filter).
     */
    enum class ProcedureType : uint8_t {
        GDS_GRAPH_PROJECT,
        GDS_GRAPH_LIST,
        GDS_GRAPH_DROP,
        GDS_GRAPH_EXPORT,
        GDS_GRAPH_FILTER,
    };

    /**
     * Return a human‑readable string for the given ProcedureType.  This is
     * primarily used for debugging and printing the logical plan.
     */
    static std::string get_procedure_string(ProcedureType procedure_type);

    /**
     * Return the list of valid yield variable names for the given procedure.
     * These lists are derived from the Neo4j Graph Data‑Science manual.  They
     * allow the parser to validate YIELD clauses and provide helpful error
     * messages when an unsupported column is requested.
     */
    static std::vector<std::string>
    get_procedure_available_yield_variable_names(ProcedureType procedure_type);

    /// The kind of procedure being invoked.
    ProcedureType procedure_type;

    /// Expressions passed as arguments to the procedure.
    std::vector<std::unique_ptr<Expr>> argument_exprs;

    /// Variables to which the procedure’s output fields will be bound.
    std::vector<VarId> yield_vars;

    /**
     * Construct a new OpProcedure.
     *
     * \param procedure_type_  The enum value identifying which procedure to invoke.
     * \param argument_exprs_  List of expressions used as arguments to the procedure.
     * \param yield_vars_      List of variables corresponding to the YIELD clause.
     */
    OpProcedure(
        ProcedureType procedure_type_,
        std::vector<std::unique_ptr<Expr>>&& argument_exprs_,
        std::vector<VarId>&& yield_vars_
    ) :
        procedure_type { procedure_type_ },
        argument_exprs { std::move(argument_exprs_) },
        yield_vars { std::move(yield_vars_) }
    {
        // At least one yield variable is expected; otherwise there is nowhere to bind
        // the procedure’s outputs.  Enforce this invariant in debug builds.
        assert(!yield_vars.empty());
    }

    /**
     * Clone this operator.  Deeply clones all argument expressions and
     * duplicates the yield variable vector.
     */
    std::unique_ptr<Op> clone() const override;

    /**
     * Accept a visitor.  Each visitor must implement a visit(OpProcedure&) method
     * to handle procedure calls.  See op_visitor.h for the visitor interface.
     */
    void accept_visitor(OpVisitor& visitor) override;

    /**
     * Return all variables mentioned in this operator.  Procedure calls only
     * introduce variables via the YIELD clause; argument expressions do not bind
     * new variables.
     */
    std::set<VarId> get_all_vars() const override;

    /**
     * Pretty‑print this operator for debugging purposes.
     */
    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override;
};

} // namespace GQL
