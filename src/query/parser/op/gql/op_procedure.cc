

#include "query/parser/op/gql/op_procedure.h"

#include <algorithm>

#include "query/parser/op/gql/op_visitor.h"

namespace GQL {

// Return a human readable string for a given procedure type.  These strings
// reflect the fully qualified procedure names used in Neo4j's Graph Data Science
// library.  They are primarily used for debugging and to help users
// understand which procedure is being invoked.
std::string OpProcedure::get_procedure_string(ProcedureType procedure_type)
{
    switch (procedure_type) {
    case ProcedureType::GDS_GRAPH_PROJECT:
        return "project";
    case ProcedureType::GDS_GRAPH_LIST:
        return "list";
    case ProcedureType::GDS_GRAPH_DROP:
        return "drop";
    case ProcedureType::GDS_GRAPH_EXPORT:
        return "export";
    case ProcedureType::GDS_GRAPH_FILTER:
        return "filter";
    default:
        return "UNKNOWN_PROCEDURE";
    }
}

// Return a list of valid yield column names for the specified procedure.  The
// lists are derived from the Neo4j Graph Data‑Science manual【45300608474768†L395-L407】【359628130627925†L370-L518】.  If a procedure
// is not yet fully supported in MillenniumDB, the list may contain the
// expected columns as documented but the execution engine may ignore some of
// them.  These names are used by the parser to validate YIELD clauses.
std::vector<std::string>
OpProcedure::get_procedure_available_yield_variable_names(ProcedureType procedure_type)
{
    switch (procedure_type) {
    case ProcedureType::GDS_GRAPH_PROJECT:
        // GdsGraphProject yields information about the newly created in‑memory graph.
        // According to the Neo4j GDS reference, the following columns may be
        // returned【45300608474768†L395-L407】: graphName, nodeProjection, nodeCount,
        // relationshipProjection, relationshipCount, projectMillis, query,
        // configuration.  Both `nodeProjection` and `relationshipProjection`
        // describe the projection specifications used to create the graph.
        return {
            "graphName",
            "nodeProjection",
            "nodeCount",
            "relationshipProjection",
            "relationshipCount",
            "projectMillis",
            "query",
            "configuration"
        };

    case ProcedureType::GDS_GRAPH_LIST:
        // GdsGraphList returns statistics about projected graphs stored in the
        // catalog.  The Neo4j manual lists a wide range of metrics【359628130627925†L370-L518】,
        // including schema and degree distribution.  We include the most common
        // fields here; additional fields can be added later as support grows.
        return {
            "graphName",
            "database",
            "databaseLocation",
            "configuration",
            "nodeCount",
            "relationshipCount",
            "schema",
            "schemaWithOrientation",
            "degreeDistribution",
            "density",
            "creationTime",
            "modificationTime",
            "sizeInBytes",
            "memoryUsage"
        };

    case ProcedureType::GDS_GRAPH_DROP:
        // drop returns similar statistics to list for the
        // graph being removed【977493777151698†L380-L520】.  We mirror the list procedure’s
        // yield columns for consistency.
        return {
            "graphName",
            "database",
            "databaseLocation",
            "configuration",
            "nodeCount",
            "relationshipCount",
            "schema",
            "schemaWithOrientation",
            "density",
            "creationTime",
            "modificationTime",
            "sizeInBytes",
            "memoryUsage"
        };

    case ProcedureType::GDS_GRAPH_EXPORT:
        // GdsGraphExport writes a projection to a new database.  The
        // procedure yields at least the graph name and counts; additional
        // timing information may be returned.  These names are speculative
        // based on the documentation of Neo4j GDS 2.x and may need
        // adjustment when full support is implemented.
        return {
            "graphName",
            "nodeCount",
            "relationshipCount",
            "configuration",
            "projectMillis",
            "loadMillis",
            "totalMillis"
        };

    case ProcedureType::GDS_GRAPH_FILTER:
        // GdsGraphFilter creates a new graph by filtering nodes and/or
        // relationships from an existing projection.  The yield columns are
        // similar to GdsGraphProject with an additional sourceGraphName.
        return {
            "graphName",
            "sourceGraphName",
            "nodeCount",
            "relationshipCount",
            "filterMillis",
            "configuration"
        };

    default:
        throw NotSupportedException(
            "OpProcedure::get_procedure_available_yield_variable_names: Unhandled procedure type"
        );
    }
}

// Deep copy this procedure operator.  Each argument expression is cloned
// independently to avoid sharing AST nodes between plan instances.  Yield
// variables are copied by value.
std::unique_ptr<Op> OpProcedure::clone() const
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

// Dispatch to the appropriate visit method implemented by the caller.
void OpProcedure::accept_visitor(OpVisitor& visitor)
{
    visitor.visit(*this);
}

// Return all variables introduced by this operator.  Procedure calls only
// bind variables via the YIELD clause; argument expressions may reference
// existing variables but they do not define new ones.  Therefore, the set of
// variables returned consists solely of the yield variables.
std::set<VarId> OpProcedure::get_all_vars() const
{
    return std::set<VarId>(yield_vars.begin(), yield_vars.end());
}

// Pretty‑print this operator.  It displays the procedure name, argument
// expressions and yield variables in a compact form similar to the MQL
// OpCall implementation.  Indentation is applied to align nested operators
// consistently.
std::ostream& OpProcedure::print_to_ostream(std::ostream& os, int indent) const
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
    os << yield_vars[0];
    for (std::size_t i = 1; i < yield_vars.size(); ++i) {
        os << ", " << yield_vars[i];
    }
    os << ")\n";
    return os;
}

} // namespace GQL