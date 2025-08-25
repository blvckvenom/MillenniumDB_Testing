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

#include "query/executor/binding_iter.h"
#include "query/var_id.h"
#include "query/parser/expr/gql/expr.h"
#include "graph_models/gql/gql_graph_catalog.h"

/**
 * Binding iterator for the GdsGraphProject() procedure.
 *
 * The iterator operates in two modes:
 *  - Legacy mode: the node and relationship projections are simple
 *    descriptors (e.g., "*", lists or maps) and the catalog generates a
 *    synthetic projection.
 *  - Subquery mode: when both projections are strings that look like GQL
 *    subqueries starting with MATCH/WITH/CALL. The strings are compiled and
 *    executed, expecting the node query to yield a single column `n` (node)
 *    and the edge query to yield `a` (node), `r` (relationship) and `b`
    (node). The resulting bindings are materialized into the catalog.
 *
 * Arguments (literals or variables are allowed):
 *   - graphName (STRING): name of the projected graph.
 *   - nodeProjection (Value): description or subquery for nodes.
 *   - relationshipProjection (Value): description or subquery for
 *     relationships.
 *   - configuration (MAP): additional projection options.
 *
 * Yielded columns, in order:
 *   - graphName (STRING)
 *   - nodeProjection (STRING representation of the argument)
 *   - nodeCount (INTEGER)
 *   - relationshipProjection (STRING representation of the argument)
 *   - relationshipCount (INTEGER)
 *   - projectMillis (INTEGER execution time)
 *   - query (currently always NULL)
 *   - configuration (STRING representation of the argument)
 */
class GdsGraphProject : public BindingIter {
public:
    /**
     * Constructs a GdsGraphProject iterator.
     *
     * @param catalog Reference to the graph catalog for storing projected graphs
     * @param argument_exprs The procedure argument expressions
     * @param yield_vars The variables to yield in the result
     */
    explicit GdsGraphProject(
        GQL::GqlGraphCatalog& catalog,
        std::vector<std::unique_ptr<GQL::Expr>> argument_exprs,
        std::vector<VarId> yield_vars
    );

    void assign_nulls() override;
    void print(std::ostream& os, int indent, bool stats) const override;

private:
    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;
    
    /// Reference to the graph catalog
    GQL::GqlGraphCatalog& catalog_;
    
    /// The procedure arguments
    std::vector<std::unique_ptr<GQL::Expr>> argument_exprs_;
    
    /// The variables to yield in the result
    std::vector<VarId> yield_vars_;
    
    /// Whether the procedure has been executed
    bool executed_ = false;
    
    /// Parent binding for writing results
    Binding* parent_binding = nullptr;
};
