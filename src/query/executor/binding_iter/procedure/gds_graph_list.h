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
#include "query/parser/expr/gql/expr.h"
#include "query/var_id.h"
#include "graph_models/gql/gql_graph_catalog.h"

/**
 * BindingIter implementation for the GdsGraphList procedure.
 * This iterator lists graph projections in the catalog and yields
 * each graph's metadata as a result row.
 */
class GdsGraphList : public BindingIter {
public:
    /**
     * Construct a GdsGraphList iterator.
     * @param catalog Reference to the graph catalog
     * @param argument_exprs The procedure arguments (optional graphName filter)
     * @param yield_vars The variables to yield (graphName, nodeCount, relationshipCount, etc.)
     */
    GdsGraphList(
        GQL::GqlGraphCatalog& catalog,
        std::vector<std::unique_ptr<GQL::Expr>>&& argument_exprs,
        std::vector<VarId>&& yield_vars
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
    
    /// The list result from the catalog
    std::vector<GQL::GqlGraphCatalog::GraphListEntry> entries_;
    
    /// Current position in the entries vector
    std::size_t current_index_ = 0;
    
    /// Whether the catalog has been queried
    bool queried_ = false;
    
    /// Parent binding for writing results
    Binding* parent_binding = nullptr;
};
