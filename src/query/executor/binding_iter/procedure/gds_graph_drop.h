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

namespace GQL {
class GqlGraphCatalog;
}

/**
 * BindingIter implementation for the GdsGraphDrop procedure.
 * This iterator drops a graph projection from the catalog. When the graph
 * exists, its metadata is yielded as a single result row.
 *
 * Arguments:
 *   - graphName (STRING): name of the graph to drop. May be supplied as a
 *     literal or via a variable in the incoming binding.
 *   - failIfMissing (BOOLEAN, optional, default true): controls behaviour
 *     when the graph does not exist. If true, an exception is thrown. If
 *     false, no row is returned.
 *
 * Yield columns mirror @ref GQL::GqlGraphCatalog::GraphListEntry and may
 * include graphName, database, databaseLocation, configuration, nodeCount,
 * relationshipCount, schema, schemaWithOrientation, degreeDistribution,
 * density, creationTime, modificationTime, sizeInBytes and memoryUsage.
 * Both arguments accept variables.
 */
class GdsGraphDrop : public BindingIter {
public:
    /**
     * Construct a GdsGraphDrop iterator.
     * @param catalog Reference to the graph catalog
     * @param argument_exprs The procedure arguments (graphName, optional failIfMissing)
     * @param yield_vars The variables to yield (graphName, nodeCount, relationshipCount, etc.)
     */
    GdsGraphDrop(
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
    
    /// Whether the procedure has been executed
    bool executed_ = false;
    
    /// Parent binding for writing results
    Binding* parent_binding = nullptr;
};
