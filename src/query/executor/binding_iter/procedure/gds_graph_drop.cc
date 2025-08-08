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

#include "gds_graph_drop.h"

#include "graph_models/gql/gql_graph_catalog.h"
#include "query/query_context.h"

GdsGraphDrop::GdsGraphDrop(
    GQL::GqlGraphCatalog& catalog,
    std::vector<std::unique_ptr<GQL::Expr>>&& argument_exprs,
    std::vector<VarId>&& yield_vars
) :
    catalog_(catalog),
    argument_exprs_(std::move(argument_exprs)),
    yield_vars_(std::move(yield_vars))
{
}

void GdsGraphDrop::_begin(Binding& parent_binding)
{
    this->parent_binding = &parent_binding;
    executed_ = false;
}

bool GdsGraphDrop::_next()
{
    if (executed_) {
        return false;
    }
    
    executed_ = true;
    
    // TODO: Evaluate argument expressions to get graph name and optional failIfMissing flag
    // For now, use placeholder values
    std::string graph_name = "demo_graph";
    bool fail_if_missing = true;
    
    try {
        // Drop the graph from the catalog
        auto result = catalog_.drop(graph_name, fail_if_missing);
        
        if (result.droppedGraph.has_value()) {
            // Set the yield variables with the dropped graph values
            const auto& entry = result.droppedGraph.value();
            
            // TODO: Map entry fields to yield variables based on their names
            // This requires knowledge of which yield variable corresponds to which field
            // get_query_ctx().get_var_name(yield_vars_[0]) = ObjectId::get_string(entry.graphName);
            // etc.
            
            return true;
        } else {
            // Graph was not found and failIfMissing was false
            assign_nulls();
            return true;
        }
    } catch (const std::exception& e) {
        // Graph was not found and failIfMissing was true, or other error
        // In a real implementation, this should propagate the error appropriately
        assign_nulls();
        return false;
    }
}

void GdsGraphDrop::_reset()
{
    executed_ = false;
}

void GdsGraphDrop::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "GdsGraphDrop(";
    // TODO: Print argument expressions and yield variables
    os << ")\n";
}

void GdsGraphDrop::assign_nulls()
{
    for (auto var : yield_vars_) {
        parent_binding->add(var, ObjectId::get_null());
    }
}
