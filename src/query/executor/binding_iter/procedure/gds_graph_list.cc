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

#include "gds_graph_list.h"

#include "graph_models/gql/gql_graph_catalog.h"
#include "query/query_context.h"

GdsGraphList::GdsGraphList(
    GQL::GqlGraphCatalog& catalog,
    std::vector<std::unique_ptr<GQL::Expr>>&& argument_exprs,
    std::vector<VarId>&& yield_vars
) :
    catalog_(catalog),
    argument_exprs_(std::move(argument_exprs)),
    yield_vars_(std::move(yield_vars))
{
}

void GdsGraphList::_begin(Binding& parent_binding)
{
    this->parent_binding = &parent_binding;
    queried_ = false;
    current_index_ = 0;
}

bool GdsGraphList::_next()
{
    if (!queried_) {
        queried_ = true;
        current_index_ = 0;
        
        // TODO: Evaluate argument expressions to get optional graph name filter
        std::optional<std::string> graph_name_filter = std::nullopt;
        
        // Query the catalog
        auto result = catalog_.list(graph_name_filter);
        entries_ = std::move(result.entries);
    }
    
    if (current_index_ >= entries_.size()) {
        return false;
    }
    
    // Set the yield variables with the current entry values
    const auto& entry = entries_[current_index_];
    
    // TODO: Map entry fields to yield variables based on their names
    // This requires knowledge of which yield variable corresponds to which field
    // For now, we'll set placeholder values
    
    // Example mapping (this would need to be dynamic based on actual yield variable names):
    // get_query_ctx().get_var_name(yield_vars_[0]) = ObjectId::get_string(entry.graphName);
    // get_query_ctx().get_var_name(yield_vars_[1]) = ObjectId::get_int64(entry.nodeCount);
    // etc.
    
    ++current_index_;
    return true;
}

void GdsGraphList::_reset()
{
    queried_ = false;
    current_index_ = 0;
    entries_.clear();
}

void GdsGraphList::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "GdsGraphList(";
    // TODO: Print argument expressions and yield variables
    os << ")\n";
}

void GdsGraphList::assign_nulls()
{
    for (auto var : yield_vars_) {
        parent_binding->add(var, ObjectId::get_null());
    }
}
