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

#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_graph_catalog.h"
#include "query/parser/expr/gql/expr_term.h"
#include "query/parser/expr/gql/expr_var.h"
#include "query/query_context.h"
#include "query/executor/binding_iter/procedure/gds_catalog_utils.h"

GdsGraphList::GdsGraphList(
    GQL::GqlGraphCatalog& catalog,
    std::vector<std::unique_ptr<GQL::Expr>>&& argument_exprs,
    std::vector<VarId>&& yield_vars
) :
    catalog_(catalog),
    argument_exprs_(std::move(argument_exprs)),
    yield_vars_(std::move(yield_vars))
{ }

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

        std::optional<std::string> graph_name_filter = std::nullopt;
        if (!argument_exprs_.empty()) {
            ObjectId oid = ObjectId::get_null();
            auto* expr = argument_exprs_[0].get();
            if (auto* term = dynamic_cast<GQL::ExprTerm*>(expr)) {
                oid = term->term;
            } else if (auto* var = dynamic_cast<GQL::ExprVar*>(expr)) {
                oid = (*parent_binding)[var->id];
            }
            if (!oid.is_null()) {
                try {
                    graph_name_filter = GQL::Conversions::unpack_string(oid);
                } catch (...) {
                    graph_name_filter = std::nullopt;
                }
            }
        }

        auto result = catalog_.list(graph_name_filter);
        entries_ = std::move(result.entries);
    }

    if (current_index_ >= entries_.size()) {
        return false;
    }

    const auto& entry = entries_[current_index_];

    for (auto var : yield_vars_) {
        const auto& name = get_query_ctx().get_var_name(var);
        parent_binding->add(var, assign_catalog_field(entry, name));
    }

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
