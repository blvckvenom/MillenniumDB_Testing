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
#include <chrono>
#include <unordered_map>
#include "query/parser/op/gql/op_procedure.h"

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

    std::unordered_map<std::string, ObjectId> row{
        { "graphName", GQL::Conversions::pack_string_simple(entry.graphName) },
        { "database", GQL::Conversions::pack_string_simple(entry.database) },
        { "databaseLocation", GQL::Conversions::pack_string_simple(entry.databaseLocation) },
        { "configuration", GQL::Conversions::pack_string_simple(entry.configuration) },
        { "schema", GQL::Conversions::pack_string_simple(entry.schema) },
        { "schemaWithOrientation", GQL::Conversions::pack_string_simple(entry.schemaWithOrientation) },
        { "degreeDistribution", GQL::Conversions::pack_string_simple(entry.degreeDistribution) },
        { "memoryUsage", GQL::Conversions::pack_string_simple(entry.memoryUsage) },
        { "nodeCount", Common::Conversions::pack_int(static_cast<int64_t>(entry.nodeCount)) },
        { "relationshipCount", Common::Conversions::pack_int(static_cast<int64_t>(entry.relationshipCount)) },
        { "density", Common::Conversions::pack_double(entry.density) },
        { "creationTime", Common::Conversions::pack_int(static_cast<int64_t>(
                              std::chrono::duration_cast<std::chrono::milliseconds>(
                                  entry.creationTime.time_since_epoch()).count())) },
        { "modificationTime", Common::Conversions::pack_int(static_cast<int64_t>(
                                 std::chrono::duration_cast<std::chrono::milliseconds>(
                                     entry.modificationTime.time_since_epoch()).count())) },
        { "sizeInBytes", Common::Conversions::pack_int(static_cast<int64_t>(entry.sizeInBytes)) }
    };

    auto available_names = GQL::OpProcedure::get_procedure_available_yield_variable_names(
        GQL::OpProcedure::ProcedureType::GDS_GRAPH_LIST);

    if (yield_vars_.size() == available_names.size()) {
        for (size_t i = 0; i < yield_vars_.size(); ++i) {
            const auto& col = available_names[i];
            auto it = row.find(col);
            if (it != row.end()) {
                parent_binding->add(yield_vars_[i], it->second);
            } else {
                parent_binding->add(yield_vars_[i], ObjectId::get_null());
            }
        }
    } else {
        for (auto var : yield_vars_) {
            const auto& name = get_query_ctx().get_var_name(var);
            auto it = row.find(name);
            if (it != row.end()) {
                parent_binding->add(var, it->second);
            } else {
                parent_binding->add(var, ObjectId::get_null());
            }
        }
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
