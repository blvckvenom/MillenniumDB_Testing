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
#include <iostream>

#include <chrono>
#include <stdexcept>
#include <unordered_map>

#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_graph_catalog.h"
#include "query/parser/expr/gql/expr_term.h"
#include "query/parser/expr/gql/expr_var.h"
#include "query/parser/op/gql/op_procedure.h"
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

    auto eval_argument = [&](std::size_t idx) -> ObjectId {
        auto* expr = argument_exprs_[idx].get();
        if (auto* var = dynamic_cast<GQL::ExprVar*>(expr)) {
            return (*parent_binding)[var->id];
        } else if (auto* term = dynamic_cast<GQL::ExprTerm*>(expr)) {
            return term->term;
        }
        throw std::runtime_error("Unsupported argument expression type");
    };

    // Evaluate graph name
    const auto graph_name_oid = eval_argument(0);
    const std::string graph_name = GQL::Conversions::unpack_string(graph_name_oid);

    // Evaluate optional failIfMissing flag (default true)
    bool fail_if_missing = true;
    if (argument_exprs_.size() > 1) {
        const auto fail_oid = eval_argument(1);
        const auto bool_oid = GQL::Conversions::to_boolean(fail_oid);
        if (bool_oid.is_null()) {
            throw std::runtime_error("failIfMissing argument could not be converted to boolean");
        }
        fail_if_missing = bool_oid.is_true();
    }

    try {
        // Drop the graph from the catalog
        auto result = catalog_.drop(graph_name, fail_if_missing);

        if (result.droppedGraph.has_value()) {
            const auto& entry = result.droppedGraph.value();

            std::unordered_map<std::string, ObjectId> row{
                { "graphName", GQL::Conversions::pack_string_simple(entry.graphName) },
                { "database", GQL::Conversions::pack_string_simple(entry.database) },
                { "databaseLocation", GQL::Conversions::pack_string_simple(entry.databaseLocation) },
                { "configuration", GQL::Conversions::pack_string_simple(entry.configuration) },
                { "nodeCount", Common::Conversions::pack_int(static_cast<int64_t>(entry.nodeCount)) },
                { "relationshipCount", Common::Conversions::pack_int(static_cast<int64_t>(entry.relationshipCount)) },
                { "schema", GQL::Conversions::pack_string_simple(entry.schema) },
                { "schemaWithOrientation", GQL::Conversions::pack_string_simple(entry.schemaWithOrientation) },
                { "degreeDistribution", GQL::Conversions::pack_string_simple(entry.degreeDistribution) },
                { "density", Common::Conversions::pack_float(static_cast<float>(entry.density)) },
                { "creationTime", Common::Conversions::pack_int(static_cast<int64_t>(
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                         entry.creationTime.time_since_epoch()).count())) },
                { "modificationTime", Common::Conversions::pack_int(static_cast<int64_t>(
                                        std::chrono::duration_cast<std::chrono::milliseconds>(
                                            entry.modificationTime.time_since_epoch()).count())) },
                { "sizeInBytes", Common::Conversions::pack_int(static_cast<int64_t>(entry.sizeInBytes)) },
                { "memoryUsage", GQL::Conversions::pack_string_simple(entry.memoryUsage) }
            };

            auto available_names = GQL::OpProcedure::get_procedure_available_yield_variable_names(
                GQL::OpProcedure::ProcedureType::GDS_GRAPH_DROP);

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

            return true;
        }

        // Graph was not found and failIfMissing was false
        assign_nulls();
        return true;
    } catch (const std::exception& e) {
        assign_nulls();
        std::cerr << "Error dropping graph: " << e.what() << '\n';
        throw;
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
