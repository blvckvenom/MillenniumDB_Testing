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

#include "gds_graph_project.h"

#include <memory>
#include <vector>
#include <stdexcept>
#include <unordered_map>

#include "graph_models/gql/gql_graph_catalog.h"
#include "graph_models/gql/gql_value.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/common/conversions.h"
#include "graph_models/object_id.h"
#include "query/executor/binding.h"
#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_term.h"
#include "query/var_id.h"
#include "query/query_context.h"

namespace {

// Convert an ObjectId representing a literal into a GQL::Value. This handles
// strings, booleans, numbers, lists and maps (encoded as lists of key/value
// pairs).
GQL::Value object_id_to_value(ObjectId oid)
{
    const auto gen_t = oid.id & ObjectId::GENERIC_TYPE_MASK;

    if (gen_t == ObjectId::MASK_STRING) {
        return GQL::Value(GQL::Conversions::unpack_string(oid));
    }

    if (gen_t == ObjectId::MASK_BOOL) {
        return GQL::Value(oid.is_true());
    }

    if (gen_t == ObjectId::MASK_LIST) {
        std::vector<ObjectId> elements;
        GQL::Conversions::unpack_list(oid, elements);

        // Determine if this list encodes a map
        bool is_map = true;
        for (const auto& el : elements) {
            if ((el.id & ObjectId::GENERIC_TYPE_MASK) != ObjectId::MASK_LIST) {
                is_map = false;
                break;
            }
            std::vector<ObjectId> pair;
            GQL::Conversions::unpack_list(el, pair);
            if (pair.size() != 2 || (pair[0].id & ObjectId::GENERIC_TYPE_MASK) != ObjectId::MASK_STRING) {
                is_map = false;
                break;
            }
        }

        if (is_map) {
            GQL::Value::ValueMap m;
            for (const auto& el : elements) {
                std::vector<ObjectId> pair;
                GQL::Conversions::unpack_list(el, pair);
                std::string key = GQL::Conversions::unpack_string(pair[0]);
                m.emplace(key, object_id_to_value(pair[1]));
            }
            return GQL::Value(m);
        }

        GQL::Value::ValueList list;
        list.reserve(elements.size());
        for (const auto& el : elements) {
            list.emplace_back(object_id_to_value(el));
        }
        return GQL::Value(list);
    }

    if (gen_t == ObjectId::MASK_NUMERIC || gen_t == ObjectId::MASK_INT
        || gen_t == ObjectId::MASK_DECIMAL || gen_t == ObjectId::MASK_FLOAT
        || gen_t == ObjectId::MASK_DOUBLE)
    {
        // Try integer first, fall back to double
        try {
            return GQL::Value(Common::Conversions::unpack_int(oid));
        } catch (...) {
            return GQL::Value(Common::Conversions::to_double(oid));
        }
    }

    // Fallback to lexical string representation for unsupported types
    return GQL::Value(GQL::Conversions::to_lexical_str(oid));
}

// Evaluate a GQL::Expr expected to be a literal and return its value.
GQL::Value evaluate_expr_to_value(const GQL::Expr* expr)
{
    if (const auto* term = dynamic_cast<const GQL::ExprTerm*>(expr)) {
        return object_id_to_value(term->term);
    }
    return GQL::Value();
}

} // namespace

// Force symbol visibility
#ifdef __GNUC__
#pragma GCC visibility push(default)
#endif

// Explicit constructor implementation with full type qualification
GdsGraphProject::GdsGraphProject(
    GQL::GqlGraphCatalog& catalog,
    std::vector<std::unique_ptr<GQL::Expr>> argument_exprs,
    std::vector<VarId> yield_vars
) :
    catalog_(catalog),
    argument_exprs_(std::move(argument_exprs)),
    yield_vars_(std::move(yield_vars)),
    executed_(false),
    parent_binding(nullptr)
{
    // Constructor body - ensure it's not empty for linking
}

// Force instantiation of constructor to ensure symbol is generated
namespace {
    [[maybe_unused]] static void force_gds_graph_project_instantiation() {
        // This function forces the compiler to generate the constructor symbol
        // It will never be called, but ensures the symbol exists for linking
        if (false) {
            GQL::GqlGraphCatalog* dummy_catalog = nullptr;
            std::vector<std::unique_ptr<GQL::Expr>> dummy_exprs;
            std::vector<VarId> dummy_vars;
            GdsGraphProject dummy(*dummy_catalog, std::move(dummy_exprs), std::move(dummy_vars));
            (void)dummy; // Suppress unused variable warning
        }
    }
}

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

void GdsGraphProject::_begin(Binding& parent_binding)
{
    this->parent_binding = &parent_binding;
    executed_ = false;
}

bool GdsGraphProject::_next()
{
    if (executed_) {
        return false;
    }

    executed_ = true;

    try {
        // Evaluate arguments
        auto graph_name_val = evaluate_expr_to_value(argument_exprs_[0].get());
        if (!graph_name_val.is_string()) {
            assign_nulls();
            return false;
        }

        auto node_proj_val = evaluate_expr_to_value(argument_exprs_[1].get());
        auto rel_proj_val  = evaluate_expr_to_value(argument_exprs_[2].get());
        auto config_val    = evaluate_expr_to_value(argument_exprs_[3].get());

        GQL::Map configuration;
        if (config_val.is_map()) {
            configuration = GQL::Map(config_val.as_map());
        }

        // Invoke catalog
        auto result = catalog_.project(
            graph_name_val.get_string(),
            node_proj_val,
            rel_proj_val,
            configuration);

        // Map column names to values
        std::unordered_map<std::string, ObjectId> values {
            {"graphName", GQL::Conversions::pack_string_simple(result.graphName)},
            {"nodeProjection", GQL::Conversions::pack_string_simple(result.nodeProjection)},
            {"nodeCount", Common::Conversions::pack_int(result.nodeCount)},
            {"relationshipProjection", GQL::Conversions::pack_string_simple(result.relationshipProjection)},
            {"relationshipCount", Common::Conversions::pack_int(result.relationshipCount)},
            {"projectMillis", Common::Conversions::pack_int(result.projectMillis)},
            {"query", ObjectId::get_null()},
            {"configuration", GQL::Conversions::pack_string_simple(result.configuration)}
        };

        // Assign only requested yield variables
        for (auto var : yield_vars_) {
            const auto& name = get_query_ctx().get_var_name(var);
            auto it = values.find(name);
            if (it != values.end()) {
                parent_binding->add(var, it->second);
            } else {
                parent_binding->add(var, ObjectId::get_null());
            }
        }

        return true;

    } catch (const std::exception&) {
        assign_nulls();
        return false;
    }
}

void GdsGraphProject::_reset()
{
    executed_ = false;
}

void GdsGraphProject::assign_nulls()
{
    for (auto var : yield_vars_) {
        parent_binding->add(var, ObjectId::get_null());
    }
}

void GdsGraphProject::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "GdsGraphProject(";
    // TODO: Print argument expressions and yield variables
    os << ")\n";
}
