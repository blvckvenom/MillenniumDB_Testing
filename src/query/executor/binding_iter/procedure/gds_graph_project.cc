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
#include <iostream>

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <set>
#include <unordered_set>

#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_graph_catalog.h"
#include "graph_models/gql/gql_value.h"
#include "graph_models/gql/gql_object_id.h"
#include "graph_models/object_id.h"
#include "query/executor/binding.h"
#include "query/optimizer/property_graph_model/binding_list_iter_constructor.h"
#include "query/parser/gql_query_parser.h"
#include "query/parser/op/gql/op_return.h"
#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_term.h"
#include "query/parser/expr/gql/expr_var.h"
#include "query/query_context.h"
#include "misc/logger.h"
#include "query/var_id.h"

// Dual execution flow:
//  * Legacy mode uses label/type projections passed as lists, maps or the
//    '*' wildcard and delegates to GqlGraphCatalog::project.
//  * Subquery mode accepts strings that look like full MATCH/WITH/CALL
//    subqueries and materialises their bindings via
//    GqlGraphCatalog::project_from_bindings.

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

    if (gen_t == ObjectId::MASK_NUMERIC || gen_t == ObjectId::MASK_INT || gen_t == ObjectId::MASK_DECIMAL
        || gen_t == ObjectId::MASK_FLOAT || gen_t == ObjectId::MASK_DOUBLE)
    {
        // Try integer first, fall back to double
        try {
            return GQL::Value(Common::Conversions::unpack_int(oid));
        } catch (const std::exception& e) {
            std::cerr << "Numeric conversion failed: " << e.what() << '\n';
            return GQL::Value(Common::Conversions::to_double(oid));
        }
    }

    // Fallback to lexical string representation for unsupported types
    return GQL::Value(GQL::Conversions::to_lexical_str(oid));
}

// Evaluate a GQL::Expr and return its value. Supports literals and variables.
GQL::Value evaluate_expr_to_value(const GQL::Expr* expr, Binding* binding)
{
    if (const auto* term = dynamic_cast<const GQL::ExprTerm*>(expr)) {
        return object_id_to_value(term->term);
    }
    if (const auto* var = dynamic_cast<const GQL::ExprVar*>(expr)) {
        if (binding == nullptr) {
            return GQL::Value();
        }
        auto oid = (*binding)[var->id];
        if (oid.is_null()) {
            return GQL::Value();
        }
        return object_id_to_value(oid);
    }
    return GQL::Value();
}

} // namespace

bool GdsGraphProject::looks_like_subquery(std::string_view s)
{
    auto starts_with_ci = [](std::string_view text, std::string_view kw) {
        return text.size() >= kw.size() && std::equal(kw.begin(), kw.end(), text.begin(),
            [](char a, char b) { return std::toupper(a) == std::toupper(b); });
    };

    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '/') {
            i += 2;
            while (i < s.size() && s[i] != '\n') {
                ++i;
            }
            continue;
        }
        if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '*') {
            i += 2;
            while (i + 1 < s.size() && !(s[i] == '*' && s[i + 1] == '/')) {
                ++i;
            }
            if (i + 1 < s.size()) {
                i += 2;
            }
            continue;
        }
        break;
    }

    auto trimmed = s.substr(i);
    return starts_with_ci(trimmed, "MATCH") || starts_with_ci(trimmed, "WITH") ||
           starts_with_ci(trimmed, "CALL");
}

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
[[maybe_unused]] static void force_gds_graph_project_instantiation()
{
    // This function forces the compiler to generate the constructor symbol
    // It will never be called, but ensures the symbol exists for linking
    if (false) {
        GQL::GqlGraphCatalog* dummy_catalog = nullptr;
        std::vector<std::unique_ptr<GQL::Expr>> dummy_exprs;
        std::vector<VarId> dummy_vars;
        GdsGraphProject dummy(*dummy_catalog, std::move(dummy_exprs), std::move(dummy_vars));
        (void) dummy; // Suppress unused variable warning
    }
}
} // namespace

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
        // Validate argument count
        if (argument_exprs_.size() < 3) {
            throw std::runtime_error(
                "gdsgraphproject requires graphName, nodeProjection and relationshipProjection arguments"
            );
        }
        if (argument_exprs_.size() > 4) {
            throw std::runtime_error("gdsgraphproject accepts at most four arguments");
        }

        // Evaluate arguments
        auto graph_name_val = evaluate_expr_to_value(argument_exprs_[0].get(), parent_binding);
        if (!graph_name_val.is_string()) {
            throw std::runtime_error("graphName argument must be a string");
        }

        auto node_proj_val = evaluate_expr_to_value(argument_exprs_[1].get(), parent_binding);
        if (!(node_proj_val.is_string() || node_proj_val.is_list() || node_proj_val.is_map())) {
            throw std::runtime_error("nodeProjection argument must be a string, list or map");
        }
        
        // Reject empty projections early to avoid undefined behavior downstream
        if (node_proj_val.is_string()) {
            const auto& s = node_proj_val.get_string();
            const bool only_ws = std::all_of(s.begin(), s.end(), [](unsigned char c){ return std::isspace(c); });
            if (s.empty() || only_ws) {
                throw std::runtime_error(
                    "nodeProjection cannot be empty; use '*' or provide a label/list/map"
                );
            }
        } else if (node_proj_val.is_list()) {
            if (node_proj_val.as_list().empty()) {
                throw std::runtime_error(
                    "nodeProjection list cannot be empty; include at least one label"
                );
            }
        } else if (node_proj_val.is_map()) {
            if (node_proj_val.as_map().empty()) {
                throw std::runtime_error(
                    "nodeProjection map cannot be empty; include at least one label entry"
                );
            }
        }

        auto rel_proj_val = evaluate_expr_to_value(argument_exprs_[2].get(), parent_binding);
        if (!(rel_proj_val.is_string() || rel_proj_val.is_list() || rel_proj_val.is_map())) {
            throw std::runtime_error("relationshipProjection argument must be a string, list or map");
        }
        
        // Same validation for relationship projection
        if (rel_proj_val.is_string()) {
            const auto& s = rel_proj_val.get_string();
            const bool only_ws = std::all_of(s.begin(), s.end(), [](unsigned char c){ return std::isspace(c); });
            if (s.empty() || only_ws) {
                throw std::runtime_error(
                    "relationshipProjection cannot be empty; use '*' or provide a type/list/map"
                );
            }
        } else if (rel_proj_val.is_list()) {
            if (rel_proj_val.as_list().empty()) {
                throw std::runtime_error(
                    "relationshipProjection list cannot be empty; include at least one type"
                );
            }
        } else if (rel_proj_val.is_map()) {
            if (rel_proj_val.as_map().empty()) {
                throw std::runtime_error(
                    "relationshipProjection map cannot be empty; include at least one type entry"
                );
            }
        }

        GQL::Map configuration;
        if (argument_exprs_.size() == 4) {
            auto config_val = evaluate_expr_to_value(argument_exprs_[3].get(), parent_binding);
            if (!config_val.is_map()) {
                throw std::runtime_error("configuration argument must be a map");
            }
            configuration = GQL::Map(config_val.as_map());
        }

        // Decide execution mode
        bool node_is_sub = node_proj_val.is_string() && looks_like_subquery(node_proj_val.get_string());
        bool rel_is_sub  = rel_proj_val.is_string() && looks_like_subquery(rel_proj_val.get_string());

        GQL::GqlGraphCatalog::ProjectResult result;
        if (node_is_sub || rel_is_sub) {
            if (!(node_is_sub && rel_is_sub)) {
                throw std::runtime_error(
                    "nodeQuery y edgeQuery deben ser ambas subconsultas o ninguna"
                );
            }

            // Subquery mode: execute provided subqueries and materialize from bindings.
            QueryContext* prev_ctx = &get_query_ctx();

            // --- Execute node subquery ---
            QueryContext node_ctx;
            node_ctx.thread_info = prev_ctx->thread_info;
            set_query_ctx(&node_ctx);

            auto node_plan = GQL::QueryParser::get_query_plan(node_proj_val.get_string());
            auto* node_return = dynamic_cast<GQL::OpReturn*>(node_plan.get());
            if (node_return == nullptr || node_return->get_expr_vars().size() != 1) {
                set_query_ctx(prev_ctx);
                throw std::runtime_error("nodeQuery debe retornar la variable 'n' de tipo nodo");
            }

            bool found_n = false;
            VarId n_var = node_ctx.get_var("n", &found_n);
            if (!found_n || node_return->get_expr_vars()[0].id != n_var.id) {
                set_query_ctx(prev_ctx);
                throw std::runtime_error("nodeQuery debe retornar la variable 'n' de tipo nodo");
            }

            GQL::PathBindingIterConstructor node_constructor;
            node_plan->accept_visitor(node_constructor);
            std::unique_ptr<BindingIter> node_iter = std::move(node_constructor.tmp_iter);

            Binding node_binding { node_ctx.get_var_size() };
            node_iter->begin(node_binding);

            std::unordered_set<std::size_t> node_set;
            std::vector<GQL::GqlGraphCatalog::OldNode> nodes;
            std::size_t read_nodes = 0;
            while (node_iter->next()) {
                ++read_nodes;
                ObjectId oid = node_binding[n_var];
                if (GQL_OID::get_generic_type(oid) != GQL_OID::GenericType::NODE) {
                    set_query_ctx(prev_ctx);
                    throw std::runtime_error("nodeQuery debe retornar la variable 'n' de tipo nodo");
                }
                std::size_t id = oid.id & ObjectId::MASK_EXTERNAL_ID;
                if (node_set.insert(id).second) {
                    nodes.push_back({id, {}});
                }
            }
            logger(Category::Info) << "read_nodes=" << read_nodes;

            // --- Execute edge subquery ---
            QueryContext edge_ctx;
            edge_ctx.thread_info = prev_ctx->thread_info;
            set_query_ctx(&edge_ctx);

            auto edge_plan = GQL::QueryParser::get_query_plan(rel_proj_val.get_string());
            auto* edge_return = dynamic_cast<GQL::OpReturn*>(edge_plan.get());
            if (edge_return == nullptr || edge_return->get_expr_vars().size() != 3) {
                set_query_ctx(prev_ctx);
                throw std::runtime_error("edgeQuery debe retornar 'a'(nodo), 'r'(relación) y 'b'(nodo) con esos nombres");
            }

            bool found_a = false, found_r = false, found_b = false;
            VarId a_var = edge_ctx.get_var("a", &found_a);
            VarId r_var = edge_ctx.get_var("r", &found_r);
            VarId b_var = edge_ctx.get_var("b", &found_b);

            std::set<VarId> expected { a_var, r_var, b_var };
            auto vars_edge = edge_return->get_expr_vars();
            if (!found_a || !found_r || !found_b || std::set<VarId>(vars_edge.begin(), vars_edge.end()) != expected) {
                set_query_ctx(prev_ctx);
                throw std::runtime_error("edgeQuery debe retornar 'a'(nodo), 'r'(relación) y 'b'(nodo) con esos nombres");
            }

            GQL::PathBindingIterConstructor edge_constructor;
            edge_plan->accept_visitor(edge_constructor);
            std::unique_ptr<BindingIter> edge_iter = std::move(edge_constructor.tmp_iter);

            Binding edge_binding { edge_ctx.get_var_size() };
            edge_iter->begin(edge_binding);

            std::vector<GQL::GqlGraphCatalog::OldEdge> edges;
            std::set<std::tuple<std::size_t,std::size_t,std::string>> edge_dedup;
            std::size_t read_edges = 0;
            while (edge_iter->next()) {
                ++read_edges;
                ObjectId a_oid = edge_binding[a_var];
                ObjectId r_oid = edge_binding[r_var];
                ObjectId b_oid = edge_binding[b_var];

                if (GQL_OID::get_generic_type(a_oid) != GQL_OID::GenericType::NODE ||
                    GQL_OID::get_generic_type(b_oid) != GQL_OID::GenericType::NODE ||
                    GQL_OID::get_generic_type(r_oid) != GQL_OID::GenericType::EDGE) {
                    set_query_ctx(prev_ctx);
                    throw std::runtime_error("edgeQuery debe retornar 'a'(nodo), 'r'(relación) y 'b'(nodo) con esos nombres");
                }

                std::size_t src = a_oid.id & ObjectId::MASK_EXTERNAL_ID;
                std::size_t dst = b_oid.id & ObjectId::MASK_EXTERNAL_ID;
                if (node_set.count(src) == 0 || node_set.count(dst) == 0) {
                    continue;
                }
                std::string type = GQL::Conversions::to_lexical_str(r_oid);
                auto key = std::make_tuple(src, dst, type);
                if (edge_dedup.insert(key).second) {
                    edges.push_back({src, dst, type});
                }
            }
            logger(Category::Info) << "read_edges=" << read_edges;
            logger(Category::Info) << "edges_after_filter=" << edges.size();
            logger(Category::Info) << "materialized_nodes=" << nodes.size();
            logger(Category::Info) << "materialized_edges=" << edges.size();

            set_query_ctx(prev_ctx);

            catalog_.project_from_bindings(graph_name_val.get_string(), nodes, edges);
            result.graphName = graph_name_val.get_string();
            result.nodeProjection = node_proj_val.to_string();
            result.relationshipProjection = rel_proj_val.to_string();
            result.nodeCount = nodes.size();
            result.relationshipCount = edges.size();
            result.projectMillis = 0;
            result.configuration = configuration.to_string();
        } else {
            // Legacy mode
            result = catalog_.project(
                graph_name_val.get_string(), node_proj_val, rel_proj_val, configuration
            );
        }

        // Map column names to values
        std::unordered_map<std::string, ObjectId> values {
            {              "graphName",              GQL::Conversions::pack_string_simple(result.graphName) },
            {         "nodeProjection",         GQL::Conversions::pack_string_simple(result.nodeProjection) },
            {              "nodeCount",                     Common::Conversions::pack_int(result.nodeCount) },
            { "relationshipProjection", GQL::Conversions::pack_string_simple(result.relationshipProjection) },
            {      "relationshipCount",             Common::Conversions::pack_int(result.relationshipCount) },
            {          "projectMillis",                 Common::Conversions::pack_int(result.projectMillis) },
            {                  "query",                                                ObjectId::get_null() },
            {          "configuration",          GQL::Conversions::pack_string_simple(result.configuration) }
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

    } catch (...) {
        assign_nulls();
        throw;
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
