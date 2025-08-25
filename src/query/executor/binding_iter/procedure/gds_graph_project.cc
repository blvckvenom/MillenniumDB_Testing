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
#include <unordered_set>
#include <vector>
#include <set>
#include <algorithm>
#include <cctype>

#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_graph_catalog.h"
#include "graph_models/gql/gql_value.h"
#include "graph_models/object_id.h"
#include "graph_models/gql/gql_model.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "query/parser/gql_query_parser.h"
#include "query/optimizer/property_graph_model/binding_list_iter_constructor.h"
#include "query/executor/binding.h"
#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_term.h"
#include "query/parser/expr/gql/expr_var.h"
#include "query/query_context.h"
#include "query/var_id.h"
#include "misc/logger.h"

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

// Heuristic to determine if a string looks like a GQL subquery.
bool looks_like_subquery(std::string_view s)
{
    size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '/') {
            i += 2; while (i < s.size() && s[i] != '\n') ++i; continue;
        }
        if (c == '-' && i + 1 < s.size() && s[i + 1] == '-') {
            i += 2; while (i < s.size() && s[i] != '\n') ++i; continue;
        }
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '*') {
            i += 2;
            while (i + 1 < s.size() && !(s[i] == '*' && s[i + 1] == '/')) ++i;
            if (i + 1 < s.size()) i += 2;
            continue;
        }
        break;
    }
    std::string token;
    while (i < s.size() && std::isalpha(static_cast<unsigned char>(s[i]))) {
        token.push_back(std::toupper(static_cast<unsigned char>(s[i])));
        ++i;
    }
    return token == "MATCH" || token == "WITH" || token == "CALL";
}

// Retrieve all labels for a node or edge
std::vector<std::string> get_labels(ObjectId oid, bool is_node)
{
    bool interruption = false;
    std::vector<std::string> labels;
    BptIter<2> it = is_node ?
        gql_model.node_label->get_range(&interruption, { oid.id, 0 }, { oid.id, UINT64_MAX }) :
        gql_model.edge_label->get_range(&interruption, { oid.id, 0 }, { oid.id, UINT64_MAX });
    auto row = it.next();
    while (row != nullptr) {
        ObjectId lbl_oid((*row)[1]);
        auto unmasked = lbl_oid.id & ObjectId::VALUE_MASK;
        if (is_node) {
            labels.push_back(gql_model.catalog.node_labels_str[unmasked]);
        } else {
            labels.push_back(gql_model.catalog.edge_labels_str[unmasked]);
        }
        row = it.next();
    }
    return labels;
}

// Retrieve properties for a node or edge
GQL::Value::ValueMap get_properties(ObjectId oid, bool is_node)
{
    bool interruption = false;
    GQL::Value::ValueMap props;
    BptIter<3> it = is_node ?
        gql_model.node_key_value->get_range(&interruption, { oid.id, 0, 0 }, { oid.id, UINT64_MAX, UINT64_MAX }) :
        gql_model.edge_key_value->get_range(&interruption, { oid.id, 0, 0 }, { oid.id, UINT64_MAX, UINT64_MAX });
    auto row = it.next();
    while (row != nullptr) {
        ObjectId key_oid((*row)[1]);
        ObjectId val_oid((*row)[2]);
        auto unmasked = key_oid.id & ObjectId::VALUE_MASK;
        std::string key = is_node ?
            gql_model.catalog.node_keys_str[unmasked] :
            gql_model.catalog.edge_keys_str[unmasked];
        props.emplace(key, object_id_to_value(val_oid));
        row = it.next();
    }
    return props;
}

// Filter to keep only scalar properties (string, int, double, bool)
GQL::Value::ValueMap filter_scalars(const GQL::Value::ValueMap& in)
{
    GQL::Value::ValueMap out;
    for (const auto& [k, v] : in) {
        if (v.is_string() || v.is_int() || v.is_double() || v.is_bool()) {
            out.emplace(k, v);
        }
    }
    return out;
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

        bool node_is_subq = node_proj_val.is_string() && looks_like_subquery(node_proj_val.get_string());
        bool edge_is_subq = rel_proj_val.is_string() && looks_like_subquery(rel_proj_val.get_string());

        GQL::GqlGraphCatalog::ProjectResult result;

        if (node_is_subq && edge_is_subq) {
            // --- Subquery mode ---
            using NodeIn = GQL::GqlGraphCatalog::ProjectedNodeInput;
            using EdgeIn = GQL::GqlGraphCatalog::ProjectedEdgeInput;

            std::unordered_map<std::size_t, NodeIn> node_map;

            // Compile and execute nodeQuery
            {
                auto plan = GQL::QueryParser::get_query_plan(node_proj_val.get_string());
                auto types = plan->get_var_types();
                bool found = false;
                VarId n_var = get_query_ctx().get_var("n", &found);
                if (!found || types[n_var].type != GQL::VarType::Node) {
                    throw std::runtime_error("nodeQuery debe retornar la variable 'n' de tipo nodo");
                }
                GQL::PathBindingIterConstructor ctor;
                plan->accept_visitor(ctor);
                auto iter = std::move(ctor.tmp_iter);
                Binding b(get_query_ctx().get_var_size());
                iter->begin(b);
                while (iter->next()) {
                    ObjectId oid = b[n_var];
                    auto id = static_cast<std::size_t>(oid.id);
                    if (node_map.count(id) == 0) {
                        NodeIn ni;
                        ni.originalId = id;
                        ni.labels = get_labels(oid, true);
                        ni.properties = filter_scalars(get_properties(oid, true));
                        node_map.emplace(id, std::move(ni));
                    }
                }
                logger(Category::Info) << "gdsgraphproject nodes: " << node_map.size();
            }

            std::vector<EdgeIn> edges;
            {
                auto plan = GQL::QueryParser::get_query_plan(rel_proj_val.get_string());
                auto types = plan->get_var_types();
                bool fa=false, fr=false, fb=false;
                VarId a_var = get_query_ctx().get_var("a", &fa);
                VarId r_var = get_query_ctx().get_var("r", &fr);
                VarId b_var = get_query_ctx().get_var("b", &fb);
                if (!fa || types[a_var].type != GQL::VarType::Node ||
                    !fr || types[r_var].type != GQL::VarType::Edge ||
                    !fb || types[b_var].type != GQL::VarType::Node) {
                    throw std::runtime_error("edgeQuery debe retornar 'a' (nodo), 'r' (relación) y 'b' (nodo) con esos nombres");
                }
                GQL::PathBindingIterConstructor ctor;
                plan->accept_visitor(ctor);
                auto iter = std::move(ctor.tmp_iter);
                Binding b(get_query_ctx().get_var_size());
                iter->begin(b);
                std::unordered_set<std::size_t> seen_edges;
                while (iter->next()) {
                    ObjectId a_oid = b[a_var];
                    ObjectId r_oid = b[r_var];
                    ObjectId b_oid = b[b_var];
                    auto a_id = static_cast<std::size_t>(a_oid.id);
                    auto b_id = static_cast<std::size_t>(b_oid.id);
                    if (node_map.count(a_id) == 0 || node_map.count(b_id) == 0) {
                        continue;
                    }
                    auto r_id = static_cast<std::size_t>(r_oid.id);
                    if (!seen_edges.insert(r_id).second) {
                        continue;
                    }
                    EdgeIn ei;
                    ei.sourceOriginal = a_id;
                    ei.targetOriginal = b_id;
                    auto lbls = get_labels(r_oid, false);
                    ei.type = lbls.empty() ? std::string() : lbls.front();
                    ei.properties = filter_scalars(get_properties(r_oid, false));
                    edges.push_back(std::move(ei));
                }
                logger(Category::Info) << "gdsgraphproject edges: " << edges.size();
            }

            std::vector<NodeIn> nodes;
            nodes.reserve(node_map.size());
            for (auto& [id, node] : node_map) {
                nodes.push_back(std::move(node));
            }

            result = catalog_.project(
                graph_name_val.get_string(),
                nodes,
                edges,
                configuration,
                node_proj_val.get_string(),
                rel_proj_val.get_string());
        } else {
            // --- Legacy mode ---
            result = catalog_.project(
                graph_name_val.get_string(),
                node_proj_val,
                rel_proj_val,
                configuration);
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
