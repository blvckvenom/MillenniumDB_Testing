#include "call_binding_iter.h"

#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"

#include <algorithm>
#include <functional>
#include <unordered_set>

namespace {

using ResultRow = std::unordered_map<std::string, ObjectId>;
using ResultSet = std::vector<ResultRow>;

ResultSet proc_db_labels()
{
    ResultSet results;
    results.reserve(gql_model.catalog.node_labels_str.size());
    for (const auto& lbl : gql_model.catalog.node_labels_str) {
        results.push_back({{"label", GQL::Conversions::pack_string_simple(lbl)}});
    }
    return results;
}

ResultSet proc_db_property_keys()
{
    std::unordered_set<std::string> keys;
    keys.insert(gql_model.catalog.node_keys_str.begin(), gql_model.catalog.node_keys_str.end());
    keys.insert(gql_model.catalog.edge_keys_str.begin(), gql_model.catalog.edge_keys_str.end());

    ResultSet results;
    results.reserve(keys.size());
    for (const auto& key : keys) {
        results.push_back({{"propertyKey", GQL::Conversions::pack_string_simple(key)}});
    }
    return results;
}

ResultSet proc_db_relationship_types()
{
    ResultSet results;
    results.reserve(gql_model.catalog.edge_labels_str.size());
    for (const auto& lbl : gql_model.catalog.edge_labels_str) {
        results.push_back({{"relationshipType", GQL::Conversions::pack_string_simple(lbl)}});
    }
    return results;
}

struct ProcedureInfo {
    std::vector<std::string> fields;
    std::function<ResultSet()> impl;
};

const std::unordered_map<std::string, ProcedureInfo>& get_procedure_registry()
{
    static const std::unordered_map<std::string, ProcedureInfo> registry = {
        {"db.labels", {{"label"}, proc_db_labels}},
        {"db.propertyKeys", {{"propertyKey"}, proc_db_property_keys}},
        {"db.relationshipTypes", {{"relationshipType"}, proc_db_relationship_types}}
    };
    return registry;
}

} // namespace

namespace GQL {

CallNamedBindingIter::CallNamedBindingIter(
    const std::string& procedure_name,
    const std::vector<std::unique_ptr<Expr>>& arguments,
    const std::vector<VarId>& yield_vars,
    const std::vector<std::string>& yield_fields,
    bool is_optional
) :
    procedure_name(procedure_name)
{
    this->yield_vars = yield_vars;
    this->yield_fields = yield_fields;
    this->is_optional = is_optional;
    // Clone arguments
    for (const auto& arg : arguments) {
        this->arguments.push_back(arg->clone());
    }
    returned_null_row = false;
}

void CallNamedBindingIter::print(std::ostream& os, int indent, bool stats) const
{
    os << std::string(indent, ' ') << "CallNamed(procedure: " << procedure_name;
    os << ", args: " << arguments.size();
    os << ", yields: " << yield_vars.size() << ")";
    if (stats) {
        print_generic_stats(os, indent + 2);
    }
}

void CallNamedBindingIter::_begin(Binding& parent_binding)
{
    this->parent_binding = &parent_binding;
    current_result_index = 0;
    result_count = 0;
    exhausted = false;
    procedure_results.clear();
    executed = false;
    returned_null_row = false;
}

bool CallNamedBindingIter::_next()
{
    if (exhausted) {
        return false;
    }

    if (!executed) {
        executed = true;

        auto& registry = get_procedure_registry();
        auto it = registry.find(procedure_name);
        if (it == registry.end()) {
            exhausted = true;
            return false;
        }

        const auto& info = it->second;

        for (const auto& field : yield_fields) {
            if (std::find(info.fields.begin(), info.fields.end(), field) == info.fields.end()) {
                throw QueryExecutionException(
                    "Procedure `" + procedure_name + "` does not yield field `" + field + "`");
            }
        }

        procedure_results = info.impl();

        if (procedure_results.empty() && is_optional) {
            if (!returned_null_row) {
                for (auto var : yield_vars) {
                    parent_binding->add(var, ObjectId::get_null());
                }
                returned_null_row = true;
                exhausted = true;
                result_count++;
                return true;
            }
            exhausted = true;
            return false;
        }
    }

    if (current_result_index < procedure_results.size()) {
        const auto& result = procedure_results[current_result_index];
        for (size_t i = 0; i < yield_vars.size(); ++i) {
            ObjectId value = ObjectId::get_null();
            if (i < yield_fields.size()) {
                auto it = result.find(yield_fields[i]);
                if (it != result.end()) {
                    value = it->second;
                }
            } else if (result.size() == 1) {
                value = result.begin()->second;
            }
            parent_binding->add(yield_vars[i], value);
        }
        current_result_index++;
        result_count++;
        return true;
    }

    if (is_optional && !returned_null_row && result_count == 0) {
        for (auto var : yield_vars) {
            parent_binding->add(var, ObjectId::get_null());
        }
        returned_null_row = true;
        exhausted = true;
        result_count++;
        return true;
    }

    exhausted = true;
    return false;
}

void CallNamedBindingIter::assign_nulls()
{
    for (auto var : yield_vars) {
        parent_binding->add(var, ObjectId::get_null());
    }
}

void CallNamedBindingIter::_reset()
{
    current_result_index = 0;
    result_count = 0;
    exhausted = false;
    procedure_results.clear();
    executed = false;
    returned_null_row = false;
}

CallInlineBindingIter::CallInlineBindingIter(
    std::unique_ptr<BindingIter> subquery_iter,
    const std::vector<VarId>& yield_vars,
    bool is_optional
) :
    subquery_iter(std::move(subquery_iter))
{
    this->yield_vars = yield_vars;
    this->is_optional = is_optional;
    returned_null_row = false;
}

void CallInlineBindingIter::print(std::ostream& os, int indent, bool stats) const
{
    os << std::string(indent, ' ') << "CallInline(yields: " << yield_vars.size() << ")";
    if (stats) {
        print_generic_stats(os, indent + 2);
    }
    if (subquery_iter) {
        subquery_iter->print(os, indent + 2, stats);
    }
}

void CallInlineBindingIter::_begin(Binding& parent_binding)
{
    this->parent_binding = &parent_binding;
    result_count = 0;
    exhausted = false;
    returned_null_row = false;
    if (subquery_iter) {
        subquery_iter->begin(parent_binding);
    }
}

bool CallInlineBindingIter::_next()
{
    if (exhausted) {
        return false;
    }

    if (subquery_iter && subquery_iter->next()) {
        result_count++;
        return true;
    }

    if (is_optional && !returned_null_row && result_count == 0) {
        for (auto var : yield_vars) {
            parent_binding->add(var, ObjectId::get_null());
        }
        if (subquery_iter) {
            subquery_iter->assign_nulls();
        }
        returned_null_row = true;
        exhausted = true;
        result_count++;
        return true;
    }

    exhausted = true;
    return false;
}

void CallInlineBindingIter::assign_nulls()
{
    for (auto var : yield_vars) {
        parent_binding->add(var, ObjectId::get_null());
    }
    if (subquery_iter) {
        subquery_iter->assign_nulls();
    }
}

void CallInlineBindingIter::_reset()
{
    result_count = 0;
    exhausted = false;
    returned_null_row = false;
    if (subquery_iter) {
        subquery_iter->reset();
    }
}

} // namespace GQL
