#include "call_binding_iter.h"

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/conversions.h"

namespace GQL {

CallNamedBindingIter::CallNamedBindingIter(
    const std::string& procedure_name,
    const std::vector<std::unique_ptr<Expr>>& arguments,
    const std::vector<VarId>& yield_vars
) :
    procedure_name(procedure_name)
{
    this->yield_vars = yield_vars;
    // Clone arguments
    for (const auto& arg : arguments) {
        this->arguments.push_back(arg->clone());
    }
}

void CallNamedBindingIter::print(std::ostream& os, int indent, bool stats) const {
    os << std::string(indent, ' ') << "CallNamed(procedure: " << procedure_name;
    os << ", args: " << arguments.size();
    os << ", yields: " << yield_vars.size() << ")";
    if (stats) {
        print_generic_stats(os, indent + 2);
    }
}

void CallNamedBindingIter::_begin(Binding& parent_binding) {
    this->parent_binding = &parent_binding;
    current_result_index = 0;
    result_count = 0;
    exhausted = false;
    procedure_results.clear();
}

bool CallNamedBindingIter::_next() {
    if (exhausted) {
        return false;
    }

    if (current_result_index == 0) {
        // First call - execute the procedure
        if (procedure_name == "db.labels") {
            execute_db_labels();
        } else if (procedure_name == "db.propertyKeys") {
            execute_db_property_keys();
        } else if (procedure_name == "db.relationshipTypes") {
            execute_db_relationship_types();
        } else {
            // Unknown procedure
            exhausted = true;
            return false;
        }
    }

    if (current_result_index < procedure_results.size()) {
        // Assign result to yield variable
        if (!yield_vars.empty()) {
            ObjectId result_oid = Conversions::pack_string_simple(procedure_results[current_result_index]);
            parent_binding->add(yield_vars[0], result_oid);
        }
        current_result_index++;
        result_count++;
        return true;
    }

    exhausted = true;
    return false;
}

void CallNamedBindingIter::assign_nulls() {
    for (auto var : yield_vars) {
        parent_binding->add(var, ObjectId::get_null());
    }
}

void CallNamedBindingIter::_reset() {
    current_result_index = 0;
    result_count = 0;
    exhausted = false;
    procedure_results.clear();
}

void CallNamedBindingIter::execute_db_labels() {
    // Simple implementation - return some common labels
    procedure_results.push_back("Person");
    procedure_results.push_back("Company");
    procedure_results.push_back("Product");
}

void CallNamedBindingIter::execute_db_property_keys() {
    // Simple implementation - return some common property keys
    procedure_results.push_back("name");
    procedure_results.push_back("age");
    procedure_results.push_back("id");
}

void CallNamedBindingIter::execute_db_relationship_types() {
    // Simple implementation - return some common relationship types
    procedure_results.push_back("KNOWS");
    procedure_results.push_back("WORKS_FOR");
    procedure_results.push_back("BOUGHT");
}

CallInlineBindingIter::CallInlineBindingIter(
    std::unique_ptr<BindingIter> subquery_iter,
    const std::vector<VarId>& yield_vars
) :
    subquery_iter(std::move(subquery_iter))
{
    this->yield_vars = yield_vars;
}

void CallInlineBindingIter::print(std::ostream& os, int indent, bool stats) const {
    os << std::string(indent, ' ') << "CallInline(yields: " << yield_vars.size() << ")";
    if (stats) {
        print_generic_stats(os, indent + 2);
    }
    if (subquery_iter) {
        subquery_iter->print(os, indent + 2, stats);
    }
}

void CallInlineBindingIter::_begin(Binding& parent_binding) {
    this->parent_binding = &parent_binding;
    result_count = 0;
    exhausted = false;
    if (subquery_iter) {
        subquery_iter->begin(parent_binding);
    }
}

bool CallInlineBindingIter::_next() {
    if (exhausted) {
        return false;
    }

    if (subquery_iter && subquery_iter->next()) {
        result_count++;
        return true;
    }

    exhausted = true;
    return false;
}

void CallInlineBindingIter::assign_nulls() {
    for (auto var : yield_vars) {
        parent_binding->add(var, ObjectId::get_null());
    }
    if (subquery_iter) {
        subquery_iter->assign_nulls();
    }
}

void CallInlineBindingIter::_reset() {
    result_count = 0;
    exhausted = false;
    if (subquery_iter) {
        subquery_iter->reset();
    }
}

} // namespace GQL
