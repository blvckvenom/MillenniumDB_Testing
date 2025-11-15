#include "call_procedure.h"

#include <iostream>

#include "query/executor/binding.h"
#include "query/exceptions.h"

using namespace GQL;

CallProcedure::CallProcedure(
    Procedure* procedure_,
    std::vector<std::unique_ptr<BindingExpr>>&& arguments_,
    std::vector<std::pair<std::string, VarId>>&& yield_items_,
    bool optional_
) :
    procedure(procedure_),
    arguments(std::move(arguments_)),
    yield_items(std::move(yield_items_)),
    optional(optional_),
    parent_binding(nullptr),
    executed(false),
    current_result_row(0)
{ }

void CallProcedure::_begin(Binding& _parent_binding)
{
    parent_binding = &_parent_binding;
    executed = false;
    current_result_row = 0;
}

void CallProcedure::_reset()
{
    executed = false;
    current_result_row = 0;
}

bool CallProcedure::_next()
{
    // Safety check: ensure _begin() was called
    if (parent_binding == nullptr) {
        throw std::runtime_error("CallProcedure::_next() called before _begin()");
    }

    // Safety check: ensure procedure is valid
    if (procedure == nullptr) {
        throw std::runtime_error("CallProcedure: procedure pointer is null");
    }

    // If already executed and returned result, we're done
    if (executed) {
        // Phase 2: If procedure has YIELD items, iterate through result rows
        if (!yield_items.empty() && context) {
            const auto& result_rows = context->get_result_rows();
            if (current_result_row < result_rows.size()) {
                // Bind YIELD variables to current result row
                const auto& row = result_rows[current_result_row];
                for (const auto& [field_name, var_id] : yield_items) {
                    auto it = row.find(field_name);
                    if (it != row.end()) {
                        parent_binding->add(var_id, it->second);
                    } else {
                        // Field not found in result row - set to NULL
                        parent_binding->add(var_id, ObjectId::get_null());
                    }
                }
                current_result_row++;
                return true;
            }
        }
        return false;
    }

    try {
        // Create procedure context
        context = std::make_unique<ProcedureContext>(*parent_binding);

        // Evaluate and store arguments
        context->arguments.reserve(arguments.size());
        for (size_t i = 0; i < arguments.size(); i++) {
            ObjectId arg_value = arguments[i]->eval(*parent_binding);
            context->arguments.push_back(arg_value);
        }

        // Execute procedure
        procedure->execute(*context);

        executed = true;

        // Phase 1: If no YIELD items, return once to indicate success
        if (yield_items.empty()) {
            return true;
        }

        // Phase 2: If YIELD items exist, start iterating through results
        const auto& result_rows = context->get_result_rows();
        if (!result_rows.empty()) {
            // Bind first result row
            const auto& row = result_rows[current_result_row];
            for (const auto& [field_name, var_id] : yield_items) {
                auto it = row.find(field_name);
                if (it != row.end()) {
                    parent_binding->add(var_id, it->second);
                } else {
                    // Field not found in result row - set to NULL
                    parent_binding->add(var_id, ObjectId::get_null());
                }
            }
            current_result_row++;
            return true;
        } else {
            // Procedure returned no results - with YIELD, this means no bindings
            return false;
        }

    } catch (const std::exception& e) {
        if (optional) {
            // ISO §15.1 OPTIONAL CALL: Return row with null fields (not empty result)
            // When procedure fails, OPTIONAL returns one record with all fields set to null
            executed = true;

            // Bind all YIELD variables to NULL
            for (const auto& [field_name, var_id] : yield_items) {
                parent_binding->add(var_id, ObjectId::get_null());
            }

            // Return true to indicate we have one result row (with null values)
            return true;
        } else {
            // Re-throw exception if not OPTIONAL
            throw;
        }
    }
}

void CallProcedure::assign_nulls()
{
    // Assign NULL to all YIELD variables
    for (const auto& [field_name, var_id] : yield_items) {
        parent_binding->add(var_id, ObjectId::get_null());
    }
}

void CallProcedure::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }

    os << std::string(indent, ' ') << "CallProcedure(";

    if (procedure) {
        os << procedure->qualified_name();
    } else {
        os << "<null procedure>";
    }

    if (optional) {
        os << " [OPTIONAL]";
    }

    if (!arguments.empty()) {
        os << ", args: " << arguments.size();
    }

    if (!yield_items.empty()) {
        os << ", yield: [";
        for (size_t i = 0; i < yield_items.size(); i++) {
            if (i > 0) os << ", ";
            os << yield_items[i].first << " AS " << yield_items[i].second;
        }
        os << "]";
    }

    os << ")\n";
}
