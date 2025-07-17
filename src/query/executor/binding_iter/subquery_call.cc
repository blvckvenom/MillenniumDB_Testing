#include "subquery_call.h"

void SubqueryCall::_begin(Binding& _parent_binding)
{
    parent_binding = &_parent_binding;
    child_binding = std::make_unique<Binding>(_parent_binding.size);

    for (auto var : imported_vars) {
        (*child_binding).add(var, (*parent_binding)[var]);
    }

    child_iter->begin(*child_binding);
}

void SubqueryCall::_reset()
{
    for (auto var : imported_vars) {
        (*child_binding).add(var, (*parent_binding)[var]);
    }
    child_iter->reset();
}

bool SubqueryCall::_next()
{
    if (child_iter->next()) {
        for (auto var : projection_vars) {
            (*parent_binding).add(var, (*child_binding)[var]);
        }
        return true;
    } else {
        return false;
    }
}

void SubqueryCall::assign_nulls()
{
    child_iter->assign_nulls();
    for (auto var : projection_vars) {
        (*parent_binding).add(var, (*child_binding)[var]);
    }
}

void SubqueryCall::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "SubqueryCall(";
    bool printed = false;
    if (!projection_vars.empty()) {
        printed = true;
        os << "proj_vars: ";
        bool first = true;
        for (auto v : projection_vars) {
            if (!first) os << ", ";
            first = false;
            os << v;
        }
    }
    if (!imported_vars.empty()) {
        if (printed) os << ", ";
        os << "imported: ";
        bool first = true;
        for (auto v : imported_vars) {
            if (!first) os << ", ";
            first = false;
            os << v;
        }
    }
    os << ")\n";
    child_iter->print(os, indent + 2, stats);
}
