#include "call_table.h"

#include <utility>

#include "graph_models/object_id.h"

CallTable::CallTable(std::unique_ptr<BindingIter> subquery_iter, std::vector<VarId> yield_vars) :
    child(std::move(subquery_iter)),
    yield_vars(std::move(yield_vars))
{
}

void CallTable::_begin(Binding& parent)
{
    parent_binding = &parent;
    row_index = 0;

    if (!materialized) {
        materialize(parent);
    }
}

void CallTable::_reset()
{
    row_index = 0;
}

bool CallTable::_next()
{
    if (row_index >= rows.size()) {
        return false;
    }

    const auto& row = rows[row_index++];
    for (size_t i = 0; i < yield_vars.size(); ++i) {
        parent_binding->add(yield_vars[i], row[i]);
    }
    return true;
}

void CallTable::assign_nulls()
{
    if (parent_binding == nullptr) {
        return;
    }
    for (auto var : yield_vars) {
        parent_binding->add(var, ObjectId::get_null());
    }
}

void CallTable::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "CallTable(yield=[";
    for (size_t i = 0; i < yield_vars.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << yield_vars[i];
    }
    os << "], rows=" << rows.size() << ")\n";
    if (child) {
        child->print(os, indent + 2, stats);
    }
}

void CallTable::materialize(Binding& parent)
{
    if (!child) {
        materialized = true;
        return;
    }

    Binding subquery_binding(parent.size);
    subquery_binding.add_all(parent);

    child->begin(subquery_binding);

    rows.clear();
    while (child->next()) {
        std::vector<ObjectId> row;
        row.reserve(yield_vars.size());
        for (auto var : yield_vars) {
            row.push_back(subquery_binding[var]);
        }
        rows.push_back(std::move(row));
    }

    materialized = true;
}
