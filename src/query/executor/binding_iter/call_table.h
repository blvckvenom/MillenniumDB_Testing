#pragma once

#include <memory>
#include <vector>

#include "graph_models/object_id.h"
#include "query/executor/binding.h"
#include "query/executor/binding_iter.h"
#include "query/var_id.h"

// Materializes the results of a CALL subquery into memory and exposes them as a
// simple table that can be scanned multiple times by the outer pipeline.
class CallTable : public BindingIter {
public:
    CallTable(std::unique_ptr<BindingIter> subquery_iter, std::vector<VarId> yield_vars);

    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;
    void assign_nulls() override;
    void print(std::ostream& os, int indent, bool stats) const override;

private:
    void materialize(Binding& parent_binding);

    std::unique_ptr<BindingIter> child;
    std::vector<VarId> yield_vars;

    Binding* parent_binding = nullptr;
    bool materialized = false;
    size_t row_index = 0;

    std::vector<std::vector<ObjectId>> rows;
};
