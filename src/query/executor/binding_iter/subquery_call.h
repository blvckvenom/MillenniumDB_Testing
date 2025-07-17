#pragma once

#include <memory>
#include <vector>

#include "query/executor/binding_iter.h"

class SubqueryCall : public BindingIter {
public:
    SubqueryCall(
        std::unique_ptr<BindingIter> child_iter,
        std::vector<VarId> projection_vars,
        std::vector<VarId> imported_vars
    ) :
        projection_vars(projection_vars),
        imported_vars(imported_vars),
        child_iter(std::move(child_iter)) { }

    void _begin(Binding& parent_binding) override;
    void _reset() override;
    bool _next() override;
    void assign_nulls() override;
    void print(std::ostream& os, int indent, bool stats) const override;

    std::vector<VarId> projection_vars;
    std::vector<VarId> imported_vars;
    std::unique_ptr<BindingIter> child_iter;

private:
    Binding* parent_binding;
    std::unique_ptr<Binding> child_binding;
};
