#pragma once

#include <tuple>
#include <vector>

#include "query/executor/binding_iter.h"
#include "query/var_id.h"

namespace Procedure {

class Jaccard : public BindingIter {
public:
    Jaccard(std::vector<VarId>&& yield_vars_);

    void print(std::ostream& os, int indent, bool stats) const override;

    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;
    void assign_nulls() override;

private:
    const std::vector<VarId> yield_vars;

    Binding* parent_binding = nullptr;
    std::vector<std::tuple<ObjectId, ObjectId, ObjectId>> results;
    std::size_t cursor = 0;
};

} // namespace Procedure
