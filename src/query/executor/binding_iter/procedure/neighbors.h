#pragma once

#include <vector>

#include "query/executor/binding_iter.h"
#include "query/var_id.h"

namespace Procedure {

class Neighbors : public BindingIter {
public:
    Neighbors(std::vector<VarId>&& yield_vars_);

    void print(std::ostream& os, int indent, bool stats) const override;

    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;
    void assign_nulls() override;

private:
    const std::vector<VarId> yield_vars;
    Binding* parent_binding;
    // campos para cursor de aristas y manejo de “segunda pasada” irán aquí
    bool emit_reverse = false;
    // añade aquí el tipo de iterador sobre aristas cuando lo definas
};

} // namespace Procedure
