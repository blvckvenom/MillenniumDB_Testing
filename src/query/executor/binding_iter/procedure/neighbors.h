#pragma once

#include <vector>

#include "query/executor/binding_iter.h"
#include "query/var_id.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"

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
    //bool emit_reverse = false;
    // añade aquí el tipo de iterador sobre aristas cuando lo definas
    BptIter<3> edge_iter; // ejemplo de iterador sobre BPlusTree de 3 campos
    //const Record<3>* current_record = nullptr;
};

} // namespace Procedure
