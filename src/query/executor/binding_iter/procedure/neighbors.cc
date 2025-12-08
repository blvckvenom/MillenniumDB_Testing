#include "neighbors.h"

#include <ostream>

using namespace Procedure;

Neighbors::Neighbors(std::vector<VarId>&& yield_vars_) :
    yield_vars { std::move(yield_vars_) }
{
}

void Neighbors::_begin(Binding& parent_binding_)
{
    this->parent_binding = &parent_binding_;
    _reset();
}

void Neighbors::_reset()
{
    emit_reverse = false;
    // aquí inicializarás el cursor de aristas cuando uses los índices reales
}

bool Neighbors::_next()
{
    // TODO: implementar la iteración sobre aristas y emisión de (node, neighbor)
    return false;
}

void Neighbors::assign_nulls()
{
    for (const auto& var : yield_vars) {
        parent_binding->add(var, ObjectId::get_null());
    }
}

void Neighbors::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "Neighbors() -> (";
    if (!yield_vars.empty()) {
        os << yield_vars[0];
        for (std::size_t i = 1; i < yield_vars.size(); ++i) {
            os << ", " << yield_vars[i];
        }
    }
    os << ")\n";
}
