#include "neighbors.h"
#include "graph_models/gql/gql_model.h"

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
    //emit_reverse = false;
    // aquí inicializarás el cursor de aristas cuando uses los índices reales
    //current_record = nullptr;

    std::array<uint64_t, 3> min_ids = {0, 0, 0};
    std::array<uint64_t, 3> max_ids = {UINT64_MAX, UINT64_MAX, UINT64_MAX};

    edge_iter = gql_model.n1_n2_edge->get_range(
        &get_query_ctx().thread_info.interruption_requested,
        min_ids,
        max_ids
    );
}

bool Neighbors::_next()
{
    // TODO: implementar la iteración sobre aristas y emisión de (node, neighbor)
    const Record<3>* current_record = edge_iter.next();
    if (current_record) {
        parent_binding->add(yield_vars[0], ObjectId((*current_record)[0]));
        parent_binding->add(yield_vars[1], ObjectId((*current_record)[1]));
        return true;
    }
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
