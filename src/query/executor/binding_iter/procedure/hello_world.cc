#include "hello_world.h"

#include "graph_models/gql/conversions.h"

using namespace Procedure;

HelloWorld::HelloWorld(
    std::vector<std::unique_ptr<BindingExpr>>&& argument_binding_exprs_,
    std::vector<VarId>&& yield_vars_
) :
    argument_binding_exprs { std::move(argument_binding_exprs_) },
    yield_vars { std::move(yield_vars_) }
{
    assert(argument_binding_exprs.empty());
    assert(yield_vars.size() == 1);
}

void HelloWorld::_begin(Binding& parent_binding_)
{
    parent_binding = &parent_binding_;
    _reset();
}

void HelloWorld::_reset()
{
    returned = false;
    message_oid = GQL::Conversions::pack_string_simple("Hello, World!");
}

bool HelloWorld::_next()
{
    if (returned) {
        return false;
    }
    parent_binding->add(yield_vars[0], message_oid);
    returned = true;
    return true;
}

void HelloWorld::assign_nulls()
{
    for (const auto& yield_var : yield_vars) {
        parent_binding->add(yield_var, ObjectId::get_null());
    }
}

void HelloWorld::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "HelloWorld() -> (";
    os << yield_vars[0];
    for (std::size_t i = 1; 1 < yield_vars.size(); ++i) {
        os << ", " << yield_vars[i];
    }
    os << ")\n";
}
