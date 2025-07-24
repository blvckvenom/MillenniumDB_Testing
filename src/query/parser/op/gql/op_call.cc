#include "query/parser/op/gql/op_call.h"

#include "query/parser/op/gql/op_visitor.h"

namespace GQL {

void OpCallNamed::accept_visitor(OpVisitor& visitor) {
    visitor.visit(*this);
}

void OpCallInline::accept_visitor(OpVisitor& visitor) {
    visitor.visit(*this);
}

} // namespace GQL
