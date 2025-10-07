#pragma once

#include "query/parser/op/gql/op_visitor.h"

namespace GQL {

class BindCall : public OpVisitor {
public:
    void visit(OpQueryStatements&) override;
    void visit(OpReturn&) override;
    void visit(OpGroupBy&) override;
    void visit(OpWhere&) override;
    void visit(OpFilter&) override;
    void visit(OpLet&) override;
    void visit(OpOrderBy&) override;
    void visit(OpGraphPattern&) override;
    void visit(OpGraphPatternList&) override;
    void visit(OpLinearPattern&) override;
    void visit(OpBasicGraphPattern&) override;
    void visit(OpPathUnion&) override;
    void visit(OpRepetition&) override;
    void visit(OpCall&) override;
    void visit(OpProject&) override;
    void visit(OpUnitTable&) override;
    void visit(OpEmpty&) override;
};

} // namespace GQL
