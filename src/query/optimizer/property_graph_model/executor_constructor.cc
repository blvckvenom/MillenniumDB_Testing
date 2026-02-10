#include "executor_constructor.h"

#include "query/executor/query_executor/gql/return_executor.h"
#include "query/optimizer/property_graph_model/binding_list_iter_constructor.h"
#include "query/parser/op/gql/op_call_procedure.h"
#include "query/parser/op/gql/op_return.h"

using namespace GQL;

void ExecutorConstructor::visit(OpReturn& op_return)
{
    PathBindingIterConstructor visitor;
    op_return.accept_visitor(visitor);

    std::unique_ptr<BindingIter> binding_iter = std::move(visitor.tmp_iter);

    std::vector<VarId> projection_vars;

    for (auto& var : op_return.get_expr_vars()) {
        projection_vars.push_back(var);
    }

    executor = std::make_unique<ReturnExecutor>(
        std::move(binding_iter),
        std::move(projection_vars),
        return_type
    );
}

void ExecutorConstructor::visit(OpCallProcedure& op_call)
{
    PathBindingIterConstructor visitor;
    op_call.accept_visitor(visitor);

    std::unique_ptr<BindingIter> binding_iter = std::move(visitor.tmp_iter);

    std::vector<VarId> projection_vars;

    // Extract yield variables as projection variables
    for (auto& yield_item : op_call.yield_items) {
        projection_vars.push_back(yield_item.var);
    }

    executor = std::make_unique<ReturnExecutor>(
        std::move(binding_iter),
        std::move(projection_vars),
        return_type
    );
}
