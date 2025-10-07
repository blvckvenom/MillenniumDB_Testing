#include "bind_call.h"

#include <algorithm>
#include <map>
#include <optional>
#include <vector>

#include "query/exceptions.h"
#include "query/parser/expr/gql/expr_var.h"
#include "query/parser/op/gql/ops.h"
#include "query/query_context.h"

using namespace GQL;

void BindCall::visit(OpQueryStatements& op_statements)
{
    for (auto& op : op_statements.ops) {
        op->accept_visitor(*this);
    }
}

void BindCall::visit(OpReturn& op_return)
{
    op_return.op->accept_visitor(*this);
    if (op_return.op_order_by != nullptr) {
        op_return.op_order_by->accept_visitor(*this);
    }
}

void BindCall::visit(OpGroupBy& op_group_by)
{
    op_group_by.op->accept_visitor(*this);
}

void BindCall::visit(OpWhere& op_where)
{
    op_where.op->accept_visitor(*this);
}

void BindCall::visit(OpFilter&)
{
    // nothing to do
}

void BindCall::visit(OpLet&)
{
    // nothing to do
}

void BindCall::visit(OpOrderBy&)
{
    // nothing to do
}

void BindCall::visit(OpGraphPattern& op_graph_pattern)
{
    op_graph_pattern.op->accept_visitor(*this);
}

void BindCall::visit(OpGraphPatternList& op_graph_pattern_list)
{
    for (auto& pattern : op_graph_pattern_list.patterns) {
        pattern->accept_visitor(*this);
    }
}

void BindCall::visit(OpLinearPattern& op_linear_pattern)
{
    for (auto& pattern : op_linear_pattern.patterns) {
        pattern->accept_visitor(*this);
    }
}

void BindCall::visit(OpBasicGraphPattern& op_basic_graph_pattern)
{
    for (auto& pattern : op_basic_graph_pattern.patterns) {
        pattern->accept_visitor(*this);
    }
}

void BindCall::visit(OpPathUnion& op_path_union)
{
    for (auto& pattern : op_path_union.op_list) {
        pattern->accept_visitor(*this);
    }
}

void BindCall::visit(OpRepetition& op_repetition)
{
    op_repetition.op->accept_visitor(*this);
}

void BindCall::visit(OpCall& op_call)
{
    if (op_call.subquery == nullptr) {
        throw QuerySemanticException("CALL subquery must end with RETURN");
    }

    auto* subquery_return = dynamic_cast<OpReturn*>(op_call.subquery.get());
    if (subquery_return == nullptr) {
        throw QuerySemanticException("CALL subquery must end with RETURN");
    }

    subquery_return->op->accept_visitor(*this);
    if (subquery_return->op_order_by != nullptr) {
        subquery_return->op_order_by->accept_visitor(*this);
    }

    auto inner_types = subquery_return->op->get_var_types();

    std::vector<VarId> available_columns;
    std::map<VarId, VarType> available_types;

    for (auto& item : subquery_return->return_items) {
        std::optional<VarId> column;
        if (item.alias.has_value()) {
            column = item.alias.value();
        } else if (auto* expr_var = dynamic_cast<ExprVar*>(item.expr.get())) {
            column = expr_var->id;
        }

        if (column.has_value()) {
            if (std::find(available_columns.begin(), available_columns.end(), *column) == available_columns.end()) {
                available_columns.push_back(*column);
            }

            auto type_it = inner_types.find(*column);
            if (type_it != inner_types.end()) {
                available_types[*column] = type_it->second;
            }
        }
    }

    if (!op_call.has_explicit_yield) {
        op_call.yield_items = available_columns;
    } else {
        std::vector<VarId> validated;
        validated.reserve(op_call.yield_items.size());

        for (auto& yield_var : op_call.yield_items) {
            if (std::find(available_columns.begin(), available_columns.end(), yield_var) == available_columns.end()) {
                throw QuerySemanticException(
                    "YIELD item not found in CALL subquery output: " + get_query_ctx().get_var_name(yield_var)
                );
            }
            validated.push_back(yield_var);
        }

        op_call.yield_items = std::move(validated);
    }

    std::map<VarId, VarType> yield_types;
    for (auto& var : op_call.yield_items) {
        auto it = available_types.find(var);
        if (it != available_types.end()) {
            yield_types.insert(*it);
        }
    }

    op_call.yield_types = std::move(yield_types);
}

void BindCall::visit(OpProject& op_project)
{
    if (op_project.subquery == nullptr) {
        throw QuerySemanticException("PROJECT subquery must end with RETURN");
    }

    auto* subquery_return = dynamic_cast<OpReturn*>(op_project.subquery.get());
    if (subquery_return == nullptr) {
        throw QuerySemanticException("PROJECT subquery must end with RETURN");
    }

    subquery_return->op->accept_visitor(*this);
    if (subquery_return->op_order_by != nullptr) {
        subquery_return->op_order_by->accept_visitor(*this);
    }

    auto inner_types = subquery_return->op->get_var_types();

    std::vector<VarId> available_columns;
    std::map<VarId, VarType> available_types;

    for (auto& item : subquery_return->return_items) {
        std::optional<VarId> column;
        if (item.alias.has_value()) {
            column = item.alias.value();
        } else if (auto* expr_var = dynamic_cast<ExprVar*>(item.expr.get())) {
            column = expr_var->id;
        }

        if (!column.has_value()) {
            continue;
        }

        if (std::find(available_columns.begin(), available_columns.end(), *column) == available_columns.end()) {
            available_columns.push_back(*column);
        }

        if (item.alias.has_value()) {
            if (auto* expr_var = dynamic_cast<ExprVar*>(item.expr.get())) {
                auto expr_type_it = inner_types.find(expr_var->id);
                if (expr_type_it != inner_types.end()) {
                    available_types[*column] = expr_type_it->second;
                }
            }
        }

        auto type_it = inner_types.find(*column);
        if (type_it != inner_types.end()) {
            available_types[*column] = type_it->second;
        }
    }

    if (available_columns.empty()) {
        throw QuerySemanticException("PROJECT must RETURN at least one node or edge column");
    }

    bool has_node_or_edge = false;
    for (auto& var : available_columns) {
        auto type_it = available_types.find(var);
        if (type_it == available_types.end()) {
            continue;
        }

        if (type_it->second.type != VarType::Node && type_it->second.type != VarType::Edge) {
            throw QuerySemanticException("PROJECT must RETURN at least one node or edge column");
        }

        has_node_or_edge = true;
    }

    if (!has_node_or_edge) {
        throw QuerySemanticException("PROJECT must RETURN at least one node or edge column");
    }

    op_project.projected_items = std::move(available_columns);
    op_project.projected_types = std::move(available_types);
}

void BindCall::visit(OpUnitTable&)
{
    // nothing to do
}

void BindCall::visit(OpEmpty&)
{
    // nothing to do
}
