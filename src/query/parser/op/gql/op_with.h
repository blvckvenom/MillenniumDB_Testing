#pragma once

#include <memory>
#include <vector>

#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_printer.h"
#include "query/parser/op/gql/op.h"
#include "query/parser/op/gql/op_return.h"

namespace GQL {

class OpWith : public Op {
public:
    std::vector<OpReturn::Item> with_items;
    bool distinct;
    std::vector<std::unique_ptr<Expr>> group_by_items;

    // might be nullptr
    std::unique_ptr<Op> op_order_by;

    OpWith(
        std::vector<OpReturn::Item>&& with_items,
        bool distinct,
        std::vector<std::unique_ptr<Expr>>&& group_by_items = {},
        std::unique_ptr<Op> op_order_by = nullptr
    ) :
        with_items(std::move(with_items)),
        distinct(distinct),
        group_by_items(std::move(group_by_items)),
        op_order_by(std::move(op_order_by))
    { }

    std::unique_ptr<Op> clone() const override
    {
        std::vector<OpReturn::Item> with_items_clone;
        with_items_clone.reserve(with_items.size());
        for (auto& item : with_items) {
            with_items_clone.emplace_back(item.expr->clone(), item.alias);
        }

        std::vector<std::unique_ptr<Expr>> group_by_items_clone;
        group_by_items_clone.reserve(group_by_items.size());
        for (auto& item : group_by_items) {
            group_by_items_clone.push_back(item->clone());
        }

        std::unique_ptr<Op> order_by_clone = op_order_by == nullptr ? nullptr : op_order_by->clone();

        return std::make_unique<OpWith>(
            std::move(with_items_clone),
            distinct,
            std::move(group_by_items_clone),
            std::move(order_by_clone)
        );
    }

    void accept_visitor(OpVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    std::set<VarId> get_all_vars() const override
    {
        std::set<VarId> res;

        for (auto& item : with_items) {
            for (auto& var : item.expr->get_all_vars()) {
                res.insert(var);
            }
            if (item.alias.has_value()) {
                res.insert(*item.alias);
            }
        }

        for (auto& expr : group_by_items) {
            for (auto& var : expr->get_all_vars()) {
                res.insert(var);
            }
        }

        if (op_order_by != nullptr) {
            auto order_by_vars = op_order_by->get_all_vars();
            res.merge(order_by_vars);
        }

        return res;
    }

    std::vector<VarId> get_expr_vars() const
    {
        std::vector<VarId> result;

        for (auto& item : with_items) {
            if (item.alias.has_value()) {
                result.push_back(item.alias.value());
            } else {
                auto expr_variables = item.expr->get_all_vars();
                result.insert(result.end(), expr_variables.begin(), expr_variables.end());
            }
        }
        return result;
    }

    std::ostream& print_to_ostream(std::ostream& os, int indent = 0) const override
    {
        os << std::string(indent, ' ');
        os << "OpWith(\n";
        ExprPrinter printer(os);

        for (auto& item : with_items) {
            os << std::string(indent + 2, ' ');
            item.expr->accept_visitor(printer);
            if (item.alias.has_value()) {
                os << " AS " << item.alias.value();
            }
            os << "\n";
        }

        if (!group_by_items.empty()) {
            os << std::string(indent + 2, ' ') << "GROUP BY ";
            bool first = true;
            for (auto& expr : group_by_items) {
                if (!first) {
                    os << ", ";
                }
                first = false;
                expr->accept_visitor(printer);
            }
            os << "\n";
        }

        os << std::string(indent, ' ') << ")";
        if (op_order_by != nullptr) {
            os << "\n";
            op_order_by->print_to_ostream(os, indent + 2);
        }
        return os;
    }
};
} // namespace GQL
