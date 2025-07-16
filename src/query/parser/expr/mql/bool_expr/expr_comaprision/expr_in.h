#pragma once

#include <memory>
#include <vector>

#include "query/parser/expr/expr.h"

namespace MQL {
class ExprIn : public Expr {
public:
    std::unique_ptr<Expr> lhs;
    std::vector<std::unique_ptr<Expr>> rhs;

    ExprIn(std::unique_ptr<Expr> lhs, std::vector<std::unique_ptr<Expr>> rhs) :
        lhs(std::move(lhs)), rhs(std::move(rhs)) { }

    virtual std::unique_ptr<Expr> clone() const override {
        std::vector<std::unique_ptr<Expr>> rhs_clone;
        rhs_clone.reserve(rhs.size());
        for (auto& e : rhs) rhs_clone.push_back(e->clone());
        return std::make_unique<ExprIn>(lhs->clone(), std::move(rhs_clone));
    }

    void accept_visitor(ExprVisitor& visitor) override {
        visitor.visit(*this);
    }

    bool has_aggregation() const override {
        if (lhs->has_aggregation()) return true;
        for (auto& e : rhs) if (e->has_aggregation()) return true;
        return false;
    }

    std::set<VarId> get_all_vars() const override {
        auto vars = lhs->get_all_vars();
        for (auto& e : rhs) {
            auto v = e->get_all_vars();
            vars.insert(v.begin(), v.end());
        }
        return vars;
    }
};
} // namespace MQL
