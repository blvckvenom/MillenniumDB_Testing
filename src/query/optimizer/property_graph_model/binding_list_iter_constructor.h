#pragma once

#include <memory>

#include "query/executor/binding_iter.h"
#include "query/executor/binding_iter/aggregation/agg.h"
#include "query/executor/binding_iter/gql/path_binding_iter.h"
#include "query/optimizer/plan/plan.h"
#include "query/parser/expr/gql/expr.h"
#include "query/parser/expr/gql/expr_property.h"
#include "query/parser/op/gql/op_return.h"
#include "query/parser/op/gql/op_visitor.h"

namespace GQL {

class PathBindingIterConstructor : public OpVisitor {
public:
    PathBindingIterConstructor() = default;

    void handle_order_by(
        std::vector<std::unique_ptr<Expr>>& items,
        std::vector<bool>& ascending_order,
        std::vector<bool>& null_order
    );

    void visit(OpQueryStatements& op) override;
    void visit(OpFilter& op) override;
    void visit(OpLet& op) override;
    void visit(OpOrderBy&) override;
    void visit(OpGroupBy&) override;
    void visit(OpWith&) override;

    void visit(OpReturn&) override;
    void visit(OpGraphPattern&) override;
    void visit(OpBasicGraphPattern&) override;
    void visit(OpGraphPatternList&) override;
    void visit(OpWhere&) override;
    void visit(OpPathUnion&) override;
    void visit(OpRepetition&) override;
    void visit(OpLinearPattern& op) override;

    void visit(OpNode&) override;
    void visit(OpEdge&) override;
    void visit(OpUnitTable& op) override;
    void visit(OpEmpty&) override;
    void visit(OpCallProcedure& op) override;
    void visit(OpOptional& op) override;

    std::unique_ptr<BindingIter> tmp_iter;

    bool grouping = false;
    std::set<VarId> group_saved_vars;
    std::map<VarId, std::unique_ptr<Agg>> aggregations;

    std::map<ExprProperty, bool> used_properties;

private:
    void build_projection_pipeline(
        std::vector<OpReturn::Item>& projection_items,
        bool distinct,
        std::vector<std::unique_ptr<Expr>>* group_by_items = nullptr,
        std::unique_ptr<Op>* op_order_by = nullptr,
        bool update_scope = false,
        bool reset_group_state = true
    );

    static std::vector<VarId> get_projection_vars(const std::vector<OpReturn::Item>& projection_items);

    std::unique_ptr<BindingIter> get_pending_properties(std::unique_ptr<BindingIter> binding_iter);

    bool is_first_iter = true;
    bool is_first_gp = true;

    std::vector<VarId> return_op_vars;

    std::set<VarId> group_vars;
    bool has_group_by = false;

    std::set<VarId> seen_nodes;
    std::set<VarId> possible_disjoint_nodes;
    std::set<VarId> assigned_vars;
    std::set<VarId> graph_pattern_vars;
    std::vector<VarId> linear_pattern_vars;
    std::set<std::pair<VarId, ObjectId>> setted_vars;
    std::vector<std::unique_ptr<Plan>> base_plans;

    std::unique_ptr<PathBindingIter> tmp_path;
};

} // namespace GQL
