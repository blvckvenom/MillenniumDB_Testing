#pragma once

#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"
#include "query/parser/expr/gql/expr_node_degree.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace GQL {
class BindingExprNodeDegree : public BindingExpr {
public:
    VarId node_var;
    NodeDegreeType degree_type;

    BindingExprNodeDegree(VarId node_var, NodeDegreeType degree_type) :
        node_var(node_var),
        degree_type(degree_type)
    { }

    ObjectId eval(const Binding& binding) override
    {
        ObjectId node_oid = binding[node_var];
        if (node_oid.is_null()) {
            return ObjectId::get_null();
        }
        if (GQL_OID::get_generic_type(node_oid) != GQL_OID::GenericType::NODE) {
            return ObjectId::get_null();
        }

        auto count_edges = [&](BPlusTree<3>& bpt) {
            bool interruption = false;
            BptIter<3> it =
                bpt.get_range(&interruption, { node_oid.id, 0, 0 }, { node_oid.id, UINT64_MAX, UINT64_MAX });

            int64_t count = 0;
            for (auto record = it.next(); record != nullptr; record = it.next()) {
                count++;
            }
            return count;
        };

        int64_t degree = 0;
        if (degree_type == NodeDegreeType::OUT || degree_type == NodeDegreeType::BOTH) {
            degree += count_edges(gql_model.get_from_to_edge());
        }
        if (degree_type == NodeDegreeType::IN || degree_type == NodeDegreeType::BOTH) {
            degree += count_edges(gql_model.get_to_from_edge());
        }

        return Conversions::pack_int(degree);
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    void print(std::ostream& os, std::vector<BindingIter*>&) const override
    {
        if (degree_type == NodeDegreeType::OUT) {
            os << "OutDegree(" << node_var << ")";
        } else if (degree_type == NodeDegreeType::IN) {
            os << "InDegree(" << node_var << ")";
        } else {
            os << "Degree(" << node_var << ")";
        }
    }
};
} // namespace GQL
