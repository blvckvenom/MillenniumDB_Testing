#include "node_plan.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/projection_query_context.h"
#include "query/executor/binding_iter/object_enum.h"
#include "query/executor/binding_iter/single_result_binding_iter.h"
#include "query/executor/binding_iter/index_scan.h"
#include "query/executor/binding_iter/scan_ranges/unassigned_var.h"

using namespace GQL;

double NodePlan::estimate_cost() const
{
    return estimate_output_size();
}

double NodePlan::estimate_output_size() const
{
    return gql_model.catalog.nodes_count;
}

std::set<VarId> NodePlan::get_vars() const
{
    std::set<VarId> result;
    result.insert(node_id);
    return result;
}

void NodePlan::set_input_vars(const std::set<VarId>& input_vars) {
    set_input_var(input_vars, node_id, &node_assigned);
}

std::unique_ptr<BindingIter> NodePlan::get_binding_iter() const
{
    if (node_assigned) {
        return std::make_unique<SingleResultBindingIter>();
    }


    // Projections preserve original node IDs, so we must scan the nodes_index B+Tree
    // instead of using ObjectEnum which assumes sequential IDs 0..n-1
    auto& qctx = get_query_ctx();
    if (qctx.is_using_projection() && qctx.projection_ctx && qctx.projection_ctx->nodes_index) {
        // Use B+Tree scan on projection's nodes_index (handles non-sequential IDs)
        std::array<std::unique_ptr<ScanRange>, 1> ranges;
        ranges[0] = std::make_unique<UnassignedVar>(node_id);

        return std::make_unique<IndexScan<1>>(
            *qctx.projection_ctx->nodes_index,
            std::move(ranges)
        );
    }

    // Default: Use ObjectEnum for main graph (sequential IDs 0..catalog.nodes_count-1)
    return std::make_unique<ObjectEnum>(node_id, ObjectId::MASK_NODE, gql_model.catalog.nodes_count);
}

void NodePlan::print(std::ostream& os, int indent) const
{
    for (int i = 0; i < indent; ++i) {
        os << ' ';
    }
    os << std::string(indent, ' ');
    os << "NodePlan(" << get_query_ctx().get_var_name(node_id) << ")";
}
