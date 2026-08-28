#include "gnn_hnsw_list_procedure.h"

#include "graph_models/gql/gql_model.h"
#include "storage/index/hnsw/hnsw_metric.h"

using namespace GQL;
using namespace GQL::Procedures;

void GnnHnswListProcedure::execute(ProcedureContext& ctx) {
    auto index_names = gql_model.catalog.hnsw_index_manager.get_index_names();
    auto metadata_map = gql_model.catalog.hnsw_index_manager.get_name2metadata();

    for (const auto& name : index_names) {
        auto* index = gql_model.catalog.hnsw_index_manager.get_hnsw_index(name);
        if (index == nullptr) {
            continue;
        }

        const auto& params = index->get_params();

        // Get metric string
        std::string metric_str = "unknown";
        auto it = metadata_map.find(name);
        if (it != metadata_map.end()) {
            metric_str = HNSW::to_string(it->second.metric_type);
        }

        ctx.yield("indexName", ctx.create_string(name));
        ctx.yield("dimension", ctx.create_int(static_cast<int64_t>(params.dimensions)));
        ctx.yield("nodeCount", ctx.create_int(static_cast<int64_t>(index->size())));
        ctx.yield("metric", ctx.create_string(metric_str));
        ctx.yield_row();
    }
}
