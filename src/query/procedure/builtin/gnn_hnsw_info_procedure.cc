#include "gnn_hnsw_info_procedure.h"

#include <stdexcept>

#include "gnn_procedure_utils.h"
#include "graph_models/gql/gql_model.h"
#include "storage/index/hnsw/hnsw_metric.h"

using namespace GQL;
using namespace GQL::Procedures;

void GnnHnswInfoProcedure::execute(ProcedureContext& ctx) {
    // Step 1: Validate argument count
    if (ctx.arguments.size() != 1) {
        throw std::runtime_error(
            "gnn.hnsw.info() requires exactly 1 argument, got " +
            std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL gnn.hnsw.info(indexName)\n"
            "  YIELD indexName, dimension, nodeCount, metric, M, efConstruction, layers"
        );
    }

    // Step 2: Parse indexName
    std::string index_name;
    try {
        index_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid indexName parameter: " + std::string(e.what())
        );
    }

    if (index_name.empty()) {
        throw std::runtime_error("Invalid index name: name cannot be empty.");
    }

    // Step 3: Get the index
    auto* index = gql_model.catalog.hnsw_index_manager.get_hnsw_index(index_name);
    if (index == nullptr) {
        auto index_names = gql_model.catalog.hnsw_index_manager.get_index_names();
        throw std::runtime_error(
            format_not_found_error("HNSW index", index_name, index_names)
        );
    }

    // Step 4: Get index info
    const auto& params = index->get_params();

    // Get metric string from metadata
    std::string metric_str = "unknown";
    auto& metadata_map = gql_model.catalog.hnsw_index_manager.get_name2metadata();
    auto it = metadata_map.find(index_name);
    if (it != metadata_map.end()) {
        metric_str = HNSW::to_string(it->second.metric_type);
    }

    // Step 5: Yield result
    ctx.yield("indexName", ctx.create_string(index_name));
    ctx.yield("dimension", ctx.create_int(static_cast<int64_t>(params.dimensions)));
    ctx.yield("nodeCount", ctx.create_int(static_cast<int64_t>(index->size())));
    ctx.yield("metric", ctx.create_string(metric_str));
    ctx.yield("M", ctx.create_int(static_cast<int64_t>(params.M)));
    ctx.yield("efConstruction", ctx.create_int(static_cast<int64_t>(params.ef_construction)));
    ctx.yield("layers", ctx.create_int(static_cast<int64_t>(params.layers)));
    ctx.yield_row();
}
