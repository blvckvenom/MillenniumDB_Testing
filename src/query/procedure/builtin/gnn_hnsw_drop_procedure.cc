#include "gnn_hnsw_drop_procedure.h"

#include <stdexcept>

#include "gnn_procedure_utils.h"
#include "graph_models/gql/gql_model.h"

using namespace GQL;
using namespace GQL::Procedures;

void GnnHnswDropProcedure::execute(ProcedureContext& ctx) {
    // Step 1: Validate argument count
    if (ctx.arguments.size() != 1) {
        throw std::runtime_error(
            "gnn.hnsw.drop() requires exactly 1 argument, got " +
            std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL gnn.hnsw.drop(indexName)\n"
            "  YIELD success"
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

    validate_safe_name(index_name, "indexName");

    // Step 3: Drop the index
    bool success = gql_model.catalog.hnsw_index_manager.drop_hnsw_index(index_name);

    if (!success) {
        // Index not found - provide helpful error
        auto index_names = gql_model.catalog.hnsw_index_manager.get_index_names();
        throw std::runtime_error(
            format_not_found_error("HNSW index", index_name, index_names)
        );
    }

    // Step 4: Yield result
    ctx.yield("success", ctx.create_bool(true));
    ctx.yield_row();
}
