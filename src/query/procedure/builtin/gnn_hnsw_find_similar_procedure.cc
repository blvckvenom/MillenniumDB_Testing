#include "gnn_hnsw_find_similar_procedure.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/gql_object_id.h"
#include "storage/index/hnsw/hnsw_index.h"
#include "gnn_procedure_utils.h"

using namespace GQL;
using namespace GQL::Procedures;

void GnnHnswFindSimilarProcedure::execute(ProcedureContext& ctx) {
    // Step 1: Validate argument count
    if (ctx.arguments.size() != 4) {
        throw std::runtime_error(
            "gnn_hnsw_find_similar() requires exactly 4 arguments, got " +
            std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL gnn_hnsw_find_similar(indexName, nodeId, k, ef)\n"
            "  YIELD similar_node, distance\n\n"
            "Parameters:\n"
            "  - indexName (STRING): Name of the HNSW index\n"
            "  - nodeId (INT): Seed node ID\n"
            "  - k (INT): Number of similar nodes to return\n"
            "  - ef (INT): Number of candidates (higher = better recall)\n\n"
            "Example:\n"
            "  CALL gnn_hnsw_find_similar('arxiv_idx', 42, 10, 100)"
        );
    }

    // Step 2: Parse indexName
    std::string index_name;
    try {
        index_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid indexName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING containing the index name."
        );
    }

    if (index_name.empty()) {
        throw std::runtime_error(
            "Invalid index name: name cannot be empty."
        );
    }

    // Step 3: Parse nodeId
    int64_t node_id;
    try {
        node_id = ctx.get_int_argument(1);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid nodeId parameter: " + std::string(e.what()) + "\n\n"
            "The second parameter must be an INT containing the seed node ID."
        );
    }

    if (node_id < 0) {
        throw std::runtime_error(
            "Invalid nodeId: must be non-negative, got " + std::to_string(node_id)
        );
    }

    // Step 4: Parse k
    int64_t k;
    try {
        k = ctx.get_int_argument(2);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid k parameter: " + std::string(e.what()) + "\n\n"
            "The third parameter must be an INT specifying number of results."
        );
    }

    if (k <= 0) {
        throw std::runtime_error(
            "Invalid k: must be positive, got " + std::to_string(k)
        );
    }

    // Step 5: Parse ef
    int64_t ef;
    try {
        ef = ctx.get_int_argument(3);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid ef parameter: " + std::string(e.what()) + "\n\n"
            "The fourth parameter must be an INT specifying number of candidates."
        );
    }

    if (ef <= 0) {
        throw std::runtime_error(
            "Invalid ef: must be positive, got " + std::to_string(ef)
        );
    }

    if (ef < k) {
        throw std::runtime_error(
            "Invalid ef: should be >= k for good results. Got ef=" +
            std::to_string(ef) + ", k=" + std::to_string(k)
        );
    }

    // Step 6: Get the HNSW index
    auto* hnsw_index = gql_model.catalog.hnsw_index_manager.get_hnsw_index(index_name);
    if (hnsw_index == nullptr) {
        // List available indexes for helpful error
        auto index_names = gql_model.catalog.hnsw_index_manager.get_index_names();
        throw std::runtime_error(
            format_not_found_error("HNSW index", index_name, index_names,
                                   "CALL gnn_hnsw_create('index_name', 'node_features')")
        );
    }

    // Step 7: Verify index uses raw embeddings mode
    if (!hnsw_index->uses_raw_embeddings()) {
        throw std::runtime_error(
            "HNSW index '" + index_name + "' was not created from GNN embeddings.\n"
            "Only indexes created with gnn_hnsw_create() can be queried with this procedure."
        );
    }

    // Step 8: Validate nodeId is within range
    uint64_t index_size = hnsw_index->size();

    if (static_cast<uint64_t>(node_id) >= index_size) {
        // R3: Guard against unsigned underflow when index_size == 0
        std::string range_str = (index_size > 0)
            ? "0-" + std::to_string(index_size - 1)
            : "empty";
        throw std::runtime_error(
            "Invalid nodeId: " + std::to_string(node_id) + " is out of range.\n"
            "Index '" + index_name + "' contains " + std::to_string(index_size) +
            " nodes (" + range_str + ")."
        );
    }

    // I2: Guard against uint32_t truncation for large node IDs
    if (static_cast<uint64_t>(node_id) > static_cast<uint64_t>(UINT32_MAX)) {
        throw std::runtime_error(
            "nodeId " + std::to_string(node_id) + " exceeds HNSW 32-bit node ID limit."
        );
    }

    // Step 9: Get the query embedding directly from the HNSW index.
    // The index already holds all raw embeddings in memory (loaded at startup),
    // so we avoid reopening the FeatureMatrix per query.
    const float* query_embedding = hnsw_index->get_raw_embedding(static_cast<uint32_t>(node_id));
    if (query_embedding == nullptr) {
        throw std::runtime_error(
            "Failed to retrieve embedding for node " + std::to_string(node_id) +
            " from HNSW index '" + index_name + "'.\n"
            "The index may be corrupted or the node ID is out of bounds."
        );
    }

    // Step 10: Execute HNSW query
    HNSW::HNSWHeap results = hnsw_index->query_raw(
        query_embedding,
        static_cast<uint64_t>(k),
        static_cast<uint64_t>(ef)
    );

    // Step 12: Extract results and sort by distance (ascending)
    std::vector<std::pair<float, uint32_t>> sorted_results;
    sorted_results.reserve(results.size());

    while (!results.empty()) {
        auto entry = results.get_min();
        results.pop_min();
        sorted_results.emplace_back(entry.distance, entry.node_id);
    }

    // Results are already sorted by distance from the heap

    // Step 13: Yield results
    for (const auto& [distance, result_node_id] : sorted_results) {
        // Read the node's ObjectId from the index instead of rebuilding it as
        // (MASK_NODE | internal_id). Both agree for indexes built from a raw
        // embedding buffer, which stores exactly that value, but rebuilding it
        // assumes row position equals node ordinal. That assumption silently
        // returns the wrong nodes for any index whose rows are not in node
        // order, which is the case for the embeddings the GNN pipeline exports
        // in batch order, and it blocks the MinHash row reordering.
        const ObjectId node_oid = hnsw_index->get_node(result_node_id).object_oid;

        ctx.yield("similar_node", node_oid);
        ctx.yield("distance", ctx.create_float(distance));
        ctx.yield_row();
    }
}
