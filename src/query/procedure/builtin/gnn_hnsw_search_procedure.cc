#include "gnn_hnsw_search_procedure.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph_models/common/conversions.h"
#include "graph_models/common/datatypes/tensor/tensor.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/gql_object_id.h"
#include "query/executor/binding.h"
#include "query/optimizer/property_graph_model/plan/node_property_plan.h"
#include "query/query_context.h"
#include "storage/index/hnsw/hnsw_heap.h"
#include "storage/index/hnsw/hnsw_index.h"
#include "storage/index/hnsw/hnsw_index_manager.h"

#include "gnn_procedure_utils.h"
#include "gnn_projection_scope.h"

using namespace GQL;
using namespace GQL::Procedures;

namespace {

bool is_tensor_oid(ObjectId oid)
{
    switch (oid.get_type()) {
    case ObjectId::MASK_TENSOR_FLOAT_INLINED:
    case ObjectId::MASK_TENSOR_FLOAT_EXTERN:
    case ObjectId::MASK_TENSOR_FLOAT_TMP:
    case ObjectId::MASK_TENSOR_DOUBLE_INLINED:
    case ObjectId::MASK_TENSOR_DOUBLE_EXTERN:
    case ObjectId::MASK_TENSOR_DOUBLE_TMP:
        return true;
    default:
        return false;
    }
}

// Reads one node's value for a property, as a point lookup rather than a scan:
// binding the node to an ObjectId sends NodePropertyPlan down its node_key_value
// branch. The caller is responsible for the projection routing.
tensor::Tensor<float> read_node_property_tensor(
    uint64_t node_id,
    const std::string& property,
    const std::string& graph_label
)
{
    const auto key_val = GQL::Conversions::pack_node_property(property);
    if (key_val.is_null()) {
        throw std::runtime_error(
            "Property '" + property + "' does not exist in " + graph_label + "."
        );
    }

    const ObjectId node_oid(ObjectId::MASK_NODE | node_id);
    const auto value_var = get_query_ctx().get_internal_var();

    const auto plan = NodePropertyPlan(node_oid, key_val, value_var);
    auto iter = plan.get_binding_iter();

    Binding binding(get_query_ctx().get_var_size());
    iter->begin(binding);

    while (iter->next()) {
        const auto value_oid = binding[value_var];
        if (is_tensor_oid(value_oid)) {
            return Common::Conversions::to_tensor<float>(value_oid);
        }
    }

    throw std::runtime_error(
        "Node " + std::to_string(node_id) + " has no tensor under property '" + property
        + "' in " + graph_label + ", so it cannot be used as a seed."
    );
}

// Builds the query vector from a literal list of numbers.
tensor::Tensor<float> tensor_from_list(ObjectId list_oid)
{
    const auto elements = GQL::Conversions::unpack_list(list_oid);
    if (elements.empty()) {
        throw std::runtime_error("Invalid seed: the query vector is empty.");
    }

    std::vector<float> values;
    values.reserve(elements.size());
    for (const auto& element : elements) {
        values.push_back(Common::Conversions::to_float(element));
    }

    return tensor::Tensor<float>(
        reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(float)
    );
}

} // namespace

void GnnHnswSearchProcedure::execute(ProcedureContext& ctx)
{
    if (ctx.arguments.size() != 4) {
        throw std::runtime_error(
            "gnn_hnsw_search() requires exactly 4 arguments, got "
            + std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL gnn_hnsw_search(indexName, seed, k, ef)\n"
            "  YIELD node, nodeId, distance\n\n"
            "The seed is the id of an indexed node, or a literal list of numbers.\n\n"
            "Example:\n"
            "  CALL gnn_hnsw_search('cora_emb_idx', 42, 10, 64)"
        );
    }

    const auto index_name = ctx.get_string_argument(0);
    const auto k = ctx.get_int_argument(2);
    const auto ef = ctx.get_int_argument(3);

    if (k <= 0) {
        throw std::runtime_error("Invalid k: must be positive, got " + std::to_string(k));
    }
    if (ef < k) {
        throw std::runtime_error(
            "Invalid ef: must be at least k. Got ef=" + std::to_string(ef)
            + ", k=" + std::to_string(k)
        );
    }

    auto& manager = gql_model.catalog.hnsw_index_manager;
    auto* hnsw_index = manager.get_hnsw_index(index_name);
    if (hnsw_index == nullptr) {
        throw std::runtime_error(
            format_not_found_error(
                "HNSW index", index_name, manager.get_index_names(),
                "CALL gnn_hnsw_create_property('index_name', 'embedding')"
            )
        );
    }

    if (hnsw_index->uses_raw_embeddings()) {
        throw std::runtime_error(
            "HNSW index '" + index_name + "' was built from a raw feature matrix and "
            "identifies its rows by position.\n"
            "Query it with gnn_hnsw_find_similar(), or rebuild it over a node property "
            "with gnn_hnsw_create_property()."
        );
    }

    const auto metadata = manager.get_name2metadata().at(index_name);
    const std::string graph_label = metadata.projection.empty()
        ? std::string("the base graph")
        : "projection '" + metadata.projection + "'";

    // --- query vector --------------------------------------------------------
    const auto seed_oid = ctx.get_argument(1);
    tensor::Tensor<float> query_tensor;

    if (GQL_OID::get_generic_type(seed_oid) == GQL_OID::GenericType::LIST) {
        query_tensor = tensor_from_list(seed_oid);
    } else {
        const auto node_id = ctx.get_int_argument(1);
        if (node_id < 0) {
            throw std::runtime_error(
                "Invalid seed node id: must be non-negative, got " + std::to_string(node_id)
            );
        }
        if (metadata.source != HNSW::HNSWSource::NODE_PROPERTY) {
            throw std::runtime_error(
                "HNSW index '" + index_name + "' is not declared over a node property, "
                "so a node id cannot be resolved to a vector. Pass a literal list instead."
            );
        }
        // Only the lookup needs the indexed graph. The tensor payloads themselves
        // live in the database-wide tensor store, so the search below does not.
        ProjectionScope scope(metadata.projection);
        query_tensor = read_node_property_tensor(
            static_cast<uint64_t>(node_id), metadata.predicate, graph_label
        );
    }

    const auto dimensions = hnsw_index->get_params().dimensions;
    if (query_tensor.size() != dimensions) {
        throw std::runtime_error(
            "Query vector has " + std::to_string(query_tensor.size())
            + " values but index '" + index_name + "' holds vectors of "
            + std::to_string(dimensions) + "."
        );
    }

    // --- search --------------------------------------------------------------
    HNSW::HNSWHeap results = hnsw_index->query(
        query_tensor,
        static_cast<uint64_t>(k),
        static_cast<uint64_t>(ef),
        nullptr,
        nullptr
    );

    while (!results.empty()) {
        const auto entry = results.get_min();
        results.pop_min();

        // Read the identity the index stored rather than rebuilding it from the
        // row position, which is what makes this correct for any row ordering.
        const ObjectId node_oid = hnsw_index->get_node(entry.node_id).object_oid;

        ctx.yield("node", node_oid);
        ctx.yield("nodeId", ctx.create_int(static_cast<int64_t>(node_oid.get_value())));
        ctx.yield("distance", ctx.create_float(entry.distance));
        ctx.yield_row();
    }
}
