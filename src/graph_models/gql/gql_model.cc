#include "gql_model.h"

#include <type_traits>

#include "graph_models/gql/conversions.h"
#include "misc/logger.h"
#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_query_context.h"
#include "query/exceptions.h"
#include "query/procedure/procedure_catalog.h"
#include "query/procedure/builtin/project_procedure.h"
#include "query/procedure/builtin/graph_exists_procedure.h"
#include "query/procedure/builtin/graph_drop_procedure.h"
#include "query/procedure/builtin/graph_list_procedure.h"
#ifdef ENABLE_GNN
#include "query/procedure/builtin/gnn_offline_sample_procedure.h"
#include "query/procedure/builtin/gnn_sample_list_procedure.h"
#include "query/procedure/builtin/gnn_sample_info_procedure.h"
#include "query/procedure/builtin/gnn_sample_drop_procedure.h"
#include "query/procedure/builtin/gnn_hnsw_create_procedure.h"
#include "query/procedure/builtin/gnn_hnsw_find_similar_procedure.h"
#include "query/procedure/builtin/gnn_hnsw_list_procedure.h"
#include "query/procedure/builtin/gnn_hnsw_drop_procedure.h"
#include "query/procedure/builtin/gnn_hnsw_info_procedure.h"
#include "query/procedure/builtin/gnn_materialize_batches_procedure.h"
#include "query/procedure/builtin/gnn_build_feature_store_procedure.h"
#include "query/procedure/builtin/gnn_train_procedure.h"
#include "query/procedure/builtin/gnn_predict_procedure.h"
#include "query/procedure/builtin/gnn_list_checkpoints_procedure.h"
#include "query/procedure/builtin/gnn_checkpoint_exists_procedure.h"
#include "query/procedure/builtin/gnn_checkpoint_delete_procedure.h"
#endif
#include "query/query_context.h"
#include "storage/index/bplus_tree/bplus_tree.h"

// memory for the object
static typename std::aligned_storage<sizeof(GQLModel), alignof(GQLModel)>::type gql_model_buf;

// global object
GQLModel& gql_model = reinterpret_cast<GQLModel&>(gql_model_buf);

std::unique_ptr<ModelDestroyer> GQLModel::init(const std::string& db_folder)
{
    new (&gql_model) GQLModel();

    // Initialize ProjectionManager if db_folder is provided
    if (!db_folder.empty()) {
        GQL::ProjectionManager::get_instance().init(db_folder);
    }

    // Register built-in procedures
    auto& catalog = GQL::ProcedureCatalog::get_instance();
    catalog.register_procedure(std::make_unique<GQL::Procedures::ProjectProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GraphExistsProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GraphDropProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GraphListProcedure>());

#ifdef ENABLE_GNN
    // Register GNN sampling procedures (only when GNN module is enabled)
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnOfflineSampleProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnSampleListProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnSampleInfoProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnSampleDropProcedure>());

    // Register GNN HNSW index procedures (ANN search over embeddings)
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnHnswCreateProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnHnswFindSimilarProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnHnswListProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnHnswDropProcedure>());
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnHnswInfoProcedure>());

    // Register GNN batch materialization procedure (L3 reorder + L4 packing)
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnMaterializeBatchesProcedure>());

    // Register GNN four-level feature store procedure (L1-L4 build)
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnBuildFeatureStoreProcedure>());

    // Register GNN training procedure (GraphSAGE training loop)
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnTrainProcedure>());

    // Register GNN prediction procedure (inference from checkpoint)
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnPredictProcedure>());

    // Register GNN checkpoint listing procedure
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnListCheckpointsProcedure>());

    // Register GNN checkpoint existence-check procedure
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnCheckpointExistsProcedure>());

    // Register GNN checkpoint delete procedure
    catalog.register_procedure(std::make_unique<GQL::Procedures::GnnCheckpointDeleteProcedure>());
#endif

    return std::make_unique<ModelDestroyer>([]() { gql_model.~GQLModel(); });
}

GQLModel::GQLModel() :
    catalog("catalog.dat")
{
    QueryContext::_debug_print = GQL::Conversions::debug_print;

    node_label = std::make_unique<BPlusTree<2>>("node_label");
    label_node = std::make_unique<BPlusTree<2>>("label_node");
    edge_label = std::make_unique<BPlusTree<2>>("edge_label");
    label_edge = std::make_unique<BPlusTree<2>>("label_edge");
    node_key_value = std::make_unique<BPlusTree<3>>("node_key_value");
    key_value_node = std::make_unique<BPlusTree<3>>("key_value_node");
    edge_key_value = std::make_unique<BPlusTree<3>>("edge_key_value");
    key_value_edge = std::make_unique<BPlusTree<3>>("key_value_edge");
    from_to_edge = std::make_unique<BPlusTree<3>>("from_to_edge");
    to_from_edge = std::make_unique<BPlusTree<3>>("to_from_edge");
    edge_from_to = std::make_unique<BPlusTree<3>>("edge_from_to");
    n1_n2_edge = std::make_unique<BPlusTree<3>>("n1_n2_edge");
    edge_n1_n2 = std::make_unique<BPlusTree<3>>("edge_n1_n2");
    equal_u_edge = std::make_unique<BPlusTree<2>>("equal_u_edge");
    equal_d_edge = std::make_unique<BPlusTree<2>>("equal_d_edge");
}

// Dynamic index selection implementations for USE GRAPH projection support.
//
// Spec #3 T3.9: when a query opens a projection built under a restricted
// IndexSet preset (GNN_MINIMAL or READONLY_TRAVERSAL) and attempts to access
// a B+Tree that was not materialized, the corresponding getter raises a
// QueryException naming:
//   1. the projection name,
//   2. the missing .leaf index (e.g. "edge_label"),
//   3. the current IndexSet preset (e.g. "GNN_MINIMAL"),
//   4. the minimum preset that would satisfy the query
//      (e.g. "READONLY_TRAVERSAL"), and
//   5. the fallback suggestion "ALL".
// Prior to this commit the reader slot held nullptr (from T3.8's gate in
// open_all_bplustree_readers_), so access would segfault or surface a
// generic null-deref error instead of an actionable message.
namespace {
// Returns true if `which` is a property-index bit (NODE_KEY_VALUE,
// KEY_VALUE_NODE, EDGE_KEY_VALUE, KEY_VALUE_EDGE). Property indexes are
// gated by the includeNodeProperties / includeEdgeProperties config keys,
// NOT by IndexSet (Spec #3 §3.4) — so the upgrade hint for these must
// mention the property flag, not just a wider preset.
bool is_property_index(GQL::ProjectionIndex which) {
    return which == GQL::ProjectionIndex::NODE_KEY_VALUE
        || which == GQL::ProjectionIndex::KEY_VALUE_NODE
        || which == GQL::ProjectionIndex::EDGE_KEY_VALUE
        || which == GQL::ProjectionIndex::KEY_VALUE_EDGE;
}

// Build the user-facing QueryException body for a missing-index access.
// `projection_name` is the USE-clause target, `which` names the index whose
// .leaf file was elided, and `active` is the preset the projection was
// built under (restored from catalog by ProjectionStorage::open()).
std::string make_missing_index_message(
    const std::string& projection_name,
    GQL::ProjectionIndex which,
    GQL::IndexSet active)
{
    const char* idx_name    = GQL::projection_index_name(which);
    const char* active_name = GQL::index_set_name(active);

    std::string msg;
    msg.reserve(512);
    msg += "Cannot execute query - index '";
    msg += idx_name;
    msg += "' is not materialized for projection '";
    msg += projection_name;
    msg += "' (indexSet='";
    msg += active_name;
    msg += "'). ";

    if (is_property_index(which)) {
        // Property indexes aren't gated by IndexSet — they're gated by
        // includeNodeProperties / includeEdgeProperties. Point the user at
        // the right config key rather than suggest widening IndexSet.
        const bool is_node = (which == GQL::ProjectionIndex::NODE_KEY_VALUE
                           || which == GQL::ProjectionIndex::KEY_VALUE_NODE);
        msg += "Property indexes are controlled by the ";
        msg += is_node ? "includeNodeProperties" : "includeEdgeProperties";
        msg += " projection config (not by indexSet). Rebuild the projection "
               "with the corresponding property config (and indexSet='ALL') "
               "to enable this query.";
    } else {
        const GQL::IndexSet min_preset = GQL::minimum_preset_for(which);
        const char* min_name = GQL::index_set_name(min_preset);
        msg += "To enable this query, rebuild the projection with indexSet='";
        msg += min_name;
        msg += "'";
        if (min_preset != GQL::IndexSet::ALL) {
            msg += " (or 'ALL')";
        }
        msg += ".";
    }
    return msg;
}
} // namespace

BPlusTree<3>& GQLModel::get_from_to_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            logger.error() << "Projection context not loaded in get_from_to_edge()";
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->from_to_edge_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::FROM_TO_EDGE,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->from_to_edge_index;
    }
    return *from_to_edge;
}

BPlusTree<3>& GQLModel::get_to_from_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->to_from_edge_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::TO_FROM_EDGE,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->to_from_edge_index;
    }
    return *to_from_edge;
}

BPlusTree<2>& GQLModel::get_node_label() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->node_label_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::NODE_LABEL,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->node_label_index;
    }
    return *node_label;
}

BPlusTree<2>& GQLModel::get_edge_label() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->edge_label_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::EDGE_LABEL,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->edge_label_index;
    }
    return *edge_label;
}

BPlusTree<2>& GQLModel::get_label_node() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->label_node_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::LABEL_NODE,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->label_node_index;
    }
    return *label_node;
}

BPlusTree<2>& GQLModel::get_label_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->label_edge_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::LABEL_EDGE,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->label_edge_index;
    }
    return *label_edge;
}

BPlusTree<3>& GQLModel::get_node_key_value() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->node_key_value_index) {
            // Property indexes are gated by includeProperties config at build
            // time, not by IndexSet (Spec #3 §3.4). The diagnostic still cites
            // the active preset for diagnostic completeness but the suggested
            // fix is to rebuild with includeProperties, not to widen IndexSet.
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::NODE_KEY_VALUE,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->node_key_value_index;
    }
    return *node_key_value;
}

BPlusTree<3>& GQLModel::get_edge_key_value() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->edge_key_value_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::EDGE_KEY_VALUE,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->edge_key_value_index;
    }
    return *edge_key_value;
}

BPlusTree<3>& GQLModel::get_key_value_node() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->key_value_node_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::KEY_VALUE_NODE,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->key_value_node_index;
    }
    return *key_value_node;
}

BPlusTree<3>& GQLModel::get_key_value_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->key_value_edge_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::KEY_VALUE_EDGE,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->key_value_edge_index;
    }
    return *key_value_edge;
}

BPlusTree<3>& GQLModel::get_edge_from_to() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->edge_from_to_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::EDGE_FROM_TO,
                ctx.projection_ctx->index_set));
        }
        return *ctx.projection_ctx->edge_from_to_index;
    }
    return *edge_from_to;
}

BPlusTree<2>& GQLModel::get_equal_d_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        throw QueryException(
            "Self-loop queries are not supported in projections. "
            "Projection '" + ctx.active_projection + "' does not have specialized self-loop indexes. "
            "Query the main graph with CURRENT_GRAPH for self-loop patterns."
        );
    }
    return *equal_d_edge;
}

BPlusTree<2>& GQLModel::get_equal_u_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        throw QueryException(
            "Self-loop queries are not supported in projections. "
            "Projection '" + ctx.active_projection + "' does not have specialized self-loop indexes. "
            "Query the main graph with CURRENT_GRAPH for self-loop patterns."
        );
    }
    return *equal_u_edge;
}

BPlusTree<3>& GQLModel::get_n1_n2_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->from_to_edge_index) {
            throw QueryException(make_missing_index_message(
                ctx.active_projection,
                GQL::ProjectionIndex::FROM_TO_EDGE,
                ctx.projection_ctx->index_set));
        }
        // For projections, n1_n2_edge queries use the from_to_edge_index
        // The caller must handle undirected edges by checking both directions
        return *ctx.projection_ctx->from_to_edge_index;
    }
    return *n1_n2_edge;
}

BPlusTree<3>& GQLModel::get_edge_n1_n2() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw QueryException("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        if (!ctx.projection_ctx->edge_n1_n2_index) {
            // Fallback to from_to_edge_index for older projections without edge_n1_n2_index.
            // If that too is absent (e.g. GNN_MINIMAL would still contain it, but a
            // hypothetical preset without FROM_TO_EDGE would not), surface the
            // richer missing-index diagnostic naming the preferred edge_n1_n2 index.
            logger.debug() << "edge_n1_n2_index not available, falling back to from_to_edge_index";
            if (!ctx.projection_ctx->from_to_edge_index) {
                throw QueryException(make_missing_index_message(
                    ctx.active_projection,
                    GQL::ProjectionIndex::EDGE_N1_N2,
                    ctx.projection_ctx->index_set));
            }
            return *ctx.projection_ctx->from_to_edge_index;
        }
        return *ctx.projection_ctx->edge_n1_n2_index;
    }
    return *edge_n1_n2;
}
