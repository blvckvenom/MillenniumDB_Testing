#include "gql_model.h"

#include <type_traits>

#include "graph_models/gql/conversions.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_query_context.h"
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

// Dynamic index selection implementations for USE GRAPH projection support

BPlusTree<3>& GQLModel::get_from_to_edge() {
    auto& ctx = get_query_ctx();
    std::cerr << "[GQLModel] get_from_to_edge() called, is_using_projection=" << ctx.is_using_projection()
              << ", active_projection='" << ctx.active_projection << "'" << std::endl;
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->from_to_edge_index) {
            std::cerr << "[GQLModel] ERROR: Projection context not loaded!" << std::endl;
            throw std::runtime_error("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        std::cerr << "[GQLModel] Using projection index" << std::endl;
        return *ctx.projection_ctx->from_to_edge_index;
    }
    std::cerr << "[GQLModel] Using main graph index" << std::endl;
    return *from_to_edge;
}

BPlusTree<3>& GQLModel::get_to_from_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->to_from_edge_index) {
            throw std::runtime_error("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        return *ctx.projection_ctx->to_from_edge_index;
    }
    return *to_from_edge;
}

BPlusTree<2>& GQLModel::get_node_label() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->node_label_index) {
            throw std::runtime_error(
                "Cannot use node labels with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include node label information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with labels:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE LABELS)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->node_label_index;
    }
    return *node_label;
}

BPlusTree<2>& GQLModel::get_edge_label() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->edge_label_index) {
            throw std::runtime_error(
                "Cannot use edge labels with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include edge label information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with labels:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE LABELS)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->edge_label_index;
    }
    return *edge_label;
}

BPlusTree<2>& GQLModel::get_label_node() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->label_node_index) {
            throw std::runtime_error(
                "Cannot use node labels with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include node label information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with labels:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE LABELS)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->label_node_index;
    }
    return *label_node;
}

BPlusTree<2>& GQLModel::get_label_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->label_edge_index) {
            throw std::runtime_error(
                "Cannot use edge labels with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include edge label information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with labels:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE LABELS)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->label_edge_index;
    }
    return *label_edge;
}

BPlusTree<3>& GQLModel::get_node_key_value() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->node_key_value_index) {
            throw std::runtime_error(
                "Cannot access node properties with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include node property information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with properties:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE PROPERTIES)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->node_key_value_index;
    }
    return *node_key_value;
}

BPlusTree<3>& GQLModel::get_edge_key_value() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->edge_key_value_index) {
            throw std::runtime_error(
                "Cannot access edge properties with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include edge property information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with properties:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE PROPERTIES)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->edge_key_value_index;
    }
    return *edge_key_value;
}

BPlusTree<3>& GQLModel::get_key_value_node() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->key_value_node_index) {
            throw std::runtime_error(
                "Cannot access node properties with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include node property information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with properties:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE PROPERTIES)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->key_value_node_index;
    }
    return *key_value_node;
}

BPlusTree<3>& GQLModel::get_key_value_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->key_value_edge_index) {
            throw std::runtime_error(
                "Cannot access edge properties with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include edge property information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with properties:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE PROPERTIES)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->key_value_edge_index;
    }
    return *key_value_edge;
}

BPlusTree<3>& GQLModel::get_edge_from_to() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        throw std::runtime_error(
            "Edge-first index ordering is not supported in projections. "
            "Projection '" + ctx.active_projection + "' uses a simplified index structure. "
            "This may indicate the query optimizer selected a plan incompatible with projections."
        );
    }
    return *edge_from_to;
}

BPlusTree<2>& GQLModel::get_equal_d_edge() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        throw std::runtime_error(
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
        throw std::runtime_error(
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
        std::cerr << "[GQLModel] get_n1_n2_edge() called for projection, returning from_to_edge_index" << std::endl;
        if (!ctx.projection_ctx || !ctx.projection_ctx->from_to_edge_index) {
            throw std::runtime_error("Projection context not loaded for '" + ctx.active_projection + "'");
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
        std::cerr << "[GQLModel] get_edge_n1_n2() called for projection, returning from_to_edge_index" << std::endl;
        if (!ctx.projection_ctx || !ctx.projection_ctx->from_to_edge_index) {
            throw std::runtime_error("Projection context not loaded for '" + ctx.active_projection + "'");
        }
        // For projections, edge_n1_n2 queries use the from_to_edge_index
        // Note: This may not be optimal since from_to_edge is ordered by (from, to, edge)
        // rather than (edge, from, to), but it should still work for scanning
        return *ctx.projection_ctx->from_to_edge_index;
    }
    return *edge_n1_n2;
}
