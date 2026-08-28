#include "gnn_hnsw_create_property_procedure.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "query/query_context.h"
#include "storage/index/hnsw/hnsw_index.h"
#include "storage/index/hnsw/hnsw_index_manager.h"
#include "storage/index/hnsw/hnsw_metric.h"
#include "system/file_manager.h"

#include "gnn_procedure_utils.h"
#include "gnn_projection_scope.h"

using namespace GQL;
using namespace GQL::Procedures;

namespace {

HNSW::MetricType parse_metric(const std::string& metric_str)
{
    std::string lower = metric_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "cosine" || lower == "cos") {
        return HNSW::MetricType::COSINE_DISTANCE;
    } else if (lower == "euclidean" || lower == "l2") {
        return HNSW::MetricType::EUCLIDEAN_DISTANCE;
    } else if (lower == "manhattan" || lower == "l1") {
        return HNSW::MetricType::MANHATTAN_DISTANCE;
    }
    throw std::runtime_error(
        "Unknown metric: '" + metric_str + "'.\n"
        "Supported metrics: 'cosine', 'euclidean', 'manhattan'"
    );
}

} // namespace

void GnnHnswCreatePropertyProcedure::execute(ProcedureContext& ctx)
{
    if (ctx.arguments.size() < 2 || ctx.arguments.size() > 3) {
        throw std::runtime_error(
            "gnn_hnsw_create_property() takes 2 or 3 arguments, got "
            + std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL gnn_hnsw_create_property(indexName, property [, options])\n"
            "  YIELD indexName, projection, property, dimension, nodeCount, buildTimeMs\n\n"
            "Example:\n"
            "  CALL gnn_hnsw_create_property('cora_emb_idx', 'embedding',\n"
            "       {projection: 'cora_proj', metric: 'cosine'})"
        );
    }

    const auto index_name = ctx.get_string_argument(0);
    validate_safe_name(index_name, "indexName");

    const auto property = ctx.get_string_argument(1);
    if (property.empty()) {
        throw std::runtime_error("Invalid property: name cannot be empty.");
    }

    // --- options -------------------------------------------------------------
    std::string projection = get_query_ctx().active_projection;
    auto metric = HNSW::MetricType::COSINE_DISTANCE;
    uint64_t max_edges = 16;
    uint64_t ef_construction = 200;
    uint64_t dimension = 0;
    uint64_t seed = 42;

    if (ctx.arguments.size() == 3) {
        const DictOptions opts(ctx.get_argument(2));
        if (auto v = opts.get_string("projection")) {
            projection = *v;
        }
        if (auto v = opts.get_string("metric")) {
            metric = parse_metric(*v);
        }
        if (auto v = opts.get_int("M")) {
            if (*v <= 0) {
                throw std::runtime_error("Invalid M: must be positive, got " + std::to_string(*v));
            }
            max_edges = static_cast<uint64_t>(*v);
        }
        if (auto v = opts.get_int("efConstruction")) {
            if (*v <= 0) {
                throw std::runtime_error(
                    "Invalid efConstruction: must be positive, got " + std::to_string(*v)
                );
            }
            ef_construction = static_cast<uint64_t>(*v);
        }
        if (auto v = opts.get_int("dimension")) {
            if (*v <= 0) {
                throw std::runtime_error(
                    "Invalid dimension: must be positive, got " + std::to_string(*v)
                );
            }
            dimension = static_cast<uint64_t>(*v);
        }
        if (auto v = opts.get_int("seed")) {
            seed = static_cast<uint64_t>(*v);
        }
    }

    if (!projection.empty()) {
        validate_safe_name(projection, "projection");
        if (!ProjectionManager::get_instance().projection_exists(projection)) {
            throw std::runtime_error(
                "Projection '" + projection + "' does not exist. "
                "Create it with graph_project() first."
            );
        }
    }

    auto& manager = gql_model.catalog.hnsw_index_manager;
    if (manager.get_hnsw_index(index_name) != nullptr) {
        throw std::runtime_error(
            "HNSW index '" + index_name + "' already exists. "
            "Drop it with gnn_hnsw_drop('" + index_name + "') before recreating it."
        );
    }

    // --- build ---------------------------------------------------------------
    const auto start_time = std::chrono::high_resolution_clock::now();
    uint_fast32_t node_count = 0;

    try {
        ProjectionScope scope(projection);

        if (dimension == 0) {
            dimension = HNSW::HNSWIndex::probe_node_property_dimension(property);
            if (dimension == 0) {
                throw std::runtime_error(
                    "Property '" + property + "' holds no tensor values in "
                    + (projection.empty() ? std::string("the base graph")
                                          : "projection '" + projection + "'")
                    + ".\nRun gnn_train(..., writeProperty: '" + property + "') first, "
                      "or pass an explicit dimension."
                );
            }
        }

        auto hnsw_index = HNSW::HNSWIndex::create(
            index_name,
            dimension,
            max_edges,
            ef_construction,
            HNSW::metric_type2metric_func(metric)
        );

        // Fix layer assignment so two builds over the same data agree. Without
        // this, ties on distance resolve differently on every build.
        hnsw_index->set_level_seed(seed);

        node_count = hnsw_index->index_node_property(property);
        if (node_count == 0) {
            throw std::runtime_error(
                "No node in " + (projection.empty() ? std::string("the base graph")
                                                    : "projection '" + projection + "'")
                + " carries a tensor of width " + std::to_string(dimension)
                + " under property '" + property + "', so the index would be empty."
            );
        }

        manager.write_to_disk(index_name, hnsw_index.get());

        HNSW::HNSWIndexManager::HNSWIndexMetadata metadata;
        metadata.metric_type = metric;
        metadata.predicate = property;
        metadata.source = HNSW::HNSWSource::NODE_PROPERTY;
        metadata.projection = projection;
        manager.load_hnsw_index(index_name, metadata);
    } catch (...) {
        // HNSWIndex::create() makes the on-disk directory before anything can
        // fail, so clear it rather than leave an orphan behind.
        namespace fs = std::filesystem;
        const auto relative_index_path =
            fs::path(HNSW::HNSWIndexManager::HNSW_INDEX_DIR) / index_name;
        std::error_code ec;
        fs::remove_all(file_manager.get_file_path(relative_index_path), ec);
        throw;
    }

    // Persist outside the projection scope so the catalog written is the base
    // database's own.
    gql_model.catalog.save();

    const auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time
    ).count();

    ctx.yield("indexName", ctx.create_string(index_name));
    ctx.yield("projection", ctx.create_string(projection));
    ctx.yield("property", ctx.create_string(property));
    ctx.yield("dimension", ctx.create_int(static_cast<int64_t>(dimension)));
    ctx.yield("nodeCount", ctx.create_int(static_cast<int64_t>(node_count)));
    ctx.yield("buildTimeMs", ctx.create_int(static_cast<int64_t>(build_ms)));
    ctx.yield_row();
}
