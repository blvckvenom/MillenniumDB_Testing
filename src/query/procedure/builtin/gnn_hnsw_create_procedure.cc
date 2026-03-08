#include "gnn_hnsw_create_procedure.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>

#include "gnn/storage/file_gnn_tensor_store.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/gql_object_id.h"
#include "storage/index/hnsw/hnsw_index.h"
#include "storage/index/hnsw/hnsw_index_manager.h"
#include "storage/index/hnsw/hnsw_metric.h"
#include "gnn_procedure_utils.h"
#include "system/file_manager.h"

using namespace GQL;
using namespace GQL::Procedures;

void GnnHnswCreateProcedure::execute(ProcedureContext& ctx) {
    // Step 1: Validate argument count
    if (ctx.arguments.size() < 2 || ctx.arguments.size() > 3) {
        throw std::runtime_error(
            "gnn_hnsw_create() requires 2-3 arguments, got " +
            std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL gnn_hnsw_create(indexName, tensorKey [, options])\n"
            "  YIELD indexName, dimension, nodeCount, buildTimeMs\n\n"
            "Parameters:\n"
            "  - indexName (STRING): Name for the HNSW index\n"
            "  - tensorKey (STRING): Key in GnnTensorStore (e.g., 'node_features')\n"
            "  - options (MAP, optional): metric, M, efConstruction\n\n"
            "Examples:\n"
            "  CALL gnn_hnsw_create('arxiv_idx', 'node_features')\n"
            "  CALL gnn_hnsw_create('arxiv_idx', 'node_features', {metric: 'cosine', M: 32})"
        );
    }

    // Step 2: Parse indexName
    std::string index_name;
    try {
        index_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid indexName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING containing the index name.\n"
            "Example: CALL gnn_hnsw_create('my_index', ...)"
        );
    }

    if (index_name.empty()) {
        throw std::runtime_error(
            "Invalid index name: name cannot be empty.\n"
            "Provide a non-empty string as the first argument.\n"
            "Example: CALL gnn_hnsw_create('my_index', ...)"
        );
    }

    // Step 3: Parse tensorKey
    std::string tensor_key;
    try {
        tensor_key = ctx.get_string_argument(1);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid tensorKey parameter: " + std::string(e.what()) + "\n\n"
            "The second parameter must be a STRING containing the tensor key.\n"
            "Example: CALL gnn_hnsw_create('my_index', 'node_features')"
        );
    }

    if (tensor_key.empty()) {
        throw std::runtime_error(
            "Invalid tensor key: key cannot be empty.\n"
            "Provide a non-empty string as the second argument.\n"
            "Example: CALL gnn_hnsw_create('my_index', 'node_features')"
        );
    }

    // Step 4: Parse optional options map
    HNSW::MetricType metric = HNSW::MetricType::COSINE_DISTANCE;
    uint64_t M = 16;
    uint64_t ef_construction = 200;
    size_t num_threads = std::thread::hardware_concurrency();  // Default: use all cores

    if (ctx.arguments.size() >= 3) {
        try {
            parse_options(ctx, 2, metric, M, ef_construction, num_threads);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Invalid options parameter: " + std::string(e.what()) + "\n\n"
                "The third parameter must be a MAP with optional keys:\n"
                "  - metric (STRING): 'cosine', 'euclidean', or 'manhattan' (default: 'cosine')\n"
                "  - M (INT): Max neighbors per node (default: 16)\n"
                "  - efConstruction (INT): Candidates during build (default: 200)\n"
                "  - threads (INT): Number of threads for parallel build (default: all cores)\n\n"
                "Example:\n"
                "  CALL gnn_hnsw_create('idx', 'node_features', {\n"
                "      metric: 'cosine',\n"
                "      M: 32,\n"
                "      efConstruction: 400,\n"
                "      threads: 4\n"
                "  })"
            );
        }
    }

    // Step 5: Check if index already exists
    auto* existing_index = gql_model.catalog.hnsw_index_manager.get_hnsw_index(index_name);
    if (existing_index != nullptr) {
        throw std::runtime_error(
            "HNSW index '" + index_name + "' already exists.\n\n"
            "Solutions:\n"
            "  1. Use a different name for the new index\n"
            "  2. Delete the existing one first:\n"
            "     CALL gnn_hnsw_drop('" + index_name + "')\n"
            "  3. List existing indexes:\n"
            "     CALL gnn_hnsw_list() YIELD indexName"
        );
    }

    // Step 6: Check if GNN tensors are available
    if (!gql_model.catalog.has_gnn_tensors) {
        throw std::runtime_error(
            "No GNN tensors found in this database.\n\n"
            "To use gnn_hnsw_create(), import your database with tensors:\n"
            "  mdb-import --with-tensors <tensor_file.npy> <db_folder> <gql_file>\n\n"
            "The tensor file should be a NumPy array with shape [num_nodes, embedding_dim]."
        );
    }

    // num_nodes and dimension are needed after tensor_store is released (for yield results)
    uint64_t num_nodes = 0;
    uint64_t dimension = 0;

    // Step 10: Create HNSW index
    // Wrap create-build-write-load in try/catch so that if any step after
    // HNSWIndex::create() throws, we clean up the on-disk directory it created.
    auto start_time = std::chrono::high_resolution_clock::now();
    uint_fast32_t indexed_count;
    decltype(std::chrono::high_resolution_clock::now() - start_time) build_duration{};

    try {

    // hnsw_index declared here so it survives the inner tensor_store scope
    std::unique_ptr<HNSW::HNSWIndex> hnsw_index;

    // Steps 7-9, 11: Load tensor store, validate, create + build the HNSW index.
    // Scoped so the tensor store's mmap (~2.3GB for 799K x 768d x float32)
    // is released as soon as HNSW indexing finishes — before write_to_disk.
    // The HNSW index keeps its own copy of the vectors, so the mmap is
    // no longer needed after index_from_raw_embeddings returns.
    {
    // Step 7: Load GNN tensor store
    std::string db_folder = get_db_folder();

    std::string gnn_path = db_folder + "/gnn_tensors";
    mdb::gnn::FileGnnTensorStore tensor_store(gnn_path, mdb::gnn::FileGnnTensorStore::DEFAULT_MAX_SHARD_SIZE, false);

    // Step 8: Load the tensor
    auto tensor_view = tensor_store.load(tensor_key);
    if (!tensor_view.valid()) {
        // List available keys for helpful error message
        auto keys = tensor_store.list_keys();
        throw std::runtime_error(
            format_not_found_error("tensor key", tensor_key, keys)
        );
    }

    // Step 9: Validate tensor shape
    const auto& shape = tensor_view.shape();
    if (shape.size() != 2) {
        throw std::runtime_error(
            "Invalid tensor shape: expected 2D tensor [num_nodes, embedding_dim], "
            "got " + std::to_string(shape.size()) + "D tensor."
        );
    }

    num_nodes = static_cast<uint64_t>(shape[0]);
    dimension = static_cast<uint64_t>(shape[1]);

    if (tensor_view.dtype() != mdb::gnn::GnnDtype::FLOAT32) {
        throw std::runtime_error(
            "Invalid tensor dtype: expected FLOAT32, got " +
            std::to_string(static_cast<int>(tensor_view.dtype()))
        );
    }

    // Step 10: Create HNSW index (after dimension is known from tensor shape)
    HNSW::MetricFuncType metric_func = HNSW::metric_type2metric_func(metric);
    hnsw_index = HNSW::HNSWIndex::create(
        index_name,
        dimension,
        M,
        ef_construction,
        metric_func
    );

    // Step 11: Index the embeddings (parallel if multiple threads specified)
    const float* embeddings = tensor_view.data_as<float>();

    // Defensive validation: ensure embeddings pointer is valid
    if (embeddings == nullptr) {
        throw std::runtime_error(
            "Failed to access tensor data for '" + tensor_key + "'.\n"
            "The tensor exists but data pointer is null - possible memory mapping failure."
        );
    }

    // Prefault pages to detect mapping issues early
    if (num_nodes > 0 && dimension > 0) {
        size_t total_bytes = static_cast<size_t>(num_nodes) * dimension * sizeof(float);
        madvise(const_cast<float*>(embeddings), total_bytes, MADV_WILLNEED);
    }

    if (num_threads > 1 && num_nodes >= 1000) {
        // Use parallel construction for large datasets
        indexed_count = hnsw_index->index_from_raw_embeddings_parallel(
            embeddings,
            num_nodes,
            dimension,
            num_threads
        );
    } else {
        // Use sequential construction for small datasets or single thread
        indexed_count = hnsw_index->index_from_raw_embeddings(
            embeddings,
            num_nodes,
            dimension
        );
    }

    } // End tensor_store scope — releases mmap (~2.3GB) before write_to_disk

    build_duration = std::chrono::high_resolution_clock::now() - start_time;

    // Step 12: Register the index with the manager
    // Note: The index is stored in the manager but we need to save it to disk
    gql_model.catalog.hnsw_index_manager.write_to_disk(index_name, hnsw_index.get());

    // Load it into the manager
    HNSW::HNSWIndexManager::HNSWIndexMetadata metadata;
    metadata.metric_type = metric;
    metadata.predicate = tensor_key;  // Use tensor key as predicate for tracking
    gql_model.catalog.hnsw_index_manager.load_hnsw_index(index_name, metadata);

    } catch (...) {
        // HNSWIndex::create() may have already created the on-disk directory.
        // Remove it so we don't leave orphaned files.
        namespace fs = std::filesystem;
        const auto relative_index_path = fs::path(HNSW::HNSWIndexManager::HNSW_INDEX_DIR) / index_name;
        const auto absolute_index_path = file_manager.get_file_path(relative_index_path);
        std::error_code ec;
        fs::remove_all(absolute_index_path, ec);
        // ec intentionally ignored — best-effort cleanup; re-throw the original error
        throw;
    }

    auto build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_duration).count();

    // Step 13: Yield results
    ctx.yield("indexName", ctx.create_string(index_name));
    ctx.yield("dimension", ctx.create_int(static_cast<int64_t>(dimension)));
    ctx.yield("nodeCount", ctx.create_int(static_cast<int64_t>(indexed_count)));
    ctx.yield("buildTimeMs", ctx.create_int(static_cast<int64_t>(build_time_ms)));
    ctx.yield_row();
}

HNSW::MetricType GnnHnswCreateProcedure::parse_metric(const std::string& metric_str) {
    std::string lower = metric_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "cosine" || lower == "cos") {
        return HNSW::MetricType::COSINE_DISTANCE;
    } else if (lower == "euclidean" || lower == "l2") {
        return HNSW::MetricType::EUCLIDEAN_DISTANCE;
    } else if (lower == "manhattan" || lower == "l1") {
        return HNSW::MetricType::MANHATTAN_DISTANCE;
    } else {
        throw std::runtime_error(
            "Unknown metric: '" + metric_str + "'.\n"
            "Supported metrics: 'cosine', 'euclidean', 'manhattan'"
        );
    }
}

void GnnHnswCreateProcedure::parse_options(
    ProcedureContext& ctx,
    size_t arg_index,
    HNSW::MetricType& metric,
    uint64_t& M,
    uint64_t& ef_construction,
    size_t& num_threads
) {
    DictOptions opts(ctx.get_argument(arg_index));

    // Parse metric
    if (auto metric_str = opts.get_string("metric")) {
        metric = parse_metric(*metric_str);
    }

    // Parse M
    if (auto v = opts.get_int("M")) {
        if (*v <= 0) {
            throw std::runtime_error("M must be positive, got: " + std::to_string(*v));
        }
        if (*v > 256) {
            throw std::runtime_error("M too large (max 256), got: " + std::to_string(*v));
        }
        M = static_cast<uint64_t>(*v);
    }

    // Parse efConstruction
    if (auto v = opts.get_int("efConstruction")) {
        if (*v <= 0) {
            throw std::runtime_error("efConstruction must be positive, got: " + std::to_string(*v));
        }
        ef_construction = static_cast<uint64_t>(*v);
    }

    // Parse threads (for parallel construction)
    if (auto v = opts.get_int("threads")) {
        if (*v < 1) {
            throw std::runtime_error("threads must be >= 1, got: " + std::to_string(*v));
        }

        // Security check: prevent requesting more threads than available
        const size_t max_threads = std::thread::hardware_concurrency();
        if (max_threads > 0 && static_cast<size_t>(*v) > max_threads) {
            throw std::runtime_error(
                "SECURITY WARNING: Requested " + std::to_string(*v) + " threads, "
                "but only " + std::to_string(max_threads) + " hardware threads available.\n\n"
                "Requesting more threads than available can cause:\n"
                "  - Excessive context switching overhead\n"
                "  - Memory pressure from thread stacks\n"
                "  - Potential system instability\n\n"
                "Please specify threads <= " + std::to_string(max_threads) + " or omit the option to use all available cores."
            );
        }

        num_threads = static_cast<size_t>(*v);
    }
}
