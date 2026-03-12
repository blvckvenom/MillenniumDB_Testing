#include "gnn_hnsw_create_procedure.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gnn/storage/feature_matrix.h"
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

namespace {
    constexpr uint64_t DEFAULT_M = 16;
    constexpr uint64_t DEFAULT_EF_CONSTRUCTION = 200;
    // Max M (neighbors per node) capped at 256.
    // Rationale: Each node stores M bidirectional links. At M=256 with 4-byte IDs,
    // that's ~2KB per node just for links. For 799K nodes, M=256 uses ~1.5GB.
    constexpr uint64_t MAX_M = 256;
} // anonymous namespace

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
            "  - tensorKey (STRING): Name of the feature matrix (e.g., 'node_features')\n"
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

    // Validate index_name is safe for filesystem paths
    validate_safe_name(index_name, "indexName");

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

    // Validate tensor_key is safe for filesystem paths
    validate_safe_name(tensor_key, "tensorKey");

    // Step 4: Parse optional options map
    HNSW::MetricType metric = HNSW::MetricType::COSINE_DISTANCE;
    uint64_t M = DEFAULT_M;
    uint64_t ef_construction = DEFAULT_EF_CONSTRUCTION;
    const unsigned int hw_threads = std::thread::hardware_concurrency();
    size_t num_threads = (hw_threads > 0) ? hw_threads : 1;  // Default: use all cores (fallback to 1 if unknown)

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
    if (!gql_model.catalog.has_gnn_features()) {
        throw std::runtime_error(
            "No GNN tensors found in this database.\n\n"
            "To use gnn_hnsw_create(), import your database with tensors:\n"
            "  mdb-import --with-tensors <tensor_file.npy> <db_folder> <gql_file>\n\n"
            "The tensor file should be a NumPy array with shape [num_nodes, embedding_dim]."
        );
    }

    // Step 6b: Validate tensor_key is registered in the catalog
    {
        const auto& names = gql_model.catalog.gnn_feature_names;
        if (std::find(names.begin(), names.end(), tensor_key) == names.end()) {
            std::string available;
            for (size_t i = 0; i < names.size(); ++i) {
                if (i > 0) available += ", ";
                available += "'" + names[i] + "'";
            }
            throw std::runtime_error(
                "Feature '" + tensor_key + "' is not registered in the catalog.\n\n"
                "Available features: " + available + "\n"
                "Only features imported via mdb-import --with-tensors are valid."
            );
        }
    }

    // num_nodes and dimension are needed after tensor_store is released (for yield results)
    uint64_t num_nodes = 0;
    uint64_t dimension = 0;

    // Step 10: Create HNSW index
    // Wrap create-build-write-load in try/catch so that if any step after
    // HNSWIndex::create() throws, we clean up the on-disk directory it created.
    auto start_time = std::chrono::high_resolution_clock::now();
    uint_fast32_t indexed_count = 0;
    decltype(std::chrono::high_resolution_clock::now() - start_time) build_duration{};

    try {

    // hnsw_index declared here so it survives the inner FeatureMatrix scope
    std::unique_ptr<HNSW::HNSWIndex> hnsw_index;

    // Steps 7-9, 11: Load FeatureMatrix, validate, create + build the HNSW index.
    // Scoped so the FeatureMatrix mmap is released as soon as HNSW indexing
    // finishes — before write_to_disk. The HNSW index keeps its own copy of
    // the vectors, so the mmap is no longer needed after indexing returns.
    {
    // Step 7: Open FeatureMatrix for the requested tensor key
    std::string db_folder = get_db_folder();
    namespace fs = std::filesystem;

    auto fmat_path = fs::path(db_folder) / "gnn_features" / (tensor_key + ".fmat");
    if (!fs::exists(fmat_path)) {
        // List available features from catalog for helpful error message
        const auto& names = gql_model.catalog.gnn_feature_names;
        throw std::runtime_error(
            format_not_found_error("feature matrix", tensor_key, names)
        );
    }

    auto feature_matrix = mdb::gnn::FeatureMatrix::open(fmat_path);

    // Step 8-9: Read dimensions from FeatureMatrix header (self-describing)
    num_nodes = feature_matrix.num_rows();
    dimension = feature_matrix.num_cols();

    // I2: HNSW internally uses uint32_t for node IDs
    if (num_nodes > static_cast<uint64_t>(UINT32_MAX)) {
        throw std::runtime_error(
            "FeatureMatrix has " + std::to_string(num_nodes) + " rows, but HNSW index "
            "supports at most " + std::to_string(UINT32_MAX) + " nodes (32-bit limit)."
        );
    }

    // I1: Support both FLOAT32 (direct mmap) and FLOAT64 (convert to float32)
    const float* embeddings = nullptr;
    std::vector<float> f32_buffer;  // Only allocated for float64 conversion

    if (feature_matrix.dtype() == mdb::gnn::GnnDtype::FLOAT32) {
        embeddings = feature_matrix.row_as<float>(0);
    } else if (feature_matrix.dtype() == mdb::gnn::GnnDtype::FLOAT64) {
        // Convert float64 -> float32 for HNSW indexing
        const double* f64_data = feature_matrix.row_as<double>(0);
        if (dimension > 0 && num_nodes > SIZE_MAX / dimension) {
            throw std::overflow_error("float64-to-float32 conversion: total_elements overflow");
        }
        size_t total_elements = static_cast<size_t>(num_nodes) * dimension;
        f32_buffer.resize(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            f32_buffer[i] = static_cast<float>(f64_data[i]);
        }
        embeddings = f32_buffer.data();
    } else {
        throw std::runtime_error(
            "Unsupported dtype for HNSW indexing: " +
            mdb::gnn::dtype_name(feature_matrix.dtype()) + ".\n\n"
            "HNSW supports FLOAT32 and FLOAT64.\n"
            "Re-import with a supported dtype, or convert with:\n"
            "  numpy: arr.astype(np.float32)"
        );
    }

    // Step 10: Create HNSW index (after dimension is known)
    HNSW::MetricFuncType metric_func = HNSW::metric_type2metric_func(metric);
    hnsw_index = HNSW::HNSWIndex::create(
        index_name,
        dimension,
        M,
        ef_construction,
        metric_func
    );

    // Step 11: Index the embeddings (parallel if multiple threads specified)
    // NOTE: HNSW find-similar reconstructs ObjectIds as (MASK_NODE | hnsw_node_id),
    // assuming a 1:1 identity mapping between row position and node ordinal.
    // If FeatureMatrix rows are ever reordered (e.g., MinHash locality optimization),
    // find-similar must be updated to use RowMapping for ObjectId lookup.
    // TODO: Use RowMapping when row reordering is introduced (Step 2+).

    // IO2: Guard overflow in total_bytes computation
    if (dimension > 0 && num_nodes > SIZE_MAX / dimension / sizeof(float)) {
        throw std::overflow_error("total_bytes computation would overflow");
    }
    size_t total_bytes = static_cast<size_t>(num_nodes) * dimension * sizeof(float);

    // IO1: Prefault pages — only for the FLOAT32 mmap path (not heap buffer).
    // Page-align the address because embeddings = mmap_base + 64-byte header,
    // which is not page-aligned. Since 64 < page_size (4096), page_addr always
    // equals mmap_base, so adj_bytes stays within the mmap region.
    if (feature_matrix.dtype() == mdb::gnn::GnnDtype::FLOAT32 && num_nodes > 0 && dimension > 0) {
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(embeddings);
            uintptr_t page_addr = addr & ~(static_cast<uintptr_t>(page_size) - 1);
            size_t adj_bytes = total_bytes + static_cast<size_t>(addr - page_addr);
            madvise(reinterpret_cast<void*>(page_addr), adj_bytes, MADV_WILLNEED);
        }
    }

    if (num_threads > 1 && num_nodes >= 1000) {
        indexed_count = hnsw_index->index_from_raw_embeddings_parallel(
            embeddings,
            num_nodes,
            dimension,
            num_threads
        );
    } else {
        indexed_count = hnsw_index->index_from_raw_embeddings(
            embeddings,
            num_nodes,
            dimension
        );
    }

    } // End FeatureMatrix scope — releases mmap before write_to_disk

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
        if (*v > static_cast<int64_t>(MAX_M)) {
            throw std::runtime_error("M too large (max " + std::to_string(MAX_M) + "), got: " + std::to_string(*v));
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

        // Resource check: prevent requesting more threads than available
        const unsigned int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) {
            // System can't determine thread count — accept any value
            num_threads = static_cast<size_t>(*v);
        } else if (static_cast<size_t>(*v) > hw_threads) {
            throw std::runtime_error(
                "RESOURCE WARNING: Requested " + std::to_string(*v) + " threads, "
                "but only " + std::to_string(hw_threads) + " hardware threads available.\n\n"
                "Requesting more threads than available can cause:\n"
                "  - Excessive context switching overhead\n"
                "  - Memory pressure from thread stacks\n"
                "  - Potential system instability\n\n"
                "Please specify threads <= " + std::to_string(hw_threads) + " or omit the option to use all available cores."
            );
        } else {
            num_threads = static_cast<size_t>(*v);
        }
    }
}
