#include "gnn_hnsw_create_procedure.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "gnn/storage/file_gnn_tensor_store.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/gql_object_id.h"
#include "storage/index/hnsw/hnsw_index.h"
#include "storage/index/hnsw/hnsw_metric.h"
#include "system/file_manager.h"

using namespace GQL;
using namespace GQL::Procedures;

void GnnHnswCreateProcedure::execute(ProcedureContext& ctx) {
    std::cerr << "[DEBUG] gnn_hnsw_create: Starting execution" << std::endl;

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

    std::cerr << "[DEBUG] gnn_hnsw_create: Parsed arguments - index_name='" << index_name
              << "', tensor_key='" << tensor_key << "', threads=" << num_threads << std::endl;

    // Step 6: Check if GNN tensors are available
    if (!gql_model.catalog.has_gnn_tensors) {
        throw std::runtime_error(
            "No GNN tensors found in this database.\n\n"
            "To use gnn_hnsw_create(), import your database with tensors:\n"
            "  mdb-import --with-tensors <tensor_file.npy> <db_folder> <gql_file>\n\n"
            "The tensor file should be a NumPy array with shape [num_nodes, embedding_dim]."
        );
    }

    // Step 7: Load GNN tensor store
    std::cerr << "[DEBUG] gnn_hnsw_create: Loading GNN tensor store" << std::endl;
    std::string db_folder = file_manager.get_file_path("");
    if (!db_folder.empty() && db_folder.back() == '/') {
        db_folder.pop_back();
    }

    std::string gnn_path = db_folder + "/gnn_tensors";
    std::cerr << "[DEBUG] gnn_hnsw_create: GNN path = " << gnn_path << std::endl;
    mdb::gnn::FileGnnTensorStore tensor_store(gnn_path, mdb::gnn::FileGnnTensorStore::DEFAULT_MAX_SHARD_SIZE, false);
    std::cerr << "[DEBUG] gnn_hnsw_create: Tensor store loaded successfully" << std::endl;

    // Step 8: Load the tensor
    std::cerr << "[DEBUG] gnn_hnsw_create: Loading tensor '" << tensor_key << "'" << std::endl;
    auto tensor_view = tensor_store.load(tensor_key);
    std::cerr << "[DEBUG] gnn_hnsw_create: tensor_view.valid() = " << tensor_view.valid() << std::endl;
    if (!tensor_view.valid()) {
        // List available keys for helpful error message
        auto keys = tensor_store.list_keys();
        std::string available;
        if (keys.empty()) {
            available = "No tensors found in the store.";
        } else {
            available = "Available tensor keys: [";
            for (size_t i = 0; i < keys.size(); i++) {
                if (i > 0) available += ", ";
                available += "'" + keys[i] + "'";
            }
            available += "]";
        }

        throw std::runtime_error(
            "Tensor '" + tensor_key + "' not found in GNN tensor store.\n\n" + available
        );
    }

    // Step 9: Validate tensor shape
    std::cerr << "[DEBUG] gnn_hnsw_create: Getting tensor shape" << std::endl;
    const auto& shape = tensor_view.shape();
    std::cerr << "[DEBUG] gnn_hnsw_create: shape.size() = " << shape.size() << std::endl;
    if (shape.size() >= 1) std::cerr << "[DEBUG] gnn_hnsw_create: shape[0] = " << shape[0] << std::endl;
    if (shape.size() >= 2) std::cerr << "[DEBUG] gnn_hnsw_create: shape[1] = " << shape[1] << std::endl;
    if (shape.size() != 2) {
        throw std::runtime_error(
            "Invalid tensor shape: expected 2D tensor [num_nodes, embedding_dim], "
            "got " + std::to_string(shape.size()) + "D tensor."
        );
    }

    uint64_t num_nodes = static_cast<uint64_t>(shape[0]);
    uint64_t dimension = static_cast<uint64_t>(shape[1]);

    if (tensor_view.dtype() != mdb::gnn::GnnDtype::FLOAT32) {
        throw std::runtime_error(
            "Invalid tensor dtype: expected FLOAT32, got " +
            std::to_string(static_cast<int>(tensor_view.dtype()))
        );
    }

    // Step 10: Create HNSW index
    std::cerr << "[DEBUG] gnn_hnsw_create: Creating HNSW index with dim=" << dimension
              << ", M=" << M << ", ef=" << ef_construction << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Get metric function
    HNSW::MetricFuncType metric_func = HNSW::metric_type2metric_func(metric);
    std::cerr << "[DEBUG] gnn_hnsw_create: Got metric function" << std::endl;

    // Create the index
    auto hnsw_index = HNSW::HNSWIndex::create(
        index_name,
        dimension,
        M,
        ef_construction,
        metric_func
    );
    std::cerr << "[DEBUG] gnn_hnsw_create: HNSW index created successfully" << std::endl;

    // Step 11: Index the embeddings (parallel if multiple threads specified)
    std::cerr << "[DEBUG] gnn_hnsw_create: Getting embeddings pointer" << std::endl;
    const float* embeddings = tensor_view.data_as<float>();
    std::cerr << "[DEBUG] gnn_hnsw_create: embeddings pointer = " << (void*)embeddings << std::endl;

    // Defensive validation: ensure embeddings pointer is valid
    if (embeddings == nullptr) {
        throw std::runtime_error(
            "Failed to access tensor data for '" + tensor_key + "'.\n"
            "The tensor exists but data pointer is null - possible memory mapping failure."
        );
    }

    // Additional validation: verify first and last embedding are accessible
    // This helps catch memory mapping issues early
    std::cerr << "[DEBUG] gnn_hnsw_create: Validating embedding access..." << std::endl;
    if (num_nodes > 0 && dimension > 0) {
        std::cerr << "[DEBUG] gnn_hnsw_create: Accessing first element" << std::endl;
        volatile float first_check = embeddings[0];
        std::cerr << "[DEBUG] gnn_hnsw_create: first_check = " << first_check << std::endl;
        std::cerr << "[DEBUG] gnn_hnsw_create: Accessing last element at offset " << ((num_nodes - 1) * dimension) << std::endl;
        volatile float last_check = embeddings[(num_nodes - 1) * dimension];
        std::cerr << "[DEBUG] gnn_hnsw_create: last_check = " << last_check << std::endl;
        (void)first_check;
        (void)last_check;
    }
    std::cerr << "[DEBUG] gnn_hnsw_create: Embedding validation passed" << std::endl;

    uint_fast32_t indexed_count;

    std::cerr << "[DEBUG] gnn_hnsw_create: Starting indexing with num_threads=" << num_threads
              << ", num_nodes=" << num_nodes << std::endl;

    if (num_threads > 1 && num_nodes >= 1000) {
        // Use parallel construction for large datasets
        std::cerr << "[DEBUG] gnn_hnsw_create: Using PARALLEL construction" << std::endl;
        indexed_count = hnsw_index->index_from_raw_embeddings_parallel(
            embeddings,
            num_nodes,
            dimension,
            num_threads
        );
    } else {
        // Use sequential construction for small datasets or single thread
        std::cerr << "[DEBUG] gnn_hnsw_create: Using SEQUENTIAL construction" << std::endl;
        indexed_count = hnsw_index->index_from_raw_embeddings(
            embeddings,
            num_nodes,
            dimension
        );
    }
    std::cerr << "[DEBUG] gnn_hnsw_create: Indexing complete, indexed_count=" << indexed_count << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    // Step 12: Register the index with the manager
    // Note: The index is stored in the manager but we need to save it to disk
    gql_model.catalog.hnsw_index_manager.write_to_disk(index_name, hnsw_index.get());

    // Load it into the manager
    HNSW::HNSWIndexManager::HNSWIndexMetadata metadata;
    metadata.metric_type = metric;
    metadata.predicate = tensor_key;  // Use tensor key as predicate for tracking
    gql_model.catalog.hnsw_index_manager.load_hnsw_index(index_name, metadata);

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
    ObjectId arg = ctx.get_argument(arg_index);
    auto type = GQL_OID::get_type(arg);

    if (type != GQL_OID::Type::DICTIONARY) {
        throw std::runtime_error(
            "options must be a MAP/DICTIONARY, got type: " +
            std::to_string(static_cast<int>(type))
        );
    }

    // Unpack dictionary
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(arg);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Failed to parse options map");
    }

    // Helper to get value from dictionary
    auto get_value = [&](const std::string& key) -> std::optional<ObjectId> {
        for (const auto& [key_oid, val_item] : dict_obj->keys) {
            std::string key_str = Conversions::unpack_string(key_oid);
            if (key_str == key) {
                auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
                if (lit) {
                    return lit->object_id;
                }
            }
        }
        return std::nullopt;
    };

    // Parse metric
    if (auto value = get_value("metric")) {
        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::STRING_SIMPLE_INLINE ||
            vtype == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
            vtype == GQL_OID::Type::STRING_SIMPLE_TMP)
        {
            std::string metric_str = Conversions::unpack_string(*value);
            metric = parse_metric(metric_str);
        } else {
            throw std::runtime_error("metric must be a string");
        }
    }

    // Parse M
    if (auto value = get_value("M")) {
        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::INT56_INLINE ||
            vtype == GQL_OID::Type::INT64_EXTERN ||
            vtype == GQL_OID::Type::INT64_TMP)
        {
            int64_t v = Conversions::unpack_int(*value);
            if (v <= 0) {
                throw std::runtime_error("M must be positive, got: " + std::to_string(v));
            }
            if (v > 256) {
                throw std::runtime_error("M too large (max 256), got: " + std::to_string(v));
            }
            M = static_cast<uint64_t>(v);
        } else {
            throw std::runtime_error("M must be an integer");
        }
    }

    // Parse efConstruction
    if (auto value = get_value("efConstruction")) {
        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::INT56_INLINE ||
            vtype == GQL_OID::Type::INT64_EXTERN ||
            vtype == GQL_OID::Type::INT64_TMP)
        {
            int64_t v = Conversions::unpack_int(*value);
            if (v <= 0) {
                throw std::runtime_error("efConstruction must be positive, got: " + std::to_string(v));
            }
            ef_construction = static_cast<uint64_t>(v);
        } else {
            throw std::runtime_error("efConstruction must be an integer");
        }
    }

    // Parse threads (for parallel construction)
    if (auto value = get_value("threads")) {
        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::INT56_INLINE ||
            vtype == GQL_OID::Type::INT64_EXTERN ||
            vtype == GQL_OID::Type::INT64_TMP)
        {
            int64_t v = Conversions::unpack_int(*value);
            if (v < 1) {
                throw std::runtime_error("threads must be >= 1, got: " + std::to_string(v));
            }

            // Security check: prevent requesting more threads than available
            const size_t max_threads = std::thread::hardware_concurrency();
            if (max_threads > 0 && static_cast<size_t>(v) > max_threads) {
                throw std::runtime_error(
                    "SECURITY WARNING: Requested " + std::to_string(v) + " threads, "
                    "but only " + std::to_string(max_threads) + " hardware threads available.\n\n"
                    "Requesting more threads than available can cause:\n"
                    "  - Excessive context switching overhead\n"
                    "  - Memory pressure from thread stacks\n"
                    "  - Potential system instability\n\n"
                    "Please specify threads <= " + std::to_string(max_threads) + " or omit the option to use all available cores."
                );
            }

            num_threads = static_cast<size_t>(v);
        } else {
            throw std::runtime_error("threads must be an integer");
        }
    }
}
