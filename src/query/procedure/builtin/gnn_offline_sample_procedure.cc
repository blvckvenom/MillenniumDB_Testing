#include "gnn_offline_sample_procedure.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "gnn/sampling/offline_sampling_engine.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_config.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_object_id.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "gnn_procedure_utils.h"

using namespace GQL;
using namespace GQL::Procedures;
using namespace mdb::gnn;

void GnnOfflineSampleProcedure::execute(ProcedureContext& ctx) {
    // Step 1: Validate argument count
    if (ctx.arguments.size() < 3 || ctx.arguments.size() > 4) {
        throw std::runtime_error(
            "gnn.offline_sample() requires 3-4 arguments, got " +
            std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL gnn.offline_sample(projectionName, sampleName, fanouts [, options])\n"
            "  YIELD sampleName, totalBatches, trainBatches, ...\n\n"
            "Parameters:\n"
            "  - projectionName (STRING): Source graph projection\n"
            "  - sampleName (STRING): Name for the sample set\n"
            "  - fanouts (LIST<INT>): Fanouts per layer, e.g., [15, 10]\n"
            "  - options (MAP, optional): batchSize, trainRatio, randomSeed, etc.\n\n"
            "Examples:\n"
            "  CALL gnn.offline_sample('social', 'samples_v1', [15, 10])\n"
            "  CALL gnn.offline_sample('social', 'samples_v1', [15, 10], {batchSize: 512})"
        );
    }

    // Step 2: Parse projectionName
    std::string projection_name;
    try {
        projection_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid projectionName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING containing the projection name.\n"
            "Example: CALL gnn.offline_sample('myProjection', ...)"
        );
    }

    if (projection_name.empty()) {
        throw std::runtime_error(
            "Invalid projection name: name cannot be empty.\n"
            "Provide a non-empty string as the first argument.\n"
            "Example: CALL gnn.offline_sample('myProjection', ...)"
        );
    }

    // Step 3: Parse sampleName
    std::string sample_name;
    try {
        sample_name = ctx.get_string_argument(1);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid sampleName parameter: " + std::string(e.what()) + "\n\n"
            "The second parameter must be a STRING containing the sample set name.\n"
            "Example: CALL gnn.offline_sample('projection', 'mySamples', ...)"
        );
    }

    if (sample_name.empty()) {
        throw std::runtime_error(
            "Invalid sample name: name cannot be empty.\n"
            "Provide a non-empty string as the second argument.\n"
            "Example: CALL gnn.offline_sample('projection', 'mySamples', ...)"
        );
    }

    // Step 4: Parse fanouts list
    std::vector<uint64_t> fanouts;
    try {
        fanouts = parse_fanouts(ctx, 2);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid fanouts parameter: " + std::string(e.what()) + "\n\n"
            "The third parameter must be a LIST of positive integers.\n"
            "Each value specifies neighbors to sample at that hop distance.\n\n"
            "Examples:\n"
            "  [15, 10]       - 2-hop: 15 neighbors at 1-hop, 10 at 2-hop\n"
            "  [15, 10, 5]    - 3-hop: 15, 10, 5 neighbors at each layer\n"
            "  [25]           - 1-hop: 25 neighbors"
        );
    }

    // Step 5: Parse optional options map
    uint64_t batch_size = 1024;
    double train_ratio = 0.7;
    double val_ratio = 0.15;
    double test_ratio = 0.15;
    uint64_t random_seed = 42;
    std::string orientation_str = "REVERSE";

    if (ctx.arguments.size() >= 4) {
        try {
            parse_options(ctx, 3, batch_size, train_ratio, val_ratio,
                          test_ratio, random_seed, orientation_str);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Invalid options parameter: " + std::string(e.what()) + "\n\n"
                "The fourth parameter must be a MAP with optional keys:\n"
                "  - batchSize (INT): Seeds per batch (default: 1024)\n"
                "  - trainRatio (FLOAT): Training fraction (default: 0.7)\n"
                "  - validationRatio (FLOAT): Validation fraction (default: 0.15)\n"
                "  - testRatio (FLOAT): Test fraction (default: 0.15)\n"
                "  - randomSeed (INT): For reproducibility (default: 42)\n"
                "  - orientation (STRING): NATURAL, REVERSE, or UNDIRECTED (default: REVERSE)\n\n"
                "Example:\n"
                "  CALL gnn.offline_sample('proj', 'samples', [15, 10], {\n"
                "      batchSize: 512,\n"
                "      trainRatio: 0.8,\n"
                "      validationRatio: 0.1,\n"
                "      testRatio: 0.1,\n"
                "      randomSeed: 12345\n"
                "  })"
            );
        }
    }

    // Step 6: Verify projection exists
    auto& manager = ProjectionManager::get_instance();
    if (!manager.projection_exists(projection_name)) {
        // Provide helpful error with available projections
        auto projections = manager.list_projections();
        std::string available;
        if (projections.empty()) {
            available = "No projections exist. Create one first with:\n"
                        "  CALL graph_project('name', 'NodeLabel', 'EdgeType')";
        } else {
            available = "Available projections: [";
            for (size_t i = 0; i < projections.size(); i++) {
                if (i > 0) available += ", ";
                available += "'" + projections[i].name + "'";
            }
            available += "]";
        }

        throw std::runtime_error(
            "Projection '" + projection_name + "' not found.\n\n" + available
        );
    }

    // Step 7: Check if sample already exists
    std::string db_folder = get_db_folder();

    if (SampleStorage::exists(db_folder, sample_name)) {
        throw std::runtime_error(
            "Sample set '" + sample_name + "' already exists.\n\n"
            "Solutions:\n"
            "  1. Use a different name for the new sample set\n"
            "  2. Delete the existing one first:\n"
            "     CALL gnn.sample_drop('" + sample_name + "')\n"
            "  3. List existing samples:\n"
            "     CALL gnn.sample_list() YIELD sampleName"
        );
    }

    // Step 8: Parse orientation
    EdgeOrientation orientation = EdgeOrientation::REVERSE;
    std::string upper_orientation = orientation_str;
    std::transform(upper_orientation.begin(), upper_orientation.end(),
                   upper_orientation.begin(), ::toupper);

    if (upper_orientation == "NATURAL") {
        orientation = EdgeOrientation::NATURAL;
    } else if (upper_orientation == "REVERSE") {
        orientation = EdgeOrientation::REVERSE;
    } else if (upper_orientation == "UNDIRECTED") {
        orientation = EdgeOrientation::UNDIRECTED;
    } else {
        throw std::runtime_error(
            "Invalid orientation: '" + orientation_str + "'.\n"
            "Must be 'NATURAL', 'REVERSE', or 'UNDIRECTED' (case-insensitive)."
        );
    }

    // Step 9: Build SamplingConfig
    SamplingConfig config;
    config.projection_name = projection_name;
    config.sample_name = sample_name;
    config.fanouts = fanouts;
    config.batch_size = batch_size;
    config.train_ratio = train_ratio;
    config.val_ratio = val_ratio;
    config.test_ratio = test_ratio;
    config.random_seed = random_seed;
    config.orientation = orientation;

    // Validate config
    try {
        config.validate();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid sampling configuration: " + std::string(e.what())
        );
    }

    // Step 10: Open projection storage
    std::string proj_dir = manager.get_projection_dir(projection_name);
    ProjectionStorage storage(proj_dir, db_folder);
    storage.open();

    // Step 11: Create and run engine
    OfflineSamplingEngine engine(storage, config);
    SamplingResult result = engine.run(db_folder);

    // Step 12: Handle result
    if (!result.success) {
        if (result.cancelled) {
            throw std::runtime_error("Sampling was cancelled.");
        }
        throw std::runtime_error(
            "Sampling failed: " + result.error_message
        );
    }

    // Step 13: Yield results
    auto storage_path = SampleStorage::get_storage_path(db_folder, sample_name);
    int64_t compute_millis = static_cast<int64_t>(result.total_seconds * 1000);

    ctx.yield("sampleName", ctx.create_string(sample_name));
    ctx.yield("projectionName", ctx.create_string(projection_name));
    ctx.yield("totalBatches", ctx.create_int(static_cast<int64_t>(result.catalog.total_batches)));
    ctx.yield("trainBatches", ctx.create_int(static_cast<int64_t>(result.catalog.train_batches)));
    ctx.yield("validationBatches", ctx.create_int(static_cast<int64_t>(result.catalog.validation_batches)));
    ctx.yield("testBatches", ctx.create_int(static_cast<int64_t>(result.catalog.test_batches)));
    ctx.yield("uniqueNodes", ctx.create_int(static_cast<int64_t>(result.catalog.unique_nodes)));
    ctx.yield("storagePath", ctx.create_string(storage_path.string()));
    ctx.yield("computeMillis", ctx.create_int(compute_millis));
    ctx.yield_row();
}

std::vector<uint64_t> GnnOfflineSampleProcedure::parse_fanouts(
    ProcedureContext& ctx,
    size_t arg_index
) {
    ObjectId arg = ctx.get_argument(arg_index);
    auto type = GQL_OID::get_type(arg);

    if (type != GQL_OID::Type::LIST) {
        throw std::runtime_error(
            "fanouts must be a LIST, got type: " +
            std::to_string(static_cast<int>(type))
        );
    }

    std::vector<ObjectId> list_items = Conversions::unpack_list(arg);

    if (list_items.empty()) {
        throw std::runtime_error(
            "fanouts list cannot be empty. "
            "Provide at least one fanout value (e.g., [15] for 1-hop sampling)."
        );
    }

    std::vector<uint64_t> fanouts;
    fanouts.reserve(list_items.size());

    for (size_t i = 0; i < list_items.size(); i++) {
        const auto& item_oid = list_items[i];
        auto item_type = GQL_OID::get_type(item_oid);

        // Accept integers
        if (item_type == GQL_OID::Type::INT56_INLINE ||
            item_type == GQL_OID::Type::INT64_EXTERN ||
            item_type == GQL_OID::Type::INT64_TMP)
        {
            int64_t value = Conversions::unpack_int(item_oid);
            if (value <= 0) {
                throw std::runtime_error(
                    "fanouts[" + std::to_string(i) + "] must be positive, got: " +
                    std::to_string(value)
                );
            }
            fanouts.push_back(static_cast<uint64_t>(value));
        } else {
            throw std::runtime_error(
                "fanouts[" + std::to_string(i) + "] must be an integer, got type: " +
                std::to_string(static_cast<int>(item_type))
            );
        }
    }

    return fanouts;
}

void GnnOfflineSampleProcedure::parse_options(
    ProcedureContext& ctx,
    size_t arg_index,
    uint64_t& batch_size,
    double& train_ratio,
    double& val_ratio,
    double& test_ratio,
    uint64_t& random_seed,
    std::string& orientation
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

    // Parse batchSize
    if (auto value = get_value("batchSize")) {
        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::INT56_INLINE ||
            vtype == GQL_OID::Type::INT64_EXTERN ||
            vtype == GQL_OID::Type::INT64_TMP)
        {
            int64_t v = Conversions::unpack_int(*value);
            if (v <= 0) {
                throw std::runtime_error("batchSize must be positive, got: " + std::to_string(v));
            }
            batch_size = static_cast<uint64_t>(v);
        } else {
            throw std::runtime_error("batchSize must be an integer");
        }
    }

    // Helper to parse double from various numeric types
    auto parse_double = [](ObjectId oid, const std::string& name) -> double {
        auto vtype = GQL_OID::get_type(oid);
        if (vtype == GQL_OID::Type::INT56_INLINE ||
            vtype == GQL_OID::Type::INT64_EXTERN ||
            vtype == GQL_OID::Type::INT64_TMP)
        {
            return static_cast<double>(Conversions::unpack_int(oid));
        }
        if (vtype == GQL_OID::Type::FLOAT32) {
            return static_cast<double>(Conversions::unpack_float(oid));
        }
        if (vtype == GQL_OID::Type::DOUBLE64_EXTERN ||
            vtype == GQL_OID::Type::DOUBLE64_TMP)
        {
            return Conversions::unpack_double(oid);
        }
        if (vtype == GQL_OID::Type::DECIMAL_INLINE ||
            vtype == GQL_OID::Type::DECIMAL_EXTERN ||
            vtype == GQL_OID::Type::DECIMAL_TMP)
        {
            return Common::Conversions::unpack_decimal(oid).to_double();
        }
        throw std::runtime_error(name + " must be a numeric value");
    };

    // Parse trainRatio
    if (auto value = get_value("trainRatio")) {
        train_ratio = parse_double(*value, "trainRatio");
        if (train_ratio < 0.0 || train_ratio > 1.0) {
            throw std::runtime_error("trainRatio must be between 0.0 and 1.0");
        }
    }

    // Parse validationRatio
    if (auto value = get_value("validationRatio")) {
        val_ratio = parse_double(*value, "validationRatio");
        if (val_ratio < 0.0 || val_ratio > 1.0) {
            throw std::runtime_error("validationRatio must be between 0.0 and 1.0");
        }
    }

    // Parse testRatio
    if (auto value = get_value("testRatio")) {
        test_ratio = parse_double(*value, "testRatio");
        if (test_ratio < 0.0 || test_ratio > 1.0) {
            throw std::runtime_error("testRatio must be between 0.0 and 1.0");
        }
    }

    // Validate ratios sum to 1.0
    double total = train_ratio + val_ratio + test_ratio;
    if (total < 0.999 || total > 1.001) {
        throw std::runtime_error(
            "trainRatio + validationRatio + testRatio must equal 1.0, got: " +
            std::to_string(total)
        );
    }

    // Parse randomSeed
    if (auto value = get_value("randomSeed")) {
        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::INT56_INLINE ||
            vtype == GQL_OID::Type::INT64_EXTERN ||
            vtype == GQL_OID::Type::INT64_TMP)
        {
            random_seed = static_cast<uint64_t>(Conversions::unpack_int(*value));
        } else {
            throw std::runtime_error("randomSeed must be an integer");
        }
    }

    // Parse orientation
    if (auto value = get_value("orientation")) {
        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::STRING_SIMPLE_INLINE ||
            vtype == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
            vtype == GQL_OID::Type::STRING_SIMPLE_TMP)
        {
            orientation = Conversions::unpack_string(*value);
        } else {
            throw std::runtime_error("orientation must be a string");
        }
    }
}
