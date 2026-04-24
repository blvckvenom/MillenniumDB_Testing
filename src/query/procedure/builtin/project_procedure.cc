#include "project_procedure.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/gql_object_id.h"
#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "query/exceptions.h"
#include "storage/dictionary/dictionary.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "system/file_manager.h"

using namespace GQL;
using namespace GQL::Procedures;

// =============================================================================
// Projection name validation (path traversal protection)
// =============================================================================

void ProjectProcedure::validate_projection_name(const std::string& name) {
    if (name.empty()) {
        throw std::runtime_error(
            "Invalid projection name: name cannot be empty.\n"
            "Provide a non-empty string as the first argument.\n"
            "Example: CALL PROJECT('myProjection', ...)");
    }
    if (name.find_first_not_of(" \t\n\r") == std::string::npos) {
        throw std::runtime_error(
            "Invalid projection name: name cannot be whitespace only.\n"
            "Provide a meaningful name for your projection.\n"
            "Example: CALL PROJECT('myProjection', ...)");
    }
    if (name == "." || name == "..") {
        throw std::runtime_error(
            "Invalid projection name: '" + name + "' is not allowed.\n"
            "Projection names cannot be '.' or '..'.");
    }
    for (char c : name) {
        if (c == '/' || c == '\\' || c == '\0') {
            throw std::runtime_error(
                "Invalid projection name: '" + name + "' contains a path separator or null byte.\n"
                "Projection names cannot contain '/', '\\', or null characters.");
        }
        if (static_cast<unsigned char>(c) < 0x20) {
            throw std::runtime_error(
                "Invalid projection name: '" + name + "' contains a control character.\n"
                "Projection names cannot contain control characters (bytes < 0x20).");
        }
    }
}

// =============================================================================
// Helper: deduplicate a string vector preserving insertion order
// =============================================================================

static void deduplicate(std::vector<std::string>& vec) {
    std::unordered_set<std::string> seen;
    auto it = std::remove_if(vec.begin(), vec.end(),
        [&seen](const std::string& s) { return !seen.insert(s).second; });
    vec.erase(it, vec.end());
}

// =============================================================================
// Main execution
// =============================================================================

void ProjectProcedure::execute(ProcedureContext& ctx) {
    // Step 1: Validate parameter count
    if (ctx.arguments.size() < 3 || ctx.arguments.size() > 4) {
        throw std::runtime_error(
            "PROJECT procedure requires 3-4 parameters, got " +
            std::to_string(ctx.arguments.size()) + ".\n\n"
            "Usage:\n"
            "  CALL PROJECT(graphName, nodeProjection, relationshipProjection [, config])\n"
            "  YIELD graphName, nodeCount, relationshipCount, projectMillis\n\n"
            "Parameters:\n"
            "  - graphName (STRING): Name of the projection to create\n"
            "  - nodeProjection (STRING | LIST<STRING> | MAP): Label(s) or per-label config\n"
            "  - relationshipProjection (STRING | LIST<STRING> | MAP): Type(s) or per-type config\n"
            "  - configuration (MAP, optional): Global config options\n\n"
            "Examples:\n"
            "  CALL PROJECT('myGraph', 'User', 'KNOWS')\n"
            "  CALL PROJECT('social', ['User', 'Post'], ['KNOWS', 'LIKES'])\n"
            "  CALL PROJECT('gnn', 'User', {KNOWS: {orientation: 'UNDIRECTED'}})"
        );
    }

    // Step 2: Parse and validate graphName
    std::string graph_name;
    try {
        graph_name = parse_graph_name(ctx);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid graphName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING containing the projection name.\n"
            "Example: CALL PROJECT('myProjection', ...)");
    }
    validate_projection_name(graph_name);

    // Step 3: Parse optional config map ONCE (needed for global defaults)
    std::vector<std::string> global_node_properties;
    std::vector<std::string> global_edge_properties;
    Orientation global_orientation = Orientation::NATURAL;
    Aggregation global_aggregation = Aggregation::SINGLE;
    std::string global_aggregation_property;

    // GNN extension fields (optional, default to empty = disabled)
    std::string include_features;
    std::string label_property;
    std::string split_property;

    // Disk-cost opt-out: when false, the projection skips the four label
    // B+Tree indexes (node_label, label_node, edge_label, label_edge).
    // Default true preserves Neo4j-GDS parity and existing behavior. See
    // analysis doc §3.A for the rationale and safety analysis.
    bool include_label_indexes = true;
    GQL::IndexSet index_set = GQL::IndexSet::ALL;

    // Spec #5 T5.11 — user-selectable leaf-page encoding. Defaults to BITSET
    // (pre-Spec-#5 byte-identical behavior); the config key `leafFormat`
    // opts the projection into DELTA_VARINT v2 leaves. Threaded through
    // NativeProjectionBuilder -> ProjectionStorage -> per-index BPlusTree
    // readers, and persisted per materialized index in catalog v1.5.
    BPT::LeafFormat leaf_format = BPT::LeafFormat::BITSET;

    // Spec #4-B T4.8 — opt-in CSR topology sidecar generation. Default
    // false so existing projections remain byte-identical on disk and
    // pre-Spec-#4-B callers see zero behavior change. The flag is parsed
    // below from the config map; non-bool values raise QueryException.
    bool build_topology_snapshot = false;

    // Keep config_holder alive so config_dict pointer remains valid
    std::unique_ptr<Dictionary> config_holder;
    DictionaryObject* config_dict = nullptr;

    if (ctx.arguments.size() >= 4) {
        ObjectId config_arg = ctx.get_argument(3);
        auto config_type = GQL_OID::get_type(config_arg);

        if (config_type == GQL_OID::Type::DICTIONARY) {
            config_holder = Common::Conversions::unpack_dictionary(config_arg);
            config_dict = dynamic_cast<DictionaryObject*>(config_holder->dictionary.get());
            if (!config_dict) {
                throw std::runtime_error("Configuration parameter must be a dictionary/map");
            }
        }
    }

    if (config_dict) {
        global_node_properties = parse_property_list_from_dict(config_dict, "nodeProperties");
        global_edge_properties = parse_property_list_from_dict(config_dict, "relationshipProperties");
        global_orientation = get_orientation_from_dict(config_dict, "orientation", Orientation::NATURAL);
        global_aggregation = get_aggregation_from_dict(config_dict, "aggregation", Aggregation::SINGLE);
        global_aggregation_property = resolve_aggregation_property(
            config_dict, global_edge_properties, global_aggregation);

        // GNN extension fields
        include_features = get_string_from_dict(config_dict, "includeFeatures", "");
        label_property   = get_string_from_dict(config_dict, "labelProperty", "");
        split_property   = get_string_from_dict(config_dict, "splitProperty", "");

        // Spec #3: user-selectable index set preset. Defaults to "ALL" which
        // preserves the pre-Spec-#3 behavior of materializing every index.
        // Invalid values raise QueryException from parse_index_set().
        {
            std::string index_set_str =
                get_string_from_dict(config_dict, "indexSet", "ALL");
            index_set = GQL::parse_index_set(index_set_str);
        }

        // Spec #5 T5.11 — user-selectable leaf-page encoding. Case-sensitive
        // parse of "BITSET" / "DELTA_VARINT" via BPT::parse_leaf_format;
        // unknown values raise std::invalid_argument which we convert to
        // QueryException so GQL callers get a proper query-level error.
        // Non-string config values are already rejected by
        // get_string_from_dict() (throws runtime_error with type info).
        {
            bool lf_found = false;
            (void) get_value_from_dict(config_dict, "leafFormat", lf_found);
            if (lf_found) {
                std::string leaf_format_str =
                    get_string_from_dict(config_dict, "leafFormat", "BITSET");
                try {
                    leaf_format = BPT::parse_leaf_format(leaf_format_str);
                } catch (const std::invalid_argument& e) {
                    throw QueryException(e.what());
                }
            }
        }

        // Disk-cost opt-out. The key is parsed inline because it is the only
        // boolean option currently supported by graph_project() and adding a
        // reusable helper is not yet justified by a second caller.
        {
            bool found = false;
            ObjectId v = get_value_from_dict(config_dict, "includeLabelIndexes", found);
            if (found) {
                auto t = GQL_OID::get_type(v);
                if (t != GQL_OID::Type::BOOL) {
                    throw std::runtime_error(
                        "Configuration value for 'includeLabelIndexes' must be a boolean "
                        "(true or false).");
                }
                include_label_indexes = Common::Conversions::unpack_bool(v);
            }
        }

        // Spec #4-B T4.8 — `buildTopologySnapshot`. Parsed inline in the same
        // style as `includeLabelIndexes`. Non-bool types are rejected up front
        // with a clear QueryException so mis-typed values never silently fall
        // through to the default.
        {
            bool found = false;
            ObjectId v = get_value_from_dict(
                config_dict, "buildTopologySnapshot", found);
            if (found) {
                auto t = GQL_OID::get_type(v);
                if (t != GQL_OID::Type::BOOL) {
                    throw QueryException(
                        "Configuration value for 'buildTopologySnapshot' must "
                        "be a boolean (true or false).");
                }
                build_topology_snapshot = Common::Conversions::unpack_bool(v);
            }
        }
    }

    // Validate includeFeatures against catalog (must be a registered FeatureMatrix)
    if (!include_features.empty()) {
        const auto& names = gql_model.catalog.gnn_feature_names;
        if (std::find(names.begin(), names.end(), include_features) == names.end()) {
            std::string msg = "feature '" + include_features + "' not found.\n\n";
            if (names.empty()) {
                msg += "No features exist. Create one first with:\n"
                       "  mdb import data.gql <db> --with-tensors features.npy";
            } else {
                msg += "Available features: [";
                for (size_t i = 0; i < names.size(); i++) {
                    if (i > 0) msg += ", ";
                    msg += "'" + names[i] + "'";
                }
                msg += "]";
            }
            throw std::runtime_error(msg);
        }
    }

    // Step 4: Parse nodeProjection (STRING, LIST, or MAP)
    NodeProjectionVariant node_projection_variant;
    try {
        node_projection_variant = parse_node_projection(ctx);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid nodeProjection parameter: " + std::string(e.what()) + "\n\n"
            "The second parameter must be either:\n"
            "  - A STRING: 'User'\n"
            "  - A LIST of STRINGs: ['User', 'Post']\n"
            "  - A MAP with per-label config: {Person: {properties: ['age']}}\n\n"
            "Examples:\n"
            "  CALL PROJECT('g', 'User', ...)\n"
            "  CALL PROJECT('g', ['User', 'Post'], ...)\n"
            "  CALL PROJECT('g', {Person: {label: 'Person', properties: ['age']}}, ...)"
        );
    }

    // Step 5: Parse relationshipProjection (STRING, LIST, or MAP)
    RelationshipProjectionVariant rel_projection_variant;
    try {
        ObjectId rel_arg = ctx.get_argument(2);
        auto rel_type = GQL_OID::get_type(rel_arg);
        if (rel_type == GQL_OID::Type::DICTIONARY) {
            // Map syntax: parse with global defaults
            rel_projection_variant = parse_relationship_projection_map(
                rel_arg, global_orientation, global_aggregation);
        } else {
            // String or list syntax
            rel_projection_variant = parse_relationship_projection(ctx);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid relationshipProjection parameter: " + std::string(e.what()) + "\n\n"
            "The third parameter must be either:\n"
            "  - A STRING: 'KNOWS'\n"
            "  - A LIST of STRINGs: ['KNOWS', 'LIKES']\n"
            "  - A MAP with per-type config: {KNOWS: {orientation: 'UNDIRECTED'}}\n\n"
            "Examples:\n"
            "  CALL PROJECT('g', 'User', 'KNOWS')\n"
            "  CALL PROJECT('g', 'User', ['KNOWS', 'LIKES'])\n"
            "  CALL PROJECT('g', 'User', {KNOWS: {orientation: 'UNDIRECTED'}})"
        );
    }

    // Step 6: Extract node labels and properties from variant
    std::vector<std::string> node_labels;
    std::vector<std::string> node_properties = global_node_properties;
    std::unordered_map<std::string, PropertyConfig> node_property_configs;

    if (std::holds_alternative<std::vector<std::string>>(node_projection_variant)) {
        node_labels = std::get<std::vector<std::string>>(node_projection_variant);
    } else {
        const auto& node_map = std::get<NodeProjectionMap>(node_projection_variant);
        for (const auto& [projected_label, config] : node_map) {
            node_labels.push_back(config.label);
            for (const auto& prop : config.simple_properties) {
                if (std::find(node_properties.begin(), node_properties.end(), prop) == node_properties.end()) {
                    node_properties.push_back(prop);
                }
            }
            for (const auto& [prop_key, prop_config] : config.property_configs) {
                const std::string& source_prop = prop_config.source_property.empty() ? prop_key : prop_config.source_property;
                if (std::find(node_properties.begin(), node_properties.end(), source_prop) == node_properties.end()) {
                    node_properties.push_back(source_prop);
                }
                node_property_configs[prop_key] = prop_config;
            }
        }
    }

    // Step 7: Extract relationship types and per-type configuration from variant
    std::vector<std::string> relationship_types;
    std::vector<std::string> edge_properties = global_edge_properties;
    std::unordered_map<std::string, PropertyConfig> edge_property_configs;
    Orientation orientation = global_orientation;
    Aggregation aggregation = global_aggregation;
    std::string aggregation_property = global_aggregation_property;

    std::unordered_map<std::string, Orientation> type_orientations;
    std::unordered_map<std::string, Aggregation> type_aggregations;
    std::unordered_map<std::string, std::string> type_agg_properties;

    if (std::holds_alternative<std::vector<std::string>>(rel_projection_variant)) {
        relationship_types = std::get<std::vector<std::string>>(rel_projection_variant);
    } else {
        const auto& rel_map = std::get<RelationshipProjectionMap>(rel_projection_variant);
        for (const auto& [projected_type, config] : rel_map) {
            relationship_types.push_back(config.type);

            type_orientations[config.type] = config.orientation;
            type_aggregations[config.type] = config.aggregation;
            if (!config.aggregation_property.empty()) {
                type_agg_properties[config.type] = config.aggregation_property;
            }

            for (const auto& prop : config.simple_properties) {
                if (std::find(edge_properties.begin(), edge_properties.end(), prop) == edge_properties.end()) {
                    edge_properties.push_back(prop);
                }
            }
            for (const auto& [prop_key, prop_config] : config.property_configs) {
                const std::string& source_prop = prop_config.source_property.empty() ? prop_key : prop_config.source_property;
                if (std::find(edge_properties.begin(), edge_properties.end(), source_prop) == edge_properties.end()) {
                    edge_properties.push_back(source_prop);
                }
                edge_property_configs[prop_key] = prop_config;
            }
        }
    }

    // Step 8: Deduplicate labels and types (user may pass duplicates)
    deduplicate(node_labels);
    deduplicate(relationship_types);

    // Step 9: Warn about missing labels/types (non-blocking)
    for (const auto& label : node_labels) {
        warn_missing_label(label);
    }
    for (const auto& type : relationship_types) {
        warn_missing_type(type);
    }

    // Step 10: Build projection
    std::string db_folder = file_manager.get_file_path("");
    if (!db_folder.empty() && db_folder.back() == '/') {
        db_folder.pop_back();
    }

    // Rollback guard: if any step of the build fails, remove the
    // half-written projection directory so the user can retry with the
    // same name. Without this we leave zombie dirs that block re-creation.
    NativeProjectionBuilder::Statistics stats;
    try {
        NativeProjectionBuilder builder(
            graph_name,
            db_folder,
            node_properties,
            edge_properties,
            orientation,
            aggregation,
            aggregation_property,
            type_orientations,
            type_aggregations,
            type_agg_properties,
            node_property_configs,
            edge_property_configs,
            include_features,
            label_property,
            split_property,
            include_label_indexes,
            index_set,
            build_topology_snapshot,
            leaf_format
        );
        builder.scan_nodes_by_labels(node_labels);
        builder.scan_edges_by_types(relationship_types);
        stats = builder.finalize();
    } catch (...) {
        // Best-effort cleanup of partial state before re-throwing so the
        // caller sees the original error (converted to HTTP 500 upstream).
        try {
            ProjectionManager::get_instance().drop_projection(graph_name);
        } catch (...) {
            // If cleanup itself fails, swallow — the user will need to
            // inspect the projections directory manually.
        }
        throw;
    }

    // Step 11: Yield results
    ctx.yield("graphName", ctx.create_string(graph_name));
    ctx.yield("nodeCount", ctx.create_int(static_cast<int64_t>(stats.node_count)));
    ctx.yield("relationshipCount", ctx.create_int(static_cast<int64_t>(stats.relationship_count)));
    ctx.yield("projectMillis", ctx.create_int(stats.duration_ms.count()));

    // GNN extension yields (populated by builder during finalize)
    ctx.yield("featureDim", ctx.create_int(static_cast<int64_t>(stats.feature_dim)));
    ctx.yield("numClasses", ctx.create_int(static_cast<int64_t>(stats.num_classes)));

    // Spec #4-B T4.8 — `topologySnapshotBytes`. Sum of the two CSR sidecar
    // files actually produced by the builder. Reports 0 when the flag was
    // false, when neither direction was eligible (IndexSet lacked both edge
    // indexes), or when stat'ing the files fails for any reason (we never
    // let stat errors surface to the client — the projection itself is the
    // real build product).
    int64_t snapshot_bytes = 0;
    try {
        std::string proj_dir = db_folder + "/projections/" + graph_name;
        std::filesystem::path fwd = std::filesystem::path(proj_dir) / "topology_fwd.csr";
        std::filesystem::path rev = std::filesystem::path(proj_dir) / "topology_rev.csr";
        std::error_code ec;
        if (std::filesystem::exists(fwd, ec)) {
            auto sz = std::filesystem::file_size(fwd, ec);
            if (!ec) {
                snapshot_bytes += static_cast<int64_t>(sz);
            }
        }
        if (std::filesystem::exists(rev, ec)) {
            auto sz = std::filesystem::file_size(rev, ec);
            if (!ec) {
                snapshot_bytes += static_cast<int64_t>(sz);
            }
        }
    } catch (...) {
        // Defensive: any filesystem exception leaves snapshot_bytes at 0.
        snapshot_bytes = 0;
    }
    ctx.yield("topologySnapshotBytes", ctx.create_int(snapshot_bytes));

    ctx.yield_row();
}

// =============================================================================
// Argument parsing
// =============================================================================

std::string ProjectProcedure::parse_graph_name(ProcedureContext& ctx) {
    return ctx.get_string_argument(0);
}

NodeProjectionVariant ProjectProcedure::parse_node_projection(ProcedureContext& ctx) {
    ObjectId arg = ctx.get_argument(1);
    auto type = GQL_OID::get_type(arg);

    // Case 1: String variant
    if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string label = Conversions::unpack_string(arg);

        if (label.empty()) {
            throw std::runtime_error(
                "nodeProjection label cannot be an empty string. "
                "Provide a valid node label or '*' for all labels.");
        }

        if (label == "*") {
            return std::vector<std::string>(
                gql_model.catalog.node_labels_str.begin(),
                gql_model.catalog.node_labels_str.end());
        }

        return std::vector<std::string>{label};
    }

    // Case 2: List variant
    if (type == GQL_OID::Type::LIST) {
        std::vector<ObjectId> list_items = Conversions::unpack_list(arg);

        if (list_items.empty()) {
            throw std::runtime_error(
                "nodeProjection list cannot be empty. "
                "Please provide at least one node label.");
        }

        std::vector<std::string> labels;
        labels.reserve(list_items.size());

        for (size_t i = 0; i < list_items.size(); i++) {
            const auto& item_oid = list_items[i];
            auto item_type = GQL_OID::get_type(item_oid);

            if (item_type != GQL_OID::Type::STRING_SIMPLE_INLINE &&
                item_type != GQL_OID::Type::STRING_SIMPLE_EXTERN &&
                item_type != GQL_OID::Type::STRING_SIMPLE_TMP)
            {
                throw std::runtime_error(
                    "nodeProjection list element at index " + std::to_string(i) +
                    " is not a string (type: " + std::to_string(static_cast<int>(item_type)) + "). "
                    "All list elements must be strings representing node labels.");
            }

            labels.push_back(Conversions::unpack_string(item_oid));
        }

        return labels;
    }

    // Case 3: Map variant (Neo4j GDS syntax)
    if (type == GQL_OID::Type::DICTIONARY) {
        return parse_node_projection_map(arg);
    }

    throw std::runtime_error(
        "nodeProjection must be STRING, LIST<STRING>, or MAP, got type: " +
        std::to_string(static_cast<int>(type)) + ". "
        "Provide either a single label string, a list of label strings, "
        "or a map with per-label configuration.");
}

RelationshipProjectionVariant ProjectProcedure::parse_relationship_projection(ProcedureContext& ctx) {
    ObjectId arg = ctx.get_argument(2);
    auto type = GQL_OID::get_type(arg);

    // Case 1: String variant
    if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string rel_type = Conversions::unpack_string(arg);

        if (rel_type.empty()) {
            throw std::runtime_error(
                "relationshipProjection type cannot be an empty string. "
                "Provide a valid relationship type or '*' for all types.");
        }

        if (rel_type == "*") {
            return std::vector<std::string>(
                gql_model.catalog.edge_labels_str.begin(),
                gql_model.catalog.edge_labels_str.end());
        }

        return std::vector<std::string>{rel_type};
    }

    // Case 2: List variant
    if (type == GQL_OID::Type::LIST) {
        std::vector<ObjectId> list_items = Conversions::unpack_list(arg);

        if (list_items.empty()) {
            throw std::runtime_error(
                "relationshipProjection list cannot be empty. "
                "Please provide at least one relationship type.");
        }

        std::vector<std::string> types;
        types.reserve(list_items.size());

        for (size_t i = 0; i < list_items.size(); i++) {
            const auto& item_oid = list_items[i];
            auto item_type = GQL_OID::get_type(item_oid);

            if (item_type != GQL_OID::Type::STRING_SIMPLE_INLINE &&
                item_type != GQL_OID::Type::STRING_SIMPLE_EXTERN &&
                item_type != GQL_OID::Type::STRING_SIMPLE_TMP)
            {
                throw std::runtime_error(
                    "relationshipProjection list element at index " + std::to_string(i) +
                    " is not a string (type: " + std::to_string(static_cast<int>(item_type)) + "). "
                    "All list elements must be strings representing relationship types.");
            }

            types.push_back(Conversions::unpack_string(item_oid));
        }

        return types;
    }

    // Case 3: Dictionary — should not reach here (handled in execute())
    if (type == GQL_OID::Type::DICTIONARY) {
        return parse_relationship_projection_map(arg, Orientation::NATURAL, Aggregation::SINGLE);
    }

    throw std::runtime_error(
        "relationshipProjection must be STRING, LIST<STRING>, or MAP, got type: " +
        std::to_string(static_cast<int>(type)) + ". "
        "Provide either a single type string, a list of type strings, "
        "or a map with per-type configuration.");
}

// =============================================================================
// Validation (non-blocking warnings)
// =============================================================================

void ProjectProcedure::warn_missing_label(const std::string& label) {
    auto it = gql_model.catalog.node_labels2id.find(label);
    if (it != gql_model.catalog.node_labels2id.end()) {
        return;
    }

    std::cerr << "[WARNING] Node label '" << label << "' does not exist in database. "
              << "Projection will not include nodes with this label." << std::endl;

    if (!gql_model.catalog.node_labels_str.empty()) {
        std::cerr << "[WARNING] Available labels: [";
        for (size_t i = 0; i < gql_model.catalog.node_labels_str.size(); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << "'" << gql_model.catalog.node_labels_str[i] << "'";
        }
        std::cerr << "]" << std::endl;
    }
}

void ProjectProcedure::warn_missing_type(const std::string& type) {
    auto it = gql_model.catalog.edge_labels2id.find(type);
    if (it != gql_model.catalog.edge_labels2id.end()) {
        return;
    }

    std::cerr << "[WARNING] Relationship type '" << type << "' does not exist in database. "
              << "Projection will not include edges with this type." << std::endl;

    if (!gql_model.catalog.edge_labels_str.empty()) {
        std::cerr << "[WARNING] Available types: [";
        for (size_t i = 0; i < gql_model.catalog.edge_labels_str.size(); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << "'" << gql_model.catalog.edge_labels_str[i] << "'";
        }
        std::cerr << "]" << std::endl;
    }
}

// =============================================================================
// Config dict helpers (single-deserialization, nullptr-safe)
// =============================================================================

std::vector<std::string> ProjectProcedure::parse_property_list_from_dict(
    DictionaryObject* dict, const std::string& key)
{
    if (!dict) return {};

    bool found = false;
    ObjectId value = get_value_from_dict(dict, key, found);
    if (!found) return {};

    auto value_type = GQL_OID::get_type(value);

    // Case 1: Single string
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        return {Conversions::unpack_string(value)};
    }

    // Case 2: List of strings
    if (value_type == GQL_OID::Type::LIST) {
        std::vector<ObjectId> list_items = Conversions::unpack_list(value);

        if (list_items.empty()) {
            throw std::runtime_error(
                "Configuration parameter '" + key + "' cannot be an empty list. "
                "Please provide at least one property name or omit the parameter.");
        }

        std::vector<std::string> properties;
        properties.reserve(list_items.size());

        for (size_t i = 0; i < list_items.size(); i++) {
            const auto& item_oid = list_items[i];
            auto item_type = GQL_OID::get_type(item_oid);

            if (item_type != GQL_OID::Type::STRING_SIMPLE_INLINE &&
                item_type != GQL_OID::Type::STRING_SIMPLE_EXTERN &&
                item_type != GQL_OID::Type::STRING_SIMPLE_TMP)
            {
                throw std::runtime_error(
                    "Configuration parameter '" + key + "' list element at index " + std::to_string(i) +
                    " is not a string (type: " + std::to_string(static_cast<int>(item_type)) + "). "
                    "All list elements must be strings representing property names.");
            }

            properties.push_back(Conversions::unpack_string(item_oid));
        }

        return properties;
    }

    // Case 3: Map of property configs — extract keys as property names
    if (value_type == GQL_OID::Type::DICTIONARY) {
        std::unique_ptr<Dictionary> prop_dict = Common::Conversions::unpack_dictionary(value);
        auto prop_dict_obj = dynamic_cast<DictionaryObject*>(prop_dict->dictionary.get());
        if (!prop_dict_obj) {
            throw std::runtime_error(
                "Configuration parameter '" + key + "' map is invalid. "
                "Provide a valid property configuration map.");
        }

        std::vector<std::string> properties;
        properties.reserve(prop_dict_obj->keys.size());
        for (const auto& [prop_alias_oid, config_item] : prop_dict_obj->keys) {
            properties.push_back(Conversions::unpack_string(prop_alias_oid));
        }
        return properties;
    }

    throw std::runtime_error(
        "Configuration parameter '" + key + "' must be STRING or LIST<STRING>, got type: " +
        std::to_string(static_cast<int>(value_type)) + ". "
        "Provide either a single property name or a list of property names.");
}

std::string ProjectProcedure::resolve_aggregation_property(
    DictionaryObject* dict,
    const std::vector<std::string>& edge_properties,
    Aggregation aggregation)
{
    // An explicit `aggregationProperty` must always be honored, even when the
    // GLOBAL aggregation is COUNT/SINGLE — otherwise per-type SUM/MIN/MAX that
    // inherit from this global value silently lose the property and read 0.
    if (dict) {
        std::string explicit_prop = get_string_from_dict(dict, "aggregationProperty", "");
        if (!explicit_prop.empty()) {
            return explicit_prop;
        }
    }

    // Implicit fallback: first relationshipProperty (preserves prior behavior
    // for the common "aggregation: 'SUM', relationshipProperties: ['x']" form).
    if (!edge_properties.empty() && aggregation != Aggregation::COUNT &&
        aggregation != Aggregation::SINGLE)
    {
        return edge_properties[0];
    }

    // For COUNT/SINGLE globals with no explicit property and no edge properties,
    // leave the resolved property empty — the aggregator (COUNT needs no property,
    // SINGLE fails on duplicates) won't use it.
    if (aggregation == Aggregation::COUNT || aggregation == Aggregation::SINGLE) {
        return "";
    }

    // MIN/MAX/SUM requires a property but none specified.
    throw QueryException(
        "Aggregation strategy '" +
        std::string(aggregation == Aggregation::MIN ? "MIN" :
                    aggregation == Aggregation::MAX ? "MAX" : "SUM") +
        "' requires a property to aggregate, but no relationshipProperties specified.\n\n"
        "Solutions:\n"
        "  1. Specify relationshipProperties: {aggregation: 'MIN', relationshipProperties: ['weight']}\n"
        "  2. Or specify aggregationProperty explicitly: {aggregation: 'MIN', aggregationProperty: 'weight'}\n"
        "  3. Or use COUNT/SINGLE which don't need properties: {aggregation: 'COUNT'}");
}

// =============================================================================
// Dictionary value extraction (nullptr-safe)
// =============================================================================

ObjectId ProjectProcedure::get_value_from_dict(DictionaryObject* dict, const std::string& key, bool& found) {
    found = false;
    if (!dict) return ObjectId::get_null();

    for (const auto& [key_oid, val_item] : dict->keys) {
        std::string key_str = Conversions::unpack_string(key_oid);
        if (key_str == key) {
            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (lit) {
                found = true;
                return lit->object_id;
            }
            auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
            if (nested_dict) {
                found = true;
                return ObjectId::get_null();
            }
            break;
        }
    }
    return ObjectId::get_null();
}

std::string ProjectProcedure::get_string_from_dict(
    DictionaryObject* dict, const std::string& key, const std::string& default_value)
{
    if (!dict) return default_value;

    bool found = false;
    ObjectId value = get_value_from_dict(dict, key, found);
    if (!found) return default_value;

    auto value_type = GQL_OID::get_type(value);
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        return Conversions::unpack_string(value);
    }

    throw std::runtime_error(
        "Configuration value for '" + key + "' must be a string, got type: " +
        std::to_string(static_cast<int>(value_type)));
}

Orientation ProjectProcedure::get_orientation_from_dict(
    DictionaryObject* dict, const std::string& key, Orientation default_value)
{
    if (!dict) return default_value;

    bool found = false;
    ObjectId value = get_value_from_dict(dict, key, found);
    if (!found) return default_value;

    auto value_type = GQL_OID::get_type(value);
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        return parse_orientation_string(Conversions::unpack_string(value));
    }

    throw std::runtime_error(
        "Configuration value for '" + key + "' must be a string, got type: " +
        std::to_string(static_cast<int>(value_type)));
}

Aggregation ProjectProcedure::get_aggregation_from_dict(
    DictionaryObject* dict, const std::string& key, Aggregation default_value)
{
    if (!dict) return default_value;

    bool found = false;
    ObjectId value = get_value_from_dict(dict, key, found);
    if (!found) return default_value;

    auto value_type = GQL_OID::get_type(value);
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        return parse_aggregation_string(Conversions::unpack_string(value));
    }

    throw std::runtime_error(
        "Configuration value for '" + key + "' must be a string, got type: " +
        std::to_string(static_cast<int>(value_type)));
}

std::optional<double> ProjectProcedure::get_optional_double_from_dict(
    DictionaryObject* dict, const std::string& key)
{
    if (!dict) return std::nullopt;

    bool found = false;
    ObjectId value = get_value_from_dict(dict, key, found);
    if (!found) return std::nullopt;

    auto value_type = GQL_OID::get_type(value);

    if (value_type == GQL_OID::Type::INT56_INLINE ||
        value_type == GQL_OID::Type::INT64_EXTERN ||
        value_type == GQL_OID::Type::INT64_TMP)
    {
        return static_cast<double>(Common::Conversions::unpack_int(value));
    }

    if (value_type == GQL_OID::Type::FLOAT32) {
        return static_cast<double>(Common::Conversions::unpack_float(value));
    }

    if (value_type == GQL_OID::Type::DOUBLE64_EXTERN ||
        value_type == GQL_OID::Type::DOUBLE64_TMP)
    {
        return Common::Conversions::unpack_double(value);
    }

    if (value_type == GQL_OID::Type::DECIMAL_INLINE ||
        value_type == GQL_OID::Type::DECIMAL_EXTERN ||
        value_type == GQL_OID::Type::DECIMAL_TMP)
    {
        return Common::Conversions::unpack_decimal(value).to_double();
    }

    throw std::runtime_error(
        "Configuration value for '" + key + "' must be a numeric value, got type: " +
        std::to_string(static_cast<int>(value_type)));
}

// =============================================================================
// String-to-enum converters
// =============================================================================

Orientation ProjectProcedure::parse_orientation_string(const std::string& orientation_str) {
    std::string upper_str = orientation_str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);

    if (upper_str == "NATURAL")    return Orientation::NATURAL;
    if (upper_str == "REVERSE")    return Orientation::REVERSE;
    if (upper_str == "UNDIRECTED") return Orientation::UNDIRECTED;

    throw std::runtime_error(
        "Invalid orientation value: '" + orientation_str + "'. "
        "Must be 'NATURAL', 'REVERSE', or 'UNDIRECTED' (case-insensitive).");
}

Aggregation ProjectProcedure::parse_aggregation_string(const std::string& aggregation_str) {
    std::string upper_str = aggregation_str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);

    if (upper_str == "SINGLE" || upper_str == "NONE") return Aggregation::SINGLE;
    if (upper_str == "MIN")   return Aggregation::MIN;
    if (upper_str == "MAX")   return Aggregation::MAX;
    if (upper_str == "SUM")   return Aggregation::SUM;
    if (upper_str == "COUNT") return Aggregation::COUNT;

    throw std::runtime_error(
        "Invalid aggregation value: '" + aggregation_str + "'. "
        "Must be 'SINGLE', 'NONE', 'MIN', 'MAX', 'SUM', or 'COUNT' (case-insensitive).");
}

// =============================================================================
// Neo4j GDS map syntax parsing
// =============================================================================

PropertyConfig ProjectProcedure::parse_property_config(ObjectId config_oid, const std::string& property_key) {
    PropertyConfig config;
    config.source_property = property_key;

    auto config_type = GQL_OID::get_type(config_oid);

    // Simple string: property name is the value
    if (config_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        config_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        config_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        config.source_property = Conversions::unpack_string(config_oid);
        return config;
    }

    if (config_type != GQL_OID::Type::DICTIONARY) {
        throw std::runtime_error(
            "Property configuration for '" + property_key + "' must be a string or map, got type: " +
            std::to_string(static_cast<int>(config_type)));
    }

    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Failed to parse property configuration for '" + property_key + "'");
    }

    config.source_property = get_string_from_dict(dict_obj, "property", property_key);
    config.default_value = get_optional_double_from_dict(dict_obj, "defaultValue");
    config.aggregation = get_aggregation_from_dict(dict_obj, "aggregation", Aggregation::SINGLE);

    return config;
}

void ProjectProcedure::parse_properties_value(
    ObjectId properties_oid,
    std::vector<std::string>& simple_properties,
    std::unordered_map<std::string, PropertyConfig>& property_configs)
{
    auto prop_type = GQL_OID::get_type(properties_oid);

    // Case 1: List of property names
    if (prop_type == GQL_OID::Type::LIST) {
        std::vector<ObjectId> list_items = Conversions::unpack_list(properties_oid);
        for (const auto& item : list_items) {
            auto item_type = GQL_OID::get_type(item);
            if (item_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
                item_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
                item_type == GQL_OID::Type::STRING_SIMPLE_TMP)
            {
                simple_properties.push_back(Conversions::unpack_string(item));
            } else {
                throw std::runtime_error("Property list elements must be strings");
            }
        }
        return;
    }

    // Case 2: Single string
    if (prop_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        prop_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        prop_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        simple_properties.push_back(Conversions::unpack_string(properties_oid));
        return;
    }

    // Case 3: Map with property configurations
    if (prop_type == GQL_OID::Type::DICTIONARY) {
        std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(properties_oid);
        auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
        if (!dict_obj) {
            throw std::runtime_error("Failed to parse properties configuration map");
        }

        for (const auto& [key_oid, val_item] : dict_obj->keys) {
            std::string property_key = Conversions::unpack_string(key_oid);

            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (lit) {
                property_configs[property_key] = parse_property_config(lit->object_id, property_key);
            } else {
                auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
                if (nested_dict) {
                    PropertyConfig prop_config;
                    prop_config.source_property = get_string_from_dict(nested_dict, "property", property_key);
                    prop_config.default_value = get_optional_double_from_dict(nested_dict, "defaultValue");
                    prop_config.aggregation = get_aggregation_from_dict(nested_dict, "aggregation", Aggregation::SINGLE);
                    property_configs[property_key] = prop_config;
                } else {
                    throw std::runtime_error("Invalid property configuration for '" + property_key + "'");
                }
            }
        }
        return;
    }

    throw std::runtime_error(
        "Properties must be a string, list, or map, got type: " +
        std::to_string(static_cast<int>(prop_type)));
}

void ProjectProcedure::extract_nested_properties(
    DictionaryObject* parent_dict,
    std::vector<std::string>& simple_properties,
    std::unordered_map<std::string, PropertyConfig>& property_configs)
{
    if (!parent_dict) return;

    for (const auto& [key_oid, val_item] : parent_dict->keys) {
        std::string key_str = Conversions::unpack_string(key_oid);
        if (key_str != "properties") continue;

        auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
        if (nested_dict) {
            for (const auto& [prop_key_oid, prop_val] : nested_dict->keys) {
                std::string prop_key = Conversions::unpack_string(prop_key_oid);
                auto prop_lit = dynamic_cast<DictionaryLiteral*>(prop_val.get());
                if (prop_lit) {
                    property_configs[prop_key] = parse_property_config(prop_lit->object_id, prop_key);
                } else {
                    auto prop_nested = dynamic_cast<DictionaryObject*>(prop_val.get());
                    if (prop_nested) {
                        PropertyConfig prop_config;
                        prop_config.source_property = get_string_from_dict(prop_nested, "property", prop_key);
                        prop_config.default_value = get_optional_double_from_dict(prop_nested, "defaultValue");
                        prop_config.aggregation = get_aggregation_from_dict(prop_nested, "aggregation", Aggregation::SINGLE);
                        property_configs[prop_key] = prop_config;
                    }
                }
            }
        } else {
            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (lit) {
                parse_properties_value(lit->object_id, simple_properties, property_configs);
            }
        }
        break;
    }
}

NodeProjectionConfig ProjectProcedure::parse_single_node_config(
    ObjectId config_oid, const std::string& projected_label)
{
    NodeProjectionConfig config;
    config.label = projected_label;

    auto config_type = GQL_OID::get_type(config_oid);

    // Simple string: label reference
    if (config_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        config_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        config_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        config.label = Conversions::unpack_string(config_oid);
        return config;
    }

    if (config_type != GQL_OID::Type::DICTIONARY) {
        throw std::runtime_error(
            "Node projection configuration for '" + projected_label + "' must be a string or map, got type: " +
            std::to_string(static_cast<int>(config_type)));
    }

    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Failed to parse node configuration for '" + projected_label + "'");
    }

    config.label = get_string_from_dict(dict_obj, "label", projected_label);

    // Try ObjectId-based properties first
    bool properties_found = false;
    ObjectId properties_oid = get_value_from_dict(dict_obj, "properties", properties_found);
    if (properties_found && !properties_oid.is_null()) {
        parse_properties_value(properties_oid, config.simple_properties, config.property_configs);
    } else {
        extract_nested_properties(dict_obj, config.simple_properties, config.property_configs);
    }

    return config;
}

RelationshipProjectionConfig ProjectProcedure::parse_single_relationship_config(
    ObjectId config_oid, const std::string& projected_type,
    Orientation global_orientation, Aggregation global_aggregation)
{
    RelationshipProjectionConfig config;
    config.type = projected_type;
    config.orientation = global_orientation;
    config.aggregation = global_aggregation;

    auto config_type = GQL_OID::get_type(config_oid);

    // Simple string: type reference
    if (config_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        config_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        config_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        config.type = Conversions::unpack_string(config_oid);
        return config;
    }

    if (config_type != GQL_OID::Type::DICTIONARY) {
        throw std::runtime_error(
            "Relationship projection configuration for '" + projected_type + "' must be a string or map, got type: " +
            std::to_string(static_cast<int>(config_type)));
    }

    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Failed to parse relationship configuration for '" + projected_type + "'");
    }

    config.type = get_string_from_dict(dict_obj, "type", projected_type);
    config.orientation = get_orientation_from_dict(dict_obj, "orientation", global_orientation);
    config.aggregation = get_aggregation_from_dict(dict_obj, "aggregation", global_aggregation);
    config.aggregation_property = get_string_from_dict(dict_obj, "aggregationProperty", "");

    // Properties
    bool properties_found = false;
    ObjectId properties_oid = get_value_from_dict(dict_obj, "properties", properties_found);
    if (properties_found && !properties_oid.is_null()) {
        parse_properties_value(properties_oid, config.simple_properties, config.property_configs);
    } else {
        extract_nested_properties(dict_obj, config.simple_properties, config.property_configs);
    }

    // Auto-select aggregation property for MIN/MAX/SUM (deterministic: first simple, then sorted configs)
    if ((config.aggregation == Aggregation::MIN ||
         config.aggregation == Aggregation::MAX ||
         config.aggregation == Aggregation::SUM) &&
        config.aggregation_property.empty())
    {
        if (!config.simple_properties.empty()) {
            config.aggregation_property = config.simple_properties[0];
        } else if (!config.property_configs.empty()) {
            // Use lexicographically first key for deterministic behavior
            std::string first_key;
            for (const auto& [k, v] : config.property_configs) {
                if (first_key.empty() || k < first_key) {
                    first_key = k;
                }
            }
            config.aggregation_property = first_key;
        }
    }

    return config;
}

NodeProjectionMap ProjectProcedure::parse_node_projection_map(ObjectId dict_oid) {
    NodeProjectionMap result;

    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(dict_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("nodeProjection map must be a dictionary/map");
    }

    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string projected_label = Conversions::unpack_string(key_oid);

        auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
        if (lit) {
            result[projected_label] = parse_single_node_config(lit->object_id, projected_label);
        } else {
            auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
            if (nested_dict) {
                NodeProjectionConfig config;
                config.label = get_string_from_dict(nested_dict, "label", projected_label);
                extract_nested_properties(nested_dict, config.simple_properties, config.property_configs);
                result[projected_label] = config;
            } else {
                throw std::runtime_error("Invalid node projection configuration for label '" + projected_label + "'");
            }
        }
    }

    if (result.empty()) {
        throw std::runtime_error("nodeProjection map cannot be empty. Please provide at least one node label.");
    }

    return result;
}

RelationshipProjectionMap ProjectProcedure::parse_relationship_projection_map(
    ObjectId dict_oid, Orientation global_orientation, Aggregation global_aggregation)
{
    RelationshipProjectionMap result;

    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(dict_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("relationshipProjection map must be a dictionary/map");
    }

    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string projected_type = Conversions::unpack_string(key_oid);

        auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
        if (lit) {
            result[projected_type] = parse_single_relationship_config(
                lit->object_id, projected_type, global_orientation, global_aggregation);
        } else {
            auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
            if (nested_dict) {
                RelationshipProjectionConfig config;
                config.type = get_string_from_dict(nested_dict, "type", projected_type);
                config.orientation = get_orientation_from_dict(nested_dict, "orientation", global_orientation);
                config.aggregation = get_aggregation_from_dict(nested_dict, "aggregation", global_aggregation);
                config.aggregation_property = get_string_from_dict(nested_dict, "aggregationProperty", "");

                extract_nested_properties(nested_dict, config.simple_properties, config.property_configs);

                // Auto-select aggregation property (deterministic)
                if ((config.aggregation == Aggregation::MIN ||
                     config.aggregation == Aggregation::MAX ||
                     config.aggregation == Aggregation::SUM) &&
                    config.aggregation_property.empty())
                {
                    if (!config.simple_properties.empty()) {
                        config.aggregation_property = config.simple_properties[0];
                    } else if (!config.property_configs.empty()) {
                        std::string first_key;
                        for (const auto& [k, v] : config.property_configs) {
                            if (first_key.empty() || k < first_key) {
                                first_key = k;
                            }
                        }
                        config.aggregation_property = first_key;
                    }
                }

                result[projected_type] = config;
            } else {
                throw std::runtime_error("Invalid relationship projection configuration for type '" + projected_type + "'");
            }
        }
    }

    if (result.empty()) {
        throw std::runtime_error("relationshipProjection map cannot be empty. Please provide at least one relationship type.");
    }

    return result;
}
