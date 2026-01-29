#include "project_procedure.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/gql_object_id.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "storage/dictionary/dictionary.h"
#include "system/file_manager.h"
//         uint64_t duration_ms;
//     };
//     class NativeProjectionBuilder;
// }

using namespace GQL;
using namespace GQL::Procedures;

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

    // Step 2: Parse graphName
    std::string graph_name;
    try {
        graph_name = parse_graph_name(ctx);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid graphName parameter: " + std::string(e.what()) + "\n\n"
            "The first parameter must be a STRING containing the projection name.\n"
            "Example: CALL PROJECT('myProjection', ...)"
        );
    }

    // Validate graph name is not empty or whitespace-only
    if (graph_name.empty()) {
        throw std::runtime_error(
            "Invalid projection name: name cannot be empty.\n"
            "Provide a non-empty string as the first argument.\n"
            "Example: CALL PROJECT('myProjection', ...)"
        );
    }
    if (graph_name.find_first_not_of(" \t\n\r") == std::string::npos) {
        throw std::runtime_error(
            "Invalid projection name: name cannot be whitespace only.\n"
            "Provide a meaningful name for your projection.\n"
            "Example: CALL PROJECT('myProjection', ...)"
        );
    }

    // Step 5: Parse optional config map for global defaults FIRST (needed for map parsing)
    // Global defaults that can be overridden by per-type configuration
    std::vector<std::string> global_node_properties;
    std::vector<std::string> global_edge_properties;
    Orientation global_orientation = Orientation::NATURAL;
    Aggregation global_aggregation = Aggregation::SINGLE;
    std::string global_aggregation_property;

    if (ctx.arguments.size() >= 4) {
        ObjectId config_arg = ctx.get_argument(3);
        auto config_type = GQL_OID::get_type(config_arg);

        if (config_type == GQL_OID::Type::DICTIONARY) {
            global_node_properties = parse_property_list_from_config(ctx, config_arg, "nodeProperties");
            global_edge_properties = parse_property_list_from_config(ctx, config_arg, "relationshipProperties");
            global_orientation = parse_orientation_from_config(ctx, config_arg);
            global_aggregation = parse_aggregation_from_config(ctx, config_arg);
            global_aggregation_property = parse_aggregation_property_from_config(
                ctx, config_arg, global_edge_properties, global_aggregation);
        }
    }

    // Step 3: Parse nodeProjection (supports STRING, LIST, or MAP)
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

    // Step 4: Parse relationshipProjection (supports STRING, LIST, or MAP)
    // For MAP syntax, we need to re-parse with global defaults
    RelationshipProjectionVariant rel_projection_variant;
    try {
        // Check if argument is a dictionary to apply global defaults
        ObjectId rel_arg = ctx.get_argument(2);
        auto rel_type = GQL_OID::get_type(rel_arg);
        if (rel_type == GQL_OID::Type::DICTIONARY) {
            // Re-parse with global defaults
            rel_projection_variant = parse_relationship_projection_map(
                ctx, rel_arg, global_orientation, global_aggregation);
        } else {
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

    // Extract node labels and properties from variant
    std::vector<std::string> node_labels;
    std::vector<std::string> node_properties = global_node_properties;
    std::unordered_map<std::string, PropertyConfig> node_property_configs;  // Phase 3
    bool using_node_map = false;

    if (std::holds_alternative<std::vector<std::string>>(node_projection_variant)) {
        // Simple list syntax - use global properties
        node_labels = std::get<std::vector<std::string>>(node_projection_variant);
    } else {
        // Map syntax - extract labels and per-label properties
        using_node_map = true;
        const auto& node_map = std::get<NodeProjectionMap>(node_projection_variant);
        for (const auto& [projected_label, config] : node_map) {
            node_labels.push_back(config.label);
            // Collect all properties from this label config
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
                // Phase 3: Store property configuration for renaming/defaults
                node_property_configs[prop_key] = prop_config;
            }
        }
        if (!node_property_configs.empty()) {
            for (const auto& [prop_name, prop_cfg] : node_property_configs) {
                if (!prop_cfg.source_property.empty()) {
                }
                if (prop_cfg.default_value.has_value()) {
                }
            }
        }
    }

    // Extract relationship types and per-type configuration from variant
    std::vector<std::string> relationship_types;
    std::vector<std::string> edge_properties = global_edge_properties;
    std::unordered_map<std::string, PropertyConfig> edge_property_configs;  // Phase 3
    Orientation orientation = global_orientation;
    Aggregation aggregation = global_aggregation;
    std::string aggregation_property = global_aggregation_property;
    bool using_rel_map = false;

    // Per-type configuration maps (for Phase 2 builder integration)
    std::unordered_map<std::string, Orientation> type_orientations;
    std::unordered_map<std::string, Aggregation> type_aggregations;
    std::unordered_map<std::string, std::string> type_agg_properties;

    if (std::holds_alternative<std::vector<std::string>>(rel_projection_variant)) {
        // Simple list syntax - use global config
        relationship_types = std::get<std::vector<std::string>>(rel_projection_variant);
    } else {
        // Map syntax - extract types and per-type configuration
        using_rel_map = true;
        const auto& rel_map = std::get<RelationshipProjectionMap>(rel_projection_variant);
        for (const auto& [projected_type, config] : rel_map) {
            relationship_types.push_back(config.type);

            // Store per-type configuration
            type_orientations[config.type] = config.orientation;
            type_aggregations[config.type] = config.aggregation;
            if (!config.aggregation_property.empty()) {
                type_agg_properties[config.type] = config.aggregation_property;
            }

            // Collect all properties from this type config
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
                // Phase 3: Store property configuration for renaming/defaults
                edge_property_configs[prop_key] = prop_config;
            }
        }

        // Phase 3: Log edge property configs
        if (!edge_property_configs.empty()) {
            for (const auto& [prop_name, prop_cfg] : edge_property_configs) {
                if (!prop_cfg.source_property.empty()) {
                }
                if (prop_cfg.default_value.has_value()) {
                }
            }
        }
    }

    // Step 6: Validate labels and types exist
    for (const auto& label : node_labels) {
        validate_label_exists(label);
    }

    for (const auto& type : relationship_types) {
        validate_type_exists(type);
    }

    // Step 7: Execute native projection using NativeProjectionBuilder
    for (size_t i = 0; i < node_labels.size(); i++) {
    }
    for (size_t i = 0; i < relationship_types.size(); i++) {
    }

    if (!node_properties.empty()) {
        for (size_t i = 0; i < node_properties.size(); i++) {
        }
    }

    if (!edge_properties.empty()) {
        for (size_t i = 0; i < edge_properties.size(); i++) {
        }
    }

    // Get db_folder from global file_manager
    std::string db_folder = file_manager.get_file_path("");
    // Remove trailing slash if present
    if (!db_folder.empty() && db_folder.back() == '/') {
        db_folder.pop_back();
    }

    // Log orientation for debugging (global default for non-map syntax)
    if (!using_rel_map) {
        switch (orientation) {
            case Orientation::NATURAL:
                break;
            case Orientation::REVERSE:
                break;
            case Orientation::UNDIRECTED:
                break;
        }

        // Log aggregation for debugging
        switch (aggregation) {
            case Aggregation::SINGLE:
                break;
            case Aggregation::MIN:
                if (!aggregation_property.empty()) {
                }
                break;
            case Aggregation::MAX:
                if (!aggregation_property.empty()) {
                }
                break;
            case Aggregation::SUM:
                if (!aggregation_property.empty()) {
                }
                break;
            case Aggregation::COUNT:
                break;
        }
    }

    // Create builder with per-type configuration (Phase 2 complete)
    // Pass per-type maps to builder - builder will use them for each type,
    // falling back to global defaults for types not in the maps
    NativeProjectionBuilder builder(
        graph_name,
        db_folder,
        node_properties,
        edge_properties,
        orientation,          // Global default orientation
        aggregation,          // Global default aggregation
        aggregation_property, // Global default aggregation property
        type_orientations,    // Per-type orientation overrides
        type_aggregations,    // Per-type aggregation overrides
        type_agg_properties,  // Per-type aggregation property overrides
        node_property_configs, // Phase 3: Per-property configuration for nodes
        edge_property_configs  // Phase 3: Per-property configuration for edges
    );
    builder.scan_nodes_by_labels(node_labels);
    builder.scan_edges_by_types(relationship_types);
    auto stats = builder.finalize();

    // Step 8: Yield results
    ctx.yield("graphName", ctx.create_string(graph_name));
    ctx.yield("nodeCount", ctx.create_int(static_cast<int64_t>(stats.node_count)));
    ctx.yield("relationshipCount", ctx.create_int(static_cast<int64_t>(stats.relationship_count)));
    ctx.yield("projectMillis", ctx.create_int(stats.duration_ms.count()));
    ctx.yield_row();
}

std::string ProjectProcedure::parse_graph_name(ProcedureContext& ctx) {
    return ctx.get_string_argument(0);
}

NodeProjectionVariant ProjectProcedure::parse_node_projection(ProcedureContext& ctx) {
    ObjectId arg = ctx.get_argument(1);
    auto type = GQL_OID::get_type(arg);

    // Case 1: String variant - single label (backward compatible)
    if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string label = Conversions::unpack_string(arg);

        // Wildcard expansion: '*' expands to all node labels from catalog
        if (label == "*") {
            return std::vector<std::string>(
                gql_model.catalog.node_labels_str.begin(),
                gql_model.catalog.node_labels_str.end()
            );
        }

        return std::vector<std::string>{label};
    }

    // Case 2: List variant - multiple labels (backward compatible)
    if (type == GQL_OID::Type::LIST) {
        std::vector<ObjectId> list_items = Conversions::unpack_list(arg);

        if (list_items.empty()) {
            throw std::runtime_error(
                "nodeProjection list cannot be empty. "
                "Please provide at least one node label."
            );
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
                    "All list elements must be strings representing node labels."
                );
            }

            std::string label = Conversions::unpack_string(item_oid);
            labels.push_back(label);
        }

        return labels;
    }

    // Case 3: Map variant - Neo4j GDS extended syntax (NEW)
    if (type == GQL_OID::Type::DICTIONARY) {
        return parse_node_projection_map(ctx, arg);
    }

    // Invalid type
    throw std::runtime_error(
        "nodeProjection must be STRING, LIST<STRING>, or MAP, got type: " +
        std::to_string(static_cast<int>(type)) + ". "
        "Provide either a single label string, a list of label strings, "
        "or a map with per-label configuration."
    );
}

RelationshipProjectionVariant ProjectProcedure::parse_relationship_projection(ProcedureContext& ctx) {
    ObjectId arg = ctx.get_argument(2);
    auto type = GQL_OID::get_type(arg);

    // Case 1: String variant - single type (backward compatible)
    if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string rel_type = Conversions::unpack_string(arg);

        // Wildcard expansion: '*' expands to all relationship types from catalog
        if (rel_type == "*") {
            return std::vector<std::string>(
                gql_model.catalog.edge_labels_str.begin(),
                gql_model.catalog.edge_labels_str.end()
            );
        }

        return std::vector<std::string>{rel_type};
    }

    // Case 2: List variant - multiple types (backward compatible)
    if (type == GQL_OID::Type::LIST) {
        std::vector<ObjectId> list_items = Conversions::unpack_list(arg);

        if (list_items.empty()) {
            throw std::runtime_error(
                "relationshipProjection list cannot be empty. "
                "Please provide at least one relationship type."
            );
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
                    "All list elements must be strings representing relationship types."
                );
            }

            std::string rel_type = Conversions::unpack_string(item_oid);
            types.push_back(rel_type);
        }

        return types;
    }

    // Case 3: Map variant - Neo4j GDS extended syntax (NEW)
    // Note: Global defaults are parsed from config (argument 3) in execute()
    // and applied when calling parse_relationship_projection_map()
    if (type == GQL_OID::Type::DICTIONARY) {
        // For map syntax, we need global defaults from config. Since we can't access them here directly,
        // we'll return a special marker and handle it in execute().
        // Actually, let's parse with defaults here and override in execute() if needed.
        return parse_relationship_projection_map(ctx, arg, Orientation::NATURAL, Aggregation::SINGLE);
    }

    // Invalid type
    throw std::runtime_error(
        "relationshipProjection must be STRING, LIST<STRING>, or MAP, got type: " +
        std::to_string(static_cast<int>(type)) + ". "
        "Provide either a single type string, a list of type strings, "
        "or a map with per-type configuration."
    );
}

void ProjectProcedure::validate_label_exists(const std::string& label) {
    // Access the global GQLModel catalog
    auto it = gql_model.catalog.node_labels2id.find(label);

    if (it == gql_model.catalog.node_labels2id.end()) {
        // Label doesn't exist - emit warning (non-blocking)
        // Decision: Use warnings instead of errors to allow dynamic label sets
        std::cerr << "[WARNING] Node label '" << label << "' does not exist in database. "
                  << "Projection will not include nodes with this label." << std::endl;

        // Add available labels hint if catalog is not empty
        if (!gql_model.catalog.node_labels_str.empty()) {
            std::cerr << "[WARNING] Available labels: [";
            for (size_t i = 0; i < gql_model.catalog.node_labels_str.size(); i++) {
                if (i > 0) std::cerr << ", ";
                std::cerr << "'" << gql_model.catalog.node_labels_str[i] << "'";
            }
            std::cerr << "]" << std::endl;
        }
    }
}

void ProjectProcedure::validate_type_exists(const std::string& type) {
    // Access the global GQLModel catalog
    auto it = gql_model.catalog.edge_labels2id.find(type);

    if (it == gql_model.catalog.edge_labels2id.end()) {
        // Type doesn't exist - emit warning (non-blocking)
        // Decision: Use warnings instead of errors to allow dynamic relationship sets
        std::cerr << "[WARNING] Relationship type '" << type << "' does not exist in database. "
                  << "Projection will not include edges with this type." << std::endl;

        // Add available types hint if catalog is not empty
        if (!gql_model.catalog.edge_labels_str.empty()) {
            std::cerr << "[WARNING] Available types: [";
            for (size_t i = 0; i < gql_model.catalog.edge_labels_str.size(); i++) {
                if (i > 0) std::cerr << ", ";
                std::cerr << "'" << gql_model.catalog.edge_labels_str[i] << "'";
            }
            std::cerr << "]" << std::endl;
        }
    }
}

std::vector<std::string> ProjectProcedure::parse_property_list_from_config(
    ProcedureContext& ctx,
    ObjectId config_map,
    const std::string& key
) {

    // Unpack dictionary to get key-value pairs
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_map);

    // Cast to DictionaryObject to access keys map
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Configuration parameter must be a dictionary/map");
    }

    // Find the key in the dictionary
    ObjectId value;
    bool found = false;
    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        // Unpack key and check if it matches
        std::string key_str = Conversions::unpack_string(key_oid);
        if (key_str == key) {
            // Extract ObjectId from DictionaryLiteral
            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (!lit) {
                throw std::runtime_error("Configuration value for '" + key + "' must be a literal (string or list)");
            }
            value = lit->object_id;
            found = true;
            break;
        }
    }

    if (!found) {
        // Key not present - return empty vector (properties not requested)
        return {};
    }
    auto value_type = GQL_OID::get_type(value);

    // Case 1: String variant - single property
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string prop_name = Conversions::unpack_string(value);
        return {prop_name};
    }

    // Case 2: List variant - multiple properties
    if (value_type == GQL_OID::Type::LIST) {
        std::vector<ObjectId> list_items = Conversions::unpack_list(value);

        if (list_items.empty()) {
            throw std::runtime_error(
                "Configuration parameter '" + key + "' cannot be an empty list. "
                "Please provide at least one property name or omit the parameter."
            );
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
                    "All list elements must be strings representing property names."
                );
            }

            std::string prop_name = Conversions::unpack_string(item_oid);
            properties.push_back(prop_name);
        }

        return properties;
    }

    // Case 3: Map variant - property configurations with renaming/defaults
    if (value_type == GQL_OID::Type::DICTIONARY) {
        std::unique_ptr<Dictionary> prop_dict = Common::Conversions::unpack_dictionary(value);

        // Cast to DictionaryObject to access keys map
        auto prop_dict_obj = dynamic_cast<DictionaryObject*>(prop_dict->dictionary.get());
        if (!prop_dict_obj) {
            throw std::runtime_error(
                "Configuration parameter '" + key + "' map is invalid. "
                "Provide a valid property configuration map."
            );
        }

        std::vector<std::string> properties;
        properties.reserve(prop_dict_obj->keys.size());

        for (const auto& [prop_alias_oid, config_item] : prop_dict_obj->keys) {
            // Extract the property alias (key name in the map)
            std::string prop_alias = Conversions::unpack_string(prop_alias_oid);
            properties.push_back(prop_alias);
        }

        return properties;
    }

    // Invalid type
    throw std::runtime_error(
        "Configuration parameter '" + key + "' must be STRING or LIST<STRING>, got type: " +
        std::to_string(static_cast<int>(value_type)) + ". "
        "Provide either a single property name or a list of property names."
    );
}

Orientation ProjectProcedure::parse_orientation_from_config(
    ProcedureContext& ctx,
    ObjectId config_map
) {

    // Unpack dictionary to get key-value pairs
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_map);

    // Cast to DictionaryObject to access keys map
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Configuration parameter must be a dictionary/map");
    }

    // Find the 'orientation' key in the dictionary
    ObjectId value;
    bool found = false;
    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string key_str = Conversions::unpack_string(key_oid);
        if (key_str == "orientation") {
            // Extract ObjectId from DictionaryLiteral
            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (!lit) {
                throw std::runtime_error("Configuration value for 'orientation' must be a literal string");
            }
            value = lit->object_id;
            found = true;
            break;
        }
    }

    if (!found) {
        // Key not present - return default NATURAL
        return Orientation::NATURAL;
    }

    // Value must be a string
    auto value_type = GQL_OID::get_type(value);
    if (value_type != GQL_OID::Type::STRING_SIMPLE_INLINE &&
        value_type != GQL_OID::Type::STRING_SIMPLE_EXTERN &&
        value_type != GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        throw std::runtime_error(
            "Configuration parameter 'orientation' must be a string ('NATURAL', 'REVERSE', or 'UNDIRECTED'), "
            "got type: " + std::to_string(static_cast<int>(value_type))
        );
    }

    // Extract and normalize string (case-insensitive)
    std::string orientation_str = Conversions::unpack_string(value);

    // Convert to uppercase for case-insensitive comparison
    std::transform(orientation_str.begin(), orientation_str.end(),
                   orientation_str.begin(), ::toupper);

    if (orientation_str == "NATURAL") {
        return Orientation::NATURAL;
    } else if (orientation_str == "REVERSE") {
        return Orientation::REVERSE;
    } else if (orientation_str == "UNDIRECTED") {
        return Orientation::UNDIRECTED;
    } else {
        throw std::runtime_error(
            "Invalid orientation value: '" + orientation_str + "'. "
            "Must be 'NATURAL', 'REVERSE', or 'UNDIRECTED' (case-insensitive).\n\n"
            "Examples:\n"
            "  CALL PROJECT('g', 'User', 'KNOWS', {orientation: 'NATURAL'})\n"
            "  CALL PROJECT('g', 'User', 'KNOWS', {orientation: 'UNDIRECTED'})"
        );
    }
}

Aggregation ProjectProcedure::parse_aggregation_from_config(
    ProcedureContext& ctx,
    ObjectId config_map
) {

    // Unpack dictionary to get key-value pairs
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_map);

    // Cast to DictionaryObject to access keys map
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Configuration parameter must be a dictionary/map");
    }

    // Find the 'aggregation' key in the dictionary
    ObjectId value;
    bool found = false;
    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string key_str = Conversions::unpack_string(key_oid);
        if (key_str == "aggregation") {
            // Extract ObjectId from DictionaryLiteral
            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (!lit) {
                throw std::runtime_error("Configuration value for 'aggregation' must be a literal string");
            }
            value = lit->object_id;
            found = true;
            break;
        }
    }

    if (!found) {
        // Key not present - return default SINGLE
        return Aggregation::SINGLE;
    }

    // Value must be a string
    auto value_type = GQL_OID::get_type(value);
    if (value_type != GQL_OID::Type::STRING_SIMPLE_INLINE &&
        value_type != GQL_OID::Type::STRING_SIMPLE_EXTERN &&
        value_type != GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        throw std::runtime_error(
            "Configuration parameter 'aggregation' must be a string ('SINGLE', 'MIN', 'MAX', 'SUM', or 'COUNT'), "
            "got type: " + std::to_string(static_cast<int>(value_type))
        );
    }

    // Extract and normalize string (case-insensitive)
    std::string aggregation_str = Conversions::unpack_string(value);

    // Convert to uppercase for case-insensitive comparison
    std::transform(aggregation_str.begin(), aggregation_str.end(),
                   aggregation_str.begin(), ::toupper);

    if (aggregation_str == "SINGLE") {
        return Aggregation::SINGLE;
    } else if (aggregation_str == "MIN") {
        return Aggregation::MIN;
    } else if (aggregation_str == "MAX") {
        return Aggregation::MAX;
    } else if (aggregation_str == "SUM") {
        return Aggregation::SUM;
    } else if (aggregation_str == "COUNT") {
        return Aggregation::COUNT;
    } else {
        throw std::runtime_error(
            "Invalid aggregation value: '" + aggregation_str + "'. "
            "Must be 'SINGLE', 'MIN', 'MAX', 'SUM', or 'COUNT' (case-insensitive).\n\n"
            "Examples:\n"
            "  CALL PROJECT('g', 'User', 'KNOWS', {aggregation: 'SINGLE'})  -- Fail on duplicates\n"
            "  CALL PROJECT('g', 'User', 'KNOWS', {aggregation: 'MIN'})     -- Keep minimum property\n"
            "  CALL PROJECT('g', 'User', 'KNOWS', {aggregation: 'COUNT'})   -- Count parallel edges"
        );
    }
}

std::string ProjectProcedure::parse_aggregation_property_from_config(
    ProcedureContext& ctx,
    ObjectId config_map,
    const std::vector<std::string>& edge_properties,
    Aggregation aggregation
) {

    // COUNT and SINGLE don't need a property
    if (aggregation == Aggregation::COUNT || aggregation == Aggregation::SINGLE) {
        return "";
    }

    // For MIN/MAX/SUM, we need a property
    // First, try to find explicit 'aggregationProperty' in config
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_map);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Configuration parameter must be a dictionary/map");
    }

    // Look for 'aggregationProperty' key
    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string key_str = Conversions::unpack_string(key_oid);
        if (key_str == "aggregationProperty") {
            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (!lit) {
                throw std::runtime_error("Configuration value for 'aggregationProperty' must be a string");
            }

            // Extract string value
            auto value_type = GQL_OID::get_type(lit->object_id);
            if (value_type != GQL_OID::Type::STRING_SIMPLE_INLINE &&
                value_type != GQL_OID::Type::STRING_SIMPLE_EXTERN &&
                value_type != GQL_OID::Type::STRING_SIMPLE_TMP)
            {
                throw std::runtime_error("Configuration parameter 'aggregationProperty' must be a string");
            }

            std::string property_name = Conversions::unpack_string(lit->object_id);
            return property_name;
        }
    }

    // No explicit aggregationProperty - use first property from relationshipProperties
    if (!edge_properties.empty()) {
        std::string property_name = edge_properties[0];
        return property_name;
    }

    // MIN/MAX/SUM requires a property but none specified
    throw std::runtime_error(
        "Aggregation strategy '" +
        std::string(aggregation == Aggregation::MIN ? "MIN" :
                    aggregation == Aggregation::MAX ? "MAX" : "SUM") +
        "' requires a property to aggregate, but no relationshipProperties specified.\n\n"
        "Solutions:\n"
        "  1. Specify relationshipProperties: {aggregation: 'MIN', relationshipProperties: ['weight']}\n"
        "  2. Or specify aggregationProperty explicitly: {aggregation: 'MIN', aggregationProperty: 'weight'}\n"
        "  3. Or use COUNT/SINGLE which don't need properties: {aggregation: 'COUNT'}"
    );
}

// =============================================================================
// Neo4j GDS Map Syntax Parsing Helpers (Phase 1)
// =============================================================================

Orientation ProjectProcedure::parse_orientation_string(const std::string& orientation_str) {
    std::string upper_str = orientation_str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);

    if (upper_str == "NATURAL") {
        return Orientation::NATURAL;
    } else if (upper_str == "REVERSE") {
        return Orientation::REVERSE;
    } else if (upper_str == "UNDIRECTED") {
        return Orientation::UNDIRECTED;
    } else {
        throw std::runtime_error(
            "Invalid orientation value: '" + orientation_str + "'. "
            "Must be 'NATURAL', 'REVERSE', or 'UNDIRECTED' (case-insensitive)."
        );
    }
}

Aggregation ProjectProcedure::parse_aggregation_string(const std::string& aggregation_str) {
    std::string upper_str = aggregation_str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);

    if (upper_str == "SINGLE" || upper_str == "NONE") {
        return Aggregation::SINGLE;
    } else if (upper_str == "MIN") {
        return Aggregation::MIN;
    } else if (upper_str == "MAX") {
        return Aggregation::MAX;
    } else if (upper_str == "SUM") {
        return Aggregation::SUM;
    } else if (upper_str == "COUNT") {
        return Aggregation::COUNT;
    } else {
        throw std::runtime_error(
            "Invalid aggregation value: '" + aggregation_str + "'. "
            "Must be 'SINGLE', 'NONE', 'MIN', 'MAX', 'SUM', or 'COUNT' (case-insensitive)."
        );
    }
}

ObjectId ProjectProcedure::get_value_from_dict(void* dict_obj_ptr, const std::string& key, bool& found) {
    auto dict_obj = static_cast<DictionaryObject*>(dict_obj_ptr);
    found = false;

    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string key_str = Conversions::unpack_string(key_oid);
        if (key_str == key) {
            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (lit) {
                found = true;
                return lit->object_id;
            }
            // If it's a nested dictionary, get the dictionary's ObjectId
            auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
            if (nested_dict) {
                // For nested dictionaries, we need to return a marker that indicates this
                // The caller should handle this case by checking the type
                found = true;
                // Return a temporary dictionary object ID - this requires special handling
                // For now, we'll handle nested dicts differently
                return ObjectId::get_null();
            }
            break;
        }
    }
    return ObjectId::get_null();
}

std::string ProjectProcedure::get_string_from_dict(
    ProcedureContext& ctx,
    void* dict_obj_ptr,
    const std::string& key,
    const std::string& default_value
) {
    bool found = false;
    ObjectId value = get_value_from_dict(dict_obj_ptr, key, found);

    if (!found) {
        return default_value;
    }

    auto value_type = GQL_OID::get_type(value);
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        return Conversions::unpack_string(value);
    }

    throw std::runtime_error(
        "Configuration value for '" + key + "' must be a string, got type: " +
        std::to_string(static_cast<int>(value_type))
    );
}

Orientation ProjectProcedure::get_orientation_from_dict(
    ProcedureContext& ctx,
    void* dict_obj_ptr,
    const std::string& key,
    Orientation default_value
) {
    bool found = false;
    ObjectId value = get_value_from_dict(dict_obj_ptr, key, found);

    if (!found) {
        return default_value;
    }

    auto value_type = GQL_OID::get_type(value);
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string orientation_str = Conversions::unpack_string(value);
        return parse_orientation_string(orientation_str);
    }

    throw std::runtime_error(
        "Configuration value for '" + key + "' must be a string, got type: " +
        std::to_string(static_cast<int>(value_type))
    );
}

Aggregation ProjectProcedure::get_aggregation_from_dict(
    ProcedureContext& ctx,
    void* dict_obj_ptr,
    const std::string& key,
    Aggregation default_value
) {
    bool found = false;
    ObjectId value = get_value_from_dict(dict_obj_ptr, key, found);

    if (!found) {
        return default_value;
    }

    auto value_type = GQL_OID::get_type(value);
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string aggregation_str = Conversions::unpack_string(value);
        return parse_aggregation_string(aggregation_str);
    }

    throw std::runtime_error(
        "Configuration value for '" + key + "' must be a string, got type: " +
        std::to_string(static_cast<int>(value_type))
    );
}

std::optional<double> ProjectProcedure::get_optional_double_from_dict(
    ProcedureContext& ctx,
    void* dict_obj_ptr,
    const std::string& key
) {
    bool found = false;
    ObjectId value = get_value_from_dict(dict_obj_ptr, key, found);

    if (!found) {
        return std::nullopt;
    }

    auto value_type = GQL_OID::get_type(value);

    // Handle integer types (INT56_INLINE, INT64_EXTERN, INT64_TMP)
    if (value_type == GQL_OID::Type::INT56_INLINE ||
        value_type == GQL_OID::Type::INT64_EXTERN ||
        value_type == GQL_OID::Type::INT64_TMP)
    {
        int64_t int_val = Common::Conversions::unpack_int(value);
        return static_cast<double>(int_val);
    }

    // Handle float types (FLOAT32)
    if (value_type == GQL_OID::Type::FLOAT32) {
        return static_cast<double>(Common::Conversions::unpack_float(value));
    }

    // Handle double types (DOUBLE64_EXTERN, DOUBLE64_TMP)
    if (value_type == GQL_OID::Type::DOUBLE64_EXTERN ||
        value_type == GQL_OID::Type::DOUBLE64_TMP)
    {
        return Common::Conversions::unpack_double(value);
    }

    // Handle decimal types (DECIMAL_INLINE, DECIMAL_EXTERN, DECIMAL_TMP)
    if (value_type == GQL_OID::Type::DECIMAL_INLINE ||
        value_type == GQL_OID::Type::DECIMAL_EXTERN ||
        value_type == GQL_OID::Type::DECIMAL_TMP)
    {
        return Common::Conversions::unpack_decimal(value).to_double();
    }

    throw std::runtime_error(
        "Configuration value for '" + key + "' must be a numeric value, got type: " +
        std::to_string(static_cast<int>(value_type))
    );
}

PropertyConfig ProjectProcedure::parse_property_config(
    ProcedureContext& ctx,
    ObjectId config_oid,
    const std::string& property_key
) {
    PropertyConfig config;
    config.source_property = property_key;  // Default: same as key

    auto config_type = GQL_OID::get_type(config_oid);

    // If it's just a string, treat as simple property name (no config)
    if (config_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        config_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        config_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        // Simple case: property name is the value
        config.source_property = Conversions::unpack_string(config_oid);
        return config;
    }

    // Must be a dictionary with property configuration
    if (config_type != GQL_OID::Type::DICTIONARY) {
        throw std::runtime_error(
            "Property configuration for '" + property_key + "' must be a string or map, got type: " +
            std::to_string(static_cast<int>(config_type))
        );
    }

    // Unpack the dictionary
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Failed to parse property configuration for '" + property_key + "'");
    }

    // Extract 'property' key (source property name for renaming)
    config.source_property = get_string_from_dict(ctx, dict_obj, "property", property_key);

    // Extract 'defaultValue' key
    config.default_value = get_optional_double_from_dict(ctx, dict_obj, "defaultValue");

    // Extract 'aggregation' key (for per-property aggregation)
    config.aggregation = get_aggregation_from_dict(ctx, dict_obj, "aggregation", Aggregation::SINGLE);

    return config;
}

void ProjectProcedure::parse_properties_value(
    ProcedureContext& ctx,
    ObjectId properties_oid,
    std::vector<std::string>& simple_properties,
    std::unordered_map<std::string, PropertyConfig>& property_configs
) {
    auto prop_type = GQL_OID::get_type(properties_oid);

    // Case 1: Simple list of property names
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

    // Case 2: Single string property
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
                // Value is a literal (string or nested dict in ObjectId form)
                PropertyConfig prop_config = parse_property_config(ctx, lit->object_id, property_key);
                property_configs[property_key] = prop_config;
            } else {
                // Value is a nested dictionary object directly
                auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
                if (nested_dict) {
                    PropertyConfig prop_config;
                    prop_config.source_property = get_string_from_dict(ctx, nested_dict, "property", property_key);
                    prop_config.default_value = get_optional_double_from_dict(ctx, nested_dict, "defaultValue");
                    prop_config.aggregation = get_aggregation_from_dict(ctx, nested_dict, "aggregation", Aggregation::SINGLE);
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
        std::to_string(static_cast<int>(prop_type))
    );
}

NodeProjectionConfig ProjectProcedure::parse_single_node_config(
    ProcedureContext& ctx,
    ObjectId config_oid,
    const std::string& projected_label
) {
    NodeProjectionConfig config;
    config.label = projected_label;  // Default: same as key

    auto config_type = GQL_OID::get_type(config_oid);

    // If it's just a string, treat as simple label reference
    if (config_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        config_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        config_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        config.label = Conversions::unpack_string(config_oid);
        return config;
    }

    // Must be a dictionary with node configuration
    if (config_type != GQL_OID::Type::DICTIONARY) {
        throw std::runtime_error(
            "Node projection configuration for '" + projected_label + "' must be a string or map, got type: " +
            std::to_string(static_cast<int>(config_type))
        );
    }

    // Unpack the dictionary
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Failed to parse node configuration for '" + projected_label + "'");
    }

    // Extract 'label' key (source label, defaults to key)
    config.label = get_string_from_dict(ctx, dict_obj, "label", projected_label);

    // Extract 'properties' key
    bool properties_found = false;
    ObjectId properties_oid = get_value_from_dict(dict_obj, "properties", properties_found);
    if (properties_found && !properties_oid.is_null()) {
        parse_properties_value(ctx, properties_oid, config.simple_properties, config.property_configs);
    } else {
        // Check if properties is a nested dictionary object directly
        for (const auto& [key_oid, val_item] : dict_obj->keys) {
            std::string key_str = Conversions::unpack_string(key_oid);
            if (key_str == "properties") {
                auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
                if (nested_dict) {
                    // Parse as nested dictionary
                    for (const auto& [prop_key_oid, prop_val] : nested_dict->keys) {
                        std::string prop_key = Conversions::unpack_string(prop_key_oid);
                        auto prop_lit = dynamic_cast<DictionaryLiteral*>(prop_val.get());
                        if (prop_lit) {
                            PropertyConfig prop_config = parse_property_config(ctx, prop_lit->object_id, prop_key);
                            config.property_configs[prop_key] = prop_config;
                        } else {
                            auto prop_nested = dynamic_cast<DictionaryObject*>(prop_val.get());
                            if (prop_nested) {
                                PropertyConfig prop_config;
                                prop_config.source_property = get_string_from_dict(ctx, prop_nested, "property", prop_key);
                                prop_config.default_value = get_optional_double_from_dict(ctx, prop_nested, "defaultValue");
                                prop_config.aggregation = get_aggregation_from_dict(ctx, prop_nested, "aggregation", Aggregation::SINGLE);
                                config.property_configs[prop_key] = prop_config;
                            }
                        }
                    }
                } else {
                    auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
                    if (lit) {
                        parse_properties_value(ctx, lit->object_id, config.simple_properties, config.property_configs);
                    }
                }
                break;
            }
        }
    }

    return config;
}

RelationshipProjectionConfig ProjectProcedure::parse_single_relationship_config(
    ProcedureContext& ctx,
    ObjectId config_oid,
    const std::string& projected_type,
    Orientation global_orientation,
    Aggregation global_aggregation
) {
    RelationshipProjectionConfig config;
    config.type = projected_type;  // Default: same as key
    config.orientation = global_orientation;  // Default from global
    config.aggregation = global_aggregation;  // Default from global

    auto config_type = GQL_OID::get_type(config_oid);

    // If it's just a string, treat as simple type reference
    if (config_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        config_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        config_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        config.type = Conversions::unpack_string(config_oid);
        return config;
    }

    // Must be a dictionary with relationship configuration
    if (config_type != GQL_OID::Type::DICTIONARY) {
        throw std::runtime_error(
            "Relationship projection configuration for '" + projected_type + "' must be a string or map, got type: " +
            std::to_string(static_cast<int>(config_type))
        );
    }

    // Unpack the dictionary
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("Failed to parse relationship configuration for '" + projected_type + "'");
    }

    // Extract 'type' key (source type, defaults to key)
    config.type = get_string_from_dict(ctx, dict_obj, "type", projected_type);

    // Extract 'orientation' key (per-type orientation, defaults to global)
    config.orientation = get_orientation_from_dict(ctx, dict_obj, "orientation", global_orientation);

    // Extract 'aggregation' key (per-type aggregation, defaults to global)
    config.aggregation = get_aggregation_from_dict(ctx, dict_obj, "aggregation", global_aggregation);

    // Extract 'aggregationProperty' key
    config.aggregation_property = get_string_from_dict(ctx, dict_obj, "aggregationProperty", "");

    // Extract 'properties' key
    bool properties_found = false;
    ObjectId properties_oid = get_value_from_dict(dict_obj, "properties", properties_found);
    if (properties_found && !properties_oid.is_null()) {
        parse_properties_value(ctx, properties_oid, config.simple_properties, config.property_configs);
    } else {
        // Check if properties is a nested dictionary object directly
        for (const auto& [key_oid, val_item] : dict_obj->keys) {
            std::string key_str = Conversions::unpack_string(key_oid);
            if (key_str == "properties") {
                auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
                if (nested_dict) {
                    // Parse as nested dictionary
                    for (const auto& [prop_key_oid, prop_val] : nested_dict->keys) {
                        std::string prop_key = Conversions::unpack_string(prop_key_oid);
                        auto prop_lit = dynamic_cast<DictionaryLiteral*>(prop_val.get());
                        if (prop_lit) {
                            PropertyConfig prop_config = parse_property_config(ctx, prop_lit->object_id, prop_key);
                            config.property_configs[prop_key] = prop_config;
                        } else {
                            auto prop_nested = dynamic_cast<DictionaryObject*>(prop_val.get());
                            if (prop_nested) {
                                PropertyConfig prop_config;
                                prop_config.source_property = get_string_from_dict(ctx, prop_nested, "property", prop_key);
                                prop_config.default_value = get_optional_double_from_dict(ctx, prop_nested, "defaultValue");
                                prop_config.aggregation = get_aggregation_from_dict(ctx, prop_nested, "aggregation", Aggregation::SINGLE);
                                config.property_configs[prop_key] = prop_config;
                            }
                        }
                    }
                } else {
                    auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
                    if (lit) {
                        parse_properties_value(ctx, lit->object_id, config.simple_properties, config.property_configs);
                    }
                }
                break;
            }
        }
    }

    // If MIN/MAX/SUM and no aggregationProperty specified, use first property
    if ((config.aggregation == Aggregation::MIN ||
         config.aggregation == Aggregation::MAX ||
         config.aggregation == Aggregation::SUM) &&
        config.aggregation_property.empty())
    {
        if (!config.simple_properties.empty()) {
            config.aggregation_property = config.simple_properties[0];
        } else if (!config.property_configs.empty()) {
            config.aggregation_property = config.property_configs.begin()->first;
        }
    }

    return config;
}

NodeProjectionMap ProjectProcedure::parse_node_projection_map(ProcedureContext& ctx, ObjectId dict_oid) {

    NodeProjectionMap result;

    // Unpack the outer dictionary
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(dict_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("nodeProjection map must be a dictionary/map");
    }


    // Iterate over each label in the map
    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string projected_label = Conversions::unpack_string(key_oid);

        auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
        if (lit) {
            // Value is stored as ObjectId
            NodeProjectionConfig config = parse_single_node_config(ctx, lit->object_id, projected_label);
            result[projected_label] = config;
        } else {
            // Value might be a nested DictionaryObject directly
            auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
            if (nested_dict) {
                NodeProjectionConfig config;
                config.label = get_string_from_dict(ctx, nested_dict, "label", projected_label);

                // Handle properties
                for (const auto& [inner_key_oid, inner_val] : nested_dict->keys) {
                    std::string inner_key = Conversions::unpack_string(inner_key_oid);
                    if (inner_key == "properties") {
                        auto inner_lit = dynamic_cast<DictionaryLiteral*>(inner_val.get());
                        if (inner_lit) {
                            parse_properties_value(ctx, inner_lit->object_id, config.simple_properties, config.property_configs);
                        }
                    }
                }

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
    ProcedureContext& ctx,
    ObjectId dict_oid,
    Orientation global_orientation,
    Aggregation global_aggregation
) {
    RelationshipProjectionMap result;

    // Unpack the outer dictionary
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(dict_oid);
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        throw std::runtime_error("relationshipProjection map must be a dictionary/map");
    }


    // Iterate over each type in the map
    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        std::string projected_type = Conversions::unpack_string(key_oid);

        auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
        if (lit) {
            // Value is stored as ObjectId
            RelationshipProjectionConfig config = parse_single_relationship_config(
                ctx, lit->object_id, projected_type, global_orientation, global_aggregation);
            result[projected_type] = config;
        } else {
            // Value might be a nested DictionaryObject directly
            auto nested_dict = dynamic_cast<DictionaryObject*>(val_item.get());
            if (nested_dict) {
                RelationshipProjectionConfig config;
                config.type = get_string_from_dict(ctx, nested_dict, "type", projected_type);
                config.orientation = get_orientation_from_dict(ctx, nested_dict, "orientation", global_orientation);
                config.aggregation = get_aggregation_from_dict(ctx, nested_dict, "aggregation", global_aggregation);
                config.aggregation_property = get_string_from_dict(ctx, nested_dict, "aggregationProperty", "");

                // Handle properties
                for (const auto& [inner_key_oid, inner_val] : nested_dict->keys) {
                    std::string inner_key = Conversions::unpack_string(inner_key_oid);
                    if (inner_key == "properties") {
                        auto inner_lit = dynamic_cast<DictionaryLiteral*>(inner_val.get());
                        if (inner_lit) {
                            parse_properties_value(ctx, inner_lit->object_id, config.simple_properties, config.property_configs);
                        }
                    }
                }

                // If MIN/MAX/SUM and no aggregationProperty specified, use first property
                if ((config.aggregation == Aggregation::MIN ||
                     config.aggregation == Aggregation::MAX ||
                     config.aggregation == Aggregation::SUM) &&
                    config.aggregation_property.empty())
                {
                    if (!config.simple_properties.empty()) {
                        config.aggregation_property = config.simple_properties[0];
                    } else if (!config.property_configs.empty()) {
                        config.aggregation_property = config.property_configs.begin()->first;
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
