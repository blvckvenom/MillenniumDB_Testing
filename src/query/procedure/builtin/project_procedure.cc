#include "project_procedure.h"

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
            "  - nodeProjection (STRING | LIST<STRING>): Label(s) to include\n"
            "  - relationshipProjection (STRING | LIST<STRING>): Type(s) to include\n"
            "  - configuration (MAP, optional): Reserved for future use\n\n"
            "Examples:\n"
            "  CALL PROJECT('myGraph', 'User', 'KNOWS')\n"
            "  CALL PROJECT('social', ['User', 'Post'], ['KNOWS', 'LIKES'])"
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

    // Step 3: Parse nodeProjection
    std::vector<std::string> node_labels;
    try {
        node_labels = parse_node_projection(ctx);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid nodeProjection parameter: " + std::string(e.what()) + "\n\n"
            "The second parameter must be either:\n"
            "  - A STRING: 'User'\n"
            "  - A LIST of STRINGs: ['User', 'Post']\n\n"
            "Examples:\n"
            "  CALL PROJECT('g', 'User', ...)\n"
            "  CALL PROJECT('g', ['User', 'Post'], ...)"
        );
    }

    // Step 4: Parse relationshipProjection
    std::vector<std::string> relationship_types;
    try {
        relationship_types = parse_relationship_projection(ctx);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid relationshipProjection parameter: " + std::string(e.what()) + "\n\n"
            "The third parameter must be either:\n"
            "  - A STRING: 'KNOWS'\n"
            "  - A LIST of STRINGs: ['KNOWS', 'LIKES']\n\n"
            "Examples:\n"
            "  CALL PROJECT('g', 'User', 'KNOWS')\n"
            "  CALL PROJECT('g', 'User', ['KNOWS', 'LIKES'])"
        );
    }

    // Step 5: Parse optional config map for properties
    std::vector<std::string> node_properties;
    std::vector<std::string> edge_properties;

    if (ctx.arguments.size() >= 4) {
        // Config map provided
        ObjectId config_arg = ctx.get_argument(3);
        auto config_type = GQL_OID::get_type(config_arg);

        if (config_type == GQL_OID::Type::DICTIONARY) {
            node_properties = parse_property_list_from_config(ctx, config_arg, "nodeProperties");
            edge_properties = parse_property_list_from_config(ctx, config_arg, "relationshipProperties");
        }
        // Ignore other types (will be used for future features)
    }

    // Step 6: Validate labels and types exist
    for (const auto& label : node_labels) {
        validate_label_exists(label);
    }

    for (const auto& type : relationship_types) {
        validate_type_exists(type);
    }

    // Step 7: Execute native projection using NativeProjectionBuilder
    std::cerr << "[ProjectProcedure] Creating NativeProjectionBuilder for graph: " << graph_name << std::endl;
    std::cerr << "[ProjectProcedure] Node labels: [";
    for (size_t i = 0; i < node_labels.size(); i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << node_labels[i];
    }
    std::cerr << "]" << std::endl;
    std::cerr << "[ProjectProcedure] Relationship types: [";
    for (size_t i = 0; i < relationship_types.size(); i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << relationship_types[i];
    }
    std::cerr << "]" << std::endl;

    if (!node_properties.empty()) {
        std::cerr << "[ProjectProcedure] Node properties: [";
        for (size_t i = 0; i < node_properties.size(); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << node_properties[i];
        }
        std::cerr << "]" << std::endl;
    }

    if (!edge_properties.empty()) {
        std::cerr << "[ProjectProcedure] Edge properties: [";
        for (size_t i = 0; i < edge_properties.size(); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << edge_properties[i];
        }
        std::cerr << "]" << std::endl;
    }

    // Get db_folder from global file_manager
    std::string db_folder = file_manager.get_file_path("");
    // Remove trailing slash if present
    if (!db_folder.empty() && db_folder.back() == '/') {
        db_folder.pop_back();
    }

    NativeProjectionBuilder builder(graph_name, db_folder, node_properties, edge_properties);
    builder.scan_nodes_by_labels(node_labels);
    builder.scan_edges_by_types(relationship_types);
    auto stats = builder.finalize();

    std::cerr << "[ProjectProcedure] Projection complete: " << stats.node_count
              << " nodes, " << stats.relationship_count << " edges" << std::endl;

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

std::vector<std::string> ProjectProcedure::parse_node_projection(ProcedureContext& ctx) {
    ObjectId arg = ctx.get_argument(1);
    auto type = GQL_OID::get_type(arg);

    // Case 1: String variant - single label
    if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string label = Conversions::unpack_string(arg);
        return {label};
    }

    // Case 2: List variant - multiple labels
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

    // Invalid type
    throw std::runtime_error(
        "nodeProjection must be STRING or LIST<STRING>, got type: " +
        std::to_string(static_cast<int>(type)) + ". "
        "Provide either a single label string or a list of label strings."
    );
}

std::vector<std::string> ProjectProcedure::parse_relationship_projection(ProcedureContext& ctx) {
    ObjectId arg = ctx.get_argument(2);
    auto type = GQL_OID::get_type(arg);

    // Case 1: String variant - single type
    if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::string rel_type = Conversions::unpack_string(arg);
        return {rel_type};
    }

    // Case 2: List variant - multiple types
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

    // Invalid type
    throw std::runtime_error(
        "relationshipProjection must be STRING or LIST<STRING>, got type: " +
        std::to_string(static_cast<int>(type)) + ". "
        "Provide either a single type string or a list of type strings."
    );
}

void ProjectProcedure::validate_label_exists(const std::string& label) {
    // Access the global GQLModel catalog
    auto it = gql_model.catalog.node_labels2id.find(label);

    if (it == gql_model.catalog.node_labels2id.end()) {
        // Label doesn't exist - build helpful error message
        std::ostringstream oss;
        oss << "Node label '" << label << "' does not exist in the database.\n\n";

        // Add available labels list for better error messages
        if (!gql_model.catalog.node_labels_str.empty()) {
            oss << "Available node labels: [";
            for (size_t i = 0; i < gql_model.catalog.node_labels_str.size(); i++) {
                if (i > 0) oss << ", ";
                oss << "'" << gql_model.catalog.node_labels_str[i] << "'";
            }
            oss << "]\n";
        } else {
            oss << "No node labels exist in the database.\n";
        }

        oss << "Hint: Labels are case-sensitive. Check your database schema.";

        throw std::runtime_error(oss.str());
    }
}

void ProjectProcedure::validate_type_exists(const std::string& type) {
    // Access the global GQLModel catalog
    auto it = gql_model.catalog.edge_labels2id.find(type);

    if (it == gql_model.catalog.edge_labels2id.end()) {
        // Type doesn't exist - build helpful error message
        std::ostringstream oss;
        oss << "Relationship type '" << type << "' does not exist in the database.\n\n";

        // Add available types list for better error messages
        if (!gql_model.catalog.edge_labels_str.empty()) {
            oss << "Available relationship types: [";
            for (size_t i = 0; i < gql_model.catalog.edge_labels_str.size(); i++) {
                if (i > 0) oss << ", ";
                oss << "'" << gql_model.catalog.edge_labels_str[i] << "'";
            }
            oss << "]\n";
        } else {
            oss << "No relationship types exist in the database.\n";
        }

        oss << "Hint: Types are case-sensitive. Check your database schema.";

        throw std::runtime_error(oss.str());
    }
}

std::vector<std::string> ProjectProcedure::parse_property_list_from_config(
    ProcedureContext& ctx,
    ObjectId config_map,
    const std::string& key
) {
    std::cerr << "[DEBUG parse_property_list_from_config] Entry: key='" << key << "'" << std::endl;
    std::cerr << "[DEBUG] config_map type: " << static_cast<int>(GQL_OID::get_type(config_map)) << std::endl;

    // Unpack dictionary to get key-value pairs
    std::unique_ptr<Dictionary> dict = Common::Conversions::unpack_dictionary(config_map);
    std::cerr << "[DEBUG] Dictionary unpacked successfully" << std::endl;

    // Cast to DictionaryObject to access keys map
    auto dict_obj = dynamic_cast<DictionaryObject*>(dict->dictionary.get());
    if (!dict_obj) {
        std::cerr << "[DEBUG] FAILED: dict_obj cast returned null" << std::endl;
        throw std::runtime_error("Configuration parameter must be a dictionary/map");
    }
    std::cerr << "[DEBUG] dict_obj keys count: " << dict_obj->keys.size() << std::endl;

    // Find the key in the dictionary
    ObjectId value;
    bool found = false;
    for (const auto& [key_oid, val_item] : dict_obj->keys) {
        // Unpack key and check if it matches
        std::string key_str = Conversions::unpack_string(key_oid);
        std::cerr << "[DEBUG] Checking dict key: '" << key_str << "' against target: '" << key << "'" << std::endl;
        if (key_str == key) {
            std::cerr << "[DEBUG] MATCH FOUND for key: '" << key << "'" << std::endl;
            // Extract ObjectId from DictionaryLiteral
            auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
            if (!lit) {
                std::cerr << "[DEBUG] FAILED: DictionaryLiteral cast returned null" << std::endl;
                throw std::runtime_error("Configuration value for '" + key + "' must be a literal (string or list)");
            }
            value = lit->object_id;
            std::cerr << "[DEBUG] Extracted value ObjectId: " << value.id << std::endl;
            found = true;
            break;
        }
    }

    if (!found) {
        std::cerr << "[DEBUG] Key '" << key << "' NOT FOUND in dictionary - returning empty vector" << std::endl;
        // Key not present - return empty vector (properties not requested)
        return {};
    }
    std::cerr << "[DEBUG] Key '" << key << "' found, proceeding to type check" << std::endl;
    auto value_type = GQL_OID::get_type(value);
    std::cerr << "[DEBUG] Value type: " << static_cast<int>(value_type) << std::endl;

    // Case 1: String variant - single property
    if (value_type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        value_type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        value_type == GQL_OID::Type::STRING_SIMPLE_TMP)
    {
        std::cerr << "[DEBUG] Processing as STRING" << std::endl;
        std::string prop_name = Conversions::unpack_string(value);
        std::cerr << "[DEBUG] Extracted single property: '" << prop_name << "'" << std::endl;
        return {prop_name};
    }

    // Case 2: List variant - multiple properties
    if (value_type == GQL_OID::Type::LIST) {
        std::cerr << "[DEBUG] Processing as LIST" << std::endl;
        std::vector<ObjectId> list_items = Conversions::unpack_list(value);
        std::cerr << "[DEBUG] List contains " << list_items.size() << " items" << std::endl;

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
            std::cerr << "[DEBUG] List item [" << i << "]: '" << prop_name << "'" << std::endl;
            properties.push_back(prop_name);
        }

        std::cerr << "[DEBUG] Returning " << properties.size() << " properties for key '" << key << "'" << std::endl;
        return properties;
    }

    // Invalid type
    throw std::runtime_error(
        "Configuration parameter '" + key + "' must be STRING or LIST<STRING>, got type: " +
        std::to_string(static_cast<int>(value_type)) + ". "
        "Provide either a single property name or a list of property names."
    );
}
