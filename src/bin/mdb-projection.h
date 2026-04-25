#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "bin/common.h"
#include "graph_models/exceptions.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "misc/fatal_error.h"
#include "system/system.h"

namespace MdbBin {

// List all projections in a database
inline int mdb_list_projections(const std::vector<std::string>& args) {
    if (args.size() != 1) {
        FATAL_ERROR("Expected 1 arg for mdb list-projections, received ", args.size());
    }

    const std::string& db_directory = args[0];

    // Check if it's a GQL database
    auto model_id = Catalog::get_model_id(db_directory);
    if (model_id != Catalog::ModelID::GQL) {
        FATAL_ERROR("Projections are only supported for GQL databases");
    }

    // Initialize system
    System system(
        db_directory,
        StringManager::DEFAULT_STATIC_BUFFER,
        StringManager::DEFAULT_DYNAMIC_BUFFER,
        BufferManager::DEFAULT_VERSIONED_PAGES_BUFFER_SIZE,
        BufferManager::DEFAULT_PRIVATE_PAGES_BUFFER_SIZE,
        TensorManager::DEFAULT_STATIC_BUFFER,
        TensorManager::DEFAULT_DYNAMIC_BUFFER,
        1
    );

    QueryContext qc;
    QueryContext::set_query_ctx(&qc);

    try {
        auto model_destroyer = GQLModel::init();

        auto& proj_manager = GQL::ProjectionManager::get_instance();
        proj_manager.init(db_directory);

        auto projections = proj_manager.list_projections();

        if (projections.empty()) {
            std::cout << "No projections found in database" << std::endl;
            return EXIT_SUCCESS;
        }

        std::cout << "\nProjections in database " << db_directory << ":\n" << std::endl;
        std::cout << std::left << std::setw(30) << "NAME"
                  << std::setw(15) << "NODES"
                  << std::setw(15) << "EDGES"
                  << "DIRECTORY" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (const auto& proj : projections) {
            std::cout << std::left << std::setw(30) << proj.name
                      << std::setw(15) << proj.node_count
                      << std::setw(15) << proj.edge_count
                      << proj.directory << std::endl;
        }

        std::cout << "\nTotal: " << projections.size() << " projection(s)" << std::endl;

    } catch (const WrongModelException& e) {
        FATAL_ERROR(e.what());
    } catch (const WrongCatalogVersionException& e) {
        FATAL_ERROR(e.what());
    } catch (const std::exception& e) {
        FATAL_ERROR("Error listing projections: ", e.what());
    }

    return EXIT_SUCCESS;
}

// Drop a projection from a database
inline int mdb_drop_projection(const std::vector<std::string>& args) {
    if (args.size() != 2) {
        FATAL_ERROR("Expected 2 args for mdb drop-projection, received ", args.size());
    }

    const std::string& db_directory = args[0];
    const std::string& projection_name = args[1];

    // Check if it's a GQL database
    auto model_id = Catalog::get_model_id(db_directory);
    if (model_id != Catalog::ModelID::GQL) {
        FATAL_ERROR("Projections are only supported for GQL databases");
    }

    // Initialize system
    System system(
        db_directory,
        StringManager::DEFAULT_STATIC_BUFFER,
        StringManager::DEFAULT_DYNAMIC_BUFFER,
        BufferManager::DEFAULT_VERSIONED_PAGES_BUFFER_SIZE,
        BufferManager::DEFAULT_PRIVATE_PAGES_BUFFER_SIZE,
        TensorManager::DEFAULT_STATIC_BUFFER,
        TensorManager::DEFAULT_DYNAMIC_BUFFER,
        1
    );

    QueryContext qc;
    QueryContext::set_query_ctx(&qc);

    try {
        auto model_destroyer = GQLModel::init();

        auto& proj_manager = GQL::ProjectionManager::get_instance();
        proj_manager.init(db_directory);

        if (!proj_manager.projection_exists(projection_name)) {
            FATAL_ERROR("Projection '", projection_name, "' does not exist");
        }

        std::cout << "Dropping projection '" << projection_name << "'..." << std::endl;

        bool success = proj_manager.drop_projection(projection_name);

        if (success) {
            std::cout << "Projection '" << projection_name << "' dropped successfully" << std::endl;
        } else {
            FATAL_ERROR("Failed to drop projection '", projection_name, "'");
        }

    } catch (const WrongModelException& e) {
        FATAL_ERROR(e.what());
    } catch (const WrongCatalogVersionException& e) {
        FATAL_ERROR(e.what());
    } catch (const std::exception& e) {
        FATAL_ERROR("Error dropping projection: ", e.what());
    }

    return EXIT_SUCCESS;
}

// Inspect a projection (show detailed statistics)
inline int mdb_inspect_projection(const std::vector<std::string>& args) {
    if (args.size() != 2) {
        FATAL_ERROR("Expected 2 args for mdb inspect-projection, received ", args.size());
    }

    const std::string& db_directory = args[0];
    const std::string& projection_name = args[1];

    // Check if it's a GQL database
    auto model_id = Catalog::get_model_id(db_directory);
    if (model_id != Catalog::ModelID::GQL) {
        FATAL_ERROR("Projections are only supported for GQL databases");
    }

    // Initialize system
    System system(
        db_directory,
        StringManager::DEFAULT_STATIC_BUFFER,
        StringManager::DEFAULT_DYNAMIC_BUFFER,
        BufferManager::DEFAULT_VERSIONED_PAGES_BUFFER_SIZE,
        BufferManager::DEFAULT_PRIVATE_PAGES_BUFFER_SIZE,
        TensorManager::DEFAULT_STATIC_BUFFER,
        TensorManager::DEFAULT_DYNAMIC_BUFFER,
        1
    );

    QueryContext qc;
    QueryContext::set_query_ctx(&qc);

    try {
        auto model_destroyer = GQLModel::init();

        auto& proj_manager = GQL::ProjectionManager::get_instance();
        proj_manager.init(db_directory);

        if (!proj_manager.projection_exists(projection_name)) {
            FATAL_ERROR("Projection '", projection_name, "' does not exist");
        }

        auto catalog = proj_manager.get_projection_catalog(projection_name);

        // Helper: renders a string vector as "[a, b, c]" or "(none)" when empty.
        auto fmt_strvec = [](const std::vector<std::string>& v) {
            if (v.empty()) {
                return std::string("(none)");
            }
            std::string out = "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) out += ", ";
                out += v[i];
            }
            out += "]";
            return out;
        };

        // Collect keys actually registered in the projection (includes synthetics
        // like "_count" and renamed projected names that don't appear in the
        // main graph catalog).
        std::vector<std::string> registered_node_keys;
        for (const auto& name : catalog->node_keys_str) {
            if (!name.empty()) registered_node_keys.push_back(name);
        }
        std::vector<std::string> registered_edge_keys;
        for (const auto& name : catalog->edge_keys_str) {
            if (!name.empty()) registered_edge_keys.push_back(name);
        }

        std::cout << "\nProjection: " << catalog->projection_name << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Directory:          " << proj_manager.get_projection_dir(projection_name) << std::endl;
        std::cout << "Created:            " << catalog->creation_timestamp << std::endl;
        std::cout << "Build time (ms):    " << catalog->projection_millis << std::endl;
        std::cout << std::endl;

        std::cout << "-- Graph statistics ------------------------------------------------------------" << std::endl;
        std::cout << "Node count:         " << catalog->node_count << std::endl;
        std::cout << "Edge count:         " << catalog->edge_count << std::endl;
        std::cout << "Directed edges:     " << catalog->directed_edge_count << std::endl;
        std::cout << "Undirected edges:   " << catalog->undirected_edge_count << std::endl;
        std::cout << std::endl;

        std::cout << "-- Index features --------------------------------------------------------------" << std::endl;
        std::cout << "Node labels:        " << (catalog->includes_node_labels ? "yes" : "no") << std::endl;
        std::cout << "Edge labels:        " << (catalog->includes_edge_labels ? "yes" : "no") << std::endl;
        std::cout << "Node properties:    " << (catalog->includes_node_properties ? "yes" : "no") << std::endl;
        std::cout << "Edge properties:    " << (catalog->includes_edge_properties ? "yes" : "no") << std::endl;
        std::cout << std::endl;

        std::cout << "-- Projected properties --------------------------------------------------------" << std::endl;
        // Prefer v1.1+ `included_*_properties`; fall back to legacy v1.0 field
        // so older projections still display something useful.
        const auto& node_props = catalog->included_node_properties.empty()
            ? catalog->node_property_names
            : catalog->included_node_properties;
        const auto& edge_props = catalog->included_edge_properties.empty()
            ? catalog->edge_property_names
            : catalog->included_edge_properties;
        std::cout << "Requested node properties: " << fmt_strvec(node_props) << std::endl;
        std::cout << "Requested edge properties: " << fmt_strvec(edge_props) << std::endl;
        std::cout << "Registered node keys:      " << fmt_strvec(registered_node_keys) << std::endl;
        std::cout << "Registered edge keys:      " << fmt_strvec(registered_edge_keys) << std::endl;
        std::cout << std::endl;

        if (!catalog->original_query.empty()) {
            std::cout << "-- Original query --------------------------------------------------------------" << std::endl;
            std::cout << catalog->original_query << std::endl;
            std::cout << std::endl;
        }

    } catch (const WrongModelException& e) {
        FATAL_ERROR(e.what());
    } catch (const WrongCatalogVersionException& e) {
        FATAL_ERROR(e.what());
    } catch (const std::exception& e) {
        FATAL_ERROR("Error inspecting projection: ", e.what());
    }

    return EXIT_SUCCESS;
}

} // namespace MdbBin
