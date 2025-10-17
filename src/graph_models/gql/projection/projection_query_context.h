#pragma once

#include <memory>
#include <string>

#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace GQL {

// Holds projection indexes for query execution
// This class manages the lifetime of projection storage during query execution
class ProjectionQueryContext {
public:
    std::string projection_name;
    std::unique_ptr<ProjectionStorage> storage;

    // Cached references to projection indexes for fast access
    BPlusTree<1>* nodes_index = nullptr;
    BPlusTree<3>* from_to_edge_index = nullptr;
    BPlusTree<3>* to_from_edge_index = nullptr;
    BPlusTree<2>* edge_direction_index = nullptr;

    explicit ProjectionQueryContext(const std::string& proj_name)
        : projection_name(proj_name)
    {
        auto& manager = ProjectionManager::get_instance();
        std::string proj_dir = manager.get_projection_dir(proj_name);
        std::string db_folder = manager.get_db_folder();

        storage = std::make_unique<ProjectionStorage>(proj_dir, db_folder);
        storage->open();  // Open existing projection

        // Cache index pointers for fast access during query execution
        nodes_index = storage->get_nodes_index();
        from_to_edge_index = storage->get_from_to_edge_index();
        to_from_edge_index = storage->get_to_from_edge_index();
        edge_direction_index = storage->get_edge_direction_index();
    }

    ~ProjectionQueryContext() = default;

    // Helper method to check if context is valid
    bool is_valid() const {
        return nodes_index != nullptr &&
               from_to_edge_index != nullptr &&
               to_from_edge_index != nullptr &&
               edge_direction_index != nullptr;
    }
};

} // namespace GQL
