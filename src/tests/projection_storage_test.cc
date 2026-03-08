#include <iostream>
#include <filesystem>

#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

// Simple test to verify projection storage layer compiles and links correctly
int main() {
    std::filesystem::remove_all("test_db_storage");

    std::cout << "Testing projection storage layer..." << std::endl;

    try {
        // Initialize system (including file_manager)
        // Use minimal buffer sizes for testing
        System system(
            "test_db_storage",      // db_folder
            1024 * 1024,            // str_static_size (1MB)
            1024 * 1024,            // str_dynamic_size (1MB)
            64 * 1024 * 1024,       // shared_buffer_size (64MB)
            32 * 1024 * 1024,       // private_buffer_size (32MB)
            1024 * 1024,            // tensor_static_size (1MB)
            1024 * 1024,            // tensor_dynamic_size (1MB)
            1                       // workers
        );

        // Set up QueryContext (required for BPlusTree operations)
        QueryContext query_ctx;
        QueryContext::set_query_ctx(&query_ctx);

        // Test 1: ProjectionManager singleton
        std::cout << "Test 1: ProjectionManager initialization...";
        auto& manager = GQL::ProjectionManager::get_instance();
        manager.init("test_db_storage");
        std::cout << " OK" << std::endl;

        // Test 2: Create a projection
        std::cout << "Test 2: Creating projection...";
        std::string proj_dir = manager.create_projection("test_projection");
        std::cout << " OK (dir: " << proj_dir << ")" << std::endl;

        // Test 3: ProjectionCatalog
        std::cout << "Test 3: ProjectionCatalog...";
        GQL::ProjectionCatalog catalog(proj_dir);
        catalog.projection_name = "test_projection";
        catalog.node_count = 10;
        catalog.edge_count = 20;
        catalog.creation_timestamp = 1234567890;
        catalog.save();
        std::cout << " OK" << std::endl;

        // Test 4: ProjectionStorage initialization
        std::cout << "Test 4: ProjectionStorage initialization...";
        GQL::ProjectionStorage storage(proj_dir, "test_db_storage");
        storage.init();
        std::cout << " OK" << std::endl;

        // Test 5: Add nodes
        std::cout << "Test 5: Adding nodes...";
        std::cout.flush();
        for (uint64_t i = 1; i <= 5; i++) {
            std::cout << " [" << i << "]";
            std::cout.flush();
            GQL::ProjectedNode node;
            node.node_id = ObjectId(i);
            storage.add_node(node);
        }
        std::cout << " OK (count: " << storage.get_node_count() << ")" << std::endl;

        // Test 6: Add edges
        std::cout << "Test 6: Adding edges...";
        for (uint64_t i = 1; i < 5; i++) {
            GQL::ProjectedEdge edge;
            edge.from_node = ObjectId(i);
            edge.to_node = ObjectId(i + 1);
            edge.edge_id = ObjectId(100 + i);
            edge.is_directed = true;
            storage.add_edge(edge);
        }
        std::cout << " OK (count: " << storage.get_edge_count() << ")" << std::endl;

        // Test 7: Check node existence
        std::cout << "Test 7: Checking node existence...";
        bool exists = storage.has_node(ObjectId(3));
        std::cout << " OK (node 3 exists: " << (exists ? "yes" : "no") << ")" << std::endl;

        // Test 8: List projections
        std::cout << "Test 8: Listing projections...";
        auto projections = manager.list_projections();
        std::cout << " OK (found: " << projections.size() << ")" << std::endl;

        // Test 9: Drop projection
        std::cout << "Test 9: Dropping projection...";
        bool dropped = manager.drop_projection("test_projection");
        std::cout << " OK (dropped: " << (dropped ? "yes" : "no") << ")" << std::endl;

        std::cout << "\nAll tests passed!" << std::endl;
        std::filesystem::remove_all("test_db_storage");
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
