#include <iostream>
#include <filesystem>

#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace fs = std::filesystem;

// Phase A3: Test backward compatibility and optional features
int main() {
    std::cout << "=== Phase A3: Backward Compatibility & Optional Features Test ===" << std::endl;

    try {
        std::filesystem::remove_all("test_db_features");

        // Initialize system
        System system(
            "test_db_features",
            1024 * 1024,      // str_static_size
            1024 * 1024,      // str_dynamic_size
            64 * 1024 * 1024, // shared_buffer_size
            32 * 1024 * 1024, // private_buffer_size
            1024 * 1024,      // tensor_static_size
            1024 * 1024,      // tensor_dynamic_size
            1                 // workers
        );

        QueryContext query_ctx;
        QueryContext::set_query_ctx(&query_ctx);

        auto& manager = GQL::ProjectionManager::get_instance();
        manager.init("test_db_features");

        // Test 1: Create v1.0-style projection (no optional features)
        std::cout << "\nTest 1: Creating v1.0-style projection (topology only)...";
        std::string proj_v1_dir = manager.create_projection("legacy_projection");
        {
            GQL::ProjectionStorage storage(proj_v1_dir, "test_db_features", "legacy_projection");
            storage.init();

            // Add some data
            for (uint64_t i = 1; i <= 3; i++) {
                GQL::ProjectedNode node;
                node.node_id = ObjectId(i);
                storage.add_node(node);
            }

            GQL::ProjectedEdge edge;
            edge.from_node = ObjectId(1);
            edge.to_node = ObjectId(2);
            edge.edge_id = ObjectId(100);
            edge.is_directed = true;
            storage.add_edge(edge);

            storage.flush();
        }
        std::cout << " OK" << std::endl;

        // Test 2: Verify v1.0 catalog has correct defaults
        std::cout << "Test 2: Verifying v1.0 catalog defaults...";
        {
            GQL::ProjectionCatalog catalog(proj_v1_dir);
            catalog.load();

            if (catalog.projection_name != "legacy_projection") {
                throw std::runtime_error("Catalog name mismatch");
            }
            if (catalog.node_count != 3) {
                throw std::runtime_error("Node count mismatch");
            }
            if (catalog.edge_count != 1) {
                throw std::runtime_error("Edge count mismatch");
            }
            // v1.0 projections should have all optional features disabled
            if (catalog.includes_node_labels) {
                throw std::runtime_error("v1.0 catalog should not include node labels");
            }
            if (catalog.includes_edge_labels) {
                throw std::runtime_error("v1.0 catalog should not include edge labels");
            }
            if (catalog.includes_node_properties) {
                throw std::runtime_error("v1.0 catalog should not include node properties");
            }
            if (catalog.includes_edge_properties) {
                throw std::runtime_error("v1.0 catalog should not include edge properties");
            }
        }
        std::cout << " OK" << std::endl;

        // Test 3: Open v1.0 projection and verify no optional indexes
        std::cout << "Test 3: Opening v1.0 projection (backward compatibility)...";
        {
            GQL::ProjectionStorage storage(proj_v1_dir, "test_db_features");
            storage.open();

            // Required indexes should exist
            if (!storage.get_nodes_index()) {
                throw std::runtime_error("Required nodes index missing");
            }
            if (!storage.get_from_to_edge_index()) {
                throw std::runtime_error("Required from_to_edge index missing");
            }

            // Optional indexes should be null
            if (storage.get_node_label_index()) {
                throw std::runtime_error("v1.0 projection should not have node label index");
            }
            if (storage.get_edge_label_index()) {
                throw std::runtime_error("v1.0 projection should not have edge label index");
            }
            if (storage.get_node_key_value_index()) {
                throw std::runtime_error("v1.0 projection should not have node property index");
            }
            if (storage.get_edge_key_value_index()) {
                throw std::runtime_error("v1.0 projection should not have edge property index");
            }

            // Verify data is accessible
            if (!storage.has_node(ObjectId(1))) {
                throw std::runtime_error("Node 1 should exist");
            }
            if (storage.get_node_count() != 3) {
                throw std::runtime_error("Node count mismatch after reopen");
            }
        }
        std::cout << " OK" << std::endl;

        // Test 4: Create v1.1 projection with all optional features enabled
        std::cout << "\nTest 4: Creating v1.1 projection with all optional features...";
        std::string proj_v1_1_dir = manager.create_projection("full_featured_projection");
        {
            GQL::ProjectionStorage::Features features;
            features.include_node_labels = true;
            features.include_edge_labels = true;
            features.include_node_properties = true;
            features.include_edge_properties = true;

            GQL::ProjectionStorage storage(proj_v1_1_dir, "test_db_features", "full_featured_projection", features);
            storage.init();

            // Add data
            for (uint64_t i = 1; i <= 5; i++) {
                GQL::ProjectedNode node;
                node.node_id = ObjectId(i);
                storage.add_node(node);
            }

            GQL::ProjectedEdge edge;
            edge.from_node = ObjectId(1);
            edge.to_node = ObjectId(2);
            edge.edge_id = ObjectId(200);
            edge.is_directed = false;
            storage.add_edge(edge);

            storage.flush();
        }
        std::cout << " OK" << std::endl;

        // Test 5: Verify v1.1 catalog has feature flags set
        std::cout << "Test 5: Verifying v1.1 catalog feature flags...";
        {
            GQL::ProjectionCatalog catalog(proj_v1_1_dir);
            catalog.load();

            if (!catalog.includes_node_labels) {
                throw std::runtime_error("v1.1 catalog should include node labels");
            }
            if (!catalog.includes_edge_labels) {
                throw std::runtime_error("v1.1 catalog should include edge labels");
            }
            if (!catalog.includes_node_properties) {
                throw std::runtime_error("v1.1 catalog should include node properties");
            }
            if (!catalog.includes_edge_properties) {
                throw std::runtime_error("v1.1 catalog should include edge properties");
            }
            if (catalog.node_count != 5) {
                throw std::runtime_error("Node count mismatch");
            }
            if (catalog.edge_count != 1) {
                throw std::runtime_error("Edge count mismatch");
            }
        }
        std::cout << " OK" << std::endl;

        // Test 6: Open v1.1 projection and verify optional indexes exist
        std::cout << "Test 6: Opening v1.1 projection (auto-detection)...";
        {
            GQL::ProjectionStorage storage(proj_v1_1_dir, "test_db_features");
            storage.open();

            // All indexes should exist (required + optional)
            if (!storage.get_nodes_index()) {
                throw std::runtime_error("Required nodes index missing");
            }
            if (!storage.get_from_to_edge_index()) {
                throw std::runtime_error("Required from_to_edge index missing");
            }
            if (!storage.get_node_label_index()) {
                throw std::runtime_error("Node label index should exist");
            }
            if (!storage.get_edge_label_index()) {
                throw std::runtime_error("Edge label index should exist");
            }
            if (!storage.get_node_key_value_index()) {
                throw std::runtime_error("Node property index should exist");
            }
            if (!storage.get_edge_key_value_index()) {
                throw std::runtime_error("Edge property index should exist");
            }

            // Verify data is accessible
            if (storage.get_node_count() != 5) {
                throw std::runtime_error("Node count mismatch after reopen");
            }
        }
        std::cout << " OK" << std::endl;

        // Test 7: Create v1.1 projection with selective features
        std::cout << "\nTest 7: Creating v1.1 projection with selective features...";
        std::string proj_selective_dir = manager.create_projection("selective_projection");
        {
            GQL::ProjectionStorage::Features features;
            features.include_node_labels = true;  // Only node labels
            features.include_edge_labels = false;
            features.include_node_properties = false;
            features.include_edge_properties = false;

            GQL::ProjectionStorage storage(proj_selective_dir, "test_db_features", "selective_projection", features);
            storage.init();

            GQL::ProjectedNode node;
            node.node_id = ObjectId(1);
            storage.add_node(node);

            storage.flush();
        }
        std::cout << " OK" << std::endl;

        // Test 8: Verify selective features
        std::cout << "Test 8: Verifying selective features...";
        {
            GQL::ProjectionStorage storage(proj_selective_dir, "test_db_features");
            storage.open();

            // Only node label index should exist
            if (!storage.get_node_label_index()) {
                throw std::runtime_error("Node label index should exist");
            }
            if (storage.get_edge_label_index()) {
                throw std::runtime_error("Edge label index should NOT exist");
            }
            if (storage.get_node_key_value_index()) {
                throw std::runtime_error("Node property index should NOT exist");
            }
            if (storage.get_edge_key_value_index()) {
                throw std::runtime_error("Edge property index should NOT exist");
            }
        }
        std::cout << " OK" << std::endl;

        // Test 9: Verify file existence for optional indexes
        std::cout << "\nTest 9: Verifying physical file existence...";
        {
            // v1.0 projection - no optional index files
            if (fs::exists(proj_v1_dir + "/node_label.leaf")) {
                throw std::runtime_error("v1.0 should not have node_label files");
            }

            // v1.1 full projection - all optional index files
            if (!fs::exists(proj_v1_1_dir + "/node_label.leaf")) {
                throw std::runtime_error("v1.1 full should have node_label files");
            }
            if (!fs::exists(proj_v1_1_dir + "/edge_label.leaf")) {
                throw std::runtime_error("v1.1 full should have edge_label files");
            }
            if (!fs::exists(proj_v1_1_dir + "/node_key_value.leaf")) {
                throw std::runtime_error("v1.1 full should have node_key_value files");
            }
            if (!fs::exists(proj_v1_1_dir + "/edge_key_value.leaf")) {
                throw std::runtime_error("v1.1 full should have edge_key_value files");
            }

            // Selective projection - only node_label files
            if (!fs::exists(proj_selective_dir + "/node_label.leaf")) {
                throw std::runtime_error("Selective should have node_label files");
            }
            if (fs::exists(proj_selective_dir + "/edge_label.leaf")) {
                throw std::runtime_error("Selective should NOT have edge_label files");
            }
        }
        std::cout << " OK" << std::endl;

        // Cleanup
        std::cout << "\nCleaning up test projections...";
        manager.drop_projection("legacy_projection");
        manager.drop_projection("full_featured_projection");
        manager.drop_projection("selective_projection");
        std::cout << " OK" << std::endl;

        std::cout << "\n=== All Phase A3 tests passed! ===" << std::endl;
        std::filesystem::remove_all("test_db_features");
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
