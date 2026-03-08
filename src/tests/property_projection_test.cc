#include <iostream>
#include <memory>
#include <cassert>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "query/query_context.h"
#include "system/system.h"

using namespace GQL;

int passed_tests = 0;
int failed_tests = 0;
std::unique_ptr<System> system_ptr;
std::unique_ptr<QueryContext> query_ctx_ptr;

#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        std::cerr << "FAIL: " << message << std::endl; \
        failed_tests++; \
        return; \
    }

// Helper to initialize database
void init_test_database() {
    std::cout << "Initializing test database..." << std::endl;

    // Initialize system (including file_manager) with minimal buffer sizes
    system_ptr = std::make_unique<System>(
        "/tmp/test_properties_db",  // db_folder
        1024 * 1024,                 // str_static_size (1MB)
        1024 * 1024,                 // str_dynamic_size (1MB)
        64 * 1024 * 1024,            // shared_buffer_size (64MB)
        32 * 1024 * 1024,            // private_buffer_size (32MB)
        1024 * 1024,                 // tensor_static_size (1MB)
        1024 * 1024,                 // tensor_dynamic_size (1MB)
        1                            // workers
    );

    // Set up QueryContext (required for BPlusTree operations)
    query_ctx_ptr = std::make_unique<QueryContext>();
    QueryContext::set_query_ctx(query_ctx_ptr.get());

    // Initialize GQL model
    gql_model.init();

    // Initialize projection manager
    ProjectionManager::get_instance().init("/tmp/test_properties_db");

    std::cout << "OK: Test database initialized" << std::endl;
}

// Test 1: Create projection WITHOUT properties
void test_baseline_projection() {
    std::cout << "\n=== Test 1: Baseline projection (no properties) ===" << std::endl;

    try {
        NativeProjectionBuilder builder("baseline_test", "/tmp/test_properties_db");
        builder.scan_nodes_by_labels({"User"});
        builder.scan_edges_by_types({"Friend"});
        auto stats = builder.finalize();

        std::cout << "OK: Baseline projection created:" << std::endl;
        std::cout << "   Nodes: " << stats.node_count << std::endl;
        std::cout << "   Relationships: " << stats.relationship_count << std::endl;
        std::cout << "   Duration: " << stats.duration_ms.count() << "ms" << std::endl;

        ASSERT_TRUE(stats.node_count > 0, "Node count should be > 0");
        ASSERT_TRUE(stats.relationship_count > 0, "Relationship count should be > 0");

        passed_tests++;
        std::cout << "OK: Test 1 PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: Test 1 FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
}

// Test 2: Create projection WITH node properties
void test_node_properties_projection() {
    std::cout << "\n=== Test 2: Node properties projection ===" << std::endl;

    try {
        std::vector<std::string> node_props = {"age", "name"};
        NativeProjectionBuilder builder("node_props_test", "/tmp/test_properties_db", node_props, {});
        builder.scan_nodes_by_labels({"User"});
        builder.scan_edges_by_types({"Friend"});
        auto stats = builder.finalize();

        std::cout << "OK: Node properties projection created:" << std::endl;
        std::cout << "   Nodes: " << stats.node_count << std::endl;
        std::cout << "   Relationships: " << stats.relationship_count << std::endl;
        std::cout << "   Duration: " << stats.duration_ms.count() << "ms" << std::endl;

        ASSERT_TRUE(stats.node_count > 0, "Node count should be > 0");
        ASSERT_TRUE(stats.relationship_count > 0, "Relationship count should be > 0");

        passed_tests++;
        std::cout << "OK: Test 2 PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: Test 2 FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
}

// Test 3: Create projection WITH edge properties
void test_edge_properties_projection() {
    std::cout << "\n=== Test 3: Edge properties projection ===" << std::endl;

    try {
        std::vector<std::string> edge_props = {"since"};
        NativeProjectionBuilder builder("edge_props_test", "/tmp/test_properties_db", {}, edge_props);
        builder.scan_nodes_by_labels({"User"});
        builder.scan_edges_by_types({"Friend"});
        auto stats = builder.finalize();

        std::cout << "OK: Edge properties projection created:" << std::endl;
        std::cout << "   Nodes: " << stats.node_count << std::endl;
        std::cout << "   Relationships: " << stats.relationship_count << std::endl;
        std::cout << "   Duration: " << stats.duration_ms.count() << "ms" << std::endl;

        ASSERT_TRUE(stats.node_count > 0, "Node count should be > 0");
        ASSERT_TRUE(stats.relationship_count > 0, "Relationship count should be > 0");

        passed_tests++;
        std::cout << "OK: Test 3 PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: Test 3 FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
}

// Test 4: Create projection WITH both node AND edge properties
void test_full_properties_projection() {
    std::cout << "\n=== Test 4: Full properties projection (nodes + edges) ===" << std::endl;

    try {
        std::vector<std::string> node_props = {"age", "name", "email"};
        std::vector<std::string> edge_props = {"since"};
        NativeProjectionBuilder builder("full_props_test", "/tmp/test_properties_db", node_props, edge_props);
        builder.scan_nodes_by_labels({"User"});
        builder.scan_edges_by_types({"Friend"});
        auto stats = builder.finalize();

        std::cout << "OK: Full properties projection created:" << std::endl;
        std::cout << "   Nodes: " << stats.node_count << std::endl;
        std::cout << "   Relationships: " << stats.relationship_count << std::endl;
        std::cout << "   Duration: " << stats.duration_ms.count() << "ms" << std::endl;
        std::cout << "   Node properties: age, name, email" << std::endl;
        std::cout << "   Edge properties: since" << std::endl;

        ASSERT_TRUE(stats.node_count > 0, "Node count should be > 0");
        ASSERT_TRUE(stats.relationship_count > 0, "Relationship count should be > 0");

        passed_tests++;
        std::cout << "OK: Test 4 PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: Test 4 FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
}

int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << "PROPERTY PROJECTION TESTS\n";
    std::cout << "================================================================================\n";

    init_test_database();

    test_baseline_projection();
    test_node_properties_projection();
    test_edge_properties_projection();
    test_full_properties_projection();

    std::cout << "\n================================================================================\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << "================================================================================\n";
    std::cout << "Passed: " << passed_tests << "/" << (passed_tests + failed_tests) << std::endl;
    std::cout << "Failed: " << failed_tests << "/" << (passed_tests + failed_tests) << std::endl;

    if (failed_tests == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cerr << "SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
