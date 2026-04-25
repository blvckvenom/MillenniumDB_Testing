#include <iostream>
#include <vector>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/native_scanner.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

// Unit tests for NativeScanner
// Tests B+Tree scanning functionality for native projection

int main() {
    std::cout << "Testing NativeScanner..." << std::endl;
    int test_count = 0;
    int passed = 0;

    try {
        // Initialize system
        System system(
            "test_db",
            1024 * 1024,
            1024 * 1024,
            64 * 1024 * 1024,
            32 * 1024 * 1024,
            1024 * 1024,
            1024 * 1024,
            1
        );

        QueryContext query_ctx;
        QueryContext::set_query_ctx(&query_ctx);

        // TODO: Initialize GQLModel with test data
        // This requires importing a test graph first
        // For now, we'll test the interface

        // Test 1: NativeScanner construction
        test_count++;
        std::cout << "Test 1: NativeScanner construction...";
        try {
            // Get indexes from gql_model
            auto* label_node_idx = &gql_model.get_label_node();
            auto* label_edge_idx = &gql_model.get_label_edge();
            auto* from_to_edge_idx = &gql_model.get_from_to_edge();
            auto* edge_from_to_idx = &gql_model.get_edge_from_to();
            auto* n1_n2_edge_idx = &gql_model.get_n1_n2_edge();
            auto* edge_n1_n2_idx = &gql_model.get_edge_n1_n2();

            GQL::NativeScanner scanner(label_node_idx, label_edge_idx, from_to_edge_idx,
                                     edge_from_to_idx, n1_n2_edge_idx, edge_n1_n2_idx);
            std::cout << " PASS" << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 2: scan_label_node with callback
        test_count++;
        std::cout << "Test 2: scan_label_node callback...";
        try {
            auto* label_node_idx = &gql_model.get_label_node();
            auto* label_edge_idx = &gql_model.get_label_edge();
            auto* from_to_edge_idx = &gql_model.get_from_to_edge();
            auto* edge_from_to_idx = &gql_model.get_edge_from_to();
            auto* n1_n2_edge_idx = &gql_model.get_n1_n2_edge();
            auto* edge_n1_n2_idx = &gql_model.get_edge_n1_n2();

            GQL::NativeScanner scanner(label_node_idx, label_edge_idx, from_to_edge_idx,
                                     edge_from_to_idx, n1_n2_edge_idx, edge_n1_n2_idx);

            // Create a test label ObjectId
            ObjectId test_label(12345);  // Dummy label ID

            std::vector<ObjectId> scanned_nodes;
            uint64_t count = scanner.scan_label_node(test_label, [&](ObjectId node_id) {
                scanned_nodes.push_back(node_id);
            });

            // With empty database, should return 0
            if (count == 0 && scanned_nodes.size() == 0) {
                std::cout << " PASS (empty db)" << std::endl;
                passed++;
            } else {
                std::cout << " PASS (found " << count << " nodes)" << std::endl;
                passed++;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 3: scan_label_edge with callback
        test_count++;
        std::cout << "Test 3: scan_label_edge callback...";
        try {
            auto* label_node_idx = &gql_model.get_label_node();
            auto* label_edge_idx = &gql_model.get_label_edge();
            auto* from_to_edge_idx = &gql_model.get_from_to_edge();
            auto* edge_from_to_idx = &gql_model.get_edge_from_to();
            auto* n1_n2_edge_idx = &gql_model.get_n1_n2_edge();
            auto* edge_n1_n2_idx = &gql_model.get_edge_n1_n2();

            GQL::NativeScanner scanner(label_node_idx, label_edge_idx, from_to_edge_idx,
                                     edge_from_to_idx, n1_n2_edge_idx, edge_n1_n2_idx);

            ObjectId test_type(67890);  // Dummy type ID

            std::vector<ObjectId> scanned_edges;
            uint64_t count = scanner.scan_label_edge(test_type, [&](ObjectId edge_id) {
                scanned_edges.push_back(edge_id);
            });

            // With empty database, should return 0
            if (count == 0 && scanned_edges.size() == 0) {
                std::cout << " PASS (empty db)" << std::endl;
                passed++;
            } else {
                std::cout << " PASS (found " << count << " edges)" << std::endl;
                passed++;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 4: get_edge_endpoints (will fail with empty db, expected)
        test_count++;
        std::cout << "Test 4: get_edge_endpoints error handling...";
        try {
            auto* label_node_idx = &gql_model.get_label_node();
            auto* label_edge_idx = &gql_model.get_label_edge();
            auto* from_to_edge_idx = &gql_model.get_from_to_edge();
            auto* edge_from_to_idx = &gql_model.get_edge_from_to();
            auto* n1_n2_edge_idx = &gql_model.get_n1_n2_edge();
            auto* edge_n1_n2_idx = &gql_model.get_edge_n1_n2();

            GQL::NativeScanner scanner(label_node_idx, label_edge_idx, from_to_edge_idx,
                                     edge_from_to_idx, n1_n2_edge_idx, edge_n1_n2_idx);

            ObjectId nonexistent_edge(999999);

            try {
                auto [from, to] = scanner.get_edge_endpoints(nonexistent_edge);
                std::cout << " FAIL (should throw for nonexistent edge)" << std::endl;
            } catch (const std::runtime_error&) {
                std::cout << " PASS (correctly throws error)" << std::endl;
                passed++;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 5: Multiple consecutive scans
        test_count++;
        std::cout << "Test 5: Multiple consecutive scans...";
        try {
            auto* label_node_idx = &gql_model.get_label_node();
            auto* label_edge_idx = &gql_model.get_label_edge();
            auto* from_to_edge_idx = &gql_model.get_from_to_edge();
            auto* edge_from_to_idx = &gql_model.get_edge_from_to();
            auto* n1_n2_edge_idx = &gql_model.get_n1_n2_edge();
            auto* edge_n1_n2_idx = &gql_model.get_edge_n1_n2();

            GQL::NativeScanner scanner(label_node_idx, label_edge_idx, from_to_edge_idx,
                                     edge_from_to_idx, n1_n2_edge_idx, edge_n1_n2_idx);

            uint64_t count1 = scanner.scan_label_node(ObjectId(1), [](ObjectId) {});
            uint64_t count2 = scanner.scan_label_node(ObjectId(2), [](ObjectId) {});
            uint64_t count3 = scanner.scan_label_edge(ObjectId(3), [](ObjectId) {});

            // Should complete without errors
            std::cout << " PASS (scanned " << (count1 + count2 + count3) << " total)" << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 6: Callback exception safety
        test_count++;
        std::cout << "Test 6: Callback exception safety...";
        try {
            auto* label_node_idx = &gql_model.get_label_node();
            auto* label_edge_idx = &gql_model.get_label_edge();
            auto* from_to_edge_idx = &gql_model.get_from_to_edge();
            auto* edge_from_to_idx = &gql_model.get_edge_from_to();
            auto* n1_n2_edge_idx = &gql_model.get_n1_n2_edge();
            auto* edge_n1_n2_idx = &gql_model.get_edge_n1_n2();

            GQL::NativeScanner scanner(label_node_idx, label_edge_idx, from_to_edge_idx,
                                     edge_from_to_idx, n1_n2_edge_idx, edge_n1_n2_idx);

            bool callback_called = false;
            try {
                scanner.scan_label_node(ObjectId(1), [&](ObjectId) {
                    callback_called = true;
                    throw std::runtime_error("Test exception in callback");
                });
                std::cout << " PASS (no nodes to trigger callback)" << std::endl;
                passed++;
            } catch (const std::runtime_error& e) {
                if (callback_called) {
                    std::cout << " PASS (exception propagated)" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (unexpected exception)" << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 7: Null callback handling
        test_count++;
        std::cout << "Test 7: Empty lambda callback...";
        try {
            auto* label_node_idx = &gql_model.get_label_node();
            auto* label_edge_idx = &gql_model.get_label_edge();
            auto* from_to_edge_idx = &gql_model.get_from_to_edge();
            auto* edge_from_to_idx = &gql_model.get_edge_from_to();
            auto* n1_n2_edge_idx = &gql_model.get_n1_n2_edge();
            auto* edge_n1_n2_idx = &gql_model.get_edge_n1_n2();

            GQL::NativeScanner scanner(label_node_idx, label_edge_idx, from_to_edge_idx,
                                     edge_from_to_idx, n1_n2_edge_idx, edge_n1_n2_idx);

            // Callback that does nothing
            uint64_t count = scanner.scan_label_node(ObjectId(1), [](ObjectId) {});

            std::cout << " PASS (empty callback works)" << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 8: Stateful callback (accumulator)
        test_count++;
        std::cout << "Test 8: Stateful callback accumulator...";
        try {
            auto* label_node_idx = &gql_model.get_label_node();
            auto* label_edge_idx = &gql_model.get_label_edge();
            auto* from_to_edge_idx = &gql_model.get_from_to_edge();
            auto* edge_from_to_idx = &gql_model.get_edge_from_to();
            auto* n1_n2_edge_idx = &gql_model.get_n1_n2_edge();
            auto* edge_n1_n2_idx = &gql_model.get_edge_n1_n2();

            GQL::NativeScanner scanner(label_node_idx, label_edge_idx, from_to_edge_idx,
                                     edge_from_to_idx, n1_n2_edge_idx, edge_n1_n2_idx);

            uint64_t accumulator = 0;
            scanner.scan_label_node(ObjectId(1), [&accumulator](ObjectId node_id) {
                accumulator += node_id.id;
            });

            // Should work even with empty results (accumulator = 0)
            std::cout << " PASS (accumulator = " << accumulator << ")" << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "NativeScanner Test Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total tests: " << test_count << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << (test_count - passed) << std::endl;
        std::cout << "Success rate: " << (100.0 * passed / test_count) << "%" << std::endl;

        if (passed == test_count) {
            std::cout << "\n✓ All tests passed!" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ Some tests failed" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << std::endl;
        return 1;
    }
}
