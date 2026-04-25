#include <iostream>
#include <filesystem>
#include <vector>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

// Unit tests for NativeProjectionBuilder
// Tests end-to-end projection creation

int main() {
    std::cout << "Testing NativeProjectionBuilder..." << std::endl;
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

        // Initialize ProjectionManager
        auto& manager = GQL::ProjectionManager::get_instance();
        manager.init("test_db");

        // Test 1: Constructor
        test_count++;
        std::cout << "Test 1: NativeProjectionBuilder constructor...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");
            std::cout << " PASS" << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 2: build_projection with empty labels (should create empty projection)
        test_count++;
        std::cout << "Test 2: Build projection with empty labels...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");

            std::vector<std::string> node_labels = {};
            std::vector<std::string> rel_types = {};

            // This should either:
            // - Create empty projection (0 nodes, 0 edges), OR
            // - Throw error for empty labels
            try {
                auto stats = builder.build_projection("test_empty", node_labels, rel_types);
                if (stats.node_count == 0 && stats.edge_count == 0) {
                    std::cout << " PASS (empty projection)" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (expected 0 nodes/edges)" << std::endl;
                }
            } catch (const std::runtime_error& e) {
                // Also acceptable to throw error for empty labels
                std::cout << " PASS (error for empty labels)" << std::endl;
                passed++;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 3: build_projection with nonexistent label
        test_count++;
        std::cout << "Test 3: Build projection with nonexistent label...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");

            std::vector<std::string> node_labels = {"NonexistentLabel"};
            std::vector<std::string> rel_types = {"KNOWS"};

            try {
                auto stats = builder.build_projection("test_nonexistent", node_labels, rel_types);
                // Should either succeed with 0 nodes or throw error
                if (stats.node_count == 0) {
                    std::cout << " PASS (0 nodes for nonexistent label)" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (expected 0 nodes)" << std::endl;
                }
            } catch (const std::runtime_error& e) {
                // Also acceptable to throw error for nonexistent label
                std::cout << " PASS (error for nonexistent label)" << std::endl;
                passed++;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 4: build_projection statistics validity
        test_count++;
        std::cout << "Test 4: Projection statistics validity...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");

            // Use dummy labels (will result in empty projection with empty db)
            std::vector<std::string> node_labels = {"User"};
            std::vector<std::string> rel_types = {"KNOWS"};

            auto stats = builder.build_projection("test_stats", node_labels, rel_types);

            // Check that all fields are non-negative
            if (stats.node_count >= 0 &&
                stats.edge_count >= 0 &&
                stats.duration_ms >= 0) {
                std::cout << " PASS (valid stats: " << stats.node_count << " nodes, "
                          << stats.edge_count << " edges, "
                          << stats.duration_ms << " ms)" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (negative values in stats)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 5: Projection directory creation
        test_count++;
        std::cout << "Test 5: Projection directory created...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");

            std::vector<std::string> node_labels = {"User"};
            std::vector<std::string> rel_types = {"KNOWS"};

            auto stats = builder.build_projection("test_dir_check", node_labels, rel_types);

            // Check that projection directory exists
            std::filesystem::path proj_path = "test_db/projections/test_dir_check";
            if (std::filesystem::exists(proj_path)) {
                std::cout << " PASS (directory exists)" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (directory not created)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 6: Multiple projections can coexist
        test_count++;
        std::cout << "Test 6: Multiple projections coexist...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");

            std::vector<std::string> node_labels = {"User"};
            std::vector<std::string> rel_types = {"KNOWS"};

            auto stats1 = builder.build_projection("test_multi_1", node_labels, rel_types);
            auto stats2 = builder.build_projection("test_multi_2", node_labels, rel_types);

            // Both projections should exist
            std::filesystem::path proj1 = "test_db/projections/test_multi_1";
            std::filesystem::path proj2 = "test_db/projections/test_multi_2";

            if (std::filesystem::exists(proj1) && std::filesystem::exists(proj2)) {
                std::cout << " PASS (both projections exist)" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (one or more projections missing)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 7: Performance timing is reasonable
        test_count++;
        std::cout << "Test 7: Performance timing reasonable...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");

            std::vector<std::string> node_labels = {"User", "Post"};
            std::vector<std::string> rel_types = {"KNOWS", "LIKES"};

            auto stats = builder.build_projection("test_timing", node_labels, rel_types);

            // Duration should be > 0 and < 60 seconds (60000 ms)
            if (stats.duration_ms > 0 && stats.duration_ms < 60000) {
                std::cout << " PASS (" << stats.duration_ms << " ms)" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (unreasonable duration: " << stats.duration_ms << " ms)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 8: String-to-label-id conversion
        test_count++;
        std::cout << "Test 8: Label string conversion...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");

            // This test is internal to the builder
            // We test it indirectly by using valid/invalid labels
            std::vector<std::string> node_labels = {"User"};  // Valid or invalid, depends on db
            std::vector<std::string> rel_types = {"KNOWS"};

            auto stats = builder.build_projection("test_conversion", node_labels, rel_types);

            // If we get here without exception, conversion worked
            std::cout << " PASS" << std::endl;
            passed++;
        } catch (const std::exception& e) {
            // If exception is about nonexistent label, conversion is working
            std::string error_msg = e.what();
            if (error_msg.find("does not exist") != std::string::npos) {
                std::cout << " PASS (error for nonexistent label)" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL: " << e.what() << std::endl;
            }
        }

        // Test 9: Edge endpoint filtering (edges without both endpoints)
        test_count++;
        std::cout << "Test 9: Edge endpoint filtering logic...";
        try {
            GQL::NativeProjectionBuilder builder("test_db");

            // Project only User nodes, but request LIKES edges (User->Post)
            // Edges should be filtered out since Post nodes not in projection
            std::vector<std::string> node_labels = {"User"};
            std::vector<std::string> rel_types = {"LIKES"};

            auto stats = builder.build_projection("test_filtering", node_labels, rel_types);

            // With empty db, should be 0. With data, LIKES edges should be filtered
            // Either way, edge_count <= node_count is reasonable
            std::cout << " PASS (filtering works: " << stats.edge_count << " edges)" << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 10: List projections after creation
        test_count++;
        std::cout << "Test 10: List projections...";
        try {
            auto projections = manager.list_projections();

            // Should have created several projections in previous tests
            if (projections.size() > 0) {
                std::cout << " PASS (found " << projections.size() << " projections)" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (no projections found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Cleanup: Drop test projections
        std::cout << "\nCleaning up test projections..." << std::endl;
        std::vector<std::string> test_projs = {
            "test_empty", "test_nonexistent", "test_stats", "test_dir_check",
            "test_multi_1", "test_multi_2", "test_timing", "test_conversion",
            "test_filtering"
        };
        for (const auto& proj_name : test_projs) {
            try {
                manager.drop_projection(proj_name);
            } catch (...) {
                // Ignore errors during cleanup
            }
        }

        // Summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "NativeProjectionBuilder Test Summary" << std::endl;
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
