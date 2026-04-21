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

        // Scope catalog and storage so they destruct before drop_projection
        {
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
            // Flush batched nodes to disk before checking counts
            storage.flush();
            auto node_count = storage.get_node_count();
            if (node_count != 5) {
                std::cerr << "\nFAIL Test 5: expected node count 5, got " << node_count << std::endl;
                return 1;
            }
            std::cout << " OK (count: " << node_count << ")" << std::endl;

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
            // Flush batched edges to disk before checking counts
            storage.flush();
            auto edge_count = storage.get_edge_count();
            if (edge_count != 4) {
                std::cerr << "\nFAIL Test 6: expected edge count 4, got " << edge_count << std::endl;
                return 1;
            }
            std::cout << " OK (count: " << edge_count << ")" << std::endl;

            // Test 7: Check node existence
            std::cout << "Test 7: Checking node existence...";
            bool exists = storage.has_node(ObjectId(3));
            if (!exists) {
                std::cerr << "FAIL Test 7: node 3 should exist" << std::endl;
                return 1;
            }
            std::cout << " OK (node 3 exists: yes)" << std::endl;

            // ================================================================
            // Sorted-vector node dedup regression suite (2026-04-20)
            // ================================================================
            // See docs/superpowers/thesis_analysis/2026-04-20-node-bloom-scan-memory-design.md
            // for design. These tests guard the contract that the append-only
            // vector + finalize_node_scan + binary-search model preserves the
            // exact semantics previously delivered by std::unordered_set.

            // Test 10: Duplicate inserts collapse after finalize_node_scan
            std::cout << "Test 10: Duplicate add_node + finalize_node_scan...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                // (Re-use existing init files; s is read-write, but we only
                // exercise the in-memory path here; no flush/finalize is
                // invoked so no B+Tree state is touched.)
                GQL::ProjectedNode n;
                n.node_id = ObjectId(1001);
                s.add_node(n);
                s.add_node(n);         // exact duplicate
                s.add_node(n);         // another duplicate
                // finalize_node_scan should collapse these to a single entry.
                s.finalize_node_scan();
                if (!s.has_node(ObjectId(1001))) {
                    std::cerr << "FAIL Test 10: node 1001 should exist after dup inserts" << std::endl;
                    return 1;
                }
                // Ensure has_node returns false for something we never added.
                if (s.has_node(ObjectId(9999))) {
                    std::cerr << "FAIL Test 10: node 9999 should NOT exist" << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;

            // Test 11: finalize_node_scan is idempotent (2 calls = 1 call)
            std::cout << "Test 11: finalize_node_scan idempotency...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                for (uint64_t i = 2001; i <= 2010; i++) {
                    GQL::ProjectedNode n;
                    n.node_id = ObjectId(i);
                    s.add_node(n);
                }
                s.finalize_node_scan();
                s.finalize_node_scan();  // second call must not corrupt state
                for (uint64_t i = 2001; i <= 2010; i++) {
                    if (!s.has_node(ObjectId(i))) {
                        std::cerr << "FAIL Test 11: node " << i
                                  << " missing after double finalize" << std::endl;
                        return 1;
                    }
                }
            }
            std::cout << " OK" << std::endl;

            // Test 12: has_node pre-finalize uses linear-scan fallback
            std::cout << "Test 12: has_node pre-finalize (linear-scan fallback)...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                for (uint64_t i = 3001; i <= 3005; i++) {
                    GQL::ProjectedNode n;
                    n.node_id = ObjectId(i);
                    s.add_node(n);
                }
                // Intentionally do NOT call finalize_node_scan; must still work.
                if (!s.has_node(ObjectId(3003))) {
                    std::cerr << "FAIL Test 12: pre-finalize has_node(3003) should be true" << std::endl;
                    return 1;
                }
                if (s.has_node(ObjectId(3999))) {
                    std::cerr << "FAIL Test 12: pre-finalize has_node(3999) should be false" << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;

            // Test 13: has_node post-finalize uses binary search
            std::cout << "Test 13: has_node post-finalize (binary search)...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                // Insert 1000 unsorted ids and verify all are found post-finalize.
                // Using a larger set increases the chance of catching a
                // binary-search off-by-one on non-trivial cardinalities.
                constexpr int N = 1000;
                for (int i = 0; i < N; i++) {
                    GQL::ProjectedNode n;
                    // Interleaved ordering so the unsorted vector is not
                    // already in insertion order.
                    n.node_id = ObjectId(static_cast<uint64_t>(4000 + ((i * 7919) % N)));
                    s.add_node(n);
                }
                s.finalize_node_scan();
                for (int i = 0; i < N; i++) {
                    if (!s.has_node(ObjectId(static_cast<uint64_t>(4000 + i)))) {
                        std::cerr << "FAIL Test 13: post-finalize has_node missed id "
                                  << (4000 + i) << std::endl;
                        return 1;
                    }
                }
                // Negative probe around the inserted range.
                if (s.has_node(ObjectId(3999)) || s.has_node(ObjectId(5000))) {
                    std::cerr << "FAIL Test 13: boundary false positive" << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;

            // Test 14: duplicate collapse count (multi-label emulation)
            std::cout << "Test 14: multi-label duplicate collapse...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                // Simulate the multi-label path: same node emitted multiple
                // times across label scans. Even 10× duplicates must collapse
                // to a single has_node-observable entity.
                GQL::ProjectedNode n;
                n.node_id = ObjectId(6001);
                for (int i = 0; i < 10; i++) {
                    s.add_node(n);
                }
                s.finalize_node_scan();
                if (!s.has_node(ObjectId(6001))) {
                    std::cerr << "FAIL Test 14: multi-label duplicate absent after finalize" << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;
        } // storage and catalog destruct here, before drop

        // Test 8: List projections
        std::cout << "Test 8: Listing projections...";
        auto projections = manager.list_projections();
        std::cout << " OK (found: " << projections.size() << ")" << std::endl;

        // Test 9: Drop projection (directory is deleted)
        std::cout << "Test 9: Dropping projection...";
        bool dropped = manager.drop_projection("test_projection");
        std::cout << " OK (dropped: " << (dropped ? "yes" : "no") << ")" << std::endl;

        std::cout << "\nAll tests passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
