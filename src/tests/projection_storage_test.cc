#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/bloom_filter.h"
#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/leaf_compression.h"
#include "storage/index/bplus_tree/bpt_mem_import.h"
#include "storage/index/record.h"
#include "graph_models/gql/projection/native_scanner.h"
#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/procedure/builtin/project_procedure.h"
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

        // ================================================================
        // Bloom-filter edge-dedup loss regression (2026-06-15)
        //
        // ProjectionStorage::add_edge can dedup edges with a Bloom filter
        // keyed on (from, to, edge_id). Because edge_id is unique per edge,
        // the filter never catches a true duplicate — its only effect is the
        // probabilistic FALSE POSITIVE, which silently DROPS a legitimate
        // unique edge (and the build-time std::unique() cannot recover an
        // edge that was never inserted). On papers100M (1.6B edges into a
        // 1%-FPR filter) this dropped 2,689,259 unique edges and left ~78k
        // nodes fully isolated. The fix: NativeProjectionBuilder::flush_edges
        // passes skip_bloom_check=true. These checks guard the mechanism.
        std::cout << "Test BLOOM-1: Bloom filter false positives near capacity...";
        {
            GQL::BloomFilter bf(1000, 0.01);
            for (uint64_t i = 0; i < 1000; ++i) {
                bf.add_edge(i, i + 5000000ULL, i + 9000000ULL);  // 1000 distinct, fills to capacity
            }
            size_t fp = 0;
            for (uint64_t i = 50000000ULL; i < 50100000ULL; ++i) {  // 100k distinct, NONE inserted
                if (bf.probably_contains_edge(i, i + 5000000ULL, i + 9000000ULL)) ++fp;
            }
            if (fp == 0) {
                std::cerr << "\nFAIL Test BLOOM-1: expected >0 false positives at "
                             "capacity (proves add_edge's Bloom dedup is lossy), got 0"
                          << std::endl;
                return 1;
            }
            std::cout << " OK (" << fp << " FP / 100k probes)" << std::endl;
        }

        std::cout << "Test BLOOM-2: add_edge(skip_bloom_check=true) conserves all distinct edges...";
        {
            std::string pdir = manager.create_projection("bloom_skip");
            {
                GQL::ProjectionStorage storage(pdir, "test_db_storage");
                storage.init();
                constexpr uint64_t K = 5000;
                for (uint64_t i = 0; i < K; ++i) {
                    GQL::ProjectedEdge edge;
                    edge.from_node = ObjectId(i + 1);
                    edge.to_node   = ObjectId(i + 1000001ULL);
                    edge.edge_id   = ObjectId(i + 2000001ULL);
                    edge.is_directed = true;
                    storage.add_edge(edge, /*skip_bloom_check=*/true);
                }
                storage.flush();
                auto ec = storage.get_edge_count();
                if (ec != K) {
                    std::cerr << "\nFAIL Test BLOOM-2: expected " << K
                              << " distinct edges conserved, got " << ec << std::endl;
                    return 1;
                }
                std::cout << " OK (count: " << ec << ")" << std::endl;
            }
            manager.drop_projection("bloom_skip");
        }

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

            // ============================================================
            // Node presence bitmap. has_node() answers from a dense bitmap
            // when the collected id span fits NODE_BITMAP_MAX_BYTES. The
            // bitmap is an exact image of collected_nodes_, so NO assertion
            // about a RESULT can tell the two apart. NB-2 is the only test
            // here that discriminates: it makes them disagree on purpose.
            // The rest pin the budget arithmetic (NB-1, NB-1b, NB-3) and the
            // invalidation contract (NB-4, NB-5).

            // NB-1: the bitmap is built over the REAL (tagged) id space.
            std::cout << "Test NB-1: bitmap over tagged ids...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                constexpr uint64_t N = 1000;
                for (uint64_t i = 0; i < N; i++) {
                    GQL::ProjectedNode n;
                    // Production ids carry a constant type tag in the high
                    // byte (MASK_NODE = 0xD4..), src/import/gql/import.cc:143.
                    // A budget computed on max_id instead of (max_id - min_id)
                    // sees ~1.5e16 here, refuses, and the whole optimization
                    // ships as a silent no-op on every real graph. Only this
                    // assertion catches that.
                    n.node_id = ObjectId(ObjectId::MASK_NODE | (7000 + i));
                    s.add_node(n);
                }
                s.finalize_node_scan();
                // span 999 -> 1000 bits -> (1000+7)/8 = 125 bytes.
                if (s.node_bitmap_bytes() != 125) {
                    std::cerr << "\nFAIL NB-1: expected 125 bitmap bytes, got "
                              << s.node_bitmap_bytes() << " status="
                              << s.node_bitmap_status() << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;

            // NB-1b: byte-count boundary (63 / 64 / 65 ids). An off-by-one in
            // the span arithmetic is an out-of-range set_kept, which
            // EdgeKeepBitmap would silently absorb by GROWING the vector -
            // i.e. no crash, just a bitmap that no longer matches its budget.
            std::cout << "Test NB-1b: span boundary...";
            {
                const uint64_t counts[3] = {63, 64, 65};
                const std::size_t expect[3] = {8, 8, 9};
                for (int k = 0; k < 3; k++) {
                    GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                    for (uint64_t i = 0; i < counts[k]; i++) {
                        GQL::ProjectedNode n;
                        n.node_id = ObjectId(ObjectId::MASK_NODE | (10000 + i));
                        s.add_node(n);
                    }
                    s.finalize_node_scan();
                    if (s.node_bitmap_bytes() != expect[k]) {
                        std::cerr << "\nFAIL NB-1b: " << counts[k] << " ids expected "
                                  << expect[k] << " B, got " << s.node_bitmap_bytes()
                                  << std::endl;
                        return 1;
                    }
                    for (uint64_t i = 0; i < counts[k]; i++) {
                        if (!s.has_node(ObjectId(ObjectId::MASK_NODE | (10000 + i)))) {
                            std::cerr << "\nFAIL NB-1b: missing id at count "
                                      << counts[k] << std::endl;
                            return 1;
                        }
                    }
                }
            }
            std::cout << " OK" << std::endl;

            // NB-2: has_node answers FROM the bitmap, not from the vector.
            // THE discriminating test. Everything else passes in both worlds.
            std::cout << "Test NB-2: bitmap is CONSULTED...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                constexpr uint64_t N = 1000;
                for (uint64_t i = 0; i < N; i++) {
                    GQL::ProjectedNode n;
                    n.node_id = ObjectId(ObjectId::MASK_NODE | (20000 + i));
                    s.add_node(n);
                }
                s.finalize_node_scan();

                // Destroy the sorted vector behind has_node's back. The bitmap
                // is a SEPARATE exact image of the same set, so every answer
                // below must be unchanged. Without a bitmap in the hot path
                // every probe returns false: binary_search over an empty
                // vector misses and nodes_index is null (this storage was
                // never init()'d). That asymmetry is what makes this the only
                // non-tautological assertion in the group.
                //
                // The const_cast is defined behaviour: `s` is a non-const
                // object, so modifying it through a reference obtained from
                // its const accessor is legal. No production code frees this
                // vector - EdgeKeepBitmapGpuBatcher still uploads it verbatim -
                // so this is a test-only poisoning, not a supported state.
                auto& v = const_cast<std::vector<uint64_t>&>(s.collected_nodes());
                v.clear();

                for (uint64_t i = 0; i < N; i++) {
                    if (!s.has_node(ObjectId(ObjectId::MASK_NODE | (20000 + i)))) {
                        std::cerr << "\nFAIL NB-2: id " << (20000 + i)
                                  << " unreachable with the vector emptied; "
                                     "has_node is not reading the bitmap"
                                  << std::endl;
                        return 1;
                    }
                }
                if (s.has_node(ObjectId(ObjectId::MASK_NODE | 19999))
                    || s.has_node(ObjectId(ObjectId::MASK_NODE | 21000))) {
                    std::cerr << "\nFAIL NB-2: bitmap boundary false positive"
                              << std::endl;
                    return 1;
                }
                // Same payload, different type tag: must NOT collide onto a
                // set bit. This is the guard against someone "simplifying"
                // the index to (id & VALUE_MASK).
                if (s.has_node(ObjectId(ObjectId::MASK_NODE_LABEL | 20500))
                    || s.has_node(ObjectId(20500))) {
                    std::cerr << "\nFAIL NB-2: type-tag collision (bitmap is "
                                 "being indexed by payload, not raw id)"
                              << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;

            // NB-3: a span over budget is refused and answers stay correct.
            // With a 64 MiB ceiling this branch is unreachable on every graph
            // in the project, so this is the ONLY place it ever executes.
            std::cout << "Test NB-3: over-budget span falls back...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                const uint64_t ids[3] = {1ULL, 1ULL << 40, 1ULL << 41};
                for (uint64_t id : ids) {
                    GQL::ProjectedNode n;
                    n.node_id = ObjectId(id);
                    s.add_node(n);
                }
                s.finalize_node_scan();
                // span = 2^41 bits = 256 GB. The budget MUST refuse it.
                if (s.node_bitmap_bytes() != 0
                    || std::string(s.node_bitmap_status()) != "span-exceeds-budget") {
                    std::cerr << "\nFAIL NB-3: budget accepted a 256 GB range ("
                              << s.node_bitmap_bytes() << " B, status="
                              << s.node_bitmap_status() << ")" << std::endl;
                    return 1;
                }
                // Oracle: identical answers to a plain binary search.
                std::vector<uint64_t> oracle(ids, ids + 3);
                std::sort(oracle.begin(), oracle.end());
                const uint64_t probes[6] = {1ULL, 2ULL, 1ULL << 39, 1ULL << 40,
                                            1ULL << 41, ~0ULL};
                for (uint64_t p : probes) {
                    const bool want = std::binary_search(oracle.begin(), oracle.end(), p);
                    if (s.has_node(ObjectId(p)) != want) {
                        std::cerr << "\nFAIL NB-3: fallback disagrees with the "
                                     "oracle on " << p << std::endl;
                        return 1;
                    }
                }
            }
            std::cout << " OK" << std::endl;

            // NB-4: mixed type tags refuse the bitmap AND stay distinguishable.
            std::cout << "Test NB-4: mixed type tags...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                GQL::ProjectedNode a, b;
                a.node_id = ObjectId(ObjectId::MASK_NODE | 1);
                b.node_id = ObjectId(ObjectId::MASK_NAMED_NODE | 1);
                s.add_node(a);
                s.add_node(b);
                s.finalize_node_scan();
                if (s.node_bitmap_bytes() != 0) {
                    std::cerr << "\nFAIL NB-4: a 0xB4<<56 span must not fit"
                              << std::endl;
                    return 1;
                }
                if (!s.has_node(a.node_id) || !s.has_node(b.node_id)) {
                    std::cerr << "\nFAIL NB-4: a mixed-tag set lost a member"
                              << std::endl;
                    return 1;
                }
                if (s.has_node(ObjectId(1)) || s.has_node(ObjectId(ObjectId::MASK_NODE | 2))) {
                    std::cerr << "\nFAIL NB-4: mixed-tag false positive" << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;

            // NB-5: add_node after finalize invalidates the bitmap.
            // Guard, not a discriminator: this passes today. It fails only if
            // the bitmap is built and NOT dropped on a late append, which is
            // the one defect of this change that yields a silently SMALLER
            // projection instead of an error.
            std::cout << "Test NB-5: late add_node invalidates...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                for (uint64_t i = 0; i < 100; i++) {
                    GQL::ProjectedNode n;
                    n.node_id = ObjectId(ObjectId::MASK_NODE | (30000 + i));
                    s.add_node(n);
                }
                s.finalize_node_scan();
                GQL::ProjectedNode late;
                late.node_id = ObjectId(ObjectId::MASK_NODE | 40000);  // outside span
                s.add_node(late);
                if (!s.has_node(late.node_id)) {
                    std::cerr << "\nFAIL NB-5: node added after finalize is "
                                 "invisible (stale bitmap)" << std::endl;
                    return 1;
                }
                if (!s.has_node(ObjectId(ObjectId::MASK_NODE | 30050))) {
                    std::cerr << "\nFAIL NB-5: pre-finalize node lost" << std::endl;
                    return 1;
                }
                // And a re-finalize rebuilds a bitmap that covers both.
                s.finalize_node_scan();
                if (s.node_bitmap_bytes() == 0
                    || !s.has_node(late.node_id)
                    || !s.has_node(ObjectId(ObjectId::MASK_NODE | 30050))) {
                    std::cerr << "\nFAIL NB-5: re-finalize did not rebuild"
                              << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;

            // NB-6: zero nodes. front()/back() on an empty vector is UB; this
            // is the most likely way the implementation crashes.
            std::cout << "Test NB-6: empty node set...";
            {
                GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                s.finalize_node_scan();
                if (s.node_bitmap_bytes() != 0
                    || std::string(s.node_bitmap_status()) != "empty-node-set"
                    || s.has_node(ObjectId(ObjectId::MASK_NODE | 1))
                    || s.has_node(ObjectId(0))) {
                    std::cerr << "\nFAIL NB-6: empty finalize misbehaved, status="
                              << s.node_bitmap_status() << std::endl;
                    return 1;
                }
            }
            std::cout << " OK" << std::endl;

            // NB-7: single node, including the UINT64_MAX edge.
            std::cout << "Test NB-7: single node...";
            {
                for (uint64_t x : {ObjectId::MASK_NODE | 42ULL, ~0ULL}) {
                    GQL::ProjectionStorage s(proj_dir, "test_db_storage");
                    GQL::ProjectedNode n;
                    n.node_id = ObjectId(x);
                    s.add_node(n);
                    s.finalize_node_scan();
                    if (s.node_bitmap_bytes() != 1) {
                        std::cerr << "\nFAIL NB-7: span 0 must be 1 byte, got "
                                  << s.node_bitmap_bytes() << std::endl;
                        return 1;
                    }
                    if (!s.has_node(ObjectId(x)) || s.has_node(ObjectId(x - 1))
                        || s.has_node(ObjectId(0))) {
                        std::cerr << "\nFAIL NB-7: single-node boundary" << std::endl;
                        return 1;
                    }
                }
            }
            std::cout << " OK" << std::endl;
        } // storage and catalog destruct here, before drop

        // ================================================================
        // Catalog v1.4 IndexSet persistence tests (reduced index-set presets)
        // ================================================================
        // These guard the round-trip of the new IndexSet byte appended to
        // the catalog in v1.4, and the backwards-compat path for v1.3
        // catalogs (which must default to IndexSet::ALL).

        // Each sub-test uses its own throwaway directory so a prior save
        // doesn't bleed its catalog bytes into the next round-trip probe.
        auto make_tmp_proj_dir = [](const std::string& tag) {
            auto base = std::filesystem::temp_directory_path()
                      / ("mdb_t3_6_" + tag);
            std::filesystem::remove_all(base);
            std::filesystem::create_directories(base);
            return base.string();
        };

        // Test 15: CatalogV4RoundtripAll — default ALL round-trips as ALL
        std::cout << "Test 15: CatalogV4RoundtripAll...";
        {
            std::string dir = make_tmp_proj_dir("v4_all");
            {
                GQL::ProjectionCatalog cat(dir);
                cat.projection_name = "t_all";
                cat.index_set = GQL::IndexSet::ALL;
                cat.save();
            }
            GQL::ProjectionCatalog read_back(dir);
            // Constructor loads if file exists.
            if (read_back.index_set != GQL::IndexSet::ALL) {
                std::cerr << "FAIL Test 15: expected IndexSet::ALL" << std::endl;
                return 1;
            }
            std::filesystem::remove_all(dir);
        }
        std::cout << " OK" << std::endl;

        // Test 16: CatalogV4RoundtripGnnMinimal — GNN_MINIMAL round-trips
        std::cout << "Test 16: CatalogV4RoundtripGnnMinimal...";
        {
            std::string dir = make_tmp_proj_dir("v4_gnn_min");
            {
                GQL::ProjectionCatalog cat(dir);
                cat.projection_name = "t_gnn_min";
                cat.index_set = GQL::IndexSet::GNN_MINIMAL;
                cat.save();
            }
            GQL::ProjectionCatalog read_back(dir);
            if (read_back.index_set != GQL::IndexSet::GNN_MINIMAL) {
                std::cerr << "FAIL Test 16: expected IndexSet::GNN_MINIMAL got ordinal "
                          << static_cast<int>(read_back.index_set) << std::endl;
                return 1;
            }
            std::filesystem::remove_all(dir);
        }
        std::cout << " OK" << std::endl;

        // Test 17: CatalogV4RoundtripReadonlyTraversal — READONLY_TRAVERSAL
        std::cout << "Test 17: CatalogV4RoundtripReadonlyTraversal...";
        {
            std::string dir = make_tmp_proj_dir("v4_readonly");
            {
                GQL::ProjectionCatalog cat(dir);
                cat.projection_name = "t_readonly";
                cat.index_set = GQL::IndexSet::READONLY_TRAVERSAL;
                cat.save();
            }
            GQL::ProjectionCatalog read_back(dir);
            if (read_back.index_set != GQL::IndexSet::READONLY_TRAVERSAL) {
                std::cerr << "FAIL Test 17: expected IndexSet::READONLY_TRAVERSAL got ordinal "
                          << static_cast<int>(read_back.index_set) << std::endl;
                return 1;
            }
            std::filesystem::remove_all(dir);
        }
        std::cout << " OK" << std::endl;

        // Test 18: CatalogV3ReadAsAll — hand-crafted v1.3 byte buffer reads
        // as IndexSet::ALL. This is the core backwards-compat guarantee: a
        // catalog whose minor version byte is 3 must never attempt to read
        // an IndexSet byte that isn't there, and must default to ALL.
        std::cout << "Test 18: CatalogV3ReadAsAll...";
        {
            std::string dir = make_tmp_proj_dir("v3_compat");

            // Synthesize a minimal v1.3 catalog by hand. We do this by first
            // saving a v1.4 catalog, then rewriting its version byte to 3
            // and truncating the trailing IndexSet byte. This is more robust
            // than hand-constructing the full binary layout (which depends
            // on the full field list) and still exercises the code path we
            // care about: version==3 ⇒ skip IndexSet read ⇒ default ALL.
            {
                GQL::ProjectionCatalog cat(dir);
                cat.projection_name = "t_v3_compat";
                cat.index_set = GQL::IndexSet::GNN_MINIMAL;  // would be wrong if read
                cat.save();
            }
            auto catalog_path = std::filesystem::path(dir) / "catalog.dat";
            // Rewrite minor version byte to 3 at offset 11
            // (6 magic + 3 mdb_ver + 1 model_id + 1 major_ver = 11 bytes
            // before the catalog minor_ver byte). Reader then treats the
            // catalog as pre-IndexSet (v1.3) and defaults index_set to ALL
            // without attempting to read the v1.4 IndexSet byte or any
            // v1.5+ trailing payload — which is the core backwards-compat
            // guarantee this test pins.
            {
                std::fstream f(catalog_path, std::ios::in | std::ios::out
                                           | std::ios::binary);
                f.seekp(11);
                char three = 3;
                f.write(&three, 1);
                f.close();
            }

            GQL::ProjectionCatalog read_back(dir);
            if (read_back.index_set != GQL::IndexSet::ALL) {
                std::cerr << "FAIL Test 18: v3 catalog must default to IndexSet::ALL, got "
                          << static_cast<int>(read_back.index_set) << std::endl;
                return 1;
            }
            std::filesystem::remove_all(dir);
        }
        std::cout << " OK" << std::endl;

        // Test 19: CatalogV4TruncatedAtIndexSetByte — v1.4 header but the
        // trailing IndexSet byte is missing. Expected behavior: read_uint8
        // hits EOF and throws std::runtime_error("Error reading uint8..."),
        // which the ProjectionCatalog constructor propagates out. This test
        // pins that contract — graceful degradation (defaulting silently)
        // would hide corrupt catalogs, so the read path should fail loudly.
        std::cout << "Test 19: CatalogV4TruncatedAtIndexSetByte...";
        {
            std::string dir = make_tmp_proj_dir("v4_trunc");
            {
                GQL::ProjectionCatalog cat(dir);
                cat.projection_name = "t_v4_trunc";
                cat.index_set = GQL::IndexSet::GNN_MINIMAL;
                cat.save();
            }
            auto catalog_path = std::filesystem::path(dir) / "catalog.dat";
            auto total_size = std::filesystem::file_size(catalog_path);
            // Drop the last byte (the v1.4 IndexSet byte) WITHOUT downgrading
            // the version. Reader should throw when it tries to read past EOF.
            std::filesystem::resize_file(catalog_path, total_size - 1);

            bool threw = false;
            try {
                GQL::ProjectionCatalog read_back(dir);
            } catch (const std::exception&) {
                threw = true;
            }
            if (!threw) {
                std::cerr << "FAIL Test 19: expected exception on truncated v4 catalog"
                          << std::endl;
                return 1;
            }
            std::filesystem::remove_all(dir);
        }
        std::cout << " OK" << std::endl;

        // ================================================================
        // Parallel index-reader-opening parity tests
        // ================================================================
        // Verify MDB_PROJECTION_PARALLEL_READERS=0 (sequential, legacy)
        // and the default (parallel TBB) produce IDENTICAL reader sets.
        // Each variant builds its own throwaway projection so file-id
        // comparison is meaningful only at "non-null reader" granularity.

        // Helper: build a tiny projection with all 4 optional feature
        // flags ON, exercising all 14 reader-open call sites. Returns a
        // bitmask describing which readers are non-null after flush().
        auto build_and_count_readers = [&manager](const std::string& proj_name) -> uint16_t {
            std::string pdir = manager.create_projection(proj_name);
            GQL::ProjectionCatalog cat(pdir);
            cat.projection_name = proj_name;
            cat.save();

            GQL::ProjectionStorage::Features feats;
            feats.include_node_labels    = true;
            feats.include_edge_labels    = true;
            feats.include_node_properties = true;
            feats.include_edge_properties = true;

            GQL::ProjectionStorage s(pdir, "test_db_storage", proj_name, feats);
            s.init();
            // Two nodes + one edge so all topology buffers have content
            // and build_all_indexes_bulk takes the real path. The 4
            // optional buffers (label_*, property_*) need their own
            // populating call for the indexes to be built non-empty —
            // but the reader-open path opens them regardless of content,
            // which is the contract we are validating.
            GQL::ProjectedNode n1; n1.node_id = ObjectId(1); s.add_node(n1);
            GQL::ProjectedNode n2; n2.node_id = ObjectId(2); s.add_node(n2);
            GQL::ProjectedEdge e1;
            e1.from_node = ObjectId(1);
            e1.to_node   = ObjectId(2);
            e1.edge_id   = ObjectId(100);
            e1.is_directed = true;
            s.add_edge(e1);
            s.flush();

            uint16_t mask = 0;
            // Topology (6) — always non-null under default IndexSet::ALL.
            if (s.get_nodes_index())          mask |= (1u << 0);
            if (s.get_from_to_edge_index())   mask |= (1u << 1);
            if (s.get_to_from_edge_index())   mask |= (1u << 2);
            if (s.get_edge_direction_index()) mask |= (1u << 3);
            if (s.get_edge_from_to_index())   mask |= (1u << 4);
            if (s.get_edge_n1_n2_index())     mask |= (1u << 5);
            // Labels (4) — gated by features.include_*_labels (true here).
            if (s.get_node_label_index())     mask |= (1u << 6);
            if (s.get_label_node_index())     mask |= (1u << 7);
            if (s.get_edge_label_index())     mask |= (1u << 8);
            if (s.get_label_edge_index())     mask |= (1u << 9);
            // Properties (4) — gated by features.include_*_properties.
            if (s.get_node_key_value_index()) mask |= (1u << 10);
            if (s.get_key_value_node_index()) mask |= (1u << 11);
            if (s.get_edge_key_value_index()) mask |= (1u << 12);
            if (s.get_key_value_edge_index()) mask |= (1u << 13);
            return mask;
        };

        // Test 20: sequential path (env var = "0")
        std::cout << "Test 20: Phase 4 reader open — sequential path"
                     " (MDB_PROJECTION_PARALLEL_READERS=0)...";
        ::setenv("MDB_PROJECTION_PARALLEL_READERS", "0", /*overwrite=*/1);
        uint16_t seq_mask = build_and_count_readers("test_proj_seq");
        ::unsetenv("MDB_PROJECTION_PARALLEL_READERS");
        const uint16_t expected_mask = 0x3FFFu;  // 14 low bits set
        if (seq_mask != expected_mask) {
            std::cerr << "\nFAIL Test 20: expected reader mask 0x" << std::hex
                      << expected_mask << " got 0x" << seq_mask << std::dec << std::endl;
            return 1;
        }
        std::cout << " OK (mask=0x" << std::hex << seq_mask << std::dec << ")" << std::endl;

        // Test 21: parallel path (env var unset → default)
        std::cout << "Test 21: Phase 4 reader open — parallel path (default)...";
        uint16_t par_mask = build_and_count_readers("test_proj_par");
        if (par_mask != expected_mask) {
            std::cerr << "\nFAIL Test 21: expected reader mask 0x" << std::hex
                      << expected_mask << " got 0x" << par_mask << std::dec << std::endl;
            return 1;
        }
        std::cout << " OK (mask=0x" << std::hex << par_mask << std::dec << ")" << std::endl;

        // Test 22: parity — both paths produced the same reader set.
        std::cout << "Test 22: Parallel == Sequential reader-set parity...";
        if (seq_mask != par_mask) {
            std::cerr << "\nFAIL Test 22: parity broken (seq=0x" << std::hex
                      << seq_mask << " par=0x" << par_mask << std::dec << ")" << std::endl;
            return 1;
        }
        std::cout << " OK" << std::endl;

        // Cleanup
        manager.drop_projection("test_proj_seq");
        manager.drop_projection("test_proj_par");

        // ================================================================
        // Parallel B+Tree node-scan parity tests
        // ================================================================
        // Verify MDB_PROJECTION_PARALLEL_NODE_SCAN=0 (sequential, legacy)
        // and the default (parallel TBB) produce IDENTICAL ordered node
        // sequences when scanning the label_node B+Tree. Synthetic data
        // is built via ProjectionStorage::add_node_label, which lands the
        // exact same {label_id, node_id} record format the scanner
        // consumes in production.

        // Helper: build a populated projection in-place, flush() it (so
        // build_all_indexes_bulk() materializes the .leaf/.dir files AND
        // open_all_bplustree_readers_() loads the BPlusTree readers into
        // the same storage object), then run the scanner against the live
        // readers. The storage object stays alive for the duration of the
        // scan, so its readers are valid. Returns observed sequence under
        // the env-controlled path + the sorted-unique expected sequence.
        auto build_and_scan = [&manager](
            const std::string& proj_name,
            uint64_t label_id_raw,
            std::size_t count,
            const char* env_value)
            -> std::pair<std::vector<uint64_t>, std::vector<uint64_t>>
        {
            if (env_value) {
                ::setenv("MDB_PROJECTION_PARALLEL_NODE_SCAN", env_value, 1);
            } else {
                ::unsetenv("MDB_PROJECTION_PARALLEL_NODE_SCAN");
            }

            std::string pdir = manager.create_projection(proj_name);
            GQL::ProjectionCatalog cat(pdir);
            cat.projection_name = proj_name;
            cat.save();

            GQL::ProjectionStorage::Features feats;
            feats.include_node_labels = true;
            GQL::ProjectionStorage s(pdir, "test_db_storage",
                                     proj_name, feats);
            s.init();

            ObjectId label_id(label_id_raw);
            std::vector<uint64_t> expected;
            expected.reserve(count);
            // Spread node ids across the value space using a coprime stride
            // so a uniform partitioner cannot pass by accident: with stride
            // 0x0001'1234'... and 8+ partitions, every partition's
            // sub-range contains at least one record on counts >= 8.
            const uint64_t stride = 0x0001'1234'5678'9ABCULL;
            uint64_t cur = ObjectId::MASK_NODE | 1;
            for (std::size_t i = 0; i < count; ++i) {
                ObjectId nid(cur);
                GQL::ProjectedNode n;
                n.node_id = nid;
                s.add_node(n);
                s.add_node_label(nid, label_id);
                expected.push_back(cur);
                cur = (cur + stride) | ObjectId::MASK_NODE;
            }
            // flush() builds .leaf/.dir AND opens BPlusTree readers, so
            // s.get_label_node_index() etc. become non-null afterwards.
            s.flush();

            std::sort(expected.begin(), expected.end());
            expected.erase(std::unique(expected.begin(), expected.end()),
                           expected.end());

            // NativeScanner ctor enforces non-null on label_node,
            // label_edge, from_to_edge, n1_n2_edge — but for
            // scan_label_node only label_node is dereferenced. Reuse the
            // projection's label_node and from_to_edge indexes for the
            // unused slots (same N=2 / N=3 types).
            GQL::NativeScanner scanner(
                s.get_label_node_index(),
                s.get_label_node_index(),    // label_edge stub
                s.get_from_to_edge_index(),
                s.get_edge_from_to_index(),
                s.get_from_to_edge_index(),  // n1_n2_edge stub
                s.get_edge_from_to_index()); // edge_n1_n2 stub

            std::vector<uint64_t> observed;
            observed.reserve(count);
            scanner.scan_label_node(ObjectId(label_id_raw),
                [&observed](ObjectId node_id) {
                    observed.push_back(node_id.id);
                });
            return {observed, expected};
        };

        const uint64_t kTestLabelId = ObjectId::MASK_NODE_LABEL | 0xABC;

        // Test 23: parallel scan returns the same sequence as sequential
        // on a non-trivial dataset (256 nodes, spread across the id range).
        std::cout << "Test 23: scan_label_node parallel == sequential (256 nodes)...";
        {
            auto [seq, expected_seq] = build_and_scan(
                "test_proj_node_scan_seq", kTestLabelId, 256, "0");
            manager.drop_projection("test_proj_node_scan_seq");
            auto [par, expected_par] = build_and_scan(
                "test_proj_node_scan_par", kTestLabelId, 256, nullptr);
            manager.drop_projection("test_proj_node_scan_par");
            ::unsetenv("MDB_PROJECTION_PARALLEL_NODE_SCAN");
            if (seq != expected_seq) {
                std::cerr << "\nFAIL Test 23: sequential output mismatch ("
                          << seq.size() << " vs " << expected_seq.size()
                          << ")" << std::endl;
                return 1;
            }
            if (par != expected_par) {
                std::cerr << "\nFAIL Test 23: parallel output mismatch ("
                          << par.size() << " vs " << expected_par.size()
                          << ")" << std::endl;
                return 1;
            }
            if (seq != par) {
                std::cerr << "\nFAIL Test 23: parity broken (seq.size="
                          << seq.size() << " par.size=" << par.size() << ")"
                          << std::endl;
                return 1;
            }
        }
        std::cout << " OK" << std::endl;

        // Test 24: parallel scan handles tiny inputs (under partition
        // count) gracefully — should still emit every record, no
        // partition starvation.
        std::cout << "Test 24: scan_label_node parallel on tiny input (3 nodes)...";
        {
            auto [par, expected] = build_and_scan(
                "test_proj_node_scan_tiny", kTestLabelId, 3, nullptr);
            manager.drop_projection("test_proj_node_scan_tiny");
            ::unsetenv("MDB_PROJECTION_PARALLEL_NODE_SCAN");
            if (par != expected) {
                std::cerr << "\nFAIL Test 24: tiny-input mismatch ("
                          << par.size() << " vs " << expected.size() << ")"
                          << std::endl;
                return 1;
            }
        }
        std::cout << " OK" << std::endl;

        // Test 25: explicit partition-count override behaves correctly.
        // 32 partitions over 64 nodes still produces the full ordered set.
        std::cout << "Test 25: scan_label_node MDB_PROJECTION_NODE_SCAN_PARTITIONS=32...";
        {
            ::setenv("MDB_PROJECTION_NODE_SCAN_PARTITIONS", "32", 1);
            auto [par, expected] = build_and_scan(
                "test_proj_node_scan_p32", kTestLabelId, 64, nullptr);
            manager.drop_projection("test_proj_node_scan_p32");
            ::unsetenv("MDB_PROJECTION_NODE_SCAN_PARTITIONS");
            ::unsetenv("MDB_PROJECTION_PARALLEL_NODE_SCAN");
            if (par != expected) {
                std::cerr << "\nFAIL Test 25: 32-partition mismatch ("
                          << par.size() << " vs " << expected.size() << ")"
                          << std::endl;
                return 1;
            }
        }
        std::cout << " OK" << std::endl;

        // ================================================================
        // Parallel B+Tree edge-scan parity tests
        // ================================================================
        // Verify MDB_PROJECTION_PARALLEL_EDGE_SCAN=0 (sequential, legacy)
        // and the default (parallel TBB) produce IDENTICAL ordered
        // (edge_id, from, to) sequences when scanning the label_edge
        // B+Tree with inline endpoint resolution. Synthetic data is
        // built via ProjectionStorage::add_edge / add_edge_label, which
        // lands the exact same {label_id, edge_id} record format the
        // scanner consumes in production AND populates the
        // edge_from_to / from_to_edge endpoint trees.

        // Helper: build a populated projection in-place, flush() it (so
        // build_all_indexes_bulk() materializes the .leaf/.dir files AND
        // open_all_bplustree_readers_() loads the BPlusTree readers), then
        // run scan_label_edge_with_endpoints against the live readers.
        // Returns observed (sorted by edge_id) sequence under the
        // env-controlled path + the sorted expected sequence.
        auto build_and_scan_edges = [&manager](
            const std::string& proj_name,
            uint64_t edge_label_id_raw,
            std::size_t count,
            const char* env_value)
            -> std::pair<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>,
                         std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>>
        {
            if (env_value) {
                ::setenv("MDB_PROJECTION_PARALLEL_EDGE_SCAN", env_value, 1);
            } else {
                ::unsetenv("MDB_PROJECTION_PARALLEL_EDGE_SCAN");
            }

            std::string pdir = manager.create_projection(proj_name);
            GQL::ProjectionCatalog cat(pdir);
            cat.projection_name = proj_name;
            cat.save();

            GQL::ProjectionStorage::Features feats;
            // Both flags ON so label_node_index AND label_edge_index are
            // materialized — the NativeScanner ctor enforces non-null on
            // both, regardless of which scan path the test exercises.
            feats.include_node_labels = true;
            feats.include_edge_labels = true;
            GQL::ProjectionStorage s(pdir, "test_db_storage",
                                     proj_name, feats);
            s.init();

            ObjectId edge_label_id(edge_label_id_raw);
            std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> expected;
            expected.reserve(count);
            // Spread edge ids and endpoint ids across the value space using
            // coprime strides so a uniform partitioner cannot pass by
            // accident.
            const uint64_t edge_stride = 0x0001'9E37'79B9'7F4AULL;
            const uint64_t from_stride = 0x0000'D34F'A765'1357ULL;
            const uint64_t to_stride   = 0x0000'C0FF'EE12'3456ULL;
            uint64_t cur_edge = ObjectId::MASK_DIRECTED_EDGE | 1;
            uint64_t cur_from = ObjectId::MASK_NODE | 1;
            uint64_t cur_to   = ObjectId::MASK_NODE | 2;
            for (std::size_t i = 0; i < count; ++i) {
                ObjectId eid(cur_edge);
                ObjectId fid(cur_from);
                ObjectId tid(cur_to);
                // The edge endpoints must exist as nodes in the projection
                // so subsequent has_node() callers downstream can succeed,
                // and so the from_to_edge / edge_from_to indexes are
                // populated by add_edge.
                GQL::ProjectedNode nf; nf.node_id = fid; s.add_node(nf);
                GQL::ProjectedNode nt; nt.node_id = tid; s.add_node(nt);
                GQL::ProjectedEdge e;
                e.from_node = fid;
                e.to_node = tid;
                e.edge_id = eid;
                e.is_directed = true;
                s.add_edge(e);
                s.add_edge_label(eid, edge_label_id);
                expected.emplace_back(cur_edge, cur_from, cur_to);
                cur_edge = ((cur_edge + edge_stride)
                            & ~ObjectId::SUB_TYPE_MASK)
                          | ObjectId::MASK_DIRECTED_EDGE;
                cur_from = ((cur_from + from_stride)
                            & ~ObjectId::SUB_TYPE_MASK)
                          | ObjectId::MASK_NODE;
                cur_to = ((cur_to + to_stride)
                          & ~ObjectId::SUB_TYPE_MASK)
                        | ObjectId::MASK_NODE;
            }
            s.flush();

            // Sort expected by edge_id ascending (B+Tree key order).
            std::sort(expected.begin(), expected.end(),
                [](const auto& a, const auto& b) {
                    return std::get<0>(a) < std::get<0>(b);
                });
            // Drop edge-id duplicates (the coprime stride should never
            // produce one, but guard so the test is robust to RNG).
            expected.erase(std::unique(expected.begin(), expected.end(),
                [](const auto& a, const auto& b) {
                    return std::get<0>(a) == std::get<0>(b);
                }),
                expected.end());

            GQL::NativeScanner scanner(
                s.get_label_node_index(),
                s.get_label_edge_index(),
                s.get_from_to_edge_index(),
                s.get_edge_from_to_index(),
                s.get_from_to_edge_index(),  // n1_n2_edge stub
                s.get_edge_from_to_index()); // edge_n1_n2 stub

            std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> observed;
            observed.reserve(count);
            scanner.scan_label_edge_with_endpoints(
                ObjectId(edge_label_id_raw),
                [&observed](ObjectId edge_id, ObjectId from_node,
                            ObjectId to_node) {
                    observed.emplace_back(edge_id.id, from_node.id, to_node.id);
                });
            return {observed, expected};
        };

        const uint64_t kTestEdgeLabelId =
            ObjectId::MASK_EDGE_LABEL | 0xDEF;

        // Test 26: parallel scan returns the same sequence as sequential
        // on a non-trivial dataset (256 edges, spread across the id range).
        std::cout << "Test 26: scan_label_edge_with_endpoints parallel == sequential (256 edges)...";
        {
            auto [seq, expected_seq] = build_and_scan_edges(
                "test_proj_edge_scan_seq", kTestEdgeLabelId, 256, "0");
            manager.drop_projection("test_proj_edge_scan_seq");
            auto [par, expected_par] = build_and_scan_edges(
                "test_proj_edge_scan_par", kTestEdgeLabelId, 256, nullptr);
            manager.drop_projection("test_proj_edge_scan_par");
            ::unsetenv("MDB_PROJECTION_PARALLEL_EDGE_SCAN");
            if (seq != expected_seq) {
                std::cerr << "\nFAIL Test 26: sequential output mismatch ("
                          << seq.size() << " vs " << expected_seq.size()
                          << ")" << std::endl;
                return 1;
            }
            if (par != expected_par) {
                std::cerr << "\nFAIL Test 26: parallel output mismatch ("
                          << par.size() << " vs " << expected_par.size()
                          << ")" << std::endl;
                return 1;
            }
            if (seq != par) {
                std::cerr << "\nFAIL Test 26: parity broken (seq.size="
                          << seq.size() << " par.size=" << par.size() << ")"
                          << std::endl;
                return 1;
            }
            // Endpoint mapping spot check: every observed triple matches
            // the corresponding expected (from, to) pair for its edge_id.
            for (std::size_t i = 0; i < par.size(); ++i) {
                if (par[i] != expected_par[i]) {
                    std::cerr << "\nFAIL Test 26: triple at " << i
                              << " mismatched expected" << std::endl;
                    return 1;
                }
            }
        }
        std::cout << " OK" << std::endl;

        // Test 27: parallel scan handles tiny inputs (under partition
        // count) gracefully — should still emit every edge with correct
        // endpoints, no partition starvation.
        std::cout << "Test 27: scan_label_edge_with_endpoints parallel on tiny input (3 edges)...";
        {
            auto [par, expected] = build_and_scan_edges(
                "test_proj_edge_scan_tiny", kTestEdgeLabelId, 3, nullptr);
            manager.drop_projection("test_proj_edge_scan_tiny");
            ::unsetenv("MDB_PROJECTION_PARALLEL_EDGE_SCAN");
            if (par != expected) {
                std::cerr << "\nFAIL Test 27: tiny-input mismatch ("
                          << par.size() << " vs " << expected.size() << ")"
                          << std::endl;
                return 1;
            }
        }
        std::cout << " OK" << std::endl;

        // Test 28: explicit partition-count override behaves correctly.
        // 32 partitions over 64 edges still produces the full ordered set
        // with correct endpoint mappings.
        std::cout << "Test 28: scan_label_edge_with_endpoints MDB_PROJECTION_EDGE_SCAN_PARTITIONS=32...";
        {
            ::setenv("MDB_PROJECTION_EDGE_SCAN_PARTITIONS", "32", 1);
            auto [par, expected] = build_and_scan_edges(
                "test_proj_edge_scan_p32", kTestEdgeLabelId, 64, nullptr);
            manager.drop_projection("test_proj_edge_scan_p32");
            ::unsetenv("MDB_PROJECTION_EDGE_SCAN_PARTITIONS");
            ::unsetenv("MDB_PROJECTION_PARALLEL_EDGE_SCAN");
            if (par != expected) {
                std::cerr << "\nFAIL Test 28: 32-partition mismatch ("
                          << par.size() << " vs " << expected.size() << ")"
                          << std::endl;
                return 1;
            }
        }
        std::cout << " OK" << std::endl;

        // ================================================================
        // graph_project duplicate-name safety
        // ================================================================
        // Calling graph_project with the name of an existing projection
        // must fail with "already exists" WITHOUT touching the existing
        // projection's directory. The build-failure rollback must only
        // remove state created by the failing invocation itself.

        // Test 29: duplicate graph_project preserves the existing projection
        std::cout << "Test 29: duplicate graph_project preserves existing projection...";
        {
            // ProjectProcedure::execute consults gql_model.catalog for the
            // non-blocking missing-label warnings, so the global model must
            // be initialized. init() also re-inits the ProjectionManager
            // (rescan of test_db_storage/projections).
            auto model_destroyer = GQLModel::init("test_db_storage");

            // Pre-existing projection with a sentinel file that must
            // survive the failed re-projection. Name kept <= 7 chars so
            // the procedure arguments inline without the string manager.
            std::string dup_dir = manager.create_projection("dup_old");
            {
                GQL::ProjectionCatalog cat(dup_dir);
                cat.projection_name = "dup_old";
                cat.save();
            }
            auto sentinel_path = std::filesystem::path(dup_dir) / "sentinel.bin";
            {
                std::ofstream sentinel(sentinel_path, std::ios::binary);
                sentinel << "keep";
            }

            Binding binding(1);
            GQL::ProcedureContext ctx(binding);
            ctx.arguments = {
                ctx.create_string("dup_old"),
                ctx.create_string("User"),
                ctx.create_string("KNOWS"),
            };

            GQL::Procedures::ProjectProcedure proc;
            bool threw = false;
            std::string error_msg;
            try {
                proc.execute(ctx);
            } catch (const std::exception& e) {
                threw = true;
                error_msg = e.what();
            }

            if (!threw || error_msg.find("already exists") == std::string::npos) {
                std::cerr << "\nFAIL Test 29: expected 'already exists' error, "
                          << (threw ? "got: " + error_msg : "but no exception was thrown")
                          << std::endl;
                return 1;
            }
            if (!std::filesystem::exists(sentinel_path)) {
                std::cerr << "\nFAIL Test 29: pre-existing projection was deleted "
                          << "by the failed duplicate graph_project" << std::endl;
                return 1;
            }
            if (!manager.projection_exists("dup_old")) {
                std::cerr << "\nFAIL Test 29: pre-existing projection was "
                          << "unregistered by the failed duplicate graph_project"
                          << std::endl;
                return 1;
            }

            manager.drop_projection("dup_old");
        }
        std::cout << " OK" << std::endl;

        // Test 30: open() must not recreate index files elided by the
        // IndexSet preset. GNN_MINIMAL omits edge_direction / edge_from_to /
        // edge_n1_n2; an unconditional BPlusTree construction in open()
        // would O_CREAT 0-byte .leaf/.dir files for them, breaking the
        // preset's file-count / disk-footprint contract.
        std::cout << "Test 30: open() respects IndexSet preset...";
        {
            std::string min_dir = manager.create_projection("gnnmin");
            {
                GQL::ProjectionStorage min_storage(min_dir, "test_db_storage");
                min_storage.requested_index_set = GQL::IndexSet::GNN_MINIMAL;
                min_storage.init();
                for (uint64_t i = 1; i <= 3; i++) {
                    GQL::ProjectedNode node;
                    node.node_id = ObjectId(i);
                    min_storage.add_node(node);
                }
                GQL::ProjectedEdge edge;
                edge.from_node = ObjectId(1);
                edge.to_node = ObjectId(2);
                edge.edge_id = ObjectId(101);
                edge.is_directed = true;
                min_storage.add_edge(edge);
                min_storage.flush();

                GQL::ProjectionCatalog cat(min_dir);
                cat.projection_name = "gnnmin";
                cat.node_count = 3;
                cat.edge_count = 1;
                cat.index_set = GQL::IndexSet::GNN_MINIMAL;
                cat.save();
            }

            const char* elided[] = { "edge_direction", "edge_from_to", "edge_n1_n2" };
            for (const char* name : elided) {
                auto leaf = std::filesystem::path(min_dir) / (std::string(name) + ".leaf");
                if (std::filesystem::exists(leaf)) {
                    std::cerr << "\nFAIL Test 30: build materialized elided index "
                              << name << std::endl;
                    return 1;
                }
            }

            {
                GQL::ProjectionStorage reopened(min_dir, "test_db_storage");
                reopened.open();
                if (reopened.get_index_set() != GQL::IndexSet::GNN_MINIMAL) {
                    std::cerr << "\nFAIL Test 30: open() did not restore the "
                              << "GNN_MINIMAL preset from the catalog" << std::endl;
                    return 1;
                }
                for (const char* name : elided) {
                    for (const char* ext : { ".leaf", ".dir" }) {
                        auto f = std::filesystem::path(min_dir)
                               / (std::string(name) + ext);
                        if (std::filesystem::exists(f)) {
                            std::cerr << "\nFAIL Test 30: open() created 0-byte "
                                      << name << ext
                                      << " for an index elided by GNN_MINIMAL"
                                      << std::endl;
                            return 1;
                        }
                    }
                }
                if (reopened.get_nodes_index() == nullptr
                    || reopened.get_from_to_edge_index() == nullptr
                    || reopened.get_to_from_edge_index() == nullptr)
                {
                    std::cerr << "\nFAIL Test 30: open() missing a GNN_MINIMAL "
                              << "index reader" << std::endl;
                    return 1;
                }
            }

            manager.drop_projection("gnnmin");
        }
        std::cout << " OK" << std::endl;

        // ================================================================
        // Leaf redundant-byte compression roundtrip (2026-06-16)
        //
        // ProjectionStorage's bulk/streaming index builders had been writing
        // BTREE+BITSET leaves UNCOMPRESSED (all-zero bitset). GQL::
        // compute_redundant_bitset + GQL::pack_compressed_page now emit the
        // redundant-byte layout the reader (BPTLeafV1) already decodes. These
        // tests drive the REAL BPTLeafWriter::process_block and read the page
        // back with the reader's exact set_record formula, asserting:
        //   (a) byte-exact roundtrip of every record,
        //   (b) the on-disk payload shrinks when bytes are redundant,
        //   (c) the control case (no shared bytes) yields count()==0 and the
        //       full uncompressed size, still roundtripping.
        //
        // The readback uses the SAME decode arithmetic as BPTLeafV1::set_record
        // (bplus_tree_leaf.cc:79) so a mismatch here is a layout bug.
        std::cout << "Test LEAFCOMP: redundant-byte compression roundtrip...";
        {
            constexpr std::size_t TN = 3;            // edge index width (src,dst,eid)
            constexpr std::size_t REC = sizeof(uint64_t) * TN; // 24 bytes

            // Decode a single page's records the way BPTLeafV1 does, directly
            // from the 4096-byte page buffer. Returns the decoded records.
            auto decode_page = [&](const char* page,
                                   std::vector<Record<TN>>& out) {
                const uint32_t value_count =
                    *reinterpret_cast<const uint32_t*>(page);
                const unsigned char* bitset_ptr =
                    reinterpret_cast<const unsigned char*>(page + 2 * sizeof(uint32_t));
                // Rebuild the bitset exactly like the reader ctor.
                std::bitset<REC> rb;
                std::size_t pos_bitset = 0;
                for (std::size_t i = 0; i < TN; ++i) {
                    for (int bit = 0; bit < 8; ++bit) {
                        rb.set(pos_bitset++, (bitset_ptr[i] >> bit) & 1);
                    }
                }
                const std::size_t redundant_count = rb.count();
                const unsigned char* redundant_bytes =
                    bitset_ptr + TN;
                const unsigned char* records =
                    bitset_ptr + TN + redundant_count;

                out.clear();
                for (uint32_t r = 0; r < value_count; ++r) {
                    Record<TN> rec{};
                    unsigned char* oc = reinterpret_cast<unsigned char*>(&rec);
                    const unsigned char* cur =
                        records + r * (REC - redundant_count);
                    std::size_t rpos = 0, upos = 0;
                    for (std::size_t i = 0; i < REC; ++i) {
                        if (rb[i]) {
                            oc[i] = redundant_bytes[rpos++];
                        } else {
                            oc[i] = cur[upos++];
                        }
                    }
                    out.push_back(rec);
                }
            };

            // Write one page via the real writer, read it back, verify.
            auto write_and_check = [&](const std::vector<Record<TN>>& recs,
                                       std::size_t expect_min_count,
                                       const char* label) -> bool {
                const auto bitset =
                    GQL::compute_redundant_bitset<TN>(recs.data(),
                                                      static_cast<uint32_t>(recs.size()));
                if (bitset.count() < expect_min_count) {
                    std::cerr << "\nFAIL LEAFCOMP[" << label << "]: bitset.count()="
                              << bitset.count() << " < expected >= "
                              << expect_min_count << std::endl;
                    return false;
                }

                // Pack + write via the real BPTLeafWriter::process_block.
                std::vector<char> buf(REC + bitset.count()
                                      + recs.size() * (REC - bitset.count()) + 16);
                GQL::pack_compressed_page<TN>(recs.data(),
                                              static_cast<uint32_t>(recs.size()),
                                              bitset, buf.data());

                const std::string fn = "test_db_storage/leafcomp_" + std::string(label) + ".leaf";
                {
                    BPTLeafWriter<TN> w(fn);
                    w.process_block(buf.data(),
                                    static_cast<uint32_t>(recs.size()),
                                    bitset, 0);
                } // dtor flushes/closes

                // Read the single 4096-byte page back.
                std::ifstream in(fn, std::ios::binary);
                std::vector<char> page(4096);
                in.read(page.data(), 4096);
                if (in.gcount() != 4096) {
                    std::cerr << "\nFAIL LEAFCOMP[" << label
                              << "]: page read short" << std::endl;
                    return false;
                }

                std::vector<Record<TN>> decoded;
                decode_page(page.data(), decoded);
                if (decoded.size() != recs.size()) {
                    std::cerr << "\nFAIL LEAFCOMP[" << label << "]: decoded "
                              << decoded.size() << " != " << recs.size() << std::endl;
                    return false;
                }
                for (std::size_t r = 0; r < recs.size(); ++r) {
                    if (decoded[r] != recs[r]) {
                        std::cerr << "\nFAIL LEAFCOMP[" << label
                                  << "]: record " << r << " mismatch" << std::endl;
                        return false;
                    }
                }
                std::filesystem::remove(fn);
                return true;
            };

            // Case A: dense IDs sharing constant high bytes (top ~5 bytes are
            // zero/constant across all records ⇒ many redundant bits).
            {
                std::vector<Record<TN>> recs = {
                    Record<TN>{0x0000001234567ULL, 0x00000022aabbULL, 0x000000300001ULL},
                    Record<TN>{0x0000001234599ULL, 0x00000022aaccULL, 0x000000300002ULL},
                    Record<TN>{0x00000012346abULL, 0x00000022aaddULL, 0x000000300003ULL},
                };
                // Compute on-disk compressed payload size vs uncompressed.
                const auto bs = GQL::compute_redundant_bitset<TN>(recs.data(), 3);
                const std::size_t compressed =
                    TN + bs.count() + recs.size() * (REC - bs.count());
                const std::size_t uncompressed = TN + recs.size() * REC;
                if (compressed >= uncompressed) {
                    std::cerr << "\nFAIL LEAFCOMP[A]: compressed " << compressed
                              << " not < uncompressed " << uncompressed << std::endl;
                    return 1;
                }
                if (!write_and_check(recs, /*expect_min_count=*/1, "A")) return 1;
            }

            // Case B (control): records share NO constant byte position in any
            // field ⇒ count()==0, full uncompressed size, still roundtrips.
            {
                std::vector<Record<TN>> recs = {
                    Record<TN>{0x0102030405060708ULL, 0x1112131415161718ULL, 0x2122232425262728ULL},
                    Record<TN>{0xF1F2F3F4F5F6F7F8ULL, 0xE1E2E3E4E5E6E7E8ULL, 0xD1D2D3D4D5D6D7D8ULL},
                };
                const auto bs = GQL::compute_redundant_bitset<TN>(recs.data(), 2);
                if (bs.count() != 0) {
                    std::cerr << "\nFAIL LEAFCOMP[B]: expected count()==0, got "
                              << bs.count() << std::endl;
                    return 1;
                }
                if (!write_and_check(recs, /*expect_min_count=*/0, "B")) return 1;
            }

            // Case C: single-record page (degenerate fully-redundant page —
            // every byte trivially constant ⇒ all bits set, records section
            // empty). The reader must reconstruct from redundant_bytes only.
            {
                std::vector<Record<TN>> recs = {
                    Record<TN>{0xdeadbeefULL, 0xcafef00dULL, 0x12345678ULL},
                };
                const auto bs = GQL::compute_redundant_bitset<TN>(recs.data(), 1);
                if (bs.count() != REC) {
                    std::cerr << "\nFAIL LEAFCOMP[C]: 1-record page expected "
                              << REC << " set bits, got " << bs.count() << std::endl;
                    return 1;
                }
                if (!write_and_check(recs, /*expect_min_count=*/REC, "C")) return 1;
            }
        }
        std::cout << " OK" << std::endl;

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
