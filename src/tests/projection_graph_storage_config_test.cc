// Unit tests for the Spec #8 T8.8 end-to-end plumbing of the
// `graphStorage` GQL config key. Covers:
//
//   1. GraphStorageConfig_Absent_DefaultsBTree      — default builder ctor
//      path + catalog roundtrip leave graph_storage == 1 (BTREE), matching
//      pre-Spec-#8 byte-for-byte behavior.
//   2. GraphStorageConfig_CSRHybrid_Accepted        — parse_graph_storage
//      returns CSR_HYBRID; persisting with ProjectionCatalog round-trips
//      the v1.6 byte as 2.
//   3. GraphStorageConfig_Unknown_Rejected          — BPT::parse_graph_storage
//      raises std::invalid_argument with the documented "Unknown
//      graphStorage" substring (project_procedure.cc wraps it in
//      QueryException — this test pins the low-level parser invariant).
//   4. GraphStorageConfig_NonString_Rejected        — non-string values are
//      owned by get_string_from_dict() in project_procedure.cc; here we
//      document the invariant at the parser level with an empty string
//      (the closest stand-in reachable without a live DictionaryObject).
//   5. GraphStorageConfig_Roundtrip_ViaStorage      — populate a catalog
//      with CSR_HYBRID, save, reload through ProjectionCatalog, assert the
//      byte round-trips. Mirrors the storage-level accessor that T8.9 will
//      consume (ProjectionStorage::requested_graph_storage).
//
// As with projection_leaffmt_config_test.cc, reaching a live builder + full
// System + QueryContext boot from a unit test is overkill for the
// invariants this task needs to prove. The catalog-roundtrip tests below
// pin the on-disk byte contract that save_catalog() writes and the end-to-
// end smoke test (config → builder → catalog byte) is covered by the
// project_procedure.cc instrumentation plus the integration suite.
//
// Spec reference: docs/superpowers/specs/2026-04-25-csr-hybrid-design.md §3.7
// Plan reference: docs/superpowers/plans/2026-04-25-csr-hybrid-plan.md §T8.8

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/projection_catalog.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"

namespace fs = std::filesystem;

namespace {

// Create a unique temporary directory for an isolated catalog.dat file.
// Uses the test name + pid so parallel test runs don't collide, identical
// discipline to projection_leaffmt_config_test.cc / projection_catalog_v6_test.cc.
std::string make_tempdir(const char* test_name) {
    const auto base = fs::temp_directory_path()
                    / ("mdb_graphstorage_config_test_" + std::string(test_name)
                       + "_" + std::to_string(::getpid()));
    fs::remove_all(base);
    fs::create_directories(base);
    return base.string();
}

// Count single-bit entries in the ProjectionIndex mask for a preset.
// Mirrors the helper in projection_leaffmt_config_test.cc — the function
// is static inside an anonymous namespace in each TU so duplication is
// the right call here.
size_t count_materialized(GQL::IndexSet preset) {
    const auto mask =
        static_cast<uint32_t>(GQL::project_index_mask_for(preset));
    return static_cast<size_t>(__builtin_popcount(mask));
}

// Populate a ProjectionCatalog with the minimum fields needed for save/
// load to exercise every pre-v1.6 section. Same shape as the helper used
// by projection_leaffmt_config_test.cc.
void populate_minimum(GQL::ProjectionCatalog& cat,
                      GQL::IndexSet preset,
                      BPT::LeafFormat leaf_format) {
    cat.projection_name = "graphstorage_test";
    cat.creation_timestamp = 1700000000;
    cat.node_count = 10;
    cat.edge_count = 20;
    cat.directed_edge_count = 20;
    cat.undirected_edge_count = 0;
    cat.includes_node_labels = true;
    cat.includes_edge_labels = true;
    cat.includes_node_properties = false;
    cat.includes_edge_properties = false;
    cat.distinct_node_labels = 1;
    cat.distinct_edge_labels = 1;
    cat.original_query = "CALL graph_project(...)";
    cat.projection_millis = 1;
    cat.add_node_key("_count", 999);
    cat.index_set = preset;
    const auto n = count_materialized(preset);
    cat.leaf_formats.assign(n, static_cast<uint8_t>(leaf_format));
}

}  // namespace

// ============================================================================
// Test 1: Absent `graphStorage` key leaves the projection at the BTREE
// default. Pins the on-disk invariant that pre-Spec-#8 builds continue to
// produce: when the GQL caller does not set the key, the catalog records
// BTREE (byte 1) at the v1.6 graph_storage slot.
// ============================================================================
TEST(GraphStorageConfig, Absent_DefaultsBTree) {
    const auto dir = make_tempdir("Absent_DefaultsBTree");

    {
        GQL::ProjectionCatalog writer(dir);
        populate_minimum(writer, GQL::IndexSet::ALL, BPT::LeafFormat::BITSET);
        // graph_storage intentionally left at the class-level default
        // (1 = BTREE). This mirrors what the builder does when GQL omits
        // the `graphStorage` key.
        ASSERT_EQ(writer.graph_storage, 1u);
        ASSERT_EQ(writer.get_graph_storage(), 1u);
        writer.save();
    }

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.graph_storage, 1u);
    EXPECT_EQ(reader.get_graph_storage(), 1u);

    fs::remove_all(dir);
}

// ============================================================================
// Test 2: `graphStorage: 'CSR_HYBRID'` is accepted by the parser and
// threaded all the way to the on-disk v1.6 catalog byte.
// ============================================================================
TEST(GraphStorageConfig, CSRHybrid_Accepted) {
    // Parser layer — mirrors what project_procedure.cc's T8.8 block does.
    const auto parsed = BPT::parse_graph_storage("CSR_HYBRID");
    ASSERT_EQ(parsed, BPT::GraphStorage::CSR_HYBRID);

    // BTREE is also accepted by the parser (default value path).
    EXPECT_EQ(BPT::parse_graph_storage("BTREE"), BPT::GraphStorage::BTREE);

    const auto dir = make_tempdir("CSRHybrid_Accepted");

    {
        GQL::ProjectionCatalog writer(dir);
        populate_minimum(writer, GQL::IndexSet::ALL, BPT::LeafFormat::BITSET);
        writer.graph_storage = static_cast<uint8_t>(parsed);
        ASSERT_EQ(writer.graph_storage, 2u);
        writer.save();
    }

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.graph_storage, 2u);
    EXPECT_EQ(reader.get_graph_storage(), 2u);

    fs::remove_all(dir);
}

// ============================================================================
// Test 3: Unknown `graphStorage` value is rejected by BPT::parse_graph_storage
// with std::invalid_argument. project_procedure.cc wraps this in
// QueryException before it surfaces to the GQL caller. The documented
// "Unknown graphStorage" message substring is asserted so future edits
// don't drop the key name from the error and leave users guessing.
// ============================================================================
TEST(GraphStorageConfig, Unknown_Rejected) {
    EXPECT_THROW({ (void) BPT::parse_graph_storage("ROARING"); },
                 std::invalid_argument);
    // Case-sensitive by design — mirrors parse_leaf_format.
    EXPECT_THROW({ (void) BPT::parse_graph_storage("btree"); },
                 std::invalid_argument);
    EXPECT_THROW({ (void) BPT::parse_graph_storage(""); },
                 std::invalid_argument);

    try {
        (void) BPT::parse_graph_storage("ROARING");
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("Unknown graphStorage"), std::string::npos)
            << "Error message must name the key it parses "
               "('Unknown graphStorage ...'); actual message: " << msg;
    }
}

// ============================================================================
// Test 4: Non-string `graphStorage` rejection is owned by the dict helper
// (get_string_from_dict in project_procedure.cc) — any non-string config
// value throws runtime_error with the type id. Since constructing a live
// DictionaryObject here would re-implement half of the GQL runtime, this
// test documents the invariant at the parse_graph_storage layer by
// asserting the parser does not silently accept an empty string (which
// is what the dict helper would surface if someone disabled the type
// check). Real non-string rejection is exercised by the GQL integration
// suite.
// ============================================================================
TEST(GraphStorageConfig, NonString_Rejected) {
    EXPECT_THROW({ (void) BPT::parse_graph_storage(""); },
                 std::invalid_argument);
}

// ============================================================================
// Test 5: Structural invariant — a catalog saved with CSR_HYBRID and then
// reopened reports the same byte through every accessor. This guards the
// contract that T8.9 relies on when it reads ProjectionStorage::
// requested_graph_storage to pick the edge-index writer. The invariant
// holds across IndexSet presets because the v1.6 graph_storage byte is a
// single per-projection slot (not per-index), unlike leaf_formats.
// ============================================================================
TEST(GraphStorageConfig, Roundtrip_ViaStorage) {
    // Parser contract — exactly two valid values.
    EXPECT_EQ(BPT::parse_graph_storage("BTREE"),
              BPT::GraphStorage::BTREE);
    EXPECT_EQ(BPT::parse_graph_storage("CSR_HYBRID"),
              BPT::GraphStorage::CSR_HYBRID);

    // Catalog contract: for each preset we iterate over, the graph_storage
    // byte round-trips independently of leaf_formats and index_set. This
    // guards against someone accidentally coupling the fields in save().
    const GQL::IndexSet presets[] = {
        GQL::IndexSet::ALL,
        GQL::IndexSet::GNN_MINIMAL,
        GQL::IndexSet::READONLY_TRAVERSAL,
    };
    for (const auto preset : presets) {
        const auto dir = make_tempdir(
            ("Roundtrip_ViaStorage_" +
             std::string(GQL::index_set_name(preset))).c_str());

        {
            GQL::ProjectionCatalog writer(dir);
            populate_minimum(writer, preset, BPT::LeafFormat::BITSET);
            writer.graph_storage =
                static_cast<uint8_t>(BPT::GraphStorage::CSR_HYBRID);
            writer.save();
        }

        GQL::ProjectionCatalog reader(dir);
        EXPECT_EQ(reader.graph_storage,
                  static_cast<uint8_t>(BPT::GraphStorage::CSR_HYBRID))
            << "catalog must round-trip CSR_HYBRID for "
            << GQL::index_set_name(preset);
        EXPECT_EQ(reader.index_set, preset)
            << "graph_storage round-trip must not disturb index_set for "
            << GQL::index_set_name(preset);

        fs::remove_all(dir);
    }
}

// ============================================================================
// Bonus: symmetry check on graph_storage_to_string — useful for debug
// logging so we pin the pair at the same time as the parser. Mirrors the
// leaf_format_to_string symmetry implicit in parse_leaf_format tests.
// ============================================================================
TEST(GraphStorageConfig, EnumStringSymmetry) {
    EXPECT_STREQ(BPT::graph_storage_to_string(BPT::GraphStorage::BTREE),
                 "BTREE");
    EXPECT_STREQ(BPT::graph_storage_to_string(BPT::GraphStorage::CSR_HYBRID),
                 "CSR_HYBRID");

    // Round-trip: parse(to_string(x)) == x
    for (auto s : { BPT::GraphStorage::BTREE, BPT::GraphStorage::CSR_HYBRID }) {
        EXPECT_EQ(BPT::parse_graph_storage(BPT::graph_storage_to_string(s)), s);
    }
}
