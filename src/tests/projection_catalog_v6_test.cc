// Unit tests for ProjectionCatalog v1.6 — per-projection graphStorage byte
// (Spec #8 T8.7). Validates:
//
//   1. CatalogV6_Roundtrip_DefaultBTree      — default (1 = BTREE) round-trips
//   2. CatalogV6_Roundtrip_CSRHybrid         — explicit 2 = CSR_HYBRID round-trips
//   3. CatalogV5_ReadAsV6_DefaultsBTree      — pre-v1.6 catalog defaults to BTREE
//   4. CatalogV6_InvalidGraphStorageByte_Rejected — value outside {1,2} rejected
//   5. CatalogV6_TruncatedGraphStorage_Rejected   — minor=6 but file ends early
//
// The "craft a v1.5 file by hand" strategy (Test 3) writes the exact v1.5
// byte layout that projection_catalog.cc's save() emitted before this task,
// including the length-prefixed leaf_formats array. This avoids temporarily
// reverting MINOR_VERSION in-tree just to regenerate fixtures. The offsets
// for the corruption tests are derived from the v1.6 save() code path and
// documented inline at each modification site.
//
// Spec reference: docs/superpowers/specs/2026-04-25-csr-hybrid-design.md §3.7
// Plan reference: docs/superpowers/plans/2026-04-25-csr-hybrid-plan.md §T8.7

#include "graph_models/gql/projection/projection_catalog.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"

namespace fs = std::filesystem;

namespace {

// Create a unique temporary directory for an isolated catalog.dat file.
// Uses the test name so parallel test runs don't collide — googletest's
// UnitTest::current_test_info() gives us a stable identifier.
std::string make_tempdir(const char* test_name) {
    const auto base = fs::temp_directory_path()
                    / ("mdb_catalog_v6_test_" + std::string(test_name)
                       + "_" + std::to_string(::getpid()));
    fs::remove_all(base);
    fs::create_directories(base);
    return base.string();
}

// Count single-bit entries in the ProjectionIndex mask for a preset.
// Mirrors the private helper inside projection_catalog.cc so tests can
// assert the expected leaf_formats array length without exposing it.
size_t count_materialized(GQL::IndexSet preset) {
    const auto mask = static_cast<uint32_t>(GQL::project_index_mask_for(preset));
    return static_cast<size_t>(__builtin_popcount(mask));
}

// Populate a ProjectionCatalog with a small but non-trivial set of fields
// so save/load exercises the v1.0..v1.5 sections too, not just the new
// v1.6 append. Returns nothing; mutates the catalog in-place.
void populate_baseline(GQL::ProjectionCatalog& cat) {
    cat.projection_name = "test_proj";
    cat.creation_timestamp = 1700000000;
    cat.node_count = 100;
    cat.edge_count = 500;
    cat.directed_edge_count = 500;
    cat.undirected_edge_count = 0;
    cat.includes_node_labels = true;
    cat.includes_edge_labels = true;
    cat.includes_node_properties = false;
    cat.includes_edge_properties = false;
    cat.distinct_node_labels = 7;
    cat.distinct_edge_labels = 3;
    cat.original_query = "CALL graph_project(...)";
    cat.projection_millis = 42;
    // At least one node key so v1.3 section is non-empty
    cat.add_node_key("_count", 999);
}

// ============================================================================
// Test 1: Roundtrip — default graph_storage (1 = BTREE)
// ============================================================================
TEST(CatalogV6, Roundtrip_DefaultBTree) {
    const auto dir = make_tempdir("Roundtrip_DefaultBTree");

    const auto n = count_materialized(GQL::IndexSet::GNN_MINIMAL);
    const std::vector<uint8_t> formats(
        n, static_cast<uint8_t>(BPT::LeafFormat::BITSET));

    {
        GQL::ProjectionCatalog writer(dir);
        populate_baseline(writer);
        writer.index_set = GQL::IndexSet::GNN_MINIMAL;
        writer.leaf_formats = formats;
        // graph_storage left at its default (1 = BTREE) on purpose.
        ASSERT_EQ(writer.graph_storage, 1u);
        ASSERT_EQ(writer.get_graph_storage(), 1u);
        writer.save();
    }

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.projection_name, "test_proj");
    EXPECT_EQ(reader.index_set, GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(reader.leaf_formats, formats);
    EXPECT_EQ(reader.graph_storage, 1u);
    EXPECT_EQ(reader.get_graph_storage(), 1u);

    fs::remove_all(dir);
}

// ============================================================================
// Test 2: Roundtrip — explicit graph_storage = 2 (CSR_HYBRID)
// ============================================================================
TEST(CatalogV6, Roundtrip_CSRHybrid) {
    const auto dir = make_tempdir("Roundtrip_CSRHybrid");

    const auto n = count_materialized(GQL::IndexSet::GNN_MINIMAL);
    const std::vector<uint8_t> formats(
        n, static_cast<uint8_t>(BPT::LeafFormat::DELTA_VARINT));

    {
        GQL::ProjectionCatalog writer(dir);
        populate_baseline(writer);
        writer.index_set = GQL::IndexSet::GNN_MINIMAL;
        writer.leaf_formats = formats;
        writer.graph_storage = 2;  // CSR_HYBRID
        writer.save();
    }

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.index_set, GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(reader.leaf_formats, formats);
    EXPECT_EQ(reader.graph_storage, 2u);
    EXPECT_EQ(reader.get_graph_storage(), 2u);

    fs::remove_all(dir);
}

// ============================================================================
// Helper: write a v1.5 catalog file by hand, matching the on-disk layout
// that projection_catalog.cc's save() emitted at MINOR_VERSION=5 (i.e.,
// identical to the current save path but with the v1.6 graphStorage byte
// omitted and the minor-version byte forced to 5).
//
// This function is the source of truth for the pre-v1.6 disk format used
// by Test 3. It deliberately constructs the bytes manually rather than
// calling the real save() path to avoid coupling the fixture to future
// code drift.
// ============================================================================
void write_v15_catalog_file(const std::string& path,
                            const std::string& proj_name,
                            GQL::IndexSet index_set_byte,
                            const std::vector<uint8_t>& leaf_formats) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.good());

    auto put_u8 = [&](uint8_t v) { out.put(static_cast<char>(v)); };
    auto put_u32_le = [&](uint32_t v) {
        uint8_t b[4];
        for (size_t i = 0, s = 0; i < 4; ++i, s += 8) b[i] = (v >> s) & 0xFF;
        out.write(reinterpret_cast<const char*>(b), 4);
    };
    auto put_u64_le = [&](uint64_t v) {
        uint8_t b[8];
        for (size_t i = 0, s = 0; i < 8; ++i, s += 8) b[i] = (v >> s) & 0xFF;
        out.write(reinterpret_cast<const char*>(b), 8);
    };
    auto put_str = [&](const std::string& s) {
        put_u32_le(static_cast<uint32_t>(s.size()));
        out.write(s.data(), static_cast<std::streamsize>(s.size()));
    };
    auto put_strvec = [&](const std::vector<std::string>& v) {
        put_u32_le(static_cast<uint32_t>(v.size()));
        for (const auto& s : v) put_str(s);
    };

    // Magic number (6 bytes)
    const uint8_t magic[6] = {0x10, 0x0D, 0xEC, 0xAD, 0xE5, 0xDB};
    out.write(reinterpret_cast<const char*>(magic), 6);
    // MDB version (3 bytes)
    put_u8(1); put_u8(0); put_u8(0);
    // Model ID
    put_u8(255);
    // Catalog version: MAJOR=1, MINOR=5
    put_u8(1); put_u8(5);

    // v1.0 body
    put_str(proj_name);
    put_u64_le(1700000000);    // creation_timestamp
    put_u64_le(10);            // node_count
    put_u64_le(20);            // edge_count
    put_u64_le(20);            // directed_edge_count
    put_u64_le(0);             // undirected_edge_count
    put_u8(0);                 // has_node_properties
    put_u8(0);                 // has_edge_properties
    put_u8(0);                 // undirected_relationships
    put_str("");               // original_query
    put_u64_le(0);             // projection_millis
    put_strvec({});            // node_property_names
    put_strvec({});            // edge_property_names

    // v1.1 body
    put_u8(0);                 // includes_node_labels
    put_u8(0);                 // includes_edge_labels
    put_u8(0);                 // includes_node_properties
    put_u8(0);                 // includes_edge_properties
    put_strvec({});            // included_node_properties
    put_strvec({});            // included_edge_properties
    put_u64_le(0);             // distinct_node_labels
    put_u64_le(0);             // distinct_edge_labels

    // v1.3 body: (key_id, key_name) pairs — zero of each
    put_u32_le(0);             // node key count
    put_u32_le(0);             // edge key count

    // v1.4 body: IndexSet preset byte
    put_u8(static_cast<uint8_t>(index_set_byte));

    // v1.5 body: length-prefixed leaf_formats byte array
    ASSERT_LE(leaf_formats.size(), 255u);
    put_u8(static_cast<uint8_t>(leaf_formats.size()));
    for (uint8_t fmt : leaf_formats) {
        put_u8(fmt);
    }

    // NOTE: v1.6 graphStorage byte INTENTIONALLY OMITTED.
    out.flush();
    out.close();
}

// ============================================================================
// Test 3: Pre-v1.6 catalog read defaults graph_storage to 1 (BTREE) and
// all other fields load normally.
// ============================================================================
TEST(CatalogV5, ReadAsV6_DefaultsBTree) {
    const auto dir = make_tempdir("ReadAsV6_DefaultsBTree");
    const auto catalog_path = dir + "/catalog.dat";

    const auto n = count_materialized(GQL::IndexSet::GNN_MINIMAL);
    const std::vector<uint8_t> formats(
        n, static_cast<uint8_t>(BPT::LeafFormat::BITSET));

    write_v15_catalog_file(catalog_path,
                           "old_proj",
                           GQL::IndexSet::GNN_MINIMAL,
                           formats);

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.projection_name, "old_proj");
    EXPECT_EQ(reader.index_set, GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(reader.leaf_formats, formats);
    // The key invariant: pre-v1.6 catalogs read as BTREE (1).
    EXPECT_EQ(reader.graph_storage, 1u);
    EXPECT_EQ(reader.get_graph_storage(), 1u);

    fs::remove_all(dir);
}

// ============================================================================
// Test 4: Invalid graph_storage byte (value outside {1, 2}) rejected
// ============================================================================
// Strategy:
//   1. Save a valid v1.6 catalog (default graph_storage = 1).
//   2. The graph_storage byte is the FINAL byte of the file. Overwrite it
//      with 99 (invalid).
//   3. Load — expected to throw.
TEST(CatalogV6, InvalidGraphStorageByte_Rejected) {
    const auto dir = make_tempdir("InvalidGraphStorageByte_Rejected");
    const auto n = count_materialized(GQL::IndexSet::GNN_MINIMAL);
    ASSERT_EQ(n, 5u);

    {
        GQL::ProjectionCatalog writer(dir);
        populate_baseline(writer);
        writer.index_set = GQL::IndexSet::GNN_MINIMAL;
        writer.leaf_formats.assign(
            n, static_cast<uint8_t>(BPT::LeafFormat::BITSET));
        // graph_storage left at default (1 = BTREE) before save.
        writer.save();
    }

    const auto catalog_path = dir + "/catalog.dat";
    const auto filesize = fs::file_size(catalog_path);
    // graph_storage byte is the last byte of the file.
    std::fstream f(catalog_path,
                   std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f.seekp(static_cast<std::streamoff>(filesize - 1));
    f.put(static_cast<char>(99));
    f.flush();
    f.close();

    EXPECT_THROW({
        GQL::ProjectionCatalog reader(dir);
        (void)reader;
    }, std::runtime_error);

    fs::remove_all(dir);
}

// ============================================================================
// Test 5: Truncated graph_storage byte rejected
// ============================================================================
// Strategy:
//   1. Save a valid v1.6 catalog.
//   2. Truncate the file by one byte, removing the graph_storage byte.
//      The catalog header still says MINOR=6, so the reader will try to
//      consume a byte that no longer exists.
//   3. Load — the underlying read_uint8 helper raises on EOF, matching the
//      pattern used by the v1.5 truncated-format-bytes test.
TEST(CatalogV6, TruncatedGraphStorage_Rejected) {
    const auto dir = make_tempdir("TruncatedGraphStorage_Rejected");
    const auto n = count_materialized(GQL::IndexSet::GNN_MINIMAL);
    ASSERT_EQ(n, 5u);

    {
        GQL::ProjectionCatalog writer(dir);
        populate_baseline(writer);
        writer.index_set = GQL::IndexSet::GNN_MINIMAL;
        writer.leaf_formats.assign(
            n, static_cast<uint8_t>(BPT::LeafFormat::BITSET));
        writer.save();
    }

    const auto catalog_path = dir + "/catalog.dat";
    const auto original_size = fs::file_size(catalog_path);
    // Drop the trailing graph_storage byte (1 byte) so the on-disk layout
    // claims v1.6 but stops just before the new field.
    fs::resize_file(catalog_path, original_size - 1);

    EXPECT_THROW({
        GQL::ProjectionCatalog reader(dir);
        (void)reader;
    }, std::runtime_error);

    fs::remove_all(dir);
}

}  // namespace
