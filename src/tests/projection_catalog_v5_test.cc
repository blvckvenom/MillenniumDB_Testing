// Unit tests for ProjectionCatalog v1.5 — per-index leaf_format byte array
// (catalog extension that persists the delta + LEB128-varint leaf encoding
// choice per B+Tree index, introduced alongside the DELTA_VARINT leaf format).
// Validates:
//
//   1. CatalogV5_Roundtrip_AllBitset          — all entries = 1
//   2. CatalogV5_Roundtrip_AllDeltaVarint     — all entries = 2
//   3. CatalogV5_Roundtrip_MixedPerIndex      — mixed 1/2 pattern
//   4. CatalogV4_ReadAsV5_DefaultsBitset      — old catalog defaults to all-BITSET
//   5. CatalogV5_TruncatedFormatBytes_Rejected— file claims N, has fewer
//   6. CatalogV5_InvalidFormatByte_Rejected   — byte outside {1,2} rejected
//
// The "craft a v1.4 file by hand" strategy writes the exact v1.4 byte
// layout that the project's save_to_file emitted before this task. This
// avoids temporarily reverting MINOR_VERSION in-tree just to regenerate
// fixtures. The offsets used for the corruption tests are derived from
// the v1.5 save_to_file code path (projection_catalog.cc) and documented
// inline at each modification site.
//
// Design reference: docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md
// Plan reference:   docs/superpowers/plans/2026-04-25-delta-varint-leaf-plan.md

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
                    / ("mdb_catalog_v5_test_" + std::string(test_name)
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
// so save/load exercises the v1.0..v1.4 sections too, not just the new
// v1.5 append. Returns nothing; mutates the catalog in-place.
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
// Test 1: Roundtrip — all BITSET
// ============================================================================
TEST(CatalogV5, Roundtrip_AllBitset) {
    const auto dir = make_tempdir("Roundtrip_AllBitset");

    const auto n = count_materialized(GQL::IndexSet::GNN_MINIMAL);
    const std::vector<uint8_t> expected_formats(
        n, static_cast<uint8_t>(BPT::LeafFormat::BITSET));

    {
        GQL::ProjectionCatalog writer(dir);
        populate_baseline(writer);
        writer.index_set = GQL::IndexSet::GNN_MINIMAL;
        writer.leaf_formats = expected_formats;
        writer.save();
    }

    GQL::ProjectionCatalog reader(dir);
    // Loaded by ctor — just read back.
    EXPECT_EQ(reader.projection_name, "test_proj");
    EXPECT_EQ(reader.index_set, GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(reader.leaf_formats.size(), n);
    EXPECT_EQ(reader.leaf_formats, expected_formats);
    for (uint8_t b : reader.leaf_formats) {
        EXPECT_EQ(b, static_cast<uint8_t>(BPT::LeafFormat::BITSET));
    }

    fs::remove_all(dir);
}

// ============================================================================
// Test 2: Roundtrip — all DELTA_VARINT
// ============================================================================
TEST(CatalogV5, Roundtrip_AllDeltaVarint) {
    const auto dir = make_tempdir("Roundtrip_AllDeltaVarint");

    const auto n = count_materialized(GQL::IndexSet::GNN_MINIMAL);
    const std::vector<uint8_t> expected_formats(
        n, static_cast<uint8_t>(BPT::LeafFormat::DELTA_VARINT));

    {
        GQL::ProjectionCatalog writer(dir);
        populate_baseline(writer);
        writer.index_set = GQL::IndexSet::GNN_MINIMAL;
        writer.leaf_formats = expected_formats;
        writer.save();
    }

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.index_set, GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(reader.leaf_formats.size(), n);
    EXPECT_EQ(reader.leaf_formats, expected_formats);
    for (uint8_t b : reader.leaf_formats) {
        EXPECT_EQ(b, static_cast<uint8_t>(BPT::LeafFormat::DELTA_VARINT));
    }

    fs::remove_all(dir);
}

// ============================================================================
// Test 3: Roundtrip — mixed per-index pattern
// ============================================================================
TEST(CatalogV5, Roundtrip_MixedPerIndex) {
    const auto dir = make_tempdir("Roundtrip_MixedPerIndex");

    // Use IndexSet::ALL so we exercise the full 14-entry array (ALL_NODE|ALL_EDGE
    // = 0x3FFF, popcount 14). Mixed pattern alternates 1, 2, 1, 2, ...
    const auto n = count_materialized(GQL::IndexSet::ALL);
    ASSERT_EQ(n, 14u);
    std::vector<uint8_t> expected_formats(n);
    for (size_t i = 0; i < n; ++i) {
        expected_formats[i] = (i % 2 == 0)
            ? static_cast<uint8_t>(BPT::LeafFormat::BITSET)
            : static_cast<uint8_t>(BPT::LeafFormat::DELTA_VARINT);
    }

    {
        GQL::ProjectionCatalog writer(dir);
        populate_baseline(writer);
        writer.index_set = GQL::IndexSet::ALL;
        writer.leaf_formats = expected_formats;
        writer.save();
    }

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.index_set, GQL::IndexSet::ALL);
    EXPECT_EQ(reader.leaf_formats, expected_formats);

    fs::remove_all(dir);
}

// ============================================================================
// Helper: write a v1.4 catalog file by hand, matching the on-disk layout
// that projection_catalog.cc's save() emitted at MINOR_VERSION=4 (i.e.,
// identical to the current save path but with the v1.5 section omitted
// and the minor-version byte forced to 4).
//
// This function is the source of truth for the pre-v1.5 disk format used
// by Test 4. It deliberately constructs the bytes manually rather than
// calling the real save() path to avoid coupling the fixture to future
// code drift.
// ============================================================================
void write_v14_catalog_file(const std::string& path,
                            const std::string& proj_name,
                            GQL::IndexSet index_set_byte) {
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
    // Catalog version: MAJOR=1, MINOR=4
    put_u8(1); put_u8(4);

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

    // NOTE: v1.5 section INTENTIONALLY OMITTED.
    out.flush();
    out.close();
}

// ============================================================================
// Test 4: Pre-v1.5 catalog read defaults every leaf_format to BITSET (1)
// and the total count equals the materialized-index count.
// ============================================================================
TEST(CatalogV4, ReadAsV5_DefaultsBitset) {
    const auto dir = make_tempdir("ReadAsV5_DefaultsBitset");
    const auto catalog_path = dir + "/catalog.dat";
    write_v14_catalog_file(catalog_path, "old_proj", GQL::IndexSet::GNN_MINIMAL);

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.projection_name, "old_proj");
    EXPECT_EQ(reader.index_set, GQL::IndexSet::GNN_MINIMAL);

    const auto expected_n = count_materialized(GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(reader.leaf_formats.size(), expected_n);
    for (uint8_t b : reader.leaf_formats) {
        EXPECT_EQ(b, static_cast<uint8_t>(BPT::LeafFormat::BITSET));
    }

    fs::remove_all(dir);
}

// ============================================================================
// Test 5: Truncated format-bytes array rejected
// ============================================================================
// Strategy:
//   1. Save a valid v1.5 catalog (IndexSet::GNN_MINIMAL, 5 entries).
//   2. Locate the length byte appended by v1.5 save (it is the LAST byte
//      written before the 5-byte format array). The byte itself is at
//      offset (filesize - 1 - 5) = filesize - 6.
//   3. Truncate the file by cutting off the last 2 format bytes (losing
//      the trailing 2 of 5) WITHOUT adjusting the length byte. The length
//      byte still says 5, but only 3 format bytes remain on disk.
//   4. Load — expected to throw.
TEST(CatalogV5, TruncatedFormatBytes_Rejected) {
    const auto dir = make_tempdir("TruncatedFormatBytes_Rejected");
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
    // Truncate last 2 bytes (two of the 5 format bytes) so length prefix
    // lies to the reader.
    fs::resize_file(catalog_path, original_size - 2);

    EXPECT_THROW({
        GQL::ProjectionCatalog reader(dir);
        (void)reader;
    }, std::runtime_error);

    fs::remove_all(dir);
}

// ============================================================================
// Test 6: Invalid format byte (value outside {1, 2}) rejected
// ============================================================================
// Strategy:
//   1. Save a valid v1.5 catalog (IndexSet::GNN_MINIMAL, 5 entries).
//   2. The 5 format bytes are the final 5 bytes of the file. Overwrite the
//      FIRST of those (offset = filesize - 5) with 99 (invalid).
//   3. Load — expected to throw.
TEST(CatalogV5, InvalidFormatByte_Rejected) {
    const auto dir = make_tempdir("InvalidFormatByte_Rejected");
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
    const auto filesize = fs::file_size(catalog_path);
    // Offset of the first leaf_format byte = filesize - 5 (5 format bytes
    // are the trailing payload).
    std::fstream f(catalog_path,
                   std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f.seekp(static_cast<std::streamoff>(filesize - n));
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
// Helper: write a minimal v1.0 catalog whose FINAL field
// (edge_property_names) ends in a string with a declared length larger
// than the bytes actually present. At MINOR=0 nothing is read after this
// field, so a reader that doesn't validate gcount() returns garbage
// SILENTLY (uninitialized heap bytes) instead of throwing.
// ============================================================================
void write_v10_catalog_with_truncated_tail_string(const std::string& path) {
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

    // Magic number (6 bytes)
    const uint8_t magic[6] = {0x10, 0x0D, 0xEC, 0xAD, 0xE5, 0xDB};
    out.write(reinterpret_cast<const char*>(magic), 6);
    // MDB version (3 bytes)
    put_u8(1); put_u8(0); put_u8(0);
    // Model ID
    put_u8(255);
    // Catalog version: MAJOR=1, MINOR=0
    put_u8(1); put_u8(0);

    // v1.0 body
    put_str("trunc_proj");
    put_u64_le(1700000000);    // creation_timestamp
    put_u64_le(1);             // node_count
    put_u64_le(1);             // edge_count
    put_u64_le(1);             // directed_edge_count
    put_u64_le(0);             // undirected_edge_count
    put_u8(0);                 // has_node_properties
    put_u8(0);                 // has_edge_properties
    put_u8(0);                 // undirected_relationships
    put_str("");               // original_query
    put_u64_le(0);             // projection_millis
    put_u32_le(0);             // node_property_names: empty strvec

    // edge_property_names: declared count 1, declared string length 64,
    // but only 4 payload bytes follow before EOF.
    put_u32_le(1);
    put_u32_le(64);
    out.write("abcd", 4);
    out.flush();
    out.close();
}

// ============================================================================
// Truncated tail string must be rejected, not returned as silent garbage.
// ============================================================================
TEST(CatalogIO, TruncatedTailString_Rejected) {
    const auto dir = make_tempdir("TruncatedTailString_Rejected");
    write_v10_catalog_with_truncated_tail_string(dir + "/catalog.dat");

    EXPECT_THROW({
        GQL::ProjectionCatalog reader(dir);
        (void)reader;
    }, std::runtime_error);

    fs::remove_all(dir);
}

// ============================================================================
// A corrupt string-length prefix claiming ~4 GB must be rejected before
// any allocation is attempted.
// ============================================================================
TEST(CatalogIO, OversizedStringLength_Rejected) {
    const auto dir = make_tempdir("OversizedStringLength_Rejected");
    const auto catalog_path = dir + "/catalog.dat";
    {
        std::ofstream out(catalog_path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        const uint8_t magic[6] = {0x10, 0x0D, 0xEC, 0xAD, 0xE5, 0xDB};
        out.write(reinterpret_cast<const char*>(magic), 6);
        // MDB version (3), model id, catalog MAJOR=1 / MINOR=6
        const uint8_t header[6] = {1, 0, 0, 255, 1, 6};
        out.write(reinterpret_cast<const char*>(header), 6);
        // projection_name length prefix claims ~4 GB with no payload.
        const uint8_t len[4] = {0xF0, 0xFF, 0xFF, 0xFF};
        out.write(reinterpret_cast<const char*>(len), 4);
    }

    EXPECT_THROW({
        GQL::ProjectionCatalog reader(dir);
        (void)reader;
    }, std::runtime_error);

    fs::remove_all(dir);
}

// ============================================================================
// save() writes via tmp + rename: no .tmp residue, and an overwrite of an
// existing catalog roundtrips the new content.
// ============================================================================
TEST(CatalogIO, SaveLeavesNoTmpResidueAndRoundtrips) {
    const auto dir = make_tempdir("SaveLeavesNoTmpResidue");
    {
        GQL::ProjectionCatalog writer(dir);
        populate_baseline(writer);
        writer.save();
        // Overwrite in place — exercises the rename-over-existing path.
        writer.projection_name = "test_proj_v2";
        writer.save();
    }
    EXPECT_FALSE(fs::exists(dir + "/catalog.dat.tmp"));
    ASSERT_TRUE(fs::exists(dir + "/catalog.dat"));

    GQL::ProjectionCatalog reader(dir);
    EXPECT_EQ(reader.projection_name, "test_proj_v2");

    fs::remove_all(dir);
}

}  // namespace
