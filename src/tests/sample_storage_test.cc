// Unit tests for SampleCatalog version compatibility and SampleStorage
// on-disk robustness:
//   - v1/v2/v3 catalog version gate (v2-era catalogs must load with
//     sample_content_fp == 0; unknown versions must be rejected).
//   - frequency.dat corruption handling (truncated payload and corrupt
//     header must degrade to "no frequency data", never partial/garbage
//     counts or unbounded allocations).
//   - batches.idx corruption handling (corrupt entry count must fail loud
//     with a runtime_error, not an unbounded allocation).
//   - create()/exists() predicate consistency (a stale sample directory
//     without catalog.dat must be reclaimable by create()).

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_config.h"
#include "graph_models/object_id.h"

namespace fs = std::filesystem;
using namespace mdb::gnn;

namespace {

// -----------------------------------------------------------------------------
// Shared fixture: per-test temp directory + tiny sample builders
// -----------------------------------------------------------------------------

class SampleStorageTest : public ::testing::Test {
protected:
    fs::path tmp_dir_;

    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        tmp_dir_ = fs::temp_directory_path()
                 / ("sample_storage_test_" + std::to_string(::getpid()) + "_"
                    + std::string(info->name()));
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_dir_, ec);
    }

    static SamplingConfig make_config(const std::string& sample_name) {
        SamplingConfig config;
        config.projection_name = "test_proj";
        config.sample_name = sample_name;
        config.fanouts = { 2 };
        config.batch_size = 2;
        return config;
    }

    // A minimal but structurally valid 1-layer sample: 2 seeds, 1 neighbor,
    // 1 edge connecting neighbor->seed.
    static GraphSample make_sample(uint64_t batch_id, SplitType split) {
        GraphSample sample;
        sample.batch_id = batch_id;
        sample.split = split;
        sample.nodes_per_layer = {
            { ObjectId(1), ObjectId(2) },  // layer 0 (seeds)
            { ObjectId(3) },               // layer 1
        };
        LayerEdges edges;
        edges.src_indices = { 0 };
        edges.dst_indices = { 0 };
        edges.edge_ids = { ObjectId(100) };
        sample.edges_per_layer = { edges };
        sample.all_unique_nodes = { ObjectId(1), ObjectId(2), ObjectId(3) };
        return sample;
    }

    // Builds a finalized on-disk sample set and returns its storage path.
    fs::path build_finalized_sample(const std::string& sample_name) {
        auto config = make_config(sample_name);
        auto storage = SampleStorage::create(tmp_dir_, config);
        storage.write_sample(make_sample(0, SplitType::TRAIN));
        storage.write_sample(make_sample(1, SplitType::VALIDATION));
        storage.finalize();
        return SampleStorage::get_storage_path(tmp_dir_, sample_name);
    }

    static std::vector<char> read_file(const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(in),
                                 std::istreambuf_iterator<char>());
    }

    static void write_file(const fs::path& p, const std::vector<char>& bytes) {
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    // Overwrite the uint32 version field at byte offset 4 of catalog.dat.
    static void patch_catalog_version(const fs::path& dir, uint32_t version) {
        auto path = dir / SampleCatalog::CATALOG_FILENAME;
        auto bytes = read_file(path);
        ASSERT_GE(bytes.size(), 8u);
        std::memcpy(bytes.data() + 4, &version, sizeof(version));
        write_file(path, bytes);
    }

    static void truncate_file(const fs::path& p, uint64_t new_size) {
        auto bytes = read_file(p);
        ASSERT_GE(bytes.size(), new_size);
        bytes.resize(new_size);
        write_file(p, bytes);
    }
};

// -----------------------------------------------------------------------------
// SampleCatalog version gate
// -----------------------------------------------------------------------------

TEST_F(SampleStorageTest, CatalogV3RoundTripPreservesFingerprint) {
    SampleCatalog catalog(make_config("v3_roundtrip"));
    catalog.sample_content_fp = 0xDEADBEEFULL;
    catalog.save(tmp_dir_);

    auto loaded = SampleCatalog::load(tmp_dir_);
    EXPECT_EQ(loaded.sample_content_fp, 0xDEADBEEFULL);
    EXPECT_EQ(loaded.sample_name, "v3_roundtrip");
    EXPECT_EQ(loaded.projection_name, "test_proj");
}

TEST_F(SampleStorageTest, CatalogV2LoadsWithUnknownFingerprint) {
    SampleCatalog catalog(make_config("v2_compat"));
    catalog.sample_content_fp = 0xDEADBEEFULL;
    catalog.save(tmp_dir_);

    // Rewrite as a v2-era file: version field = 2, no trailing 8-byte
    // sample_content_fp.
    patch_catalog_version(tmp_dir_, 2);
    auto path = tmp_dir_ / SampleCatalog::CATALOG_FILENAME;
    auto bytes = read_file(path);
    ASSERT_GT(bytes.size(), 8u);
    bytes.resize(bytes.size() - sizeof(uint64_t));
    write_file(path, bytes);

    SampleCatalog loaded;
    ASSERT_NO_THROW(loaded = SampleCatalog::load(tmp_dir_));
    EXPECT_EQ(loaded.sample_content_fp, 0u);  // UNKNOWN -> recompute
    EXPECT_EQ(loaded.sample_name, "v2_compat");
    EXPECT_EQ(loaded.projection_name, "test_proj");
    EXPECT_EQ(loaded.batch_size, 2u);
}

TEST_F(SampleStorageTest, CatalogFutureVersionRejected) {
    SampleCatalog catalog(make_config("v4_reject"));
    catalog.save(tmp_dir_);
    patch_catalog_version(tmp_dir_, SampleCatalog::VERSION + 1);
    EXPECT_THROW(SampleCatalog::load(tmp_dir_), std::runtime_error);
}

TEST_F(SampleStorageTest, CatalogVersionZeroRejected) {
    SampleCatalog catalog(make_config("v0_reject"));
    catalog.save(tmp_dir_);
    patch_catalog_version(tmp_dir_, 0);
    EXPECT_THROW(SampleCatalog::load(tmp_dir_), std::runtime_error);
}

// -----------------------------------------------------------------------------
// frequency.dat robustness
// -----------------------------------------------------------------------------

TEST_F(SampleStorageTest, FrequenciesRoundTrip) {
    auto path = build_finalized_sample("freq_roundtrip");

    auto storage = SampleStorage::open(path);
    auto freqs = storage.get_node_frequencies();
    ASSERT_EQ(freqs.size(), 3u);
    EXPECT_EQ(freqs[1], 2u);  // node 1 appears in both batches
    EXPECT_EQ(freqs[2], 2u);
    EXPECT_EQ(freqs[3], 2u);
}

TEST_F(SampleStorageTest, TruncatedFrequencyPayloadDegradesToEmpty) {
    auto path = build_finalized_sample("freq_truncated");

    // v1 sparse layout: 16-byte header (magic + version + num_entries) then
    // 16 bytes per (oid, count) pair. Keep the header (which claims 3
    // entries) plus half a pair.
    truncate_file(path / "frequency.dat", 16 + 8);

    auto storage = SampleStorage::open(path);
    std::unordered_map<uint64_t, uint64_t> freqs;
    ASSERT_NO_THROW(freqs = storage.get_node_frequencies());
    EXPECT_TRUE(freqs.empty());  // degraded, not partial/garbage counts
}

TEST_F(SampleStorageTest, CorruptFrequencyHeaderDoesNotAllocate) {
    auto path = build_finalized_sample("freq_corrupt_header");

    // Forge a header claiming 2^60 dense entries (8 EiB payload).
    {
        std::ofstream out(path / "frequency.dat", std::ios::binary | std::ios::trunc);
        uint32_t magic = 0x51455246;  // "FREQ"
        uint32_t version = 2;
        uint64_t num_entries = uint64_t(1) << 60;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));
    }

    auto storage = SampleStorage::open(path);
    std::unordered_map<uint64_t, uint64_t> freqs;
    ASSERT_NO_THROW(freqs = storage.get_node_frequencies());
    EXPECT_TRUE(freqs.empty());
}

// -----------------------------------------------------------------------------
// batches.idx robustness
// -----------------------------------------------------------------------------

TEST_F(SampleStorageTest, CorruptIndexHeaderFailsLoud) {
    auto path = build_finalized_sample("idx_corrupt_header");

    // Patch the uint64 entry count at offset 8 (after magic + version) to a
    // value far beyond what the file holds.
    auto idx_path = path / "batches.idx";
    auto bytes = read_file(idx_path);
    ASSERT_GE(bytes.size(), 16u);
    uint64_t huge = uint64_t(1) << 60;
    std::memcpy(bytes.data() + 8, &huge, sizeof(huge));
    write_file(idx_path, bytes);

    // Must be a runtime_error (clear diagnostics), not bad_alloc from an
    // attempted 2^60-entry resize.
    EXPECT_THROW(SampleStorage::open(path), std::runtime_error);
}

// -----------------------------------------------------------------------------
// create()/exists() predicate consistency
// -----------------------------------------------------------------------------

TEST_F(SampleStorageTest, StaleDirectoryWithoutCatalogIsReclaimed) {
    auto config = make_config("stale_dir");
    auto path = SampleStorage::get_storage_path(tmp_dir_, config.sample_name);

    // Simulate a hard crash mid-run: directory + partial data, no catalog.
    fs::create_directories(path);
    {
        std::ofstream out(path / "batches.dat", std::ios::binary);
        out << "partial";
    }

    ASSERT_FALSE(SampleStorage::exists(tmp_dir_, config.sample_name));

    // create() must agree with exists(): reclaim the stale leftover instead
    // of throwing "already exists" that no force-flag can clear.
    {
        auto storage = SampleStorage::create(tmp_dir_, config);
        storage.write_sample(make_sample(0, SplitType::TRAIN));
        storage.finalize();
    }
    EXPECT_TRUE(SampleStorage::exists(tmp_dir_, config.sample_name));
}

TEST_F(SampleStorageTest, FinalizedSampleRejectsSecondCreate) {
    build_finalized_sample("no_clobber");
    auto config = make_config("no_clobber");
    EXPECT_THROW(SampleStorage::create(tmp_dir_, config), std::runtime_error);
}

}  // namespace
