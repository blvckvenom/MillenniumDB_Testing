#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "gnn/storage/cache_file.h"
#include "gnn/storage/cpu_cache.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/gnn_dtype.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/object_id.h"
#include "gnn/tests/test_helpers.h"

namespace fs = std::filesystem;
using namespace mdb::gnn;

// =============================================================================
// Test Fixture
// =============================================================================

class FourLevelStoreTest : public GnnStorageTest {
protected:
    static constexpr uint64_t N = 8;
    static constexpr uint64_t D = 4;

    // Known feature values: row r, col c -> value = (r+1)*100 + (c+1)
    // Row 0: [101, 102, 103, 104]
    // Row 7: [801, 802, 803, 804]
    std::vector<ObjectId> node_oids_;
    std::vector<float> features_;
    std::unique_ptr<FeatureMatrix> fm_;
    std::unique_ptr<RowMapping> rm_;

    void SetUp() override {
        GnnStorageTest::SetUp();

        node_oids_.resize(N);
        for (uint64_t i = 0; i < N; ++i)
            node_oids_[i] = ObjectId(0xD400000000000000ULL | i);

        features_.resize(N * D);
        for (uint64_t r = 0; r < N; ++r)
            for (uint64_t c = 0; c < D; ++c)
                features_[r * D + c] = static_cast<float>((r + 1) * 100 + (c + 1));

        auto fmat_path = test_dir_ / "test.fmat";
        auto rmap_path = test_dir_ / "test.rmap";
        auto fm_tmp = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features_.data());
        fm_ = std::make_unique<FeatureMatrix>(std::move(fm_tmp));
        auto rm_tmp = RowMapping::create(rmap_path, node_oids_);
        rm_ = std::make_unique<RowMapping>(std::move(rm_tmp));
    }

    float expected_feature(uint64_t row, uint64_t dim) const {
        return static_cast<float>((row + 1) * 100 + (dim + 1));
    }
};

// =============================================================================
// CpuCache: Build and Load — verify contents round-trip
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_BuildAndLoad) {
    auto cache_path = test_dir_ / "cpu_cache.bin";
    std::vector<ObjectId> nodes = {node_oids_[0], node_oids_[1], node_oids_[2]};
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    CpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), 3u);
    EXPECT_EQ(cache.feature_dim(), D);
    EXPECT_TRUE(cache.contains(node_oids_[0]));
    EXPECT_TRUE(cache.contains(node_oids_[1]));
    EXPECT_TRUE(cache.contains(node_oids_[2]));
    EXPECT_FALSE(cache.contains(node_oids_[5]));
    EXPECT_FALSE(cache.contains(node_oids_[7]));

    // Lookup: 2 hits (node 0, node 2), 1 miss (node 5)
    auto result = cache.lookup({node_oids_[0], node_oids_[5], node_oids_[2]});
    ASSERT_EQ(result.hit_positions.size(), 2u);
    ASSERT_EQ(result.miss_positions.size(), 1u);
    EXPECT_EQ(result.hit_positions[0], 0u);   // node_oids_[0] at input position 0
    EXPECT_EQ(result.hit_positions[1], 2u);   // node_oids_[2] at input position 2
    EXPECT_EQ(result.miss_positions[0], 1u);  // node_oids_[5] at input position 1
    EXPECT_EQ(result.feature_dim, D);
    EXPECT_EQ(result.elem_size, sizeof(float));

    auto* data = reinterpret_cast<const float*>(result.features.data());
    // First hit is node 0 -> features [101, 102, 103, 104]
    EXPECT_FLOAT_EQ(data[0], 101.0f);
    EXPECT_FLOAT_EQ(data[1], 102.0f);
    EXPECT_FLOAT_EQ(data[2], 103.0f);
    EXPECT_FLOAT_EQ(data[3], 104.0f);
    // Second hit is node 2 -> features [301, 302, 303, 304]
    EXPECT_FLOAT_EQ(data[D + 0], 301.0f);
    EXPECT_FLOAT_EQ(data[D + 1], 302.0f);
    EXPECT_FLOAT_EQ(data[D + 2], 303.0f);
    EXPECT_FLOAT_EQ(data[D + 3], 304.0f);
}

// =============================================================================
// CacheFile: GNNC magic and header validation
// =============================================================================

TEST_F(FourLevelStoreTest, CacheFile_MagicAndHeader) {
    auto cache_path = test_dir_ / "cpu_header.bin";
    std::vector<ObjectId> nodes = {node_oids_[0]};
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    // Read raw header from file and validate
    std::ifstream f(cache_path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    CacheFileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    ASSERT_TRUE(f.good());

    EXPECT_EQ(hdr.magic, CacheFileHeader::MAGIC);
    EXPECT_EQ(hdr.version, 1u);
    EXPECT_EQ(hdr.num_nodes, 1u);
    EXPECT_EQ(hdr.feature_dim, D);
    EXPECT_EQ(hdr.get_dtype(), GnnDtype::FLOAT32);
    EXPECT_TRUE(hdr.is_valid());

    // Validate total file size
    auto file_size = fs::file_size(cache_path);
    size_t expected = CacheFileHeader::SIZE        // 32 bytes
                    + 1 * sizeof(uint64_t)         // 1 ObjectId
                    + 1 * D * sizeof(float);       // 1 row of features
    EXPECT_EQ(file_size, expected);
}

// =============================================================================
// CpuCache: Empty cache (0 nodes)
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_EmptyCache) {
    auto cache_path = test_dir_ / "cpu_empty.bin";
    std::vector<ObjectId> nodes = {};
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    CpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), 0u);
    EXPECT_EQ(cache.feature_dim(), D);
    EXPECT_EQ(cache.memory_bytes(), 0u);
    EXPECT_EQ(cache.data_ptr(), nullptr);

    // All lookups should miss
    auto result = cache.lookup({node_oids_[0], node_oids_[1]});
    EXPECT_EQ(result.hit_positions.size(), 0u);
    EXPECT_EQ(result.miss_positions.size(), 2u);
    EXPECT_EQ(result.miss_positions[0], 0u);
    EXPECT_EQ(result.miss_positions[1], 1u);
    EXPECT_TRUE(result.features.empty());
}

// =============================================================================
// CpuCache: All nodes cached — every lookup is a hit
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_AllNodesCached) {
    auto cache_path = test_dir_ / "cpu_full.bin";
    CpuCache::build(node_oids_, *fm_, *rm_, cache_path);

    CpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), N);
    EXPECT_EQ(cache.memory_bytes(), N * D * sizeof(float));
    EXPECT_NE(cache.data_ptr(), nullptr);

    // Lookup all nodes
    auto result = cache.lookup(node_oids_);
    EXPECT_EQ(result.hit_positions.size(), N);
    EXPECT_EQ(result.miss_positions.size(), 0u);

    // Verify all feature values
    auto* data = reinterpret_cast<const float*>(result.features.data());
    for (uint64_t r = 0; r < N; ++r) {
        for (uint64_t c = 0; c < D; ++c) {
            EXPECT_FLOAT_EQ(data[r * D + c], expected_feature(r, c))
                << "Node " << r << ", dim " << c;
        }
    }
}

// =============================================================================
// CpuCache: Single node — minimal non-empty cache
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_SingleNode) {
    auto cache_path = test_dir_ / "cpu_single.bin";
    std::vector<ObjectId> nodes = {node_oids_[5]};
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    CpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), 1u);
    EXPECT_TRUE(cache.contains(node_oids_[5]));
    EXPECT_FALSE(cache.contains(node_oids_[0]));

    // Hit on the only cached node
    auto result = cache.lookup({node_oids_[5]});
    ASSERT_EQ(result.hit_positions.size(), 1u);
    EXPECT_EQ(result.hit_positions[0], 0u);

    auto* data = reinterpret_cast<const float*>(result.features.data());
    // Node 5 -> features [601, 602, 603, 604]
    EXPECT_FLOAT_EQ(data[0], 601.0f);
    EXPECT_FLOAT_EQ(data[1], 602.0f);
    EXPECT_FLOAT_EQ(data[2], 603.0f);
    EXPECT_FLOAT_EQ(data[3], 604.0f);
}

// =============================================================================
// CpuCache: Lookup with duplicate ObjectIds in request
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_DuplicateLookup) {
    auto cache_path = test_dir_ / "cpu_dup.bin";
    std::vector<ObjectId> nodes = {node_oids_[3]};
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    CpuCache cache(cache_path);

    // Same node requested twice -> two hits, same features
    auto result = cache.lookup({node_oids_[3], node_oids_[3]});
    ASSERT_EQ(result.hit_positions.size(), 2u);
    EXPECT_EQ(result.hit_positions[0], 0u);
    EXPECT_EQ(result.hit_positions[1], 1u);
    EXPECT_EQ(result.miss_positions.size(), 0u);

    auto* data = reinterpret_cast<const float*>(result.features.data());
    // Both hits return features for node 3 -> [401, 402, 403, 404]
    for (int h = 0; h < 2; ++h) {
        EXPECT_FLOAT_EQ(data[h * D + 0], 401.0f) << "hit " << h;
        EXPECT_FLOAT_EQ(data[h * D + 1], 402.0f) << "hit " << h;
        EXPECT_FLOAT_EQ(data[h * D + 2], 403.0f) << "hit " << h;
        EXPECT_FLOAT_EQ(data[h * D + 3], 404.0f) << "hit " << h;
    }
}

// =============================================================================
// CpuCache: Lookup with empty request
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_EmptyLookup) {
    auto cache_path = test_dir_ / "cpu_elookup.bin";
    std::vector<ObjectId> nodes = {node_oids_[0]};
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    CpuCache cache(cache_path);
    auto result = cache.lookup({});
    EXPECT_EQ(result.hit_positions.size(), 0u);
    EXPECT_EQ(result.miss_positions.size(), 0u);
    EXPECT_TRUE(result.features.empty());
}

// =============================================================================
// CpuCache: Move semantics
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_MoveConstructor) {
    auto cache_path = test_dir_ / "cpu_move.bin";
    std::vector<ObjectId> nodes = {node_oids_[0], node_oids_[1]};
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    CpuCache original(cache_path);
    ASSERT_EQ(original.num_nodes(), 2u);

    CpuCache moved(std::move(original));
    EXPECT_EQ(moved.num_nodes(), 2u);
    EXPECT_TRUE(moved.contains(node_oids_[0]));
    EXPECT_TRUE(moved.contains(node_oids_[1]));
    EXPECT_NE(moved.data_ptr(), nullptr);

    // Moved-from should be empty
    EXPECT_EQ(original.num_nodes(), 0u);
    EXPECT_EQ(original.data_ptr(), nullptr);
}

TEST_F(FourLevelStoreTest, CpuCache_MoveAssignment) {
    auto cache_path1 = test_dir_ / "cpu_move1.bin";
    auto cache_path2 = test_dir_ / "cpu_move2.bin";
    CpuCache::build({node_oids_[0]}, *fm_, *rm_, cache_path1);
    CpuCache::build({node_oids_[7]}, *fm_, *rm_, cache_path2);

    CpuCache cache1(cache_path1);
    CpuCache cache2(cache_path2);
    ASSERT_TRUE(cache1.contains(node_oids_[0]));
    ASSERT_TRUE(cache2.contains(node_oids_[7]));

    cache1 = std::move(cache2);
    EXPECT_TRUE(cache1.contains(node_oids_[7]));
    EXPECT_FALSE(cache1.contains(node_oids_[0]));
    EXPECT_EQ(cache2.num_nodes(), 0u);
}

// =============================================================================
// CpuCache: Non-sequential node subset preserves correct features
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_NonSequentialNodes) {
    auto cache_path = test_dir_ / "cpu_nonseq.bin";
    // Cache nodes 1, 4, 7 (non-contiguous)
    std::vector<ObjectId> nodes = {node_oids_[1], node_oids_[4], node_oids_[7]};
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    CpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), 3u);

    // Lookup in reverse order
    auto result = cache.lookup({node_oids_[7], node_oids_[4], node_oids_[1]});
    ASSERT_EQ(result.hit_positions.size(), 3u);
    EXPECT_EQ(result.miss_positions.size(), 0u);

    auto* data = reinterpret_cast<const float*>(result.features.data());
    // Hit 0: node 7 -> [801, 802, 803, 804]
    EXPECT_FLOAT_EQ(data[0 * D + 0], 801.0f);
    EXPECT_FLOAT_EQ(data[0 * D + 3], 804.0f);
    // Hit 1: node 4 -> [501, 502, 503, 504]
    EXPECT_FLOAT_EQ(data[1 * D + 0], 501.0f);
    EXPECT_FLOAT_EQ(data[1 * D + 3], 504.0f);
    // Hit 2: node 1 -> [201, 202, 203, 204]
    EXPECT_FLOAT_EQ(data[2 * D + 0], 201.0f);
    EXPECT_FLOAT_EQ(data[2 * D + 3], 204.0f);
}

// =============================================================================
// CpuCache: FLOAT64 dtype round-trip
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_Float64) {
    // Create FLOAT64 FeatureMatrix
    auto fmat64_path = test_dir_ / "test64.fmat";
    auto rmap64_path = test_dir_ / "test64.rmap";

    std::vector<double> features64(N * D);
    for (uint64_t r = 0; r < N; ++r)
        for (uint64_t c = 0; c < D; ++c)
            features64[r * D + c] = static_cast<double>((r + 1) * 1000 + (c + 1));

    auto fm64 = FeatureMatrix::create(fmat64_path, N, D, GnnDtype::FLOAT64, features64.data());
    auto rm64 = RowMapping::create(rmap64_path, node_oids_);

    auto cache_path = test_dir_ / "cpu_f64.bin";
    std::vector<ObjectId> nodes = {node_oids_[2], node_oids_[6]};
    CpuCache::build(nodes, fm64, rm64, cache_path);

    CpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), 2u);
    EXPECT_EQ(cache.memory_bytes(), 2 * D * sizeof(double));

    auto result = cache.lookup({node_oids_[2], node_oids_[6]});
    ASSERT_EQ(result.hit_positions.size(), 2u);
    EXPECT_EQ(result.elem_size, sizeof(double));

    auto* data = reinterpret_cast<const double*>(result.features.data());
    // Node 2 -> [3001, 3002, 3003, 3004]
    EXPECT_DOUBLE_EQ(data[0], 3001.0);
    EXPECT_DOUBLE_EQ(data[1], 3002.0);
    EXPECT_DOUBLE_EQ(data[2], 3003.0);
    EXPECT_DOUBLE_EQ(data[3], 3004.0);
    // Node 6 -> [7001, 7002, 7003, 7004]
    EXPECT_DOUBLE_EQ(data[D + 0], 7001.0);
    EXPECT_DOUBLE_EQ(data[D + 3], 7004.0);
}

// =============================================================================
// CpuCache: is_pinned() returns false (no CUDA in test environment)
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_PinnedFlag) {
    auto cache_path = test_dir_ / "cpu_pinned.bin";
    CpuCache::build({node_oids_[0]}, *fm_, *rm_, cache_path);

    CpuCache cache(cache_path);
    // In non-CUDA builds, memory is never pinned
#ifndef GNN_CUDA_ENABLED
    EXPECT_FALSE(cache.is_pinned());
#endif
    // In either case, data_ptr() should be valid for non-empty cache
    EXPECT_NE(cache.data_ptr(), nullptr);
}

// =============================================================================
// Error: build() with ObjectId not in RowMapping
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_BuildMissingOid) {
    auto cache_path = test_dir_ / "cpu_bad.bin";
    ObjectId bad_oid(0xD4000000DEADBEEFULL);
    std::vector<ObjectId> nodes = {node_oids_[0], bad_oid};

    EXPECT_THROW(
        CpuCache::build(nodes, *fm_, *rm_, cache_path),
        std::runtime_error
    );
}

// =============================================================================
// Error: Loading a truncated file
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_TruncatedFile) {
    auto cache_path = test_dir_ / "cpu_trunc.bin";
    CpuCache::build({node_oids_[0], node_oids_[1]}, *fm_, *rm_, cache_path);

    // Truncate the file to just the header (remove OID table and features)
    fs::resize_file(cache_path, CacheFileHeader::SIZE);

    EXPECT_THROW(CpuCache cache(cache_path), std::runtime_error);
}

// =============================================================================
// Error: Loading a file with bad magic
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_BadMagic) {
    auto cache_path = test_dir_ / "cpu_badmagic.bin";

    // Write a file with corrupted magic
    CacheFileHeader hdr = CacheFileHeader::make(1, D, GnnDtype::FLOAT32);
    hdr.magic = 0xDEADDEAD;

    std::ofstream f(cache_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.close();

    EXPECT_THROW(CpuCache cache(cache_path), std::runtime_error);
}

// =============================================================================
// Error: Loading a nonexistent file
// =============================================================================

TEST_F(FourLevelStoreTest, CpuCache_NonexistentFile) {
    auto cache_path = test_dir_ / "does_not_exist.bin";
    EXPECT_THROW(CpuCache cache(cache_path), std::runtime_error);
}
