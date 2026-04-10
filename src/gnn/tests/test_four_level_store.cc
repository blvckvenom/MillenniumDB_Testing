#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <vector>

#include "gnn/storage/cache_file.h"
#include "gnn/storage/cpu_cache.h"
#include "gnn/storage/four_level_store.h"
#include "gnn/storage/gpu_cache.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/gnn_dtype.h"
#include "gnn/storage/packed_batch_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_config.h"
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

// =============================================================================
// GpuCache Tests
// =============================================================================

// =============================================================================
// GpuCache: Build and load empty cache
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_BuildAndLoad_Empty) {
    auto cache_path = test_dir_ / "gpu_cache_empty.bin";
    std::vector<ObjectId> nodes = {};
    GpuCache::build(nodes, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), 0u);
    EXPECT_EQ(cache.memory_bytes(), 0u);
    EXPECT_FALSE(cache.is_on_gpu());  // no data to put on GPU

    // Lookup against empty cache: all misses
    auto result = cache.lookup({node_oids_[0]});
    EXPECT_EQ(result.hit_positions.size(), 0u);
    EXPECT_EQ(result.miss_positions.size(), 1u);
}

// =============================================================================
// GpuCache: Build with specific nodes, verify contains + lookup features
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_BuildWithNodes) {
    auto cache_path = test_dir_ / "gpu_cache_nodes.bin";
    std::vector<ObjectId> nodes = {node_oids_[3], node_oids_[5]};
    GpuCache::build(nodes, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), 2u);
    EXPECT_EQ(cache.feature_dim(), D);
    EXPECT_TRUE(cache.contains(node_oids_[3]));
    EXPECT_TRUE(cache.contains(node_oids_[5]));
    EXPECT_FALSE(cache.contains(node_oids_[0]));

    // Lookup: 2 hits (3, 5) + 1 miss (0)
    auto result = cache.lookup({node_oids_[3], node_oids_[0], node_oids_[5]});
    EXPECT_EQ(result.hit_positions.size(), 2u);
    EXPECT_EQ(result.miss_positions.size(), 1u);

    // Hit positions (indices in input): 0(node3), 2(node5)
    EXPECT_EQ(result.hit_positions[0], 0u);
    EXPECT_EQ(result.hit_positions[1], 2u);
    EXPECT_EQ(result.miss_positions[0], 1u);

    // Verify features on CPU (even if tensor was on GPU)
    auto cpu_features = result.features.cpu();
    auto accessor = cpu_features.accessor<float, 2>();
    // First hit: node 3 -> row 3 -> features [401, 402, 403, 404]
    EXPECT_FLOAT_EQ(accessor[0][0], 401.0f);
    EXPECT_FLOAT_EQ(accessor[0][1], 402.0f);
    EXPECT_FLOAT_EQ(accessor[0][2], 403.0f);
    EXPECT_FLOAT_EQ(accessor[0][3], 404.0f);
    // Second hit: node 5 -> row 5 -> features [601, 602, 603, 604]
    EXPECT_FLOAT_EQ(accessor[1][0], 601.0f);
    EXPECT_FLOAT_EQ(accessor[1][1], 602.0f);
    EXPECT_FLOAT_EQ(accessor[1][2], 603.0f);
    EXPECT_FLOAT_EQ(accessor[1][3], 604.0f);
}

// =============================================================================
// GpuCache: Zero nodes -> empty cache, still functional
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_ZeroNodes) {
    auto cache_path = test_dir_ / "gpu_zero.bin";
    GpuCache::build({}, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);
    EXPECT_FALSE(cache.is_on_gpu());
    EXPECT_EQ(cache.num_nodes(), 0u);
    EXPECT_EQ(cache.feature_dim(), D);  // feature_dim preserved from header
    EXPECT_EQ(cache.memory_bytes(), 0u);
}

// =============================================================================
// GpuCache: All nodes cached — every lookup is a hit
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_AllNodes) {
    auto cache_path = test_dir_ / "gpu_all.bin";
    GpuCache::build(node_oids_, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), N);
    EXPECT_EQ(cache.feature_dim(), D);
    EXPECT_GT(cache.memory_bytes(), 0u);

    // All nodes should be found
    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_TRUE(cache.contains(node_oids_[i])) << "node " << i;
    }

    // Lookup all nodes: all hits, no misses
    auto result = cache.lookup(node_oids_);
    EXPECT_EQ(result.hit_positions.size(), N);
    EXPECT_EQ(result.miss_positions.size(), 0u);

    // Verify every row
    auto cpu_features = result.features.cpu();
    auto accessor = cpu_features.accessor<float, 2>();
    for (uint64_t r = 0; r < N; ++r) {
        for (uint64_t c = 0; c < D; ++c) {
            EXPECT_FLOAT_EQ(accessor[r][c], expected_feature(r, c))
                << "node " << r << ", dim " << c;
        }
    }
}

// =============================================================================
// GpuCache: Lookup all misses
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_LookupAllMisses) {
    auto cache_path = test_dir_ / "gpu_partial.bin";
    // Cache only nodes 0 and 1
    std::vector<ObjectId> cached = {node_oids_[0], node_oids_[1]};
    GpuCache::build(cached, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);

    // Lookup nodes that are NOT in the cache
    std::vector<ObjectId> query = {node_oids_[5], node_oids_[6], node_oids_[7]};
    auto result = cache.lookup(query);
    EXPECT_EQ(result.hit_positions.size(), 0u);
    EXPECT_EQ(result.miss_positions.size(), 3u);
    EXPECT_EQ(result.features.size(0), 0);
    EXPECT_EQ(result.features.size(1), static_cast<int64_t>(D));
}

// =============================================================================
// GpuCache: Lookup empty query
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_LookupEmptyQuery) {
    auto cache_path = test_dir_ / "gpu_for_empty_q.bin";
    std::vector<ObjectId> cached = {node_oids_[0]};
    GpuCache::build(cached, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);

    auto result = cache.lookup({});
    EXPECT_EQ(result.hit_positions.size(), 0u);
    EXPECT_EQ(result.miss_positions.size(), 0u);
    EXPECT_EQ(result.features.size(0), 0);
}

// =============================================================================
// GpuCache: Verify GNNC file format (header + OID table + data)
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_FileFormat) {
    auto cache_path = test_dir_ / "gpu_format.bin";
    std::vector<ObjectId> nodes = {node_oids_[2], node_oids_[4]};
    GpuCache::build(nodes, *fm_, *rm_, cache_path);

    // Expected file size: header(32) + OID table(2*8) + data(2*4*4)
    size_t expected = CacheFileHeader::SIZE + 2 * sizeof(uint64_t) + 2 * D * sizeof(float);
    EXPECT_EQ(fs::file_size(cache_path), expected);

    // Reload and verify
    GpuCache cache(cache_path);
    EXPECT_EQ(cache.num_nodes(), 2u);
    EXPECT_EQ(cache.feature_dim(), D);

    auto result = cache.lookup({node_oids_[2], node_oids_[4]});
    ASSERT_EQ(result.hit_positions.size(), 2u);
    auto cpu = result.features.cpu();
    auto acc = cpu.accessor<float, 2>();
    // node 2 -> row 2 -> [301, 302, 303, 304]
    EXPECT_FLOAT_EQ(acc[0][0], 301.0f);
    EXPECT_FLOAT_EQ(acc[0][3], 304.0f);
    // node 4 -> row 4 -> [501, 502, 503, 504]
    EXPECT_FLOAT_EQ(acc[1][0], 501.0f);
    EXPECT_FLOAT_EQ(acc[1][3], 504.0f);
}

// =============================================================================
// GpuCache: memory_bytes matches expected tensor size
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_MemoryBytes) {
    auto cache_path = test_dir_ / "gpu_mem.bin";
    std::vector<ObjectId> nodes = {node_oids_[0], node_oids_[1], node_oids_[2]};
    GpuCache::build(nodes, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);
    // 3 nodes * 4 dims * 4 bytes (float32) = 48 bytes
    EXPECT_EQ(cache.memory_bytes(), 3u * D * sizeof(float));
}

// =============================================================================
// GpuCache: Lookup preserves input order (hit_positions are input indices)
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_LookupOrderPreserved) {
    auto cache_path = test_dir_ / "gpu_order.bin";
    // Cache nodes 1, 3, 5
    std::vector<ObjectId> cached = {node_oids_[1], node_oids_[3], node_oids_[5]};
    GpuCache::build(cached, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);

    // Query in reverse order with a miss interleaved:
    // input[0]=node5(hit), input[1]=node4(miss), input[2]=node3(hit), input[3]=node1(hit)
    std::vector<ObjectId> query = {node_oids_[5], node_oids_[4], node_oids_[3], node_oids_[1]};
    auto result = cache.lookup(query);

    ASSERT_EQ(result.hit_positions.size(), 3u);
    ASSERT_EQ(result.miss_positions.size(), 1u);

    // Hit positions (indices in query): 0(node5), 2(node3), 3(node1)
    EXPECT_EQ(result.hit_positions[0], 0u);
    EXPECT_EQ(result.hit_positions[1], 2u);
    EXPECT_EQ(result.hit_positions[2], 3u);
    EXPECT_EQ(result.miss_positions[0], 1u);

    // Features should be in hit order: [node5, node3, node1]
    auto cpu = result.features.cpu();
    auto acc = cpu.accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc[0][0], 601.0f);  // node 5
    EXPECT_FLOAT_EQ(acc[1][0], 401.0f);  // node 3
    EXPECT_FLOAT_EQ(acc[2][0], 201.0f);  // node 1
}

// =============================================================================
// GpuCache: Duplicate ObjectIds in lookup request
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_DuplicateLookup) {
    auto cache_path = test_dir_ / "gpu_dup.bin";
    std::vector<ObjectId> nodes = {node_oids_[3]};
    GpuCache::build(nodes, *fm_, *rm_, cache_path);

    GpuCache cache(cache_path);

    // Same node requested twice -> two hits
    auto result = cache.lookup({node_oids_[3], node_oids_[3]});
    ASSERT_EQ(result.hit_positions.size(), 2u);
    EXPECT_EQ(result.hit_positions[0], 0u);
    EXPECT_EQ(result.hit_positions[1], 1u);
    EXPECT_EQ(result.miss_positions.size(), 0u);

    auto cpu = result.features.cpu();
    auto acc = cpu.accessor<float, 2>();
    // Both hits return features for node 3 -> [401, 402, 403, 404]
    for (int h = 0; h < 2; ++h) {
        EXPECT_FLOAT_EQ(acc[h][0], 401.0f) << "hit " << h;
        EXPECT_FLOAT_EQ(acc[h][1], 402.0f) << "hit " << h;
        EXPECT_FLOAT_EQ(acc[h][2], 403.0f) << "hit " << h;
        EXPECT_FLOAT_EQ(acc[h][3], 404.0f) << "hit " << h;
    }
}

// =============================================================================
// GpuCache: build() with ObjectId not in RowMapping throws
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_BuildMissingOid) {
    auto cache_path = test_dir_ / "gpu_bad.bin";
    ObjectId bad_oid(0xD4000000DEADBEEFULL);
    std::vector<ObjectId> nodes = {node_oids_[0], bad_oid};

    EXPECT_THROW(
        GpuCache::build(nodes, *fm_, *rm_, cache_path),
        std::runtime_error
    );
}

// =============================================================================
// GpuCache: Loading a truncated file throws
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_TruncatedFile) {
    auto cache_path = test_dir_ / "gpu_trunc.bin";
    GpuCache::build({node_oids_[0], node_oids_[1]}, *fm_, *rm_, cache_path);

    // Truncate the file to just the header
    fs::resize_file(cache_path, CacheFileHeader::SIZE);

    EXPECT_THROW(GpuCache cache(cache_path), std::runtime_error);
}

// =============================================================================
// GpuCache: Loading a file with bad magic throws
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_BadMagic) {
    auto cache_path = test_dir_ / "gpu_badmagic.bin";

    CacheFileHeader hdr = CacheFileHeader::make(1, D, GnnDtype::FLOAT32);
    hdr.magic = 0xDEADDEAD;

    std::ofstream f(cache_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.close();

    EXPECT_THROW(GpuCache cache(cache_path), std::runtime_error);
}

// =============================================================================
// GpuCache: Loading a nonexistent file throws
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_NonexistentFile) {
    auto cache_path = test_dir_ / "gpu_does_not_exist.bin";
    EXPECT_THROW(GpuCache cache(cache_path), std::runtime_error);
}

// =============================================================================
// GpuCache: GNNC file is interchangeable with CpuCache (same format)
// =============================================================================

TEST_F(FourLevelStoreTest, GpuCache_InterchangeableWithCpuCache) {
    auto cache_path = test_dir_ / "shared_cache.bin";
    std::vector<ObjectId> nodes = {node_oids_[2], node_oids_[6]};

    // Build using CpuCache::build, load with GpuCache
    CpuCache::build(nodes, *fm_, *rm_, cache_path);

    GpuCache gpu_cache(cache_path);
    EXPECT_EQ(gpu_cache.num_nodes(), 2u);
    EXPECT_EQ(gpu_cache.feature_dim(), D);
    EXPECT_TRUE(gpu_cache.contains(node_oids_[2]));
    EXPECT_TRUE(gpu_cache.contains(node_oids_[6]));

    auto result = gpu_cache.lookup({node_oids_[2], node_oids_[6]});
    ASSERT_EQ(result.hit_positions.size(), 2u);

    auto cpu = result.features.cpu();
    auto acc = cpu.accessor<float, 2>();
    // node 2 -> [301, 302, 303, 304]
    EXPECT_FLOAT_EQ(acc[0][0], 301.0f);
    EXPECT_FLOAT_EQ(acc[0][3], 304.0f);
    // node 6 -> [701, 702, 703, 704]
    EXPECT_FLOAT_EQ(acc[1][0], 701.0f);
    EXPECT_FLOAT_EQ(acc[1][3], 704.0f);
}

// #############################################################################
// FourLevelStore Coordinator Tests
// #############################################################################

/**
 * Test fixture for the FourLevelStore coordinator.
 *
 * Uses a db_folder layout matching what build() expects, plus a
 * create_samples() helper that writes SampleStorage with known batches.
 * Feature value scheme: row r, dim c -> (r+1)*100 + (c+1).
 */
class FourLevelStoreCoordTest : public GnnStorageTest {
protected:
    static constexpr uint64_t N = 8;
    static constexpr uint64_t D = 4;

    std::vector<ObjectId> node_oids_;
    std::vector<float> features_;

    fs::path db_folder_;
    fs::path gnn_dir_;
    fs::path fmat_path_;
    fs::path rmap_path_;

    void SetUp() override {
        GnnStorageTest::SetUp();

        db_folder_ = test_dir_ / "test_db";
        gnn_dir_   = db_folder_ / "gnn_features";
        fs::create_directories(gnn_dir_);

        fmat_path_ = gnn_dir_ / "test_feat.fmat";
        rmap_path_ = gnn_dir_ / "test_feat.rmap";

        node_oids_.resize(N);
        for (uint64_t i = 0; i < N; ++i) {
            node_oids_[i] = ObjectId(0xD400000000000000ULL | i);
        }

        features_.resize(N * D);
        for (uint64_t r = 0; r < N; ++r) {
            for (uint64_t c = 0; c < D; ++c) {
                features_[r * D + c] = static_cast<float>((r + 1) * 100 + (c + 1));
            }
        }

        FeatureMatrix::create(fmat_path_, N, D, GnnDtype::FLOAT32, features_.data());
        RowMapping::create(rmap_path_, node_oids_);
    }

    float expected_feature(uint64_t row, uint64_t dim) const {
        return static_cast<float>((row + 1) * 100 + (dim + 1));
    }

    /// Create SampleStorage with configurable batches.
    /// Each batch is defined by a list of node indices.
    SampleStorage create_samples(
        const std::string& name,
        const std::vector<std::vector<uint64_t>>& batch_node_indices)
    {
        SamplingConfig config;
        config.projection_name = "test_proj";
        config.sample_name = name;
        config.fanouts = {2};
        config.batch_size = 4;
        config.train_ratio = 1.0;
        config.val_ratio = 0.0;
        config.test_ratio = 0.0;

        auto storage = SampleStorage::create(db_folder_, config);

        for (uint64_t bid = 0; bid < batch_node_indices.size(); ++bid) {
            GraphSample s;
            s.batch_id = bid;
            s.split = SplitType::TRAIN;
            for (uint64_t idx : batch_node_indices[bid]) {
                s.nodes_per_layer.resize(1);
                s.nodes_per_layer[0].push_back(node_oids_[idx]);
                s.all_unique_nodes.push_back(node_oids_[idx]);
            }
            storage.write_sample(s);
        }
        storage.finalize();

        return SampleStorage::open(
            SampleStorage::get_storage_path(db_folder_, name));
    }

    /// Standard 5-batch setup with varying frequencies:
    /// Batch 0: nodes {0, 1, 2}      -> freq: 0 += 1, 1 += 1, 2 += 1
    /// Batch 1: nodes {0, 1, 3}      -> freq: 0 += 1, 1 += 1, 3 += 1
    /// Batch 2: nodes {0, 3, 4}      -> freq: 0 += 1, 3 += 1, 4 += 1
    /// Batch 3: nodes {0, 2, 5}      -> freq: 0 += 1, 2 += 1, 5 += 1
    /// Batch 4: nodes {0, 6, 7}      -> freq: 0 += 1, 6 += 1, 7 += 1
    ///
    /// Final frequencies: 0=5, 1=2, 2=2, 3=2, 4=1, 5=1, 6=1, 7=1
    /// With budget for 1 node each: L2 gets node 0 (no GPU in test)
    /// L3: nodes 1,2,3 (freq > 1)
    /// L4: nodes 4,5,6,7 (freq == 1)
    SampleStorage create_frequency_samples(
        const std::string& name = "fls_sample")
    {
        return create_samples(name, {
            {0, 1, 2},
            {0, 1, 3},
            {0, 3, 4},
            {0, 2, 5},
            {0, 6, 7}
        });
    }

    /// Build config with zero GPU budget and a specific CPU budget.
    FourLevelStore::Config make_config(
        size_t cpu_budget_nodes = 1,
        bool reorder = false,
        bool force = false)
    {
        FourLevelStore::Config config;
        config.gpu.budget_bytes = 0;           // zero GPU budget (test exercises CPU path)
        config.cpu.budget_bytes = cpu_budget_nodes * D * sizeof(float);
        config.reorder = reorder;
        config.force = force;
        config.minhash.num_hashes = 2;
        config.minhash.random_seed = 42;
        return config;
    }
};

// =============================================================================
// Build: Node classification by frequency
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_NodeClassification) {
    auto samples = create_frequency_samples();
    auto config = make_config(/*cpu_budget_nodes=*/1, /*reorder=*/false);

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    // gpu.budget_bytes=0 so L1 is always empty, regardless of GPU presence.
    EXPECT_EQ(result.l1_nodes, 0u);
    // CPU budget fits 1 node: only node 0 (freq=5, highest)
    EXPECT_EQ(result.l2_nodes, 1u);
    // gpu_available reflects actual CUDA detection (true on GPU machines,
    // false on CPU-only CI). The L1=0 assertion above already validates
    // that a zero GPU budget is respected regardless of hardware.
    EXPECT_EQ(result.gpu_available, torch::cuda::is_available());
    EXPECT_EQ(result.total_batches, 5u);
    EXPECT_GT(result.build_time_ms, -1);
    EXPECT_FALSE(result.packed_slim_dir.empty());
}

// =============================================================================
// Build: Larger CPU budget caches more nodes
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_LargerBudget) {
    auto samples = create_frequency_samples("fls_big");
    auto config = make_config(/*cpu_budget_nodes=*/3, /*reorder=*/false);

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    // Budget for 3 nodes: top-3 by frequency are 0(5), 1(2), 2(2) or 3(2)
    EXPECT_EQ(result.l1_nodes, 0u);
    EXPECT_EQ(result.l2_nodes, 3u);
}

// =============================================================================
// Build: Already exists -> error
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_AlreadyExistsError) {
    auto samples = create_frequency_samples();
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    // Second build without force should throw
    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_sample"));
    EXPECT_THROW(
        FourLevelStore::build(
            FeatureMatrix::open(fmat_path_),
            RowMapping::open(rmap_path_),
            samples2, config, db_folder_, "test_feat"),
        std::runtime_error);
}

// =============================================================================
// Build: Force overwrite succeeds
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_ForceOverwrite) {
    auto samples = create_frequency_samples();
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    // Second build with force should succeed
    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_sample"));
    auto config_force = make_config(1, false, /*force=*/true);

    EXPECT_NO_THROW(
        FourLevelStore::build(
            FeatureMatrix::open(fmat_path_),
            RowMapping::open(rmap_path_),
            samples2, config_force, db_folder_, "test_feat"));
}

// =============================================================================
// Build: GFLS metadata file written with correct values
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_MetadataFileCorrect) {
    auto samples = create_frequency_samples("fls_meta");
    auto config = make_config(1, false);

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto meta_path = gnn_dir_ / "test_feat_store.meta";
    ASSERT_TRUE(fs::exists(meta_path));
    EXPECT_EQ(fs::file_size(meta_path), StoreMetaHeader::SIZE);

    // Read and validate
    std::ifstream f(meta_path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    StoreMetaHeader meta{};
    f.read(reinterpret_cast<char*>(&meta), sizeof(meta));
    ASSERT_TRUE(f.good());

    EXPECT_EQ(meta.magic, StoreMetaHeader::MAGIC);
    EXPECT_EQ(meta.version, StoreMetaHeader::VERSION);
    EXPECT_TRUE(meta.is_valid());
    EXPECT_EQ(meta.l1_count, result.l1_nodes);
    EXPECT_EQ(meta.l2_count, result.l2_nodes);
    EXPECT_EQ(meta.feature_dim, D);
    EXPECT_EQ(meta.get_dtype(), GnnDtype::FLOAT32);
    EXPECT_EQ(meta.get_packed_slim_dir(), result.packed_slim_dir);
}

// =============================================================================
// Build: Slim files are v2 (have OID table)
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_SlimFilesAreV2) {
    auto samples = create_frequency_samples("fls_v2");
    auto config = make_config(1, false);

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto slim_dir = fs::path(result.packed_slim_dir);
    ASSERT_TRUE(fs::exists(slim_dir));

    // Check batch 0
    auto batch_path = slim_dir / "batch_000000.bin";
    ASSERT_TRUE(fs::exists(batch_path));

    std::ifstream f(batch_path, std::ios::binary);
    PackedBatchHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    ASSERT_TRUE(f.good());

    EXPECT_EQ(hdr.magic, PackedBatchHeader::MAGIC);
    EXPECT_EQ(hdr.version, 2u);
    EXPECT_TRUE(hdr.has_oid_table());
    EXPECT_EQ(hdr.feature_dim, D);
    EXPECT_EQ(hdr.get_dtype(), GnnDtype::FLOAT32);
}

// =============================================================================
// Build: Cache files are created
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_CacheFilesCreated) {
    auto samples = create_frequency_samples("fls_cache");
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    EXPECT_TRUE(fs::exists(gnn_dir_ / "test_feat_gpu_cache.bin"));
    EXPECT_TRUE(fs::exists(gnn_dir_ / "test_feat_cpu_cache.bin"));
    EXPECT_TRUE(fs::exists(gnn_dir_ / "test_feat_store.meta"));
}

// =============================================================================
// Build with reorder: reordered files created
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_WithReorder) {
    auto samples = create_frequency_samples("fls_reord");
    auto config = make_config(1, /*reorder=*/true);

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    EXPECT_TRUE(fs::exists(gnn_dir_ / "test_feat_reordered.fmat"));
    EXPECT_TRUE(fs::exists(gnn_dir_ / "test_feat_reordered.rmap"));

    // Reordered RowMapping must be a bijection
    auto reord_rm = RowMapping::open(gnn_dir_ / "test_feat_reordered.rmap");
    std::set<uint64_t> oid_set;
    for (uint64_t i = 0; i < N; ++i) {
        oid_set.insert(reord_rm.get(i).id);
    }
    EXPECT_EQ(oid_set.size(), N);
}

// =============================================================================
// Runtime: Constructor loads from metadata
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Constructor_LoadsFromMeta) {
    auto samples = create_frequency_samples("fls_load");
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_load"));

    // Should not throw
    FourLevelStore store(db_folder_, "test_feat", samples2);
    EXPECT_EQ(store.feature_dim(), D);
}

// =============================================================================
// Runtime: Constructor fails without metadata
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Constructor_NoMetadataThrows) {
    auto samples = create_frequency_samples("fls_nometa");

    EXPECT_THROW(
        FourLevelStore store(db_folder_, "nonexistent", samples),
        std::runtime_error);
}

// =============================================================================
// S4 Property: load_batch_features matches original features for ALL batches
// =============================================================================

TEST_F(FourLevelStoreCoordTest, LoadBatchFeatures_S4Property) {
    auto samples = create_frequency_samples("fls_s4");
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_s4"));
    FourLevelStore store(db_folder_, "test_feat", samples2);

    auto rm = RowMapping::open(rmap_path_);

    // S4: for every node in every batch, features match original
    for (uint64_t b = 0; b < 5; ++b) {
        auto sample = samples2.read_sample(b);
        auto tensor = store.load_batch_features(b);
        ASSERT_EQ(tensor.size(0), static_cast<int64_t>(sample.all_unique_nodes.size()))
            << "Batch " << b << " output size mismatch";
        ASSERT_EQ(tensor.size(1), static_cast<int64_t>(D));

        auto acc = tensor.accessor<float, 2>();
        for (size_t i = 0; i < sample.all_unique_nodes.size(); ++i) {
            auto oid = sample.all_unique_nodes[i];
            auto row = rm.find(oid);
            ASSERT_TRUE(row.has_value()) << "Batch " << b << " node " << i;
            for (uint64_t d = 0; d < D; ++d) {
                EXPECT_FLOAT_EQ(acc[i][d], expected_feature(*row, d))
                    << "S4 violation: batch " << b << " node " << i
                    << " (row " << *row << ") dim " << d;
            }
        }
    }
}

// =============================================================================
// S4 Property with reorder: features still match after MinHash reordering
// =============================================================================

TEST_F(FourLevelStoreCoordTest, LoadBatchFeatures_S4WithReorder) {
    auto samples = create_frequency_samples("fls_s4r");
    auto config = make_config(1, /*reorder=*/true);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_s4r"));
    FourLevelStore store(db_folder_, "test_feat", samples2);

    auto rm = RowMapping::open(rmap_path_);

    for (uint64_t b = 0; b < 5; ++b) {
        auto sample = samples2.read_sample(b);
        auto tensor = store.load_batch_features(b);
        auto acc = tensor.accessor<float, 2>();
        for (size_t i = 0; i < sample.all_unique_nodes.size(); ++i) {
            auto oid = sample.all_unique_nodes[i];
            auto row = rm.find(oid);
            ASSERT_TRUE(row.has_value());
            for (uint64_t d = 0; d < D; ++d) {
                EXPECT_FLOAT_EQ(acc[i][d], expected_feature(*row, d))
                    << "S4+reorder violation: batch " << b << " node " << i
                    << " dim " << d;
            }
        }
    }
}

// =============================================================================
// Stats: Counts are correct after load_batch_features
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Stats_CountsCorrect) {
    auto samples = create_frequency_samples("fls_stats");
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_stats"));
    FourLevelStore store(db_folder_, "test_feat", samples2);

    store.load_batch_features(0);
    const auto& stats = store.get_stats();

    // Batch 0: nodes {0, 1, 2}
    // node 0 is in L2 (cached), nodes 1,2 are not cached
    uint64_t total = stats.l1_hits.load() + stats.l2_hits.load()
                   + stats.l3_reads.load() + stats.l4_reads.load();

    auto sample = samples2.read_sample(0);
    uint64_t expected_total = sample.all_unique_nodes.size();
    EXPECT_EQ(total, expected_total)
        << "Total hits+reads must equal batch node count";
    EXPECT_EQ(stats.total_requests.load(), expected_total);

    // At least 1 L2 hit (node 0)
    EXPECT_GE(stats.l2_hits.load(), 1u);
}

// =============================================================================
// Stats: reset clears all counters
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Stats_ResetClears) {
    auto samples = create_frequency_samples("fls_rst");
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_rst"));
    FourLevelStore store(db_folder_, "test_feat", samples2);

    store.load_batch_features(0);
    store.reset_stats();

    const auto& stats = store.get_stats();
    EXPECT_EQ(stats.l1_hits.load(), 0u);
    EXPECT_EQ(stats.l2_hits.load(), 0u);
    EXPECT_EQ(stats.l3_reads.load(), 0u);
    EXPECT_EQ(stats.l4_reads.load(), 0u);
    EXPECT_EQ(stats.total_requests.load(), 0u);
}

// =============================================================================
// Stats: Accumulate across multiple batches
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Stats_AccumulateAcrossBatches) {
    auto samples = create_frequency_samples("fls_accum");
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_accum"));
    FourLevelStore store(db_folder_, "test_feat", samples2);

    // Load all 5 batches
    uint64_t total_nodes = 0;
    for (uint64_t b = 0; b < 5; ++b) {
        auto sample = samples2.read_sample(b);
        total_nodes += sample.all_unique_nodes.size();
        store.load_batch_features(b);
    }

    const auto& stats = store.get_stats();
    EXPECT_EQ(stats.total_requests.load(), total_nodes);

    uint64_t sum = stats.l1_hits.load() + stats.l2_hits.load()
                 + stats.l3_reads.load() + stats.l4_reads.load();
    EXPECT_EQ(sum, total_nodes);
}

// =============================================================================
// load_features: L1->L2->L3 only (no L4)
// =============================================================================

TEST_F(FourLevelStoreCoordTest, LoadFeatures_L1L2L3Only) {
    auto samples = create_frequency_samples("fls_lf");
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_lf"));
    FourLevelStore store(db_folder_, "test_feat", samples2);

    // Node 0 is in L2 cache: should get correct features
    auto tensor = store.load_features({node_oids_[0]});
    ASSERT_EQ(tensor.size(0), 1);
    ASSERT_EQ(tensor.size(1), static_cast<int64_t>(D));

    auto acc = tensor.accessor<float, 2>();
    for (uint64_t d = 0; d < D; ++d) {
        EXPECT_FLOAT_EQ(acc[0][d], expected_feature(0, d))
            << "load_features node 0, dim " << d;
    }
}

// =============================================================================
// load_features: Empty request returns empty tensor
// =============================================================================

TEST_F(FourLevelStoreCoordTest, LoadFeatures_EmptyRequest) {
    auto samples = create_frequency_samples("fls_empty");
    auto config = make_config(1, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_empty"));
    FourLevelStore store(db_folder_, "test_feat", samples2);

    auto tensor = store.load_features({});
    EXPECT_EQ(tensor.size(0), 0);
    EXPECT_EQ(tensor.size(1), static_cast<int64_t>(D));
}

// =============================================================================
// Build: Zero CPU budget -> all nodes go to L3/L4
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_ZeroBudget) {
    auto samples = create_frequency_samples("fls_zero");
    FourLevelStore::Config config;
    config.gpu.budget_bytes = 0;
    config.cpu.budget_bytes = 0;
    config.reorder = false;

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    EXPECT_EQ(result.l1_nodes, 0u);
    EXPECT_EQ(result.l2_nodes, 0u);
    // All accessed nodes go to L3/L4
    EXPECT_GT(result.l3_nodes + result.l4_nodes, 0u);
}

// =============================================================================
// Build: Zero budget + S4 still holds
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_ZeroBudget_S4) {
    auto samples = create_frequency_samples("fls_z4");
    FourLevelStore::Config config;
    config.gpu.budget_bytes = 0;
    config.cpu.budget_bytes = 0;
    config.reorder = false;

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_z4"));
    FourLevelStore store(db_folder_, "test_feat", samples2);
    auto rm = RowMapping::open(rmap_path_);

    for (uint64_t b = 0; b < 5; ++b) {
        auto sample = samples2.read_sample(b);
        auto tensor = store.load_batch_features(b);
        auto acc = tensor.accessor<float, 2>();
        for (size_t i = 0; i < sample.all_unique_nodes.size(); ++i) {
            auto oid = sample.all_unique_nodes[i];
            auto row = rm.find(oid);
            ASSERT_TRUE(row.has_value());
            for (uint64_t d = 0; d < D; ++d) {
                EXPECT_FLOAT_EQ(acc[i][d], expected_feature(*row, d))
                    << "Zero-budget S4: batch " << b << " node " << i;
            }
        }
    }
}

// =============================================================================
// Build: Large budget caches everything -> no L3/L4
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_LargeBudget_AllCached) {
    auto samples = create_frequency_samples("fls_all");
    auto config = make_config(/*cpu_budget_nodes=*/N, false);

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    // Budget for N nodes: all accessed nodes fit in L2
    // (no GPU -> L1=0, everything in L2)
    EXPECT_EQ(result.l1_nodes, 0u);
    // At most N nodes can be cached
    EXPECT_GE(result.l2_nodes, 1u);
}

// =============================================================================
// Single batch: All nodes in one batch
// =============================================================================

TEST_F(FourLevelStoreCoordTest, SingleBatch_S4) {
    auto samples = create_samples("fls_single", {{0,1,2,3,4,5,6,7}});
    auto config = make_config(2, false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_single"));
    FourLevelStore store(db_folder_, "test_feat", samples2);
    auto rm = RowMapping::open(rmap_path_);

    auto tensor = store.load_batch_features(0);
    ASSERT_EQ(tensor.size(0), static_cast<int64_t>(N));

    auto acc = tensor.accessor<float, 2>();
    for (uint64_t i = 0; i < N; ++i) {
        auto oid = samples2.read_sample(0).all_unique_nodes[i];
        auto row = rm.find(oid);
        ASSERT_TRUE(row.has_value());
        for (uint64_t d = 0; d < D; ++d) {
            EXPECT_FLOAT_EQ(acc[i][d], expected_feature(*row, d))
                << "Single batch S4: node " << i << " dim " << d;
        }
    }
}

// =============================================================================
// S4 Property with FLOAT64 end-to-end
// Catches dtype conversion bugs in the pipeline
// =============================================================================

TEST_F(FourLevelStoreCoordTest, LoadBatchFeatures_S4WithFloat64) {
    // Create FLOAT64 FeatureMatrix + RowMapping
    auto fmat64_path = gnn_dir_ / "test_feat64.fmat";
    auto rmap64_path = gnn_dir_ / "test_feat64.rmap";

    std::vector<double> features64(N * D);
    for (uint64_t r = 0; r < N; ++r)
        for (uint64_t c = 0; c < D; ++c)
            features64[r * D + c] = static_cast<double>((r + 1) * 1000 + (c + 1));

    FeatureMatrix::create(fmat64_path, N, D, GnnDtype::FLOAT64, features64.data());
    RowMapping::create(rmap64_path, node_oids_);

    // Create samples for the FLOAT64 test
    auto samples = create_frequency_samples("fls_f64");
    auto config = make_config(/*cpu_budget_nodes=*/1, /*reorder=*/false);

    FourLevelStore::build(
        FeatureMatrix::open(fmat64_path),
        RowMapping::open(rmap64_path),
        samples, config, db_folder_, "test_feat64");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_f64"));
    FourLevelStore store(db_folder_, "test_feat64", samples2);

    auto rm = RowMapping::open(rmap64_path);

    // S4: for every node in every batch, features match original FLOAT64 values
    for (uint64_t b = 0; b < 5; ++b) {
        auto sample = samples2.read_sample(b);
        auto tensor = store.load_batch_features(b);
        ASSERT_EQ(tensor.size(0), static_cast<int64_t>(sample.all_unique_nodes.size()))
            << "FLOAT64 batch " << b << " output size mismatch";
        ASSERT_EQ(tensor.size(1), static_cast<int64_t>(D));

        // Tensor dtype should be float64
        EXPECT_EQ(tensor.scalar_type(), torch::kFloat64)
            << "FLOAT64 S4: expected double tensor at batch " << b;

        auto acc = tensor.accessor<double, 2>();
        for (size_t i = 0; i < sample.all_unique_nodes.size(); ++i) {
            auto oid = sample.all_unique_nodes[i];
            auto row = rm.find(oid);
            ASSERT_TRUE(row.has_value()) << "FLOAT64 batch " << b << " node " << i;
            for (uint64_t d = 0; d < D; ++d) {
                double expected = static_cast<double>((*row + 1) * 1000 + (d + 1));
                EXPECT_DOUBLE_EQ(acc[i][d], expected)
                    << "FLOAT64 S4 violation: batch " << b << " node " << i
                    << " (row " << *row << ") dim " << d;
            }
        }
    }
}

// =============================================================================
// Large Feature Dimension D=2048
// Tests CUDA kernel tiling (D > blockDim.x=256) and large row handling
// =============================================================================

TEST_F(FourLevelStoreTest, LargeFeatureDim_2048) {
    static constexpr uint64_t BIG_D = 2048;
    static constexpr uint64_t BIG_N = 8;

    // Create 8-node FeatureMatrix with D=2048
    std::vector<float> big_features(BIG_N * BIG_D);
    for (uint64_t r = 0; r < BIG_N; ++r)
        for (uint64_t c = 0; c < BIG_D; ++c)
            big_features[r * BIG_D + c] = static_cast<float>(r * 10000 + c);

    auto fmat_big_path = test_dir_ / "big.fmat";
    auto rmap_big_path = test_dir_ / "big.rmap";

    auto fm_big = FeatureMatrix::create(fmat_big_path, BIG_N, BIG_D, GnnDtype::FLOAT32,
                                        big_features.data());
    auto rm_big = RowMapping::create(rmap_big_path, node_oids_);

    // Build CpuCache with 2 nodes
    auto cpu_cache_path = test_dir_ / "big_cpu_cache.bin";
    std::vector<ObjectId> cpu_nodes = {node_oids_[0], node_oids_[3]};
    CpuCache::build(cpu_nodes, fm_big, rm_big, cpu_cache_path);

    CpuCache cache(cpu_cache_path);
    EXPECT_EQ(cache.num_nodes(), 2u);
    EXPECT_EQ(cache.feature_dim(), BIG_D);
    EXPECT_EQ(cache.memory_bytes(), 2u * BIG_D * sizeof(float));

    // Verify correct features for node 0 (row 0)
    auto result = cache.lookup({node_oids_[0]});
    ASSERT_EQ(result.hit_positions.size(), 1u);
    auto* data = reinterpret_cast<const float*>(result.features.data());
    for (uint64_t c = 0; c < BIG_D; ++c) {
        EXPECT_FLOAT_EQ(data[c], static_cast<float>(0 * 10000 + c))
            << "Big D cache mismatch at dim " << c;
    }

    // Build GpuCache with 2 nodes
    auto gpu_cache_path = test_dir_ / "big_gpu_cache.bin";
    std::vector<ObjectId> gpu_nodes = {node_oids_[5], node_oids_[7]};
    GpuCache::build(gpu_nodes, fm_big, rm_big, gpu_cache_path);

    GpuCache gpu_cache(gpu_cache_path);
    EXPECT_EQ(gpu_cache.num_nodes(), 2u);
    EXPECT_EQ(gpu_cache.feature_dim(), BIG_D);

    auto gpu_result = gpu_cache.lookup({node_oids_[5]});
    ASSERT_EQ(gpu_result.hit_positions.size(), 1u);
    auto cpu_tensor = gpu_result.features.cpu();
    auto acc = cpu_tensor.accessor<float, 2>();
    // Row 5: values 5*10000 + c
    EXPECT_FLOAT_EQ(acc[0][0], 50000.0f);
    EXPECT_FLOAT_EQ(acc[0][1023], 51023.0f);
    EXPECT_FLOAT_EQ(acc[0][2047], 52047.0f);
}

// =============================================================================
// Empty Sample Storage: 0 batches
// Build should succeed with all counts = 0
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_EmptySampleStorage) {
    // Create SampleStorage with 0 batches
    auto samples = create_samples("fls_empty_samples", {});
    auto config = make_config(1, false);

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    EXPECT_EQ(result.l1_nodes, 0u);
    EXPECT_EQ(result.l2_nodes, 0u);
    EXPECT_EQ(result.l3_nodes, 0u);
    // With 0 batches, all N nodes have freq=0, classified as L4
    EXPECT_EQ(result.l4_nodes, N);
    EXPECT_EQ(result.total_batches, 0u);
}

// =============================================================================
// Budget Boundary: exactly fits all N nodes
// All nodes should go to L2, L3=0, L4=0
// =============================================================================

TEST_F(FourLevelStoreCoordTest, Build_BudgetExactlyFitsAll) {
    auto samples = create_frequency_samples("fls_exact");
    // Set cpu_budget to exactly N * D * sizeof(float) — fits all 8 nodes
    auto config = make_config(/*cpu_budget_nodes=*/N, /*reorder=*/false);

    auto result = FourLevelStore::build(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    // All accessed nodes should fit in L2 (no GPU available)
    EXPECT_EQ(result.l1_nodes, 0u);
    // All 8 unique nodes accessed across batches should be cached
    EXPECT_EQ(result.l2_nodes, N);
    // Nothing should spill to L3/L4
    EXPECT_EQ(result.l3_nodes, 0u);
    EXPECT_EQ(result.l4_nodes, 0u);

    // S4 property should still hold
    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "fls_exact"));
    FourLevelStore store(db_folder_, "test_feat", samples2);
    auto rm = RowMapping::open(rmap_path_);

    for (uint64_t b = 0; b < 5; ++b) {
        auto sample = samples2.read_sample(b);
        auto tensor = store.load_batch_features(b);
        auto acc = tensor.accessor<float, 2>();
        for (size_t i = 0; i < sample.all_unique_nodes.size(); ++i) {
            auto oid = sample.all_unique_nodes[i];
            auto row = rm.find(oid);
            ASSERT_TRUE(row.has_value());
            for (uint64_t d = 0; d < D; ++d) {
                EXPECT_FLOAT_EQ(acc[i][d], expected_feature(*row, d))
                    << "Exact budget S4: batch " << b << " node " << i
                    << " dim " << d;
            }
        }
    }
}
