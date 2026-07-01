#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <set>
#include <vector>

#include "gnn/storage/batch_materializer.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/feature_matrix_header.h"
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

class BatchMaterializerTest : public GnnStorageTest {
protected:
    static constexpr uint64_t N = 8;
    static constexpr uint64_t D = 4;

    // Known feature values: row r, col c → value = (r+1)*100 + (c+1)
    // Row 0: [101, 102, 103, 104]
    // Row 1: [201, 202, 203, 204]
    // ...
    // Row 7: [801, 802, 803, 804]
    // This makes each value globally unique → any mismatch is instantly visible.
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

    /// Get KNOWN expected feature for node index r, dimension c.
    float expected_feature(uint64_t node_index, uint64_t dim) const {
        return static_cast<float>((node_index + 1) * 100 + (dim + 1));
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

    /// Standard 3-batch setup:
    /// Batch 0: nodes {0, 1, 2}       — 3 nodes
    /// Batch 1: nodes {2, 3, 4, 5}    — 4 nodes (node 2 shared)
    /// Batch 2: nodes {5, 6, 7}        — 3 nodes (node 5 shared)
    SampleStorage create_standard_samples(const std::string& name = "std_sample") {
        return create_samples(name, {{0,1,2}, {2,3,4,5}, {5,6,7}});
    }
};

// =============================================================================
// L4 WITHOUT REORDER: Verify exact feature values in ALL batches
// =============================================================================

TEST_F(BatchMaterializerTest, PackWithoutReorder_AllBatchesExactValues) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = false;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    // Expected values for each batch
    // Batch 0: nodes {0,1,2} → features [[101..104], [201..204], [301..304]]
    // Batch 1: nodes {2,3,4,5} → features [[301..304], [401..404], [501..504], [601..604]]
    // Batch 2: nodes {5,6,7} → features [[601..604], [701..704], [801..804]]
    std::vector<std::vector<uint64_t>> expected_nodes = {{0,1,2}, {2,3,4,5}, {5,6,7}};

    ASSERT_EQ(result.total_batches, 3u);
    ASSERT_FALSE(result.reordered);
    ASSERT_EQ(result.reorder_time_ms, 0);

    auto packed_dir = fs::path(result.packed_dir);
    PackedBatchReader reader(packed_dir, 3, D, GnnDtype::FLOAT32);

    for (uint64_t bid = 0; bid < 3; ++bid) {
        auto hdr = reader.read_header(bid);
        ASSERT_EQ(hdr.num_nodes, expected_nodes[bid].size())
            << "Batch " << bid << " node count mismatch";
        ASSERT_EQ(hdr.feature_dim, D);
        ASSERT_EQ(hdr.magic, PackedBatchHeader::MAGIC);
        ASSERT_EQ(hdr.version, PackedBatchHeader::VERSION);

        std::vector<float> buf(hdr.num_nodes * D);
        reader.read_batch(bid, buf.data(), buf.size() * sizeof(float));

        for (uint64_t i = 0; i < hdr.num_nodes; ++i) {
            uint64_t node_idx = expected_nodes[bid][i];
            for (uint64_t j = 0; j < D; ++j) {
                float got = buf[i * D + j];
                float want = expected_feature(node_idx, j);
                EXPECT_FLOAT_EQ(got, want)
                    << "Batch " << bid << ", pos " << i
                    << " (node " << node_idx << "), dim " << j
                    << ": got " << got << ", want " << want;
            }
        }
    }
}

// =============================================================================
// L4 WITHOUT REORDER: Verify exact file sizes
// =============================================================================

TEST_F(BatchMaterializerTest, PackWithoutReorder_ExactFileSizes) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = false;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto packed_dir = fs::path(result.packed_dir);
    // Expected: header(32) + num_nodes * D * sizeof(float)
    std::vector<uint64_t> expected_nodes = {3, 4, 3};
    for (uint64_t bid = 0; bid < 3; ++bid) {
        auto path = packed_dir / ("batch_" + std::string(6 - std::to_string(bid).size(), '0') + std::to_string(bid) + ".bin");
        uint64_t expected_size = 32 + expected_nodes[bid] * D * sizeof(float);
        EXPECT_EQ(fs::file_size(path), expected_size)
            << "Batch " << bid << " file size mismatch";
    }

    // No reordered files
    EXPECT_FALSE(fs::exists(gnn_dir_ / "test_feat_reordered.fmat"));
    EXPECT_FALSE(fs::exists(gnn_dir_ / "test_feat_reordered.rmap"));
}

// =============================================================================
// L3+L4: S4 critical property on ALL batches with exact values
// =============================================================================

TEST_F(BatchMaterializerTest, PackWithReorder_S4CriticalProperty) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.num_hashes = 2;
    config.minhash.random_seed = 42;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    ASSERT_TRUE(result.reordered);
    ASSERT_EQ(result.total_batches, 3u);

    // Re-read samples for node lists
    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "std_sample"));
    auto packed_dir = fs::path(result.packed_dir);
    PackedBatchReader reader(packed_dir, 3, D, GnnDtype::FLOAT32);

    for (uint64_t bid = 0; bid < 3; ++bid) {
        auto sample = samples2.read_sample(bid);
        auto hdr = reader.read_header(bid);
        ASSERT_EQ(hdr.num_nodes, sample.all_unique_nodes.size());

        std::vector<float> packed(hdr.num_nodes * D);
        reader.read_batch(bid, packed.data(), packed.size() * sizeof(float));

        for (uint64_t i = 0; i < hdr.num_nodes; ++i) {
            // Extract node index from ObjectId
            uint64_t node_idx = sample.all_unique_nodes[i].id & 0x00FFFFFFFFFFFFFFULL;
            for (uint64_t j = 0; j < D; ++j) {
                float got = packed[i * D + j];
                float want = expected_feature(node_idx, j);
                EXPECT_FLOAT_EQ(got, want)
                    << "Batch " << bid << ", pos " << i
                    << " (node " << node_idx << "), dim " << j;
            }
        }
    }
}

// =============================================================================
// L3: Reordered RowMapping is a bijection (same OID set as original)
// =============================================================================

TEST_F(BatchMaterializerTest, ReorderedRowMappingIsBijection) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.num_hashes = 2;
    config.minhash.random_seed = 42;

    BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto orig_rm = RowMapping::open(rmap_path_);
    auto reord_rm = RowMapping::open(gnn_dir_ / "test_feat_reordered.rmap");

    ASSERT_EQ(reord_rm.size(), N);

    // Collect all OIDs from both mappings
    std::set<uint64_t> orig_oids, reord_oids;
    for (uint64_t i = 0; i < N; ++i) {
        orig_oids.insert(orig_rm.get(i).id);
        reord_oids.insert(reord_rm.get(i).id);
    }

    // Same set = bijection (each OID appears exactly once in each)
    EXPECT_EQ(orig_oids, reord_oids)
        << "Reordered RowMapping must contain the same ObjectIds as original";
    EXPECT_EQ(orig_oids.size(), N)
        << "All ObjectIds must be unique";
}

// =============================================================================
// L3: RowMapping coherence — rm[i] ↔ fm[i] for all i
// =============================================================================

TEST_F(BatchMaterializerTest, RowMappingCoherenceChain) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.num_hashes = 2;
    config.minhash.random_seed = 42;

    BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto reord_fm = FeatureMatrix::open(gnn_dir_ / "test_feat_reordered.fmat");
    auto reord_rm = RowMapping::open(gnn_dir_ / "test_feat_reordered.rmap");

    ASSERT_EQ(reord_fm.num_rows(), N);
    ASSERT_EQ(reord_fm.num_cols(), D);

    for (uint64_t i = 0; i < N; ++i) {
        ObjectId oid = reord_rm.get(i);
        uint64_t node_idx = oid.id & 0x00FFFFFFFFFFFFFFULL;

        const float* feat = reord_fm.row_as<float>(i);
        for (uint64_t j = 0; j < D; ++j) {
            float got = feat[j];
            float want = expected_feature(node_idx, j);
            EXPECT_FLOAT_EQ(got, want)
                << "Reordered row " << i << " (node " << node_idx
                << "), dim " << j << ": got " << got << ", want " << want;
        }
    }
}

// =============================================================================
// L3: Reordered FM has same metadata as original
// =============================================================================

TEST_F(BatchMaterializerTest, ReorderedFMMetadataMatchesOriginal) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.num_hashes = 2;

    BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto orig_fm = FeatureMatrix::open(fmat_path_);
    auto reord_fm = FeatureMatrix::open(gnn_dir_ / "test_feat_reordered.fmat");

    EXPECT_EQ(reord_fm.num_rows(), orig_fm.num_rows());
    EXPECT_EQ(reord_fm.num_cols(), orig_fm.num_cols());
    EXPECT_EQ(reord_fm.dtype(), orig_fm.dtype());

    // File sizes must be identical
    EXPECT_EQ(fs::file_size(gnn_dir_ / "test_feat_reordered.fmat"),
              fs::file_size(fmat_path_));
    EXPECT_EQ(fs::file_size(gnn_dir_ / "test_feat_reordered.rmap"),
              fs::file_size(rmap_path_));
}

// =============================================================================
// Edge case: Single batch containing ALL nodes
// =============================================================================

TEST_F(BatchMaterializerTest, SingleBatchAllNodes) {
    auto samples = create_samples("all_in_one", {{0,1,2,3,4,5,6,7}});

    BatchMaterializer::Config config;
    config.reorder = false;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    ASSERT_EQ(result.total_batches, 1u);

    auto packed_dir = fs::path(result.packed_dir);
    PackedBatchReader reader(packed_dir, 1, D, GnnDtype::FLOAT32);
    auto hdr = reader.read_header(0);
    ASSERT_EQ(hdr.num_nodes, N);

    std::vector<float> buf(N * D);
    reader.read_batch(0, buf.data(), buf.size() * sizeof(float));

    for (uint64_t r = 0; r < N; ++r) {
        for (uint64_t c = 0; c < D; ++c) {
            EXPECT_FLOAT_EQ(buf[r * D + c], expected_feature(r, c))
                << "Node " << r << ", dim " << c;
        }
    }
}

// =============================================================================
// Edge case: Single batch (MinHash degenerate — no access diversity)
// =============================================================================

TEST_F(BatchMaterializerTest, SingleBatchReorderDegenerate) {
    auto samples = create_samples("one_batch", {{0,1,2,3}});

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.num_hashes = 2;
    config.minhash.random_seed = 42;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    ASSERT_TRUE(result.reordered);
    ASSERT_EQ(result.total_batches, 1u);

    // S4 must still hold even with degenerate MinHash
    auto packed_dir = fs::path(result.packed_dir);
    PackedBatchReader reader(packed_dir, 1, D, GnnDtype::FLOAT32);
    auto hdr = reader.read_header(0);
    ASSERT_EQ(hdr.num_nodes, 4u);

    std::vector<float> buf(4 * D);
    reader.read_batch(0, buf.data(), buf.size() * sizeof(float));

    // Read the actual node order from the sample
    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "one_batch"));
    auto sample = samples2.read_sample(0);

    for (uint64_t i = 0; i < 4; ++i) {
        uint64_t node_idx = sample.all_unique_nodes[i].id & 0x00FFFFFFFFFFFFFFULL;
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(buf[i * D + j], expected_feature(node_idx, j))
                << "Degenerate batch, pos " << i << ", dim " << j;
        }
    }
}

// =============================================================================
// Edge case: Shared node appears in ALL batches (max frequency)
// =============================================================================

TEST_F(BatchMaterializerTest, NodeInAllBatches) {
    // Node 0 appears in every batch
    auto samples = create_samples("shared_node", {{0,1}, {0,2}, {0,3}});

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.num_hashes = 2;
    config.minhash.random_seed = 42;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    ASSERT_EQ(result.total_batches, 3u);

    // Verify node 0's features appear correctly in EVERY packed batch
    auto packed_dir = fs::path(result.packed_dir);
    PackedBatchReader reader(packed_dir, 3, D, GnnDtype::FLOAT32);

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "shared_node"));

    for (uint64_t bid = 0; bid < 3; ++bid) {
        auto sample = samples2.read_sample(bid);
        auto hdr = reader.read_header(bid);
        std::vector<float> buf(hdr.num_nodes * D);
        reader.read_batch(bid, buf.data(), buf.size() * sizeof(float));

        // Find node 0 in this batch and verify its features
        bool found_node0 = false;
        for (uint64_t i = 0; i < hdr.num_nodes; ++i) {
            uint64_t node_idx = sample.all_unique_nodes[i].id & 0x00FFFFFFFFFFFFFFULL;
            if (node_idx == 0) {
                found_node0 = true;
                for (uint64_t j = 0; j < D; ++j) {
                    EXPECT_FLOAT_EQ(buf[i * D + j], expected_feature(0, j))
                        << "Batch " << bid << ": node 0, dim " << j;
                }
            }
        }
        EXPECT_TRUE(found_node0) << "Node 0 should appear in batch " << bid;
    }
}

// =============================================================================
// Force: Verify output correctness AFTER force overwrite
// =============================================================================

TEST_F(BatchMaterializerTest, ForceOverwriteProducesCorrectOutput) {
    auto samples = create_standard_samples();
    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);

    BatchMaterializer::Config config;
    config.reorder = false;

    // First run
    BatchMaterializer::materialize(fm, rm, samples, config, db_folder_, "test_feat");

    // Second run with force — verify output is STILL correct
    config.force = true;
    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "std_sample"));
    auto result = BatchMaterializer::materialize(fm, rm, samples2, config, db_folder_, "test_feat");

    auto packed_dir = fs::path(result.packed_dir);
    PackedBatchReader reader(packed_dir, 3, D, GnnDtype::FLOAT32);

    // Spot-check batch 1: nodes {2,3,4,5}
    auto hdr = reader.read_header(1);
    ASSERT_EQ(hdr.num_nodes, 4u);
    std::vector<float> buf(4 * D);
    reader.read_batch(1, buf.data(), buf.size() * sizeof(float));

    uint64_t expected_indices[] = {2, 3, 4, 5};
    for (uint64_t i = 0; i < 4; ++i) {
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(buf[i * D + j], expected_feature(expected_indices[i], j))
                << "After force, batch 1, pos " << i << ", dim " << j;
        }
    }
}

// =============================================================================
// Error: Already materialized → helpful message
// =============================================================================

TEST_F(BatchMaterializerTest, AlreadyMaterializedError) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = false;

    BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "std_sample"));

    try {
        BatchMaterializer::materialize(
            FeatureMatrix::open(fmat_path_),
            RowMapping::open(rmap_path_),
            samples2, config, db_folder_, "test_feat");
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("already exist"), std::string::npos) << "Error: " << msg;
        EXPECT_NE(msg.find("force"), std::string::npos) << "Should suggest force option";
    }
}

// =============================================================================
// Error: Missing ObjectId → descriptive error with batch ID
// =============================================================================

TEST_F(BatchMaterializerTest, MissingObjectIdError) {
    SamplingConfig sc;
    sc.projection_name = "bad_proj";
    sc.sample_name = "bad_sample";
    sc.fanouts = {2};
    sc.batch_size = 4;
    sc.train_ratio = 1.0;
    sc.val_ratio = 0.0;
    sc.test_ratio = 0.0;

    auto storage = SampleStorage::create(db_folder_, sc);

    GraphSample bad;
    bad.batch_id = 0;
    bad.split = SplitType::TRAIN;
    ObjectId bad_oid(0xD4000000DEADBEEFULL);
    bad.nodes_per_layer = {{bad_oid}};
    bad.all_unique_nodes = {bad_oid};
    storage.write_sample(bad);
    storage.finalize();

    auto samples = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "bad_sample"));

    BatchMaterializer::Config config;
    config.reorder = false;

    try {
        BatchMaterializer::materialize(
            FeatureMatrix::open(fmat_path_),
            RowMapping::open(rmap_path_),
            samples, config, db_folder_, "test_feat");
        FAIL() << "Expected runtime_error for missing ObjectId";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("no corresponding feature row"), std::string::npos)
            << "Error: " << msg;
        EXPECT_NE(msg.find("batch 0"), std::string::npos)
            << "Should identify which batch";
    }
}

// =============================================================================
// Error: Without force, reordered files block re-run
// =============================================================================

TEST_F(BatchMaterializerTest, ReorderedFilesBlockRerun) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.num_hashes = 2;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    // Remove packed dir but keep reordered files → should hit reordered check
    fs::remove_all(result.packed_dir);
    ASSERT_TRUE(fs::exists(gnn_dir_ / "test_feat_reordered.fmat"));
    ASSERT_FALSE(fs::exists(result.packed_dir));

    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "std_sample"));

    try {
        BatchMaterializer::materialize(
            FeatureMatrix::open(fmat_path_),
            RowMapping::open(rmap_path_),
            samples2, config, db_folder_, "test_feat");
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Reordered matrix already exists"), std::string::npos)
            << "Error: " << msg;
    }
}

// =============================================================================
// Result struct: Timing values are plausible
// =============================================================================

TEST_F(BatchMaterializerTest, TimingValuesPlausible) {
    auto samples = create_standard_samples();

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.num_hashes = 2;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    EXPECT_GE(result.reorder_time_ms, 0);
    EXPECT_GE(result.pack_time_ms, 0);
    EXPECT_GE(result.total_time_ms, 0);
    EXPECT_GE(result.total_time_ms, result.reorder_time_ms + result.pack_time_ms);
    EXPECT_FALSE(result.packed_dir.empty());
    EXPECT_TRUE(fs::exists(result.packed_dir));
}

// =============================================================================
// translate_to_rows: Direct unit tests (now public)
// =============================================================================

TEST_F(BatchMaterializerTest, TranslateToRows_NoInverse) {
    auto rm = RowMapping::open(rmap_path_);

    // Without inverse: ObjectId → direct row index
    std::vector<ObjectId> oids = {node_oids_[3], node_oids_[0], node_oids_[7]};
    auto rows = BatchMaterializer::translate_to_rows(oids, rm, nullptr, 0);

    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0], 3u);  // node_oids_[3] → row 3
    EXPECT_EQ(rows[1], 0u);  // node_oids_[0] → row 0
    EXPECT_EQ(rows[2], 7u);  // node_oids_[7] → row 7
}

TEST_F(BatchMaterializerTest, TranslateToRows_WithKnownInverse) {
    auto rm = RowMapping::open(rmap_path_);

    // Known inverse: old_row 0→5, 1→3, 2→7, 3→0, 4→1, 5→2, 6→6, 7→4
    std::vector<uint64_t> inverse = {5, 3, 7, 0, 1, 2, 6, 4};

    std::vector<ObjectId> oids = {node_oids_[3], node_oids_[0], node_oids_[7]};
    auto rows = BatchMaterializer::translate_to_rows(oids, rm, &inverse, 0);

    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0], inverse[3]);  // node 3 → old_row 3 → inverse[3] = 0
    EXPECT_EQ(rows[1], inverse[0]);  // node 0 → old_row 0 → inverse[0] = 5
    EXPECT_EQ(rows[2], inverse[7]);  // node 7 → old_row 7 → inverse[7] = 4
}

TEST_F(BatchMaterializerTest, TranslateToRows_MissingOidThrows) {
    auto rm = RowMapping::open(rmap_path_);

    ObjectId bad_oid(0xD4000000DEADBEEFULL);
    std::vector<ObjectId> oids = {node_oids_[0], bad_oid};

    try {
        BatchMaterializer::translate_to_rows(oids, rm, nullptr, 42);
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("no corresponding feature row"), std::string::npos);
        EXPECT_NE(msg.find("batch 42"), std::string::npos);
    }
}

TEST_F(BatchMaterializerTest, TranslateToRows_InverseOutOfBoundsThrows) {
    auto rm = RowMapping::open(rmap_path_);

    // Inverse too small — only 4 entries but node 7 maps to old_row 7
    std::vector<uint64_t> small_inverse = {0, 1, 2, 3};

    std::vector<ObjectId> oids = {node_oids_[7]};  // old_row = 7 >= inverse.size() = 4

    try {
        BatchMaterializer::translate_to_rows(oids, rm, &small_inverse, 99);
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("out of bounds"), std::string::npos);
        EXPECT_NE(msg.find("batch 99"), std::string::npos);
    }
}

// =============================================================================
// FLOAT64: Pipeline works with double precision
// =============================================================================

TEST_F(BatchMaterializerTest, Float64Pipeline) {
    // Create FLOAT64 FeatureMatrix
    auto fmat64_path = gnn_dir_ / "test_f64.fmat";
    auto rmap64_path = gnn_dir_ / "test_f64.rmap";

    std::vector<double> features64(N * D);
    for (uint64_t r = 0; r < N; ++r) {
        for (uint64_t c = 0; c < D; ++c) {
            features64[r * D + c] = static_cast<double>((r + 1) * 1000 + (c + 1));
        }
    }

    FeatureMatrix::create(fmat64_path, N, D, GnnDtype::FLOAT64, features64.data());
    RowMapping::create(rmap64_path, node_oids_);

    auto samples = create_samples("f64_sample", {{0, 1, 2}, {3, 4, 5}});

    BatchMaterializer::Config config;
    config.reorder = false;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat64_path),
        RowMapping::open(rmap64_path),
        samples, config, db_folder_, "test_f64");

    ASSERT_EQ(result.total_batches, 2u);

    // Verify FLOAT64 features in packed batch 0: nodes {0, 1, 2}
    auto packed_dir = fs::path(result.packed_dir);
    PackedBatchReader reader(packed_dir, 2, D, GnnDtype::FLOAT64);
    auto hdr = reader.read_header(0);
    ASSERT_EQ(hdr.num_nodes, 3u);

    std::vector<double> buf(3 * D);
    reader.read_batch(0, buf.data(), buf.size() * sizeof(double));

    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < D; ++j) {
            double got = buf[i * D + j];
            double want = static_cast<double>((i + 1) * 1000 + (j + 1));
            EXPECT_DOUBLE_EQ(got, want)
                << "FLOAT64 batch 0, node " << i << ", dim " << j;
        }
    }

    // Verify file size: header(32) + 3 nodes × 4 dims × 8 bytes
    uint64_t expected_size = 32 + 3 * D * sizeof(double);
    EXPECT_EQ(fs::file_size(packed_dir / "batch_000000.bin"), expected_size);
}

// =============================================================================
// MULTIPASS_BOUNDED: Second MinHash strategy works
// =============================================================================

TEST_F(BatchMaterializerTest, MultipassBoundedStrategy) {
    auto samples = create_standard_samples("mp_sample");

    BatchMaterializer::Config config;
    config.reorder = true;
    config.minhash.strategy = MinHashReorderer::Strategy::MULTIPASS_BOUNDED;
    config.minhash.num_hashes = 2;
    config.minhash.hashes_per_pass = 2;
    config.minhash.random_seed = 42;

    auto result = BatchMaterializer::materialize(
        FeatureMatrix::open(fmat_path_),
        RowMapping::open(rmap_path_),
        samples, config, db_folder_, "test_feat");

    ASSERT_TRUE(result.reordered);
    ASSERT_EQ(result.total_batches, 3u);

    // S4 property must hold regardless of strategy
    auto samples2 = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "mp_sample"));
    auto packed_dir = fs::path(result.packed_dir);
    PackedBatchReader reader(packed_dir, 3, D, GnnDtype::FLOAT32);

    for (uint64_t bid = 0; bid < 3; ++bid) {
        auto sample = samples2.read_sample(bid);
        auto hdr = reader.read_header(bid);
        ASSERT_EQ(hdr.num_nodes, sample.all_unique_nodes.size());

        std::vector<float> packed(hdr.num_nodes * D);
        reader.read_batch(bid, packed.data(), packed.size() * sizeof(float));

        for (uint64_t i = 0; i < hdr.num_nodes; ++i) {
            uint64_t node_idx = sample.all_unique_nodes[i].id & 0x00FFFFFFFFFFFFFFULL;
            for (uint64_t j = 0; j < D; ++j) {
                EXPECT_FLOAT_EQ(packed[i * D + j], expected_feature(node_idx, j))
                    << "MULTIPASS batch " << bid << ", pos " << i << ", dim " << j;
            }
        }
    }

    // Reordered RowMapping must also be a bijection
    auto reord_rm = RowMapping::open(gnn_dir_ / "test_feat_reordered.rmap");
    std::set<uint64_t> oid_set;
    for (uint64_t i = 0; i < N; ++i) {
        oid_set.insert(reord_rm.get(i).id);
    }
    EXPECT_EQ(oid_set.size(), N);
}
