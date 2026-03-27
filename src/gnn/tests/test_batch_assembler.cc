#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <vector>

#include "gnn/training/batch_assembler.h"
#include "gnn/training/label_store.h"
#include "gnn/training/split_store.h"
#include "gnn/storage/feature_matrix.h"
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

/**
 * Synthetic graph for testing:
 *
 *   Nodes (N=6, D=3):
 *     OID 0xD4..00 → row 0 → features [101, 102, 103]
 *     OID 0xD4..01 → row 1 → features [201, 202, 203]
 *     OID 0xD4..02 → row 2 → features [301, 302, 303]
 *     OID 0xD4..03 → row 3 → features [401, 402, 403]
 *     OID 0xD4..04 → row 4 → features [501, 502, 503]
 *     OID 0xD4..05 → row 5 → features [601, 602, 603]
 *
 *   Seed labels:
 *     row 0 → label 0 (class 0)
 *     row 1 → label 1 (class 1)
 *     row 2 → label -1 (unlabeled)
 *     row 3 → label 0
 *     row 4 → label 1
 *     row 5 → label -1
 *
 *   Splits: all TRAIN (0)
 *
 *   Two-layer topology for batch 0:
 *     Layer 0 (seeds):    nodes 0, 1, 2       (indices 0..2)
 *     Layer 1 (1-hop):    nodes 3, 4, 5       (indices 0..2)
 *     edges_per_layer[0]: (src in layer1, dst in layer0)
 *       edge 0: src_idx=0 (node 3) -> dst_idx=0 (node 0)
 *       edge 1: src_idx=1 (node 4) -> dst_idx=1 (node 1)
 *       edge 2: src_idx=2 (node 5) -> dst_idx=2 (node 2)
 *
 *   all_unique_nodes = [0, 1, 2, 3, 4, 5]  (layer 0 first, then layer 1)
 *   Global indices:   node0→0, node1→1, node2→2, node3→3, node4→4, node5→5
 *
 *   Expected edge_indices[0] after global remap:
 *     row 0 (sources): [3, 4, 5]
 *     row 1 (dests):   [0, 1, 2]
 */
class BatchAssemblerTest : public GnnStorageTest {
protected:
    static constexpr uint64_t N = 6;  // total nodes
    static constexpr uint64_t D = 3;  // feature dimensions
    static constexpr uint64_t NUM_CLASSES = 2;

    std::vector<ObjectId> node_oids_;
    std::vector<float>    features_;

    fs::path db_folder_;
    fs::path gnn_dir_;
    fs::path fmat_path_;
    fs::path rmap_path_;
    fs::path labels_path_;
    fs::path splits_path_;

    void SetUp() override {
        GnnStorageTest::SetUp();

        db_folder_ = test_dir_ / "test_db";
        gnn_dir_   = db_folder_ / "gnn_data";
        fs::create_directories(gnn_dir_);

        fmat_path_   = gnn_dir_ / "features.fmat";
        rmap_path_   = gnn_dir_ / "nodes.rmap";
        labels_path_ = gnn_dir_ / "labels.bin";
        splits_path_ = gnn_dir_ / "splits.bin";

        // Build node OIDs: use a GQL-node prefix similar to other tests
        node_oids_.resize(N);
        for (uint64_t i = 0; i < N; ++i) {
            node_oids_[i] = ObjectId(0xD400000000000000ULL | i);
        }

        // Feature values: row r, col c → (r+1)*100 + (c+1)
        features_.resize(N * D);
        for (uint64_t r = 0; r < N; ++r) {
            for (uint64_t c = 0; c < D; ++c) {
                features_[r * D + c] = static_cast<float>((r + 1) * 100 + (c + 1));
            }
        }

        FeatureMatrix::create(fmat_path_, N, D, GnnDtype::FLOAT32, features_.data());
        RowMapping::create(rmap_path_, node_oids_);

        // Labels: 0, 1, -1, 0, 1, -1
        std::vector<int64_t> labels = {0, 1, -1, 0, 1, -1};
        LabelStore::write(labels_path_, labels, NUM_CLASSES);

        // Splits: all TRAIN (0)
        std::vector<uint8_t> splits(N, SplitStore::TRAIN);
        SplitStore::write(splits_path_, splits);
    }

    // -------------------------------------------------------------------
    // Helpers

    float expected_feature(uint64_t node_row, uint64_t dim) const {
        return static_cast<float>((node_row + 1) * 100 + (dim + 1));
    }

    /**
     * Build a two-layer GraphSample:
     *   Layer 0 (seeds): node_oids_[0..2]
     *   Layer 1 (1-hop): node_oids_[3..5]
     *   edges:  3->0, 4->1, 5->2  (by layer-local index: src0->dst0, etc.)
     */
    GraphSample make_two_layer_sample(uint64_t batch_id = 0,
                                      SplitType split = SplitType::TRAIN) const
    {
        GraphSample s;
        s.batch_id = batch_id;
        s.split    = split;

        // Layer 0: seeds (nodes 0, 1, 2)
        s.nodes_per_layer.resize(2);
        s.nodes_per_layer[0] = { node_oids_[0], node_oids_[1], node_oids_[2] };

        // Layer 1: 1-hop neighbors (nodes 3, 4, 5)
        s.nodes_per_layer[1] = { node_oids_[3], node_oids_[4], node_oids_[5] };

        // Edges: local src in layer1 -> local dst in layer0
        s.edges_per_layer.resize(1);
        LayerEdges& le = s.edges_per_layer[0];
        le.src_indices = { 0, 1, 2 };   // nodes_per_layer[1][0,1,2] = nodes 3,4,5
        le.dst_indices = { 0, 1, 2 };   // nodes_per_layer[0][0,1,2] = nodes 0,1,2
        le.edge_ids    = { ObjectId(0), ObjectId(0), ObjectId(0) };  // dummy

        // all_unique_nodes: layer 0 first, then layer 1 (deduplication)
        s.all_unique_nodes = {
            node_oids_[0], node_oids_[1], node_oids_[2],
            node_oids_[3], node_oids_[4], node_oids_[5]
        };

        return s;
    }

    /**
     * Build a one-layer GraphSample (no edges, just seed nodes 0 and 1).
     * Used for simpler label/mask tests.
     */
    GraphSample make_one_layer_sample(uint64_t batch_id = 0) const {
        GraphSample s;
        s.batch_id = batch_id;
        s.split    = SplitType::TRAIN;

        s.nodes_per_layer.resize(1);
        s.nodes_per_layer[0] = { node_oids_[0], node_oids_[1], node_oids_[2] };

        // no edges (0-hop)
        s.all_unique_nodes = { node_oids_[0], node_oids_[1], node_oids_[2] };

        return s;
    }

    /**
     * Create a SampleStorage with the two-layer sample stored at batch_id=0.
     */
    SampleStorage create_sample_storage(const std::string& name = "test_samples") {
        SamplingConfig config;
        config.projection_name = "test_proj";
        config.sample_name     = name;
        config.fanouts         = {3};  // 1-hop, fanout 3
        config.batch_size      = 3;
        config.train_ratio     = 1.0;
        config.val_ratio       = 0.0;
        config.test_ratio      = 0.0;

        auto storage = SampleStorage::create(db_folder_, config);
        storage.write_sample(make_two_layer_sample(0));
        storage.finalize();

        return SampleStorage::open(SampleStorage::get_storage_path(db_folder_, name));
    }
};

// =============================================================================
// Test 1: AssembleProducesCorrectShapes
// =============================================================================

TEST_F(BatchAssemblerTest, AssembleProducesCorrectShapes) {
    auto fm  = FeatureMatrix::open(fmat_path_);
    auto rm  = RowMapping::open(rmap_path_);
    auto ls  = LabelStore::open(labels_path_);
    auto ss  = SplitStore::open(splits_path_);
    auto storage = create_sample_storage();

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    MiniBatch batch = assembler.assemble(0);

    // features: [N_all, D] = [6, 3]
    ASSERT_TRUE(batch.features.defined());
    ASSERT_EQ(batch.features.dim(), 2);
    EXPECT_EQ(batch.features.size(0), 6);  // all 6 unique nodes
    EXPECT_EQ(batch.features.size(1), static_cast<int64_t>(D));
    EXPECT_EQ(batch.features.scalar_type(), torch::kFloat32);

    // edge_indices: 1 layer, each [2, 3]
    ASSERT_EQ(batch.edge_indices.size(), 1u);
    EXPECT_EQ(batch.edge_indices[0].size(0), 2);
    EXPECT_EQ(batch.edge_indices[0].size(1), 3);  // 3 edges

    // labels: [num_seeds] = [3]
    ASSERT_TRUE(batch.labels.defined());
    EXPECT_EQ(batch.labels.size(0), 3);
    EXPECT_EQ(batch.labels.scalar_type(), torch::kInt64);

    // label_mask: [3] bool
    ASSERT_TRUE(batch.label_mask.defined());
    EXPECT_EQ(batch.label_mask.size(0), 3);

    // metadata
    EXPECT_EQ(batch.num_seeds, 3u);
    EXPECT_EQ(batch.num_nodes, 6u);
    EXPECT_EQ(batch.batch_id, 0u);
    EXPECT_EQ(batch.split, SplitType::TRAIN);
}

// =============================================================================
// Test 2: EdgeIndicesAreGloballyRemapped
// =============================================================================

TEST_F(BatchAssemblerTest, EdgeIndicesAreGloballyRemapped) {
    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto storage = create_sample_storage();

    // No labels needed for this test
    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    MiniBatch batch = assembler.assemble(0);

    ASSERT_EQ(batch.edge_indices.size(), 1u);
    auto ei = batch.edge_indices[0];  // [2, 3]
    ASSERT_EQ(ei.dim(), 2);
    ASSERT_EQ(ei.size(0), 2);
    ASSERT_EQ(ei.size(1), 3);

    // Row 0 = sources (layer 1 nodes: 3, 4, 5) → global indices 3, 4, 5
    // Row 1 = dests   (layer 0 nodes: 0, 1, 2) → global indices 0, 1, 2
    auto acc = ei.accessor<int64_t, 2>();

    // Sources must be valid global indices [0, 6)
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_GE(acc[0][i], 0) << "source index negative at edge " << i;
        EXPECT_LT(acc[0][i], 6) << "source index out of range at edge " << i;
    }

    // Destinations must be valid global indices [0, 6)
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_GE(acc[1][i], 0) << "dest index negative at edge " << i;
        EXPECT_LT(acc[1][i], 6) << "dest index out of range at edge " << i;
    }

    // Verify exact remapping:
    //   edge 0: src=layer1[0]=node3 → global 3, dst=layer0[0]=node0 → global 0
    //   edge 1: src=layer1[1]=node4 → global 4, dst=layer0[1]=node1 → global 1
    //   edge 2: src=layer1[2]=node5 → global 5, dst=layer0[2]=node2 → global 2
    EXPECT_EQ(acc[0][0], 3);
    EXPECT_EQ(acc[0][1], 4);
    EXPECT_EQ(acc[0][2], 5);
    EXPECT_EQ(acc[1][0], 0);
    EXPECT_EQ(acc[1][1], 1);
    EXPECT_EQ(acc[1][2], 2);
}

// =============================================================================
// Test 3: LabelsMatchSeedNodes
// =============================================================================

TEST_F(BatchAssemblerTest, LabelsMatchSeedNodes) {
    auto fm  = FeatureMatrix::open(fmat_path_);
    auto rm  = RowMapping::open(rmap_path_);
    auto ls  = LabelStore::open(labels_path_);
    auto storage = create_sample_storage();

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);
    MiniBatch batch = assembler.assemble(0);

    // Seed nodes 0, 1, 2 → labels 0, 1, -1
    ASSERT_EQ(batch.labels.size(0), 3);
    auto label_acc = batch.labels.accessor<int64_t, 1>();
    EXPECT_EQ(label_acc[0], 0);
    EXPECT_EQ(label_acc[1], 1);
    EXPECT_EQ(label_acc[2], -1);
}

// =============================================================================
// Test 4: NullLabelsProducesZerosAndEmptyMask
// =============================================================================

TEST_F(BatchAssemblerTest, NullLabelsProducesZerosAndEmptyMask) {
    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto storage = create_sample_storage();

    // Pass nullptr for labels — unsupervised path
    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    MiniBatch batch = assembler.assemble(0);

    // labels: all zeros
    ASSERT_EQ(batch.labels.size(0), 3);
    EXPECT_TRUE(batch.labels.equal(torch::zeros({3}, torch::kInt64)));

    // label_mask: all false
    ASSERT_EQ(batch.label_mask.size(0), 3);
    EXPECT_TRUE(batch.label_mask.equal(torch::zeros({3}, torch::kBool)));
}

// =============================================================================
// Test 5: LabelMaskCorrectlyMasksMinusOne
// =============================================================================

TEST_F(BatchAssemblerTest, LabelMaskCorrectlyMasksMinusOne) {
    auto fm  = FeatureMatrix::open(fmat_path_);
    auto rm  = RowMapping::open(rmap_path_);
    auto ls  = LabelStore::open(labels_path_);
    auto storage = create_sample_storage();

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);
    MiniBatch batch = assembler.assemble(0);

    // Seed nodes 0, 1, 2 → labels 0, 1, -1
    // mask should be: true, true, false
    ASSERT_EQ(batch.label_mask.size(0), 3);
    auto mask_acc = batch.label_mask.accessor<bool, 1>();
    EXPECT_TRUE(mask_acc[0]);   // label 0 → labeled
    EXPECT_TRUE(mask_acc[1]);   // label 1 → labeled
    EXPECT_FALSE(mask_acc[2]);  // label -1 → unlabeled
}

// =============================================================================
// Test 6: FeaturesHaveCorrectValues
// =============================================================================

TEST_F(BatchAssemblerTest, FeaturesHaveCorrectValues) {
    auto fm  = FeatureMatrix::open(fmat_path_);
    auto rm  = RowMapping::open(rmap_path_);
    auto storage = create_sample_storage();

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    MiniBatch batch = assembler.assemble(0);

    // all_unique_nodes = [node0, node1, node2, node3, node4, node5]
    // features[i] should match expected_feature(i, j)
    auto f = batch.features.accessor<float, 2>();
    for (int64_t r = 0; r < 6; ++r) {
        for (int64_t c = 0; c < static_cast<int64_t>(D); ++c) {
            float expected = expected_feature(static_cast<uint64_t>(r), static_cast<uint64_t>(c));
            EXPECT_FLOAT_EQ(f[r][c], expected)
                << "Feature mismatch at node " << r << ", dim " << c;
        }
    }
}

// =============================================================================
// Test 7: AssembleFromSampleBypassesStorage
// =============================================================================

TEST_F(BatchAssemblerTest, AssembleFromSampleBypassesStorage) {
    // Use assemble_from_sample() to skip SampleStorage setup entirely.
    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto storage = create_sample_storage();  // still needed for constructor

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);

    GraphSample sample = make_two_layer_sample(42, SplitType::VALIDATION);
    MiniBatch batch = assembler.assemble_from_sample(sample);

    EXPECT_EQ(batch.batch_id, 42u);
    EXPECT_EQ(batch.split, SplitType::VALIDATION);
    EXPECT_EQ(batch.num_seeds, 3u);
    EXPECT_EQ(batch.num_nodes, 6u);
    ASSERT_EQ(batch.edge_indices.size(), 1u);
    EXPECT_EQ(batch.edge_indices[0].size(1), 3);
}

// =============================================================================
// Test 8: EmptyEdgesLayer
// =============================================================================

TEST_F(BatchAssemblerTest, EmptyEdgesLayer) {
    // A sample with zero edges in the single layer.
    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto storage = create_sample_storage();

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);

    GraphSample s;
    s.batch_id = 0;
    s.split    = SplitType::TRAIN;
    s.nodes_per_layer.resize(2);
    s.nodes_per_layer[0] = { node_oids_[0] };
    s.nodes_per_layer[1] = { node_oids_[1] };
    s.edges_per_layer.resize(1);
    // Leave layer_edges empty
    s.all_unique_nodes = { node_oids_[0], node_oids_[1] };

    MiniBatch batch = assembler.assemble_from_sample(s);

    ASSERT_EQ(batch.edge_indices.size(), 1u);
    EXPECT_EQ(batch.edge_indices[0].size(0), 2);
    EXPECT_EQ(batch.edge_indices[0].size(1), 0);  // 0 edges
    EXPECT_EQ(batch.num_seeds, 1u);
    EXPECT_EQ(batch.num_nodes, 2u);
}
