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
// Nested-aggregation edge wiring (2026-06-02)
//
// 3-layer sample (2 edge layers) over the 6-node fixture:
//   Layer 0 (seeds): node0, node1     (global pos 0,1; A_0 = {0,1})
//   Layer 1 (1-hop): node2, node3     (A_1 = {0,1,2,3})
//   Layer 2 (2-hop): node4, node5     (A_2 = {0,1,2,3,4,5})
//   E_0 (edges_per_layer[0]): node2->node0, node3->node1   (1-hop -> seed)
//   E_1 (edges_per_layer[1]): node4->node2, node5->node3   (2-hop -> 1-hop)
//
// LEGACY: edge_index[1] = E_1 only (2 edges, dst in {node2,node3} = A_1-local
//         {2,3}); the seeds do NOT aggregate at conv 1.
// NESTED: edge_index[1] = E_0 ∪ E_1 (4 edges); the seeds (A_1-local {0,1}) now
//         aggregate at conv 1 too — the standard nested-neighbourhood wiring.
// edge_index[0] is identical in both (k=0 → only E_0).
// =============================================================================

static GraphSample make_three_layer_sample(const std::vector<ObjectId>& n) {
    GraphSample s;
    s.batch_id = 0;
    s.split    = SplitType::TRAIN;
    s.nodes_per_layer.resize(3);
    s.nodes_per_layer[0] = { n[0], n[1] };   // seeds
    s.nodes_per_layer[1] = { n[2], n[3] };   // 1-hop
    s.nodes_per_layer[2] = { n[4], n[5] };   // 2-hop
    s.edges_per_layer.resize(2);
    s.edges_per_layer[0].src_indices = { 0, 1 };  // layer1[0,1] = n2,n3
    s.edges_per_layer[0].dst_indices = { 0, 1 };  // layer0[0,1] = n0,n1
    s.edges_per_layer[0].edge_ids    = { ObjectId(0), ObjectId(0) };
    s.edges_per_layer[1].src_indices = { 0, 1 };  // layer2[0,1] = n4,n5
    s.edges_per_layer[1].dst_indices = { 0, 1 };  // layer1[0,1] = n2,n3
    s.edges_per_layer[1].edge_ids    = { ObjectId(0), ObjectId(0) };
    s.all_unique_nodes = { n[0], n[1], n[2], n[3], n[4], n[5] };
    return s;
}

TEST_F(BatchAssemblerTest, LegacyEdgeWiringIsPerHop) {
    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto storage = create_sample_storage();

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    assembler.set_nested_aggregation(false);
    MiniBatch batch = assembler.assemble_from_sample(make_three_layer_sample(node_oids_));

    ASSERT_EQ(batch.edge_indices.size(), 2u);
    EXPECT_EQ(batch.edge_indices[0].size(1), 2);  // E_0
    EXPECT_EQ(batch.edge_indices[1].size(1), 2);  // E_1 ONLY (no seed aggregation)

    // conv-1 dst are A_1-local indices of the 1-hop nodes {node2,node3} = {2,3};
    // the seeds (A_1-local {0,1}) are absent.
    auto acc = batch.edge_indices[1].accessor<int64_t, 2>();
    std::vector<int64_t> dsts = { acc[1][0], acc[1][1] };
    std::sort(dsts.begin(), dsts.end());
    EXPECT_EQ(dsts, (std::vector<int64_t>{2, 3}));
}

TEST_F(BatchAssemblerTest, NestedEdgeWiringIsCumulativeAndIncludesSeeds) {
    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto storage = create_sample_storage();

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    assembler.set_nested_aggregation(true);
    MiniBatch batch = assembler.assemble_from_sample(make_three_layer_sample(node_oids_));

    ASSERT_EQ(batch.edge_indices.size(), 2u);
    EXPECT_EQ(batch.edge_indices[0].size(1), 2);  // k=0 unchanged: only E_0
    EXPECT_EQ(batch.edge_indices[1].size(1), 4);  // E_0 ∪ E_1

    // conv-1 dst must now include the seeds (A_1-local {0,1}) AND the 1-hop
    // nodes ({2,3}) — every node within 1 hop re-aggregates at conv 1.
    auto acc = batch.edge_indices[1].accessor<int64_t, 2>();
    std::vector<int64_t> dsts, srcs;
    for (int64_t i = 0; i < 4; ++i) { dsts.push_back(acc[1][i]); srcs.push_back(acc[0][i]); }
    std::sort(dsts.begin(), dsts.end());
    EXPECT_EQ(dsts, (std::vector<int64_t>{0, 1, 2, 3}));
    // every src must be a valid A_2-local index [0,6)
    for (int64_t v : srcs) { EXPECT_GE(v, 0); EXPECT_LT(v, 6); }

    // edge_index[0] (conv 0, seeds) is identical to legacy: src=1-hop {2,3}, dst=seeds {0,1}.
    auto acc0 = batch.edge_indices[0].accessor<int64_t, 2>();
    std::vector<int64_t> d0 = { acc0[1][0], acc0[1][1] };
    std::sort(d0.begin(), d0.end());
    EXPECT_EQ(d0, (std::vector<int64_t>{0, 1}));
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

// Tests for active-set computation (active-set-shrinking refactor).
TEST_F(BatchAssemblerTest, ActiveIndicesAreCumulative) {
    // 4-layer sample manually constructed:
    //   nodes_per_layer[0] = [s1, s2]            (seeds)
    //   nodes_per_layer[1] = [n3, n4, n5]        (1-hop frontier)
    //   nodes_per_layer[2] = [n6, n7]            (2-hop frontier)
    //   nodes_per_layer[3] = [n8, n9, n10]       (3-hop frontier; deepest)
    //
    // After rebuild_unique_nodes (layer order, dedup):
    //   all_unique_nodes = [s1, s2, n3, n4, n5, n6, n7, n8, n9, n10]
    //   positions:         [0, 1,  2,  3,  4,  5,  6,  7,  8,  9 ]
    //
    // Expected active sets (cumulative):
    //   A_0 = {s1, s2}                     positions [0, 1]
    //   A_1 = A_0 ∪ {n3,n4,n5}             positions [0, 1, 2, 3, 4]
    //   A_2 = A_1 ∪ {n6, n7}               positions [0, 1, 2, 3, 4, 5, 6]
    //   A_3 = A_2 ∪ {n8, n9, n10}          positions [0..9] (all)
    GraphSample sample;
    sample.batch_id = 0;
    sample.split    = SplitType::TRAIN;
    sample.nodes_per_layer = {
        { ObjectId(1), ObjectId(2) },
        { ObjectId(3), ObjectId(4), ObjectId(5) },
        { ObjectId(6), ObjectId(7) },
        { ObjectId(8), ObjectId(9), ObjectId(10) }
    };
    sample.edges_per_layer.resize(3);  // 3 edge layers, can be empty for this test
    sample.rebuild_unique_nodes();
    ASSERT_EQ(sample.all_unique_nodes.size(), 10u);

    // Bypass storage: build an assembler with the FeatureMatrix path. The
    // assemble_from_sample call does NOT need features for the active-set
    // logic, but load_features needs a RowMapping that contains the OIDs.
    // To keep this test focused, use a synthetic feature setup matching
    // the OIDs we just used.
    fs::path local_fmat = gnn_dir_ / "active_test.fmat";
    fs::path local_rmap = gnn_dir_ / "active_test.rmap";
    constexpr uint64_t LN = 10;
    constexpr uint64_t LD = 1;
    std::vector<ObjectId> local_oids;
    local_oids.reserve(LN);
    for (uint64_t i = 1; i <= LN; ++i) local_oids.emplace_back(i);
    std::vector<float> local_feats(LN * LD, 0.0f);
    FeatureMatrix::create(local_fmat, LN, LD, GnnDtype::FLOAT32, local_feats.data());
    RowMapping::create(local_rmap, local_oids);
    auto fm = FeatureMatrix::open(local_fmat);
    auto rm = RowMapping::open(local_rmap);
    auto storage = create_sample_storage();
    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);

    auto mini = assembler.assemble_from_sample(sample);

    ASSERT_EQ(mini.active_indices_per_layer.size(), 4u);
    ASSERT_EQ(mini.active_sizes_per_layer.size(),   4u);

    auto check = [&](size_t k, std::vector<int64_t> expected) {
        const auto& t = mini.active_indices_per_layer[k];
        ASSERT_EQ(t.dim(), 1);
        ASSERT_EQ(t.scalar_type(), torch::kInt64);
        ASSERT_EQ(t.size(0), (int64_t)expected.size());
        ASSERT_EQ(mini.active_sizes_per_layer[k], (int64_t)expected.size());
        auto acc = t.accessor<int64_t, 1>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(acc[i], expected[i]) << "k=" << k << " i=" << i;
        }
    };
    check(0, {0, 1});
    check(1, {0, 1, 2, 3, 4});
    check(2, {0, 1, 2, 3, 4, 5, 6});
    check(3, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
}

// Tests for Task 3: edges remapped to local active-set indices.
TEST_F(BatchAssemblerTest, EdgeIndicesAreLocalToActiveSets) {
    // 3-layer sample, concrete edges. Seeds = [s1], 1-hop = [n2,n3], 2-hop = [n4].
    //
    //   nodes_per_layer[0] = [s1]              global pos 0
    //   nodes_per_layer[1] = [n2, n3]          global pos 1, 2
    //   nodes_per_layer[2] = [n4]              global pos 3
    //   nodes_per_layer[3] = []                (empty deepest)
    //
    //   edges_per_layer[0]: src in layer 1, dst in layer 0
    //                       (n2, s1), (n3, s1)
    //   edges_per_layer[1]: src in layer 2, dst in layer 1
    //                       (n4, n2)
    //   edges_per_layer[2]: empty
    //
    // Active sets (cumulative):
    //   A_0 = [s1]                 = local indices [0]
    //   A_1 = [s1, n2, n3]         = local indices [0, 1, 2]
    //   A_2 = [s1, n2, n3, n4]     = local indices [0, 1, 2, 3]
    //   A_3 = same as A_2 (layer 3 empty)
    //
    // Expected edge_indices (LOCAL):
    //   edge_indices[0]: src local in A_1 = [1, 2] (n2, n3); dst local in A_0 = [0, 0] (s1, s1)
    //   edge_indices[1]: src local in A_2 = [3] (n4);        dst local in A_1 = [1] (n2)
    //   edge_indices[2]: empty
    GraphSample sample;
    sample.batch_id = 0;
    sample.split    = SplitType::TRAIN;
    sample.nodes_per_layer = {
        { ObjectId(1) },                  // s1
        { ObjectId(2), ObjectId(3) },     // n2, n3
        { ObjectId(4) },                  // n4
        {}                                // empty deepest layer
    };
    sample.edges_per_layer.resize(3);
    // edges_per_layer[0]: layer 1 -> layer 0
    sample.edges_per_layer[0].src_indices = {0, 1};  // local in layer 1: n2, n3
    sample.edges_per_layer[0].dst_indices = {0, 0};  // local in layer 0: s1, s1
    sample.edges_per_layer[0].edge_ids = { ObjectId(100), ObjectId(101) };
    // edges_per_layer[1]: layer 2 -> layer 1
    sample.edges_per_layer[1].src_indices = {0};     // local in layer 2: n4
    sample.edges_per_layer[1].dst_indices = {0};     // local in layer 1: n2
    sample.edges_per_layer[1].edge_ids = { ObjectId(200) };
    // edges_per_layer[2]: empty
    sample.rebuild_unique_nodes();
    ASSERT_EQ(sample.all_unique_nodes.size(), 4u);

    // Synthetic FeatureMatrix/RowMapping with the OIDs 1..4 used above.
    fs::path local_fmat = gnn_dir_ / "edge_local_test.fmat";
    fs::path local_rmap = gnn_dir_ / "edge_local_test.rmap";
    constexpr uint64_t LN = 4;
    constexpr uint64_t LD = 1;
    std::vector<ObjectId> local_oids;
    local_oids.reserve(LN);
    for (uint64_t i = 1; i <= LN; ++i) local_oids.emplace_back(i);
    std::vector<float> local_feats(LN * LD, 0.0f);
    FeatureMatrix::create(local_fmat, LN, LD, GnnDtype::FLOAT32, local_feats.data());
    RowMapping::create(local_rmap, local_oids);
    auto fm = FeatureMatrix::open(local_fmat);
    auto rm = RowMapping::open(local_rmap);
    auto storage = create_sample_storage();
    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);

    auto mini = assembler.assemble_from_sample(sample);

    ASSERT_EQ(mini.edge_indices.size(), 3u);

    // edge_indices[0] should be src=[1, 2] (local A_1), dst=[0, 0] (local A_0)
    auto e0 = mini.edge_indices[0];
    ASSERT_EQ(e0.size(0), 2);  // 2 rows (src + dst)
    ASSERT_EQ(e0.size(1), 2);  // 2 edges
    auto a0 = e0.accessor<int64_t, 2>();
    EXPECT_EQ(a0[0][0], 1);  // src for edge 0: local pos of n2 in A_1
    EXPECT_EQ(a0[0][1], 2);  // src for edge 1: local pos of n3 in A_1
    EXPECT_EQ(a0[1][0], 0);  // dst for edge 0: local pos of s1 in A_0
    EXPECT_EQ(a0[1][1], 0);  // dst for edge 1: local pos of s1 in A_0

    // edge_indices[1] should be src=[3] (local A_2), dst=[1] (local A_1)
    auto e1 = mini.edge_indices[1];
    ASSERT_EQ(e1.size(0), 2);
    ASSERT_EQ(e1.size(1), 1);
    auto a1 = e1.accessor<int64_t, 2>();
    EXPECT_EQ(a1[0][0], 3);  // n4 local in A_2
    EXPECT_EQ(a1[1][0], 1);  // n2 local in A_1

    // edge_indices[2] empty
    auto e2 = mini.edge_indices[2];
    ASSERT_EQ(e2.size(0), 2);
    ASSERT_EQ(e2.size(1), 0);
}

// =============================================================================
// Structural per-batch cache (train hot path)
//
// A cache hit must produce structurally-identical tensors to a fresh build —
// the cache only avoids recomputing the deterministic index/label build, never
// changes content. Features are still re-loaded each call.
// =============================================================================

TEST_F(BatchAssemblerTest, StructCacheMatchesFreshBuild) {
    auto fm  = FeatureMatrix::open(fmat_path_);
    auto rm  = RowMapping::open(rmap_path_);
    auto ls  = LabelStore::open(labels_path_);
    auto ss  = SplitStore::open(splits_path_);
    auto storage = create_sample_storage();

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    // Reference build with the cache disabled.
    MiniBatch ref = assembler.assemble(0);
    {
        auto s = assembler.struct_cache_stats();
        EXPECT_EQ(s.budget, 0u);
        EXPECT_EQ(s.hits, 0u);
        EXPECT_EQ(s.misses, 0u);
    }

    // Enable the cache: first assemble misses + caches, second hits.
    assembler.set_struct_cache_budget_bytes(8 * 1024 * 1024);
    MiniBatch miss = assembler.assemble(0);
    MiniBatch hit  = assembler.assemble(0);

    auto s = assembler.struct_cache_stats();
    EXPECT_EQ(s.misses, 1u);
    EXPECT_EQ(s.hits, 1u);
    EXPECT_EQ(s.entries, 1u);
    EXPECT_GT(s.bytes, 0u);

    // Structural content identical across ref / miss / hit.
    ASSERT_EQ(hit.edge_indices.size(), ref.edge_indices.size());
    for (size_t k = 0; k < ref.edge_indices.size(); ++k) {
        EXPECT_TRUE(torch::equal(hit.edge_indices[k], ref.edge_indices[k]));
        EXPECT_TRUE(torch::equal(miss.edge_indices[k], ref.edge_indices[k]));
    }
    ASSERT_EQ(hit.active_indices_per_layer.size(), ref.active_indices_per_layer.size());
    for (size_t k = 0; k < ref.active_indices_per_layer.size(); ++k) {
        EXPECT_TRUE(torch::equal(hit.active_indices_per_layer[k],
                                 ref.active_indices_per_layer[k]));
    }
    EXPECT_EQ(hit.active_sizes_per_layer, ref.active_sizes_per_layer);
    EXPECT_TRUE(torch::equal(hit.labels, ref.labels));
    EXPECT_TRUE(torch::equal(hit.label_mask, ref.label_mask));
    EXPECT_EQ(hit.num_seeds, ref.num_seeds);
    EXPECT_EQ(hit.num_nodes, ref.num_nodes);
    EXPECT_EQ(hit.num_labeled, ref.num_labeled);

    // Features are still produced on a hit (re-loaded, not cached).
    ASSERT_TRUE(hit.features.defined());
    EXPECT_TRUE(torch::equal(hit.features, ref.features));

    // Disable clears.
    assembler.set_struct_cache_budget_bytes(0);
    EXPECT_EQ(assembler.struct_cache_stats().entries, 0u);
}
