#pragma once

// =============================================================================
// Shared fixture for TrainingLoop tests.
//
// Extracted from test_training_loop.cc so that multiple test translation units
// (test_training_loop.cc and test_training_loop_resume.cc) can share identical
// setup without duplicating ~200 lines of harness code.
//
// Consumers should simply:
//     #include "gnn/tests/training_loop_test_fixture.h"
//     using TrainingLoopResumeTest = TrainingLoopTestFixture;  // or TEST_F directly
// =============================================================================

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gnn/models/graphsage_model.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_config.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/batch_assembler.h"
#include "gnn/training/label_store.h"
#include "gnn/training/split_store.h"
#include "gnn/training/training_loop.h"
#include "gnn/tests/test_helpers.h"
#include "graph_models/object_id.h"

namespace mdb::gnn::testing_util {

// =============================================================================
// Synthetic graph (N=8 nodes, D=4 features, 2 classes):
//
//   Class 0 nodes (OIDs 0..3): features are large positive values (+10..+40).
//   Class 1 nodes (OIDs 4..7): features are large negative values (-10..-40).
//
// This deliberate separation makes the two classes linearly separable so a
// small GraphSAGE model trained for a handful of epochs can clearly decrease
// cross-entropy loss.
//
// Batch layout (2 train batches, 1 validation batch, no test batches):
//   Train  batch 0: seeds = nodes 0, 1, 2, 3   (all class 0)
//   Train  batch 1: seeds = nodes 4, 5, 6, 7   (all class 1)
//   Val    batch 2: seeds = nodes 0, 2, 4, 6   (2 from each class)
//
// Each batch is a 0-layer GraphSample (self-loop) to keep the fixture simple.
// =============================================================================

class TrainingLoopTestFixture : public ::GnnStorageTest {
protected:
    static constexpr uint64_t N           = 8;
    static constexpr uint64_t D           = 4;
    static constexpr uint64_t NUM_CLASSES = 2;

    // Batch layout
    static constexpr uint64_t NUM_TRAIN_BATCHES = 2;
    static constexpr uint64_t NUM_VAL_BATCHES   = 1;

    std::vector<ObjectId>   node_oids_;
    std::filesystem::path   db_folder_;
    std::filesystem::path   gnn_dir_;
    std::filesystem::path   fmat_path_;
    std::filesystem::path   rmap_path_;
    std::filesystem::path   labels_path_;
    std::filesystem::path   splits_path_;

    void SetUp() override {
        ::GnnStorageTest::SetUp();

        db_folder_ = test_dir_ / "train_db";
        gnn_dir_   = db_folder_ / "gnn_data";
        std::filesystem::create_directories(gnn_dir_);

        fmat_path_   = gnn_dir_ / "features.fmat";
        rmap_path_   = gnn_dir_ / "nodes.rmap";
        labels_path_ = gnn_dir_ / "labels.bin";
        splits_path_ = gnn_dir_ / "splits.bin";

        // Build node OIDs
        node_oids_.resize(N);
        for (uint64_t i = 0; i < N; ++i) {
            node_oids_[i] = ObjectId(0xD400000000000000ULL | i);
        }

        // Features: class 0 (i=0..3) -> all-positive, class 1 (i=4..7) -> all-negative.
        // Large magnitude (±100 per dim) makes the two classes easily separable.
        std::vector<float> features(N * D);
        for (uint64_t r = 0; r < N; ++r) {
            float sign = (r < 4) ? +1.0f : -1.0f;
            for (uint64_t c = 0; c < D; ++c) {
                features[r * D + c] = sign * static_cast<float>((c + 1) * 100);
            }
        }
        FeatureMatrix::create(fmat_path_, N, D, GnnDtype::FLOAT32, features.data());
        RowMapping::create(rmap_path_, node_oids_);

        // Labels: nodes 0..3 → class 0, nodes 4..7 → class 1
        std::vector<int64_t> labels(N);
        for (uint64_t i = 0; i < N; ++i) {
            labels[i] = (i < 4) ? 0 : 1;
        }
        LabelStore::write(labels_path_, labels, NUM_CLASSES);

        // Splits: all TRAIN (the SplitStore is only used by BatchAssembler for
        // per-node split metadata; the TrainingLoop uses SampleCatalog counters).
        std::vector<uint8_t> splits(N, SplitStore::TRAIN);
        SplitStore::write(splits_path_, splits);
    }

    // -----------------------------------------------------------------------
    // Build a one-layer GraphSample where seeds are self-connected (identity
    // edges).  This produces exactly 1 edge_index tensor in the MiniBatch,
    // matching num_layers=1 in GraphSAGEModel.
    //
    // Layout:
    //   nodes_per_layer[0] = seeds        (layer 0: target nodes)
    //   nodes_per_layer[1] = seeds        (layer 1: same nodes as "neighbors")
    //   edges_per_layer[0]: each seed i -> itself (self-loop in global index)
    //   all_unique_nodes   = seeds (de-duplicated: seeds appear once)
    // -----------------------------------------------------------------------
    GraphSample make_seed_sample(
        const std::vector<ObjectId>& seeds,
        uint64_t batch_id,
        SplitType split = SplitType::TRAIN
    ) const
    {
        GraphSample s;
        s.batch_id = batch_id;
        s.split    = split;

        const size_t M = seeds.size();

        // Layer 0: seed nodes (targets)
        // Layer 1: same seeds acting as their own 1-hop neighbors
        s.nodes_per_layer.resize(2);
        s.nodes_per_layer[0] = seeds;
        s.nodes_per_layer[1] = seeds;

        // Self-loop edges: layer1[i] -> layer0[i]  (local indices)
        s.edges_per_layer.resize(1);
        LayerEdges& le = s.edges_per_layer[0];
        le.src_indices.resize(M);
        le.dst_indices.resize(M);
        le.edge_ids.resize(M, ObjectId(0));
        for (size_t i = 0; i < M; ++i) {
            le.src_indices[i] = static_cast<int32_t>(i);
            le.dst_indices[i] = static_cast<int32_t>(i);
        }

        // all_unique_nodes: seeds only (layer 0 and layer 1 are the same nodes)
        s.all_unique_nodes = seeds;
        return s;
    }

    // -----------------------------------------------------------------------
    // Create SampleStorage with 3 batches (2 train + 1 val).
    // Returns the SampleCatalog that TrainingLoop needs.
    // -----------------------------------------------------------------------
    SampleCatalog create_sample_storage(const std::string& name = "test_samples")
    {
        SamplingConfig config;
        config.projection_name = "test_proj";
        config.sample_name     = name;
        config.fanouts         = {};      // 0-hop
        config.batch_size      = 4;
        config.train_ratio     = 2.0 / 3.0;
        config.val_ratio       = 1.0 / 3.0;
        config.test_ratio      = 0.0;

        auto storage = SampleStorage::create(db_folder_, config);

        // Batch 0: train — nodes 0, 1, 2, 3 (class 0)
        std::vector<ObjectId> train0 = {
            node_oids_[0], node_oids_[1], node_oids_[2], node_oids_[3]
        };
        storage.write_sample(make_seed_sample(train0, 0, SplitType::TRAIN));

        // Batch 1: train — nodes 4, 5, 6, 7 (class 1)
        std::vector<ObjectId> train1 = {
            node_oids_[4], node_oids_[5], node_oids_[6], node_oids_[7]
        };
        storage.write_sample(make_seed_sample(train1, 1, SplitType::TRAIN));

        // Batch 2: validation — 2 from each class
        std::vector<ObjectId> val0 = {
            node_oids_[0], node_oids_[2], node_oids_[4], node_oids_[6]
        };
        storage.write_sample(make_seed_sample(val0, 2, SplitType::VALIDATION));

        storage.finalize();

        // Build the catalog manually since SampleStorage::create/finalize
        // sets train_batches from the ratio, not from explicit counts.
        // We mirror what the assembler will see: batches 0..1 = train, batch 2 = val.
        SampleCatalog cat;
        cat.projection_name    = "test_proj";
        cat.sample_name        = name;
        cat.batch_size         = 4;
        cat.random_seed        = 0;
        cat.fanouts            = {};
        cat.total_batches      = 3;
        cat.train_batches      = NUM_TRAIN_BATCHES;
        cat.validation_batches = NUM_VAL_BATCHES;
        cat.test_batches       = 0;
        cat.unique_nodes       = N;
        cat.total_edges        = 0;

        return cat;
    }
};

} // namespace mdb::gnn::testing_util

// Legacy public name used by existing tests.
using TrainingLoopTest = mdb::gnn::testing_util::TrainingLoopTestFixture;
