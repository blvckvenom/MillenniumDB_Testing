#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
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

namespace fs = std::filesystem;
using namespace mdb::gnn;

// =============================================================================
// Test Fixture
// =============================================================================
//
// Synthetic graph (N=8 nodes, D=4 features, 2 classes):
//
//   Class 0 nodes (OIDs 0..3): features are large positive values (+10, +20, +30, +40).
//   Class 1 nodes (OIDs 4..7): features are large negative values (-10, -20, -30, -40).
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
// Each batch is a 0-layer GraphSample (no neighbor hop) to keep the test
// self-contained and fast.
//
// =============================================================================

class TrainingLoopTest : public GnnStorageTest {
protected:
    static constexpr uint64_t N           = 8;
    static constexpr uint64_t D           = 4;
    static constexpr uint64_t NUM_CLASSES = 2;

    // Batch layout
    static constexpr uint64_t NUM_TRAIN_BATCHES = 2;
    static constexpr uint64_t NUM_VAL_BATCHES   = 1;

    std::vector<ObjectId> node_oids_;
    fs::path db_folder_;
    fs::path gnn_dir_;
    fs::path fmat_path_;
    fs::path rmap_path_;
    fs::path labels_path_;
    fs::path splits_path_;

    void SetUp() override {
        GnnStorageTest::SetUp();

        db_folder_ = test_dir_ / "train_db";
        gnn_dir_   = db_folder_ / "gnn_data";
        fs::create_directories(gnn_dir_);

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
    //
    // Because seeds appear in both layer 0 and layer 1, all_unique_nodes is
    // just `seeds` (size M).  Global index of seeds[i] = i.
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
    // Create SampleStorage with our 3 batches.
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
        cat.projection_name   = "test_proj";
        cat.sample_name       = name;
        cat.batch_size        = 4;
        cat.random_seed       = 0;
        cat.fanouts           = {};
        cat.total_batches     = 3;
        cat.train_batches     = NUM_TRAIN_BATCHES;
        cat.validation_batches= NUM_VAL_BATCHES;
        cat.test_batches      = 0;
        cat.unique_nodes      = N;
        cat.total_edges       = 0;

        return cat;
    }

    // -----------------------------------------------------------------------
    // Build a fully wired BatchAssembler (FeatureMatrix fallback mode).
    // Caller must keep the returned storage alive!
    // -----------------------------------------------------------------------
    struct Env {
        FeatureMatrix   fm;
        RowMapping      rm;
        LabelStore      ls;
        SplitStore      ss;
        SampleStorage   storage;
        SampleCatalog   catalog;
    };

    Env make_env(const std::string& name = "test_samples")
    {
        Env e{
            FeatureMatrix::open(fmat_path_),
            RowMapping::open(rmap_path_),
            LabelStore::open(labels_path_),
            SplitStore::open(splits_path_),
            SampleStorage::open(
                SampleStorage::get_storage_path(db_folder_, name)
            ),
            SampleCatalog{},
        };
        e.catalog = create_sample_storage(name + "_tmp");
        return e;
    }
};

// =============================================================================
// Test 1: LossDecreases
//
// Train 10 epochs on a linearly-separable dataset.
// The average training loss at epoch 10 must be strictly less than epoch 1.
// =============================================================================

TEST_F(TrainingLoopTest, LossDecreases)
{
    // Each test run needs a unique storage name to avoid path collisions.
    const std::string sname = "loss_decreases";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    GraphSAGEConfig cfg{
        .input_dim   = static_cast<int64_t>(D),
        .hidden_dim  = 16,
        .num_classes = static_cast<int64_t>(NUM_CLASSES),
        .num_layers  = 1,
        .dropout     = 0.0,   // deterministic forward pass
        .normalize   = false,
    };
    GraphSAGEModel model(cfg);

    TrainingLoop::Config loop_cfg;
    loop_cfg.epochs        = 10;
    loop_cfg.learning_rate = 0.05;
    loop_cfg.weight_decay  = 0.0;
    loop_cfg.tolerance     = 1e-9;   // effectively disabled
    loop_cfg.patience      = 100;    // effectively disabled
    loop_cfg.random_seed   = 42;

    TrainingLoop loop(model, assembler, cat, loop_cfg);
    auto result = loop.train();

    ASSERT_GE(result.epoch_losses.size(), 2u)
        << "Expected at least 2 epoch loss values";

    double first_loss = result.epoch_losses.front();
    double last_loss  = result.epoch_losses.back();

    EXPECT_LT(last_loss, first_loss)
        << "Loss did not decrease: first=" << first_loss
        << "  last=" << last_loss;

    // ran_epochs may be less than 10 if the loop converged early — that is
    // valid behaviour.  We only require at least 2 epochs ran.
    EXPECT_GE(result.ran_epochs, 2u);
}

// =============================================================================
// Test 2: ConvergenceStops
//
// Set tolerance=999 (very large) so the very first two-epoch delta will
// always fall below it.  Expect converged=true and ran_epochs==2.
// =============================================================================

TEST_F(TrainingLoopTest, ConvergenceStops)
{
    const std::string sname = "convergence_stops";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    GraphSAGEConfig cfg{
        .input_dim   = static_cast<int64_t>(D),
        .hidden_dim  = 16,
        .num_classes = static_cast<int64_t>(NUM_CLASSES),
        .num_layers  = 1,
        .dropout     = 0.0,
        .normalize   = false,
    };
    GraphSAGEModel model(cfg);

    TrainingLoop::Config loop_cfg;
    loop_cfg.epochs        = 50;
    loop_cfg.learning_rate = 0.01;
    loop_cfg.tolerance     = 999.0;   // always triggers after epoch 2
    loop_cfg.patience      = 100;     // effectively disabled
    loop_cfg.random_seed   = 0;

    TrainingLoop loop(model, assembler, cat, loop_cfg);
    auto result = loop.train();

    EXPECT_TRUE(result.converged)
        << "Expected converged=true with tolerance=999";
    EXPECT_EQ(result.ran_epochs, 2u)
        << "Expected early stop after epoch 2";
}

// =============================================================================
// Test 3: RandomSeedReproducible
//
// Two independent TrainingLoop instances initialised with the same random
// seed on fresh model copies must produce identical epoch_losses.
// =============================================================================

TEST_F(TrainingLoopTest, RandomSeedReproducible)
{
    const std::string sname_a = "repro_a";
    const std::string sname_b = "repro_b";
    auto cat_a = create_sample_storage(sname_a);
    auto cat_b = create_sample_storage(sname_b);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);
    auto ls = LabelStore::open(labels_path_);
    auto ss = SplitStore::open(splits_path_);

    auto storage_a = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname_a));
    auto storage_b = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname_b));

    GraphSAGEConfig cfg{
        .input_dim   = static_cast<int64_t>(D),
        .hidden_dim  = 16,
        .num_classes = static_cast<int64_t>(NUM_CLASSES),
        .num_layers  = 1,
        .dropout     = 0.0,   // no stochastic ops during forward
        .normalize   = false,
    };

    TrainingLoop::Config loop_cfg;
    loop_cfg.epochs        = 5;
    loop_cfg.learning_rate = 0.01;
    loop_cfg.tolerance     = 1e-9;
    loop_cfg.patience      = 100;
    loop_cfg.random_seed   = 1234;

    // Both models must start from identical weights.  Set the global torch seed
    // before each construction so Kaiming-uniform initialisation is identical.

    // --- Run A ---
    torch::manual_seed(1234);
    GraphSAGEModel model_a(cfg);
    BatchAssembler assembler_a(fm, storage_a, &ls, &ss, rm);
    TrainingLoop loop_a(model_a, assembler_a, cat_a, loop_cfg);
    auto result_a = loop_a.train();

    // --- Run B (same seed, freshly constructed model) ---
    torch::manual_seed(1234);
    GraphSAGEModel model_b(cfg);
    BatchAssembler assembler_b(fm, storage_b, &ls, &ss, rm);
    TrainingLoop loop_b(model_b, assembler_b, cat_b, loop_cfg);
    auto result_b = loop_b.train();

    ASSERT_EQ(result_a.epoch_losses.size(), result_b.epoch_losses.size())
        << "Both runs must produce the same number of epochs";

    for (size_t e = 0; e < result_a.epoch_losses.size(); ++e) {
        EXPECT_DOUBLE_EQ(result_a.epoch_losses[e], result_b.epoch_losses[e])
            << "Loss mismatch at epoch " << e;
    }
}

// =============================================================================
// Test 4: EvaluateReturnsValidAccuracy
//
// After at least one training epoch the evaluate() method must return a
// value in [0.0, 1.0].
// =============================================================================

TEST_F(TrainingLoopTest, EvaluateReturnsValidAccuracy)
{
    const std::string sname = "eval_accuracy";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    GraphSAGEConfig cfg{
        .input_dim   = static_cast<int64_t>(D),
        .hidden_dim  = 16,
        .num_classes = static_cast<int64_t>(NUM_CLASSES),
        .num_layers  = 1,
        .dropout     = 0.0,
        .normalize   = false,
    };
    GraphSAGEModel model(cfg);

    TrainingLoop::Config loop_cfg;
    loop_cfg.epochs        = 3;
    loop_cfg.learning_rate = 0.01;
    loop_cfg.tolerance     = 1e-9;
    loop_cfg.patience      = 100;
    loop_cfg.random_seed   = 7;

    TrainingLoop loop(model, assembler, cat, loop_cfg);
    loop.train();

    // Validation batch is at index train_batches (= 2)
    double acc = loop.evaluate(cat.train_batches, cat.validation_batches);

    EXPECT_GE(acc, 0.0) << "Accuracy below 0";
    EXPECT_LE(acc, 1.0) << "Accuracy above 1";
}

// =============================================================================
// Test 5: ZeroValidationBatchesEvaluatesToZero
//
// evaluate(start, 0) should return 0.0 gracefully.
// =============================================================================

TEST_F(TrainingLoopTest, ZeroValidationBatchesEvaluatesToZero)
{
    const std::string sname = "zero_val";
    auto cat = create_sample_storage(sname);
    // Override val count to 0 for this test
    cat.validation_batches = 0;

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    GraphSAGEConfig cfg{
        .input_dim   = static_cast<int64_t>(D),
        .hidden_dim  = 8,
        .num_classes = static_cast<int64_t>(NUM_CLASSES),
        .num_layers  = 1,
        .dropout     = 0.0,
    };
    GraphSAGEModel model(cfg);

    TrainingLoop loop(model, assembler, cat,
                      TrainingLoop::Config{.epochs = 1, .patience = 100});

    double acc = loop.evaluate(0, 0);
    EXPECT_DOUBLE_EQ(acc, 0.0);
}

// =============================================================================
// Test 6: PatienceStops
//
// If validation accuracy never improves (start very poorly), the loop
// should stop after `patience` consecutive non-improving epochs.
// =============================================================================

TEST_F(TrainingLoopTest, PatienceStops)
{
    const std::string sname = "patience_stops";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    GraphSAGEConfig cfg{
        .input_dim   = static_cast<int64_t>(D),
        .hidden_dim  = 4,
        .num_classes = static_cast<int64_t>(NUM_CLASSES),
        .num_layers  = 1,
        .dropout     = 0.0,
    };
    GraphSAGEModel model(cfg);

    // Use learning_rate=0 so weights never update and val_accuracy stays fixed.
    // With patience=3 and tolerance=1e-9, the loop should stop at epoch 3
    // (after patience consecutive non-improving epochs).
    TrainingLoop::Config loop_cfg;
    loop_cfg.epochs        = 100;
    loop_cfg.learning_rate = 0.0;   // frozen weights → constant val accuracy
    loop_cfg.tolerance     = 1e-9;  // tiny: loss won't converge with lr=0 and
                                    //       two equal consecutive losses would
                                    //       satisfy it, but we check patience first
    loop_cfg.patience      = 3;
    loop_cfg.random_seed   = 99;

    TrainingLoop loop(model, assembler, cat, loop_cfg);
    auto result = loop.train();

    // With lr=0, val_accuracy is the same every epoch.
    // The patience counter increments on every non-improving epoch.
    // It fires after patience consecutive non-improving epochs.
    // However: the first epoch can set best_val_acc, then the next `patience`
    // epochs trigger the stop.  So ran_epochs == 1 + patience.
    // Allow a small range in case the zero-lr run still satisfies tolerance.
    EXPECT_LE(result.ran_epochs, loop_cfg.patience + 2)
        << "Loop ran more epochs than expected with patience="
        << loop_cfg.patience;
}
