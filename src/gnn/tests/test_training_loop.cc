#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <vector>

// Fixture (+ all GNN includes + using-directives) is shared with
// test_training_loop_resume.cc via this header.
#include "gnn/tests/training_loop_test_fixture.h"

namespace fs = std::filesystem;

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

    torch::optim::Adam opt(
        model.parameters(),
        torch::optim::AdamOptions(loop_cfg.learning_rate)
            .weight_decay(loop_cfg.weight_decay)
    );
    TrainingLoop loop(model, assembler, cat, opt, loop_cfg);
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

    torch::optim::Adam opt(
        model.parameters(),
        torch::optim::AdamOptions(loop_cfg.learning_rate)
            .weight_decay(loop_cfg.weight_decay)
    );
    TrainingLoop loop(model, assembler, cat, opt, loop_cfg);
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
    torch::optim::Adam opt_a(
        model_a.parameters(),
        torch::optim::AdamOptions(loop_cfg.learning_rate)
            .weight_decay(loop_cfg.weight_decay)
    );
    TrainingLoop loop_a(model_a, assembler_a, cat_a, opt_a, loop_cfg);
    auto result_a = loop_a.train();

    // --- Run B (same seed, freshly constructed model) ---
    torch::manual_seed(1234);
    GraphSAGEModel model_b(cfg);
    BatchAssembler assembler_b(fm, storage_b, &ls, &ss, rm);
    torch::optim::Adam opt_b(
        model_b.parameters(),
        torch::optim::AdamOptions(loop_cfg.learning_rate)
            .weight_decay(loop_cfg.weight_decay)
    );
    TrainingLoop loop_b(model_b, assembler_b, cat_b, opt_b, loop_cfg);
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

    torch::optim::Adam opt(
        model.parameters(),
        torch::optim::AdamOptions(loop_cfg.learning_rate)
            .weight_decay(loop_cfg.weight_decay)
    );
    TrainingLoop loop(model, assembler, cat, opt, loop_cfg);
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

    TrainingLoop::Config zero_val_cfg{.epochs = 1, .patience = 100};
    torch::optim::Adam opt(
        model.parameters(),
        torch::optim::AdamOptions(zero_val_cfg.learning_rate)
            .weight_decay(zero_val_cfg.weight_decay)
    );
    TrainingLoop loop(model, assembler, cat, opt, zero_val_cfg);

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
    // With patience=3 and tolerance=0.0 (convergence gated by strict <), the
    // loop must stop from patience exhaustion, not convergence: one
    // improvement epoch sets best_val_acc, the next `patience` non-improving
    // epochs trigger the stop, yielding ran_epochs == patience + 1.
    TrainingLoop::Config loop_cfg;
    loop_cfg.epochs        = 100;
    loop_cfg.learning_rate = 0.0;   // frozen weights → constant val accuracy
    loop_cfg.tolerance     = 0.0;   // convergence uses strict <, so |delta|<0
                                    // is never true and patience must fire.
    loop_cfg.patience      = 3;
    loop_cfg.random_seed   = 99;

    torch::optim::Adam opt(
        model.parameters(),
        torch::optim::AdamOptions(loop_cfg.learning_rate)
            .weight_decay(loop_cfg.weight_decay)
    );
    TrainingLoop loop(model, assembler, cat, opt, loop_cfg);
    auto result = loop.train();

    // With lr=0, val_accuracy is the same every epoch.
    // The patience counter increments on every non-improving epoch.
    // It fires after patience consecutive non-improving epochs.
    // However: the first epoch can set best_val_acc, then the next `patience`
    // epochs trigger the stop.  So ran_epochs == 1 + patience.
    // Allow a small range in case the zero-lr run still satisfies tolerance.
    EXPECT_EQ(result.ran_epochs, loop_cfg.patience + 1)
        << "Loop ran more epochs than expected with patience="
        << loop_cfg.patience;
}

// =============================================================================
// Spec C3 stage 0 (2026-05-07): per-stage timing instrumentation
//
// Validates that the new assemble_seconds, forward_seconds, backward_seconds
// fields populate non-zero values during a real training run, and that their
// sum is bounded above by train_seconds (since train_seconds also includes
// the validation phase, the inequality is strict in general).
// =============================================================================

TEST_F(TrainingLoopTest, SpecC3_PerStageTimingPopulated)
{
    const std::string sname = "spec_c3_timing";
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
    loop_cfg.weight_decay  = 0.0;
    loop_cfg.tolerance     = 1e-9;
    loop_cfg.patience      = 100;
    loop_cfg.random_seed   = 42;

    torch::optim::Adam opt(
        model.parameters(),
        torch::optim::AdamOptions(loop_cfg.learning_rate)
            .weight_decay(loop_cfg.weight_decay)
    );
    TrainingLoop loop(model, assembler, cat, opt, loop_cfg);
    auto result = loop.train();

    // All three stages should have measured non-zero time. Even on a tiny
    // toy fixture, each stage takes at least microseconds, well above the
    // 1e-9 noise floor of steady_clock.
    EXPECT_GT(result.assemble_seconds, 0.0)
        << "assemble_seconds must be > 0 after a real run";
    EXPECT_GT(result.forward_seconds, 0.0)
        << "forward_seconds must be > 0 after a real run";
    EXPECT_GT(result.backward_seconds, 0.0)
        << "backward_seconds must be > 0 after a real run";

    // Sum of stages should be bounded above by train_seconds (the latter
    // also covers validation, callbacks, and per-batch zero_grad/loss-mask
    // checks that aren't bucketed into the three stages).
    const double stage_sum = result.assemble_seconds
                           + result.forward_seconds
                           + result.backward_seconds;
    EXPECT_LE(stage_sum, result.train_seconds + 1e-6)
        << "stage_sum=" << stage_sum
        << " > train_seconds=" << result.train_seconds
        << "; stages should be a strict subset of total train wall-time";
}
