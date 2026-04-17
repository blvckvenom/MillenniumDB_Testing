// =============================================================================
// Tests for TrainingLoop resume semantics + EpochEvent callback behavior.
//
// Covers:
//   - Callback fires exactly once per epoch (including the final epoch that
//     triggers early stop — verifies the fix in commit 5ffbe8c).
//   - EpochEvent fields are populated consistently.
//   - start_epoch / start_patience / start_best_val / seed_losses are honored.
//   - Strict-greater is_best semantics (verifies fix in commit 5ffbe8c).
//   - External optimizer state survives train() invocation.
//   - Callback absence (default-constructed std::function) is safe.
//
// These tests serve double duty:
//   (a) Validate the resume feature used by Phase 5 auto-checkpointing.
//   (b) Act as regression guards for the callback-every-epoch + strict-greater
//       is_best fixes.
// =============================================================================

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "gnn/tests/training_loop_test_fixture.h"

namespace fs = std::filesystem;

// =============================================================================
// Test fixture.
//
// Derives from the shared TrainingLoopTestFixture and exposes a few protected
// members as public accessors so that helper utilities in an anonymous
// namespace can access them without friend declarations. This avoids touching
// the shared header.
// =============================================================================

class TrainingLoopResumeTest : public mdb::gnn::testing_util::TrainingLoopTestFixture {
public:
    const std::filesystem::path& fmat_path()   const { return fmat_path_;   }
    const std::filesystem::path& rmap_path()   const { return rmap_path_;   }
    const std::filesystem::path& labels_path() const { return labels_path_; }
    const std::filesystem::path& splits_path() const { return splits_path_; }
    const std::filesystem::path& db_folder()   const { return db_folder_;   }

    SampleCatalog create_sample_storage_public(const std::string& name) {
        return create_sample_storage(name);
    }
};

namespace {

// -----------------------------------------------------------------------------
// Helper bundle keeping the heavy objects alive for a single test.
// -----------------------------------------------------------------------------
struct TrainRig {
    FeatureMatrix   fm;
    RowMapping      rm;
    LabelStore      ls;
    SplitStore      ss;
    SampleStorage   storage;
    SampleCatalog   catalog;

    // Heap-allocated so we never accidentally copy-construct a model/optimizer.
    std::unique_ptr<GraphSAGEModel>     model;
    std::unique_ptr<BatchAssembler>     assembler;
    std::unique_ptr<torch::optim::Adam> optimizer;
};

// Build a ready-to-train rig with the standard fixture paths + a fresh sample
// storage named `sname`. Each test needs a unique `sname` to keep on-disk
// state isolated between runs.
TrainRig make_rig(TrainingLoopResumeTest& f,
                  const std::string& sname,
                  double learning_rate = 0.01,
                  double weight_decay  = 0.0)
{
    // create_sample_storage_public writes + finalizes a SampleStorage on disk.
    SampleCatalog cat = f.create_sample_storage_public(sname);

    TrainRig r{
        FeatureMatrix::open(f.fmat_path()),
        RowMapping::open(f.rmap_path()),
        LabelStore::open(f.labels_path()),
        SplitStore::open(f.splits_path()),
        SampleStorage::open(
            SampleStorage::get_storage_path(f.db_folder(), sname)
        ),
        std::move(cat),
        nullptr, nullptr, nullptr
    };

    // Mirror the fixture constants (D=4, NUM_CLASSES=2). They are protected
    // inside the fixture, so we duplicate the literals here rather than
    // widening visibility in the shared header.
    GraphSAGEConfig cfg;
    cfg.input_dim   = 4;
    cfg.hidden_dim  = 8;
    cfg.num_classes = 2;
    cfg.num_layers  = 1;
    cfg.dropout     = 0.0;
    cfg.normalize   = false;

    // Make model construction deterministic.
    torch::manual_seed(1234);
    r.model = std::make_unique<GraphSAGEModel>(cfg);

    r.assembler = std::make_unique<BatchAssembler>(
        r.fm, r.storage, &r.ls, &r.ss, r.rm
    );

    r.optimizer = std::make_unique<torch::optim::Adam>(
        r.model->parameters(),
        torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay)
    );

    return r;
}

} // namespace

// =============================================================================
// Test 1: FreshTrainingFiresCallbackOncePerEpoch
//
// With patience and tolerance effectively disabled, 3 configured epochs must
// produce exactly 3 callback invocations. Also verifies monotonic `epoch`
// field (0, 1, 2) and that result.ran_epochs equals 3.
// =============================================================================

TEST_F(TrainingLoopResumeTest, FreshTrainingFiresCallbackOncePerEpoch)
{
    torch::manual_seed(42);

    auto r = make_rig(*this, "fresh_cb_once");

    int cb_count = 0;
    std::vector<uint64_t> seen_epochs;

    TrainingLoop::Config cfg;
    cfg.epochs        = 3;
    cfg.patience      = 100;
    cfg.tolerance     = 0.0;     // convergence uses strict <, so never fires
    cfg.learning_rate = 0.01;
    cfg.random_seed   = 42;
    cfg.on_epoch_end  = [&](const TrainingLoop::EpochEvent& e) {
        ++cb_count;
        seen_epochs.push_back(e.epoch);
    };

    TrainingLoop loop(*r.model, *r.assembler, r.catalog, *r.optimizer, cfg);
    auto result = loop.train();

    EXPECT_EQ(cb_count, 3);
    EXPECT_EQ(result.ran_epochs, 3u);
    ASSERT_EQ(seen_epochs.size(), 3u);
    EXPECT_EQ(seen_epochs[0], 0u);
    EXPECT_EQ(seen_epochs[1], 1u);
    EXPECT_EQ(seen_epochs[2], 2u);
}

// =============================================================================
// Test 2: CallbackReceivesCorrectEpochEventFields
//
// Verify the shape + value ranges of the EpochEvent for the first epoch.
// With start_best_val default (0.0), our linearly-separable toy dataset
// produces val_accuracy > 0 so is_best == true.
// =============================================================================

TEST_F(TrainingLoopResumeTest, CallbackReceivesCorrectEpochEventFields)
{
    torch::manual_seed(7);

    auto r = make_rig(*this, "cb_fields");

    std::vector<TrainingLoop::EpochEvent> events;

    TrainingLoop::Config cfg;
    cfg.epochs        = 2;
    cfg.patience      = 100;
    cfg.tolerance     = 0.0;
    cfg.learning_rate = 0.01;
    cfg.random_seed   = 7;
    cfg.on_epoch_end  = [&](const TrainingLoop::EpochEvent& e) {
        events.push_back(e);
    };

    TrainingLoop loop(*r.model, *r.assembler, r.catalog, *r.optimizer, cfg);
    auto result = loop.train();

    ASSERT_EQ(events.size(), 2u);

    const auto& first = events[0];
    EXPECT_EQ(first.epoch, 0u);
    // Loss must be finite and non-negative. On this linearly-separable toy
    // dataset with extreme features (±400) the initial cross-entropy can be
    // arbitrarily close to 0 depending on random init, so we only require
    // >= 0 here.
    EXPECT_GE(first.train_loss, 0.0);
    EXPECT_FALSE(std::isnan(first.train_loss));
    EXPECT_FALSE(std::isinf(first.train_loss));
    EXPECT_GE(first.val_accuracy, 0.0);
    EXPECT_LE(first.val_accuracy, 1.0);

    // Our linearly-separable toy dataset essentially guarantees val_accuracy > 0,
    // so the first epoch should set a new best (strict > 0.0 comparator).
    EXPECT_TRUE(first.is_best);
    EXPECT_EQ(first.patience_counter, 0u);

    // Sanity check on result.
    EXPECT_EQ(result.ran_epochs, 2u);
}

// =============================================================================
// Test 3: ResumeStartEpochOffsetEpochFieldInCallback
//
// When start_epoch = 10 and epochs = 3, the callback must receive
// EpochEvent::epoch values 10, 11, 12 (absolute epoch ids), and
// result.ran_epochs must be 3 (epochs run *this invocation*, not total).
// =============================================================================

TEST_F(TrainingLoopResumeTest, ResumeStartEpochOffsetEpochFieldInCallback)
{
    torch::manual_seed(11);

    auto r = make_rig(*this, "start_epoch");

    std::vector<uint64_t> seen_epochs;

    TrainingLoop::Config cfg;
    cfg.epochs        = 3;
    cfg.start_epoch   = 10;
    cfg.patience      = 100;
    cfg.tolerance     = 0.0;
    cfg.learning_rate = 0.01;
    cfg.random_seed   = 11;
    cfg.on_epoch_end  = [&](const TrainingLoop::EpochEvent& e) {
        seen_epochs.push_back(e.epoch);
    };

    TrainingLoop loop(*r.model, *r.assembler, r.catalog, *r.optimizer, cfg);
    auto result = loop.train();

    ASSERT_EQ(seen_epochs.size(), 3u);
    EXPECT_EQ(seen_epochs[0], 10u);
    EXPECT_EQ(seen_epochs[1], 11u);
    EXPECT_EQ(seen_epochs[2], 12u);

    // ran_epochs counts epochs run IN THIS INVOCATION, not absolute epoch id.
    EXPECT_EQ(result.ran_epochs, 3u);
}

// =============================================================================
// Test 4: ResumeSeedLossesPrependedToResult
//
// Seed losses from a prior run must appear at the front of result.epoch_losses,
// followed by the newly computed losses for this invocation's epochs.
// =============================================================================

TEST_F(TrainingLoopResumeTest, ResumeSeedLossesPrependedToResult)
{
    torch::manual_seed(22);

    auto r = make_rig(*this, "seed_losses");

    TrainingLoop::Config cfg;
    cfg.epochs        = 2;
    cfg.patience      = 100;
    cfg.tolerance     = 0.0;
    cfg.learning_rate = 0.01;
    cfg.random_seed   = 22;
    cfg.seed_losses   = {2.0, 1.5, 1.0};

    TrainingLoop loop(*r.model, *r.assembler, r.catalog, *r.optimizer, cfg);
    auto result = loop.train();

    // 3 seed losses + 2 newly computed = 5 total.
    ASSERT_EQ(result.epoch_losses.size(), 5u);
    EXPECT_DOUBLE_EQ(result.epoch_losses[0], 2.0);
    EXPECT_DOUBLE_EQ(result.epoch_losses[1], 1.5);
    EXPECT_DOUBLE_EQ(result.epoch_losses[2], 1.0);
    // [3] and [4] are the actual computed losses — must be finite and >= 0.
    // On this extreme toy dataset the actual loss can underflow to 0, so we
    // only enforce finiteness and non-negativity here.
    EXPECT_GE(result.epoch_losses[3], 0.0);
    EXPECT_FALSE(std::isnan(result.epoch_losses[3]));
    EXPECT_FALSE(std::isinf(result.epoch_losses[3]));
    EXPECT_GE(result.epoch_losses[4], 0.0);
    EXPECT_FALSE(std::isnan(result.epoch_losses[4]));
    EXPECT_FALSE(std::isinf(result.epoch_losses[4]));

    // ran_epochs counts THIS invocation, so = 2 not 5.
    EXPECT_EQ(result.ran_epochs, 2u);
}

// =============================================================================
// Test 5: ResumeStartBestValSeedAffectsIsBest
//
// With start_best_val = 1.0 (the maximum possible accuracy), no epoch can
// strictly beat it. The first epoch callback must observe is_best == false
// and patience_counter == 1.
//
// This directly verifies the strict-greater semantics fix (commit 5ffbe8c):
// a `>=` comparator would mark is_best==true when val_accuracy matches 1.0,
// which is WRONG for checkpointing.
// =============================================================================

TEST_F(TrainingLoopResumeTest, ResumeStartBestValSeedAffectsIsBest)
{
    torch::manual_seed(33);

    auto r = make_rig(*this, "seed_best_val");

    std::vector<TrainingLoop::EpochEvent> events;

    TrainingLoop::Config cfg;
    cfg.epochs         = 1;
    cfg.patience       = 100;      // don't stop on patience
    cfg.tolerance      = 0.0;
    cfg.start_best_val = 1.0;      // unbeatable — even val=1.0 is NOT strictly greater
    cfg.learning_rate  = 0.01;
    cfg.random_seed    = 33;
    cfg.on_epoch_end   = [&](const TrainingLoop::EpochEvent& e) {
        events.push_back(e);
    };

    TrainingLoop loop(*r.model, *r.assembler, r.catalog, *r.optimizer, cfg);
    loop.train();

    ASSERT_EQ(events.size(), 1u);
    EXPECT_FALSE(events[0].is_best)
        << "start_best_val=1.0 must make is_best=false (strict > comparator).";
    EXPECT_EQ(events[0].patience_counter, 1u);
}

// =============================================================================
// Test 6: ResumeStartPatienceTriggersEarlyStop
//
// CRITICAL — directly verifies Fix #1 (callback fires on patience-exhaust epoch).
//
// Setup: start_patience = 4, patience = 5, start_best_val = 1.0 (unbeatable).
// The single epoch of this invocation cannot improve, so patience_counter
// increments 4 → 5, triggering the stop.
// Expectations:
//   - ran_epochs == 1          (one epoch ran before stop)
//   - converged == false       (stopped on patience, not tolerance)
//   - cb_count == 1            (callback fired on the final epoch — THE FIX)
//   - last event's patience_counter == 5
//   - last event's is_best == false
// =============================================================================

TEST_F(TrainingLoopResumeTest, ResumeStartPatienceTriggersEarlyStop)
{
    torch::manual_seed(44);

    auto r = make_rig(*this, "start_patience");

    int cb_count = 0;
    TrainingLoop::EpochEvent last_event{};

    TrainingLoop::Config cfg;
    cfg.epochs         = 10;       // plenty, but we expect to stop after 1
    cfg.patience       = 5;
    cfg.start_patience = 4;        // one non-improvement away from exhaustion
    cfg.start_best_val = 1.0;      // unbeatable — forces non-improvement
    cfg.tolerance      = 0.0;      // disable convergence early-stop
    cfg.learning_rate  = 0.01;
    cfg.random_seed    = 44;
    cfg.on_epoch_end   = [&](const TrainingLoop::EpochEvent& e) {
        ++cb_count;
        last_event = e;
    };

    TrainingLoop loop(*r.model, *r.assembler, r.catalog, *r.optimizer, cfg);
    auto result = loop.train();

    EXPECT_EQ(result.ran_epochs, 1u);
    EXPECT_FALSE(result.converged);

    // THE FIX: callback MUST fire on the final patience-exhaustion epoch.
    EXPECT_EQ(cb_count, 1)
        << "Fix #1 regression: callback failed to fire on the "
           "patience-exhaustion epoch.";
    EXPECT_EQ(last_event.patience_counter, 5u);
    EXPECT_FALSE(last_event.is_best);
}

// =============================================================================
// Test 7: ResumeExternalOptimizerBuffersPreserved
//
// Ensures the TrainingLoop does NOT reset / reconstruct the caller-owned
// Adam optimizer. After train() returns, the optimizer's state dict must be
// non-empty (Adam populates per-parameter m/v buffers after at least one step).
// If the loop re-created the optimizer internally, the caller's optimizer
// would remain in its fresh state (state_dict entries zero/unpopulated).
// =============================================================================

TEST_F(TrainingLoopResumeTest, ResumeExternalOptimizerBuffersPreserved)
{
    torch::manual_seed(55);

    auto r = make_rig(*this, "ext_opt_preserved", /*lr=*/0.01, /*wd=*/0.0);

    // Sanity: before any training, Adam has no per-parameter state (it
    // populates state lazily on the first step()).
    ASSERT_TRUE(r.optimizer->state().empty())
        << "Precondition: fresh Adam has no state before the first step.";

    TrainingLoop::Config cfg;
    cfg.epochs        = 2;
    cfg.patience      = 100;
    cfg.tolerance     = 0.0;
    cfg.learning_rate = 0.01;
    cfg.random_seed   = 55;

    TrainingLoop loop(*r.model, *r.assembler, r.catalog, *r.optimizer, cfg);
    auto result = loop.train();
    EXPECT_EQ(result.ran_epochs, 2u);

    // The caller-owned optimizer MUST retain per-parameter state after training.
    // If the loop had reconstructed a local optimizer, ours would still be empty.
    EXPECT_FALSE(r.optimizer->state().empty())
        << "Adam state is empty after train(); "
           "TrainingLoop likely reconstructed the optimizer internally, "
           "which would discard resumed momenta.";
}

// =============================================================================
// Test 8: NoCallbackProvidedDoesNotCrash
//
// When Config::on_epoch_end is left default-constructed (empty std::function),
// train() must not crash, throw, or misbehave. Verifies the defensive nullity
// check around the callback invocation.
// =============================================================================

TEST_F(TrainingLoopResumeTest, NoCallbackProvidedDoesNotCrash)
{
    torch::manual_seed(66);

    auto r = make_rig(*this, "no_callback");

    TrainingLoop::Config cfg;
    cfg.epochs        = 2;
    cfg.patience      = 100;
    cfg.tolerance     = 0.0;
    cfg.learning_rate = 0.01;
    cfg.random_seed   = 66;
    // cfg.on_epoch_end intentionally left default (empty std::function)

    TrainingLoop loop(*r.model, *r.assembler, r.catalog, *r.optimizer, cfg);

    ASSERT_NO_THROW({
        auto result = loop.train();
        EXPECT_EQ(result.ran_epochs, 2u);
    });
}
