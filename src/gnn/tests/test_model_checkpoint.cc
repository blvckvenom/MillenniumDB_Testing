#include "test_helpers.h"

#include <array>
#include <cstdint>
#include <fstream>

#include "gnn/output/model_checkpoint.h"
#include "gnn/models/graphsage_model.h"
#include "gnn/projection/gnn_meta.h"

using ModelCheckpointTest = GnnStorageTest;

// ---------------------------------------------------------------------------
// ComputeHashStable — same gnn_meta.bin produces same hash on repeated calls
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ComputeHashStable) {
    mdb::gnn::GnnMeta m;
    m.feature_name = "features";
    m.feature_dim  = 128;
    m.num_nodes    = 1000;
    m.num_classes  = 7;
    m.has_labels   = true;
    m.has_splits   = false;
    auto path = test_dir_ / "gnn_meta.bin";
    m.write(path);

    auto h1 = mdb::gnn::ModelCheckpoint::compute_gnn_meta_hash(path);
    auto h2 = mdb::gnn::ModelCheckpoint::compute_gnn_meta_hash(path);

    EXPECT_EQ(h1, h2);

    // And it's not all zeros (sanity)
    std::array<uint8_t, 32> zero{};
    EXPECT_NE(h1, zero);
}

// ---------------------------------------------------------------------------
// ComputeHashDistinct — different gnn_meta.bin contents produce different hashes
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ComputeHashDistinct) {
    mdb::gnn::GnnMeta m1;
    m1.feature_name = "features";
    m1.feature_dim  = 128;
    m1.num_nodes    = 1000;
    m1.num_classes  = 7;
    auto p1 = test_dir_ / "m1.bin";
    m1.write(p1);

    mdb::gnn::GnnMeta m2 = m1;
    m2.feature_name = "different_features";   // single-field change → different hash
    auto p2 = test_dir_ / "m2.bin";
    m2.write(p2);

    auto h1 = mdb::gnn::ModelCheckpoint::compute_gnn_meta_hash(p1);
    auto h2 = mdb::gnn::ModelCheckpoint::compute_gnn_meta_hash(p2);

    EXPECT_NE(h1, h2);
}

// ---------------------------------------------------------------------------
// Helper: build a representative state for roundtrip tests
// ---------------------------------------------------------------------------
static mdb::gnn::TrainingState sample_state() {
    mdb::gnn::TrainingState s;
    s.epoch               = 12;
    s.patience_counter    = 3;
    s.best_val_accuracy   = 0.79f;
    s.epoch_losses        = {1.3, 1.1, 0.9, 0.8, 0.75};
    s.input_dim           = 1433;
    s.hidden_dim          = 16;
    s.num_classes         = 7;
    s.num_layers          = 2;
    s.dropout             = 0.5;
    s.normalize           = false;
    s.model_type          = "graphsage";
    s.projection_name     = "cora_projection";
    for (size_t i = 0; i < s.gnn_meta_hash.size(); ++i) {
        s.gnn_meta_hash[i] = static_cast<uint8_t>(i * 7 + 1);
    }
    s.creation_time_unix       = 1735000000;
    s.total_training_time_sec  = 12.345;
    return s;
}

// ---------------------------------------------------------------------------
// CkptmetaRoundtrip — every field written is recovered bit-exact
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, CkptmetaRoundtrip) {
    auto s    = sample_state();
    auto path = test_dir_ / "best_model.ckptmeta";

    mdb::gnn::ModelCheckpoint::write_ckptmeta(path, s);
    auto s2 = mdb::gnn::ModelCheckpoint::read_ckptmeta(path);

    EXPECT_EQ(s2.epoch,               s.epoch);
    EXPECT_EQ(s2.patience_counter,    s.patience_counter);
    EXPECT_FLOAT_EQ(s2.best_val_accuracy, s.best_val_accuracy);
    EXPECT_EQ(s2.epoch_losses,        s.epoch_losses);
    EXPECT_EQ(s2.input_dim,           s.input_dim);
    EXPECT_EQ(s2.hidden_dim,          s.hidden_dim);
    EXPECT_EQ(s2.num_classes,         s.num_classes);
    EXPECT_EQ(s2.num_layers,          s.num_layers);
    EXPECT_DOUBLE_EQ(s2.dropout,      s.dropout);
    EXPECT_EQ(s2.normalize,           s.normalize);
    EXPECT_EQ(s2.model_type,          s.model_type);
    EXPECT_EQ(s2.projection_name,     s.projection_name);
    EXPECT_EQ(s2.gnn_meta_hash,       s.gnn_meta_hash);
    EXPECT_EQ(s2.creation_time_unix,  s.creation_time_unix);
    EXPECT_DOUBLE_EQ(s2.total_training_time_sec, s.total_training_time_sec);
}

// ---------------------------------------------------------------------------
// CkptmetaRejectBadMagic — corrupted magic bytes → throw
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, CkptmetaRejectBadMagic) {
    auto s    = sample_state();
    auto path = test_dir_ / "bad.ckptmeta";
    mdb::gnn::ModelCheckpoint::write_ckptmeta(path, s);

    // Overwrite first byte of magic
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(f);
        char bad = 'Z';
        f.seekp(0);
        f.write(&bad, 1);
    }

    EXPECT_THROW(mdb::gnn::ModelCheckpoint::read_ckptmeta(path),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// CkptmetaRejectBadVersion — unknown format_version → throw
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, CkptmetaRejectBadVersion) {
    auto s    = sample_state();
    auto path = test_dir_ / "v99.ckptmeta";
    mdb::gnn::ModelCheckpoint::write_ckptmeta(path, s);

    // Overwrite version (uint32 at offset 8) with 99
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(f);
        uint32_t v = 99;
        f.seekp(8);
        f.write(reinterpret_cast<const char*>(&v), 4);
    }

    EXPECT_THROW(mdb::gnn::ModelCheckpoint::read_ckptmeta(path),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// CkptmetaRejectTruncated — file ends mid-payload → throw
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, CkptmetaRejectTruncated) {
    auto s    = sample_state();
    auto path = test_dir_ / "trunc.ckptmeta";
    mdb::gnn::ModelCheckpoint::write_ckptmeta(path, s);

    // Truncate to 50 bytes (should be ~204)
    std::filesystem::resize_file(path, 50);

    EXPECT_THROW(mdb::gnn::ModelCheckpoint::read_ckptmeta(path),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// CkptmetaEmptyEpochLosses — num_epoch_losses=0 handled correctly
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, CkptmetaEmptyEpochLosses) {
    auto s    = sample_state();
    s.epoch_losses.clear();
    auto path = test_dir_ / "empty_losses.ckptmeta";

    mdb::gnn::ModelCheckpoint::write_ckptmeta(path, s);
    auto s2 = mdb::gnn::ModelCheckpoint::read_ckptmeta(path);

    EXPECT_TRUE(s2.epoch_losses.empty());
    EXPECT_EQ(s2.epoch, s.epoch);  // other fields still recovered
}

// ---------------------------------------------------------------------------
// CkptmetaRejectEmpty — 0-byte .ckptmeta → throw (magic read fails)
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, CkptmetaRejectEmpty) {
    auto path = test_dir_ / "empty.ckptmeta";
    { std::ofstream f(path, std::ios::binary); }  // create empty

    EXPECT_THROW(mdb::gnn::ModelCheckpoint::read_ckptmeta(path),
                 std::runtime_error);
}

namespace {

// Helper: create a small GraphSAGEModel with deterministic seed
std::unique_ptr<mdb::gnn::GraphSAGEModel> make_test_model(int64_t seed = 42) {
    torch::manual_seed(seed);
    mdb::gnn::GraphSAGEConfig cfg;
    cfg.input_dim   = 16;
    cfg.hidden_dim  = 8;
    cfg.num_classes = 4;
    cfg.num_layers  = 2;
    cfg.dropout     = 0.5;
    cfg.normalize   = false;
    return std::make_unique<mdb::gnn::GraphSAGEModel>(cfg);
}

// Helper: mutate the optimizer once so its buffers (m/v) become non-trivial.
// Reseeds the global torch RNG so the generated input x (and any dropout
// sampling inside forward) are deterministic — required for the I2 fold-in
// check that compares parameters after a single step on two separately-loaded
// models.
void step_optimizer_once(mdb::gnn::GraphSAGEModel& model,
                         torch::optim::Adam& opt) {
    torch::manual_seed(1234);
    opt.zero_grad();
    auto x     = torch::randn({3, 16});
    auto ei    = torch::tensor({{0, 1}, {1, 2}}, torch::kLong);
    auto ei2   = torch::tensor({{0, 1}, {1, 2}}, torch::kLong);
    auto logits = model.forward(x, {ei, ei2}, std::vector<int64_t>{3, 3, 3});
    auto loss   = logits.sum();
    loss.backward();
    opt.step();
}

mdb::gnn::TrainingState state_from_model(const mdb::gnn::GraphSAGEModel& m) {
    mdb::gnn::TrainingState s;
    s.epoch              = 5;
    s.patience_counter   = 0;
    s.best_val_accuracy  = 0.72f;
    s.epoch_losses       = {1.0, 0.8, 0.7, 0.65, 0.6};
    s.input_dim          = m.config().input_dim;
    s.hidden_dim         = m.config().hidden_dim;
    s.num_classes        = m.config().num_classes;
    s.num_layers         = m.config().num_layers;
    s.dropout            = m.config().dropout;
    s.normalize          = m.config().normalize;
    s.model_type         = "graphsage";
    s.projection_name    = "test_projection";
    for (size_t i = 0; i < s.gnn_meta_hash.size(); ++i) {
        s.gnn_meta_hash[i] = static_cast<uint8_t>(i);
    }
    s.creation_time_unix       = 1735000000;
    s.total_training_time_sec  = 1.5;
    return s;
}

bool params_allclose(const mdb::gnn::GraphSAGEModel& a,
                     const mdb::gnn::GraphSAGEModel& b,
                     double rtol = 1e-6, double atol = 1e-6) {
    auto pa = a.parameters();
    auto pb = b.parameters();
    if (pa.size() != pb.size()) return false;
    for (size_t i = 0; i < pa.size(); ++i) {
        if (!torch::allclose(pa[i], pb[i], rtol, atol)) return false;
    }
    return true;
}

} // anon

// ---------------------------------------------------------------------------
// SaveFullRoundTrip — save_full + load_full preserves weights + optim state
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, SaveFullRoundTrip) {
    auto m1 = make_test_model(42);
    torch::optim::Adam opt1(m1->parameters(), torch::optim::AdamOptions(0.01));
    step_optimizer_once(*m1, opt1);

    auto base = test_dir_ / "best_model";
    auto state = state_from_model(*m1);
    mdb::gnn::ModelCheckpoint::save_full(*m1, opt1, base, state);

    ASSERT_TRUE(std::filesystem::exists(base.string() + ".pt"));
    ASSERT_TRUE(std::filesystem::exists(base.string() + ".ckptmeta"));

    auto m2 = make_test_model(99);  // different seed → different weights
    torch::optim::Adam opt2(m2->parameters(), torch::optim::AdamOptions(0.01));

    auto loaded = mdb::gnn::ModelCheckpoint::load_full(*m2, opt2, base);

    EXPECT_TRUE(params_allclose(*m1, *m2));
    EXPECT_EQ(loaded.epoch,             state.epoch);
    EXPECT_EQ(loaded.patience_counter,  state.patience_counter);
    EXPECT_FLOAT_EQ(loaded.best_val_accuracy, state.best_val_accuracy);
    EXPECT_EQ(loaded.epoch_losses,      state.epoch_losses);
    EXPECT_EQ(loaded.projection_name,   state.projection_name);

    // I2 fold-in: verify optimizer buffers (Adam m/v) were actually serialized.
    // If state was silently reset to zero, the next step from m2+opt2 would
    // diverge from the next step of m1+opt1 (which continues from the already-
    // mutated Adam state). Running one identical gradient step on both and
    // comparing params is the canonical check.
    step_optimizer_once(*m1, opt1);
    step_optimizer_once(*m2, opt2);
    EXPECT_TRUE(params_allclose(*m1, *m2));
}

// ---------------------------------------------------------------------------
// LoadFullRejectsMissingFile — load with non-existent basename throws
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, LoadFullRejectsMissingFile) {
    auto m1 = make_test_model(42);
    torch::optim::Adam opt1(m1->parameters(), torch::optim::AdamOptions(0.01));

    EXPECT_THROW(
        mdb::gnn::ModelCheckpoint::load_full(*m1, opt1, test_dir_ / "no_such"),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// SaveWeightsRoundTrip — save_weights / load_weights preserves params
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, SaveWeightsRoundTrip) {
    auto m1 = make_test_model(42);
    auto base  = test_dir_ / "weights_only";
    auto state = state_from_model(*m1);
    state.epoch_losses.clear();  // weights-only carries no loss history

    mdb::gnn::ModelCheckpoint::save_weights(*m1, base, state);

    auto m2 = make_test_model(99);
    auto loaded = mdb::gnn::ModelCheckpoint::load_weights(*m2, base);

    EXPECT_TRUE(params_allclose(*m1, *m2));
    EXPECT_EQ(loaded.model_type,      "graphsage");
    EXPECT_EQ(loaded.projection_name, state.projection_name);
}

// ---------------------------------------------------------------------------
// LoadFullRejectsWeightsOnly — load_full on a weights-only checkpoint throws
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, LoadFullRejectsWeightsOnly) {
    auto m1 = make_test_model(42);
    auto base  = test_dir_ / "wo_ckpt";
    auto state = state_from_model(*m1);

    mdb::gnn::ModelCheckpoint::save_weights(*m1, base, state);

    auto m2 = make_test_model(99);
    torch::optim::Adam opt2(m2->parameters(), torch::optim::AdamOptions(0.01));

    EXPECT_THROW(
        mdb::gnn::ModelCheckpoint::load_full(*m2, opt2, base),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// LoadWeightsAcceptsFullCheckpoint — inverse direction works (ignores optim)
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, LoadWeightsAcceptsFullCheckpoint) {
    auto m1 = make_test_model(42);
    torch::optim::Adam opt1(m1->parameters(), torch::optim::AdamOptions(0.01));
    step_optimizer_once(*m1, opt1);

    auto base  = test_dir_ / "full_ckpt";
    auto state = state_from_model(*m1);
    mdb::gnn::ModelCheckpoint::save_full(*m1, opt1, base, state);

    auto m2 = make_test_model(99);
    auto loaded = mdb::gnn::ModelCheckpoint::load_weights(*m2, base);

    EXPECT_TRUE(params_allclose(*m1, *m2));
    EXPECT_EQ(loaded.projection_name, state.projection_name);
}

// ---------------------------------------------------------------------------
// ValidateCompatAccepts — identical state + projection → no throw
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ValidateCompatAccepts) {
    mdb::gnn::GnnMeta m;
    m.feature_name = "features";
    m.feature_dim  = 16;
    m.num_nodes    = 100;
    m.num_classes  = 4;
    auto meta_path = test_dir_ / "gnn_meta.bin";
    m.write(meta_path);

    mdb::gnn::TrainingState s;
    s.projection_name = "cora";
    s.gnn_meta_hash   = mdb::gnn::ModelCheckpoint::compute_gnn_meta_hash(meta_path);

    EXPECT_NO_THROW(
        mdb::gnn::ModelCheckpoint::validate_compat(s, meta_path, "cora"));
}

// ---------------------------------------------------------------------------
// ValidateCompatProjectionMismatch — different projection name → throw
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ValidateCompatProjectionMismatch) {
    mdb::gnn::GnnMeta m;
    m.feature_name = "features";
    m.feature_dim  = 16;
    m.num_nodes    = 100;
    m.num_classes  = 4;
    auto meta_path = test_dir_ / "gnn_meta.bin";
    m.write(meta_path);

    mdb::gnn::TrainingState s;
    s.projection_name = "cora";
    s.gnn_meta_hash   = mdb::gnn::ModelCheckpoint::compute_gnn_meta_hash(meta_path);

    EXPECT_THROW(
        mdb::gnn::ModelCheckpoint::validate_compat(s, meta_path, "imdb"),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// ValidateCompatHashMismatch — gnn_meta.bin content changed → throw
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ValidateCompatHashMismatch) {
    mdb::gnn::GnnMeta m1;
    m1.feature_name = "v1";
    m1.feature_dim  = 16;
    m1.num_nodes    = 100;
    m1.num_classes  = 4;
    auto p1 = test_dir_ / "gnn_meta.bin";
    m1.write(p1);

    mdb::gnn::TrainingState s;
    s.projection_name = "cora";
    s.gnn_meta_hash   = mdb::gnn::ModelCheckpoint::compute_gnn_meta_hash(p1);

    // Regenerate with different feature_name — hash differs
    mdb::gnn::GnnMeta m2 = m1;
    m2.feature_name = "v2";
    m2.write(p1);

    EXPECT_THROW(
        mdb::gnn::ModelCheckpoint::validate_compat(s, p1, "cora"),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// ListEmptyDir — non-existent dir → empty vector, no throw
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ListEmptyDir) {
    auto v = mdb::gnn::ModelCheckpoint::list_checkpoints(test_dir_ / "no_such");
    EXPECT_TRUE(v.empty());
}

// ---------------------------------------------------------------------------
// ListIgnoresOrphans — a lone .pt without .ckptmeta is not listed
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ListIgnoresOrphans) {
    // Create an orphan .pt file
    std::ofstream(test_dir_ / "orphan.pt") << "fake torch data";

    auto v = mdb::gnn::ModelCheckpoint::list_checkpoints(test_dir_);
    EXPECT_TRUE(v.empty());
}

// ---------------------------------------------------------------------------
// ListSortedByTimeDesc — newest checkpoint first
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ListSortedByTimeDesc) {
    auto m   = make_test_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));

    auto s1 = state_from_model(*m); s1.creation_time_unix = 1000;
    auto s2 = state_from_model(*m); s2.creation_time_unix = 2000;
    auto s3 = state_from_model(*m); s3.creation_time_unix = 3000;

    mdb::gnn::ModelCheckpoint::save_full(*m, opt, test_dir_ / "c1", s1);
    mdb::gnn::ModelCheckpoint::save_full(*m, opt, test_dir_ / "c2", s2);
    mdb::gnn::ModelCheckpoint::save_full(*m, opt, test_dir_ / "c3", s3);

    auto v = mdb::gnn::ModelCheckpoint::list_checkpoints(test_dir_);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0].creation_time_unix, 3000u);
    EXPECT_EQ(v[1].creation_time_unix, 2000u);
    EXPECT_EQ(v[2].creation_time_unix, 1000u);
}

// ---------------------------------------------------------------------------
// ListNameFilter — filter returns 0 or 1 rows
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ListNameFilter) {
    auto m   = make_test_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto s = state_from_model(*m);
    mdb::gnn::ModelCheckpoint::save_full(*m, opt, test_dir_ / "best_model",  s);
    mdb::gnn::ModelCheckpoint::save_full(*m, opt, test_dir_ / "final_model", s);

    auto v = mdb::gnn::ModelCheckpoint::list_checkpoints(test_dir_, "best_model");
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].basename.filename().string(), "best_model");

    auto none = mdb::gnn::ModelCheckpoint::list_checkpoints(test_dir_, "nonexistent");
    EXPECT_TRUE(none.empty());
}

// ---------------------------------------------------------------------------
// ExistsTrueFalse
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ExistsTrueFalse) {
    auto m   = make_test_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto s = state_from_model(*m);
    mdb::gnn::ModelCheckpoint::save_full(*m, opt, test_dir_ / "present", s);

    EXPECT_TRUE (mdb::gnn::ModelCheckpoint::exists(test_dir_ / "present"));
    EXPECT_FALSE(mdb::gnn::ModelCheckpoint::exists(test_dir_ / "absent"));
}

// ---------------------------------------------------------------------------
// DeleteIdempotent — delete of missing is no-op; delete of present removes both files
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, DeleteIdempotent) {
    // No throw when neither file exists
    EXPECT_NO_THROW(
        mdb::gnn::ModelCheckpoint::delete_checkpoint(test_dir_ / "ghost"));

    // Now create one and delete it
    auto m   = make_test_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto s = state_from_model(*m);
    mdb::gnn::ModelCheckpoint::save_full(*m, opt, test_dir_ / "doomed", s);

    mdb::gnn::ModelCheckpoint::delete_checkpoint(test_dir_ / "doomed");
    EXPECT_FALSE(std::filesystem::exists(test_dir_ / "doomed.pt"));
    EXPECT_FALSE(std::filesystem::exists(test_dir_ / "doomed.ckptmeta"));

    // Second call is idempotent
    EXPECT_NO_THROW(
        mdb::gnn::ModelCheckpoint::delete_checkpoint(test_dir_ / "doomed"));
}

// ---------------------------------------------------------------------------
// ValidateCompatMissingMetaFile — fold-in from Task 2.3 review
// compute_gnn_meta_hash throws cleanly when the file is absent
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointTest, ValidateCompatMissingMetaFile) {
    mdb::gnn::TrainingState s;
    s.projection_name = "cora";
    // s.gnn_meta_hash left all-zero (default)

    EXPECT_THROW(
        mdb::gnn::ModelCheckpoint::validate_compat(
            s, test_dir_ / "does_not_exist.bin", "cora"),
        std::runtime_error);
}
