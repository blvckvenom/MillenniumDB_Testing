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

// Helper: mutate the optimizer once so its buffers (m/v) become non-trivial
void step_optimizer_once(mdb::gnn::GraphSAGEModel& model,
                         torch::optim::Adam& opt) {
    opt.zero_grad();
    auto x     = torch::randn({3, 16});
    auto ei    = torch::tensor({{0, 1}, {1, 2}}, torch::kLong);
    auto ei2   = torch::tensor({{0, 1}, {1, 2}}, torch::kLong);
    auto logits = model.forward(x, {ei, ei2}, 3);
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
