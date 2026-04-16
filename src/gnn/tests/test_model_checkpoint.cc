#include "test_helpers.h"

#include <array>
#include <cstdint>
#include <fstream>

#include "gnn/output/model_checkpoint.h"
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

    // Truncate to 50 bytes (should be ~564)
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
