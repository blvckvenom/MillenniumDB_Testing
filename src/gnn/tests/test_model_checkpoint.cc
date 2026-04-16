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
