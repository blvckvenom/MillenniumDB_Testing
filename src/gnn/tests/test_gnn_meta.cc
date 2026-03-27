#include "test_helpers.h"

#include <fstream>
#include <stdexcept>

#include "gnn/projection/gnn_meta.h"

using GnnMetaTest = GnnStorageTest;

// ===========================================================================
// WriteAndReadRoundtrip
// ===========================================================================

TEST_F(GnnMetaTest, WriteAndReadRoundtrip) {
    auto meta_path = test_dir_ / "gnn_meta.bin";

    mdb::gnn::GnnMeta m;
    m.feature_name = "node_features";
    m.feature_dim  = 128;
    m.num_nodes    = 1000;
    m.num_classes  = 7;
    m.has_labels   = true;
    m.has_splits   = false;

    m.write(meta_path);

    auto m2 = mdb::gnn::GnnMeta::read(meta_path);

    EXPECT_EQ(m2.feature_name, "node_features");
    EXPECT_EQ(m2.feature_dim,  128u);
    EXPECT_EQ(m2.num_nodes,    1000u);
    EXPECT_EQ(m2.num_classes,  7u);
    EXPECT_TRUE(m2.has_labels);
    EXPECT_FALSE(m2.has_splits);
}

// ===========================================================================
// ExistsReturnsFalseWhenMissing
// ===========================================================================

TEST_F(GnnMetaTest, ExistsReturnsFalseWhenMissing) {
    // test_dir_ exists but gnn_meta.bin has not been written
    EXPECT_FALSE(mdb::gnn::GnnMeta::exists(test_dir_));
}

// ===========================================================================
// ExistsReturnsTrueWhenPresent
// ===========================================================================

TEST_F(GnnMetaTest, ExistsReturnsTrueWhenPresent) {
    mdb::gnn::GnnMeta m;
    m.feature_name = "features";
    m.feature_dim  = 64;
    m.num_nodes    = 500;
    m.num_classes  = 3;
    m.has_labels   = false;
    m.has_splits   = true;

    m.write(test_dir_ / "gnn_meta.bin");

    EXPECT_TRUE(mdb::gnn::GnnMeta::exists(test_dir_));
}

// ===========================================================================
// InvalidMagicThrows
// ===========================================================================

TEST_F(GnnMetaTest, InvalidMagicThrows) {
    auto bad_path = test_dir_ / "bad_meta.bin";

    // Write garbage bytes
    {
        std::ofstream f(bad_path, std::ios::binary);
        const char garbage[] = "XXXX\x00\x00\x00\x00\x01\x00\x00\x00";
        f.write(garbage, sizeof(garbage) - 1);
    }

    EXPECT_THROW(mdb::gnn::GnnMeta::read(bad_path), std::runtime_error);
}

// ===========================================================================
// EmptyFeatureName
// ===========================================================================

TEST_F(GnnMetaTest, EmptyFeatureName) {
    auto meta_path = test_dir_ / "gnn_meta_empty_name.bin";

    mdb::gnn::GnnMeta m;
    m.feature_name = "";
    m.feature_dim  = 32;
    m.num_nodes    = 100;
    m.num_classes  = 2;
    m.has_labels   = false;
    m.has_splits   = false;

    m.write(meta_path);

    auto m2 = mdb::gnn::GnnMeta::read(meta_path);
    EXPECT_EQ(m2.feature_name, "");
    EXPECT_EQ(m2.feature_dim,  32u);
    EXPECT_EQ(m2.num_nodes,    100u);
    EXPECT_EQ(m2.num_classes,  2u);
    EXPECT_FALSE(m2.has_labels);
    EXPECT_FALSE(m2.has_splits);
}

// ===========================================================================
// WriteCreatesParentDirs
// ===========================================================================

TEST_F(GnnMetaTest, WriteCreatesParentDirs) {
    // Write to a nested path that does not exist yet
    auto nested = test_dir_ / "subdir" / "deep" / "gnn_meta.bin";

    mdb::gnn::GnnMeta m;
    m.feature_name = "feat";
    m.feature_dim  = 16;
    m.num_nodes    = 50;
    m.num_classes  = 2;

    // Should not throw — write() must create parent dirs
    EXPECT_NO_THROW(m.write(nested));
    EXPECT_TRUE(std::filesystem::exists(nested));
}

// ===========================================================================
// BothFlagsTrue
// ===========================================================================

TEST_F(GnnMetaTest, BothFlagsTrue) {
    auto meta_path = test_dir_ / "gnn_meta_flags.bin";

    mdb::gnn::GnnMeta m;
    m.feature_name = "x";
    m.feature_dim  = 8;
    m.num_nodes    = 10;
    m.num_classes  = 5;
    m.has_labels   = true;
    m.has_splits   = true;

    m.write(meta_path);
    auto m2 = mdb::gnn::GnnMeta::read(meta_path);

    EXPECT_TRUE(m2.has_labels);
    EXPECT_TRUE(m2.has_splits);
}

// ===========================================================================
// UnsupportedVersionThrows
// ===========================================================================

TEST_F(GnnMetaTest, UnsupportedVersionThrows) {
    auto bad_path = test_dir_ / "bad_version.bin";

    // Write a valid magic but version=99
    {
        std::ofstream f(bad_path, std::ios::binary);
        // magic
        const uint8_t magic[8] = {'G','N','N','M','\0','\0','\0','\0'};
        f.write(reinterpret_cast<const char*>(magic), 8);
        // version = 99
        uint32_t bad_ver = 99;
        f.write(reinterpret_cast<const char*>(&bad_ver), 4);
        // remaining fields (zeros)
        const uint8_t zeros[28] = {};
        f.write(reinterpret_cast<const char*>(zeros), 28);
    }

    EXPECT_THROW(mdb::gnn::GnnMeta::read(bad_path), std::runtime_error);
}
