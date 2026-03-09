#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/feature_matrix_header.h"
#include "gnn/storage/gnn_dtype.h"

namespace fs = std::filesystem;
using namespace mdb::gnn;

class FeatureMatrixTest : public ::testing::Test {
protected:
    fs::path test_dir_;

    void SetUp() override {
        test_dir_ = fs::temp_directory_path() / "mdb_test_fmat";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    fs::path test_path(const std::string& name) {
        return test_dir_ / name;
    }
};

// --- Header Tests ---

TEST(FeatureMatrixHeaderTest, MakeAndValidate) {
    auto h = FeatureMatrixHeader::make(100, 768, GnnDtype::FLOAT32);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.magic, FeatureMatrixHeader::MAGIC);
    EXPECT_EQ(h.version, FeatureMatrixHeader::VERSION);
    EXPECT_EQ(h.num_rows, 100u);
    EXPECT_EQ(h.num_cols, 768u);
    EXPECT_EQ(h.get_dtype(), GnnDtype::FLOAT32);
    EXPECT_EQ(h.row_bytes(), 768 * sizeof(float));
    EXPECT_EQ(h.data_bytes(), 100 * 768 * sizeof(float));
}

TEST(FeatureMatrixHeaderTest, InvalidMagic) {
    auto h = FeatureMatrixHeader::make(10, 10, GnnDtype::FLOAT32);
    h.magic = 0xDEADBEEF;
    EXPECT_FALSE(h.is_valid());
}

TEST(FeatureMatrixHeaderTest, ZeroRows) {
    auto h = FeatureMatrixHeader::make(0, 10, GnnDtype::FLOAT32);
    EXPECT_FALSE(h.is_valid());
}

TEST(FeatureMatrixHeaderTest, SizeIs64Bytes) {
    EXPECT_EQ(sizeof(FeatureMatrixHeader), 64u);
}

// --- Create + Open Roundtrip ---

TEST_F(FeatureMatrixTest, CreateAndOpenSmallFloat32) {
    const uint64_t N = 5, D = 3;
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f
    };

    auto path = test_path("small.fmat");
    auto fm_write = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, data.data());
    EXPECT_EQ(fm_write.num_rows(), N);
    EXPECT_EQ(fm_write.num_cols(), D);
    EXPECT_EQ(fm_write.dtype(), GnnDtype::FLOAT32);

    // Re-open from disk
    auto fm_read = FeatureMatrix::open(path);
    EXPECT_EQ(fm_read.num_rows(), N);
    EXPECT_EQ(fm_read.num_cols(), D);
    EXPECT_EQ(fm_read.dtype(), GnnDtype::FLOAT32);

    // Verify data
    for (uint64_t i = 0; i < N; ++i) {
        const float* row = fm_read.row_as<float>(i);
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(row[j], data[i * D + j])
                << "Mismatch at row=" << i << " col=" << j;
        }
    }
}

TEST_F(FeatureMatrixTest, CreateAndOpenFloat64) {
    const uint64_t N = 3, D = 2;
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto path = test_path("f64.fmat");
    auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT64, data.data());

    auto fm2 = FeatureMatrix::open(path);
    for (uint64_t i = 0; i < N; ++i) {
        const double* row = fm2.row_as<double>(i);
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_DOUBLE_EQ(row[j], data[i * D + j]);
        }
    }
}

TEST_F(FeatureMatrixTest, OpenNonExistentThrows) {
    EXPECT_THROW(FeatureMatrix::open(test_path("nonexistent.fmat")), std::runtime_error);
}

TEST_F(FeatureMatrixTest, RowOutOfBoundsThrows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm = FeatureMatrix::create(test_path("bounds.fmat"), N, D, GnnDtype::FLOAT32, data.data());
    EXPECT_THROW(fm.row(N), std::out_of_range);
    EXPECT_THROW(fm.row(N + 100), std::out_of_range);
    EXPECT_NO_THROW(fm.row(N - 1));
}

// --- Scan ---

TEST_F(FeatureMatrixTest, ScanVisitsAllRowsInOrder) {
    const uint64_t N = 10, D = 4;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto fm = FeatureMatrix::create(test_path("scan.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> visited_rows;
    fm.scan([&](uint64_t row_id, const void* row_data) {
        visited_rows.push_back(row_id);
        const float* row = static_cast<const float*>(row_data);
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(row[j], data[row_id * D + j]);
        }
    });

    ASSERT_EQ(visited_rows.size(), N);
    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_EQ(visited_rows[i], i);
    }
}

// --- Extract Rows ---

TEST_F(FeatureMatrixTest, ExtractRowsPreservesInputOrder) {
    const uint64_t N = 10, D = 3;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto fm = FeatureMatrix::create(test_path("extract.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    // Request rows out of order: 7, 2, 9, 0
    std::vector<uint64_t> row_ids = {7, 2, 9, 0};
    std::vector<float> out(row_ids.size() * D);
    fm.extract_rows(row_ids, out.data());

    // Output should be in INPUT order: row7, row2, row9, row0
    for (size_t i = 0; i < row_ids.size(); ++i) {
        uint64_t src_row = row_ids[i];
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(out[i * D + j], data[src_row * D + j])
                << "Mismatch at output_row=" << i << " (source_row=" << src_row << ") col=" << j;
        }
    }
}

TEST_F(FeatureMatrixTest, ExtractRowsEmpty) {
    const uint64_t N = 5, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm = FeatureMatrix::create(test_path("extract_empty.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> empty_ids;
    std::vector<float> out;
    EXPECT_NO_THROW(fm.extract_rows(empty_ids, out.data()));
}

// --- Streaming Create ---

TEST_F(FeatureMatrixTest, CreateStreamingRoundtrip) {
    const uint64_t N = 8, D = 4;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i * 0.5f);

    auto path = test_path("streaming.fmat");
    auto fm = FeatureMatrix::create_streaming(path, N, D, GnnDtype::FLOAT32,
        [&](uint64_t row_id, void* dest, uint64_t rb) {
            std::memcpy(dest, &data[row_id * D], rb);
        });

    auto fm2 = FeatureMatrix::open(path);
    for (uint64_t i = 0; i < N; ++i) {
        const float* row = fm2.row_as<float>(i);
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(row[j], data[i * D + j]);
        }
    }
}

// --- File Persistence ---

TEST_F(FeatureMatrixTest, PersistsAcrossOpenClose) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto path = test_path("persist.fmat");

    // Create and immediately destroy
    { auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, data.data()); }

    // Open fresh — data should still be there
    auto fm = FeatureMatrix::open(path);
    EXPECT_EQ(fm.num_rows(), N);
    const float* row1 = fm.row_as<float>(1);
    EXPECT_FLOAT_EQ(row1[0], 3.0f);
    EXPECT_FLOAT_EQ(row1[1], 4.0f);
}
