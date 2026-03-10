#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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

// ===========================================================================
// Edge Case Tests
// ===========================================================================

// --- Moved-from object ---

TEST_F(FeatureMatrixTest, MovedFromObjectThrowsOnAccess) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm1 = FeatureMatrix::create(test_path("move_src.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    // Move fm1 into fm2
    auto fm2 = std::move(fm1);

    // fm2 should work fine
    EXPECT_EQ(fm2.num_rows(), N);
    EXPECT_NO_THROW(fm2.row(0));

    // fm1 is moved-from: num_rows should be 0 (header zeroed)
    EXPECT_EQ(fm1.num_rows(), 0u);
    // row() should throw (null mmap_ptr_)
    EXPECT_THROW(fm1.row(0), std::runtime_error);
}

TEST_F(FeatureMatrixTest, MoveAssignReleasesOldMapping) {
    const uint64_t N = 2, D = 2;
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {10.0f, 20.0f, 30.0f, 40.0f};

    auto fm = FeatureMatrix::create(test_path("assign_a.fmat"), N, D, GnnDtype::FLOAT32, data_a.data());
    EXPECT_FLOAT_EQ(fm.row_as<float>(0)[0], 1.0f);

    // Move-assign over it — old mapping should be released (no leak)
    fm = FeatureMatrix::create(test_path("assign_b.fmat"), N, D, GnnDtype::FLOAT32, data_b.data());
    EXPECT_FLOAT_EQ(fm.row_as<float>(0)[0], 10.0f);
}

// --- Single row / single column ---

TEST_F(FeatureMatrixTest, SingleRowSingleCol) {
    const uint64_t N = 1, D = 1;
    std::vector<float> data = {42.0f};
    auto fm = FeatureMatrix::create(test_path("1x1.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    EXPECT_EQ(fm.num_rows(), 1u);
    EXPECT_EQ(fm.num_cols(), 1u);
    EXPECT_FLOAT_EQ(fm.row_as<float>(0)[0], 42.0f);
    EXPECT_THROW(fm.row(1), std::out_of_range);

    // Scan should visit exactly 1 row
    uint64_t count = 0;
    fm.scan([&](uint64_t row_id, const void*) { ++count; EXPECT_EQ(row_id, 0u); });
    EXPECT_EQ(count, 1u);
}

TEST_F(FeatureMatrixTest, SingleRowManyColumns) {
    const uint64_t N = 1, D = 1000;
    std::vector<float> data(D);
    for (uint64_t j = 0; j < D; ++j) data[j] = static_cast<float>(j);

    auto fm = FeatureMatrix::create(test_path("1xD.fmat"), N, D, GnnDtype::FLOAT32, data.data());
    EXPECT_EQ(fm.row_bytes(), D * sizeof(float));

    const float* row = fm.row_as<float>(0);
    for (uint64_t j = 0; j < D; ++j) {
        EXPECT_FLOAT_EQ(row[j], static_cast<float>(j));
    }
}

TEST_F(FeatureMatrixTest, ManyRowsSingleColumn) {
    const uint64_t N = 500, D = 1;
    std::vector<float> data(N);
    for (uint64_t i = 0; i < N; ++i) data[i] = static_cast<float>(i * 3);

    auto fm = FeatureMatrix::create(test_path("Nx1.fmat"), N, D, GnnDtype::FLOAT32, data.data());
    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(fm.row_as<float>(i)[0], static_cast<float>(i * 3));
    }
}

// --- create() validation ---

TEST_F(FeatureMatrixTest, CreateZeroRowsThrows) {
    std::vector<float> data = {1.0f};
    EXPECT_THROW(
        FeatureMatrix::create(test_path("zero_rows.fmat"), 0, 1, GnnDtype::FLOAT32, data.data()),
        std::invalid_argument);
}

TEST_F(FeatureMatrixTest, CreateZeroColsThrows) {
    std::vector<float> data = {1.0f};
    EXPECT_THROW(
        FeatureMatrix::create(test_path("zero_cols.fmat"), 1, 0, GnnDtype::FLOAT32, data.data()),
        std::invalid_argument);
}

TEST_F(FeatureMatrixTest, CreateNullDataThrows) {
    EXPECT_THROW(
        FeatureMatrix::create(test_path("null.fmat"), 1, 1, GnnDtype::FLOAT32, nullptr),
        std::invalid_argument);
}

// --- open() validation ---

TEST_F(FeatureMatrixTest, OpenCorruptedHeaderThrows) {
    auto path = test_path("corrupt.fmat");
    // Write garbage that looks like a file but has bad magic
    std::vector<char> garbage(128, 0);
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(garbage.data(), garbage.size());
    ofs.close();

    EXPECT_THROW(FeatureMatrix::open(path), std::runtime_error);
}

TEST_F(FeatureMatrixTest, OpenTruncatedFileThrows) {
    const uint64_t N = 10, D = 100;
    std::vector<float> data(N * D, 1.0f);
    auto path = test_path("truncated.fmat");

    // Create valid file
    { auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, data.data()); }

    // Truncate it (keep header but remove data)
    fs::resize_file(path, FeatureMatrixHeader::SIZE + 10);

    EXPECT_THROW(FeatureMatrix::open(path), std::runtime_error);
}

// --- extract_rows() validation ---

TEST_F(FeatureMatrixTest, ExtractRowsOutOfBoundsThrows) {
    const uint64_t N = 5, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm = FeatureMatrix::create(test_path("extract_oob.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> bad_ids = {1, 3, 99};  // 99 is out of range
    std::vector<float> out(bad_ids.size() * D);
    EXPECT_THROW(fm.extract_rows(bad_ids, out.data()), std::out_of_range);
}

TEST_F(FeatureMatrixTest, ExtractRowsDuplicateIds) {
    const uint64_t N = 5, D = 2;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto fm = FeatureMatrix::create(test_path("extract_dup.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    // Duplicate row IDs should work — each output position gets its own copy
    std::vector<uint64_t> ids = {2, 2, 2};
    std::vector<float> out(ids.size() * D);
    EXPECT_NO_THROW(fm.extract_rows(ids, out.data()));
    for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i * D + 0], data[2 * D + 0]);
        EXPECT_FLOAT_EQ(out[i * D + 1], data[2 * D + 1]);
    }
}

// --- create_reordered() ---

TEST_F(FeatureMatrixTest, CreateReorderedRoundtrip) {
    const uint64_t N = 4, D = 3;
    std::vector<float> data = {
        10.0f, 11.0f, 12.0f,  // row 0
        20.0f, 21.0f, 22.0f,  // row 1
        30.0f, 31.0f, 32.0f,  // row 2
        40.0f, 41.0f, 42.0f,  // row 3
    };
    auto src = FeatureMatrix::create(test_path("reorder_src.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    // Reverse permutation: output[0]=row3, output[1]=row2, output[2]=row1, output[3]=row0
    std::vector<uint64_t> perm = {3, 2, 1, 0};
    auto dst = FeatureMatrix::create_reordered(src, perm, test_path("reorder_dst.fmat"));

    EXPECT_EQ(dst.num_rows(), N);
    EXPECT_FLOAT_EQ(dst.row_as<float>(0)[0], 40.0f);  // was row 3
    EXPECT_FLOAT_EQ(dst.row_as<float>(1)[0], 30.0f);  // was row 2
    EXPECT_FLOAT_EQ(dst.row_as<float>(2)[0], 20.0f);  // was row 1
    EXPECT_FLOAT_EQ(dst.row_as<float>(3)[0], 10.0f);  // was row 0
}

TEST_F(FeatureMatrixTest, CreateReorderedBadSizeThrows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto src = FeatureMatrix::create(test_path("reorder_bad.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> bad_perm = {0, 1};  // size 2 != N=3
    EXPECT_THROW(
        FeatureMatrix::create_reordered(src, bad_perm, test_path("reorder_bad_out.fmat")),
        std::invalid_argument);
}

TEST_F(FeatureMatrixTest, CreateReorderedOutOfBoundsThrows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto src = FeatureMatrix::create(test_path("reorder_oob.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> bad_perm = {0, 1, 99};  // 99 out of range
    EXPECT_THROW(
        FeatureMatrix::create_reordered(src, bad_perm, test_path("reorder_oob_out.fmat")),
        std::out_of_range);
}

// --- All dtypes ---

TEST_F(FeatureMatrixTest, AllDtypesRoundtrip) {
    auto path = test_path("dtypes.fmat");

    // INT32
    {
        std::vector<int32_t> data = {-1, 0, 1, 42};
        auto fm = FeatureMatrix::create(path, 2, 2, GnnDtype::INT32, data.data());
        EXPECT_EQ(fm.row_as<int32_t>(1)[0], 1);
        EXPECT_EQ(fm.row_as<int32_t>(1)[1], 42);
    }

    // INT64
    {
        std::vector<int64_t> data = {-100, 200};
        auto fm = FeatureMatrix::create(path, 1, 2, GnnDtype::INT64, data.data());
        EXPECT_EQ(fm.row_as<int64_t>(0)[0], -100);
        EXPECT_EQ(fm.row_as<int64_t>(0)[1], 200);
    }

    // UINT8
    {
        std::vector<uint8_t> data = {0, 128, 255};
        auto fm = FeatureMatrix::create(path, 1, 3, GnnDtype::UINT8, data.data());
        EXPECT_EQ(fm.row_as<uint8_t>(0)[0], 0);
        EXPECT_EQ(fm.row_as<uint8_t>(0)[1], 128);
        EXPECT_EQ(fm.row_as<uint8_t>(0)[2], 255);
    }

    // BOOL
    {
        uint8_t data[] = {1, 0, 1};
        auto fm = FeatureMatrix::create(path, 1, 3, GnnDtype::BOOL, data);
        EXPECT_EQ(fm.row_as<uint8_t>(0)[0], 1);
        EXPECT_EQ(fm.row_as<uint8_t>(0)[1], 0);
    }
}
