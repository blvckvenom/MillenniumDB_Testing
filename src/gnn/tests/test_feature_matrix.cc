#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/feature_matrix_header.h"
#include "gnn/storage/gnn_dtype.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/object_id.h"

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

// ===========================================================================
// RowMapping Tests
// ===========================================================================

TEST_F(FeatureMatrixTest, RowMappingRoundtrip) {
    std::vector<ObjectId> ids;
    for (uint64_t i = 0; i < 100; ++i) {
        ids.push_back(ObjectId(i * 7 + 3)); // arbitrary IDs
    }

    auto path = test_path("mapping.rmap");
    auto rm_write = RowMapping::create(path, ids);
    EXPECT_EQ(rm_write.size(), 100u);

    auto rm_read = RowMapping::open(path);
    EXPECT_EQ(rm_read.size(), 100u);

    for (uint64_t i = 0; i < 100; ++i) {
        EXPECT_EQ(rm_read.get(i).id, ids[i].id);
    }
}

TEST_F(FeatureMatrixTest, RowMappingFind) {
    std::vector<ObjectId> ids = {ObjectId(42), ObjectId(99), ObjectId(7)};
    auto rm = RowMapping::create(test_path("find.rmap"), ids);

    auto idx = rm.find(ObjectId(99));
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(idx.value(), 1u);

    auto miss = rm.find(ObjectId(12345));
    EXPECT_FALSE(miss.has_value());
}

TEST_F(FeatureMatrixTest, RowMappingEmpty) {
    std::vector<ObjectId> ids;
    auto rm = RowMapping::create(test_path("empty.rmap"), ids);
    EXPECT_EQ(rm.size(), 0u);
    EXPECT_THROW(rm.get(0), std::out_of_range);
    EXPECT_FALSE(rm.find(ObjectId(1)).has_value());
}

TEST_F(FeatureMatrixTest, RowMappingPersistence) {
    std::vector<ObjectId> ids = {ObjectId(100), ObjectId(200)};
    auto path = test_path("persist.rmap");

    // Create and destroy
    { auto rm = RowMapping::create(path, ids); }

    // Re-open
    auto rm = RowMapping::open(path);
    EXPECT_EQ(rm.size(), 2u);
    EXPECT_EQ(rm.get(0).id, 100u);
    EXPECT_EQ(rm.get(1).id, 200u);
}

TEST_F(FeatureMatrixTest, RowMappingOpenNonExistentThrows) {
    EXPECT_THROW(RowMapping::open(test_path("nope.rmap")), std::runtime_error);
}

TEST_F(FeatureMatrixTest, RowMappingOutOfBoundsThrows) {
    std::vector<ObjectId> ids = {ObjectId(1)};
    auto rm = RowMapping::create(test_path("oob.rmap"), ids);
    EXPECT_THROW(rm.get(1), std::out_of_range);
    EXPECT_THROW(rm.get(999), std::out_of_range);
    EXPECT_NO_THROW(rm.get(0));
}

TEST_F(FeatureMatrixTest, RowMappingMovedFrom) {
    std::vector<ObjectId> ids = {ObjectId(42)};
    auto rm1 = RowMapping::create(test_path("move.rmap"), ids);
    auto rm2 = std::move(rm1);

    EXPECT_EQ(rm2.size(), 1u);
    EXPECT_EQ(rm2.get(0).id, 42u);

    EXPECT_EQ(rm1.size(), 0u);
    EXPECT_THROW(rm1.get(0), std::runtime_error);
}

// ===========================================================================
// Additional Edge Case Tests (robustness)
// ===========================================================================

// --- RowMapping edge cases ---

TEST_F(FeatureMatrixTest, RowMappingCorruptedHeaderThrows) {
    auto path = test_path("corrupt.rmap");
    // Write a file with bad magic but enough size for header
    std::vector<char> garbage(32, static_cast<char>(0xFF));
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(garbage.data(), garbage.size());
    }
    EXPECT_THROW(RowMapping::open(path), std::runtime_error);
}

TEST_F(FeatureMatrixTest, RowMappingTruncatedFileThrows) {
    auto path = test_path("truncated.rmap");

    // Create a valid mapping with 10 entries
    std::vector<ObjectId> ids;
    for (uint64_t i = 0; i < 10; ++i) ids.push_back(ObjectId(i));
    { auto rm = RowMapping::create(path, ids); }

    // Truncate to just the header + 2 entries (should expect 10)
    fs::resize_file(path, RowMapping::HEADER_SIZE + 2 * sizeof(ObjectId));

    EXPECT_THROW(RowMapping::open(path), std::runtime_error);
}

TEST_F(FeatureMatrixTest, RowMappingFileTooSmallForHeaderThrows) {
    auto path = test_path("tiny.rmap");
    // Write a file smaller than the 16-byte header
    std::ofstream ofs(path, std::ios::binary);
    char byte = 0;
    ofs.write(&byte, 1);
    ofs.close();

    EXPECT_THROW(RowMapping::open(path), std::runtime_error);
}

TEST_F(FeatureMatrixTest, RowMappingLargeRoundtrip) {
    const uint64_t N = 10000;
    std::vector<ObjectId> ids;
    ids.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        ids.push_back(ObjectId(i * 13 + 7)); // arbitrary but deterministic
    }

    auto path = test_path("large.rmap");
    { auto rm = RowMapping::create(path, ids); }

    auto rm = RowMapping::open(path);
    EXPECT_EQ(rm.size(), N);

    // Spot-check boundaries
    EXPECT_EQ(rm.get(0).id, ids[0].id);
    EXPECT_EQ(rm.get(N / 2).id, ids[N / 2].id);
    EXPECT_EQ(rm.get(N - 1).id, ids[N - 1].id);
    EXPECT_THROW(rm.get(N), std::out_of_range);

    // find() at boundaries
    EXPECT_TRUE(rm.find(ids[0]).has_value());
    EXPECT_TRUE(rm.find(ids[N - 1]).has_value());
    EXPECT_FALSE(rm.find(ObjectId(0xDEADBEEF)).has_value());
}

TEST_F(FeatureMatrixTest, RowMappingMoveAssignReleasesOld) {
    std::vector<ObjectId> ids_a = {ObjectId(10), ObjectId(20)};
    std::vector<ObjectId> ids_b = {ObjectId(30), ObjectId(40), ObjectId(50)};

    auto rm = RowMapping::create(test_path("assign_a.rmap"), ids_a);
    EXPECT_EQ(rm.size(), 2u);
    EXPECT_EQ(rm.get(0).id, 10u);

    // Move-assign over it — old mapping should be released (no leak)
    rm = RowMapping::create(test_path("assign_b.rmap"), ids_b);
    EXPECT_EQ(rm.size(), 3u);
    EXPECT_EQ(rm.get(0).id, 30u);
}

TEST_F(FeatureMatrixTest, RowMappingFindFirstOccurrence) {
    // If duplicates exist, find() should return the first index
    std::vector<ObjectId> ids = {ObjectId(5), ObjectId(10), ObjectId(5), ObjectId(10)};
    auto rm = RowMapping::create(test_path("dup_find.rmap"), ids);

    auto idx = rm.find(ObjectId(5));
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(idx.value(), 0u);  // first occurrence

    auto idx2 = rm.find(ObjectId(10));
    ASSERT_TRUE(idx2.has_value());
    EXPECT_EQ(idx2.value(), 1u);  // first occurrence
}

// --- FeatureMatrix edge cases ---

TEST_F(FeatureMatrixTest, ScanOnMovedFromThrows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm1 = FeatureMatrix::create(test_path("scan_moved.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    auto fm2 = std::move(fm1);

    // fm1 is moved-from — scan should throw
    EXPECT_THROW(fm1.scan([](uint64_t, const void*) {}), std::runtime_error);

    // fm2 should scan fine
    uint64_t count = 0;
    EXPECT_NO_THROW(fm2.scan([&](uint64_t, const void*) { ++count; }));
    EXPECT_EQ(count, N);
}

TEST_F(FeatureMatrixTest, ExtractRowsOnMovedFromThrows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm1 = FeatureMatrix::create(test_path("extract_moved.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    auto fm2 = std::move(fm1);

    std::vector<uint64_t> ids = {0};
    std::vector<float> out(D);
    EXPECT_THROW(fm1.extract_rows(ids, out.data()), std::runtime_error);
}

TEST_F(FeatureMatrixTest, ExtractRowsLargeRandomAccess) {
    // Test with 1000 rows, extracting random subset in reverse order
    const uint64_t N = 1000, D = 8;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto fm = FeatureMatrix::create(test_path("large_extract.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    // Extract rows in reverse order
    std::vector<uint64_t> ids;
    for (uint64_t i = N; i > 0; i -= 10) {
        ids.push_back(i - 1);  // 999, 989, 979, ..., 9
    }

    std::vector<float> out(ids.size() * D);
    EXPECT_NO_THROW(fm.extract_rows(ids, out.data()));

    // Verify each extracted row matches source
    for (size_t i = 0; i < ids.size(); ++i) {
        uint64_t src_row = ids[i];
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(out[i * D + j], data[src_row * D + j])
                << "Mismatch at output[" << i << "] (source row " << src_row << ") col " << j;
        }
    }
}

TEST_F(FeatureMatrixTest, CreateStreamingZeroDimsThrows) {
    EXPECT_THROW(
        FeatureMatrix::create_streaming(test_path("stream_zero.fmat"), 0, 10, GnnDtype::FLOAT32,
            [](uint64_t, void*, uint64_t) {}),
        std::invalid_argument);

    EXPECT_THROW(
        FeatureMatrix::create_streaming(test_path("stream_zero2.fmat"), 10, 0, GnnDtype::FLOAT32,
            [](uint64_t, void*, uint64_t) {}),
        std::invalid_argument);
}

TEST_F(FeatureMatrixTest, CreateReorderedIdentityPermutation) {
    const uint64_t N = 5, D = 3;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto src = FeatureMatrix::create(test_path("identity_src.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    // Identity permutation — output should be identical to input
    std::vector<uint64_t> perm = {0, 1, 2, 3, 4};
    auto dst = FeatureMatrix::create_reordered(src, perm, test_path("identity_dst.fmat"));

    for (uint64_t i = 0; i < N; ++i) {
        const float* src_row = src.row_as<float>(i);
        const float* dst_row = dst.row_as<float>(i);
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(src_row[j], dst_row[j]);
        }
    }
}

TEST_F(FeatureMatrixTest, HeaderFileLayoutSize) {
    // Verify the header is exactly 64 bytes (critical for mmap alignment)
    EXPECT_EQ(FeatureMatrixHeader::SIZE, 64u);
    EXPECT_EQ(sizeof(FeatureMatrixHeader), FeatureMatrixHeader::SIZE);

    // Verify RowMapping header is exactly 16 bytes
    EXPECT_EQ(RowMapping::HEADER_SIZE, 16u);
}

TEST_F(FeatureMatrixTest, OpenFileWithExtraTrailingBytesSucceeds) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto path = test_path("extra_bytes.fmat");

    { auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, data.data()); }

    // Append extra bytes to file — open() should still work
    // (file_size >= expected is OK, file_size < expected is an error)
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::app);
        char padding[128] = {};
        ofs.write(padding, sizeof(padding));
    }

    auto fm = FeatureMatrix::open(path);
    EXPECT_EQ(fm.num_rows(), N);
    EXPECT_EQ(fm.num_cols(), D);
    EXPECT_FLOAT_EQ(fm.row_as<float>(0)[0], 1.0f);
}

TEST_F(FeatureMatrixTest, RowMappingOpenWithExtraTrailingBytesSucceeds) {
    std::vector<ObjectId> ids = {ObjectId(1), ObjectId(2)};
    auto path = test_path("extra.rmap");
    { auto rm = RowMapping::create(path, ids); }

    // Append extra bytes — should not break open()
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::app);
        char padding[64] = {};
        ofs.write(padding, sizeof(padding));
    }

    auto rm = RowMapping::open(path);
    EXPECT_EQ(rm.size(), 2u);
    EXPECT_EQ(rm.get(0).id, 1u);
    EXPECT_EQ(rm.get(1).id, 2u);
}

TEST_F(FeatureMatrixTest, ScanAllDtypesVerifyRowSize) {
    // Verify that row_bytes() matches dtype_size * num_cols for all dtypes
    struct DtypeInfo {
        GnnDtype dtype;
        size_t expected_elem_size;
    };
    std::vector<DtypeInfo> infos = {
        {GnnDtype::FLOAT32, 4},
        {GnnDtype::FLOAT64, 8},
        {GnnDtype::INT32,   4},
        {GnnDtype::INT64,   8},
        {GnnDtype::UINT8,   1},
        {GnnDtype::BOOL,    1},
    };

    const uint64_t D = 10;
    for (const auto& info : infos) {
        auto header = FeatureMatrixHeader::make(1, D, info.dtype);
        EXPECT_EQ(header.row_bytes(), D * info.expected_elem_size)
            << "row_bytes mismatch for dtype " << static_cast<int>(info.dtype);
    }
}

TEST_F(FeatureMatrixTest, CreateReorderedDuplicateSourceRows) {
    // Permutation with duplicate source rows (not a true permutation)
    // create_reordered doesn't enforce bijection — it's just an index mapping
    const uint64_t N = 3, D = 2;
    std::vector<float> data = {10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    auto src = FeatureMatrix::create(test_path("dup_perm_src.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    // All output rows come from source row 1
    std::vector<uint64_t> perm = {1, 1, 1};
    auto dst = FeatureMatrix::create_reordered(src, perm, test_path("dup_perm_dst.fmat"));

    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(dst.row_as<float>(i)[0], 20.0f);
        EXPECT_FLOAT_EQ(dst.row_as<float>(i)[1], 21.0f);
    }
}
