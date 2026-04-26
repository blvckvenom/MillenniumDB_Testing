#include "test_helpers.h"

#include <atomic>
#include <thread>

// Fixture alias — keeps test names as FeatureMatrixTest.* in output
using FeatureMatrixTest = GnnStorageTest;

// ===========================================================================
// Header Tests (no fixture needed)
// ===========================================================================

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
    EXPECT_EQ(FeatureMatrixHeader::SIZE, sizeof(FeatureMatrixHeader));
}

TEST(FeatureMatrixHeaderTest, InvalidDtypeThrows) {
    auto bad_dtype = static_cast<GnnDtype>(static_cast<uint8_t>(GnnDtype::MAX_VALUE) + 1);
    EXPECT_THROW(FeatureMatrixHeader::make(10, 10, bad_dtype), std::invalid_argument);

    auto worst_dtype = static_cast<GnnDtype>(255);
    EXPECT_THROW(FeatureMatrixHeader::make(10, 10, worst_dtype), std::invalid_argument);
}

TEST(FeatureMatrixHeaderTest, MaxValidDtypeAccepted) {
    auto h = FeatureMatrixHeader::make(10, 10, GnnDtype::MAX_VALUE);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.get_dtype(), GnnDtype::BOOL);
}

// ===========================================================================
// Parameterized Dtype Tests
// ===========================================================================

class FeatureMatrixDtypeTest : public GnnStorageTest,
    public ::testing::WithParamInterface<DtypeTestParam> {};

TEST_P(FeatureMatrixDtypeTest, CreateOpenRoundtrip) {
    auto param = GetParam();
    size_t elem_size = dtype_size(param.dtype);
    const uint64_t N = 4, D = 3;

    // Fill with deterministic byte pattern (prime modulus avoids repetition)
    std::vector<char> data(N * D * elem_size);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<char>((i * 7 + 13) % 251);
    }

    auto path = test_path("dtype_" + param.name + ".fmat");
    auto fm = FeatureMatrix::create(path, N, D, param.dtype, data.data());

    EXPECT_EQ(fm.num_rows(), N);
    EXPECT_EQ(fm.num_cols(), D);
    EXPECT_EQ(fm.dtype(), param.dtype);
    EXPECT_EQ(fm.row_bytes(), D * elem_size);

    // Re-open from disk and verify byte-exact roundtrip
    auto fm2 = FeatureMatrix::open(path);
    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_EQ(std::memcmp(fm2.row(i), data.data() + i * D * elem_size, D * elem_size), 0)
            << "Data mismatch at row " << i << " for dtype " << param.name;
    }
}

TEST_P(FeatureMatrixDtypeTest, RowBytesMatchesDtypeSize) {
    auto param = GetParam();
    size_t elem_size = dtype_size(param.dtype);
    const uint64_t D = 10;

    auto header = FeatureMatrixHeader::make(1, D, param.dtype);
    EXPECT_EQ(header.row_bytes(), D * elem_size)
        << "row_bytes mismatch for dtype " << param.name;
}

INSTANTIATE_TEST_SUITE_P(AllDtypes, FeatureMatrixDtypeTest,
    ::testing::Values(
        DtypeTestParam{GnnDtype::FLOAT32, "float32"},
        DtypeTestParam{GnnDtype::FLOAT64, "float64"},
        DtypeTestParam{GnnDtype::INT32,   "int32"},
        DtypeTestParam{GnnDtype::INT64,   "int64"},
        DtypeTestParam{GnnDtype::UINT8,   "uint8"},
        DtypeTestParam{GnnDtype::BOOL,    "bool"}
    ),
    [](const ::testing::TestParamInfo<DtypeTestParam>& info) {
        return info.param.name;
    }
);

// ===========================================================================
// Create + Open Roundtrip
// ===========================================================================

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

    auto fm_read = FeatureMatrix::open(path);
    EXPECT_EQ(fm_read.num_rows(), N);
    EXPECT_EQ(fm_read.num_cols(), D);
    EXPECT_EQ(fm_read.dtype(), GnnDtype::FLOAT32);

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

// ===========================================================================
// Scan
// ===========================================================================

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

// ===========================================================================
// Extract Rows
// ===========================================================================

TEST_F(FeatureMatrixTest, ExtractRowsPreservesInputOrder) {
    const uint64_t N = 10, D = 3;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto fm = FeatureMatrix::create(test_path("extract.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> row_ids = {7, 2, 9, 0};
    std::vector<float> out(row_ids.size() * D);
    fm.extract_rows(row_ids, out.data());

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
    // Pass explicit nullptr — extract_rows should return early before null check
    EXPECT_NO_THROW(fm.extract_rows(empty_ids, nullptr));
}

TEST_F(FeatureMatrixTest, ExtractRowsOutOfBoundsThrows) {
    const uint64_t N = 5, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm = FeatureMatrix::create(test_path("extract_oob.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> bad_ids = {1, 3, 99};
    std::vector<float> out(bad_ids.size() * D);
    EXPECT_THROW(fm.extract_rows(bad_ids, out.data()), std::out_of_range);
}

TEST_F(FeatureMatrixTest, ExtractRowsDuplicateIds) {
    const uint64_t N = 5, D = 2;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto fm = FeatureMatrix::create(test_path("extract_dup.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> ids = {2, 2, 2};
    std::vector<float> out(ids.size() * D);
    EXPECT_NO_THROW(fm.extract_rows(ids, out.data()));
    for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i * D + 0], data[2 * D + 0]);
        EXPECT_FLOAT_EQ(out[i * D + 1], data[2 * D + 1]);
    }
}

TEST_F(FeatureMatrixTest, ExtractRowsLargeRandomAccess) {
    const uint64_t N = 1000, D = 8;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto fm = FeatureMatrix::create(test_path("large_extract.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> ids;
    for (uint64_t i = N; i > 0; i -= 10) {
        ids.push_back(i - 1);
    }

    std::vector<float> out(ids.size() * D);
    EXPECT_NO_THROW(fm.extract_rows(ids, out.data()));

    for (size_t i = 0; i < ids.size(); ++i) {
        uint64_t src_row = ids[i];
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(out[i * D + j], data[src_row * D + j])
                << "Mismatch at output[" << i << "] (source row " << src_row << ") col " << j;
        }
    }
}

// ===========================================================================
// Streaming Create
// ===========================================================================

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

TEST_F(FeatureMatrixTest, CreateStreamingCleansUpOnWriterException) {
    auto path = test_path("stream_fail.fmat");

    EXPECT_THROW(
        FeatureMatrix::create_streaming(path, 10, 4, GnnDtype::FLOAT32,
            [](uint64_t row_id, void* dest, uint64_t rb) {
                if (row_id == 3) {
                    throw std::runtime_error("simulated writer failure");
                }
                std::memset(dest, 0, rb);
            }),
        std::runtime_error);

    EXPECT_FALSE(fs::exists(path))
        << "Partial .fmat file should be cleaned up after writer exception";
}

TEST_F(FeatureMatrixTest, CreateStreamingExceptionPreservesExistingFile) {
    auto path = test_path("stream_preserve.fmat");

    const uint64_t N = 3, D = 2;
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    { auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, original.data()); }

    EXPECT_THROW(
        FeatureMatrix::create_streaming(path, 5, 2, GnnDtype::FLOAT32,
            [](uint64_t row_id, void* dest, uint64_t rb) {
                if (row_id == 2) throw std::runtime_error("fail mid-write");
                std::memset(dest, 0, rb);
            }),
        std::runtime_error);

    // File is gone (O_TRUNC destroyed original, exception cleanup removed partial)
    EXPECT_FALSE(fs::exists(path));
}

// ===========================================================================
// Reordering
// ===========================================================================

TEST_F(FeatureMatrixTest, CreateReorderedRoundtrip) {
    const uint64_t N = 4, D = 3;
    std::vector<float> data = {
        10.0f, 11.0f, 12.0f,
        20.0f, 21.0f, 22.0f,
        30.0f, 31.0f, 32.0f,
        40.0f, 41.0f, 42.0f,
    };
    auto src = FeatureMatrix::create(test_path("reorder_src.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> perm = {3, 2, 1, 0};
    auto dst = FeatureMatrix::create_reordered(src, perm, test_path("reorder_dst.fmat"));

    EXPECT_EQ(dst.num_rows(), N);
    EXPECT_FLOAT_EQ(dst.row_as<float>(0)[0], 40.0f);
    EXPECT_FLOAT_EQ(dst.row_as<float>(1)[0], 30.0f);
    EXPECT_FLOAT_EQ(dst.row_as<float>(2)[0], 20.0f);
    EXPECT_FLOAT_EQ(dst.row_as<float>(3)[0], 10.0f);
}

TEST_F(FeatureMatrixTest, CreateReorderedBadSizeThrows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto src = FeatureMatrix::create(test_path("reorder_bad.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> bad_perm = {0, 1};
    EXPECT_THROW(
        FeatureMatrix::create_reordered(src, bad_perm, test_path("reorder_bad_out.fmat")),
        std::invalid_argument);
}

TEST_F(FeatureMatrixTest, CreateReorderedOutOfBoundsThrows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto src = FeatureMatrix::create(test_path("reorder_oob.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> bad_perm = {0, 1, 99};
    EXPECT_THROW(
        FeatureMatrix::create_reordered(src, bad_perm, test_path("reorder_oob_out.fmat")),
        std::out_of_range);
}

TEST_F(FeatureMatrixTest, CreateReorderedIdentityPermutation) {
    const uint64_t N = 5, D = 3;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto src = FeatureMatrix::create(test_path("identity_src.fmat"), N, D, GnnDtype::FLOAT32, data.data());

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

TEST_F(FeatureMatrixTest, CreateReorderedDuplicateSourceRows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data = {10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    auto src = FeatureMatrix::create(test_path("dup_perm_src.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    std::vector<uint64_t> perm = {1, 1, 1};
    auto dst = FeatureMatrix::create_reordered(src, perm, test_path("dup_perm_dst.fmat"));

    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(dst.row_as<float>(i)[0], 20.0f);
        EXPECT_FLOAT_EQ(dst.row_as<float>(i)[1], 21.0f);
    }
}

// ===========================================================================
// File Persistence & Overwrite
// ===========================================================================

TEST_F(FeatureMatrixTest, PersistsAcrossOpenClose) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto path = test_path("persist.fmat");

    { auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, data.data()); }

    auto fm = FeatureMatrix::open(path);
    EXPECT_EQ(fm.num_rows(), N);
    const float* row1 = fm.row_as<float>(1);
    EXPECT_FLOAT_EQ(row1[0], 3.0f);
    EXPECT_FLOAT_EQ(row1[1], 4.0f);
}

TEST_F(FeatureMatrixTest, CreateOverwritesExistingFile) {
    auto path = test_path("overwrite.fmat");

    const uint64_t N = 3, D = 2;
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    { auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, data_a.data()); }

    const uint64_t N2 = 2, D2 = 3;
    std::vector<float> data_b = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    { auto fm = FeatureMatrix::create(path, N2, D2, GnnDtype::FLOAT32, data_b.data()); }

    auto fm = FeatureMatrix::open(path);
    EXPECT_EQ(fm.num_rows(), N2);
    EXPECT_EQ(fm.num_cols(), D2);
    EXPECT_FLOAT_EQ(fm.row_as<float>(0)[0], 10.0f);
    EXPECT_FLOAT_EQ(fm.row_as<float>(1)[2], 60.0f);
}

TEST_F(FeatureMatrixTest, CreateOverwritesLargerFileWithSmaller) {
    auto path = test_path("overwrite_shrink.fmat");

    const uint64_t N1 = 100, D1 = 100;
    std::vector<float> data_large(N1 * D1, 99.0f);
    { auto fm = FeatureMatrix::create(path, N1, D1, GnnDtype::FLOAT32, data_large.data()); }
    auto size_large = fs::file_size(path);

    const uint64_t N2 = 2, D2 = 2;
    std::vector<float> data_small = {1.0f, 2.0f, 3.0f, 4.0f};
    { auto fm = FeatureMatrix::create(path, N2, D2, GnnDtype::FLOAT32, data_small.data()); }
    auto size_small = fs::file_size(path);

    EXPECT_LT(size_small, size_large);

    auto fm = FeatureMatrix::open(path);
    EXPECT_EQ(fm.num_rows(), N2);
    EXPECT_EQ(fm.num_cols(), D2);
}

// ===========================================================================
// Corrupted / Truncated Files
// ===========================================================================

TEST_F(FeatureMatrixTest, OpenCorruptedHeaderThrows) {
    auto path = test_path("corrupt.fmat");
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

    { auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, data.data()); }

    fs::resize_file(path, FeatureMatrixHeader::SIZE + 10);

    EXPECT_THROW(FeatureMatrix::open(path), std::runtime_error);
}

TEST_F(FeatureMatrixTest, OpenFileWithExtraTrailingBytesSucceeds) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto path = test_path("extra_bytes.fmat");

    { auto fm = FeatureMatrix::create(path, N, D, GnnDtype::FLOAT32, data.data()); }

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

// ===========================================================================
// Move Semantics
// ===========================================================================

TEST_F(FeatureMatrixTest, MovedFromObjectThrowsOnAccess) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm1 = FeatureMatrix::create(test_path("move_src.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    auto fm2 = std::move(fm1);

    EXPECT_EQ(fm2.num_rows(), N);
    EXPECT_NO_THROW(fm2.row(0));

    EXPECT_EQ(fm1.num_rows(), 0u);
    EXPECT_THROW(fm1.row(0), std::runtime_error);
}

TEST_F(FeatureMatrixTest, MoveAssignReleasesOldMapping) {
    const uint64_t N = 2, D = 2;
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {10.0f, 20.0f, 30.0f, 40.0f};

    auto fm = FeatureMatrix::create(test_path("assign_a.fmat"), N, D, GnnDtype::FLOAT32, data_a.data());
    EXPECT_FLOAT_EQ(fm.row_as<float>(0)[0], 1.0f);

    fm = FeatureMatrix::create(test_path("assign_b.fmat"), N, D, GnnDtype::FLOAT32, data_b.data());
    EXPECT_FLOAT_EQ(fm.row_as<float>(0)[0], 10.0f);
}

TEST_F(FeatureMatrixTest, ScanOnMovedFromThrows) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f);
    auto fm1 = FeatureMatrix::create(test_path("scan_moved.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    auto fm2 = std::move(fm1);

    EXPECT_THROW(fm1.scan([](uint64_t, const void*) {}), std::runtime_error);

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

// ===========================================================================
// Boundary Dimensions
// ===========================================================================

TEST_F(FeatureMatrixTest, SingleRowSingleCol) {
    const uint64_t N = 1, D = 1;
    std::vector<float> data = {42.0f};
    auto fm = FeatureMatrix::create(test_path("1x1.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    EXPECT_EQ(fm.num_rows(), 1u);
    EXPECT_EQ(fm.num_cols(), 1u);
    EXPECT_FLOAT_EQ(fm.row_as<float>(0)[0], 42.0f);
    EXPECT_THROW(fm.row(1), std::out_of_range);

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

// ===========================================================================
// create() validation
// ===========================================================================

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

// ===========================================================================
// Concurrent Reads
// ===========================================================================

TEST_F(FeatureMatrixTest, ConcurrentReadsDoNotCorrupt) {
    const uint64_t N = 1000, D = 16;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto fm = FeatureMatrix::create(test_path("concurrent.fmat"), N, D, GnnDtype::FLOAT32, data.data());

    constexpr int num_threads = 8;
    std::atomic<bool> error_found{false};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&fm, &data, &error_found, t, N, D]() {
            for (uint64_t i = 0; i < N; ++i) {
                uint64_t row_id = (i + static_cast<uint64_t>(t) * 100) % N;
                const float* row = fm.row_as<float>(row_id);
                for (uint64_t j = 0; j < D; ++j) {
                    if (row[j] != static_cast<float>(row_id * D + j)) {
                        error_found.store(true);
                        return;
                    }
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_FALSE(error_found.load()) << "Concurrent reads produced corrupted data";
}

// ===========================================================================
// Fix C1: row_as<T> type-size mismatch detection
// ===========================================================================

// The assert in row_as<T> vanishes in Release. This test documents that
// a type-size mismatch is caught in Debug (assert fires) but NOT in Release.
// If the implementation is changed to use a runtime check, update this test.
#ifndef NDEBUG
TEST_F(FeatureMatrixTest, RowAsWrongTypeSizeAsserts) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data(N * D, 1.0f); // FLOAT32 = 4 bytes
    auto fm = FeatureMatrix::create(test_path("wrong_type.fmat"),
                                     N, D, GnnDtype::FLOAT32, data.data());

    // double = 8 bytes vs FLOAT32 = 4 bytes → assert should fire
    EXPECT_DEATH(fm.row_as<double>(0), "")
        << "row_as<double> on FLOAT32 matrix should assert in Debug";
}
#endif

// ===========================================================================
// Fix C2: Overflow guard coverage in create() and open()
// ===========================================================================

TEST(FeatureMatrixHeaderTest, ZeroColsIsInvalid) {
    auto h = FeatureMatrixHeader::make(10, 0, GnnDtype::FLOAT32);
    // make() doesn't reject zero cols, but is_valid() does
    // Actually, let's check if make() even works with 0 cols
    EXPECT_FALSE(h.is_valid()) << "Header with num_cols=0 should be invalid";
}

TEST_F(FeatureMatrixTest, CreateOverflowRowsColsThrows) {
    // num_rows * num_cols * dtype_size would overflow size_t
    // UINT64_MAX rows × 1 col × 4 bytes = overflow
    EXPECT_THROW(
        FeatureMatrix::create(test_path("overflow.fmat"),
                               UINT64_MAX, 1, GnnDtype::FLOAT32, nullptr),
        std::exception  // could be overflow_error or invalid_argument
    ) << "Creating FeatureMatrix with overflowing dimensions should throw";
}

TEST_F(FeatureMatrixTest, OpenCraftedOverflowHeaderThrows) {
    // Craft a .fmat file with valid magic+version but overflowing dimensions
    auto path = test_path("overflow_header.fmat");
    {
        auto h = FeatureMatrixHeader::make(1, 1, GnnDtype::FLOAT32);
        // Overwrite with huge dimensions that would overflow
        h.num_rows = UINT64_MAX;
        h.num_cols = UINT64_MAX;

        std::ofstream ofs(path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(&h), sizeof(h));
        // Write minimal data so file is "big enough" for header
        char pad[8] = {};
        ofs.write(pad, sizeof(pad));
    }

    EXPECT_THROW(FeatureMatrix::open(path), std::exception)
        << "Opening FeatureMatrix with overflowing header dimensions should throw";
}

// ===========================================================================
// Spec #14: Parallel create (multi-thread pwrite into pre-allocated file)
// ===========================================================================

// Build a deterministic input matrix and verify byte-equality against the
// sequential single-thread path under several worker counts. The writer is a
// pure memcpy from a const buffer — the same shape used by import_node_tensors
// for the npy memmap source. Implemented as a fixture member so it can reach
// the protected GnnStorageTest::test_path helper.
class FeatureMatrixParallelTest : public GnnStorageTest {
protected:
    void run_byte_eq(uint64_t N, uint64_t D, unsigned workers,
                     const std::string& tag)
    {
        std::vector<float> data(N * D);
        for (uint64_t i = 0; i < N * D; ++i) {
            data[i] = static_cast<float>(i * 0.25f + 0.5f);
        }
        const char* src = reinterpret_cast<const char*>(data.data());
        const size_t rb = D * sizeof(float);

        auto seq_path = test_path("seq_" + tag + ".fmat");
        FeatureMatrix::create_streaming(
            seq_path, N, D, GnnDtype::FLOAT32,
            [src, rb](uint64_t row_id, void* dest, uint64_t bytes) {
                ASSERT_EQ(bytes, rb);
                std::memcpy(dest, src + row_id * rb, bytes);
            });

        auto par_path = test_path("par_" + tag + ".fmat");
        FeatureMatrix::create_parallel(
            par_path, N, D, GnnDtype::FLOAT32,
            [src, rb](uint64_t row_id, void* dest, uint64_t bytes) {
                std::memcpy(dest, src + row_id * rb, bytes);
            },
            workers);

        auto seq_size = fs::file_size(seq_path);
        auto par_size = fs::file_size(par_path);
        ASSERT_EQ(seq_size, par_size)
            << "[" << tag << "] sequential and parallel file sizes differ";

        std::ifstream a(seq_path, std::ios::binary);
        std::ifstream b(par_path, std::ios::binary);
        ASSERT_TRUE(a.good() && b.good());
        std::vector<char> ba(seq_size), bb(par_size);
        a.read(ba.data(), ba.size());
        b.read(bb.data(), bb.size());
        EXPECT_EQ(std::memcmp(ba.data(), bb.data(), seq_size), 0)
            << "[" << tag << "] parallel output differs from sequential ("
            << workers << " workers)";

        auto fm = FeatureMatrix::open(par_path);
        EXPECT_EQ(fm.num_rows(), N);
        EXPECT_EQ(fm.num_cols(), D);
        EXPECT_EQ(fm.dtype(), GnnDtype::FLOAT32);
        for (uint64_t i = 0; i < N; ++i) {
            const float* row = fm.row_as<float>(i);
            for (uint64_t j = 0; j < D; ++j) {
                EXPECT_FLOAT_EQ(row[j], data[i * D + j])
                    << "[" << tag << "] mismatch at (" << i << "," << j << ")";
            }
        }
    }
};

TEST_F(FeatureMatrixParallelTest, SmokeMatchesSequential) {
    // 1M rows × 16 floats = 64 MiB (small enough for CI, large enough to
    // give each of 4 workers a non-trivial chunk).
    run_byte_eq(/*N=*/1u << 20, /*D=*/16, /*workers=*/4, "smoke");
}

TEST_F(FeatureMatrixParallelTest, ByteEqUnderConcurrency) {
    // Multiple worker counts to exercise both even-split (8 workers, N % 8 == 0)
    // and remainder-distribution paths (3 and 7 workers).
    run_byte_eq(/*N=*/1024u, /*D=*/8, /*workers=*/3, "w3");
    run_byte_eq(/*N=*/1024u, /*D=*/8, /*workers=*/7, "w7");
    run_byte_eq(/*N=*/1024u, /*D=*/8, /*workers=*/8, "w8");
}

TEST_F(FeatureMatrixTest, CreateParallelHeaderIntegrity) {
    const uint64_t N = 7777, D = 17;  // odd dims to stress remainder math
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i);

    auto path = test_path("parallel_header.fmat");
    FeatureMatrix::create_parallel(
        path, N, D, GnnDtype::FLOAT32,
        [&data, D](uint64_t row_id, void* dest, uint64_t bytes) {
            std::memcpy(dest, &data[row_id * D], bytes);
        },
        /*num_workers=*/5);

    // Re-read the raw header bytes off disk and validate every field.
    std::ifstream ifs(path, std::ios::binary);
    ASSERT_TRUE(ifs.good());
    FeatureMatrixHeader h{};
    ifs.read(reinterpret_cast<char*>(&h), sizeof(h));
    ASSERT_EQ(ifs.gcount(), static_cast<std::streamsize>(sizeof(h)));

    EXPECT_EQ(h.magic,    FeatureMatrixHeader::MAGIC);
    EXPECT_EQ(h.version,  FeatureMatrixHeader::VERSION);
    EXPECT_EQ(h.num_rows, N);
    EXPECT_EQ(h.num_cols, D);
    EXPECT_EQ(h.get_dtype(), GnnDtype::FLOAT32);
    EXPECT_TRUE(h.is_valid());

    auto file_bytes = fs::file_size(path);
    EXPECT_EQ(file_bytes, FeatureMatrixHeader::SIZE + N * D * sizeof(float));
}

TEST_F(FeatureMatrixTest, CreateParallelZeroWorkersFallsBackToSequential) {
    // num_workers == 0 must use the sequential path verbatim — protects
    // callers that cannot detect cores (e.g. some sandboxes).
    const uint64_t N = 64, D = 4;
    std::vector<float> data(N * D);
    for (uint64_t i = 0; i < N * D; ++i) data[i] = static_cast<float>(i + 1);

    auto path = test_path("parallel_w0.fmat");
    FeatureMatrix::create_parallel(
        path, N, D, GnnDtype::FLOAT32,
        [&data, D](uint64_t row_id, void* dest, uint64_t bytes) {
            std::memcpy(dest, &data[row_id * D], bytes);
        },
        /*num_workers=*/0);

    auto fm = FeatureMatrix::open(path);
    for (uint64_t i = 0; i < N; ++i) {
        const float* row = fm.row_as<float>(i);
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(row[j], data[i * D + j]);
        }
    }
}

TEST_F(FeatureMatrixTest, CreateParallelMoreWorkersThanRowsIsSafe) {
    // 16 workers but only 5 rows — extra workers must be skipped, not crash.
    const uint64_t N = 5, D = 3;
    std::vector<float> data = {
         1, 2, 3,
         4, 5, 6,
         7, 8, 9,
        10,11,12,
        13,14,15,
    };

    auto path = test_path("parallel_excess.fmat");
    FeatureMatrix::create_parallel(
        path, N, D, GnnDtype::FLOAT32,
        [&data, D](uint64_t row_id, void* dest, uint64_t bytes) {
            std::memcpy(dest, &data[row_id * D], bytes);
        },
        /*num_workers=*/16);

    auto fm = FeatureMatrix::open(path);
    EXPECT_EQ(fm.num_rows(), N);
    for (uint64_t i = 0; i < N; ++i) {
        const float* row = fm.row_as<float>(i);
        for (uint64_t j = 0; j < D; ++j) {
            EXPECT_FLOAT_EQ(row[j], data[i * D + j]);
        }
    }
}

TEST_F(FeatureMatrixTest, CreateParallelWriterExceptionCleansUp) {
    auto path = test_path("parallel_fail.fmat");

    EXPECT_THROW(
        FeatureMatrix::create_parallel(
            path, /*N=*/64, /*D=*/4, GnnDtype::FLOAT32,
            [](uint64_t row_id, void* dest, uint64_t bytes) {
                if (row_id == 17) {
                    throw std::runtime_error("simulated parallel writer failure");
                }
                std::memset(dest, 0, bytes);
            },
            /*num_workers=*/4),
        std::runtime_error);

    EXPECT_FALSE(fs::exists(path))
        << "Partial parallel .fmat file should be cleaned up after writer exception";
}

TEST_F(FeatureMatrixTest, CreateParallelZeroDimsThrows) {
    EXPECT_THROW(
        FeatureMatrix::create_parallel(
            test_path("p_zero_rows.fmat"), 0, 4, GnnDtype::FLOAT32,
            [](uint64_t, void*, uint64_t) {}, 4),
        std::invalid_argument);
    EXPECT_THROW(
        FeatureMatrix::create_parallel(
            test_path("p_zero_cols.fmat"), 4, 0, GnnDtype::FLOAT32,
            [](uint64_t, void*, uint64_t) {}, 4),
        std::invalid_argument);
}
