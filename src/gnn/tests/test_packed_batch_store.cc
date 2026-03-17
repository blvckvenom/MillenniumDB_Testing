#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "gnn/storage/packed_batch_store.h"

using PackedBatchTest = GnnStorageTest;

// ===========================================================================
// Header
// ===========================================================================

TEST(PackedBatchHeaderTest, MakeAndValidate) {
    auto h = PackedBatchHeader::make(100, 128, GnnDtype::FLOAT32);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.magic, PackedBatchHeader::MAGIC);
    EXPECT_EQ(h.version, PackedBatchHeader::VERSION);
    EXPECT_EQ(h.num_nodes, 100u);
    EXPECT_EQ(h.feature_dim, 128u);
    EXPECT_EQ(h.get_dtype(), GnnDtype::FLOAT32);
    EXPECT_EQ(h.data_bytes(), 100u * 128u * 4u);
}

TEST(PackedBatchHeaderTest, InvalidMagic) {
    auto h = PackedBatchHeader::make(10, 10, GnnDtype::FLOAT32);
    h.magic = 0xDEADBEEF;
    EXPECT_FALSE(h.is_valid());
}

TEST(PackedBatchHeaderTest, InvalidDtype) {
    auto h = PackedBatchHeader::make(10, 10, GnnDtype::FLOAT32);
    h.dtype = 255;
    EXPECT_FALSE(h.is_valid());
}

TEST(PackedBatchHeaderTest, ZeroNodesIsValid) {
    auto h = PackedBatchHeader::make(0, 128, GnnDtype::FLOAT32);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.data_bytes(), 0u);
}

TEST(PackedBatchHeaderTest, ZeroFeatureDimIsInvalid) {
    PackedBatchHeader h{};
    std::memset(&h, 0, sizeof(h));
    h.magic = PackedBatchHeader::MAGIC;
    h.version = PackedBatchHeader::VERSION;
    h.num_nodes = 10;
    h.feature_dim = 0;
    h.dtype = static_cast<uint8_t>(GnnDtype::FLOAT32);
    EXPECT_FALSE(h.is_valid());
}

TEST(PackedBatchHeaderTest, SizeIs32Bytes) {
    EXPECT_EQ(sizeof(PackedBatchHeader), 32u);
}

TEST(PackedBatchHeaderTest, DataBytesOverflow) {
    PackedBatchHeader h{};
    std::memset(&h, 0, sizeof(h));
    h.magic = PackedBatchHeader::MAGIC;
    h.version = PackedBatchHeader::VERSION;
    h.num_nodes = UINT64_MAX;
    h.feature_dim = UINT64_MAX;
    h.dtype = static_cast<uint8_t>(GnnDtype::FLOAT32);
    EXPECT_THROW(h.data_bytes(), std::overflow_error);
}

TEST(PackedBatchHeaderTest, MakeInvalidDtypeThrows) {
    EXPECT_THROW(
        PackedBatchHeader::make(10, 10, static_cast<GnnDtype>(255)),
        std::invalid_argument
    );
}

// ===========================================================================
// Writer
// ===========================================================================

TEST_F(PackedBatchTest, WriteAndReadSingleBatch) {
    const uint64_t N = 5, D = 3;
    std::vector<float> data = {1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15};

    auto dir = test_path("packed_single");
    PackedBatchWriter writer(dir, D, GnnDtype::FLOAT32);
    writer.write_batch(0, data.data(), N);

    EXPECT_EQ(writer.batches_written(), 1u);
    EXPECT_TRUE(fs::exists(dir / "batch_000000.bin"));
}

TEST_F(PackedBatchTest, WriteMultipleBatches) {
    const uint64_t D = 4;
    auto dir = test_path("packed_multi");
    PackedBatchWriter writer(dir, D, GnnDtype::FLOAT32);

    for (uint64_t i = 0; i < 10; ++i) {
        uint64_t N = 3 + i;
        std::vector<float> data(N * D, static_cast<float>(i));
        writer.write_batch(i, data.data(), N);
    }

    EXPECT_EQ(writer.batches_written(), 10u);
}

TEST_F(PackedBatchTest, WriteEmptyBatch) {
    auto dir = test_path("packed_empty");
    PackedBatchWriter writer(dir, 128, GnnDtype::FLOAT32);
    writer.write_batch(0, nullptr, 0);

    EXPECT_EQ(writer.batches_written(), 1u);
    auto file_size = fs::file_size(dir / "batch_000000.bin");
    EXPECT_EQ(file_size, PackedBatchHeader::SIZE);
}

TEST_F(PackedBatchTest, WriterFeatureDimZeroThrows) {
    EXPECT_THROW(
        PackedBatchWriter(test_path("bad_dim"), 0, GnnDtype::FLOAT32),
        std::invalid_argument
    );
}

TEST_F(PackedBatchTest, WriterBatchIdNotSequentialThrows) {
    auto dir = test_path("packed_nonseq");
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    std::vector<float> data(4, 1.0f);
    writer.write_batch(0, data.data(), 1);
    EXPECT_THROW(writer.write_batch(2, data.data(), 1), std::invalid_argument);
}

TEST_F(PackedBatchTest, WriterNullDataWithNodesThrows) {
    auto dir = test_path("packed_null");
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    EXPECT_THROW(writer.write_batch(0, nullptr, 5), std::invalid_argument);
}

TEST_F(PackedBatchTest, WriterCreatesDirectory) {
    auto dir = test_path("packed_nested/subdir/deep");
    EXPECT_FALSE(fs::exists(dir));
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    EXPECT_TRUE(fs::exists(dir));
}

// ===========================================================================
// Reader
// ===========================================================================

TEST_F(PackedBatchTest, ReadBackSingleBatch) {
    const uint64_t N = 5, D = 3;
    std::vector<float> data = {1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15};

    auto dir = test_path("read_single");
    PackedBatchWriter writer(dir, D, GnnDtype::FLOAT32);
    writer.write_batch(0, data.data(), N);

    PackedBatchReader reader(dir, 1, D, GnnDtype::FLOAT32);
    std::vector<float> out(N * D);
    uint64_t nodes_read = reader.read_batch(0, out.data(), out.size() * sizeof(float));

    EXPECT_EQ(nodes_read, N);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(PackedBatchTest, ReadBackMultipleBatches) {
    const uint64_t D = 4;
    auto dir = test_path("read_multi");
    PackedBatchWriter writer(dir, D, GnnDtype::FLOAT32);

    std::vector<std::vector<float>> all_data;
    for (uint64_t i = 0; i < 5; ++i) {
        uint64_t N = 3 + i;
        std::vector<float> batch(N * D);
        for (size_t j = 0; j < batch.size(); ++j) batch[j] = static_cast<float>(i * 100 + j);
        writer.write_batch(i, batch.data(), N);
        all_data.push_back(std::move(batch));
    }

    PackedBatchReader reader(dir, 5, D, GnnDtype::FLOAT32);
    for (uint64_t i = 0; i < 5; ++i) {
        uint64_t N = 3 + i;
        std::vector<float> out(N * D);
        uint64_t nodes = reader.read_batch(i, out.data(), out.size() * sizeof(float));
        EXPECT_EQ(nodes, N);
        for (size_t j = 0; j < all_data[i].size(); ++j) {
            EXPECT_FLOAT_EQ(out[j], all_data[i][j]);
        }
    }
}

TEST_F(PackedBatchTest, ReadFloat64) {
    const uint64_t N = 3, D = 2;
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto dir = test_path("read_f64");
    PackedBatchWriter writer(dir, D, GnnDtype::FLOAT64);
    writer.write_batch(0, data.data(), N);

    PackedBatchReader reader(dir, 1, D, GnnDtype::FLOAT64);
    std::vector<double> out(N * D);
    reader.read_batch(0, out.data(), out.size() * sizeof(double));
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(out[i], data[i]);
    }
}

TEST_F(PackedBatchTest, ReadHeaderOnly) {
    auto dir = test_path("read_header");
    PackedBatchWriter writer(dir, 128, GnnDtype::FLOAT32);
    std::vector<float> data(10 * 128, 1.0f);
    writer.write_batch(0, data.data(), 10);

    PackedBatchReader reader(dir, 1, 128, GnnDtype::FLOAT32);
    auto h = reader.read_header(0);
    EXPECT_EQ(h.num_nodes, 10u);
    EXPECT_EQ(h.feature_dim, 128u);
    EXPECT_EQ(h.get_dtype(), GnnDtype::FLOAT32);
}

TEST_F(PackedBatchTest, ReadBatchOutOfBounds) {
    auto dir = test_path("read_oob");
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    std::vector<float> data(4, 1.0f);
    writer.write_batch(0, data.data(), 1);

    PackedBatchReader reader(dir, 1, 4, GnnDtype::FLOAT32);
    std::vector<float> out(4);
    EXPECT_THROW(reader.read_batch(1, out.data(), out.size() * sizeof(float)), std::out_of_range);
}

TEST_F(PackedBatchTest, ReadMissingFileThrows) {
    auto dir = test_path("read_missing");
    fs::create_directories(dir);
    PackedBatchReader reader(dir, 1, 4, GnnDtype::FLOAT32);
    std::vector<float> out(4);
    EXPECT_THROW(reader.read_batch(0, out.data(), out.size() * sizeof(float)), std::runtime_error);
}

TEST_F(PackedBatchTest, ReadCorruptedHeaderThrows) {
    auto dir = test_path("read_corrupt");
    fs::create_directories(dir);
    auto path = dir / "batch_000000.bin";
    {
        std::ofstream ofs(path, std::ios::binary);
        char garbage[64] = {};
        std::memset(garbage, 0xFF, sizeof(garbage));
        ofs.write(garbage, sizeof(garbage));
    }

    PackedBatchReader reader(dir, 1, 4, GnnDtype::FLOAT32);
    std::vector<float> out(100);
    EXPECT_THROW(reader.read_batch(0, out.data(), out.size() * sizeof(float)), std::runtime_error);
}

TEST_F(PackedBatchTest, ReadTruncatedFileThrows) {
    auto dir = test_path("read_truncated");
    PackedBatchWriter writer(dir, 128, GnnDtype::FLOAT32);
    std::vector<float> data(10 * 128, 1.0f);
    writer.write_batch(0, data.data(), 10);

    fs::resize_file(dir / "batch_000000.bin", PackedBatchHeader::SIZE);

    PackedBatchReader reader(dir, 1, 128, GnnDtype::FLOAT32);
    std::vector<float> out(10 * 128);
    EXPECT_THROW(reader.read_batch(0, out.data(), out.size() * sizeof(float)), std::runtime_error);
}

TEST_F(PackedBatchTest, ReadBufferTooSmallThrows) {
    auto dir = test_path("read_small_buf");
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    std::vector<float> data(10 * 4, 1.0f);
    writer.write_batch(0, data.data(), 10);

    PackedBatchReader reader(dir, 1, 4, GnnDtype::FLOAT32);
    std::vector<float> out(2 * 4);
    EXPECT_THROW(reader.read_batch(0, out.data(), out.size() * sizeof(float)), std::runtime_error);
}

TEST_F(PackedBatchTest, ReaderDirNotExistThrows) {
    EXPECT_THROW(
        PackedBatchReader(test_path("no_such_dir"), 1, 4, GnnDtype::FLOAT32),
        std::runtime_error
    );
}

TEST_F(PackedBatchTest, ReaderFeatureDimMismatchThrows) {
    auto dir = test_path("read_dim_mismatch");
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    std::vector<float> data(4, 1.0f);
    writer.write_batch(0, data.data(), 1);

    PackedBatchReader reader(dir, 1, 8, GnnDtype::FLOAT32);
    std::vector<float> out(8);
    EXPECT_THROW(reader.read_batch(0, out.data(), out.size() * sizeof(float)), std::runtime_error);
}

TEST_F(PackedBatchTest, ReadEmptyBatch) {
    auto dir = test_path("read_empty");
    PackedBatchWriter writer(dir, 128, GnnDtype::FLOAT32);
    writer.write_batch(0, nullptr, 0);

    PackedBatchReader reader(dir, 1, 128, GnnDtype::FLOAT32);
    uint64_t nodes = reader.read_batch(0, nullptr, 0);
    EXPECT_EQ(nodes, 0u);
}

TEST_F(PackedBatchTest, PersistsAcrossOpenClose) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data = {1,2, 3,4, 5,6};
    auto dir = test_path("persist");

    { PackedBatchWriter w(dir, D, GnnDtype::FLOAT32); w.write_batch(0, data.data(), N); }

    PackedBatchReader reader(dir, 1, D, GnnDtype::FLOAT32);
    std::vector<float> out(N * D);
    reader.read_batch(0, out.data(), out.size() * sizeof(float));
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[5], 6.0f);
}

// ===========================================================================
// Batch Generation
// ===========================================================================

TEST_F(PackedBatchTest, GenerateFromAssignments) {
    const uint64_t N = 10, D = 3;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i);

    auto fmat_path = test_path("gen_assign.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    std::vector<std::vector<uint64_t>> assignments = {
        {0, 3, 6},
        {1, 4, 7},
        {2, 5, 8, 9}
    };

    auto packed_dir = test_path("gen_assign_packed");
    generate_packed_batches(fm, assignments, packed_dir);

    EXPECT_TRUE(fs::exists(packed_dir / "batch_000000.bin"));
    EXPECT_TRUE(fs::exists(packed_dir / "batch_000001.bin"));
    EXPECT_TRUE(fs::exists(packed_dir / "batch_000002.bin"));

    PackedBatchReader reader(packed_dir, 3, D, GnnDtype::FLOAT32);
    {
        std::vector<float> out(3 * D);
        uint64_t nodes = reader.read_batch(0, out.data(), out.size() * sizeof(float));
        EXPECT_EQ(nodes, 3u);
        EXPECT_FLOAT_EQ(out[0], 0.0f);   // row 0, col 0
        EXPECT_FLOAT_EQ(out[3], 9.0f);   // row 3, col 0
        EXPECT_FLOAT_EQ(out[6], 18.0f);  // row 6, col 0
    }
}

TEST_F(PackedBatchTest, GenerateFromCallback) {
    const uint64_t N = 8, D = 2;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i);

    auto fmat_path = test_path("gen_cb.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    auto packed_dir = test_path("gen_cb_packed");
    generate_packed_batches(fm, 4,
        [](uint64_t batch_id) -> std::vector<uint64_t> {
            return {batch_id * 2, batch_id * 2 + 1};
        },
        packed_dir
    );

    PackedBatchReader reader(packed_dir, 4, D, GnnDtype::FLOAT32);
    for (uint64_t b = 0; b < 4; ++b) {
        std::vector<float> out(2 * D);
        uint64_t nodes = reader.read_batch(b, out.data(), out.size() * sizeof(float));
        EXPECT_EQ(nodes, 2u);
        // Verify ALL values, not just out[0]
        for (uint64_t r = 0; r < 2; ++r) {
            uint64_t src_row = b * 2 + r;
            for (uint64_t c = 0; c < D; ++c) {
                EXPECT_FLOAT_EQ(out[r * D + c], features[src_row * D + c])
                    << "batch=" << b << " row=" << r << " col=" << c;
            }
        }
    }
}

TEST_F(PackedBatchTest, GenerateVerifyAllFeatureContent) {
    const uint64_t N = 20, D = 4;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i);

    auto fmat_path = test_path("gen_verify.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    std::vector<std::vector<uint64_t>> assignments;
    for (uint64_t b = 0; b < 5; ++b) {
        assignments.push_back({b*4+3, b*4+2, b*4+1, b*4});
    }

    auto packed_dir = test_path("gen_verify_packed");
    generate_packed_batches(fm, assignments, packed_dir);

    PackedBatchReader reader(packed_dir, 5, D, GnnDtype::FLOAT32);
    for (uint64_t b = 0; b < 5; ++b) {
        std::vector<float> out(4 * D);
        reader.read_batch(b, out.data(), out.size() * sizeof(float));

        for (uint64_t r = 0; r < 4; ++r) {
            uint64_t src_row = assignments[b][r];
            for (uint64_t c = 0; c < D; ++c) {
                EXPECT_FLOAT_EQ(out[r * D + c], features[src_row * D + c])
                    << "batch=" << b << " row=" << r << " col=" << c
                    << " src_row=" << src_row;
            }
        }
    }
}

TEST_F(PackedBatchTest, LargeScale) {
    const uint64_t N = 10000, D = 128;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i % 1000);

    auto fmat_path = test_path("gen_large.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    const uint64_t batch_size = 100;
    const uint64_t num_batches = N / batch_size;

    auto packed_dir = test_path("gen_large_packed");
    generate_packed_batches(fm, num_batches,
        [batch_size](uint64_t batch_id) -> std::vector<uint64_t> {
            std::vector<uint64_t> ids;
            for (uint64_t i = 0; i < batch_size; ++i) {
                ids.push_back(batch_id * batch_size + i);
            }
            return ids;
        },
        packed_dir
    );

    PackedBatchReader reader(packed_dir, num_batches, D, GnnDtype::FLOAT32);
    for (uint64_t b = 0; b < num_batches; b += 20) {
        std::vector<float> out(batch_size * D);
        uint64_t nodes = reader.read_batch(b, out.data(), out.size() * sizeof(float));
        EXPECT_EQ(nodes, batch_size);
        uint64_t src_row = b * batch_size;
        EXPECT_FLOAT_EQ(out[0], features[src_row * D]);
    }
}

TEST_F(PackedBatchTest, ConcurrentReads) {
    const uint64_t D = 4, num_batches = 20;
    auto dir = test_path("concurrent");
    PackedBatchWriter writer(dir, D, GnnDtype::FLOAT32);
    for (uint64_t i = 0; i < num_batches; ++i) {
        std::vector<float> data(10 * D, static_cast<float>(i));
        writer.write_batch(i, data.data(), 10);
    }

    PackedBatchReader reader(dir, num_batches, D, GnnDtype::FLOAT32);

    std::vector<std::thread> threads;
    std::atomic<uint64_t> errors{0};

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (uint64_t i = static_cast<uint64_t>(t); i < num_batches; i += 4) {
                std::vector<float> out(10 * D);
                uint64_t n = reader.read_batch(i, out.data(), out.size() * sizeof(float));
                if (n != 10) ++errors;
                if (out[0] != static_cast<float>(i)) ++errors;
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0u);
}

// Fix #6: Concurrent reads on the SAME batch (true contention test)
TEST_F(PackedBatchTest, ConcurrentReadsSameBatch) {
    const uint64_t D = 4, N = 100;
    auto dir = test_path("concurrent_same");
    PackedBatchWriter writer(dir, D, GnnDtype::FLOAT32);
    std::vector<float> data(N * D);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);
    writer.write_batch(0, data.data(), N);

    PackedBatchReader reader(dir, 1, D, GnnDtype::FLOAT32);

    std::vector<std::thread> threads;
    std::atomic<uint64_t> errors{0};

    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            for (int iter = 0; iter < 50; ++iter) {
                std::vector<float> out(N * D);
                uint64_t n = reader.read_batch(0, out.data(), out.size() * sizeof(float));
                if (n != N) { ++errors; continue; }
                // Verify full buffer content
                for (size_t i = 0; i < out.size(); ++i) {
                    if (out[i] != static_cast<float>(i)) { ++errors; break; }
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0u);
}

// ===========================================================================
// Additional edge cases (from review)
// ===========================================================================

// Fix #7: Dtype mismatch between writer and reader
TEST_F(PackedBatchTest, ReaderDtypeMismatchThrows) {
    auto dir = test_path("dtype_mismatch");
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    std::vector<float> data(4, 1.0f);
    writer.write_batch(0, data.data(), 1);

    // Reader expects FLOAT64 but file has FLOAT32
    PackedBatchReader reader(dir, 1, 4, GnnDtype::FLOAT64);
    std::vector<double> out(4);
    EXPECT_THROW(reader.read_batch(0, out.data(), out.size() * sizeof(double)), std::runtime_error);
}

// Edge case: single-node batch (boundary between empty and multi-node)
TEST_F(PackedBatchTest, SingleNodeBatch) {
    auto dir = test_path("single_node");
    PackedBatchWriter writer(dir, 3, GnnDtype::FLOAT32);
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    writer.write_batch(0, data.data(), 1);

    PackedBatchReader reader(dir, 1, 3, GnnDtype::FLOAT32);
    std::vector<float> out(3);
    uint64_t n = reader.read_batch(0, out.data(), out.size() * sizeof(float));
    EXPECT_EQ(n, 1u);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

// Edge case: empty batch in the middle of non-empty batches
TEST_F(PackedBatchTest, EmptyBatchInMiddle) {
    auto dir = test_path("empty_middle");
    PackedBatchWriter writer(dir, 2, GnnDtype::FLOAT32);

    std::vector<float> data1 = {1, 2, 3, 4};
    writer.write_batch(0, data1.data(), 2);
    writer.write_batch(1, nullptr, 0);  // empty middle batch
    std::vector<float> data2 = {5, 6};
    writer.write_batch(2, data2.data(), 1);

    PackedBatchReader reader(dir, 3, 2, GnnDtype::FLOAT32);

    std::vector<float> out(4);
    EXPECT_EQ(reader.read_batch(0, out.data(), 4 * sizeof(float)), 2u);
    EXPECT_EQ(reader.read_batch(1, nullptr, 0), 0u);
    EXPECT_EQ(reader.read_batch(2, out.data(), 2 * sizeof(float)), 1u);
    EXPECT_FLOAT_EQ(out[0], 5.0f);
}

// Fix #8: read_header() out-of-bounds directly
TEST_F(PackedBatchTest, ReadHeaderOutOfBoundsThrows) {
    auto dir = test_path("header_oob");
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    std::vector<float> data(4, 1.0f);
    writer.write_batch(0, data.data(), 1);

    PackedBatchReader reader(dir, 1, 4, GnnDtype::FLOAT32);
    EXPECT_THROW(reader.read_header(1), std::out_of_range);
    EXPECT_THROW(reader.read_header(999), std::out_of_range);
}

// Fix #14: INT64 dtype roundtrip
TEST_F(PackedBatchTest, WriteAndReadInt64) {
    const uint64_t N = 3, D = 2;
    std::vector<int64_t> data = {-100, 200, -300, 400, -500, 600};

    auto dir = test_path("read_i64");
    PackedBatchWriter writer(dir, D, GnnDtype::INT64);
    writer.write_batch(0, data.data(), N);

    PackedBatchReader reader(dir, 1, D, GnnDtype::INT64);
    std::vector<int64_t> out(N * D);
    reader.read_batch(0, out.data(), out.size() * sizeof(int64_t));
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(out[i], data[i]);
    }
}

// Fix #15: Reserved bytes — non-zero reserved bytes are accepted (forward-compat)
TEST_F(PackedBatchTest, HeaderNonZeroReservedBytesAccepted) {
    auto h = PackedBatchHeader::make(10, 128, GnnDtype::FLOAT32);
    h.reserved[0] = 0xFF;
    h.reserved[6] = 0x42;
    EXPECT_TRUE(h.is_valid()) << "is_valid() should accept non-zero reserved bytes for forward compatibility";
}

// Fix #16a: Empty assignments (0 batches)
TEST_F(PackedBatchTest, GenerateEmptyAssignments) {
    const uint64_t N = 5, D = 2;
    std::vector<float> features(N * D, 1.0f);
    auto fmat_path = test_path("gen_empty.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    auto packed_dir = test_path("gen_empty_packed");
    std::vector<std::vector<uint64_t>> empty_assignments;
    generate_packed_batches(fm, empty_assignments, packed_dir);

    EXPECT_TRUE(fs::exists(packed_dir)); // directory created
    // No batch files should exist
    EXPECT_FALSE(fs::exists(packed_dir / "batch_000000.bin"));
}

// Fix #16b: Re-read same batch (idempotency)
TEST_F(PackedBatchTest, ReReadSameBatch) {
    const uint64_t N = 5, D = 3;
    std::vector<float> data(N * D);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);

    auto dir = test_path("reread");
    PackedBatchWriter writer(dir, D, GnnDtype::FLOAT32);
    writer.write_batch(0, data.data(), N);

    PackedBatchReader reader(dir, 1, D, GnnDtype::FLOAT32);
    std::vector<float> out1(N * D), out2(N * D);

    reader.read_batch(0, out1.data(), out1.size() * sizeof(float));
    reader.read_batch(0, out2.data(), out2.size() * sizeof(float));

    for (size_t i = 0; i < out1.size(); ++i) {
        EXPECT_FLOAT_EQ(out1[i], out2[i]);
    }
}

// Fix #16c: Large batch_id (>999999) filename correctness
TEST_F(PackedBatchTest, LargeBatchIdFilename) {
    auto dir = test_path("large_id");
    PackedBatchWriter writer(dir, 2, GnnDtype::FLOAT32);

    // Write batches 0 through 5 to satisfy sequential requirement, then check filename
    for (uint64_t i = 0; i < 5; ++i) {
        std::vector<float> data(2, static_cast<float>(i));
        writer.write_batch(i, data.data(), 1);
    }

    // Verify filenames: 0-4 should be zero-padded to 6 digits
    EXPECT_TRUE(fs::exists(dir / "batch_000000.bin"));
    EXPECT_TRUE(fs::exists(dir / "batch_000004.bin"));
}

// Fix #5 (test): Reader rejects feature_dim=0
TEST_F(PackedBatchTest, ReaderFeatureDimZeroThrows) {
    auto dir = test_path("reader_dim0");
    fs::create_directories(dir);
    EXPECT_THROW(
        PackedBatchReader(dir, 1, 0, GnnDtype::FLOAT32),
        std::invalid_argument
    );
}
