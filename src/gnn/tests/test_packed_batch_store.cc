#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#include <fstream>

#include "gnn/storage/packed_batch_store.h"
#include "gnn/storage/consolidated_slim.h"
#include "graph_models/object_id.h"

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

// Verify ALL values after close+reopen, not just the first and last elements
TEST_F(PackedBatchTest, PersistsAcrossOpenClose) {
    const uint64_t N = 3, D = 2;
    std::vector<float> data = {1,2, 3,4, 5,6};
    auto dir = test_path("persist");

    { PackedBatchWriter w(dir, D, GnnDtype::FLOAT32); w.write_batch(0, data.data(), N); }

    PackedBatchReader reader(dir, 1, D, GnnDtype::FLOAT32);
    std::vector<float> out(N * D);
    reader.read_batch(0, out.data(), out.size() * sizeof(float));
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], data[i]) << "Mismatch at index " << i << " after reopen";
    }
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

// Concurrent reads on the SAME batch from multiple threads (true contention test)
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

// Dtype mismatch between writer (FLOAT32) and reader (FLOAT64) must throw
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

// read_header() must throw std::out_of_range for batch ids beyond the declared count
TEST_F(PackedBatchTest, ReadHeaderOutOfBoundsThrows) {
    auto dir = test_path("header_oob");
    PackedBatchWriter writer(dir, 4, GnnDtype::FLOAT32);
    std::vector<float> data(4, 1.0f);
    writer.write_batch(0, data.data(), 1);

    PackedBatchReader reader(dir, 1, 4, GnnDtype::FLOAT32);
    EXPECT_THROW(reader.read_header(1), std::out_of_range);
    EXPECT_THROW(reader.read_header(999), std::out_of_range);
}

// INT64 dtype roundtrip: values including negatives must survive write+read unchanged
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

// Reserved bytes — non-zero reserved bytes must be accepted for forward compatibility
TEST_F(PackedBatchTest, HeaderNonZeroReservedBytesAccepted) {
    auto h = PackedBatchHeader::make(10, 128, GnnDtype::FLOAT32);
    h.reserved[0] = 0xFF;
    h.reserved[6] = 0x42;
    EXPECT_TRUE(h.is_valid()) << "is_valid() should accept non-zero reserved bytes for forward compatibility";
}

// Empty assignment list (0 batches): directory is created but no batch files are written
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

// Re-reading the same batch twice must return identical data (idempotency)
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

// Large batch_id (>999999) filename correctness: format string widens past 6 digits
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

    // 7-digit rollover: "%06" is a *minimum* width, so id 1000000 must map to
    // batch_1000000.bin (widened, not truncated/wrapped). The writer enforces
    // sequential ids, so place a valid batch file at that name by hand and
    // verify the reader's name construction resolves it round-trip.
    auto header = PackedBatchHeader::make(1, 2, GnnDtype::FLOAT32);
    std::vector<float> row = { 7.0f, 8.0f };
    {
        std::ofstream ofs(dir / "batch_1000000.bin", std::ios::binary);
        ASSERT_TRUE(ofs.good());
        ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
        ofs.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size() * sizeof(float)));
    }

    PackedBatchReader reader(dir, 1000001, 2, GnnDtype::FLOAT32);
    auto h = reader.read_header(1000000);
    EXPECT_EQ(h.num_nodes, 1u);

    std::vector<float> out(2, 0.0f);
    uint64_t nodes = reader.read_batch(1000000, out.data(), out.size() * sizeof(float));
    EXPECT_EQ(nodes, 1u);
    EXPECT_EQ(out[0], 7.0f);
    EXPECT_EQ(out[1], 8.0f);
}

// Reader must reject feature_dim=0 with std::invalid_argument at construction
TEST_F(PackedBatchTest, ReaderFeatureDimZeroThrows) {
    auto dir = test_path("reader_dim0");
    fs::create_directories(dir);
    EXPECT_THROW(
        PackedBatchReader(dir, 1, 0, GnnDtype::FLOAT32),
        std::invalid_argument
    );
}

// ===========================================================================
// PackedBatchHeader v2 (ObjectId table support)
// ===========================================================================

TEST(PackedBatchHeaderV2, IsValidAcceptsBothVersions) {
    auto h1 = PackedBatchHeader::make(10, 4, GnnDtype::FLOAT32);
    EXPECT_TRUE(h1.is_valid());
    EXPECT_EQ(h1.version, 1u);
    EXPECT_FALSE(h1.has_oid_table());
    EXPECT_EQ(h1.data_offset(), 32u);

    auto h2 = PackedBatchHeader::make_v2(10, 4, GnnDtype::FLOAT32);
    EXPECT_TRUE(h2.is_valid());
    EXPECT_EQ(h2.version, 2u);
    EXPECT_TRUE(h2.has_oid_table());
    EXPECT_EQ(h2.data_offset(), 32u + 10u * 8u);
}

TEST(PackedBatchHeaderV2, DataOffsetAccountsForOidTable) {
    auto h2 = PackedBatchHeader::make_v2(100, 128, GnnDtype::FLOAT32);
    EXPECT_EQ(h2.data_offset(), 32u + 100u * 8u);

    auto h1 = PackedBatchHeader::make(100, 128, GnnDtype::FLOAT32);
    EXPECT_EQ(h1.data_offset(), 32u);
}

TEST_F(PackedBatchTest, ReaderSkipsOidTableCorrectly) {
    // Manually write a v2 file and verify reader reads features, not OID bytes
    auto tmp = test_path("test_v2_reader");
    fs::create_directories(tmp);
    auto path = tmp / "batch_000000.bin";

    constexpr uint64_t N = 3, D = 2;
    auto header = PackedBatchHeader::make_v2(N, D, GnnDtype::FLOAT32);

    // OID table (should be SKIPPED by reader)
    std::vector<uint64_t> oids = {100, 200, 300};
    // Feature data (this is what reader should return)
    std::vector<float> features = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Write file: header + OID table + features
    {
        int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        ASSERT_GE(fd, 0);
        ::write(fd, &header, sizeof(header));
        ::write(fd, oids.data(), oids.size() * sizeof(uint64_t));
        ::write(fd, features.data(), features.size() * sizeof(float));
        ::fsync(fd);
        ::close(fd);
    }

    // Read with PackedBatchReader — should get features, NOT oid bytes
    std::vector<float> out(N * D);
    PackedBatchReader reader(tmp.string(), 1, D, GnnDtype::FLOAT32);
    reader.read_batch(0, out.data(), out.size() * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);
    EXPECT_FLOAT_EQ(out[4], 5.0f);
    EXPECT_FLOAT_EQ(out[5], 6.0f);
}

// ===========================================================================
// Partitioned packer: functional equivalence with the classic (sequential) packer
// ===========================================================================
//
// The partitioned packer writes rows in PARTITION ITERATION ORDER, not
// sample-input order. That breaks bit-identity with the classic packer, but
// preserves the multiset of feature rows per batch. The tests below compare
// the multisets row-by-row to verify functional equivalence.

namespace {

// Read the v1 (no OID table) data section of a packed batch file as a vector
// of rows of `row_bytes` bytes each. Throws on header mismatch.
std::vector<std::vector<char>>
read_packed_batch_rows_v1(const fs::path& path, uint64_t row_bytes)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    PackedBatchHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!hdr.is_valid()) {
        throw std::runtime_error("invalid header in " + path.string());
    }
    if (hdr.has_oid_table()) {
        // Skip OID table for v2.
        f.seekg(static_cast<std::streamoff>(hdr.num_nodes * sizeof(uint64_t)),
                std::ios::cur);
    }
    std::vector<std::vector<char>> rows(hdr.num_nodes);
    for (uint64_t i = 0; i < hdr.num_nodes; ++i) {
        rows[i].resize(row_bytes);
        f.read(rows[i].data(), static_cast<std::streamsize>(row_bytes));
    }
    return rows;
}

// Helper: assert that two batch directories contain the same set of files
// and the data section of each pair contains the same MULTISET of rows
// (order-insensitive). Headers are verified for equal num_nodes only.
void expect_packed_dirs_functionally_equivalent(const fs::path& dir_a,
                                                const fs::path& dir_b,
                                                uint64_t expected_batches,
                                                uint64_t row_bytes)
{
    for (uint64_t b = 0; b < expected_batches; ++b) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "batch_%06" PRIu64 ".bin", b);
        auto path_a = dir_a / buf;
        auto path_b = dir_b / buf;

        ASSERT_TRUE(fs::exists(path_a)) << path_a;
        ASSERT_TRUE(fs::exists(path_b)) << path_b;

        auto rows_a = read_packed_batch_rows_v1(path_a, row_bytes);
        auto rows_b = read_packed_batch_rows_v1(path_b, row_bytes);

        ASSERT_EQ(rows_a.size(), rows_b.size())
            << "Batch " << b << " num_nodes mismatch: " << rows_a.size()
            << " vs " << rows_b.size();

        std::sort(rows_a.begin(), rows_a.end());
        std::sort(rows_b.begin(), rows_b.end());
        for (size_t i = 0; i < rows_a.size(); ++i) {
            ASSERT_EQ(rows_a[i], rows_b[i])
                << "Batch " << b << " row " << i
                << " content mismatch (after sort)";
        }
    }
}

} // anonymous namespace

TEST_F(PackedBatchTest, Partitioned_FunctionallyEquivalentToClassic) {
    // Simple dataset that exercises multiple partitions:
    // 64 rows × 4 dim → 64 × 16 B = 1024 B total.
    // Partition size 256 B → 16 rows per partition → 4 partitions.
    const uint64_t N = 64, D = 4;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i);

    auto fmat_path = test_path("b1_iden.fmat");
    auto fm = FeatureMatrix::create(
        fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    // 8 batches with mixed-locality assignments (some adjacent, some spread).
    std::vector<std::vector<uint64_t>> assignments = {
        {0, 1, 2, 3},        // dense in partition 0
        {16, 17, 18, 19},    // dense in partition 1
        {0, 16, 32, 48},     // one row per partition
        {3, 19, 35, 51},     // one row per partition (different positions)
        {63, 0, 31, 32},     // wrap and cross
        {7, 7, 7},           // duplicate row indices in one batch
        {15, 16},            // straddle partition boundary 0/1
        {47, 48}             // straddle partition boundary 2/3
    };

    auto classic_dir     = test_path("b1_iden_classic");
    auto partitioned_dir = test_path("b1_iden_partitioned");

    generate_packed_batches(fm, assignments, classic_dir);

    // Force partition_bytes = 256 → 16 rows per partition → 4 partitions.
    generate_packed_batches_partitioned(
        fm, assignments.size(),
        [&](uint64_t bid) { return assignments.at(bid); },
        partitioned_dir, /*partition_bytes=*/256);

    const uint64_t row_bytes = D * sizeof(float);
    expect_packed_dirs_functionally_equivalent(
        classic_dir, partitioned_dir, assignments.size(), row_bytes);
}

TEST_F(PackedBatchTest, Partitioned_HandlesEmptyBatches) {
    const uint64_t N = 16, D = 2;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i + 1);

    auto fmat_path = test_path("b1_empty.fmat");
    auto fm = FeatureMatrix::create(
        fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    // Mix of empty and populated batches — empty batches should produce
    // header-only files, identical between paths.
    std::vector<std::vector<uint64_t>> assignments = {
        {0, 5, 10},
        {},                   // empty
        {3, 7, 11, 15},
        {},                   // empty
        {0}                   // single row
    };

    auto classic_dir     = test_path("b1_empty_classic");
    auto partitioned_dir = test_path("b1_empty_partitioned");

    generate_packed_batches(fm, assignments, classic_dir);
    generate_packed_batches_partitioned(
        fm, assignments.size(),
        [&](uint64_t bid) { return assignments.at(bid); },
        partitioned_dir, /*partition_bytes=*/64);  // 8 rows per partition

    const uint64_t row_bytes = D * sizeof(float);
    expect_packed_dirs_functionally_equivalent(
        classic_dir, partitioned_dir, assignments.size(), row_bytes);

    // Sanity: empty batch files are exactly header_size on disk.
    EXPECT_EQ(fs::file_size(partitioned_dir / "batch_000001.bin"),
              PackedBatchHeader::SIZE);
}

TEST_F(PackedBatchTest, Partitioned_PartitionSmallerThanSingleRow) {
    // partition_bytes < row_bytes triggers the "round up to 1 row" branch.
    // Each partition holds exactly 1 row → maximum number of partitions.
    const uint64_t N = 8, D = 4;   // row_bytes = 16
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(100 + i);

    auto fmat_path = test_path("b1_tiny.fmat");
    auto fm = FeatureMatrix::create(
        fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    std::vector<std::vector<uint64_t>> assignments = {
        {7, 0, 4, 1},
        {2, 5, 3, 6}
    };

    auto classic_dir     = test_path("b1_tiny_classic");
    auto partitioned_dir = test_path("b1_tiny_partitioned");

    generate_packed_batches(fm, assignments, classic_dir);
    // partition_bytes = 8 < row_bytes = 16 → 1 row per partition, 8 partitions
    generate_packed_batches_partitioned(
        fm, assignments.size(),
        [&](uint64_t bid) { return assignments.at(bid); },
        partitioned_dir, /*partition_bytes=*/8);

    const uint64_t row_bytes = D * sizeof(float);
    expect_packed_dirs_functionally_equivalent(
        classic_dir, partitioned_dir, assignments.size(), row_bytes);
}

// ===========================================================================
// Consolidated cold-feature file (consolidated.slim): round-trip correctness
// ===========================================================================

// The consolidated file concatenates per-batch feature payloads into a single
// file at 4096-aligned offsets, with permutation fingerprint and meta SHA-256
// in the header for stale-rejection. Each per-batch payload must be byte-identical
// to the data section of the corresponding per-batch .bin file. Empty batches
// occupy zero payload bytes.
TEST_F(PackedBatchTest, ConsolidatedRoundTrip) {
    const uint64_t N = 64, D = 4;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i + 1);
    auto fmat_path = test_path("cons_rt.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    std::vector<std::vector<uint64_t>> assignments = {
        {0, 1, 2, 3},        // dense
        {},                  // empty batch -> zero payload
        {16, 0, 48, 32},     // cross-partition
        {7, 7},              // duplicate rows
        {63},                // single
    };
    const uint64_t row_bytes = D * sizeof(float);  // 16

    auto dir       = test_path("cons_rt_pack");
    auto cons_path = test_path("cons_rt_pack/consolidated.slim");

    std::vector<uint64_t> offsets, lengths;
    const uint64_t PERM_FP  = 0xABCDEF0123456789ull;
    const uint64_t META_SHA = 0x1122334455667788ull;

    generate_packed_batches_partitioned(
        fm, assignments.size(),
        [&](uint64_t bid) { return assignments.at(bid); },
        dir, /*partition_bytes=*/row_bytes * 8,  // 8 rows/partition -> 8 partitions
        /*oid_provider=*/[&](uint64_t bid) {
            std::vector<ObjectId> v;
            for (uint64_t r : assignments.at(bid)) v.push_back(ObjectId(r));
            return v;
        },
        cons_path, PERM_FP, META_SHA, &offsets, &lengths);

    // --- Header ---
    ASSERT_TRUE(fs::exists(cons_path));
    ConsolidatedSlimHeader hdr{};
    {
        std::ifstream f(cons_path, std::ios::binary);
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        ASSERT_TRUE(f.good());
    }
    EXPECT_TRUE(hdr.is_valid());
    EXPECT_EQ(hdr.num_batches, assignments.size());
    EXPECT_EQ(hdr.feature_dim, D);
    EXPECT_EQ(hdr.perm_fingerprint, PERM_FP);
    EXPECT_EQ(hdr.meta_sha256_head, META_SHA);
    EXPECT_EQ(hdr.data_start, 4096u);

    ASSERT_EQ(offsets.size(), assignments.size());
    ASSERT_EQ(lengths.size(), assignments.size());

    // --- Per-batch: length, alignment, ordering, byte-identity vs .bin ---
    uint64_t prev_end = hdr.data_start;
    for (uint64_t b = 0; b < assignments.size(); ++b) {
        const uint64_t N_b = assignments[b].size();
        EXPECT_EQ(lengths[b], N_b * row_bytes) << "batch " << b;
        EXPECT_GE(offsets[b], hdr.data_start) << "batch " << b;
        EXPECT_EQ(offsets[b] % hdr.alignment(), 0u) << "offset not aligned, batch " << b;
        EXPECT_GE(offsets[b], prev_end) << "offsets overlap/regress, batch " << b;
        prev_end = offsets[b] + ConsolidatedSlimHeader::align_up(lengths[b], hdr.alignment());

        if (N_b == 0) { EXPECT_EQ(lengths[b], 0u); continue; }

        // Consolidated payload at offsets[b].
        std::vector<char> cons_payload(lengths[b]);
        {
            std::ifstream f(cons_path, std::ios::binary);
            f.seekg(static_cast<std::streamoff>(offsets[b]));
            f.read(cons_payload.data(), static_cast<std::streamsize>(lengths[b]));
            ASSERT_TRUE(f.good()) << "cons read batch " << b;
        }

        // Per-batch .bin data section (use the header's own data_offset()).
        char fn[64];
        std::snprintf(fn, sizeof(fn), "batch_%06" PRIu64 ".bin", b);
        auto bin_path = dir / fn;
        std::vector<char> bin_payload(lengths[b]);
        {
            std::ifstream f(bin_path, std::ios::binary);
            PackedBatchHeader bh{};
            f.read(reinterpret_cast<char*>(&bh), sizeof(bh));
            ASSERT_TRUE(bh.is_valid()) << "bin header batch " << b;
            f.seekg(static_cast<std::streamoff>(bh.data_offset()));
            f.read(bin_payload.data(), static_cast<std::streamsize>(lengths[b]));
            ASSERT_TRUE(f.good()) << "bin read batch " << b;
        }

        EXPECT_EQ(std::memcmp(cons_payload.data(), bin_payload.data(), lengths[b]), 0)
            << "consolidated payload != per-batch .bin data section for batch " << b;
    }
}

// Disabled (empty consolidated_path) => no consolidated file, behaviour
// byte-identical to the plain partitioned packer.
TEST_F(PackedBatchTest, ConsolidatedDisabledByDefault) {
    const uint64_t N = 16, D = 2;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i + 1);
    auto fm = FeatureMatrix::create(test_path("cons_off.fmat"), N, D,
                                    GnnDtype::FLOAT32, features.data());
    std::vector<std::vector<uint64_t>> assignments = {{0, 1}, {2, 3}, {0}};
    auto dir = test_path("cons_off_pack");
    generate_packed_batches_partitioned(
        fm, assignments.size(),
        [&](uint64_t bid) { return assignments.at(bid); },
        dir, /*partition_bytes=*/64);  // no oid_provider, no consolidated_path
    EXPECT_FALSE(fs::exists(dir / "consolidated.slim"));
}
