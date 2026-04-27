#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include <unistd.h> // getpid

#include "gnn/storage/direct_io_reader.h"

namespace fs = std::filesystem;
using namespace mdb::gnn;

// =============================================================================
// Test Fixture
// =============================================================================

class DirectIoReaderTest : public ::testing::Test {
protected:
    fs::path test_dir_;

    void SetUp() override {
        test_dir_ = fs::temp_directory_path()
            / ("mdb_test_dio_" + std::to_string(getpid()));
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    /// Create a test file: HEADER_SIZE-byte header + ROWS x ROW_BYTES data.
    /// Each row r has floats: (r+1)*100+1, (r+1)*100+2, ..., (r+1)*100+FLOATS_PER_ROW.
    fs::path create_test_file(
        const std::string& name,
        uint64_t header_size,
        uint64_t rows,
        uint64_t row_bytes)
    {
        auto path = test_dir_ / name;
        std::vector<char> data(header_size + rows * row_bytes, 0);

        // Write a recognizable header magic
        if (header_size >= 4) {
            uint32_t magic = 0xDEADBEEF;
            std::memcpy(data.data(), &magic, sizeof(magic));
        }

        // Fill row data with known float patterns
        uint64_t floats_per_row = row_bytes / sizeof(float);
        for (uint64_t r = 0; r < rows; ++r) {
            auto* row = reinterpret_cast<float*>(
                data.data() + header_size + r * row_bytes);
            for (uint64_t c = 0; c < floats_per_row; ++c) {
                row[c] = static_cast<float>((r + 1) * 100 + (c + 1));
            }
        }

        std::ofstream f(path, std::ios::binary);
        f.write(data.data(), static_cast<std::streamsize>(data.size()));
        f.close();
        return path;
    }

    /// Create a test file with arbitrary byte content.
    fs::path create_raw_file(const std::string& name, const std::vector<char>& content) {
        auto path = test_dir_ / name;
        std::ofstream f(path, std::ios::binary);
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
        f.close();
        return path;
    }
};

// =============================================================================
// Constructor tests
// =============================================================================

TEST_F(DirectIoReaderTest, ConstructorOpensExistingFile) {
    auto path = create_test_file("basic.bin", 64, 10, 16);
    DirectIoReader reader(path);

    // File should be open; file_size should match
    EXPECT_EQ(reader.file_size(), 64u + 10u * 16u);
}

TEST_F(DirectIoReaderTest, ConstructorThrowsOnMissingFile) {
    EXPECT_THROW(
        DirectIoReader(test_dir_ / "nonexistent.bin"),
        std::runtime_error);
}

TEST_F(DirectIoReaderTest, FallbackWorksOnTmpfs) {
    // tmpfs (where /tmp usually lives) doesn't support O_DIRECT.
    // The reader should open the file and fall back gracefully.
    auto path = create_test_file("tmpfs.bin", 64, 5, 16);
    DirectIoReader reader(path);

    // Whether direct or not depends on filesystem. Just verify it works.
    EXPECT_EQ(reader.file_size(), 64u + 5u * 16u);

    // Read some rows to verify data path works regardless of O_DIRECT state
    auto result = reader.read_rows({0, 1}, 16, 64);
    EXPECT_EQ(result.num_rows, 2u);
    EXPECT_EQ(result.size, 32u);
}

// =============================================================================
// read_rows tests
// =============================================================================

TEST_F(DirectIoReaderTest, ReadRowsReturnsCorrectData) {
    constexpr uint64_t HEADER = 64, ROWS = 10, ROW_BYTES = 16;
    auto path = create_test_file("rows.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    auto result = reader.read_rows({2, 7}, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 2u);
    ASSERT_EQ(result.size, 2u * ROW_BYTES);

    // Row 2 should have: 301, 302, 303, 304
    auto* r0 = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(r0[0], 301.0f);
    EXPECT_FLOAT_EQ(r0[1], 302.0f);
    EXPECT_FLOAT_EQ(r0[2], 303.0f);
    EXPECT_FLOAT_EQ(r0[3], 304.0f);

    // Row 7 should have: 801, 802, 803, 804
    auto* r1 = reinterpret_cast<const float*>(result.data.get() + ROW_BYTES);
    EXPECT_FLOAT_EQ(r1[0], 801.0f);
    EXPECT_FLOAT_EQ(r1[1], 802.0f);
    EXPECT_FLOAT_EQ(r1[2], 803.0f);
    EXPECT_FLOAT_EQ(r1[3], 804.0f);
}

TEST_F(DirectIoReaderTest, ReadRowsPreservesInputOrder) {
    // Read rows in reverse order — output must match input order, not file order
    constexpr uint64_t HEADER = 32, ROWS = 8, ROW_BYTES = 16;
    auto path = create_test_file("order.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    auto result = reader.read_rows({5, 3, 1, 7}, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 4u);

    auto get_first_float = [&](size_t idx) -> float {
        return *reinterpret_cast<const float*>(
            result.data.get() + idx * ROW_BYTES);
    };

    EXPECT_FLOAT_EQ(get_first_float(0), 601.0f);  // row 5
    EXPECT_FLOAT_EQ(get_first_float(1), 401.0f);  // row 3
    EXPECT_FLOAT_EQ(get_first_float(2), 201.0f);  // row 1
    EXPECT_FLOAT_EQ(get_first_float(3), 801.0f);  // row 7
}

TEST_F(DirectIoReaderTest, ReadRowsSingleRow) {
    constexpr uint64_t HEADER = 64, ROWS = 10, ROW_BYTES = 16;
    auto path = create_test_file("single.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    auto result = reader.read_rows({0}, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 1u);
    ASSERT_EQ(result.size, ROW_BYTES);

    auto* r = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(r[0], 101.0f);
    EXPECT_FLOAT_EQ(r[1], 102.0f);
    EXPECT_FLOAT_EQ(r[2], 103.0f);
    EXPECT_FLOAT_EQ(r[3], 104.0f);
}

TEST_F(DirectIoReaderTest, ReadRowsLastRow) {
    constexpr uint64_t HEADER = 64, ROWS = 10, ROW_BYTES = 16;
    auto path = create_test_file("last.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    auto result = reader.read_rows({9}, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 1u);
    auto* r = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(r[0], 1001.0f);  // row 9: (9+1)*100+1 = 1001
}

TEST_F(DirectIoReaderTest, ReadRowsAllRows) {
    constexpr uint64_t HEADER = 64, ROWS = 5, ROW_BYTES = 16;
    auto path = create_test_file("all.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    std::vector<uint64_t> all_indices = {0, 1, 2, 3, 4};
    auto result = reader.read_rows(all_indices, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 5u);
    ASSERT_EQ(result.size, 5u * ROW_BYTES);

    for (uint64_t r = 0; r < ROWS; ++r) {
        auto* row = reinterpret_cast<const float*>(
            result.data.get() + r * ROW_BYTES);
        EXPECT_FLOAT_EQ(row[0], static_cast<float>((r + 1) * 100 + 1))
            << "Mismatch at row " << r;
    }
}

TEST_F(DirectIoReaderTest, ReadRowsDuplicateIndices) {
    constexpr uint64_t HEADER = 64, ROWS = 10, ROW_BYTES = 16;
    auto path = create_test_file("dup.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    // Same row requested twice — both copies should be identical
    auto result = reader.read_rows({3, 3}, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 2u);
    auto* r0 = reinterpret_cast<const float*>(result.data.get());
    auto* r1 = reinterpret_cast<const float*>(result.data.get() + ROW_BYTES);
    for (int c = 0; c < 4; ++c) {
        EXPECT_FLOAT_EQ(r0[c], r1[c]) << "Duplicate mismatch at col " << c;
    }
    EXPECT_FLOAT_EQ(r0[0], 401.0f);
}

TEST_F(DirectIoReaderTest, EmptyReadReturnsEmpty) {
    constexpr uint64_t HEADER = 64, ROWS = 10, ROW_BYTES = 16;
    auto path = create_test_file("empty.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);

    // Empty row_indices
    auto result = reader.read_rows({}, ROW_BYTES, HEADER);
    EXPECT_EQ(result.num_rows, 0u);
    EXPECT_EQ(result.size, 0u);
    EXPECT_EQ(result.data.get(), nullptr);
}

TEST_F(DirectIoReaderTest, EmptyReadZeroRowBytes) {
    constexpr uint64_t HEADER = 64, ROWS = 10, ROW_BYTES = 16;
    auto path = create_test_file("zero_rb.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);

    // row_bytes == 0 => empty result
    auto result = reader.read_rows({0, 1}, 0, HEADER);
    EXPECT_EQ(result.num_rows, 0u);
    EXPECT_EQ(result.size, 0u);
}

// =============================================================================
// read_rows with non-aligned row sizes (typical for feature matrices)
// =============================================================================

TEST_F(DirectIoReaderTest, ReadRowsNonAlignedRowSize) {
    // Simulate a feature matrix with 1433 dims x float32 = 5732 bytes per row.
    // This is NOT aligned to any filesystem block boundary.
    constexpr uint64_t HEADER = 64;
    constexpr uint64_t ROWS = 4;
    constexpr uint64_t DIMS = 7;  // 7 floats = 28 bytes (not aligned to 512)
    constexpr uint64_t ROW_BYTES = DIMS * sizeof(float);

    auto path = create_test_file("nonalign.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    auto result = reader.read_rows({1, 3}, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 2u);

    auto* r0 = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(r0[0], 201.0f);  // row 1
    EXPECT_FLOAT_EQ(r0[6], 207.0f);

    auto* r1 = reinterpret_cast<const float*>(result.data.get() + ROW_BYTES);
    EXPECT_FLOAT_EQ(r1[0], 401.0f);  // row 3
    EXPECT_FLOAT_EQ(r1[6], 407.0f);
}

// =============================================================================
// read_range tests
// =============================================================================

TEST_F(DirectIoReaderTest, ReadRangeWorks) {
    constexpr uint64_t HEADER = 64, ROWS = 10, ROW_BYTES = 16;
    auto path = create_test_file("range.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);

    // Read the entire data section (past header)
    auto result = reader.read_range(HEADER, ROWS * ROW_BYTES);
    ASSERT_EQ(result.size, ROWS * ROW_BYTES);

    // Verify first and last row
    auto* first = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(first[0], 101.0f);

    auto* last = reinterpret_cast<const float*>(
        result.data.get() + 9 * ROW_BYTES);
    EXPECT_FLOAT_EQ(last[0], 1001.0f);
}

TEST_F(DirectIoReaderTest, ReadRangePartialRow) {
    constexpr uint64_t HEADER = 64, ROWS = 10, ROW_BYTES = 16;
    auto path = create_test_file("partial.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);

    // Read just the first float of row 5
    uint64_t offset = HEADER + 5 * ROW_BYTES;
    auto result = reader.read_range(offset, sizeof(float));
    ASSERT_EQ(result.size, sizeof(float));

    auto* val = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(*val, 601.0f);
}

TEST_F(DirectIoReaderTest, ReadRangeHeader) {
    constexpr uint64_t HEADER = 64;
    auto path = create_test_file("hdr.bin", HEADER, 1, 16);

    DirectIoReader reader(path);

    // Read just the header magic (first 4 bytes)
    auto result = reader.read_range(0, 4);
    ASSERT_EQ(result.size, 4u);

    uint32_t magic;
    std::memcpy(&magic, result.data.get(), sizeof(magic));
    EXPECT_EQ(magic, 0xDEADBEEFu);
}

TEST_F(DirectIoReaderTest, ReadRangeZeroSize) {
    auto path = create_test_file("zero.bin", 64, 1, 16);

    DirectIoReader reader(path);
    auto result = reader.read_range(0, 0);
    EXPECT_EQ(result.size, 0u);
    EXPECT_EQ(result.data.get(), nullptr);
}

// =============================================================================
// Capability queries
// =============================================================================

TEST_F(DirectIoReaderTest, IsDirectReportsState) {
    auto path = create_test_file("cap.bin", 64, 1, 16);
    DirectIoReader reader(path);

    // On tmpfs, O_DIRECT typically fails, so direct_ should be false.
    // On ext4/xfs, it should be true. Either way, the value is consistent.
    // Just verify the method doesn't crash and returns a bool.
    bool d = reader.is_direct();
    (void)d;  // suppress unused warning
}

TEST_F(DirectIoReaderTest, IsIoUringReportsState) {
    auto path = create_test_file("uring.bin", 64, 1, 16);
    DirectIoReader reader(path);

    bool u = reader.is_io_uring();
#ifndef ENABLE_IO_URING
    // Without ENABLE_IO_URING, this must always be false
    EXPECT_FALSE(u);
#else
    (void)u;  // With ENABLE_IO_URING, depends on kernel support
#endif
}

// =============================================================================
// Large row count — stress the batch loop
// =============================================================================

TEST_F(DirectIoReaderTest, ReadManyRows) {
    constexpr uint64_t HEADER = 32;
    constexpr uint64_t ROWS = 256;
    constexpr uint64_t ROW_BYTES = 16;
    auto path = create_test_file("many.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);

    // Read every other row
    std::vector<uint64_t> indices;
    for (uint64_t i = 0; i < ROWS; i += 2) {
        indices.push_back(i);
    }

    auto result = reader.read_rows(indices, ROW_BYTES, HEADER);
    ASSERT_EQ(result.num_rows, 128u);
    ASSERT_EQ(result.size, 128u * ROW_BYTES);

    // Verify first and last of the selected rows
    auto* first = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(first[0], 101.0f);  // row 0

    auto* last = reinterpret_cast<const float*>(
        result.data.get() + 127u * ROW_BYTES);
    EXPECT_FLOAT_EQ(last[0], 25501.0f);  // row 254: (254+1)*100+1 = 25501
}

// =============================================================================
// Scattered reads — rows far apart in the file
// =============================================================================

TEST_F(DirectIoReaderTest, ReadRowsScatteredAccess) {
    constexpr uint64_t HEADER = 64;
    constexpr uint64_t ROWS = 1000;
    constexpr uint64_t ROW_BYTES = 16;
    auto path = create_test_file("scatter.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);

    // Read rows that are far apart: 0, 499, 999
    auto result = reader.read_rows({0, 499, 999}, ROW_BYTES, HEADER);
    ASSERT_EQ(result.num_rows, 3u);

    auto get_first = [&](size_t idx) -> float {
        return *reinterpret_cast<const float*>(
            result.data.get() + idx * ROW_BYTES);
    };

    EXPECT_FLOAT_EQ(get_first(0), 101.0f);      // row 0
    EXPECT_FLOAT_EQ(get_first(1), 50001.0f);     // row 499: 500*100+1
    EXPECT_FLOAT_EQ(get_first(2), 100001.0f);    // row 999: 1000*100+1
}

// =============================================================================
// Large file with 4K+ rows — exercises io_uring batch chunking
// =============================================================================

TEST_F(DirectIoReaderTest, ReadRowsExceedsQueueDepth) {
    // QUEUE_DEPTH is 64; request more than that to exercise batching logic
    constexpr uint64_t HEADER = 32;
    constexpr uint64_t ROWS = 200;
    constexpr uint64_t ROW_BYTES = 16;
    auto path = create_test_file("qdepth.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);

    // Request 100 rows (> QUEUE_DEPTH of 64)
    std::vector<uint64_t> indices(100);
    std::iota(indices.begin(), indices.end(), 0);  // 0..99

    auto result = reader.read_rows(indices, ROW_BYTES, HEADER);
    ASSERT_EQ(result.num_rows, 100u);

    // Spot-check a few rows
    auto get_first = [&](size_t idx) -> float {
        return *reinterpret_cast<const float*>(
            result.data.get() + idx * ROW_BYTES);
    };

    EXPECT_FLOAT_EQ(get_first(0), 101.0f);      // row 0
    EXPECT_FLOAT_EQ(get_first(50), 5101.0f);     // row 50
    EXPECT_FLOAT_EQ(get_first(99), 10001.0f);    // row 99
}

// =============================================================================
// Various header sizes and data offsets
// =============================================================================

TEST_F(DirectIoReaderTest, ReadRowsWithZeroHeader) {
    constexpr uint64_t ROWS = 5, ROW_BYTES = 16;
    // No header at all — data starts at offset 0
    std::vector<char> data(ROWS * ROW_BYTES, 0);
    for (uint64_t r = 0; r < ROWS; ++r) {
        auto* row = reinterpret_cast<float*>(data.data() + r * ROW_BYTES);
        for (int c = 0; c < 4; ++c) {
            row[c] = static_cast<float>((r + 1) * 10 + c);
        }
    }
    auto path = create_raw_file("nohdr.bin", data);

    DirectIoReader reader(path);
    auto result = reader.read_rows({0, 4}, ROW_BYTES, 0);

    ASSERT_EQ(result.num_rows, 2u);
    auto* r0 = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(r0[0], 10.0f);  // row 0

    auto* r1 = reinterpret_cast<const float*>(result.data.get() + ROW_BYTES);
    EXPECT_FLOAT_EQ(r1[0], 50.0f);  // row 4
}

TEST_F(DirectIoReaderTest, ReadRowsWithLargeHeader) {
    // Header larger than typical block size (4096)
    constexpr uint64_t HEADER = 8192;
    constexpr uint64_t ROWS = 3;
    constexpr uint64_t ROW_BYTES = 16;
    auto path = create_test_file("bighdr.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    auto result = reader.read_rows({0, 2}, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 2u);
    auto* r0 = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(r0[0], 101.0f);  // row 0

    auto* r1 = reinterpret_cast<const float*>(result.data.get() + ROW_BYTES);
    EXPECT_FLOAT_EQ(r1[0], 301.0f);  // row 2
}

// =============================================================================
// Non-aligned row size matching Cora feature dimension
// Cora: 1433 dims x 4 bytes = 5732 bytes per row (NOT aligned to 512 or 4096)
// Tests O_DIRECT alignment handling with realistic feature matrix sizes
// =============================================================================

TEST_F(DirectIoReaderTest, NonAlignedRowSizeCora) {
    constexpr uint64_t HEADER = 64;
    constexpr uint64_t ROWS = 10;
    constexpr uint64_t DIMS = 1433;
    constexpr uint64_t ROW_BYTES = DIMS * sizeof(float);  // 5732 bytes

    // Create file with known float patterns
    std::vector<char> data(HEADER + ROWS * ROW_BYTES, 0);
    for (uint64_t r = 0; r < ROWS; ++r) {
        auto* row = reinterpret_cast<float*>(data.data() + HEADER + r * ROW_BYTES);
        for (uint64_t c = 0; c < DIMS; ++c) {
            row[c] = static_cast<float>((r + 1) * 1000 + c);
        }
    }
    auto path = create_raw_file("cora.bin", data);

    DirectIoReader reader(path);
    auto result = reader.read_rows({0, 5, 9}, ROW_BYTES, HEADER);

    ASSERT_EQ(result.num_rows, 3u);
    ASSERT_EQ(result.size, 3u * ROW_BYTES);

    // Row 0: (0+1)*1000 + dim
    auto* r0 = reinterpret_cast<const float*>(result.data.get());
    EXPECT_FLOAT_EQ(r0[0], 1000.0f);     // row 0, dim 0
    EXPECT_FLOAT_EQ(r0[1432], 2432.0f);  // row 0, dim 1432

    // Row 5: (5+1)*1000 + dim
    auto* r5 = reinterpret_cast<const float*>(result.data.get() + ROW_BYTES);
    EXPECT_FLOAT_EQ(r5[0], 6000.0f);     // row 5, dim 0
    EXPECT_FLOAT_EQ(r5[716], 6716.0f);   // row 5, middle dim
    EXPECT_FLOAT_EQ(r5[1432], 7432.0f);  // row 5, last dim

    // Row 9: (9+1)*1000 + dim
    auto* r9 = reinterpret_cast<const float*>(result.data.get() + 2 * ROW_BYTES);
    EXPECT_FLOAT_EQ(r9[0], 10000.0f);    // row 9, dim 0
    EXPECT_FLOAT_EQ(r9[1432], 11432.0f); // row 9, last dim
}

// =============================================================================
// Spec A1 (2026-04-27): bytes_disk in ReadResult — paper-comparable accounting
// =============================================================================

TEST_F(DirectIoReaderTest, BytesDiskMonotone_ReadRows) {
    // Cora-like row size (5732 B) is intentionally non-aligned so O_DIRECT
    // padding shows up clearly. With 4 KB blocks, each scattered row needs
    // ceil((skip + 5732) / 4096) * 4096 bytes — usually 8192, sometimes 12288.
    constexpr uint64_t HEADER    = 64;
    constexpr uint64_t DIMS      = 1433;
    constexpr uint64_t ROW_BYTES = DIMS * sizeof(float);  // 5732
    constexpr uint64_t ROWS      = 16;

    std::vector<char> data(HEADER + ROWS * ROW_BYTES, 0);
    auto path = create_raw_file("amp.bin", data);

    DirectIoReader reader(path);
    auto result = reader.read_rows({0, 3, 7, 11, 15}, ROW_BYTES, HEADER);

    EXPECT_GE(result.bytes_disk, result.size)
        << "bytes_disk must always be >= size (alignment is monotone overhead)";
    if (reader.is_direct()) {
        EXPECT_EQ(result.bytes_disk % 4096u, 0u)
            << "Under O_DIRECT every read is block-aligned";
    } else {
        // Fallback path: bytes_disk should equal the wanted bytes exactly
        // (plain pread, no alignment inflation).
        EXPECT_EQ(result.bytes_disk, result.size);
    }
}

TEST_F(DirectIoReaderTest, BytesDiskZero_EmptyInput) {
    auto path = create_raw_file("empty.bin", std::vector<char>(4096, 0));
    DirectIoReader reader(path);

    // Empty row list — no read should happen.
    auto r1 = reader.read_rows({}, 16, 0);
    EXPECT_EQ(r1.size, 0u);
    EXPECT_EQ(r1.num_rows, 0u);
    EXPECT_EQ(r1.bytes_disk, 0u);

    // Zero row_bytes — same.
    auto r2 = reader.read_rows({0}, 0, 0);
    EXPECT_EQ(r2.bytes_disk, 0u);

    // read_range with size=0 — same.
    auto r3 = reader.read_range(0, 0);
    EXPECT_EQ(r3.size, 0u);
    EXPECT_EQ(r3.bytes_disk, 0u);
}

TEST_F(DirectIoReaderTest, BytesDiskMonotone_ReadRange) {
    // Allocate a file large enough that requesting a sub-range from the
    // middle forces alignment expansion in both directions under O_DIRECT.
    constexpr uint64_t FILE_SIZE = 64 * 1024;
    std::vector<char> data(FILE_SIZE, static_cast<char>(0xCD));
    auto path = create_raw_file("range.bin", data);

    DirectIoReader reader(path);
    // Read 100 bytes starting at offset 1234 — neither end aligned.
    auto result = reader.read_range(1234, 100);

    EXPECT_EQ(result.size, 100u);
    EXPECT_GE(result.bytes_disk, result.size);
    if (reader.is_direct()) {
        EXPECT_EQ(result.bytes_disk % 4096u, 0u);
        // 100 bytes spanning [1234, 1334) lives entirely inside page [0, 4096),
        // so a single 4 KB aligned read is sufficient.
        EXPECT_EQ(result.bytes_disk, 4096u);
    } else {
        EXPECT_EQ(result.bytes_disk, 100u);
    }
}

// =============================================================================
// Spec A2 (2026-04-27): page-level dedup — merge overlapping aligned regions
// =============================================================================

TEST_F(DirectIoReaderTest, Dedup_SamePageMultipleRows) {
    // 4 rows of 512 B each, no header, all sharing the same 4 KB page.
    // Pre-A2: 4 × 4096 = 16 KB physical reads.
    // Post-A2: 1 × 4096 = 4 KB (4× reduction in disk traffic).
    constexpr uint64_t HEADER    = 0;
    constexpr uint64_t ROW_BYTES = 512;
    constexpr uint64_t ROWS      = 8;

    auto path = create_test_file("dedup_same_page.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    auto result = reader.read_rows({0, 1, 2, 3}, ROW_BYTES, HEADER);

    EXPECT_EQ(result.num_rows, 4u);
    EXPECT_EQ(result.size, 4u * ROW_BYTES);  // 2048 B wanted

    if (reader.is_direct()) {
        // All 4 rows share the same aligned [0, 4096) region — single read.
        EXPECT_EQ(result.bytes_disk, 4096u)
            << "Dedup must collapse same-page rows into one 4 KB read";
    } else {
        // tmpfs / fallback: pread per row, no alignment overhead.
        EXPECT_EQ(result.bytes_disk, 4u * ROW_BYTES);
    }

    // Output data correctness: each row must have its expected pattern.
    for (uint64_t r = 0; r < 4; ++r) {
        auto* row = reinterpret_cast<const float*>(
            result.data.get() + r * ROW_BYTES);
        EXPECT_FLOAT_EQ(row[0], static_cast<float>((r + 1) * 100 + 1));
    }
}

TEST_F(DirectIoReaderTest, Dedup_DisjointRowsNotMerged) {
    // 2 rows in distant pages — aligned regions do not touch, so dedup
    // produces 2 separate spans of 4 KB each.
    constexpr uint64_t HEADER    = 0;
    constexpr uint64_t ROW_BYTES = 512;
    constexpr uint64_t ROWS      = 32;  // 4 pages worth of rows (8 rows/page)

    auto path = create_test_file("dedup_disjoint.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    // Row 0 is in page [0, 4096); row 16 is in page [8192, 12288).
    auto result = reader.read_rows({0, 16}, ROW_BYTES, HEADER);

    EXPECT_EQ(result.num_rows, 2u);
    EXPECT_EQ(result.size, 2u * ROW_BYTES);

    if (reader.is_direct()) {
        // Two disjoint pages — no merging, 2 × 4096 = 8 KB.
        EXPECT_EQ(result.bytes_disk, 8192u);
    } else {
        EXPECT_EQ(result.bytes_disk, 2u * ROW_BYTES);
    }

    // Output correctness.
    auto* r0 = reinterpret_cast<const float*>(result.data.get());
    auto* r1 = reinterpret_cast<const float*>(result.data.get() + ROW_BYTES);
    EXPECT_FLOAT_EQ(r0[0], 101.0f);            // row 0
    EXPECT_FLOAT_EQ(r1[0], 17.0f * 100 + 1);   // row 16 = 1701
}

// =============================================================================
// Spec A3 (2026-04-27): multi-ring parallel submission stress test
// =============================================================================

TEST_F(DirectIoReaderTest, MultiRing_StressManySpans) {
    // Each row is exactly one 4 KB page, so dedup cannot merge any spans —
    // 2048 rows produce 2048 distinct spans, exceeding QUEUE_DEPTH=1024.
    // This forces submit_io_uring_spans to leave the single-ring fast path
    // and dispatch across multiple rings via std::async.
    constexpr uint64_t HEADER    = 0;
    constexpr uint64_t ROW_BYTES = 4096;
    constexpr uint64_t ROWS      = 2048;

    auto path = create_test_file("multi_ring.bin", HEADER, ROWS, ROW_BYTES);
    DirectIoReader reader(path);

    std::vector<uint64_t> all_indices(ROWS);
    for (uint64_t i = 0; i < ROWS; ++i) all_indices[i] = i;

    auto result = reader.read_rows(all_indices, ROW_BYTES, HEADER);

    EXPECT_EQ(result.num_rows, ROWS);
    EXPECT_EQ(result.size, ROWS * ROW_BYTES);

    if (reader.is_direct()) {
        // Each row exactly fills one aligned 4 KB block — total bytes_disk
        // equals the wanted bytes (no alignment overhead per row).
        EXPECT_EQ(result.bytes_disk, ROWS * ROW_BYTES);
    }

    // Output correctness across all 2048 rows — verifies the multi-ring
    // scatter assembled them in original-row order with no cross-thread
    // data corruption.
    for (uint64_t r = 0; r < ROWS; ++r) {
        auto* row = reinterpret_cast<const float*>(
            result.data.get() + r * ROW_BYTES);
        EXPECT_FLOAT_EQ(row[0], static_cast<float>((r + 1) * 100 + 1))
            << "Row " << r << " first float mismatch — multi-ring scatter bug?";
    }
}

TEST_F(DirectIoReaderTest, Dedup_StraddlePageBoundary) {
    // Cora-like row size (5732 B) — each row spans ~1.4 pages, so adjacent
    // rows have aligned regions that touch but do not overlap with one row.
    // Two adjacent rows share their boundary page → merge into one span.
    constexpr uint64_t HEADER    = 64;
    constexpr uint64_t DIMS      = 1433;
    constexpr uint64_t ROW_BYTES = DIMS * sizeof(float);  // 5732
    constexpr uint64_t ROWS      = 4;

    auto path = create_test_file("dedup_straddle.bin", HEADER, ROWS, ROW_BYTES);

    DirectIoReader reader(path);
    // Read consecutive rows — their aligned regions span overlapping pages.
    auto result = reader.read_rows({0, 1}, ROW_BYTES, HEADER);

    EXPECT_EQ(result.num_rows, 2u);
    EXPECT_EQ(result.size, 2u * ROW_BYTES);

    if (reader.is_direct()) {
        // Row 0: file_offset=64, end=5796 → aligned [0, 8192).
        // Row 1: file_offset=5796, end=11528 → aligned [4096, 12288).
        // Merged: [0, 12288) = 12 KB. Pre-A2 would be 8 + 8 = 16 KB.
        EXPECT_EQ(result.bytes_disk, 12288u)
            << "Adjacent rows on shared page must merge to single span";
    } else {
        EXPECT_EQ(result.bytes_disk, 2u * ROW_BYTES);
    }

    // Output correctness — row 0 starts with 101.0, row 1 with 201.0.
    auto* r0 = reinterpret_cast<const float*>(result.data.get());
    auto* r1 = reinterpret_cast<const float*>(result.data.get() + ROW_BYTES);
    EXPECT_FLOAT_EQ(r0[0], 101.0f);
    EXPECT_FLOAT_EQ(r0[DIMS - 1], static_cast<float>(100 + DIMS));   // 1533
    EXPECT_FLOAT_EQ(r1[0], 201.0f);
    EXPECT_FLOAT_EQ(r1[DIMS - 1], static_cast<float>(200 + DIMS));   // 1633
}
