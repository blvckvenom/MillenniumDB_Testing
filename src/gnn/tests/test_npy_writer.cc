#include "test_helpers.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "gnn/training/npy_writer.h"

using NpyWriterTest = GnnStorageTest;

namespace fs = std::filesystem;

// ============================================================================
// Test 1: magic bytes
// ============================================================================

TEST_F(NpyWriterTest, WriteFloat32MagicBytes) {
    auto t    = torch::randn({10, 4});
    auto path = test_path("magic_f32.npy");
    mdb::gnn::NpyWriter::write_float32(path, t);

    ASSERT_TRUE(fs::exists(path));

    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    char magic[6];
    f.read(magic, 6);
    EXPECT_EQ(std::string(magic, 6), std::string("\x93NUMPY", 6));

    char ver_major, ver_minor;
    f.get(ver_major);
    f.get(ver_minor);
    EXPECT_EQ(static_cast<uint8_t>(ver_major), 1u);
    EXPECT_EQ(static_cast<uint8_t>(ver_minor), 0u);
}

// ============================================================================
// Test 2: file size for 2D float32
// ============================================================================

TEST_F(NpyWriterTest, WriteFloat32CorrectSize) {
    // 10 rows x 4 columns of float32 → data = 10*4*4 = 160 bytes
    auto t    = torch::randn({10, 4});
    auto path = test_path("size_f32.npy");
    mdb::gnn::NpyWriter::write_float32(path, t);

    ASSERT_TRUE(fs::exists(path));

    // Preamble is 10 bytes; file must be multiple of 64 (from preamble) + data.
    auto file_size = fs::file_size(path);
    constexpr size_t data_bytes = 10 * 4 * 4;  // 160 bytes
    EXPECT_GE(file_size, data_bytes);

    // The preamble+header together must be a multiple of 64.
    size_t header_section = file_size - data_bytes;
    EXPECT_EQ(header_section % 64, 0u);
}

// ============================================================================
// Test 3: correct HEADER_LEN field (uint16 LE, bytes 8-9)
// ============================================================================

TEST_F(NpyWriterTest, WriteFloat32HeaderLen) {
    auto t    = torch::randn({10, 4});
    auto path = test_path("headerlen_f32.npy");
    mdb::gnn::NpyWriter::write_float32(path, t);

    std::ifstream f(path, std::ios::binary);
    f.seekg(8);  // skip magic (6) + version (2)
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);

    // Verify it is a multiple of 8 (NumPy guarantees multiples of 64 total,
    // so HEADER_LEN is always a multiple of 64 - 10 + k*64 which is ≡54 mod64
    // ... actually just check preamble+header is a multiple of 64).
    EXPECT_EQ((10u + header_len) % 64, 0u);

    // Also verify that the file total = 10 + header_len + data_bytes
    auto file_size = fs::file_size(path);
    constexpr size_t data_bytes = 10 * 4 * 4;
    EXPECT_EQ(file_size, static_cast<size_t>(10 + header_len + data_bytes));
}

// ============================================================================
// Test 4: 2D float32 — header contains correct dtype and shape
// ============================================================================

TEST_F(NpyWriterTest, WriteFloat32HeaderContent) {
    auto t    = torch::randn({5, 3});
    auto path = test_path("content_f32.npy");
    mdb::gnn::NpyWriter::write_float32(path, t);

    // Read the header string
    std::ifstream f(path, std::ios::binary);
    f.seekg(8);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    EXPECT_NE(header.find("<f4"), std::string::npos);
    EXPECT_NE(header.find("False"), std::string::npos);
    EXPECT_NE(header.find("(5, 3)"), std::string::npos);
    EXPECT_EQ(header.back(), '\n');
}

// ============================================================================
// Test 5: 1D float32 — trailing comma in shape
// ============================================================================

TEST_F(NpyWriterTest, WriteFloat32_1D_Shape) {
    auto t    = torch::randn({20});
    auto path = test_path("1d_f32.npy");
    mdb::gnn::NpyWriter::write_float32(path, t);

    std::ifstream f(path, std::ios::binary);
    f.seekg(8);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // 1D shape must have trailing comma: "(20,)"
    EXPECT_NE(header.find("(20,)"), std::string::npos);
    EXPECT_NE(header.find("<f4"), std::string::npos);
}

// ============================================================================
// Test 6: int64 magic bytes and size
// ============================================================================

TEST_F(NpyWriterTest, WriteInt64CorrectSize) {
    // 15 elements × 8 bytes = 120 bytes data
    auto t    = torch::zeros({15}, torch::kInt64);
    auto path = test_path("size_i64.npy");
    mdb::gnn::NpyWriter::write_int64(path, t);

    ASSERT_TRUE(fs::exists(path));

    auto file_size = fs::file_size(path);
    constexpr size_t data_bytes = 15 * 8;
    EXPECT_GE(file_size, data_bytes);

    size_t header_section = file_size - data_bytes;
    EXPECT_EQ(header_section % 64, 0u);
}

// ============================================================================
// Test 7: int64 header contains correct dtype descriptor
// ============================================================================

TEST_F(NpyWriterTest, WriteInt64HeaderContent) {
    auto t    = torch::arange(6, torch::kInt64).reshape({2, 3});
    auto path = test_path("content_i64.npy");
    mdb::gnn::NpyWriter::write_int64(path, t);

    std::ifstream f(path, std::ios::binary);
    f.seekg(8);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    EXPECT_NE(header.find("<i8"),  std::string::npos);
    EXPECT_NE(header.find("(2, 3)"), std::string::npos);
}

// ============================================================================
// Test 8: dtype mismatch throws
// ============================================================================

TEST_F(NpyWriterTest, DtypeMismatchFloat32Throws) {
    auto t    = torch::zeros({5}, torch::kInt64);
    auto path = test_path("mismatch_f32.npy");
    EXPECT_THROW(mdb::gnn::NpyWriter::write_float32(path, t),
                 std::runtime_error);
}

TEST_F(NpyWriterTest, DtypeMismatchInt64Throws) {
    auto t    = torch::randn({5});
    auto path = test_path("mismatch_i64.npy");
    EXPECT_THROW(mdb::gnn::NpyWriter::write_int64(path, t),
                 std::runtime_error);
}

// ============================================================================
// Test 9: data bytes round-trip (write then read back raw data)
// ============================================================================

TEST_F(NpyWriterTest, WriteFloat32DataRoundtrip) {
    // Known values so we can read them back.
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t    = torch::tensor(vals).reshape({2, 3});
    auto path = test_path("roundtrip_f32.npy");
    mdb::gnn::NpyWriter::write_float32(path, t);

    // Compute data offset
    std::ifstream f(path, std::ios::binary);
    f.seekg(8);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    f.seekg(10 + header_len);

    // Read back the 6 floats
    std::vector<float> readback(6);
    f.read(reinterpret_cast<char*>(readback.data()),
           static_cast<std::streamsize>(6 * sizeof(float)));

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(readback[i], vals[i]);
    }
}

// ============================================================================
// Test 10: int64 data round-trip
// ============================================================================

TEST_F(NpyWriterTest, WriteInt64DataRoundtrip) {
    std::vector<int64_t> vals = {0, 1, 2, 3, 100, -1};
    auto t    = torch::tensor(vals);
    auto path = test_path("roundtrip_i64.npy");
    mdb::gnn::NpyWriter::write_int64(path, t);

    std::ifstream f(path, std::ios::binary);
    f.seekg(8);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    f.seekg(10 + header_len);

    std::vector<int64_t> readback(6);
    f.read(reinterpret_cast<char*>(readback.data()),
           static_cast<std::streamsize>(6 * sizeof(int64_t)));

    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(readback[i], vals[i]);
    }
}

// ============================================================================
// Test 11: nested directory is created automatically
// ============================================================================

TEST_F(NpyWriterTest, ParentDirCreatedAutomatically) {
    auto path = test_path("nested/subdir/out.npy");
    ASSERT_FALSE(fs::exists(path.parent_path()));

    auto t = torch::randn({3, 3});
    mdb::gnn::NpyWriter::write_float32(path, t);

    EXPECT_TRUE(fs::exists(path));
}

// ============================================================================
// Test 12: non-contiguous tensor is handled (transpose)
// ============================================================================

TEST_F(NpyWriterTest, NonContiguousTensorHandled) {
    // Transpose makes it non-contiguous in memory.
    auto t    = torch::randn({4, 3}).t();  // shape [3, 4], non-contiguous
    auto path = test_path("noncontig.npy");
    // Should not throw — write_impl calls .contiguous() internally.
    EXPECT_NO_THROW(mdb::gnn::NpyWriter::write_float32(path, t));

    auto file_size = fs::file_size(path);
    constexpr size_t data_bytes = 3 * 4 * 4;  // 48 bytes
    EXPECT_GE(file_size, data_bytes);
}
