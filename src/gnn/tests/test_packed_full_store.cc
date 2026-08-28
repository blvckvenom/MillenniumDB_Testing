#include <gtest/gtest.h>
#include "gnn/storage/packed_full_format.h"
#include "gnn/storage/packed_full_store.h"
#include "gnn/storage/feature_matrix.h"
#include <filesystem>
#include <vector>
using namespace mdb::gnn;

TEST(PackedFullFormat, MakeAndValidate) {
    auto h = PackedFullHeader::make(/*store_fp=*/0xABCDull, /*feature_dim=*/128,
                                    /*dtype=*/0, /*num_batches=*/10, /*row_bytes=*/512);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.magic, PackedFullHeader::MAGIC);
    EXPECT_EQ(h.version, PackedFullHeader::VERSION);
    EXPECT_EQ(h.store_fp, 0xABCDull);
    EXPECT_EQ(h.feature_dim, 128u);
    EXPECT_EQ(h.num_batches, 10u);
    EXPECT_EQ(h.row_bytes, 512u);
}
TEST(PackedFullFormat, RejectsBadMagic) {
    PackedFullHeader h{};  // zeroed
    EXPECT_FALSE(h.is_valid());
}
static_assert(sizeof(PackedFullHeader) == 64, "packed_full header 64 B");
static_assert(sizeof(PackedFullEntry) == 24, "packed_full entry 24 B");

TEST(PackedFullStore, WriteThenReadRoundTrip) {
    auto tmp = std::filesystem::temp_directory_path() / "pf_rt";
    std::filesystem::remove_all(tmp); std::filesystem::create_directories(tmp);
    const uint32_t D = 4; const uint64_t row_bytes = D * sizeof(float);
    PackedFullWriter w(tmp, /*store_fp=*/0xFEEDull, /*feature_dim=*/D, /*dtype=*/0, row_bytes);
    std::vector<float> b0 = {1,2,3,4, 5,6,7,8};      // 2 nodes
    std::vector<float> b1 = {9,10,11,12};            // 1 node
    w.write_batch(0, b0.data(), /*num_nodes=*/2);
    w.write_batch(1, b1.data(), /*num_nodes=*/1);
    w.finalize();
    auto r = PackedFullReader::open(tmp, /*expected_store_fp=*/0xFEEDull);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->num_batches(), 2u);
    auto e0 = r->entry(0); EXPECT_EQ(e0.num_nodes, 2u); EXPECT_EQ(e0.length, 2*row_bytes);
    auto e1 = r->entry(1); EXPECT_EQ(e1.num_nodes, 1u); EXPECT_EQ(e1.length, 1*row_bytes);
    EXPECT_EQ(e1.offset % PackedFullHeader::ALIGNMENT, 0u);  // O_DIRECT aligned
    std::vector<float> got(2*D);
    r->read_payload_for_test(0, got.data());
    EXPECT_EQ(got, b0);
}
TEST(PackedFullStore, StalenessRejectsWrongStoreFp) {
    auto tmp = std::filesystem::temp_directory_path() / "pf_stale";
    std::filesystem::remove_all(tmp); std::filesystem::create_directories(tmp);
    PackedFullWriter w(tmp, 0x1111ull, 4, 0, 16);
    std::vector<float> b = {1,2,3,4}; w.write_batch(0, b.data(), 1); w.finalize();
    EXPECT_FALSE(PackedFullReader::open(tmp, /*expected=*/0x2222ull).has_value());
    EXPECT_TRUE (PackedFullReader::open(tmp, /*expected=*/0x1111ull).has_value());
}
TEST(PackedFullStore, OutOfOrderWriteBatch) {
    auto tmp = std::filesystem::temp_directory_path() / "pf_ooo";
    std::filesystem::remove_all(tmp); std::filesystem::create_directories(tmp);
    const uint32_t D = 4; const uint64_t row_bytes = D * sizeof(float);
    PackedFullWriter w(tmp, 0xC0DEull, D, 0, row_bytes);
    std::vector<float> b2 = {20,21,22,23};
    std::vector<float> b0 = {0,1,2,3};
    std::vector<float> b1 = {10,11,12,13};
    w.write_batch(2, b2.data(), 1);  // out of order: 2, then 0, then 1
    w.write_batch(0, b0.data(), 1);
    w.write_batch(1, b1.data(), 1);
    w.finalize();
    auto r = PackedFullReader::open(tmp, 0xC0DEull);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->num_batches(), 3u);
    for (uint64_t i = 0; i < 3; ++i) {
        auto e = r->entry(i);
        EXPECT_EQ(e.num_nodes, 1u);
        EXPECT_EQ(e.length, row_bytes);
        EXPECT_EQ(e.offset % PackedFullHeader::ALIGNMENT, 0u);
    }
    std::vector<float> got(D);
    r->read_payload_for_test(0, got.data()); EXPECT_EQ(got, b0);
    r->read_payload_for_test(1, got.data()); EXPECT_EQ(got, b1);
    r->read_payload_for_test(2, got.data()); EXPECT_EQ(got, b2);
}
TEST(PackedFullStore, ZeroNodeBatch) {
    auto tmp = std::filesystem::temp_directory_path() / "pf_zero";
    std::filesystem::remove_all(tmp); std::filesystem::create_directories(tmp);
    const uint32_t D = 4; const uint64_t row_bytes = D * sizeof(float);
    PackedFullWriter w(tmp, 0xBEEFull, D, 0, row_bytes);
    w.write_batch(0, nullptr, /*num_nodes=*/0);   // empty batch, length 0
    std::vector<float> b1 = {1,2,3,4};
    w.write_batch(1, b1.data(), 1);
    w.finalize();
    auto r = PackedFullReader::open(tmp, 0xBEEFull);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->num_batches(), 2u);
    auto e0 = r->entry(0); EXPECT_EQ(e0.num_nodes, 0u); EXPECT_EQ(e0.length, 0u);
    auto e1 = r->entry(1);
    EXPECT_EQ(e1.num_nodes, 1u); EXPECT_EQ(e1.length, row_bytes);
    EXPECT_EQ(e1.offset % PackedFullHeader::ALIGNMENT, 0u);
    std::vector<float> got(D);
    r->read_payload_for_test(1, got.data()); EXPECT_EQ(got, b1);
}

TEST(PackedFullStore, PayloadEqualsRowGather) {
    auto tmp = std::filesystem::temp_directory_path() / "pf_gather";
    std::filesystem::remove_all(tmp); std::filesystem::create_directories(tmp);

    // Source FeatureMatrix: N=6 nodes, D=4 float32, row r = {r*10+0, r*10+1, r*10+2, r*10+3}.
    const uint64_t N = 6, D = 4;
    std::vector<float> src(N * D);
    for (uint64_t r = 0; r < N; ++r)
        for (uint64_t c = 0; c < D; ++c)
            src[r * D + c] = static_cast<float>(r * 10 + c);
    auto fmat_path = tmp / "src.fmat";
    FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, src.data());
    auto fm = FeatureMatrix::open(fmat_path);

    // A fake "all_unique_nodes order": rows 5,0,3,1 (4 nodes), arbitrary order.
    std::vector<uint64_t> rows = {5, 0, 3, 1};
    const uint64_t row_bytes = fm.row_bytes();           // D * sizeof(float)
    std::vector<char> expected(rows.size() * row_bytes);
    fm.extract_rows(rows, expected.data());              // the row-gather output

    // Write the SAME gather through the packed-full writer.
    PackedFullWriter w(tmp / "packed_full", /*store_fp=*/0xC0FFEEull,
                       static_cast<uint32_t>(fm.num_cols()),
                       static_cast<uint32_t>(fm.dtype()), row_bytes);
    w.write_batch(0, expected.data(), rows.size());
    w.finalize();

    // Read it back and assert byte-for-byte equality with the row-gather.
    auto r = PackedFullReader::open(tmp / "packed_full", /*expected_store_fp=*/0xC0FFEEull);
    ASSERT_TRUE(r.has_value());
    auto e0 = r->entry(0);
    EXPECT_EQ(e0.num_nodes, rows.size());
    EXPECT_EQ(e0.length, rows.size() * row_bytes);
    std::vector<char> got(rows.size() * row_bytes);
    r->read_payload_for_test(0, got.data());
    EXPECT_EQ(got, expected);   // packed_full payload == row-gather, bit-identical

    // And spot-check the actual float values match the source rows in order.
    const float* gf = reinterpret_cast<const float*>(got.data());
    for (size_t i = 0; i < rows.size(); ++i)
        for (uint64_t c = 0; c < D; ++c)
            EXPECT_FLOAT_EQ(gf[i * D + c], src[rows[i] * D + c]);
}
