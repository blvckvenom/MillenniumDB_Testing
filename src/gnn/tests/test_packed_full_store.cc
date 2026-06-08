#include <gtest/gtest.h>
#include "gnn/storage/packed_full_format.h"
#include "gnn/storage/packed_full_store.h"
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
