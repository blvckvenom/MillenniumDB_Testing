#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <torch/torch.h>
#include "gnn/storage/block_format.h"
#include "gnn/storage/block_store.h"
using namespace mdb::gnn;

TEST(BlockFormat, MakeAndValidate) {
    auto h = BlockBatchHeader::make(/*num_layers=*/3, /*sample_fp=*/0xABCDull);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.magic, BlockBatchHeader::MAGIC);
    EXPECT_EQ(h.version, BlockBatchHeader::VERSION);
    EXPECT_EQ(h.num_layers, 3u);
    EXPECT_EQ(h.sample_fp, 0xABCDull);
}
TEST(BlockFormat, RejectsBadMagic) {
    BlockBatchHeader h{};  // zeroed
    EXPECT_FALSE(h.is_valid());
}
static_assert(sizeof(BlockBatchHeader) == 64, "block header 64 B");

TEST(BlockStore, WriteThenReadRoundTrip) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_rt";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {5, 3, 2};                   // M_0..M_2 (K+1=3 node layers => 2 conv layers)
    std::vector<torch::Tensor> edges = {                       // 2 conv layers
        torch::tensor({{0,1,2},{2,2,1}}, torch::kInt64),      // [2,3]
        torch::tensor({{0},{1}}, torch::kInt64),              // [2,1]
    };
    BlockWriter::write(tmp / "block_000000.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges);
    auto blk = BlockReader::open(tmp / "block_000000.blk", /*expected_sample_fp=*/0xFEEDull);
    ASSERT_TRUE(blk.has_value());
    EXPECT_EQ(blk->active_sizes, sizes);
    ASSERT_EQ(blk->edge_indices.size(), 2u);
    EXPECT_TRUE(torch::equal(blk->edge_indices[0], edges[0]));  // widened back to int64
    EXPECT_TRUE(torch::equal(blk->edge_indices[1], edges[1]));
}
TEST(BlockStore, StalenessRejectsWrongFp) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_stale";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {2, 1};                       // K=1 conv layer
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    BlockWriter::write(tmp / "b.blk", 0x1111ull, 0, sizes, edges);
    EXPECT_FALSE(BlockReader::open(tmp / "b.blk", /*expected=*/0x2222ull).has_value()); // stale → reject
    EXPECT_TRUE (BlockReader::open(tmp / "b.blk", /*expected=*/0x1111ull).has_value()); // fresh → accept
}
TEST(BlockStore, MissingFileReturnsNullopt) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_missing";
    std::filesystem::create_directories(tmp);
    EXPECT_FALSE(BlockReader::open(tmp / "does_not_exist.blk", 0).has_value());
}
