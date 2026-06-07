#include <gtest/gtest.h>
#include "gnn/storage/block_format.h"
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
