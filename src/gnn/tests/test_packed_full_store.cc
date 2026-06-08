#include <gtest/gtest.h>
#include "gnn/storage/packed_full_format.h"
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
