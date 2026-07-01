// src/gnn/tests/test_consolidated_slim_format.cc
#include "gnn/storage/consolidated_slim.h"

#include <gtest/gtest.h>
#include <cstring>
#include <type_traits>

using namespace mdb::gnn;

TEST(ConsolidatedSlimFormat, HeaderIs64Bytes) {
    EXPECT_EQ(sizeof(ConsolidatedSlimHeader), 64u);
    EXPECT_EQ(ConsolidatedSlimHeader::SIZE, 64u);
    EXPECT_TRUE(std::is_standard_layout_v<ConsolidatedSlimHeader>);
    EXPECT_TRUE(std::is_trivially_copyable_v<ConsolidatedSlimHeader>);
}

TEST(ConsolidatedSlimFormat, MakeSetsMagicVersionAndFields) {
    auto h = ConsolidatedSlimHeader::make(/*num_batches=*/1512, /*D=*/128,
                                          /*dtype=*/1, /*perm_fp=*/0xDEADBEEFCAFEBABEull,
                                          /*meta_sha=*/0x0123456789ABCDEFull);
    EXPECT_EQ(h.magic, 0x43534C4Du);
    EXPECT_EQ(h.version, 1u);
    EXPECT_EQ(h.num_batches, 1512u);
    EXPECT_EQ(h.feature_dim, 128u);
    EXPECT_EQ(h.dtype, 1u);
    EXPECT_EQ(h.alignment_log2, 12u);
    EXPECT_EQ(h.alignment(), 4096u);
    EXPECT_EQ(h.perm_fingerprint, 0xDEADBEEFCAFEBABEull);
    EXPECT_EQ(h.meta_sha256_head, 0x0123456789ABCDEFull);
    EXPECT_TRUE(h.is_valid());
}

TEST(ConsolidatedSlimFormat, DataStartIsAlignedPastHeader) {
    auto h = ConsolidatedSlimHeader::make(10, 128, 1, 0, 0);
    // align_up(64, 4096) == 4096
    EXPECT_EQ(h.data_start, 4096u);
    EXPECT_GE(h.data_start, ConsolidatedSlimHeader::SIZE);
    EXPECT_EQ(h.data_start % h.alignment(), 0u);
}

TEST(ConsolidatedSlimFormat, AlignUp) {
    EXPECT_EQ(ConsolidatedSlimHeader::align_up(0, 4096), 0u);
    EXPECT_EQ(ConsolidatedSlimHeader::align_up(1, 4096), 4096u);
    EXPECT_EQ(ConsolidatedSlimHeader::align_up(4096, 4096), 4096u);
    EXPECT_EQ(ConsolidatedSlimHeader::align_up(4097, 4096), 8192u);
    EXPECT_EQ(ConsolidatedSlimHeader::align_up(64, 4096), 4096u);
    EXPECT_EQ(ConsolidatedSlimHeader::align_up(123, 0), 123u);  // a==0 => identity
}

TEST(ConsolidatedSlimFormat, IsValidRejectsBadMagic) {
    auto h = ConsolidatedSlimHeader::make(1, 128, 1, 0, 0);
    h.magic = 0xDEADBEEFu;
    EXPECT_FALSE(h.is_valid());
}

TEST(ConsolidatedSlimFormat, IsValidRejectsBadVersion) {
    auto h = ConsolidatedSlimHeader::make(1, 128, 1, 0, 0);
    h.version = 99;
    EXPECT_FALSE(h.is_valid());
}

TEST(ConsolidatedSlimFormat, IsValidRejectsZeroFeatureDim) {
    auto h = ConsolidatedSlimHeader::make(1, /*D=*/0, 1, 0, 0);
    EXPECT_FALSE(h.is_valid());
}

TEST(ConsolidatedSlimFormat, IsValidRejectsInconsistentDataStart) {
    auto h = ConsolidatedSlimHeader::make(1, 128, 1, 0, 0);
    h.data_start = 100;  // not align_up(64, 4096)
    EXPECT_FALSE(h.is_valid());
}

TEST(ConsolidatedSlimFormat, IsValidRejectsInsaneAlignment) {
    auto h = ConsolidatedSlimHeader::make(1, 128, 1, 0, 0, /*alignment_log2=*/40);
    EXPECT_FALSE(h.is_valid());  // 2^40 alignment is out of [9,30]
}

TEST(ConsolidatedSlimFormat, IsValidRejectsZeroMemory) {
    ConsolidatedSlimHeader h{};
    EXPECT_EQ(h.magic, 0u);
    EXPECT_FALSE(h.is_valid());
}

TEST(ConsolidatedSlimFormat, BinaryRoundTrip) {
    auto orig = ConsolidatedSlimHeader::make(1512, 128, 1,
                                             0xAAAABBBBCCCCDDDDull, 0x1111222233334444ull);
    uint8_t buf[ConsolidatedSlimHeader::SIZE];
    std::memcpy(buf, &orig, ConsolidatedSlimHeader::SIZE);
    ConsolidatedSlimHeader copy{};
    std::memcpy(&copy, buf, ConsolidatedSlimHeader::SIZE);
    EXPECT_TRUE(copy.is_valid());
    EXPECT_EQ(copy.num_batches, 1512u);
    EXPECT_EQ(copy.feature_dim, 128u);
    EXPECT_EQ(copy.perm_fingerprint, 0xAAAABBBBCCCCDDDDull);
    EXPECT_EQ(copy.meta_sha256_head, 0x1111222233334444ull);
    EXPECT_EQ(copy.data_start, 4096u);
}

TEST(ConsolidatedSlimFormat, NonDefaultAlignment) {
    auto h = ConsolidatedSlimHeader::make(1, 128, 1, 0, 0, /*alignment_log2=*/9);  // 512B
    EXPECT_EQ(h.alignment(), 512u);
    EXPECT_EQ(h.data_start, 512u);  // align_up(64, 512) == 512
    EXPECT_TRUE(h.is_valid());
}
