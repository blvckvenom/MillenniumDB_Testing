// src/gnn/tests/test_addr_table_format.cc
#include "gnn/storage/addr_table.h"

#include <gtest/gtest.h>
#include <cstring>

using namespace mdb::gnn;

TEST(AddrTableFormat, MakeSetsMagicAndVersion) {
    auto h = AddrTableHeader::make(1, 0, 0, 0, 0, 0);
    EXPECT_EQ(h.magic, 0x41444452u);
    EXPECT_EQ(h.version, 1u);
}

TEST(AddrTableFormat, HeaderStructIs56Bytes) {
    // v2 in-memory struct (adds slim_offset/slim_length).
    EXPECT_EQ(sizeof(AddrTableHeader), 56u);
}

TEST(AddrTableFormat, OnDiskV1HeaderIs40Bytes) {
    // Backwards-compat: a v1 header still occupies 40 bytes on disk so existing
    // addr_tables round-trip byte-identically.
    EXPECT_EQ(AddrTableHeader::SIZE_V1, 40u);
    auto h = AddrTableHeader::make(1, 2, 3, 4, 5, 0);
    EXPECT_EQ(h.version, 1u);
    EXPECT_EQ(h.header_bytes(), 40u);
}

TEST(AddrTableFormat, MakeV2SetsVersionAndSlimFields) {
    auto h = AddrTableHeader::make_v2(1, 2, 3, 4, 5, 0xABCD, 0x4000, 0x800);
    EXPECT_EQ(h.version, AddrTableHeader::VERSION_V2);
    EXPECT_EQ(h.version, 2u);
    EXPECT_EQ(h.slim_offset, 0x4000ull);
    EXPECT_EQ(h.slim_length, 0x800ull);
    EXPECT_EQ(h.header_bytes(), 56u);
    EXPECT_EQ(h.total, 15u);
    EXPECT_TRUE(h.is_valid());
}

TEST(AddrTableFormat, V1SlimFieldsAreZero) {
    auto h = AddrTableHeader::make(7, 0, 0, 0, 0, 0);
    EXPECT_EQ(h.slim_offset, 0ull);
    EXPECT_EQ(h.slim_length, 0ull);
}

TEST(AddrTableFormat, IsValidAcceptsV1AndV2RejectsV3) {
    auto v1 = AddrTableHeader::make(1, 0, 0, 0, 0, 0);
    auto v2 = AddrTableHeader::make_v2(1, 0, 0, 0, 0, 0, 0, 0);
    EXPECT_TRUE(v1.is_valid());
    EXPECT_TRUE(v2.is_valid());
    v2.version = 3;
    EXPECT_FALSE(v2.is_valid());
}

TEST(AddrTableFormat, V2ExpectedFileSizeUses56ByteHeader) {
    auto h = AddrTableHeader::make_v2(2, 3, 5, 7, 0, 0, 0x1000, 0x40);
    size_t expected =
        56 + (2*4 + 2*4) + (3*4 + 3*4) + (5*4 + 5*8) + (7*4 + 7*4) + 0;
    EXPECT_EQ(h.expected_file_size(), expected);
}

TEST(AddrTableFormat, MakeBuildsValidHeader) {
    auto h = AddrTableHeader::make(10, 20, 30, 40, 5, 0xDEADBEEFCAFEBABE);
    EXPECT_EQ(h.magic, AddrTableHeader::MAGIC);
    EXPECT_EQ(h.version, AddrTableHeader::VERSION);
    EXPECT_EQ(h.num_l1, 10u);
    EXPECT_EQ(h.num_l2, 20u);
    EXPECT_EQ(h.num_l3, 30u);
    EXPECT_EQ(h.num_l4, 40u);
    EXPECT_EQ(h.num_zero, 5u);
    EXPECT_EQ(h.total, 105u);
    EXPECT_EQ(h.meta_sha256_head, 0xDEADBEEFCAFEBABEull);
    EXPECT_TRUE(h.is_valid());
}

TEST(AddrTableFormat, IsValidRejectsBadMagic) {
    AddrTableHeader h = AddrTableHeader::make(0, 0, 0, 0, 0, 0);
    h.magic = 0xDEADBEEFu;
    EXPECT_FALSE(h.is_valid());
}

TEST(AddrTableFormat, IsValidRejectsBadVersion) {
    AddrTableHeader h = AddrTableHeader::make(0, 0, 0, 0, 0, 0);
    h.version = 99;
    EXPECT_FALSE(h.is_valid());
}

TEST(AddrTableFormat, IsValidRejectsTotalMismatch) {
    AddrTableHeader h = AddrTableHeader::make(10, 20, 30, 40, 5, 0);
    h.total = 999;
    EXPECT_FALSE(h.is_valid());
}

TEST(AddrTableFormat, BodyOffsetsAreCorrect) {
    auto h = AddrTableHeader::make(2, 3, 5, 7, 0, 0);
    size_t expected =
        40 + (2*4 + 2*4) + (3*4 + 3*4) + (5*4 + 5*8) + (7*4 + 7*4) + 0;
    EXPECT_EQ(h.expected_file_size(), expected);
}

TEST(AddrTableFormat, BinaryRoundTrip) {
    auto orig = AddrTableHeader::make(10, 20, 30, 40, 5, 0xDEADBEEFCAFEBABEull);
    uint8_t buf[AddrTableHeader::SIZE];
    std::memcpy(buf, &orig, AddrTableHeader::SIZE);
    AddrTableHeader copy{};
    std::memcpy(&copy, buf, AddrTableHeader::SIZE);
    EXPECT_TRUE(copy.is_valid());
    EXPECT_EQ(copy.magic, AddrTableHeader::MAGIC);
    EXPECT_EQ(copy.version, 1u);
    EXPECT_EQ(copy.num_l3, 30u);
    EXPECT_EQ(copy.num_zero, 5u);
    EXPECT_EQ(copy.total, 105u);
    EXPECT_EQ(copy.meta_sha256_head, 0xDEADBEEFCAFEBABEull);
}

TEST(AddrTableFormat, BodyOffsetsWithZeroPositions) {
    auto h = AddrTableHeader::make(2, 3, 5, 7, 4, 0);
    // Header (40) + l1(2*4*2) + l2(3*4*2) + l3(5*4 + 5*8) + l4(7*4*2) + zero(4*4)
    size_t expected =
        40 + (2*4*2) + (3*4*2) + (5*4 + 5*8) + (7*4*2) + (4*4);
    EXPECT_EQ(h.expected_file_size(), expected);
}

TEST(AddrTableFormat, IsValidRejectsZeroMemory) {
    AddrTableHeader h{};
    // All fields zero-initialised by value-init braces
    EXPECT_EQ(h.magic, 0u);
    EXPECT_FALSE(h.is_valid());
}
