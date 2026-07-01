// src/gnn/tests/test_consolidated_slim_reader.cc
#include "gnn/storage/consolidated_slim.h"
#include "gnn/storage/consolidated_slim_reader.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace mdb::gnn;

namespace {
fs::path tmp_file(const std::string& tag) {
    return fs::temp_directory_path() /
           ("cons_reader_" + tag + "_" + std::to_string(::getpid()) + ".bin");
}
}  // namespace

// --- validate_consolidated_header ---

TEST(ConsolidatedSlimReader, ValidateAcceptsMatching) {
    auto h = ConsolidatedSlimHeader::make(100, 128, /*dtype=*/1,
                                          /*perm_fp=*/0xAA, /*meta_sha=*/0xBB);
    EXPECT_TRUE(validate_consolidated_header(h, 128, 1, 0xAA, 0xBB));
}

TEST(ConsolidatedSlimReader, ValidateRejectsBadMagic) {
    auto h = ConsolidatedSlimHeader::make(100, 128, 1, 0xAA, 0xBB);
    h.magic = 0xDEADBEEF;
    EXPECT_FALSE(validate_consolidated_header(h, 128, 1, 0xAA, 0xBB));
}

TEST(ConsolidatedSlimReader, ValidateRejectsDimMismatch) {
    auto h = ConsolidatedSlimHeader::make(100, 128, 1, 0xAA, 0xBB);
    EXPECT_FALSE(validate_consolidated_header(h, /*dim=*/256, 1, 0xAA, 0xBB));
}

TEST(ConsolidatedSlimReader, ValidateRejectsDtypeMismatch) {
    auto h = ConsolidatedSlimHeader::make(100, 128, 1, 0xAA, 0xBB);
    EXPECT_FALSE(validate_consolidated_header(h, 128, /*dtype=*/2, 0xAA, 0xBB));
}

TEST(ConsolidatedSlimReader, ValidateRejectsPermFpMismatch) {
    auto h = ConsolidatedSlimHeader::make(100, 128, 1, /*perm_fp=*/0xAA, 0xBB);
    EXPECT_FALSE(validate_consolidated_header(h, 128, 1, /*expected_perm=*/0xCC, 0xBB));
}

TEST(ConsolidatedSlimReader, ValidatePermFpZeroDisablesCheck) {
    auto h = ConsolidatedSlimHeader::make(100, 128, 1, /*perm_fp=*/0xAA, 0xBB);
    // expected_perm_fp == 0 => the permutation check is skipped (reorder-off store).
    EXPECT_TRUE(validate_consolidated_header(h, 128, 1, /*expected_perm=*/0, 0xBB));
}

TEST(ConsolidatedSlimReader, ValidateRejectsMetaShaMismatch) {
    auto h = ConsolidatedSlimHeader::make(100, 128, 1, 0xAA, /*meta_sha=*/0xBB);
    EXPECT_FALSE(validate_consolidated_header(h, 128, 1, 0xAA, /*expected_meta=*/0xDD));
}

TEST(ConsolidatedSlimReader, ValidateMetaShaZeroDisablesCheck) {
    auto h = ConsolidatedSlimHeader::make(100, 128, 1, 0xAA, /*meta_sha=*/0xBB);
    EXPECT_TRUE(validate_consolidated_header(h, 128, 1, 0xAA, /*expected_meta=*/0));
}

// --- pread_exact ---

TEST(ConsolidatedSlimReader, PreadExactRoundTrip) {
    auto p = tmp_file("pread_rt");
    fs::remove(p);
    std::vector<uint8_t> data(4096);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i * 7 + 1);
    {
        int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        ASSERT_GE(fd, 0);
        ASSERT_EQ(::write(fd, data.data(), data.size()),
                  static_cast<ssize_t>(data.size()));
        ::close(fd);
    }
    int fd = ::open(p.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);

    // Full read.
    std::vector<uint8_t> out(data.size());
    EXPECT_TRUE(pread_exact(fd, out.data(), out.size(), 0));
    EXPECT_EQ(std::memcmp(out.data(), data.data(), data.size()), 0);

    // Sub-range at an offset.
    std::vector<uint8_t> mid(1000);
    EXPECT_TRUE(pread_exact(fd, mid.data(), mid.size(), 512));
    EXPECT_EQ(std::memcmp(mid.data(), data.data() + 512, mid.size()), 0);

    ::close(fd);
    fs::remove(p);
}

TEST(ConsolidatedSlimReader, PreadExactPastEofFails) {
    auto p = tmp_file("pread_eof");
    fs::remove(p);
    {
        int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        ASSERT_GE(fd, 0);
        uint8_t buf[64] = {0};
        ASSERT_EQ(::write(fd, buf, sizeof(buf)), 64);
        ::close(fd);
    }
    int fd = ::open(p.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    std::vector<uint8_t> out(128);  // ask for more than the file holds
    EXPECT_FALSE(pread_exact(fd, out.data(), out.size(), 0));
    ::close(fd);
    fs::remove(p);
}

TEST(ConsolidatedSlimReader, PreadExactBadFdFails) {
    std::vector<uint8_t> out(16);
    EXPECT_FALSE(pread_exact(-1, out.data(), out.size(), 0));
}
