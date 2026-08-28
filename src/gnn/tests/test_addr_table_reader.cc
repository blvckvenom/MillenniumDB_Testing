// src/gnn/tests/test_addr_table_reader.cc
#include "gnn/storage/addr_table.h"
#include "gnn/storage/addr_table_reader.h"
#include "gnn/storage/addr_table_writer.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace mdb::gnn;

namespace {

fs::path tmp_addrtab(const std::string& tag) {
    return fs::temp_directory_path() /
           ("addrtab_reader_test_" + tag + "_" +
            std::to_string(::getpid()) + ".addr");
}

void write_synthetic_addrtab(const fs::path& path,
                              uint64_t meta_sha = 0xCAFEBABE)
{
    AddrTableBuffers buf;
    buf.header = AddrTableHeader::make(2, 3, 1, 4, 0, meta_sha);
    buf.l1_positions = {0, 1};
    buf.l1_indices   = {100, 101};
    buf.l2_positions = {2, 3, 4};
    buf.l2_indices   = {200, 201, 202};
    buf.l3_positions = {5};
    buf.l3_row_idxs  = {0xDEADBEEFull};
    buf.l4_positions = {6, 7, 8, 9};
    buf.l4_indices   = {300, 301, 302, 303};
    AddrTableWriter::write_atomic(path, buf);
}

} // namespace

TEST(AddrTableReaderOpen, ParsesValidFile) {
    auto p = tmp_addrtab("valid");
    fs::remove(p);
    write_synthetic_addrtab(p, 0xCAFEBABE);

    auto res = AddrTableReader::open(p, 0xCAFEBABE);
    EXPECT_EQ(res.header.num_l1, 2u);
    EXPECT_EQ(res.header.num_l2, 3u);
    EXPECT_EQ(res.header.num_l3, 1u);
    EXPECT_EQ(res.header.num_l4, 4u);
    EXPECT_EQ(res.header.total, 10u);

    ASSERT_EQ(res.l1_positions.size(), 2u);
    EXPECT_EQ(res.l1_positions[0], 0u);
    EXPECT_EQ(res.l1_indices[1], 101u);
    EXPECT_EQ(res.l3_row_idxs[0], 0xDEADBEEFull);
    EXPECT_EQ(res.l4_indices[3], 303u);

    fs::remove(p);
}

TEST(AddrTableReaderOpen, RejectsBadMagic) {
    auto p = tmp_addrtab("bad_magic");
    fs::remove(p);
    write_synthetic_addrtab(p, 0);

    std::fstream f(p.string(), std::ios::in | std::ios::out | std::ios::binary);
    uint32_t bogus = 0xDEADBEEFu;
    f.seekp(0);
    f.write(reinterpret_cast<const char*>(&bogus), sizeof(bogus));
    f.close();

    EXPECT_THROW(AddrTableReader::open(p, 0), std::runtime_error);
    fs::remove(p);
}

TEST(AddrTableReaderOpen, RejectsBadVersion) {
    auto p = tmp_addrtab("bad_ver");
    fs::remove(p);
    write_synthetic_addrtab(p, 0);

    std::fstream f(p.string(), std::ios::in | std::ios::out | std::ios::binary);
    uint32_t bogus_ver = 99;
    f.seekp(4);  // offset of version field
    f.write(reinterpret_cast<const char*>(&bogus_ver), sizeof(bogus_ver));
    f.close();

    EXPECT_THROW(AddrTableReader::open(p, 0), std::runtime_error);
    fs::remove(p);
}

TEST(AddrTableReaderOpen, RejectsMetaShaMismatch) {
    auto p = tmp_addrtab("sha_mismatch");
    fs::remove(p);
    write_synthetic_addrtab(p, 0xAAAA);
    EXPECT_THROW(AddrTableReader::open(p, 0xBBBB), AddrTableStaleException);
    fs::remove(p);
}

TEST(AddrTableReaderOpen, RejectsTruncatedFile) {
    auto p = tmp_addrtab("truncated");
    fs::remove(p);
    write_synthetic_addrtab(p, 0);

    auto sz = fs::file_size(p);
    fs::resize_file(p, sz / 2);
    EXPECT_THROW(AddrTableReader::open(p, 0), std::runtime_error);
    fs::remove(p);
}

TEST(AddrTableReaderOpen, V1FileSurfacesZeroSlimFields) {
    // A v1 (40-byte header) addr_table written by make() must read back with
    // slim_offset == slim_length == 0 (the reader zero-inits the v2 extension).
    auto p = tmp_addrtab("v1_slim_zero");
    fs::remove(p);
    write_synthetic_addrtab(p, 0xCAFEBABE);

    auto res = AddrTableReader::open(p, 0xCAFEBABE);
    EXPECT_EQ(res.header.version, 1u);
    EXPECT_EQ(res.header.header_bytes(), 40u);
    EXPECT_EQ(res.header.slim_offset, 0ull);
    EXPECT_EQ(res.header.slim_length, 0ull);
    // arrays still parse correctly off the 40-byte header base
    ASSERT_EQ(res.l4_indices.size(), 4u);
    EXPECT_EQ(res.l4_indices[3], 303u);
    fs::remove(p);
}

TEST(AddrTableReaderOpen, V2RoundTripSurfacesSlimFields) {
    // A v2 (56-byte header) addr_table written by make_v2() must read back its
    // slim_offset/slim_length AND parse the 9 arrays off the bumped 56-byte base.
    auto p = tmp_addrtab("v2_roundtrip");
    fs::remove(p);

    AddrTableBuffers buf;
    buf.header = AddrTableHeader::make_v2(2, 0, 1, 3, 0, 0xCAFEBABE,
                                          /*slim_off=*/0x40000, /*slim_len=*/0x1800);
    buf.l1_positions = {0, 1};
    buf.l1_indices   = {100, 101};
    buf.l3_positions = {2};
    buf.l3_row_idxs  = {0xFEEDFACEull};
    buf.l4_positions = {3, 4, 5};
    buf.l4_indices   = {300, 301, 302};
    AddrTableWriter::write_atomic(p, buf);

    // File must be exactly 56-byte header + arrays (v2 on-disk size).
    EXPECT_EQ(fs::file_size(p), buf.header.expected_file_size());

    auto res = AddrTableReader::open(p, 0xCAFEBABE);
    EXPECT_EQ(res.header.version, 2u);
    EXPECT_EQ(res.header.header_bytes(), 56u);
    EXPECT_EQ(res.header.slim_offset, 0x40000ull);
    EXPECT_EQ(res.header.slim_length, 0x1800ull);
    EXPECT_EQ(res.header.total, 6u);
    ASSERT_EQ(res.l1_indices.size(), 2u);
    EXPECT_EQ(res.l1_indices[1], 101u);
    ASSERT_EQ(res.l3_row_idxs.size(), 1u);
    EXPECT_EQ(res.l3_row_idxs[0], 0xFEEDFACEull);
    ASSERT_EQ(res.l4_indices.size(), 3u);
    EXPECT_EQ(res.l4_indices[2], 302u);
    fs::remove(p);
}

TEST(AddrTableReaderOpen, ParsesOddL3AlignmentSafely) {
    // num_l3 = 1 (odd) puts l3_row_idxs at a 4-byte-aligned but not
    // 8-byte-aligned offset within the read buffer. The aligned-storage
    // path in open() should copy into Result::l3_row_idxs_storage so
    // the uint64 load is well-defined.
    auto p = tmp_addrtab("odd_l3");
    fs::remove(p);

    AddrTableBuffers buf;
    buf.header = AddrTableHeader::make(0, 0, 1, 0, 0, 0xDEAD);
    buf.l3_positions = {7};
    buf.l3_row_idxs  = {0x0123456789ABCDEFull};
    AddrTableWriter::write_atomic(p, buf);

    auto res = AddrTableReader::open(p, 0xDEAD);
    ASSERT_EQ(res.l3_row_idxs.size(), 1u);
    EXPECT_EQ(res.l3_row_idxs[0], 0x0123456789ABCDEFull);

    fs::remove(p);
}
