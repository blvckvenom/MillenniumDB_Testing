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
