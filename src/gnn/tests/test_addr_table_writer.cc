// src/gnn/tests/test_addr_table_writer.cc
#include "gnn/storage/addr_table_writer.h"
#include "gnn/storage/addr_table.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace mdb::gnn;

namespace {

/// Mock cache supporting find_index lookups by OID payload (uint64 id).
struct MockCache {
    std::unordered_map<uint64_t, uint32_t> oid_to_idx;
    std::optional<uint32_t> find_index(ObjectId oid) const {
        auto it = oid_to_idx.find(oid.id);
        if (it == oid_to_idx.end()) return std::nullopt;
        return it->second;
    }
};

ObjectId mkoid(uint64_t v) { return ObjectId(v); }

fs::path tmp_path(const std::string& tag) {
    return fs::temp_directory_path() /
           ("addrtab_writer_test_" + tag + "_" +
            std::to_string(::getpid()) + ".addr");
}

} // namespace

TEST(AddrTableWriterBuild, AllNodesGoToL1) {
    std::vector<ObjectId> unique_nodes = {mkoid(10), mkoid(20), mkoid(30)};
    MockCache l1; l1.oid_to_idx = {{10, 0}, {20, 1}, {30, 2}};
    MockCache l2;
    std::unordered_map<uint64_t, uint32_t> slim;
    auto rmap_find = [](ObjectId) -> std::optional<uint64_t> { return std::nullopt; };

    AddrTableBuffers buf;
    AddrTableWriter::build(unique_nodes, &l1, &l2, slim, rmap_find,
                           /*meta_sha_head=*/0xCAFEBABE, buf);

    EXPECT_EQ(buf.l1_positions.size(), 3u);
    EXPECT_EQ(buf.l1_indices.size(), 3u);
    EXPECT_EQ(buf.l2_positions.size(), 0u);
    EXPECT_EQ(buf.l3_positions.size(), 0u);
    EXPECT_EQ(buf.l4_positions.size(), 0u);
    EXPECT_EQ(buf.zero_positions.size(), 0u);
    EXPECT_EQ(buf.header.num_l1, 3u);
    EXPECT_EQ(buf.header.total, 3u);
    EXPECT_EQ(buf.header.meta_sha256_head, 0xCAFEBABEull);
}

TEST(AddrTableWriterBuild, NodesPartitionInTierOrder) {
    // Node 0 -> L1, 1 -> L2, 2 -> L4, 3 -> L3, 4 -> zero
    std::vector<ObjectId> unique_nodes = {
        mkoid(100), mkoid(200), mkoid(300), mkoid(400), mkoid(500)
    };
    MockCache l1; l1.oid_to_idx = {{100, 7}};
    MockCache l2; l2.oid_to_idx = {{200, 8}};
    std::unordered_map<uint64_t, uint32_t> slim{{300, 11}};
    auto rmap_find = [](ObjectId oid) -> std::optional<uint64_t> {
        if (oid.id == 400) return 42ull;
        return std::nullopt;
    };

    AddrTableBuffers buf;
    AddrTableWriter::build(unique_nodes, &l1, &l2, slim, rmap_find, 0, buf);

    ASSERT_EQ(buf.l1_positions.size(), 1u);
    EXPECT_EQ(buf.l1_positions[0], 0u);
    EXPECT_EQ(buf.l1_indices[0], 7u);

    ASSERT_EQ(buf.l2_positions.size(), 1u);
    EXPECT_EQ(buf.l2_positions[0], 1u);
    EXPECT_EQ(buf.l2_indices[0], 8u);

    ASSERT_EQ(buf.l4_positions.size(), 1u);
    EXPECT_EQ(buf.l4_positions[0], 2u);
    EXPECT_EQ(buf.l4_indices[0], 11u);

    ASSERT_EQ(buf.l3_positions.size(), 1u);
    EXPECT_EQ(buf.l3_positions[0], 3u);
    EXPECT_EQ(buf.l3_row_idxs[0], 42ull);

    ASSERT_EQ(buf.zero_positions.size(), 1u);
    EXPECT_EQ(buf.zero_positions[0], 4u);

    EXPECT_EQ(buf.header.total, 5u);
}

TEST(AddrTableWriterBuild, EmptyInputProducesEmptyBuffers) {
    std::vector<ObjectId> empty;
    MockCache l1, l2;
    std::unordered_map<uint64_t, uint32_t> slim;
    auto noop = [](ObjectId) -> std::optional<uint64_t> { return std::nullopt; };

    AddrTableBuffers buf;
    AddrTableWriter::build(empty, &l1, &l2, slim, noop, 0, buf);

    EXPECT_EQ(buf.header.total, 0u);
    EXPECT_EQ(buf.total_bytes(), AddrTableHeader::SIZE);
}

TEST(AddrTableWriterWrite, RoundTripsViaTempFile) {
    auto tmp = tmp_path("roundtrip");
    fs::remove(tmp);

    std::vector<ObjectId> unique_nodes = {mkoid(1), mkoid(2)};
    MockCache l1; l1.oid_to_idx = {{1, 100}, {2, 101}};
    MockCache l2;
    std::unordered_map<uint64_t, uint32_t> slim;
    auto noop = [](ObjectId) -> std::optional<uint64_t> { return std::nullopt; };

    AddrTableBuffers buf;
    AddrTableWriter::build(unique_nodes, &l1, &l2, slim, noop, 0xABCD, buf);
    AddrTableWriter::write_atomic(tmp, buf);

    ASSERT_TRUE(fs::exists(tmp));
    EXPECT_EQ(fs::file_size(tmp), buf.total_bytes());

    // Read header back
    std::ifstream f(tmp, std::ios::binary);
    AddrTableHeader hdr_back{};
    f.read(reinterpret_cast<char*>(&hdr_back), sizeof(hdr_back));
    EXPECT_EQ(hdr_back.magic, AddrTableHeader::MAGIC);
    EXPECT_EQ(hdr_back.num_l1, 2u);
    EXPECT_EQ(hdr_back.meta_sha256_head, 0xABCDull);

    fs::remove(tmp);
}

TEST(AddrTableWriterWrite, NoPartialFileOnMissingParent) {
    auto bad = fs::path("/dev/null/this/cannot/exist/addrtab.addr");
    std::vector<ObjectId> unique_nodes = {mkoid(1)};
    MockCache l1, l2;
    std::unordered_map<uint64_t, uint32_t> slim;
    auto noop = [](ObjectId) -> std::optional<uint64_t> { return std::nullopt; };

    AddrTableBuffers buf;
    AddrTableWriter::build(unique_nodes, &l1, &l2, slim, noop, 0, buf);
    EXPECT_THROW(AddrTableWriter::write_atomic(bad, buf), std::runtime_error);
}
