// Unit tests for TopologySnapshotWriter.
//
// TopologySnapshotWriter produces mmap-backed CSR sidecar files
// (topology_fwd.csr / topology_rev.csr) that enable O(1) neighbor slices
// for GNN sampling, replacing O(log N) B+Tree lookups.
//
// Scope:
//   - Header byte values in finalized file (magic / version / flags / counts).
//   - ROW_PTR prefix-sum contract (ROW_PTR[0] == 0, monotone
//     non-decreasing, ROW_PTR[N] == total edge count).
//   - COL_IDX ordering matches append order.
//   - Source-.leaf SHA-256 correctness (against a hand-computed digest).
//   - Atomic commit: `.tmp` absent after finalize, final present.
//   - `has_edge_ids` flag + EDGE_IDS section round-trip.
//
// What is NOT tested here (covered by the reader and integration test suites):
//   - mmap + slice access via the reader.
//   - Staleness rejection on hash / size mismatch.
//   - Parallel reader safety.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <openssl/evp.h>
#include <gtest/gtest.h>

#include "graph_models/gql/projection/topology_snapshot.h"
#include "graph_models/gql/projection/topology_snapshot_writer.h"
#include "graph_models/object_id.h"

using GQL::Projection::kTopologySnapshotHeaderSize;
using GQL::Projection::TopologySnapshotFlags::kHasEdgeIds;
using GQL::Projection::TopologySnapshotHeader;
using GQL::Projection::TopologySnapshotWriter;
using GQL::Projection::parse_topology_snapshot_header;

namespace {

// ---------------------------------------------------------------------------
// Test fixture helpers
// ---------------------------------------------------------------------------

// Creates a hermetic, unique temporary directory. Cleaned in TearDown.
class TopologySnapshotWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto base = std::filesystem::temp_directory_path();
        std::random_device rd;
        // Re-seed between tests to avoid collisions if two test cases run in
        // the same process and one rerolls an already-created directory.
        std::mt19937_64 rng(rd());
        for (int attempt = 0; attempt < 64; ++attempt) {
            dir_ = base / ("mdb_topo_writer_test_" + std::to_string(rng()));
            if (!std::filesystem::exists(dir_)) {
                std::filesystem::create_directories(dir_);
                return;
            }
        }
        FAIL() << "Could not allocate unique temp dir under " << base;
    }

    void TearDown() override {
        if (!dir_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(dir_, ec);  // best-effort
        }
    }

    // Write a deterministic fake `from_to_edge.leaf` (for FORWARD) so the
    // SHA-256 pass in finalize() has real content to hash.
    void write_fake_source_leaf(TopologySnapshotWriter::Direction d,
                                const std::string&                content) {
        const char* name = (d == TopologySnapshotWriter::Direction::FORWARD)
                         ? "from_to_edge.leaf"
                         : "to_from_edge.leaf";
        std::ofstream f(dir_ / name, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(f.good());
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
        f.close();
    }

    // Read the on-disk 64-byte header into a parsed struct. Fails the test
    // if the file is shorter than 64 bytes or unreadable.
    TopologySnapshotHeader read_header(const std::filesystem::path& p) {
        std::ifstream f(p, std::ios::binary);
        EXPECT_TRUE(f.good());
        uint8_t buf[kTopologySnapshotHeaderSize];
        f.read(reinterpret_cast<char*>(buf), kTopologySnapshotHeaderSize);
        EXPECT_EQ(f.gcount(),
                  static_cast<std::streamsize>(kTopologySnapshotHeaderSize));
        return parse_topology_snapshot_header(buf);
    }

    // Read the full file body (after header). Used to decode ROW_PTR / COL_IDX.
    std::vector<uint8_t> read_all(const std::filesystem::path& p) {
        std::ifstream f(p, std::ios::binary);
        EXPECT_TRUE(f.good());
        f.seekg(0, std::ios::end);
        auto size = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<uint8_t> v(static_cast<std::size_t>(size));
        f.read(reinterpret_cast<char*>(v.data()), size);
        return v;
    }

    // Hand-compute SHA-256 of an arbitrary buffer via EVP — used to confirm
    // the hash stored in the header matches an externally-computed digest.
    std::array<uint8_t, 32> sha256(const std::string& content) {
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
        EVP_DigestUpdate(ctx, content.data(), content.size());
        std::array<uint8_t, 32> digest{};
        unsigned int len = 0;
        EVP_DigestFinal_ex(ctx, digest.data(), &len);
        EVP_MD_CTX_free(ctx);
        return digest;
    }

    std::filesystem::path dir_;
};

// Shorthand.
ObjectId oid(uint64_t v) { return ObjectId(v); }

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — empty projection: N=0 → file is exactly 64 bytes of header
// (plus the 8-byte ROW_PTR[0]=0). `num_edges=0`, `has_edge_ids=0`.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, EmptyProjectionWritesValidHeaderOnly) {
    // No source .leaf at all — N=0, M=0 path accepts this.
    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/0,
        std::vector<uint64_t>{},  // zero-length degrees
        /*include_edge_ids=*/false);
    writer.finalize();

    auto final_path = dir_ / "topology_fwd.csr";
    ASSERT_TRUE(std::filesystem::exists(final_path));

    // Expected file size: 64 (header) + 8 (row_ptr[0]=0) + 0 (col_idx) = 72.
    EXPECT_EQ(std::filesystem::file_size(final_path), 72u);
    EXPECT_EQ(writer.bytes_written(), 72u);

    TopologySnapshotHeader h = read_header(final_path);
    EXPECT_EQ(h.num_nodes, 0u);
    EXPECT_EQ(h.num_edges, 0u);
    EXPECT_EQ(h.flags & kHasEdgeIds, 0u);

    // ROW_PTR[0] == 0 immediately after the header.
    auto bytes = read_all(final_path);
    uint64_t row_ptr_0 = 0;
    std::memcpy(&row_ptr_0, bytes.data() + kTopologySnapshotHeaderSize, 8);
    EXPECT_EQ(row_ptr_0, 0u);
}

// ---------------------------------------------------------------------------
// Test 2 — N=2 with one edge (0 → 1). Degrees = [1, 0].
// Expected ROW_PTR = [0, 1, 1]; COL_IDX = [1].
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, SingleEdgeWritesCorrectRowPtr) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "one-edge-source-leaf");

    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/2,
        /*degrees=*/{1, 0},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.finalize();

    auto bytes = read_all(dir_ / "topology_fwd.csr");
    // Layout: 64 header | 24 row_ptr (N+1=3 × 8) | 8 col_idx (M=1 × 8)
    ASSERT_EQ(bytes.size(), 64u + 24u + 8u);

    uint64_t row_ptr[3];
    std::memcpy(row_ptr, bytes.data() + 64, 24);
    EXPECT_EQ(row_ptr[0], 0u);
    EXPECT_EQ(row_ptr[1], 1u);
    EXPECT_EQ(row_ptr[2], 1u);

    uint64_t col_idx_0 = 0;
    std::memcpy(&col_idx_0, bytes.data() + 64 + 24, 8);
    EXPECT_EQ(col_idx_0, 1u);
}

// ---------------------------------------------------------------------------
// Test 3 — Multi-edge fan-out; verifies both ROW_PTR prefix sum and that
// COL_IDX reflects append order. N=4, degrees=[2,1,1,1], edges in
// source-monotonic order.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, MultiEdgeMonotonicSourceOrder) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "multi-edge-fake-leaf-contents");

    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/4,
        /*degrees=*/{2, 1, 1, 1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(0), oid(2), ObjectId());
    writer.append_edge(oid(1), oid(3), ObjectId());
    writer.append_edge(oid(2), oid(0), ObjectId());
    writer.append_edge(oid(3), oid(1), ObjectId());
    writer.finalize();

    auto bytes = read_all(dir_ / "topology_fwd.csr");
    // Layout: 64 + 8*(4+1)=40 + 8*5=40  → 144 bytes total.
    ASSERT_EQ(bytes.size(), 144u);

    uint64_t row_ptr[5];
    std::memcpy(row_ptr, bytes.data() + 64, 40);
    EXPECT_EQ(row_ptr[0], 0u);
    EXPECT_EQ(row_ptr[1], 2u);
    EXPECT_EQ(row_ptr[2], 3u);
    EXPECT_EQ(row_ptr[3], 4u);
    EXPECT_EQ(row_ptr[4], 5u);

    uint64_t col_idx[5];
    std::memcpy(col_idx, bytes.data() + 64 + 40, 40);
    EXPECT_EQ(col_idx[0], 1u);
    EXPECT_EQ(col_idx[1], 2u);
    EXPECT_EQ(col_idx[2], 3u);
    EXPECT_EQ(col_idx[3], 0u);
    EXPECT_EQ(col_idx[4], 1u);
}

// ---------------------------------------------------------------------------
// Test 4 — SHA-256 stored in the header matches a hand-computed digest of
// the exact bytes written to the fake `from_to_edge.leaf`.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, SourceSha256MatchesHandComputed) {
    const std::string leaf = "deterministic test payload for SHA-256 check";
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, leaf);

    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/1,
        /*degrees=*/{1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(0), ObjectId());
    writer.finalize();

    auto expected = sha256(leaf);
    TopologySnapshotHeader h = read_header(dir_ / "topology_fwd.csr");
    EXPECT_EQ(std::memcmp(h.source_sha256, expected.data(), 32), 0)
        << "SHA-256 of from_to_edge.leaf must match the digest stored in the header";
}

// ---------------------------------------------------------------------------
// Test 5 — atomic rename: the `.tmp` exists during the write phase and is
// gone after finalize, replaced by the final name.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, AtomicRenameOnFinalize) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "x");

    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/1,
        /*degrees=*/{0},
        /*include_edge_ids=*/false);

    const auto tmp_path   = dir_ / "topology_fwd.csr.tmp";
    const auto final_path = dir_ / "topology_fwd.csr";

    // Mid-write: .tmp present (it's created by the ctor), final absent.
    EXPECT_TRUE(std::filesystem::exists(tmp_path));
    EXPECT_FALSE(std::filesystem::exists(final_path));

    writer.finalize();

    // Post-finalize: .tmp gone, final present.
    EXPECT_FALSE(std::filesystem::exists(tmp_path));
    EXPECT_TRUE(std::filesystem::exists(final_path));
}

// ---------------------------------------------------------------------------
// Test 6 — after finalize, there is no `.csr.tmp` left behind.
// Distinct from Test 5 because it runs `directory_iterator` to guarantee no
// lingering temp entries of any kind (not just the one we expected).
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, TmpFileCleanedAfterFinalize) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "y");

    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/0,
        /*degrees=*/{},
        /*include_edge_ids=*/false);
    writer.finalize();

    for (const auto& entry : std::filesystem::directory_iterator(dir_)) {
        const auto name = entry.path().filename().string();
        // No `.tmp` files of any kind should remain — the writer's atomic
        // rename must have either cleaned them up or converted them to the
        // final `.csr`.
        EXPECT_FALSE(name.size() >= 4 && name.compare(name.size() - 4, 4, ".tmp") == 0)
            << "no `.tmp` file should remain: " << entry.path().string();
        // The directory should contain at most our final .csr and the
        // fixture's source .leaf — nothing else.
        EXPECT_TRUE(name == "topology_fwd.csr" || name == "from_to_edge.leaf")
            << "unexpected residue in dir: " << name;
    }
}

// ---------------------------------------------------------------------------
// Test 7 — ROW_PTR is monotonic and terminates at M. This duplicates the
// arithmetic of Test 3 but exercises a wider graph with random gaps
// (zero-degree nodes interleaved with high-degree ones).
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, InvariantRowPtrMonotonicIncreasing) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "invariant-check-payload");

    const std::vector<uint64_t> degrees = {3, 0, 2, 0, 1, 0, 4};
    const uint64_t N = degrees.size();
    uint64_t expected_M = 0;
    for (auto d : degrees) expected_M += d;

    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        N,
        degrees,
        /*include_edge_ids=*/false);

    // Emit the required edges in source-monotonic order.
    uint64_t counter = 0;
    for (uint64_t s = 0; s < N; ++s) {
        for (uint64_t k = 0; k < degrees[s]; ++k) {
            writer.append_edge(oid(s), oid(counter++ % N), ObjectId());
        }
    }
    writer.finalize();

    auto bytes = read_all(dir_ / "topology_fwd.csr");
    std::vector<uint64_t> row_ptr(N + 1);
    std::memcpy(row_ptr.data(), bytes.data() + 64, 8 * (N + 1));

    EXPECT_EQ(row_ptr.front(), 0u);
    EXPECT_EQ(row_ptr.back(), expected_M);
    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_LE(row_ptr[i], row_ptr[i + 1])
            << "row_ptr must be monotonic nondecreasing at i=" << i;
    }

    TopologySnapshotHeader h = read_header(dir_ / "topology_fwd.csr");
    EXPECT_EQ(h.num_nodes, N);
    EXPECT_EQ(h.num_edges, expected_M);
}

// ---------------------------------------------------------------------------
// Test 8 — include_edge_ids=true round-trip: flag bit set, file size grows
// by `M * 8`, and EDGE_IDS section holds the exact ids in append order.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, HasEdgeIdsFlagRoundTrip) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "edge-id-payload");

    const std::vector<uint64_t> degrees = {2, 1};
    const uint64_t N = degrees.size();
    const uint64_t M = 3;

    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        N,
        degrees,
        /*include_edge_ids=*/true);

    writer.append_edge(oid(0), oid(1), oid(100));
    writer.append_edge(oid(0), oid(2), oid(101));
    writer.append_edge(oid(1), oid(0), oid(102));
    writer.finalize();

    const auto path = dir_ / "topology_fwd.csr";
    // Expected size: 64 + 8*(N+1) + 8*M + 8*M = 64 + 24 + 24 + 24 = 136.
    EXPECT_EQ(std::filesystem::file_size(path),
              64u + 8u * (N + 1) + 8u * M * 2u);
    EXPECT_EQ(writer.bytes_written(),
              64u + 8u * (N + 1) + 8u * M * 2u);

    TopologySnapshotHeader h = read_header(path);
    EXPECT_NE(h.flags & kHasEdgeIds, 0u) << "kHasEdgeIds flag bit must be set";
    EXPECT_EQ(h.num_nodes, N);
    EXPECT_EQ(h.num_edges, M);

    auto bytes = read_all(path);
    // EDGE_IDS section starts at 64 + 8*(N+1) + 8*M = 112.
    const std::size_t edge_ids_off = 64 + 8 * (N + 1) + 8 * M;
    uint64_t ids[3];
    std::memcpy(ids, bytes.data() + edge_ids_off, 24);
    EXPECT_EQ(ids[0], 100u);
    EXPECT_EQ(ids[1], 101u);
    EXPECT_EQ(ids[2], 102u);
}

// ---------------------------------------------------------------------------
// Test 9 — a second writer targeting the same output path while the first
// is still holding `.tmp` is rejected. Verifies the O_EXCL discipline that
// stands in for a directory-level lock: two concurrent builds of the same
// sidecar must not interleave writes into one `.tmp` file.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotWriterTest, ParallelWritersRejectedOnSameFile) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "z");

    TopologySnapshotWriter writer_a(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/0,
        /*degrees=*/{},
        /*include_edge_ids=*/false);
    // writer_a has already created `.tmp` in its ctor. A second writer for
    // the same final path must fail — the ctor itself opens with O_EXCL.
    EXPECT_THROW({
        TopologySnapshotWriter writer_b(
            dir_,
            TopologySnapshotWriter::Direction::FORWARD,
            /*num_nodes=*/0,
            /*degrees=*/{},
            /*include_edge_ids=*/false);
    }, std::runtime_error);

    // Finish the first writer cleanly so the fixture can tear down without
    // dangling fd warnings.
    writer_a.finalize();
}

// ---------------------------------------------------------------------------
// Always-on invariant checks (promoted from assert to runtime throws so
// Release builds catch B+Tree scan bugs before they corrupt the CSR body
// in a way that would only surface as wrong GNN sampling output).
// ---------------------------------------------------------------------------

TEST_F(TopologySnapshotWriterTest, AppendEdgeRejectsOutOfRangeSrc) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "x");
    // N=2 nodes, src=5 is out of range.
    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/2,
        /*degrees=*/{1, 0},
        /*include_edge_ids=*/false);
    EXPECT_THROW(
        writer.append_edge(ObjectId{5}, ObjectId{0}, ObjectId{0}),
        std::runtime_error);
}

TEST_F(TopologySnapshotWriterTest, AppendEdgeRejectsDecreasingSrc) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "x");
    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/3,
        /*degrees=*/{1, 1, 0},
        /*include_edge_ids=*/false);
    // First edge at src=1 advances last_src_idx_; a following edge at src=0
    // violates monotonicity.
    writer.append_edge(ObjectId{1}, ObjectId{2}, ObjectId{0});
    EXPECT_THROW(
        writer.append_edge(ObjectId{0}, ObjectId{1}, ObjectId{0}),
        std::runtime_error);
}

TEST_F(TopologySnapshotWriterTest, AppendEdgeRejectsDegreeOverflow) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "x");
    // N=2, src=0 has declared degree 1. A second append_edge with src=0
    // exceeds the declared degree.
    TopologySnapshotWriter writer(
        dir_,
        TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/2,
        /*degrees=*/{1, 0},
        /*include_edge_ids=*/false);
    writer.append_edge(ObjectId{0}, ObjectId{1}, ObjectId{0});
    EXPECT_THROW(
        writer.append_edge(ObjectId{0}, ObjectId{1}, ObjectId{0}),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// Symmetric / explicit-basename writer ctor
// ---------------------------------------------------------------------------

// Hand-compute SHA-256 of two concatenated payloads, matching the writer's
// fixed-order chaining (from_to_edge.leaf first, to_from_edge.leaf second).
static std::array<uint8_t, 32> sha256_concat(const std::string& a,
                                             const std::string& b) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, a.data(), a.size());
    EVP_DigestUpdate(ctx, b.data(), b.size());
    std::array<uint8_t, 32> d{};
    unsigned int len = 0;
    EVP_DigestFinal_ex(ctx, d.data(), &len);
    EVP_MD_CTX_free(ctx);
    return d;
}

TEST_F(TopologySnapshotWriterTest, SymmetricWriterEmitsSymMagicAndExplicitBasename) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "fwd-leaf");
    write_fake_source_leaf(TopologySnapshotWriter::Direction::REVERSE, "rev-leaf");

    TopologySnapshotWriter writer(
        dir_,
        std::string("topology_sym.csr"),
        std::vector<std::filesystem::path>{dir_ / "from_to_edge.leaf",
                                           dir_ / "to_from_edge.leaf"},
        /*num_nodes=*/2,
        /*degrees=*/{1, 1},
        /*include_edge_ids=*/false,
        /*symmetric_format=*/true);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(1), oid(0), ObjectId());
    writer.finalize();

    const auto path = dir_ / "topology_sym.csr";
    ASSERT_TRUE(std::filesystem::exists(path));
    EXPECT_EQ(writer.output_path(), path);

    std::ifstream f(path, std::ios::binary);
    uint8_t buf[GQL::Projection::kTopologySnapshotHeaderSize];
    f.read(reinterpret_cast<char*>(buf),
           GQL::Projection::kTopologySnapshotHeaderSize);
    auto h = GQL::Projection::parse_topology_snapshot_sym_header(buf);
    EXPECT_EQ(h.num_nodes, 2u);
    EXPECT_EQ(h.num_edges, 2u);
    EXPECT_EQ(h.flags & kHasEdgeIds, 0u);
}

TEST_F(TopologySnapshotWriterTest, SymmetricWriterCombinedHashChainsBothLeavesInOrder) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "AAA-fwd");
    write_fake_source_leaf(TopologySnapshotWriter::Direction::REVERSE, "BBB-rev");

    TopologySnapshotWriter writer(
        dir_,
        std::string("topology_sym.csr"),
        std::vector<std::filesystem::path>{dir_ / "from_to_edge.leaf",
                                           dir_ / "to_from_edge.leaf"},
        /*num_nodes=*/1,
        /*degrees=*/{1},
        /*include_edge_ids=*/false,
        /*symmetric_format=*/true);
    writer.append_edge(oid(0), oid(0), ObjectId());
    writer.finalize();

    std::ifstream f(dir_ / "topology_sym.csr", std::ios::binary);
    uint8_t buf[GQL::Projection::kTopologySnapshotHeaderSize];
    f.read(reinterpret_cast<char*>(buf),
           GQL::Projection::kTopologySnapshotHeaderSize);
    auto h = GQL::Projection::parse_topology_snapshot_sym_header(buf);

    auto expected = sha256_concat("AAA-fwd", "BBB-rev");
    EXPECT_EQ(std::memcmp(h.source_sha256, expected.data(), 32), 0)
        << "combined digest must chain from_to_edge.leaf then to_from_edge.leaf";
    auto wrong = sha256_concat("BBB-rev", "AAA-fwd");
    EXPECT_NE(std::memcmp(h.source_sha256, wrong.data(), 32), 0);
}

TEST_F(TopologySnapshotWriterTest, SymmetricWriterBodyIsPlainCsrNoEdgeIds) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "f");
    write_fake_source_leaf(TopologySnapshotWriter::Direction::REVERSE, "r");
    // N=3, undirected row of node 0 = {1,2}; node1={0}; node2={0}.
    TopologySnapshotWriter writer(
        dir_,
        std::string("topology_sym.csr"),
        std::vector<std::filesystem::path>{dir_ / "from_to_edge.leaf",
                                           dir_ / "to_from_edge.leaf"},
        /*num_nodes=*/3,
        /*degrees=*/{2, 1, 1},
        /*include_edge_ids=*/false,
        /*symmetric_format=*/true);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(0), oid(2), ObjectId());
    writer.append_edge(oid(1), oid(0), ObjectId());
    writer.append_edge(oid(2), oid(0), ObjectId());
    writer.finalize();

    const auto path = dir_ / "topology_sym.csr";
    // No EDGE_IDS section; symmetric body is NARROW uint32 col_idx (ROW_PTR stays
    // uint64): 64 + 8*(3+1) + 4*4 = 64+32+16 = 112.
    EXPECT_EQ(std::filesystem::file_size(path), 112u);

    auto bytes = read_all(path);
    uint64_t row_ptr[4];
    std::memcpy(row_ptr, bytes.data() + 64, 32);
    EXPECT_EQ(row_ptr[0], 0u);
    EXPECT_EQ(row_ptr[1], 2u);
    EXPECT_EQ(row_ptr[2], 3u);
    EXPECT_EQ(row_ptr[3], 4u);
    uint32_t col[4];
    std::memcpy(col, bytes.data() + 64 + 32, 16);
    EXPECT_EQ(col[0], 1u);
    EXPECT_EQ(col[1], 2u);
    EXPECT_EQ(col[2], 0u);
    EXPECT_EQ(col[3], 0u);
}
