// Unit tests for TopologySnapshotReader — the mmap-backed reader for the
// topology CSR sidecar files (topology_fwd.csr / topology_rev.csr). Each
// sidecar stores a 64-byte header, a ROW_PTR prefix-sum array, a COL_IDX
// neighbor array, and an optional parallel EDGE_IDS array, allowing O(1)
// neighbor slicing without a B+Tree lookup.
//
// Scope:
//   - Absent / truncated / magic-invalid / version-invalid → has_data()=false.
//   - File-size vs declared N/M mismatch → has_data()=false.
//   - ROW_PTR invariant violations → has_data()=false.
//   - Valid file: neighbors()/edge_ids() return byte-exact slices vs direct decode.
//   - Empty neighbor list (degree-0 node) → empty span.
//   - has_edge_ids flag clear → edge_ids(v) empty span.
//   - has_edge_ids flag set → edge_ids(v) returns expected values.
//   - Out-of-range node_idx → throws (always-on bounds enforcement).
//   - Two concurrent readers produce identical results (no shared state).
//
// Not tested here: SHA-256 staleness (see VerifySourceSha256MatchesHeader and
// surrounding tests), TopologyAccessor fast-path integration.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/topology_snapshot.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "graph_models/gql/projection/topology_snapshot_writer.h"
#include "graph_models/object_id.h"

using GQL::Projection::kTopologySnapshotHeaderSize;
using GQL::Projection::kTopologySnapshotMagic;
using GQL::Projection::kTopologySnapshotVersion;
using GQL::Projection::kTopologySnapshotIdWidth;
using GQL::Projection::kTopologySnapshotIdWidthNarrow;
using GQL::Projection::TopologySnapshotFlags::kHasEdgeIds;
using GQL::Projection::TopologySnapshotFormatError;
using GQL::Projection::TopologySnapshotHeader;
using GQL::Projection::TopologySnapshotReader;
using GQL::Projection::TopologySnapshotWriter;
using GQL::Projection::make_default_topology_snapshot_header;
using GQL::Projection::serialize_topology_snapshot_header;

namespace {

// ---------------------------------------------------------------------------
// Fixture: hermetic temp dir + helpers to craft raw / valid files.
// ---------------------------------------------------------------------------

class TopologySnapshotReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto base = std::filesystem::temp_directory_path();
        std::random_device rd;
        std::mt19937_64 rng(rd());
        for (int attempt = 0; attempt < 64; ++attempt) {
            dir_ = base / ("mdb_topo_reader_test_" + std::to_string(rng()));
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
            std::filesystem::remove_all(dir_, ec);
        }
    }

    // Write a deterministic fake .leaf file next to the .csr so the writer's
    // finalize() has something to SHA-256 when we build valid fixtures.
    void write_fake_source_leaf(TopologySnapshotWriter::Direction d,
                                const std::string&                content) {
        const char* name = (d == TopologySnapshotWriter::Direction::FORWARD)
                         ? "from_to_edge.leaf"
                         : "to_from_edge.leaf";
        std::ofstream f(dir_ / name, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(f.good());
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
    }

    // Overwrite `topology_fwd.csr` with the given raw bytes. Used to craft
    // malformed files without going through the writer.
    void write_raw_fwd_csr(const std::vector<uint8_t>& bytes) {
        std::ofstream f(dir_ / "topology_fwd.csr",
                        std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(f.good());
        f.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    }

    std::filesystem::path dir_;
};

ObjectId oid(uint64_t v) { return ObjectId(v); }

// Construct a minimally valid header block + body, where the caller supplies
// version / flags / N / M overrides. Body is zero-filled enough to hit the
// declared file size. Useful for building "reject this specific field"
// negative cases without hand-hexing magic etc.
std::vector<uint8_t> build_raw_file_bytes(uint32_t version,
                                          uint8_t  flags,
                                          uint64_t N,
                                          uint64_t M,
                                          bool     corrupt_row_ptr_head = false,
                                          bool     corrupt_row_ptr_tail = false) {
    TopologySnapshotHeader header = make_default_topology_snapshot_header();
    header.version   = version;
    header.flags     = flags;
    header.num_nodes = N;
    header.num_edges = M;

    const bool has_eids = (flags & kHasEdgeIds) != 0;
    const std::size_t body_u64s = (N + 1) + M * (has_eids ? 2 : 1);

    std::vector<uint8_t> bytes(kTopologySnapshotHeaderSize
                               + body_u64s * sizeof(uint64_t), 0);
    uint8_t hdr_buf[kTopologySnapshotHeaderSize];
    serialize_topology_snapshot_header(header, hdr_buf);
    std::memcpy(bytes.data(), hdr_buf, kTopologySnapshotHeaderSize);

    // Write a well-formed ROW_PTR prefix sum where every node has degree
    // ceil(M / max(N,1)) (simplistic, just enough for invariant tests).
    // If N=0 there is only ROW_PTR[0] = 0.
    uint64_t* row_ptr = reinterpret_cast<uint64_t*>(
        bytes.data() + kTopologySnapshotHeaderSize);
    if (N == 0) {
        row_ptr[0] = 0;
    } else {
        uint64_t running = 0;
        for (uint64_t i = 0; i < N; ++i) {
            row_ptr[i] = running;
            // Spread M across N evenly-ish.
            uint64_t deg = (i < (M % N)) ? (M / N + 1) : (M / N);
            running += deg;
        }
        row_ptr[N] = running;  // Must equal M (if no corruption requested).
        if (corrupt_row_ptr_head) {
            row_ptr[0] = 42;  // break "ROW_PTR[0] == 0"
        }
        if (corrupt_row_ptr_tail) {
            row_ptr[N] = running + 1;  // break "ROW_PTR[N] == M"
        }
    }
    return bytes;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — Absent file: has_data() == false, no throw, no log noise.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, AbsentFileHasNoData) {
    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
    EXPECT_EQ(reader.num_nodes(), 0u);
    EXPECT_EQ(reader.num_edges(), 0u);
    EXPECT_FALSE(reader.has_edge_ids());
}

// ---------------------------------------------------------------------------
// Test 2 — Bad magic bytes: has_data()=false.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, BadMagicHasNoData) {
    // Build a structurally plausible file but with BADMAGIC in the first 8 B.
    std::vector<uint8_t> bytes = build_raw_file_bytes(
        kTopologySnapshotVersion, /*flags=*/0, /*N=*/2, /*M=*/1);
    const char bad[] = "BADMAGIC";
    std::memcpy(bytes.data(), bad, 8);
    write_raw_fwd_csr(bytes);

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
}

// ---------------------------------------------------------------------------
// Test 3 — Bad version (0, 2, or anything != 1): has_data()=false.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, BadVersionZeroHasNoData) {
    std::vector<uint8_t> bytes = build_raw_file_bytes(
        /*version=*/0, /*flags=*/0, /*N=*/2, /*M=*/1);
    write_raw_fwd_csr(bytes);
    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
}

TEST_F(TopologySnapshotReaderTest, BadVersionTwoHasNoData) {
    std::vector<uint8_t> bytes = build_raw_file_bytes(
        /*version=*/2, /*flags=*/0, /*N=*/2, /*M=*/1);
    write_raw_fwd_csr(bytes);
    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
}

// ---------------------------------------------------------------------------
// Test 4 — Truncated file (< 64 bytes): has_data()=false.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, TruncatedFileHasNoData) {
    // 32 bytes of zero — well short of the 64-byte header.
    std::vector<uint8_t> bytes(32, 0);
    write_raw_fwd_csr(bytes);
    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
}

// ---------------------------------------------------------------------------
// Test 5 — File size inconsistent with declared N/M: has_data()=false.
// Craft a file whose header claims M=10 but whose COL_IDX section only
// contains 5 u64 slots.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, WrongFileSizeHasNoData) {
    // Start from a valid N=2,M=10 file. Then truncate 40 bytes from the end
    // so the body no longer matches the header's claim.
    std::vector<uint8_t> bytes = build_raw_file_bytes(
        kTopologySnapshotVersion, /*flags=*/0, /*N=*/2, /*M=*/10);
    ASSERT_GT(bytes.size(), 40u);
    bytes.resize(bytes.size() - 40);  // short by 5 × u64
    write_raw_fwd_csr(bytes);

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
}

// ---------------------------------------------------------------------------
// Test 6a — ROW_PTR[0] != 0: reject.
// Test 6b — ROW_PTR[N] != M: reject.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, RowPtrHeadInvariantViolatedRejected) {
    auto bytes = build_raw_file_bytes(
        kTopologySnapshotVersion, /*flags=*/0, /*N=*/4, /*M=*/5,
        /*corrupt_row_ptr_head=*/true);
    write_raw_fwd_csr(bytes);
    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
}

TEST_F(TopologySnapshotReaderTest, RowPtrTailInvariantViolatedRejected) {
    // Corrupting the tail makes ROW_PTR[N] != M. The file size is still
    // consistent with M=5 (we just wrote a bad last element).
    auto bytes = build_raw_file_bytes(
        kTopologySnapshotVersion, /*flags=*/0, /*N=*/4, /*M=*/5,
        /*corrupt_row_ptr_head=*/false,
        /*corrupt_row_ptr_tail=*/true);
    write_raw_fwd_csr(bytes);
    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
}

// ---------------------------------------------------------------------------
// Test 7 — Valid file round-trip: use the writer, open via reader, confirm
// neighbors() slices match the intended graph.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, ValidFileNeighborsSliceCorrect) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "valid-graph-payload");

    // Graph: 4 nodes. Adjacency:
    //   0 → {1, 2}
    //   1 → {3}
    //   2 → {0}
    //   3 → {1}
    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/4,
        /*degrees=*/{2, 1, 1, 1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(0), oid(2), ObjectId());
    writer.append_edge(oid(1), oid(3), ObjectId());
    writer.append_edge(oid(2), oid(0), ObjectId());
    writer.append_edge(oid(3), oid(1), ObjectId());
    writer.finalize();

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(reader.has_data());
    EXPECT_EQ(reader.num_nodes(), 4u);
    EXPECT_EQ(reader.num_edges(), 5u);
    EXPECT_FALSE(reader.has_edge_ids());

    auto n0 = reader.neighbors(0);
    ASSERT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], 1u);
    EXPECT_EQ(n0[1], 2u);

    auto n1 = reader.neighbors(1);
    ASSERT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1[0], 3u);

    auto n2 = reader.neighbors(2);
    ASSERT_EQ(n2.size(), 1u);
    EXPECT_EQ(n2[0], 0u);

    auto n3 = reader.neighbors(3);
    ASSERT_EQ(n3.size(), 1u);
    EXPECT_EQ(n3[0], 1u);
}

// ---------------------------------------------------------------------------
// Test 8 — Node with degree 0 gives an empty span (not null, not UB).
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, EmptyNeighborListReturnsEmptySpan) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "degree-0-payload");

    // Graph: 3 nodes, only node 0 has an edge. Node 1 and node 2 are isolated.
    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/3,
        /*degrees=*/{1, 0, 0},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(2), ObjectId());
    writer.finalize();

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(reader.has_data());

    auto n0 = reader.neighbors(0);
    EXPECT_EQ(n0.size(), 1u);

    auto n1 = reader.neighbors(1);
    EXPECT_EQ(n1.size(), 0u);  // empty — node is isolated

    auto n2 = reader.neighbors(2);
    EXPECT_EQ(n2.size(), 0u);
}

// ---------------------------------------------------------------------------
// Test 9 — has_edge_ids flag clear → edge_ids(v) is empty for all v.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, EdgeIdsEmptyWhenFlagNotSet) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "no-eids");

    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/2,
        /*degrees=*/{1, 1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(1), oid(0), ObjectId());
    writer.finalize();

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(reader.has_data());
    EXPECT_FALSE(reader.has_edge_ids());
    EXPECT_EQ(reader.edge_ids(0).size(), 0u);
    EXPECT_EQ(reader.edge_ids(1).size(), 0u);
}

// ---------------------------------------------------------------------------
// Test 10 — has_edge_ids flag set → edge_ids(v) returns exact values.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, EdgeIdsPresentWhenFlagSet) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "eids");

    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/2,
        /*degrees=*/{2, 1},
        /*include_edge_ids=*/true);
    writer.append_edge(oid(0), oid(1), oid(100));
    writer.append_edge(oid(0), oid(0), oid(101));
    writer.append_edge(oid(1), oid(0), oid(102));
    writer.finalize();

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(reader.has_data());
    ASSERT_TRUE(reader.has_edge_ids());

    auto e0 = reader.edge_ids(0);
    ASSERT_EQ(e0.size(), 2u);
    EXPECT_EQ(e0[0], 100u);
    EXPECT_EQ(e0[1], 101u);

    auto e1 = reader.edge_ids(1);
    ASSERT_EQ(e1.size(), 1u);
    EXPECT_EQ(e1[0], 102u);

    // Neighbor slices are still correct too.
    auto n0 = reader.neighbors(0);
    ASSERT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], 1u);
    EXPECT_EQ(n0[1], 0u);
}

// ---------------------------------------------------------------------------
// Test 11 — Out-of-range node_idx → throws std::out_of_range.
// Bounds errors must surface, not silently read past the mmap.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, OutOfBoundsNodeIdxThrows) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "bounds-check");

    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/3,
        /*degrees=*/{1, 1, 1},
        /*include_edge_ids=*/true);
    writer.append_edge(oid(0), oid(1), oid(10));
    writer.append_edge(oid(1), oid(2), oid(11));
    writer.append_edge(oid(2), oid(0), oid(12));
    writer.finalize();

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(reader.has_data());

    EXPECT_THROW(reader.neighbors(3),  std::out_of_range);
    EXPECT_THROW(reader.neighbors(99), std::out_of_range);
    EXPECT_THROW(reader.edge_ids(3),   std::out_of_range);
    EXPECT_THROW(reader.edge_ids(99),  std::out_of_range);
}

// ---------------------------------------------------------------------------
// Test 12 — Two concurrent readers on the same file produce identical
// results under contention. Confirms mmap + the reader's const surface are
// thread-safe for read-only sharing.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, ConcurrentReadersNoCorruption) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "concurrent-readers");

    // Small but structured graph: 8 nodes, chain 0→1→2→…→7 plus fanout at 0.
    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/8,
        /*degrees=*/{3, 1, 1, 1, 1, 1, 1, 0},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(0), oid(3), ObjectId());
    writer.append_edge(oid(0), oid(5), ObjectId());
    writer.append_edge(oid(1), oid(2), ObjectId());
    writer.append_edge(oid(2), oid(3), ObjectId());
    writer.append_edge(oid(3), oid(4), ObjectId());
    writer.append_edge(oid(4), oid(5), ObjectId());
    writer.append_edge(oid(5), oid(6), ObjectId());
    writer.append_edge(oid(6), oid(7), ObjectId());
    writer.finalize();

    auto reader_a = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    auto reader_b = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(reader_a.has_data());
    ASSERT_TRUE(reader_b.has_data());

    constexpr int kIters = 100;
    auto hammer = [&](const TopologySnapshotReader& r,
                      std::vector<uint64_t>*       out) {
        for (int it = 0; it < kIters; ++it) {
            for (uint64_t v = 0; v < r.num_nodes(); ++v) {
                auto s = r.neighbors(v);
                for (uint64_t n : s) {
                    out->push_back(n);
                }
            }
        }
    };

    std::vector<uint64_t> out_a, out_b;
    std::thread t1(hammer, std::cref(reader_a), &out_a);
    std::thread t2(hammer, std::cref(reader_b), &out_b);
    t1.join();
    t2.join();

    ASSERT_EQ(out_a.size(), out_b.size());
    EXPECT_EQ(std::memcmp(out_a.data(), out_b.data(),
                          out_a.size() * sizeof(uint64_t)),
              0);

    // Sanity: the per-iter sum of neighbor ids across all nodes is
    // 1+3+5 + 2 + 3 + 4 + 5 + 6 + 7 = 36, × 8 nodes scan... actually, the
    // walk already sums every neighbor across every source, so expected =
    // (1+3+5 + 2 + 3 + 4 + 5 + 6 + 7) × kIters = 36 × 100 = 3600.
    uint64_t sum_a = 0;
    for (auto v : out_a) sum_a += v;
    EXPECT_EQ(sum_a, 36u * static_cast<uint64_t>(kIters));
}

// ---------------------------------------------------------------------------
// Test 13 — Producer/consumer SHA-256 invariant.
// The writer stamps SHA-256(source.leaf) into the header; the reader
// recomputes the same digest and must accept it. This is the matching
// half of the staleness gate: when the source B+Tree leaf file matches the
// digest embedded in the sidecar header, the reader accepts the sidecar
// as current. Paired with the two negative tests below (mutated leaf and
// deleted leaf) that exercise the fallback-to-absent behavior.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, VerifySourceSha256MatchesHeader) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "sha-match-payload");

    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/1,
        /*degrees=*/{1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(0), ObjectId());
    writer.finalize();

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(reader.has_data());

    // Matching content → matching hash.
    EXPECT_TRUE(reader.verify_source_sha256(dir_ / "from_to_edge.leaf"));

    // Conservative failure modes: path absent or unreadable → false
    // (treated as mismatch). No throw.
    EXPECT_FALSE(reader.verify_source_sha256("/nonexistent/path.leaf"));
}

// ---------------------------------------------------------------------------
// Test 13b — Mutating a single byte of the source `.leaf` after finalize
// invalidates the embedded digest; the next open() falls back to B+Tree
// (has_data()==false), matching the §3.4 fallback contract.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, MutatedSourceLeafTriggersFallback) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "pristine-source-leaf-contents");

    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/2,
        /*degrees=*/{1, 1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(1), oid(0), ObjectId());
    writer.finalize();

    // Baseline: the fresh sidecar + unmutated source open cleanly.
    auto baseline = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(baseline.has_data());

    // Flip a single byte in the source .leaf. The CSR header still
    // stores the *original* digest, so recomputation must diverge.
    const auto source_path = dir_ / "from_to_edge.leaf";
    {
        std::fstream f(source_path,
                       std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(f.good());
        f.seekp(0, std::ios::beg);
        char first = 0;
        f.read(&first, 1);
        first = static_cast<char>(first ^ 0xFF);
        f.seekp(0, std::ios::beg);
        f.write(&first, 1);
        ASSERT_TRUE(f.good());
    }

    // New reader over the same (still-valid) .csr must now see a
    // mismatched digest and fall back — has_data()==false.
    auto after_mutation = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(after_mutation.has_data());
    EXPECT_EQ(after_mutation.num_nodes(), 0u);
    EXPECT_EQ(after_mutation.num_edges(), 0u);
}

// ---------------------------------------------------------------------------
// Test 13c — Deleting the source `.leaf` means the reader cannot verify
// the digest at all; conservative policy is fall back, not trust.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, MissingSourceLeafTriggersFallback) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "will-be-deleted");

    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/2,
        /*degrees=*/{1, 1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(1), oid(0), ObjectId());
    writer.finalize();

    // Remove the source .leaf — the sidecar now has no verifiable origin.
    std::error_code ec;
    std::filesystem::remove(dir_ / "from_to_edge.leaf", ec);
    ASSERT_FALSE(ec);

    auto reader = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(reader.has_data());
}

// ---------------------------------------------------------------------------
// Test 14 — REVERSE direction loads topology_rev.csr and sees the right data.
// Distinct from the FORWARD tests to catch basename-routing regressions.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, ReverseDirectionOpensRevSidecar) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::REVERSE,
                           "reverse-payload");

    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::REVERSE,
        /*num_nodes=*/2,
        /*degrees=*/{1, 1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(1), oid(0), ObjectId());
    writer.finalize();

    // FORWARD reader finds nothing.
    auto fwd = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    EXPECT_FALSE(fwd.has_data());

    // REVERSE reader opens the file.
    auto rev = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::REVERSE);
    ASSERT_TRUE(rev.has_data());
    EXPECT_EQ(rev.num_nodes(), 2u);
    EXPECT_EQ(rev.num_edges(), 2u);

    auto n0 = rev.neighbors(0);
    ASSERT_EQ(n0.size(), 1u);
    EXPECT_EQ(n0[0], 1u);
    auto n1 = rev.neighbors(1);
    ASSERT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1[0], 0u);
}

// ---------------------------------------------------------------------------
// Test 15 — Move construction / assignment transfer ownership cleanly:
// moved-from reader is empty, moved-to reader serves slices correctly.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, MoveSemanticsPreserveData) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "move-payload");

    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/2,
        /*degrees=*/{1, 1},
        /*include_edge_ids=*/false);
    writer.append_edge(oid(0), oid(1), ObjectId());
    writer.append_edge(oid(1), oid(0), ObjectId());
    writer.finalize();

    auto a = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(a.has_data());

    // Move-construct b from a. a must now have has_data() == false.
    TopologySnapshotReader b(std::move(a));
    EXPECT_FALSE(a.has_data());  // NOLINT(bugprone-use-after-move) — intentional
    ASSERT_TRUE(b.has_data());
    auto n0 = b.neighbors(0);
    ASSERT_EQ(n0.size(), 1u);
    EXPECT_EQ(n0[0], 1u);

    // Move-assign over b from a freshly-opened reader. b's old mmap is
    // released without leaking.
    auto c = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    b = std::move(c);
    EXPECT_FALSE(c.has_data());  // NOLINT — intentional
    ASSERT_TRUE(b.has_data());
    auto n1 = b.neighbors(1);
    ASSERT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1[0], 0u);
}

// ===========================================================================
// Narrow (uint32) topology sidecar layout — round-trip + losslessness vs the
// default uint64 layout. When the MDB_GNN_TOPOLOGY_UINT32 environment variable
// is set, the writer strips the 8-bit ObjectId type tag from each node and
// edge id, stores only the lower 32 bits, and records the type tag once in
// the sidecar header. On read, the reader reconstructs the full tagged
// ObjectId by OR-ing the stored tag back in — achieving lossless round-trip
// at roughly half the sidecar size. These tests force the narrow layout via a
// RAII environment guard and verify that copy_neighbors / copy_edge_ids return
// byte-identical tagged values compared with the default wide layout.
// ===========================================================================

namespace {

// Forces the narrow (uint32) writer opt-in for the test's duration; restores
// the prior env on teardown so sibling tests stay on the default wide layout.
class NarrowEnvGuard {
public:
    NarrowEnvGuard() {
        if (const char* prev = std::getenv("MDB_GNN_TOPOLOGY_UINT32")) {
            had_prev_ = true;
            prev_     = prev;
        }
        ::setenv("MDB_GNN_TOPOLOGY_UINT32", "1", /*overwrite=*/1);
    }
    ~NarrowEnvGuard() {
        if (had_prev_) {
            ::setenv("MDB_GNN_TOPOLOGY_UINT32", prev_.c_str(), 1);
        } else {
            ::unsetenv("MDB_GNN_TOPOLOGY_UINT32");
        }
    }
private:
    bool        had_prev_ = false;
    std::string prev_;
};

// Per-section ObjectId type tags used by the narrow tests. Picked to exercise
// a real top-byte tag (the papers100M sidecar bug f71b3bf0 surfaced 0xD4).
constexpr uint64_t kDstTag = 0xD4ull << 56;
constexpr uint64_t kEidTag = 0xE2ull << 56;

}  // namespace

// Build the SAME tagged graph twice — once narrow (env on), once wide (env
// off) — and assert the reader yields byte-identical tagged ObjectIds for
// every node via the width-agnostic copy accessors, while the narrow file is
// strictly smaller and the zero-copy uint64 span throws for the narrow layout.
TEST_F(TopologySnapshotReaderTest, NarrowRoundTripLosslessVsWide) {
    auto build = [&](const std::filesystem::path& d) {
        std::filesystem::create_directories(d);
        {
            std::ofstream f(d / "from_to_edge.leaf",
                            std::ios::binary | std::ios::trunc);
            f << "narrow-vs-wide-payload";
        }
        // 0 → {1, 2}, 1 → {3}, 2 → {0}, 3 → {1}; dst + edge ids carry tags.
        TopologySnapshotWriter w(d, TopologySnapshotWriter::Direction::FORWARD,
                                 /*num_nodes=*/4, /*degrees=*/{2, 1, 1, 1},
                                 /*include_edge_ids=*/true);
        w.append_edge(oid(0), oid(kDstTag | 1), oid(kEidTag | 10));
        w.append_edge(oid(0), oid(kDstTag | 2), oid(kEidTag | 11));
        w.append_edge(oid(1), oid(kDstTag | 3), oid(kEidTag | 12));
        w.append_edge(oid(2), oid(kDstTag | 0), oid(kEidTag | 13));
        w.append_edge(oid(3), oid(kDstTag | 1), oid(kEidTag | 14));
        w.finalize();
    };

    const auto narrow_dir = dir_ / "narrow";
    const auto wide_dir   = dir_ / "wide";
    { NarrowEnvGuard g; build(narrow_dir); }
    build(wide_dir);  // env restored → wide layout

    auto rn = TopologySnapshotReader::open(
        narrow_dir, TopologySnapshotReader::Direction::FORWARD);
    auto rw = TopologySnapshotReader::open(
        wide_dir, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(rn.has_data());
    ASSERT_TRUE(rw.has_data());

    EXPECT_EQ(rn.id_width(), kTopologySnapshotIdWidthNarrow);
    EXPECT_EQ(rw.id_width(), kTopologySnapshotIdWidth);
    EXPECT_EQ(rn.dst_type_tag(), 0xD4u);
    EXPECT_EQ(rn.edge_type_tag(), 0xE2u);
    // Wide header carries no tag bytes.
    EXPECT_EQ(rw.dst_type_tag(), 0u);
    EXPECT_EQ(rw.edge_type_tag(), 0u);

    // Narrow file is strictly smaller: COL_IDX + EDGE_IDS halved (4B vs 8B),
    // ROW_PTR identical.
    EXPECT_LT(std::filesystem::file_size(narrow_dir / "topology_fwd.csr"),
              std::filesystem::file_size(wide_dir / "topology_fwd.csr"));

    // Per-node: degree + reconstructed tagged neighbors/edge_ids identical.
    for (uint64_t v = 0; v < 4; ++v) {
        EXPECT_EQ(rn.degree(v), rw.degree(v)) << "degree mismatch at v=" << v;

        std::vector<uint64_t> nn, wn;
        rn.copy_neighbors(v, nn);
        rw.copy_neighbors(v, wn);
        EXPECT_EQ(nn, wn) << "neighbor reconstruction mismatch at v=" << v;
        for (uint64_t x : nn) {
            EXPECT_EQ(x >> 56, 0xD4u)
                << "narrow reader must restore the dst type tag at v=" << v;
        }

        std::vector<uint64_t> ne, we;
        rn.copy_edge_ids(v, ne);
        rw.copy_edge_ids(v, we);
        EXPECT_EQ(ne, we) << "edge_id reconstruction mismatch at v=" << v;
        for (uint64_t x : ne) {
            EXPECT_EQ(x >> 56, 0xE2u)
                << "narrow reader must restore the edge type tag at v=" << v;
        }
    }

    // The zero-copy uint64 span is unavailable for the narrow layout.
    EXPECT_THROW(rn.neighbors(0), TopologySnapshotFormatError);
    EXPECT_THROW(rn.edge_ids(0), TopologySnapshotFormatError);
    // ... but still works for the wide layout.
    EXPECT_NO_THROW(rw.neighbors(0));
    EXPECT_NO_THROW(rw.edge_ids(0));
}

// The narrow zero-copy raw-uint32 accessors (used by the four-level store's
// hot tier-3 dispatch) return non-null pointers whose widened+tagged values
// match copy_neighbors; the wide layout returns nullptr from them.
TEST_F(TopologySnapshotReaderTest, NarrowZeroCopyU32PointersReconstruct) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "narrow-u32-ptr-payload");
    {
        NarrowEnvGuard g;
        TopologySnapshotWriter w(dir_, TopologySnapshotWriter::Direction::FORWARD,
                                 /*num_nodes=*/3, /*degrees=*/{2, 1, 0},
                                 /*include_edge_ids=*/true);
        w.append_edge(oid(0), oid(kDstTag | 1), oid(kEidTag | 7));
        w.append_edge(oid(0), oid(kDstTag | 2), oid(kEidTag | 8));
        w.append_edge(oid(1), oid(kDstTag | 0), oid(kEidTag | 9));
        w.finalize();
    }
    auto r = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(r.has_data());
    ASSERT_EQ(r.id_width(), kTopologySnapshotIdWidthNarrow);

    const uint32_t* col0 = r.col_idx32_row(0);
    const uint32_t* eid0 = r.edge_ids32_row(0);
    ASSERT_NE(col0, nullptr);
    ASSERT_NE(eid0, nullptr);
    const uint64_t dtag = static_cast<uint64_t>(r.dst_type_tag()) << 56;
    const uint64_t etag = static_cast<uint64_t>(r.edge_type_tag()) << 56;
    EXPECT_EQ(dtag | col0[0], kDstTag | 1);
    EXPECT_EQ(dtag | col0[1], kDstTag | 2);
    EXPECT_EQ(etag | eid0[0], kEidTag | 7);
    EXPECT_EQ(etag | eid0[1], kEidTag | 8);

    // degree-0 node: pointer is in-range but spans nothing.
    EXPECT_NO_THROW((void)r.col_idx32_row(2));
}

// The parallel append_subrange path narrows + captures the tag via CAS and is
// lossless on read.
TEST_F(TopologySnapshotReaderTest, NarrowAppendSubrangeRoundTrip) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "narrow-subrange-payload");
    {
        NarrowEnvGuard g;
        TopologySnapshotWriter w(dir_, TopologySnapshotWriter::Direction::FORWARD,
                                 /*num_nodes=*/4, /*degrees=*/{2, 1, 1, 1},
                                 /*include_edge_ids=*/true);
        // One subrange covering all sources; dst/eid carry tags (raw, like the
        // leaf record fields the parallel builder passes).
        std::vector<uint64_t> dst = {kDstTag | 1, kDstTag | 2, kDstTag | 3,
                                     kDstTag | 0, kDstTag | 1};
        std::vector<uint64_t> eid = {kEidTag | 10, kEidTag | 11, kEidTag | 12,
                                     kEidTag | 13, kEidTag | 14};
        w.append_subrange(/*lo_src=*/0, /*hi_src=*/4, dst, eid);
        w.finalize();
    }
    auto r = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(r.has_data());
    ASSERT_EQ(r.id_width(), kTopologySnapshotIdWidthNarrow);

    std::vector<uint64_t> n0;
    r.copy_neighbors(0, n0);
    ASSERT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], kDstTag | 1);
    EXPECT_EQ(n0[1], kDstTag | 2);
    std::vector<uint64_t> e0;
    r.copy_edge_ids(0, e0);
    ASSERT_EQ(e0.size(), 2u);
    EXPECT_EQ(e0[0], kEidTag | 10);
    EXPECT_EQ(e0[1], kEidTag | 11);
}

// A narrow section must carry a single ObjectId type tag; mixing tags is not
// representable and must fail loud rather than silently corrupt the read.
TEST_F(TopologySnapshotReaderTest, NarrowInconsistentDstTagThrows) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "narrow-bad-tag-payload");
    NarrowEnvGuard g;
    TopologySnapshotWriter w(dir_, TopologySnapshotWriter::Direction::FORWARD,
                             /*num_nodes=*/2, /*degrees=*/{2, 0},
                             /*include_edge_ids=*/false);
    w.append_edge(oid(0), oid((0xD4ull << 56) | 1), ObjectId());
    // Second dst carries a different top-byte tag → capture_tag_ rejects it.
    EXPECT_THROW(
        w.append_edge(oid(0), oid((0xC2ull << 56) | 0), ObjectId()),
        std::runtime_error);
}

// The default (env unset) layout stays wide even for a small graph — proves
// the narrow path is strictly opt-in and the legacy layout is unchanged.
TEST_F(TopologySnapshotReaderTest, DefaultLayoutStaysWideWithoutOptIn) {
    ::unsetenv("MDB_GNN_TOPOLOGY_UINT32");  // ensure clean default
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD,
                           "default-wide-payload");
    TopologySnapshotWriter w(dir_, TopologySnapshotWriter::Direction::FORWARD,
                             /*num_nodes=*/2, /*degrees=*/{1, 0},
                             /*include_edge_ids=*/false);
    w.append_edge(oid(0), oid(kDstTag | 1), ObjectId());
    w.finalize();

    auto r = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    ASSERT_TRUE(r.has_data());
    EXPECT_EQ(r.id_width(), kTopologySnapshotIdWidth);
    // Wide layout keeps the full tagged ObjectId in the zero-copy span.
    auto n0 = r.neighbors(0);
    ASSERT_EQ(n0.size(), 1u);
    EXPECT_EQ(n0[0], kDstTag | 1);
}

// ===========================================================================
// Symmetric reader: open_symmetric() + two-source staleness gate (TOPOSYM1)
// ===========================================================================
//
// The symmetric sidecar topology_sym.csr carries a distinct magic ("TOPOSYM1")
// and a COMBINED digest chaining BOTH source .leaf streams (from_to_edge then
// to_from_edge, fixed order). open_symmetric() runs the SAME structural
// validation as open() but parses with the sym parser and verifies against the
// two-source chained digest. Same fallback-first contract: absent / malformed /
// stale → has_data()==false, never throws.

// Build a valid topology_sym.csr in `dir` from an undirected adjacency, hashing
// from_to_edge.leaf then to_from_edge.leaf (fixed chaining order).
// N=3 undirected: 0-{1,2}, 1-{0}, 2-{0}.
static void build_sym_fixture(const std::filesystem::path& dir,
                              const std::string& fwd_leaf,
                              const std::string& rev_leaf) {
    { std::ofstream f(dir / "from_to_edge.leaf",
                      std::ios::binary | std::ios::trunc);
      f.write(fwd_leaf.data(), static_cast<std::streamsize>(fwd_leaf.size())); }
    { std::ofstream f(dir / "to_from_edge.leaf",
                      std::ios::binary | std::ios::trunc);
      f.write(rev_leaf.data(), static_cast<std::streamsize>(rev_leaf.size())); }
    TopologySnapshotWriter w(
        dir,
        std::string("topology_sym.csr"),
        std::vector<std::filesystem::path>{dir / "from_to_edge.leaf",
                                           dir / "to_from_edge.leaf"},
        /*num_nodes=*/3,
        /*degrees=*/{2, 1, 1},
        /*include_edge_ids=*/false,
        /*symmetric_format=*/true);
    w.append_edge(oid(0), oid(1), ObjectId());
    w.append_edge(oid(0), oid(2), ObjectId());
    w.append_edge(oid(1), oid(0), ObjectId());
    w.append_edge(oid(2), oid(0), ObjectId());
    w.finalize();
}

TEST_F(TopologySnapshotReaderTest, SymOpenSucceedsAndSlicesNeighbors) {
    build_sym_fixture(dir_, "fwd-bytes", "rev-bytes");
    auto r = TopologySnapshotReader::open_symmetric(dir_);
    ASSERT_TRUE(r.has_data());
    EXPECT_EQ(r.num_nodes(), 3u);
    EXPECT_EQ(r.num_edges(), 4u);
    EXPECT_FALSE(r.has_edge_ids());

    auto n0 = r.neighbors(0);
    ASSERT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], 1u);
    EXPECT_EQ(n0[1], 2u);
    auto n1 = r.neighbors(1);
    ASSERT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1[0], 0u);
}

TEST_F(TopologySnapshotReaderTest, SymOpenAbsentFileHasNoData) {
    // No topology_sym.csr written at all.
    auto r = TopologySnapshotReader::open_symmetric(dir_);
    EXPECT_FALSE(r.has_data());
}

TEST_F(TopologySnapshotReaderTest, SymOpenRejectsDirectionalMagic) {
    // A directional fwd writer produces "TOPOCSR1"; place it under the sym name
    // by writing a directional file and renaming it.
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "x");
    TopologySnapshotWriter w(
        dir_, TopologySnapshotWriter::Direction::FORWARD,
        /*num_nodes=*/1, /*degrees=*/{1}, /*include_edge_ids=*/false);
    w.append_edge(oid(0), oid(0), ObjectId());
    w.finalize();
    std::filesystem::rename(dir_ / "topology_fwd.csr", dir_ / "topology_sym.csr");

    auto r = TopologySnapshotReader::open_symmetric(dir_);
    EXPECT_FALSE(r.has_data())
        << "directional magic must be rejected by the symmetric opener";
}

TEST_F(TopologySnapshotReaderTest, SymStalenessFallsBackWhenLeafChanges) {
    build_sym_fixture(dir_, "fwd-original", "rev-original");
    // Mutate one source .leaf AFTER the sidecar was written → combined digest
    // no longer matches → open_symmetric must fall back to has_data()==false.
    { std::ofstream f(dir_ / "to_from_edge.leaf",
                      std::ios::binary | std::ios::trunc);
      const std::string m = "rev-MUTATED";
      f.write(m.data(), static_cast<std::streamsize>(m.size())); }

    auto r = TopologySnapshotReader::open_symmetric(dir_);
    EXPECT_FALSE(r.has_data())
        << "a post-write edit to either source .leaf must stale the sym sidecar";
}

TEST_F(TopologySnapshotReaderTest, SymVerifyCombinedSha256MatchesFreshLeaves) {
    build_sym_fixture(dir_, "fwd-bytes", "rev-bytes");
    auto r = TopologySnapshotReader::open_symmetric(dir_);
    ASSERT_TRUE(r.has_data());
    EXPECT_TRUE(r.verify_combined_sha256(
        {dir_ / "from_to_edge.leaf", dir_ / "to_from_edge.leaf"}));
    // Reversed order must NOT verify (chaining order is load-bearing).
    EXPECT_FALSE(r.verify_combined_sha256(
        {dir_ / "to_from_edge.leaf", dir_ / "from_to_edge.leaf"}));
}

TEST_F(TopologySnapshotReaderTest, SymTrustSidecarBypassesStaleness) {
    build_sym_fixture(dir_, "fwd-x", "rev-x");
    { std::ofstream f(dir_ / "from_to_edge.leaf",
                      std::ios::binary | std::ios::trunc);
      const std::string m = "fwd-CHANGED";
      f.write(m.data(), static_cast<std::streamsize>(m.size())); }
    ::setenv("MDB_GNN_TRUST_SIDECAR", "1", /*overwrite=*/1);
    auto r = TopologySnapshotReader::open_symmetric(dir_);
    ::unsetenv("MDB_GNN_TRUST_SIDECAR");
    EXPECT_TRUE(r.has_data())
        << "MDB_GNN_TRUST_SIDECAR must skip the combined staleness gate";
}
