// Unit tests for TopologySnapshotReader (Spec #4-B, T4.5).
//
// Scope (per plan T4.5 acceptance criteria):
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
// Not tested here: SHA-256 staleness (T4.10), TopologyAccessor fast-path
// integration (T4.7).

#include <cstdint>
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
using GQL::Projection::TopologySnapshotFlags::kHasEdgeIds;
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
// Test 13 — The T4.5 verify_source_sha256() stub returns true unconditionally.
// Pin that so that T4.10 (which replaces this body) has an observable
// transition point: this test will need to be updated / moved to the real
// SHA-256 suite when T4.10 lands.
// ---------------------------------------------------------------------------
TEST_F(TopologySnapshotReaderTest, VerifySourceSha256StubReturnsTrue) {
    write_fake_source_leaf(TopologySnapshotWriter::Direction::FORWARD, "stub-payload");

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

    // Stub: returns true regardless of the path argument, including paths
    // that don't exist. T4.10 replaces this with a real streaming hash.
    EXPECT_TRUE(reader.verify_source_sha256(dir_ / "from_to_edge.leaf"));
    EXPECT_TRUE(reader.verify_source_sha256("/nonexistent/path.leaf"));
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
