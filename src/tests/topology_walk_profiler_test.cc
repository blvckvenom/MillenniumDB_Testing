// Unit tests for TopologyWalkProfiler — the cold-start topology random-walk
// profiler that runs degree-weighted Vose-alias random walks over the
// topology CSR sidecar (mmap, O(1) neighbor slice) to estimate per-node
// access frequencies and write node_counts.bin before the Four-Level
// Topology Store (L1 RAM hash / L2 compact uint32 CSR / L3 mmap sidecar /
// L4 direct B+Tree) is built for the first time.
//
// Covers:
//   1. Empty reader (has_data=false) → empty Result.
//   2. Single chain graph → counts populated, sum == lookups_done.
//   3. Deterministic seed → identical counts across two calls.
//   4. Different seeds → different counts (no accidental clash).
//   5. Isolated node fixture → walks restart, restarts counter ≥ 1.
//   6. counts.size() == reader.num_nodes() (post-condition).
//   7. lookups_done bounded by num_walks × walk_length.
//   8. Default parameters (num_walks=0 or walk_length=0) use kDefault constants.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "graph_models/gql/projection/topology_snapshot_writer.h"
#include "graph_models/object_id.h"

#include "gnn/projection/topology_walk_profiler.h"

using GQL::Projection::TopologySnapshotReader;
using GQL::Projection::TopologySnapshotWriter;
using mdb::gnn::TopologyWalkProfiler;

namespace {

class TopologyWalkProfilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto base = std::filesystem::temp_directory_path();
        std::random_device rd;
        std::mt19937_64 rng(rd());
        for (int attempt = 0; attempt < 64; ++attempt) {
            dir_ = base / ("mdb_walk_profiler_test_" + std::to_string(rng()));
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

    // Write a placeholder .leaf file so the writer's SHA-256 step has
    // something to hash. The content is irrelevant — the reader will
    // compare the hash against this same file at open() time.
    void write_fake_source_leaf(TopologySnapshotWriter::Direction d,
                                const std::string&                content) {
        const char* name = (d == TopologySnapshotWriter::Direction::FORWARD)
                         ? "from_to_edge.leaf"
                         : "to_from_edge.leaf";
        std::ofstream f(dir_ / name, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(f.good());
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
    }

    // Build a tiny 4-node chain graph in REVERSE direction:
    //   0 ← 1 ← 2 ← 3   (each non-zero node points to its predecessor)
    // Sampling REVERSE means we follow these arrows backward.
    void build_chain_graph_(uint64_t n) {
        write_fake_source_leaf(TopologySnapshotWriter::Direction::REVERSE,
                               "chain-graph-payload");

        std::vector<uint64_t> degrees(static_cast<std::size_t>(n), 0);
        // Every node except the source-of-everything has one neighbor.
        for (uint64_t i = 0; i + 1 < n; ++i) {
            degrees[static_cast<std::size_t>(i)] = 1;
        }

        TopologySnapshotWriter writer(
            dir_, TopologySnapshotWriter::Direction::REVERSE,
            n, degrees, /*include_edge_ids=*/false);
        for (uint64_t i = 0; i + 1 < n; ++i) {
            writer.append_edge(ObjectId(i), ObjectId(i + 1), ObjectId());
        }
        writer.finalize();
    }

    // Build a graph where node 0 is isolated (degree 0) and 1..N-1 form
    // a cycle. Used to exercise the dead-end / restart logic without
    // making the entire graph unwalkable.
    void build_with_isolated_zero_(uint64_t n) {
        ASSERT_GE(n, 2u) << "need ≥2 nodes for cycle";
        write_fake_source_leaf(TopologySnapshotWriter::Direction::REVERSE,
                               "isolated-zero-payload");

        std::vector<uint64_t> degrees(static_cast<std::size_t>(n), 0);
        for (uint64_t i = 1; i < n; ++i) {
            degrees[static_cast<std::size_t>(i)] = 1;
        }

        TopologySnapshotWriter writer(
            dir_, TopologySnapshotWriter::Direction::REVERSE,
            n, degrees, /*include_edge_ids=*/false);
        for (uint64_t i = 1; i < n; ++i) {
            const uint64_t dst = (i + 1 < n) ? (i + 1) : 1;
            writer.append_edge(ObjectId(i), ObjectId(dst), ObjectId());
        }
        writer.finalize();
    }

    TopologySnapshotReader open_reverse_() {
        return TopologySnapshotReader::open(
            dir_, TopologySnapshotReader::Direction::REVERSE);
    }

    std::filesystem::path dir_;
};

// =========================================================================
// 1. Empty reader → empty Result.
// =========================================================================
TEST_F(TopologyWalkProfilerTest, EmptyReaderReturnsEmptyResult) {
    // No leaf, no .csr — reader will fail to open and has_data()=false.
    auto reader = open_reverse_();
    ASSERT_FALSE(reader.has_data())
        << "precondition: reader without sidecar should report has_data=false";

    auto result = TopologyWalkProfiler::profile(
        reader, /*num_walks=*/100, /*walk_length=*/5, /*seed=*/42);

    EXPECT_TRUE(result.counts.empty());
    EXPECT_EQ(result.lookups_done, 0u);
    EXPECT_EQ(result.restarts,     0u);
    EXPECT_DOUBLE_EQ(result.elapsed_seconds, 0.0);
}

// =========================================================================
// 2. Counts populated; sum equals lookups_done (every step ticks a count).
// =========================================================================
TEST_F(TopologyWalkProfilerTest, ChainGraphCountsSumMatchesLookups) {
    constexpr uint64_t N = 8;
    build_chain_graph_(N);
    auto reader = open_reverse_();
    ASSERT_TRUE(reader.has_data());

    auto result = TopologyWalkProfiler::profile(
        reader, /*num_walks=*/50, /*walk_length=*/4, /*seed=*/123);

    ASSERT_EQ(result.counts.size(), static_cast<std::size_t>(N));
    uint64_t sum = 0;
    for (uint64_t c : result.counts) sum += c;
    EXPECT_EQ(sum, result.lookups_done);
    EXPECT_GT(result.lookups_done, 0u);
    EXPECT_LE(result.lookups_done, 50u * 4u);
}

// =========================================================================
// 3. Deterministic seed → identical counts on a second call.
// =========================================================================
TEST_F(TopologyWalkProfilerTest, DeterministicSeedReproduces) {
    constexpr uint64_t N = 16;
    build_chain_graph_(N);
    auto reader = open_reverse_();
    ASSERT_TRUE(reader.has_data());

    auto a = TopologyWalkProfiler::profile(reader, 200, 5, 7777);
    auto b = TopologyWalkProfiler::profile(reader, 200, 5, 7777);

    ASSERT_EQ(a.counts.size(), b.counts.size());
    EXPECT_EQ(a.counts, b.counts);
    EXPECT_EQ(a.lookups_done, b.lookups_done);
    EXPECT_EQ(a.restarts,     b.restarts);
}

// =========================================================================
// 4. Different seeds produce different walks. Equal counts would suggest
//    the RNG isn't actually being consumed.
// =========================================================================
TEST_F(TopologyWalkProfilerTest, DifferentSeedsProduceDifferentCounts) {
    constexpr uint64_t N = 64;
    build_chain_graph_(N);
    auto reader = open_reverse_();
    ASSERT_TRUE(reader.has_data());

    auto a = TopologyWalkProfiler::profile(reader, 200, 5, 1);
    auto b = TopologyWalkProfiler::profile(reader, 200, 5, 2);

    EXPECT_NE(a.counts, b.counts);
}

// =========================================================================
// 5a. Papers100M-style pathology: 1 hub + many isolated leaves.
//
// Replicates the empirically-observed access skew in citation graphs:
// the vast majority of nodes (papers nobody ever cited) have degree=0
// in REVERSE direction. Naive uniform seed selection would land 99%
// of walks on isolated nodes; even a buggy alias-method (e.g. one
// whose leftover-large cleanup leaves alias_[i]=0 pointing at an
// isolated node 0) would silently degrade to 100% restarts. The
// post-fix degree-weighted alias MUST land 100% of walks on the hub
// (the only node with neighbors) on this fixture.
//
// This test is the regression guard for commit 42c6970b's papers100M
// failure mode.
// =========================================================================
TEST_F(TopologyWalkProfilerTest, MostlyIsolatedGraphLandsOnHub) {
    // 20 nodes: node 5 has degree 19 (cites all others), the rest are
    // isolated under REVERSE direction. Without the eligible-only
    // filter, leftover-large bucket entries can leave alias slots
    // pointing at node 0 (which is isolated here), reproducing the
    // papers100M pathology.
    constexpr uint64_t N    = 20;
    constexpr uint64_t HUB  = 5;
    write_fake_source_leaf(TopologySnapshotWriter::Direction::REVERSE,
                           "hub-fixture-payload");
    std::vector<uint64_t> degrees(static_cast<std::size_t>(N), 0);
    degrees[HUB] = N - 1;  // hub points to every other node
    TopologySnapshotWriter writer(
        dir_, TopologySnapshotWriter::Direction::REVERSE,
        N, degrees, /*include_edge_ids=*/false);
    for (uint64_t j = 0; j < N; ++j) {
        if (j == HUB) continue;
        writer.append_edge(ObjectId(HUB), ObjectId(j), ObjectId());
    }
    writer.finalize();

    auto reader = open_reverse_();
    ASSERT_TRUE(reader.has_data());

    constexpr std::size_t W = 2000;
    constexpr std::size_t L = 4;
    auto result = TopologyWalkProfiler::profile(
        reader, /*num_walks=*/W, /*walk_length=*/L, /*seed=*/2026);

    // Every walk MUST seed on the hub (only eligible node). The
    // walk then bounces hub → leaf and dead-ends (leaves have no
    // reverse-neighbors), so each walk performs exactly 2 lookups
    // (seed + 1 step) and accumulates 1 restart.
    EXPECT_EQ(result.counts[HUB], W)
        << "hub must be selected as seed for every walk under the "
        << "eligible-only alias filter";
    EXPECT_EQ(result.restarts, W)
        << "each walk should dead-end on its first step (leaves are isolated)";
    EXPECT_EQ(result.lookups_done, W * 2u)
        << "two lookups per walk: seed + one bounce to a leaf";

    // Sanity: leaf counts sum to W (one bounce per walk distributed
    // uniformly across the N-1 leaves).
    uint64_t leaf_total = 0;
    for (std::size_t i = 0; i < result.counts.size(); ++i) {
        if (i != HUB) leaf_total += result.counts[i];
    }
    EXPECT_EQ(leaf_total, W);
}

// =========================================================================
// 5b. Cycle-with-isolated-zero fixture (degree-weighted alias version).
// =========================================================================
TEST_F(TopologyWalkProfilerTest, DegreeWeightedSkipsIsolatedSeeds) {
    constexpr uint64_t N = 10;
    build_with_isolated_zero_(N);
    auto reader = open_reverse_();
    ASSERT_TRUE(reader.has_data());

    // Run many walks. Each non-isolated node has degree=1 in this
    // fixture, so the alias table puts uniform weight on nodes 1..N-1
    // and zero on node 0. Node 0 should never be picked.
    auto result = TopologyWalkProfiler::profile(
        reader, /*num_walks=*/500, /*walk_length=*/3, /*seed=*/42);

    EXPECT_EQ(result.counts[0], 0u)
        << "isolated node must not be selected by degree-weighted seed sampler";

    // Sanity: non-isolated nodes accumulated counts.
    uint64_t non_zero = 0;
    for (std::size_t i = 1; i < result.counts.size(); ++i) {
        if (result.counts[i] > 0) non_zero++;
    }
    EXPECT_GT(non_zero, 0u);
}

// =========================================================================
// 6. counts.size() always matches num_nodes() post-call.
// =========================================================================
TEST_F(TopologyWalkProfilerTest, CountsSizeEqualsNumNodes) {
    constexpr uint64_t N = 12;
    build_chain_graph_(N);
    auto reader = open_reverse_();
    ASSERT_TRUE(reader.has_data());
    ASSERT_EQ(reader.num_nodes(), N);

    auto result = TopologyWalkProfiler::profile(reader, 10, 3, 99);
    EXPECT_EQ(result.counts.size(), static_cast<std::size_t>(reader.num_nodes()));
}

// =========================================================================
// 7. lookups_done is bounded by num_walks × walk_length. The cap can be
//    reached on a fully-connected (or chain) graph if no walk dead-ends.
// =========================================================================
TEST_F(TopologyWalkProfilerTest, LookupsBoundedByWalksTimesLength) {
    constexpr uint64_t N = 32;
    build_chain_graph_(N);
    auto reader = open_reverse_();
    ASSERT_TRUE(reader.has_data());

    constexpr std::size_t W = 17;
    constexpr std::size_t L = 6;
    auto result = TopologyWalkProfiler::profile(reader, W, L, 1234);
    EXPECT_LE(result.lookups_done, W * L);
}

// =========================================================================
// 8. Default constants kick in when caller passes 0.
// =========================================================================
TEST_F(TopologyWalkProfilerTest, DefaultParametersWhenZero) {
    constexpr uint64_t N = 8;
    build_chain_graph_(N);
    auto reader = open_reverse_();
    ASSERT_TRUE(reader.has_data());

    // Use a tiny ceiling override via custom defaults indirectly: confirm
    // the defaults yielded lookups in a band consistent with kDefault*.
    // We don't want this test to run 100k×5 walks during ctest, so we
    // pass explicit but small values to verify the "0 = default" path
    // executes by checking the bound against the default constants.
    auto result0 = TopologyWalkProfiler::profile(reader, /*num_walks=*/0,
                                                 /*walk_length=*/0,
                                                 /*seed=*/5);

    // 0/0 ⇒ kDefault*. We assert lookups_done is at least 1 (some walk
    // ran) and at most kDefaultNumWalks × kDefaultWalkLength (no overshoot).
    EXPECT_GT(result0.lookups_done, 0u);
    EXPECT_LE(result0.lookups_done,
              TopologyWalkProfiler::kDefaultNumWalks *
              TopologyWalkProfiler::kDefaultWalkLength);
}

}  // namespace
