// four_level_topology_release_directional_test.cc
//
// Unit tests for FourLevelTopologyStore::release_directional_after_symmetric_pin
// and the in-process RSS probes (process_memory.h).
//
// The directional fwd/rev tiers are dead weight once the merged undirected slice
// is materialized + pinned on the symmetric GPU path (the GPU kernel walks only
// the pinned slice; the CPU UNDIRECTED dispatch uses the symmetric tier). These
// tests drive the store through the dispatcher (Phase 2) ctor with two narrow
// (id_width==4) L3 sidecar readers — the form materialize_symmetric_arrays()
// requires — and assert the release contract: no-op without a slice, frees +
// nulls the directional aliases + leaves the slice intact when a slice exists,
// and is idempotent. CI-friendly: no GPU and no projection build needed.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/common/process_memory.h"
#include "gnn/projection/four_level_topology_store.h"
#include "gnn/projection/l1_hash_cache.h"
#include "gnn/projection/l2_compact_csr.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "graph_models/gql/projection/topology_snapshot_writer.h"
#include "graph_models/object_id.h"

using GQL::Projection::TopologySnapshotReader;
using GQL::Projection::TopologySnapshotWriter;
using mdb::gnn::FourLevelTopologyStore;
using mdb::gnn::L1HashCache;
using mdb::gnn::L2CompactCsr;

namespace {

ObjectId oid(uint64_t v) { return ObjectId(v); }

// Hermetic temp dir + helpers to write narrow CSR sidecars through the real
// writer (so the reader's staleness gate + narrow uint32 layout are exercised).
class ReleaseDirectionalTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Force the narrow uint32 layout so materialize_symmetric_arrays() (which
        // requires id_width()==4) engages on these tiny fixtures.
        ::setenv("MDB_GNN_TOPOLOGY_UINT32", "1", /*overwrite=*/1);
        auto base = std::filesystem::temp_directory_path();
        std::random_device rd;
        std::mt19937_64 rng(rd());
        for (int attempt = 0; attempt < 64; ++attempt) {
            dir_ = base / ("mdb_release_dir_test_" + std::to_string(rng()));
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

    void write_fake_source_leaf(TopologySnapshotWriter::Direction d,
                                const std::string&                content) {
        const char* name = (d == TopologySnapshotWriter::Direction::FORWARD)
                         ? "from_to_edge.leaf"
                         : "to_from_edge.leaf";
        std::ofstream f(dir_ / name, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(f.good());
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
    }

    // Write a directional sidecar; edges is a list of (src,dst) with src
    // ascending and degrees derived from it.
    void write_sidecar(TopologySnapshotWriter::Direction d,
                       uint64_t num_nodes,
                       const std::vector<std::pair<uint64_t, uint64_t>>& edges) {
        write_fake_source_leaf(d, "release-dir-payload");
        std::vector<uint64_t> degrees(num_nodes, 0);
        for (auto& e : edges) ++degrees[e.first];
        TopologySnapshotWriter writer(dir_, d, num_nodes, degrees,
                                      /*include_edge_ids=*/false);
        for (auto& e : edges) writer.append_edge(oid(e.first), oid(e.second),
                                                 ObjectId());
        writer.finalize();
    }

    std::filesystem::path dir_;
};

// A dispatcher store with NO L3 sidecars never materializes a slice, so the
// release is a no-op: nothing freed, not flagged released, directional fetches
// still work.
TEST_F(ReleaseDirectionalTest, NoOpWhenSliceNeverMaterialized) {
    const std::vector<uint8_t> tiers = {1, 1};
    L1HashCache l1f(tiers), l1r(tiers);
    L2CompactCsr l2f, l2r;
    l2f.freeze();
    l2r.freeze();
    FourLevelTopologyStore::Config cfg;  // UNDIRECTED default
    FourLevelTopologyStore store(
        l1f, l1r, l2f, l2r,
        /*l3_fwd=*/nullptr, /*l3_rev=*/nullptr,
        /*l4_fwd=*/{}, /*l4_rev=*/{},
        tiers, [](ObjectId v) { return v.id; }, cfg);

    EXPECT_EQ(store.release_directional_after_symmetric_pin(), 0u);
    EXPECT_FALSE(store.directional_released());
    // Directional fetch still serviceable (no throw).
    EXPECT_NO_THROW((void)store.get_out_neighbors(oid(0)));
}

// With two narrow L3 readers, materialize builds the merged slice; the release
// then flags released, nulls the directional L3 aliases, leaves the merged
// slice intact, and makes directional fetches throw — while get_neighbors stays
// callable (the symmetric path).
TEST_F(ReleaseDirectionalTest, FreesAndGuardsAfterMaterialize) {
    // fwd: 0->{1,2} 1->{3} 2->{0} 3->{1} ; rev = transpose
    write_sidecar(TopologySnapshotWriter::Direction::FORWARD, 4,
                  {{0, 1}, {0, 2}, {1, 3}, {2, 0}, {3, 1}});
    write_sidecar(TopologySnapshotWriter::Direction::REVERSE, 4,
                  {{0, 2}, {1, 0}, {1, 3}, {2, 0}, {3, 1}});
    auto rf = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    auto rr = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::REVERSE);
    ASSERT_TRUE(rf.has_data());
    ASSERT_TRUE(rr.has_data());
    ASSERT_EQ(rf.id_width(), 4);  // narrow — required by materialize
    ASSERT_EQ(rr.id_width(), 4);

    const std::vector<uint8_t> tiers = {1, 1, 1, 1};
    L1HashCache l1f(tiers), l1r(tiers);
    L2CompactCsr l2f, l2r;
    l2f.freeze();
    l2r.freeze();
    FourLevelTopologyStore::Config cfg;  // UNDIRECTED default
    FourLevelTopologyStore store(
        l1f, l1r, l2f, l2r,
        &rf, &rr,
        /*l4_fwd=*/{}, /*l4_rev=*/{},
        tiers, [](ObjectId v) { return v.id; }, cfg);

    // Materialize the merged undirected slice (CPU vector merge; no GPU needed).
    const auto* sym = store.materialize_symmetric_arrays();
    ASSERT_NE(sym, nullptr);
    const std::size_t slice_bytes = store.symmetric_ram_bytes();
    EXPECT_GT(slice_bytes, 0u);
    ASSERT_NE(store.l3_fwd(), nullptr);
    ASSERT_NE(store.l3_rev(), nullptr);

    // Release. In the dispatcher ctor the store does NOT own the L3 readers
    // (the test keeps them), so the reported freed bytes are 0 here — but the
    // observable post-conditions (flag, nulled aliases, slice survival, throw)
    // are exactly those of the owned (build()) path used in production.
    (void)store.release_directional_after_symmetric_pin();
    EXPECT_TRUE(store.directional_released());
    EXPECT_EQ(store.l3_fwd(), nullptr);
    EXPECT_EQ(store.l3_rev(), nullptr);
    // The merged slice is untouched by the release.
    EXPECT_EQ(store.symmetric_ram_bytes(), slice_bytes);
    EXPECT_EQ(sym, store.materialize_symmetric_arrays());  // still cached
    // Directional fetches are no longer serviceable.
    EXPECT_THROW((void)store.get_out_neighbors(oid(0)), std::logic_error);
    EXPECT_THROW((void)store.get_in_neighbors(oid(0)), std::logic_error);
}

// Releasing twice is harmless: the second call is a no-op returning 0.
TEST_F(ReleaseDirectionalTest, Idempotent) {
    write_sidecar(TopologySnapshotWriter::Direction::FORWARD, 3,
                  {{0, 1}, {1, 2}});
    write_sidecar(TopologySnapshotWriter::Direction::REVERSE, 3,
                  {{1, 0}, {2, 1}});
    auto rf = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::FORWARD);
    auto rr = TopologySnapshotReader::open(
        dir_, TopologySnapshotReader::Direction::REVERSE);
    ASSERT_TRUE(rf.has_data());
    ASSERT_TRUE(rr.has_data());

    const std::vector<uint8_t> tiers = {1, 1, 1};
    L1HashCache l1f(tiers), l1r(tiers);
    L2CompactCsr l2f, l2r;
    l2f.freeze();
    l2r.freeze();
    FourLevelTopologyStore::Config cfg;
    FourLevelTopologyStore store(
        l1f, l1r, l2f, l2r, &rf, &rr, {}, {},
        tiers, [](ObjectId v) { return v.id; }, cfg);

    ASSERT_NE(store.materialize_symmetric_arrays(), nullptr);
    (void)store.release_directional_after_symmetric_pin();
    EXPECT_TRUE(store.directional_released());
    EXPECT_EQ(store.release_directional_after_symmetric_pin(), 0u);  // 2nd no-op
    EXPECT_TRUE(store.directional_released());
}

// ---------------------------------------------------------------------------
// In-process RSS probes (process_memory.h).
// ---------------------------------------------------------------------------

TEST(ProcessMemory, CurrentAndPeakAreSaneOrZero) {
    const std::size_t cur = mdb::gnn::current_rss_bytes();
    const std::size_t peak = mdb::gnn::peak_rss_bytes();
    // On Linux with procfs both are > 0 and peak >= current. In a sandbox
    // without /proc both read 0; tolerate that (the probes are best-effort).
    if (cur > 0 || peak > 0) {
        EXPECT_GT(cur, 0u);
        EXPECT_GT(peak, 0u);
        EXPECT_GE(peak, cur);
    }
}

TEST(ProcessMemory, ResetPeakDoesNotThrowAndKeepsPeakSane) {
    // Allocate + touch ~32 MB so VmHWM is meaningfully above current after free.
    {
        std::vector<char> buf(32u * 1024u * 1024u, 1);
        EXPECT_GT(buf.size(), 0u);
    }
    const bool ok = mdb::gnn::reset_peak_rss();  // may be false in a sandbox
    (void)ok;
    const std::size_t cur = mdb::gnn::current_rss_bytes();
    const std::size_t peak = mdb::gnn::peak_rss_bytes();
    if (cur > 0 || peak > 0) {
        EXPECT_GE(peak, cur);  // peak is never below current
    }
}

}  // namespace
