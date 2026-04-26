// four_level_topology_warm_start_test.cc
//
// Spec #13 Phase 5 (T13.2 + T13.11) — warm-start activation tests.
//
// These tests cover the round-trip between the gnn_offline_sample-side
// `node_counts.bin` writer (in offline_sampling_engine.cc) and the
// FourLevelTopologyStore-side reader chain
// (TopologyFrequencyProfiler::compute_from_node_counts_ →
// FourLevelTopologyStore::compute_l3_minhash_reorder_).
//
// To stay independent of the full gnn_offline_sample procedure (HTTP
// server + GQL parser), the writer half is exercised by hand-crafting
// `node_counts.bin` files in the projection directory with the same
// 8-byte-magic + uint64 header that offline_sampling_engine writes.
// The actual writer's atomic-rename + fsync semantics are covered by
// the bench-script smoke documented in the parent prompt.
//
// Coverage:
//   1. ColdStart_NoNodeCountsFile — fresh projection, no warm-start
//      file. Profiler reports cold-start, frequency comes from degree.
//   2. WarmStart_RoundTrip — plant a valid node_counts.bin, confirm
//      profiler picks it up and frequency_ matches the file's payload.
//   3. WarmStart_RejectsCorruptMagic — bad magic header → cold-start
//      fallback, no throw.
//   4. WarmStart_RejectsStaleNumNodes — num_nodes mismatch → cold-start
//      fallback, no throw.
//   5. WarmStart_TierAssignmentDiffers — synthetic counts where a
//      low-degree node has very high access frequency. Tier assignment
//      promotes the high-frequency node to L1 even though degree-based
//      ranking would have left it in L3.
//   6. MinHashWarmStart_GeneratesPermutation — when warm_start_used is
//      true, FourLevelTopologyStore::build() populates a non-empty
//      l3_reorder_permutation_ with length == tier_lookup_.size().

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/four_level_topology_store.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/projection/topology_frequency_profiler.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace {

namespace fs = std::filesystem;

// Process-lifetime fixture (System + ProjectionManager singletons can
// only be bound once per process). Mirrors the convention used by
// topology_frequency_profiler_test.cc + topology_accessor_four_level_
// integration_test.cc.
class MdbFixture {
public:
    static MdbFixture& instance() {
        static MdbFixture f;
        return f;
    }

    const std::string& db_folder() const { return db_folder_; }

    ~MdbFixture() noexcept {
        try {
            fs::remove_all(db_folder_);
        } catch (...) {
            // best-effort
        }
    }

private:
    MdbFixture() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        db_folder_ = "test_db_four_level_warm_start_" + std::to_string(rng());
        fs::remove_all(db_folder_);

        system_.reset(new System(
            db_folder_,
            1024 * 1024,
            1024 * 1024,
            64 * 1024 * 1024,
            32 * 1024 * 1024,
            1024 * 1024,
            1024 * 1024,
            1
        ));

        query_ctx_.reset(new QueryContext());
        QueryContext::set_query_ctx(query_ctx_.get());

        auto& manager = GQL::ProjectionManager::get_instance();
        manager.init(db_folder_);
    }

    std::string                     db_folder_;
    std::unique_ptr<System>         system_;
    std::unique_ptr<QueryContext>   query_ctx_;
};

// 6-node fixture, mirrors the asymmetric-degree shape used by
// topology_frequency_profiler_test.cc:
//
//   0 -> 1, 0 -> 2, 0 -> 3   (out=3, in=0)
//   1 -> 2                   (out=1, in=1)
//   2 -> 3                   (out=1, in=2)
//   4 -> 0                   (out=1, in=1)
//   (node 3: out=0, in=2; node 5: isolated)
struct FixtureGraph {
    static constexpr uint64_t kNumNodes = 6;
    static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& edges() {
        static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> E = {
            {0, 1, 100},
            {0, 2, 101},
            {0, 3, 102},
            {1, 2, 103},
            {2, 3, 104},
            {4, 0, 105},
        };
        return E;
    }
    static std::vector<uint64_t> out_degrees() { return {3, 1, 1, 0, 1, 0}; }
};

struct BuiltFixture {
    std::unique_ptr<GQL::ProjectionStorage> storage;
    fs::path projection_dir;
};

BuiltFixture build_fixture(const std::string& projection_name) {
    auto& manager = GQL::ProjectionManager::get_instance();
    std::string proj_dir = manager.create_projection(projection_name);

    auto storage = std::make_unique<GQL::ProjectionStorage>(
        proj_dir,
        MdbFixture::instance().db_folder(),
        projection_name);
    storage->init();

    for (uint64_t i = 0; i < FixtureGraph::kNumNodes; ++i) {
        GQL::ProjectedNode node;
        node.node_id = ObjectId(i);
        storage->add_node(node);
    }
    for (const auto& [from, to, eid] : FixtureGraph::edges()) {
        GQL::ProjectedEdge edge;
        edge.from_node   = ObjectId(from);
        edge.to_node     = ObjectId(to);
        edge.edge_id     = ObjectId(eid);
        edge.is_directed = true;
        storage->add_edge(edge);
    }
    storage->flush();

    BuiltFixture out;
    out.storage = std::move(storage);
    out.projection_dir = fs::path(proj_dir);
    return out;
}

constexpr uint8_t kNodeCountsMagic[8] = {'N','O','D','E','C','N','T','0'};

// Plant a valid `node_counts.bin` with the given counts vector.
// `direction_bitmask` defaults to UNDIRECTED (=3).
void plant_node_counts(const fs::path& projection_dir,
                       const std::vector<uint64_t>& counts,
                       uint64_t direction_bitmask = 3)
{
    fs::create_directories(projection_dir);
    std::ofstream f(projection_dir / "node_counts.bin", std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(kNodeCountsMagic), 8);
    const uint64_t num_nodes = counts.size();
    f.write(reinterpret_cast<const char*>(&num_nodes),         sizeof(num_nodes));
    f.write(reinterpret_cast<const char*>(&direction_bitmask), sizeof(direction_bitmask));
    if (!counts.empty()) {
        f.write(reinterpret_cast<const char*>(counts.data()),
                static_cast<std::streamsize>(num_nodes * sizeof(uint64_t)));
    }
}

void plant_corrupt_magic(const fs::path& projection_dir) {
    fs::create_directories(projection_dir);
    std::ofstream f(projection_dir / "node_counts.bin", std::ios::binary | std::ios::trunc);
    const char garbage[16] = "BADMAGIC";
    f.write(garbage, 16);
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — ColdStart_NoNodeCountsFile.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyWarmStart, ColdStart_NoNodeCountsFile) {
    (void)MdbFixture::instance();
    auto fx = build_fixture("warm_start_cold_start");

    // No node_counts.bin planted — the fixture path is fresh.
    ASSERT_FALSE(fs::exists(fx.projection_dir / "node_counts.bin"));

    mdb::gnn::TopologyAccessor acc(*fx.storage);
    mdb::gnn::TopologyFrequencyProfiler profiler(acc, fx.projection_dir);
    profiler.compute(mdb::gnn::EdgeOrientation::NATURAL);

    EXPECT_FALSE(profiler.warm_start_used());

    // Cold-start frequency vector matches degree.
    const auto& freq   = profiler.frequency();
    const auto  expect = FixtureGraph::out_degrees();
    ASSERT_EQ(freq.size(), expect.size());
    for (std::size_t i = 0; i < expect.size(); ++i) {
        EXPECT_EQ(freq[i], expect[i]) << "node " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 2 — WarmStart_RoundTrip.
//
// Plant a `node_counts.bin` with handcrafted per-node counts, then verify
// the profiler sees them exactly. Use values that are unmistakably distinct
// from any degree-based vector so the assertion can't be passed by the
// cold-start fallback.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyWarmStart, WarmStart_RoundTrip) {
    (void)MdbFixture::instance();
    auto fx = build_fixture("warm_start_round_trip");

    // Counts are intentionally NOT the degree vector so we can tell the
    // two code paths apart in the assertion below.
    const std::vector<uint64_t> counts = {500, 200, 100, 50, 25, 10};
    ASSERT_EQ(counts.size(), FixtureGraph::kNumNodes);
    plant_node_counts(fx.projection_dir, counts);

    mdb::gnn::TopologyAccessor acc(*fx.storage);
    mdb::gnn::TopologyFrequencyProfiler profiler(acc, fx.projection_dir);
    profiler.compute(mdb::gnn::EdgeOrientation::NATURAL);

    EXPECT_TRUE(profiler.warm_start_used());
    const auto& freq = profiler.frequency();
    ASSERT_EQ(freq.size(), counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        EXPECT_EQ(freq[i], counts[i]) << "node " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 3 — WarmStart_RejectsCorruptMagic.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyWarmStart, WarmStart_RejectsCorruptMagic) {
    (void)MdbFixture::instance();
    auto fx = build_fixture("warm_start_corrupt_magic");

    plant_corrupt_magic(fx.projection_dir);
    ASSERT_TRUE(fs::exists(fx.projection_dir / "node_counts.bin"));

    mdb::gnn::TopologyAccessor acc(*fx.storage);
    mdb::gnn::TopologyFrequencyProfiler profiler(acc, fx.projection_dir);
    EXPECT_NO_THROW(profiler.compute(mdb::gnn::EdgeOrientation::NATURAL));

    // Cold-start fallback engaged.
    EXPECT_FALSE(profiler.warm_start_used());

    const auto& freq   = profiler.frequency();
    const auto  expect = FixtureGraph::out_degrees();
    ASSERT_EQ(freq.size(), expect.size());
    for (std::size_t i = 0; i < expect.size(); ++i) {
        EXPECT_EQ(freq[i], expect[i]) << "node " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 4 — WarmStart_RejectsStaleNumNodes.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyWarmStart, WarmStart_RejectsStaleNumNodes) {
    (void)MdbFixture::instance();
    auto fx = build_fixture("warm_start_stale_num_nodes");

    // Plant counts whose num_nodes (3) doesn't match the projection's 6.
    const std::vector<uint64_t> stale_counts = {1, 2, 3};
    plant_node_counts(fx.projection_dir, stale_counts);

    mdb::gnn::TopologyAccessor acc(*fx.storage);
    mdb::gnn::TopologyFrequencyProfiler profiler(acc, fx.projection_dir);
    EXPECT_NO_THROW(profiler.compute(mdb::gnn::EdgeOrientation::NATURAL));

    // Cold-start fallback engaged.
    EXPECT_FALSE(profiler.warm_start_used());

    const auto& freq   = profiler.frequency();
    const auto  expect = FixtureGraph::out_degrees();
    ASSERT_EQ(freq.size(), expect.size());
}

// ---------------------------------------------------------------------------
// Test 5 — WarmStart_TierAssignmentDiffers.
//
// Synthetic frequency where a node with low degree has very high access
// counts (e.g., a popular hub that sampling visits frequently), and a
// node with high degree is never sampled. Tier assignment must follow
// the warm-start frequencies, not the degree vector.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyWarmStart, WarmStart_TierAssignmentDiffers) {
    // Use the helper directly — no projection needed.
    //
    // Node 3 has degree 0 (per FixtureGraph::out_degrees()) but a huge
    // synthetic access count of 1000. Node 0 has degree 3 but 0
    // accesses. Tier assignment should rank 3 first.
    const std::vector<uint64_t> warm_freq = {0, 5, 10, 1000, 50, 0};

    // Tight L1 budget: only the top-1 node fits. Avg degree = 0.5 →
    // l1_per_node = 0.5*16 + 56 = 64 bytes. l1_budget = 64 ⇒ exactly 1.
    auto tiers_warm = mdb::gnn::compute_tier_assignment(
        warm_freq, /*l1=*/64, /*l2=*/0, /*avg_degree=*/0.5);
    ASSERT_EQ(tiers_warm.size(), warm_freq.size());

    // Node 3 (max frequency = 1000) lands in tier 1.
    EXPECT_EQ(tiers_warm[3], 1u);
    // Node 0 (degree 3, frequency 0) does NOT land in tier 1.
    EXPECT_NE(tiers_warm[0], 1u);

    // Cold-start (degree-based) ranking would land node 0 in tier 1.
    auto degree_freq = FixtureGraph::out_degrees();
    auto tiers_cold = mdb::gnn::compute_tier_assignment(
        degree_freq, /*l1=*/64, /*l2=*/0, /*avg_degree=*/0.5);
    EXPECT_EQ(tiers_cold[0], 1u);
}

// ---------------------------------------------------------------------------
// Test 6 — MinHashWarmStart_GeneratesPermutation.
//
// End-to-end smoke: build a larger projection (≥ ~1000 nodes) so the
// 1 MiB L1 budget is too tight to absorb every node; that forces a
// non-empty L3 tier set in the production build path. With the
// synthetic node_counts.bin planted, `compute_l3_minhash_reorder_`
// must therefore return a non-empty permutation whose length equals
// the projection's row count.
// ---------------------------------------------------------------------------
TEST(FourLevelTopologyWarmStart, MinHashWarmStart_GeneratesPermutation) {
    (void)MdbFixture::instance();

    // Build a larger synthetic projection with ~2000 nodes and a chain
    // of edges (each node points to the next). With l1_budget=1 MiB
    // and avg_degree≈1, l1_per_node ≈ 16 + 56 = 72 B → ~14k nodes
    // fit. We need l1 cost much higher per node: synthetic counts
    // skewed so avg_degree (used as the per-edge cost driver in the
    // tier sizing model) is ~5000, pushing l1_per_node ≈ 80 KB and
    // forcing every node past the first 13 into L3.
    constexpr uint64_t kN = 2000;
    auto& manager = GQL::ProjectionManager::get_instance();
    std::string proj_name = "warm_start_minhash_big";
    std::string proj_dir = manager.create_projection(proj_name);
    auto storage = std::make_unique<GQL::ProjectionStorage>(
        proj_dir, MdbFixture::instance().db_folder(), proj_name);
    storage->init();
    for (uint64_t i = 0; i < kN; ++i) {
        GQL::ProjectedNode node;
        node.node_id = ObjectId(i);
        storage->add_node(node);
    }
    for (uint64_t i = 0; i + 1 < kN; ++i) {
        GQL::ProjectedEdge edge;
        edge.from_node   = ObjectId(i);
        edge.to_node     = ObjectId(i + 1);
        edge.edge_id     = ObjectId(1000000 + i);
        edge.is_directed = true;
        storage->add_edge(edge);
    }
    storage->flush();

    // Plant high-magnitude counts so avg_degree (= mean(counts)) is
    // large enough that the per-node cost overflows a 1 MiB budget
    // for most nodes. A few high-frequency nodes dominate the average
    // — typical real-world long-tail.
    std::vector<uint64_t> counts(kN, 0);
    for (uint64_t i = 0; i < 10; ++i)   counts[i] = 1000000;  // top 10
    for (uint64_t i = 10; i < kN; ++i)  counts[i] = (kN - i); // tail
    plant_node_counts(fs::path(proj_dir), counts);

    // Drive the production build with tight L1 + L2 budgets, forcing
    // the tail of nodes into L3. NOTE: a budget of 0 means "auto-detect
    // from MemAvailable" (not "0 bytes"), so we must use 1 MiB to opt
    // out of the auto-detection path.
    mdb::gnn::FourLevelTopologyStore::Config cfg;
    cfg.l1_budget_mb        = 1;
    cfg.l2_budget_mb        = 1;
    cfg.use_l3_mmap_sidecar = false;
    cfg.orientation         = mdb::gnn::EdgeOrientation::NATURAL;

    mdb::gnn::FourLevelTopologyStore store(
        storage->get_from_to_edge_index(),
        storage->get_to_from_edge_index(),
        storage.get(),
        fs::path(proj_dir),
        cfg);
    store.build();

    const auto& perm = store.l3_reorder_permutation();
    EXPECT_FALSE(perm.empty()) << "Expected non-empty permutation under "
                                  "warm start with L3 tier non-empty";
    EXPECT_EQ(perm.size(), kN) << "Permutation must span every row";

    // Every row appears exactly once.
    std::vector<uint64_t> seen = perm;
    std::sort(seen.begin(), seen.end());
    for (std::size_t i = 0; i < seen.size(); ++i) {
        EXPECT_EQ(seen[i], i) << "permutation missing or duplicate at " << i;
    }
}
