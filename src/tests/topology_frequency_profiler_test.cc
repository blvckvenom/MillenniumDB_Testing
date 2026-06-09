// topology_frequency_profiler_test.cc
//
// Spec #13 Phase 1 — TopologyFrequencyProfiler unit tests.
//
// Coverage:
//   1. ColdStart_FallsBackToDegree — without `node_counts.bin`, the cold
//      path emits a frequency vector that matches per-node degree.
//   2. WarmStart_StubbedReturnsFalse — even with a fake `node_counts.bin`
//      present in the projection directory, Phase 1's stubbed reader
//      MUST report `warm_start_used() == false` and fall through to the
//      cold path. Sanity-checks that no Phase 2 code accidentally landed.
//   3. MixedOrientation_DegreesCombined — for a graph with asymmetric
//      in/out degrees, NATURAL → out, REVERSE → in,
//      UNDIRECTED → out + in.
//   4. TierAssignment_RespectsBudgets — exercises `compute_tier_assignment`
//      with controlled frequency + budgets: highest-frequency nodes pack
//      into tier 1, next into tier 2, rest into tier 3. Includes the
//      "budget covers the graph → all tier 1" edge case.
//   5. ColdStart_TaggedNodeIds_ProfilesRealDegrees — same as (1)/(3) but
//      over a projection whose node ObjectIds carry the production type
//      tag (`ObjectId::MASK_NODE`, 0xD4...), so the edge B+Tree keys are
//      tagged uint64s. Guards against the cold path ranging the B+Tree
//      with bare ordinals (which matches nothing and silently profiles
//      every node as degree 0).
//
// Mirrors the fixture style of topology_accessor_adjacency_cache_test.cc
// so both suites can share the process-lifetime System singleton when run
// in the same process.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/topology_accessor.h"
#include "gnn/projection/topology_frequency_profiler.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace {

namespace fs = std::filesystem;

// Process-lifetime fixture (System + ProjectionManager singletons can only
// be bound once per process).
class MdbFixture {
public:
    static MdbFixture& instance() {
        static MdbFixture f;
        return f;
    }

    const std::string& db_folder() const { return db_folder_; }

    // Cleanup the per-process scratch directory on teardown so repeated
    // ctest runs do not leak `test_db_topo_freq_profiler_*` directories
    // into the working tree. `noexcept` because filesystem ops can throw
    // and a destructor must not propagate.
    ~MdbFixture() noexcept {
        try {
            fs::remove_all(db_folder_);
        } catch (...) {
            // Swallow — best-effort cleanup; nothing the test can do.
        }
    }

private:
    MdbFixture() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        db_folder_ = "test_db_topo_freq_profiler_" + std::to_string(rng());
        fs::remove_all(db_folder_);

        system_.reset(new System(
            db_folder_,
            1024 * 1024,        // str_static_size
            1024 * 1024,        // str_dynamic_size
            64 * 1024 * 1024,   // shared_buffer_size
            32 * 1024 * 1024,   // private_buffer_size
            1024 * 1024,        // tensor_static_size
            1024 * 1024,        // tensor_dynamic_size
            1                   // workers
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

// Asymmetric fixture so NATURAL / REVERSE / UNDIRECTED produce visibly
// different frequency vectors:
//
//   0 -> 1, 0 -> 2, 0 -> 3   (out=3, in=0)
//   1 -> 2                   (out=1, in=1)
//   2 -> 3                   (out=1, in=2)
//   4 -> 0                   (out=1, in=1)
//   (node 3 has out=0, in=2; node 5 isolated)
struct AsymGraph {
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
    // Degrees (computed by hand; mirror in the test).
    static std::vector<uint64_t> out_degrees() { return {3, 1, 1, 0, 1, 0}; }
    static std::vector<uint64_t> in_degrees()  { return {1, 1, 2, 2, 0, 0}; }
};

// Builds a projection populated with `AsymGraph`. Returns the storage
// (caller owns lifetime) and the absolute path to the projection dir so
// the test can plant a fake `node_counts.bin` there.
struct BuiltFixture {
    std::unique_ptr<GQL::ProjectionStorage> storage;
    fs::path projection_dir;
};

// `node_id_tag` / `edge_id_tag` let a test mirror the production id shape
// (projected nodes carry `ObjectId::MASK_NODE` in the top 8 bits, edges
// `ObjectId::MASK_DIRECTED_EDGE`); the defaults preserve the original
// untagged synthetic shape.
BuiltFixture build_asym_storage(const std::string& projection_name,
                                uint64_t node_id_tag = 0,
                                uint64_t edge_id_tag = 0) {
    auto& manager = GQL::ProjectionManager::get_instance();
    std::string proj_dir = manager.create_projection(projection_name);

    auto storage = std::make_unique<GQL::ProjectionStorage>(
        proj_dir,
        MdbFixture::instance().db_folder(),
        projection_name);
    storage->init();

    for (uint64_t i = 0; i < AsymGraph::kNumNodes; ++i) {
        GQL::ProjectedNode node;
        node.node_id = ObjectId(node_id_tag | i);
        storage->add_node(node);
    }
    for (const auto& [from, to, eid] : AsymGraph::edges()) {
        GQL::ProjectedEdge edge;
        edge.from_node   = ObjectId(node_id_tag | from);
        edge.to_node     = ObjectId(node_id_tag | to);
        edge.edge_id     = ObjectId(edge_id_tag | eid);
        edge.is_directed = true;
        storage->add_edge(edge);
    }
    storage->flush();

    BuiltFixture out;
    out.storage = std::move(storage);
    out.projection_dir = fs::path(proj_dir);
    return out;
}

// Plants a non-empty file at `<projection_dir>/node_counts.bin` so we can
// confirm that Phase 1's stub does NOT consume it (Phase 2 will).
void plant_fake_node_counts(const fs::path& projection_dir) {
    fs::create_directories(projection_dir);
    std::ofstream out(projection_dir / "node_counts.bin", std::ios::binary);
    const char garbage[] = "FAKE_NODE_COUNTS_BIN_PHASE1";
    out.write(garbage, sizeof(garbage));
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — ColdStart_FallsBackToDegree.
// ---------------------------------------------------------------------------
TEST(TopologyFrequencyProfiler, ColdStart_FallsBackToDegree) {
    (void)MdbFixture::instance();
    auto fx = build_asym_storage("freq_profiler_cold_start");

    mdb::gnn::TopologyAccessor acc(*fx.storage);
    mdb::gnn::TopologyFrequencyProfiler profiler(acc, fx.projection_dir);

    profiler.compute(mdb::gnn::EdgeOrientation::NATURAL);

    EXPECT_FALSE(profiler.warm_start_used());

    const auto& freq    = profiler.frequency();
    const auto  expect  = AsymGraph::out_degrees();
    ASSERT_EQ(freq.size(), expect.size());
    for (std::size_t i = 0; i < expect.size(); ++i) {
        EXPECT_EQ(freq[i], expect[i]) << "node " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 2 — WarmStart_StubbedReturnsFalse.
//
// Even with a fake node_counts.bin sitting in the projection directory,
// the Phase 1 stub MUST not claim warm-start. If a future Phase 2 patch
// makes this start returning true accidentally, this test will catch it.
// ---------------------------------------------------------------------------
TEST(TopologyFrequencyProfiler, WarmStart_StubbedReturnsFalse) {
    (void)MdbFixture::instance();
    auto fx = build_asym_storage("freq_profiler_warm_start_stubbed");

    plant_fake_node_counts(fx.projection_dir);
    ASSERT_TRUE(fs::exists(fx.projection_dir / "node_counts.bin"));

    mdb::gnn::TopologyAccessor acc(*fx.storage);
    mdb::gnn::TopologyFrequencyProfiler profiler(acc, fx.projection_dir);

    profiler.compute(mdb::gnn::EdgeOrientation::NATURAL);

    EXPECT_FALSE(profiler.warm_start_used());

    // Fall-through path must still produce a valid degree vector.
    const auto& freq   = profiler.frequency();
    const auto  expect = AsymGraph::out_degrees();
    ASSERT_EQ(freq.size(), expect.size());
    for (std::size_t i = 0; i < expect.size(); ++i) {
        EXPECT_EQ(freq[i], expect[i]) << "node " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 3 — MixedOrientation_DegreesCombined.
// ---------------------------------------------------------------------------
TEST(TopologyFrequencyProfiler, MixedOrientation_DegreesCombined) {
    (void)MdbFixture::instance();
    auto fx = build_asym_storage("freq_profiler_mixed_orientation");

    mdb::gnn::TopologyAccessor acc(*fx.storage);
    mdb::gnn::TopologyFrequencyProfiler profiler(acc, fx.projection_dir);

    const auto out_d = AsymGraph::out_degrees();
    const auto in_d  = AsymGraph::in_degrees();

    // NATURAL -> out_degree
    profiler.compute(mdb::gnn::EdgeOrientation::NATURAL);
    {
        const auto& f = profiler.frequency();
        ASSERT_EQ(f.size(), out_d.size());
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_EQ(f[i], out_d[i]) << "NATURAL node " << i;
        }
    }

    // REVERSE -> in_degree
    profiler.compute(mdb::gnn::EdgeOrientation::REVERSE);
    {
        const auto& f = profiler.frequency();
        ASSERT_EQ(f.size(), in_d.size());
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_EQ(f[i], in_d[i]) << "REVERSE node " << i;
        }
    }

    // UNDIRECTED -> out + in
    profiler.compute(mdb::gnn::EdgeOrientation::UNDIRECTED);
    {
        const auto& f = profiler.frequency();
        ASSERT_EQ(f.size(), out_d.size());
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_EQ(f[i], out_d[i] + in_d[i]) << "UNDIRECTED node " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4 — TierAssignment_RespectsBudgets.
//
// Drives compute_tier_assignment directly with synthetic data so the unit
// test is independent of any real projection. Two scenarios:
//
//   (a) Tight budgets: only the top-2 frequency nodes fit in L1, next 2
//       fit in L2, rest spill to L3.
//   (b) Generous budgets: total cache capacity exceeds the graph's
//       notional cost → every node lands in tier 1.
// ---------------------------------------------------------------------------
TEST(TopologyFrequencyProfiler, TierAssignment_RespectsBudgets) {
    // 6 nodes, frequencies chosen so the descending-sorted order is
    // unambiguous: index 4 > 0 > 2 > 1 > 3 > 5.
    const std::vector<uint64_t> frequency = {50, 10, 30, 5, 99, 0};

    // For avg_degree = 4.0:
    //   l1_per_node = 4 * 16 + 56 = 120 bytes
    //   l2_per_node = 4 * 8  + 8  =  40 bytes
    const double avg_degree = 4.0;

    // (a) Tight budgets — fit exactly 2 nodes in L1 and 2 in L2.
    const std::size_t l1_budget = 240;   // 2 * 120
    const std::size_t l2_budget = 80;    // 2 * 40

    auto tiers = mdb::gnn::compute_tier_assignment(
        frequency, l1_budget, l2_budget, avg_degree);

    ASSERT_EQ(tiers.size(), frequency.size());

    // Top-2 by frequency: indexes 4 (99) and 0 (50) → tier 1.
    EXPECT_EQ(tiers[4], 1u);
    EXPECT_EQ(tiers[0], 1u);

    // Next-2 by frequency: indexes 2 (30) and 1 (10) → tier 2.
    EXPECT_EQ(tiers[2], 2u);
    EXPECT_EQ(tiers[1], 2u);

    // Remaining: indexes 3 (5) and 5 (0) → tier 3.
    EXPECT_EQ(tiers[3], 3u);
    EXPECT_EQ(tiers[5], 3u);

    // (b) Generous budgets — every node fits in L1.
    auto tiers_all_l1 = mdb::gnn::compute_tier_assignment(
        frequency,
        /*l1_budget_bytes=*/ 1ULL << 30,
        /*l2_budget_bytes=*/ 1ULL << 30,
        avg_degree);
    ASSERT_EQ(tiers_all_l1.size(), frequency.size());
    for (std::size_t i = 0; i < tiers_all_l1.size(); ++i) {
        EXPECT_EQ(tiers_all_l1[i], 1u) << "all-L1 node " << i;
    }

    // (c) Zero budgets — every node falls to tier 3.
    auto tiers_all_l3 = mdb::gnn::compute_tier_assignment(
        frequency, /*l1=*/0, /*l2=*/0, avg_degree);
    ASSERT_EQ(tiers_all_l3.size(), frequency.size());
    for (std::size_t i = 0; i < tiers_all_l3.size(); ++i) {
        EXPECT_EQ(tiers_all_l3[i], 3u) << "all-L3 node " << i;
    }

    // (d) Empty frequency vector → empty result.
    auto tiers_empty = mdb::gnn::compute_tier_assignment({}, 1024, 1024, 4.0);
    EXPECT_TRUE(tiers_empty.empty());
}

// ---------------------------------------------------------------------------
// Test 5 — ColdStart_TaggedNodeIds_ProfilesRealDegrees.
//
// Production projections store node ObjectIds WITH the 8-bit type tag
// (`ObjectId::MASK_NODE` = 0xD4ULL << 56 | ordinal), so the projection's
// edge B+Tree keys are tagged uint64s. The cold-start degree pass must
// range those B+Trees with ids that carry the tag; querying bare ordinals
// matches no key and silently yields an all-zero frequency vector (and
// thus a blind tier assignment that accounts only fixed per-node overhead).
// The frequency vector must stay indexed by dense row ordinal — i.e. the
// tag-stripped value — matching the untagged expectations of Test 1/3.
// ---------------------------------------------------------------------------
TEST(TopologyFrequencyProfiler, ColdStart_TaggedNodeIds_ProfilesRealDegrees) {
    (void)MdbFixture::instance();
    auto fx = build_asym_storage("freq_profiler_tagged_cold_start",
                                 ObjectId::MASK_NODE,
                                 ObjectId::MASK_DIRECTED_EDGE);

    mdb::gnn::TopologyAccessor acc(*fx.storage);
    mdb::gnn::TopologyFrequencyProfiler profiler(acc, fx.projection_dir);

    const auto out_d = AsymGraph::out_degrees();
    const auto in_d  = AsymGraph::in_degrees();

    // NATURAL -> out_degree
    profiler.compute(mdb::gnn::EdgeOrientation::NATURAL);
    EXPECT_FALSE(profiler.warm_start_used());
    {
        const auto& f = profiler.frequency();
        ASSERT_EQ(f.size(), out_d.size());
        uint64_t total = 0;
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_EQ(f[i], out_d[i]) << "NATURAL tagged node " << i;
            total += f[i];
        }
        // The core regression check: a tagged projection with edges must
        // never profile as all-zero.
        EXPECT_GT(total, 0u);
    }

    // REVERSE -> in_degree
    profiler.compute(mdb::gnn::EdgeOrientation::REVERSE);
    {
        const auto& f = profiler.frequency();
        ASSERT_EQ(f.size(), in_d.size());
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_EQ(f[i], in_d[i]) << "REVERSE tagged node " << i;
        }
    }

    // UNDIRECTED -> out + in
    profiler.compute(mdb::gnn::EdgeOrientation::UNDIRECTED);
    {
        const auto& f = profiler.frequency();
        ASSERT_EQ(f.size(), out_d.size());
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_EQ(f[i], out_d[i] + in_d[i]) << "UNDIRECTED tagged node " << i;
        }
    }
}
