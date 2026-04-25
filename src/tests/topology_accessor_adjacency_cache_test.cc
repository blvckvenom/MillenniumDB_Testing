// topology_accessor_adjacency_cache_test.cc
//
// Spec #11 unit tests — the in-memory adjacency cache mode of
// `mdb::gnn::TopologyAccessor`. The cache is opt-in (default off) and is
// expected to return bit-identical neighbour sets compared to the B+Tree
// path under a fixed RNG seed. Coverage:
//
//   1. Default state: cache disabled, no entries, no resident bytes.
//   2. enable_adjacency_cache(true) without prebuild: subsequent
//      get_*_neighbors falls back to BPT until prebuild_adjacency_cache().
//   3. After prebuild_adjacency_cache(NATURAL): get_out_neighbors returns
//      the same set as a BPT-only accessor for every node.
//   4. After prebuild_adjacency_cache(REVERSE): get_in_neighbors matches.
//   5. After prebuild_adjacency_cache(UNDIRECTED): get_neighbors(UNDIRECTED)
//      matches, both directions are reported as built, and entry counts
//      match the fixture's directed edge total.
//   6. enable_adjacency_cache(false): cache cleared, get_*_neighbors falls
//      back to BPT and returns the same data again.
//   7. K-hop sampling parity: BasicKHopSampler-equivalent loop with a
//      fixed RNG seed produces bit-identical neighbour multisets across
//      cache vs no-cache paths.
//   8. Isolated nodes (no edges) → empty Neighbors via cache, no crash.
//
// Mirrors the fixture style of topology_accessor_csr_path_test.cc so the
// process-lifetime System singleton is shared by both suites cleanly.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/topology_accessor.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

namespace {

namespace fs = std::filesystem;

// Process-lifetime fixture (System + ProjectionManager singletons can only
// be bound once per process). Uses a unique randomly-named db folder so the
// test never collides with a previous run.
class MdbFixture {
public:
    static MdbFixture& instance() {
        static MdbFixture f;
        return f;
    }

    const std::string& db_folder() const { return db_folder_; }

private:
    MdbFixture() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        db_folder_ = "test_db_topo_acc_adj_cache_" + std::to_string(rng());
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

// Modest fixture: 6 nodes, 8 directed edges, includes one isolated node (5)
// to exercise the absent-key path of the cache. Edge ids start at 200 to
// distinguish them from node ids in any failure log.
//
//   0 -> 1 (e=200)
//   0 -> 2 (e=201)
//   1 -> 3 (e=202)
//   2 -> 3 (e=203)
//   2 -> 4 (e=204)
//   3 -> 4 (e=205)
//   4 -> 0 (e=206)   -- back-edge
//   1 -> 4 (e=207)
//   (node 5 is isolated)
struct FixtureGraph {
    static constexpr uint64_t kNumNodes = 6;
    static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& edges() {
        static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> E = {
            {0, 1, 200},
            {0, 2, 201},
            {1, 3, 202},
            {2, 3, 203},
            {2, 4, 204},
            {3, 4, 205},
            {4, 0, 206},
            {1, 4, 207},
        };
        return E;
    }
};

std::unique_ptr<GQL::ProjectionStorage> build_fixture_storage(
    const std::string& projection_name)
{
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
    return storage;
}

std::vector<std::pair<uint64_t, uint64_t>> neighbors_as_sorted_pairs(
    const mdb::gnn::Neighbors& n)
{
    std::vector<std::pair<uint64_t, uint64_t>> out;
    out.reserve(n.node_ids.size());
    for (std::size_t i = 0; i < n.node_ids.size(); ++i) {
        const uint64_t eid = (i < n.edge_ids.size()) ? n.edge_ids[i].id : 0ULL;
        out.emplace_back(n.node_ids[i].id, eid);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Sampling helper (mirrors BasicKHopSampler::Impl::sample_neighbors_uniform
// without depending on the sampler library). Single-layer k-hop with a
// fixed RNG seed, returns the sorted set of selected neighbour ids per
// seed for parity comparison.
std::vector<std::vector<uint64_t>> sample_layer_under_seed(
    mdb::gnn::TopologyAccessor& acc,
    const std::vector<ObjectId>& seeds,
    std::size_t fanout,
    uint64_t seed,
    mdb::gnn::EdgeOrientation orientation)
{
    std::mt19937_64 rng(seed);
    std::vector<std::vector<uint64_t>> per_seed;
    per_seed.reserve(seeds.size());
    for (const auto& s : seeds) {
        auto n = acc.get_neighbors(s, orientation);
        const std::size_t total = n.node_ids.size();
        const std::size_t k     = std::min(fanout, total);
        std::vector<std::size_t> idx(total);
        std::iota(idx.begin(), idx.end(), 0);
        for (std::size_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<std::size_t> dist(i, total - 1);
            std::size_t j = dist(rng);
            std::swap(idx[i], idx[j]);
        }
        std::vector<uint64_t> selected;
        selected.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            selected.push_back(n.node_ids[idx[i]].id);
        }
        std::sort(selected.begin(), selected.end());
        per_seed.push_back(std::move(selected));
    }
    return per_seed;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — Default state: cache is disabled, reports zero entries and bytes.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorAdjacencyCache, DisabledByDefault) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("adj_cache_default_off");
    mdb::gnn::TopologyAccessor acc(*storage);

    EXPECT_FALSE(acc.is_adjacency_cache_enabled());
    EXPECT_FALSE(acc.is_adjacency_cache_built(mdb::gnn::EdgeOrientation::NATURAL));
    EXPECT_FALSE(acc.is_adjacency_cache_built(mdb::gnn::EdgeOrientation::REVERSE));
    EXPECT_FALSE(acc.is_adjacency_cache_built(mdb::gnn::EdgeOrientation::UNDIRECTED));
    EXPECT_EQ(0u, acc.get_adjacency_cache_size_bytes());
    EXPECT_EQ(0u, acc.get_adjacency_cache_fwd_entries());
    EXPECT_EQ(0u, acc.get_adjacency_cache_rev_entries());
}

// ---------------------------------------------------------------------------
// Test 2 — enable() without prebuild: cache reports enabled but not built;
// get_*_neighbors should silently fall back to the BPT path so callers see
// no behaviour change.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorAdjacencyCache, EnabledWithoutPrebuildFallsBack) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("adj_cache_enable_no_prebuild");

    mdb::gnn::TopologyAccessor acc_cache(*storage);
    acc_cache.enable_adjacency_cache(true);
    EXPECT_TRUE(acc_cache.is_adjacency_cache_enabled());
    EXPECT_FALSE(acc_cache.is_adjacency_cache_built(mdb::gnn::EdgeOrientation::NATURAL));

    mdb::gnn::TopologyAccessor acc_bpt(*storage);  // no cache

    for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
        auto via_cache = acc_cache.get_out_neighbors(ObjectId(nid));
        auto via_bpt   = acc_bpt.get_out_neighbors(ObjectId(nid));
        EXPECT_EQ(neighbors_as_sorted_pairs(via_cache),
                  neighbors_as_sorted_pairs(via_bpt))
            << "node " << nid;
    }
}

// ---------------------------------------------------------------------------
// Test 3 — Prebuild NATURAL: cached out-neighbours match the BPT path
// for every fixture node.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorAdjacencyCache, PrebuildNaturalMatchesBpt) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("adj_cache_prebuild_natural");

    mdb::gnn::TopologyAccessor acc_cache(*storage);
    mdb::gnn::TopologyAccessor acc_bpt(*storage);

    acc_cache.enable_adjacency_cache(true);
    acc_cache.prebuild_adjacency_cache(mdb::gnn::EdgeOrientation::NATURAL);

    EXPECT_TRUE(acc_cache.is_adjacency_cache_built(
        mdb::gnn::EdgeOrientation::NATURAL));
    EXPECT_FALSE(acc_cache.is_adjacency_cache_built(
        mdb::gnn::EdgeOrientation::REVERSE));
    EXPECT_EQ(FixtureGraph::edges().size(),
              acc_cache.get_adjacency_cache_fwd_entries());
    EXPECT_EQ(0u, acc_cache.get_adjacency_cache_rev_entries());

    for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
        auto via_cache = acc_cache.get_out_neighbors(ObjectId(nid));
        auto via_bpt   = acc_bpt.get_out_neighbors(ObjectId(nid));
        EXPECT_EQ(neighbors_as_sorted_pairs(via_cache),
                  neighbors_as_sorted_pairs(via_bpt))
            << "node " << nid;
    }
}

// ---------------------------------------------------------------------------
// Test 4 — Prebuild REVERSE: cached in-neighbours match the BPT path.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorAdjacencyCache, PrebuildReverseMatchesBpt) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("adj_cache_prebuild_reverse");

    mdb::gnn::TopologyAccessor acc_cache(*storage);
    mdb::gnn::TopologyAccessor acc_bpt(*storage);

    acc_cache.enable_adjacency_cache(true);
    acc_cache.prebuild_adjacency_cache(mdb::gnn::EdgeOrientation::REVERSE);

    EXPECT_TRUE(acc_cache.is_adjacency_cache_built(
        mdb::gnn::EdgeOrientation::REVERSE));
    EXPECT_FALSE(acc_cache.is_adjacency_cache_built(
        mdb::gnn::EdgeOrientation::NATURAL));
    EXPECT_EQ(0u, acc_cache.get_adjacency_cache_fwd_entries());
    EXPECT_EQ(FixtureGraph::edges().size(),
              acc_cache.get_adjacency_cache_rev_entries());

    for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
        auto via_cache = acc_cache.get_in_neighbors(ObjectId(nid));
        auto via_bpt   = acc_bpt.get_in_neighbors(ObjectId(nid));
        EXPECT_EQ(neighbors_as_sorted_pairs(via_cache),
                  neighbors_as_sorted_pairs(via_bpt))
            << "node " << nid;
    }
}

// ---------------------------------------------------------------------------
// Test 5 — Prebuild UNDIRECTED: cached union matches the BPT path.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorAdjacencyCache, PrebuildUndirectedMatchesBpt) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("adj_cache_prebuild_undirected");

    mdb::gnn::TopologyAccessor acc_cache(*storage);
    mdb::gnn::TopologyAccessor acc_bpt(*storage);

    acc_cache.enable_adjacency_cache(true);
    acc_cache.prebuild_adjacency_cache(mdb::gnn::EdgeOrientation::UNDIRECTED);

    EXPECT_TRUE(acc_cache.is_adjacency_cache_built(
        mdb::gnn::EdgeOrientation::UNDIRECTED));
    EXPECT_EQ(FixtureGraph::edges().size(),
              acc_cache.get_adjacency_cache_fwd_entries());
    EXPECT_EQ(FixtureGraph::edges().size(),
              acc_cache.get_adjacency_cache_rev_entries());
    EXPECT_GT(acc_cache.get_adjacency_cache_size_bytes(), 0u);

    for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
        auto via_cache = acc_cache.get_neighbors(ObjectId(nid),
            mdb::gnn::EdgeOrientation::UNDIRECTED);
        auto via_bpt   = acc_bpt.get_neighbors(ObjectId(nid),
            mdb::gnn::EdgeOrientation::UNDIRECTED);
        EXPECT_EQ(neighbors_as_sorted_pairs(via_cache),
                  neighbors_as_sorted_pairs(via_bpt))
            << "node " << nid;
    }
}

// ---------------------------------------------------------------------------
// Test 6 — Disable after build: cache is cleared, lookups revert to BPT.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorAdjacencyCache, DisableClearsCache) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("adj_cache_disable_clears");

    mdb::gnn::TopologyAccessor acc(*storage);
    acc.enable_adjacency_cache(true);
    acc.prebuild_adjacency_cache(mdb::gnn::EdgeOrientation::UNDIRECTED);
    ASSERT_GT(acc.get_adjacency_cache_size_bytes(), 0u);

    acc.enable_adjacency_cache(false);
    EXPECT_FALSE(acc.is_adjacency_cache_enabled());
    EXPECT_FALSE(acc.is_adjacency_cache_built(
        mdb::gnn::EdgeOrientation::NATURAL));
    EXPECT_FALSE(acc.is_adjacency_cache_built(
        mdb::gnn::EdgeOrientation::REVERSE));
    EXPECT_EQ(0u, acc.get_adjacency_cache_size_bytes());
    EXPECT_EQ(0u, acc.get_adjacency_cache_fwd_entries());
    EXPECT_EQ(0u, acc.get_adjacency_cache_rev_entries());

    // Lookups must still work via the BPT fallback.
    mdb::gnn::TopologyAccessor acc_bpt(*storage);
    for (uint64_t nid = 0; nid < FixtureGraph::kNumNodes; ++nid) {
        auto a = acc.get_out_neighbors(ObjectId(nid));
        auto b = acc_bpt.get_out_neighbors(ObjectId(nid));
        EXPECT_EQ(neighbors_as_sorted_pairs(a), neighbors_as_sorted_pairs(b))
            << "node " << nid;
    }
}

// ---------------------------------------------------------------------------
// Test 7 — K-hop sampling parity under fixed RNG seed.
//
// Spec #11 must NOT change sampler output under the same seed: the cache
// stores the same edges the BPT path returns and the sample_layer helper
// drives both with mt19937_64(42). The cached and the BPT-only iterations
// must therefore produce identical per-seed neighbour selections.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorAdjacencyCache, KHopSamplingDeterministic) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("adj_cache_khop_deterministic");

    mdb::gnn::TopologyAccessor acc_cache(*storage);
    mdb::gnn::TopologyAccessor acc_bpt(*storage);

    acc_cache.enable_adjacency_cache(true);
    acc_cache.prebuild_adjacency_cache(mdb::gnn::EdgeOrientation::UNDIRECTED);

    std::vector<ObjectId> seeds;
    for (uint64_t i = 0; i < FixtureGraph::kNumNodes; ++i) {
        seeds.emplace_back(i);
    }

    for (auto orientation : {
            mdb::gnn::EdgeOrientation::NATURAL,
            mdb::gnn::EdgeOrientation::REVERSE,
            mdb::gnn::EdgeOrientation::UNDIRECTED,
        })
    {
        for (std::size_t fanout : {1u, 2u, 3u, 5u}) {
            auto via_cache = sample_layer_under_seed(
                acc_cache, seeds, fanout, /*seed=*/42, orientation);
            auto via_bpt   = sample_layer_under_seed(
                acc_bpt,   seeds, fanout, /*seed=*/42, orientation);
            EXPECT_EQ(via_cache, via_bpt)
                << "orientation=" << static_cast<int>(orientation)
                << " fanout=" << fanout;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 8 — Isolated nodes: empty neighbours via cache, no crash.
// ---------------------------------------------------------------------------
TEST(TopologyAccessorAdjacencyCache, IsolatedNodeReturnsEmpty) {
    (void)MdbFixture::instance();
    auto storage = build_fixture_storage("adj_cache_isolated_node");

    mdb::gnn::TopologyAccessor acc(*storage);
    acc.enable_adjacency_cache(true);
    acc.prebuild_adjacency_cache(mdb::gnn::EdgeOrientation::UNDIRECTED);

    // Node 5 in the fixture has no edges in either direction.
    auto out_neigh = acc.get_out_neighbors(ObjectId(5));
    EXPECT_TRUE(out_neigh.node_ids.empty());
    auto in_neigh = acc.get_in_neighbors(ObjectId(5));
    EXPECT_TRUE(in_neigh.node_ids.empty());
    auto und = acc.get_neighbors(ObjectId(5),
        mdb::gnn::EdgeOrientation::UNDIRECTED);
    EXPECT_TRUE(und.node_ids.empty());

    // Out-of-range node id far past the fixture: BPT range query returns
    // empty; cache must report empty too rather than crashing on the absent
    // hash key.
    auto missing = acc.get_out_neighbors(ObjectId(999999ULL));
    EXPECT_TRUE(missing.node_ids.empty());
}
