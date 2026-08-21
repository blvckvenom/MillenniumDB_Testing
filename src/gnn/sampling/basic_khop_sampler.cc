#include "gnn/sampling/basic_khop_sampler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "gnn/projection/four_level_topology_store.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/projection/topology_walk_profiler.h"
#include "gnn/sampling/leapfrog_gnn_sampler.h"
#include "gnn/sampling/node_counts_io.h"
#include "gnn/sampling/seek_based_gnn_sampler.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "misc/ablation_registry.h"

#include <filesystem>
#include <iostream>

namespace mdb::gnn {

// ---------------------------------------------------------------------------
// Expand-stage profiling (env MDB_GNN_EXPAND_PROFILE=1, default off). Splits the
// k-hop expand into four sub-costs so we measure where the time actually goes
// before optimizing. Microsecond accumulators, summed across all worker threads
// via relaxed atomics; a few clock reads per batch when enabled, zero overhead
// (one branch) when off. Never affects the sampled output.
// ---------------------------------------------------------------------------
namespace {

struct ExpandProfile {
    std::atomic<uint64_t> sample_us{0};   // per-node neighbor sampling loop
    std::atomic<uint64_t> fetch_us{0};    // ...of which: topology get_neighbors
    std::atomic<uint64_t> convert_us{0};  // convert-to-edges + next-layer dedup
    std::atomic<uint64_t> edges_us{0};    // build_edges local-index maps
    std::atomic<uint64_t> unique_us{0};   // rebuild_unique_nodes dedup
};
ExpandProfile g_expand_profile;

// choice() and not flag(): only "1" ever turned the profile on, so every other
// value already meant off. The accepted list is what makes a mistyped value say
// so, instead of looking like a deliberately disabled arm.
// A FUNCTION and not a namespace-scope global. As a global it resolved during
// static initialisation, before main(), so every invocation of the binary
// declared this switch first: `mdb help` opened with an [ABLATION] line ahead
// of its banner. Worse, the order of static initialisation across translation
// units is unspecified, so which switch got declared first was not even stable.
// A function-local static defers the decision to the first sampler that asks,
// which is also when the declaration carries information.
inline bool expand_profile_on() {
    static const bool on =
        Ablation::choice("MDB_GNN_EXPAND_PROFILE", "0", {"0", "1"}) == "1";
    return on;
}

inline uint64_t now_us_() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

// =============================================================================
// Implementation Details
// =============================================================================

struct BasicKHopSampler::Impl {
    GQL::ProjectionStorage& storage;
    SamplingConfig config;
    // Split ownership for parallel offline sampling: `owned_topology` is
    // non-null on the primary sampler and null on workers (which borrow the
    // primary's already-initialised TopologyAccessor). `topology` is the raw
    // pointer used throughout the code, so existing call sites keep using
    // `topology->...` regardless of who actually owns it. Workers must
    // not destroy their borrowed topology; that's the primary's job.
    std::unique_ptr<TopologyAccessor> owned_topology;
    TopologyAccessor* topology = nullptr;
    std::unique_ptr<LeapfrogGnnSampler> leapfrog_sampler;    ///< Sweep-based sampler
    std::unique_ptr<SeekBasedGnnSampler> seek_sampler;       ///< Seek-based sampler
    std::mt19937_64 rng;
    bool use_leapfrog = true;  ///< Enable batch optimization by default

    // Per-node access tally used by the Four-Level Topology Store to build
    // the frequency-based tier assignment on the next warm-start run.
    // Indexed by dense row_idx == ObjectId::get_value(). The vector is
    // sized lazily on first access so test doubles / synthetic graphs
    // that never query topology->get_node_count() avoid the alloc.
    std::vector<uint64_t> node_access_counts;

    // Reusable per-sampler neighbor buffer for the hot fetch path. Each sampler
    // (one per worker thread) owns its own, so get_neighbors_into can fill it
    // without a fresh per-node allocation. The measured dominant expand cost.
    Neighbors nbr_scratch_;

    // Optional shared atomic tally. When non-null, all parallel workers
    // fetch_add into this single N-sized array instead of each growing its
    // own `node_access_counts` vector (which would cost 0.83 GB × numWorkers
    // on papers100M). Owned by OfflineSamplingEngine; pre-sized + zero-init;
    // outlives every sampler. nullptr ⇒ legacy private-vector path.
    std::atomic<uint64_t>* shared_counts_   = nullptr;
    std::size_t            shared_counts_n_  = 0;

    // Increment the access count for `id`, growing the vector on demand.
    // The 8-bit ObjectId type tag is masked off here at the boundary —
    // the same convention the FourLevelTopologyStore uses to align with
    // `tier_lookup_` and the topology CSR sidecar's dense ROW_PTR layout
    // (the sidecar stores stripped uint64 node indices, not tagged ObjectIds).
    inline void tally_(ObjectId id) {
        const uint64_t row_idx = id.get_value();
        // Shared-array fast path for parallel sampling runs. Relaxed is
        // sufficient: we only need the final per-node sum, which is
        // interleave-invariant (commutative add); no inter-thread ordering
        // is implied by a tally.
        if (shared_counts_ != nullptr) {
            if (row_idx < shared_counts_n_) {
                shared_counts_[static_cast<std::size_t>(row_idx)]
                    .fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        if (row_idx >= node_access_counts.size()) {
            // Lazy growth — start at the projection's node count when we
            // can (the typical path; skips amortised reallocs on the
            // hot loop), fall back to plain push-out resize for the
            // rare case where the topology accessor isn't ready yet.
            uint64_t target = static_cast<uint64_t>(topology->get_node_count());
            if (target <= row_idx) target = row_idx + 1;
            node_access_counts.resize(static_cast<std::size_t>(target), 0);
        }
        node_access_counts[static_cast<std::size_t>(row_idx)]++;
    }

    Impl(GQL::ProjectionStorage& storage_, const SamplingConfig& config_)
        : storage(storage_)
        , config(config_)
        , owned_topology(std::make_unique<TopologyAccessor>(storage_))
        , leapfrog_sampler(std::make_unique<LeapfrogGnnSampler>(storage_))
        , seek_sampler(std::make_unique<SeekBasedGnnSampler>(storage_))
        , rng(config_.random_seed)
    {
        topology = owned_topology.get();
        config.validate();
        // Sync random seed with all samplers
        leapfrog_sampler->set_random_seed(config_.random_seed);
        seek_sampler->set_random_seed(config_.random_seed);

        // The Four-Level Topology Store (L1 RAM hash / L2 compact uint32 CSR /
        // L3 mmap sidecar / L4 direct B+Tree) supersedes the simpler
        // in-memory adjacency cache. When the four-level store is enabled,
        // we delegate the entire neighbor lookup pipeline to it and skip
        // building the single-tier adjacency cache to avoid double-RAM
        // accounting.
        if (config_.use_four_level_topology_store) {
            // Bootstrap `node_counts.bin` via a cheap random-walk profile
            // when no warm-start file exists yet and the mmap'd topology
            // CSR sidecar (topology_{fwd,rev}.csr) is on disk. Without this,
            // the Four-Level Topology Store's cold-start path skips the L3
            // MinHash reorder, and on graphs whose sidecar exceeds RAM
            // (e.g. papers100M topology_*.csr at 53 GB on a 30 GB host)
            // the resulting random mmap access pattern thrashes the page
            // cache and the sample never completes. The profiler issues
            // ~500k lookups (vs ~5 B for a full 3-layer sample) and
            // produces counts good enough to activate the warm-start
            // reorder on this very build. Setting
            // `auto_profile_on_cold_start:false` preserves the legacy
            // cold path for A/B benchmarks.
            run_phase0_auto_profile_if_needed_();

            FourLevelTopologyStore::Config tcfg;
            tcfg.l1_budget_mb        = config_.l1_cache_mb;
            tcfg.l2_budget_mb        = config_.l2_cache_mb;
            tcfg.use_l3_mmap_sidecar = config_.use_l3_mmap_sidecar;
            tcfg.orientation         = config_.orientation;
            // useSymmetricTopology AUTO/ON -> build the pre-merged undirected
            // tier; OFF -> keep the runtime out+in+merge fallback (the
            // bit-reproducible reference, byte-identical via the same dedup).
            tcfg.build_symmetric_tier =
                config_.symmetric_resolved_on(config_.orientation);
            // Lean tiled-symmetric GPU path (set by the engine's AUTO check):
            // build() skips the L1/L2 tiers + whole-COL_IDX copy and pins the
            // symmetric slice tiled. use_four_level_topology_store stays true so
            // node_counts.bin persistence + the store-backed plumbing are intact.
            tcfg.lean_symmetric_gpu = config_.lean_symmetric_gpu;
            topology->enable_four_level_store(tcfg);
            // Force PER_NODE: the four-level store dispatch is O(1) per
            // node (L1 hash lookup or L2 CSR slice), which beats the
            // Leapfrog range-sweep and SeekBased B+Tree seeks under
            // cached paths.
            use_leapfrog = false;
        }
        // In-memory adjacency cache path: perform a single full B+Tree scan
        // up front (O(|E|)) to populate an unordered_map<src, vector<AdjEntry>>
        // so that the O(|E|) cost is amortised across every batch in this
        // sampling run. When the cache is built we force PER_NODE for every
        // layer: PER_NODE → topology->get_neighbors → O(degree) hash lookup,
        // which beats Leapfrog's O(|E|) per-batch range scan and SeekBased's
        // per-node O(log E) seek under the cached path. Bit-identical sampling
        // output is preserved because the cache holds the same edges the
        // B+Tree path returns.
        else if (config_.use_adjacency_cache) {
            topology->enable_adjacency_cache(true);
            topology->prebuild_adjacency_cache(config_.orientation);
            // Force PER_NODE: the cache makes per-node sampling
            // unconditionally optimal compared to Leapfrog/Seek.
            use_leapfrog = false;
        }
    }

    // Worker constructor for parallel offline sampling. Borrows the primary's
    // TopologyAccessor (which has already completed the cold-start profile,
    // four-level store initialisation, and adjacency-cache build) and creates
    // fresh Leapfrog/Seek samplers owning their own RNG state. Skips the
    // expensive Phase 0 / cache setup since the primary did all that work.
    Impl(GQL::ProjectionStorage& storage_,
         const SamplingConfig& config_,
         TopologyAccessor* shared_topology,
         uint32_t worker_offset)
        : storage(storage_)
        , config(config_)
        , owned_topology(nullptr)
        , leapfrog_sampler(std::make_unique<LeapfrogGnnSampler>(storage_))
        , seek_sampler(std::make_unique<SeekBasedGnnSampler>(storage_))
        // Seed workers with `random_seed XOR worker_offset` so they start
        // from distinct but deterministic states. Callers are expected to
        // re-seed via `reseed_for_batch()` before each batch anyway, which
        // makes the output invariant to scheduling; this initial seed is a
        // safety net for callers that forget.
        , rng(config_.random_seed ^ static_cast<uint64_t>(worker_offset))
    {
        topology = shared_topology;
        config.validate();
        const uint64_t worker_seed =
            config_.random_seed ^ static_cast<uint64_t>(worker_offset);
        leapfrog_sampler->set_random_seed(worker_seed);
        seek_sampler->set_random_seed(worker_seed);
        // Mirror the primary's strategy choice. The primary forced PER_NODE
        // iff the Four-Level Topology Store or in-memory adjacency cache is
        // active; workers inherit the same toggle by inspecting the same
        // config flags. We never rebuild the cache or re-enable the four-
        // level store here — that's the primary's responsibility.
        if (config_.use_four_level_topology_store ||
            config_.use_adjacency_cache) {
            use_leapfrog = false;
        }
    }

    // Cold-start topology profiler — see comment in the Four-Level Store
    // branch of the primary constructor above. Runs `TopologyWalkProfiler`
    // over the mmap'd reverse topology CSR sidecar (topology_rev.csr) using
    // degree-weighted Vose-alias seed selection, and persists the resulting
    // per-node access counts to `node_counts.bin` so the Four-Level Topology
    // Store can activate its L3 MinHash reorder on this very build. No-op
    // when:
    //   - `auto_profile_on_cold_start` is false (user opt-out)
    //   - `node_counts.bin` already exists (warm-start ready)
    //   - the sidecar is absent (projection built without
    //     `buildTopologySnapshot:true`)
    //   - the projection_dir cannot be resolved (synthetic test paths
    //     that don't have a real on-disk projection)
    //
    // Outcomes are also surfaced via the public `last_phase0_result_`
    // member so `gnn_offline_sample_procedure` can yield them.
    struct Phase0Telemetry {
        bool          triggered     = false;
        bool          succeeded     = false;
        std::size_t   walks_done    = 0;
        std::size_t   lookups_done  = 0;
        double        elapsed_seconds = 0.0;
    };
    Phase0Telemetry last_phase0_result;

    void run_phase0_auto_profile_if_needed_() {
        if (!config.auto_profile_on_cold_start) return;

        std::filesystem::path proj_dir;
        try {
            proj_dir = std::filesystem::path(storage.get_projection_dir());
        } catch (...) {
            return;  // synthetic test path with no real storage
        }
        if (proj_dir.empty()) return;

        std::error_code ec;
        const auto counts_file = proj_dir / "node_counts.bin";
        if (std::filesystem::exists(counts_file, ec) && !ec) {
            // Warm-start ready; nothing to do.
            return;
        }

        // Try to open the REVERSE sidecar (the direction used by the
        // typical GraphSAGE in-neighbor sampling). The reader is
        // fallback-first: open() never throws, has_data() reports if
        // the sidecar is usable.
        auto reader = GQL::Projection::TopologySnapshotReader::open(
            proj_dir,
            GQL::Projection::TopologySnapshotReader::Direction::REVERSE);

        if (!reader.has_data()) {
            // No sidecar → can't profile cheaply. Fall through to legacy
            // cold-start path (FourLevelTopologyStore will log its own
            // "skipping L3 MinHash reorder" message).
            return;
        }

        last_phase0_result.triggered = true;

        std::cerr << "[BasicKHopSampler] Phase 0 auto-profile: "
                  << "no node_counts.bin found, running "
                  << (config.profile_num_walks
                          ? config.profile_num_walks
                          : TopologyWalkProfiler::kDefaultNumWalks)
                  << " random walks over the mmap'd topology CSR sidecar to bootstrap "
                  << "warm-start...\n";

        auto result = TopologyWalkProfiler::profile(
            reader,
            config.profile_num_walks,
            config.profile_walk_length,
            config.random_seed);

        last_phase0_result.walks_done      = config.profile_num_walks
            ? config.profile_num_walks
            : TopologyWalkProfiler::kDefaultNumWalks;
        last_phase0_result.lookups_done    = result.lookups_done;
        last_phase0_result.elapsed_seconds = result.elapsed_seconds;

        node_counts_io::persist(proj_dir, result.counts,
                                config.orientation);
        last_phase0_result.succeeded = true;

        std::cerr << "[BasicKHopSampler] Phase 0 done: "
                  << result.lookups_done << " lookups in "
                  << result.elapsed_seconds << "s ("
                  << result.restarts << " restarts). "
                  << "node_counts.bin persisted; warm-start activates "
                  << "on enable_four_level_store().\n";
    }

    /**
     * @brief Choose optimal batch strategy based on batch characteristics.
     *
     * Decision logic:
     * 1. If batch_strategy is not AUTO, use that strategy
     * 2. If batch is very small (<10 nodes), use PER_NODE
     * 3. Compare estimated costs of SWEEP vs SEEK
     *
     * Cost model:
     * - sweep_cost ≈ edges_in_range (from min_node to max_node)
     * - seek_cost ≈ batch_size × log2(total_edges) × overhead_factor
     *
     * @param sorted_nodes Sorted batch of node IDs
     * @param total_edges Total edges in projection
     * @return Selected strategy
     */
    BatchStrategy choose_strategy(
        const std::vector<ObjectId>& nodes,
        uint64_t total_edges
    ) {
        // If explicitly configured, use that strategy
        if (config.batch_strategy != BatchStrategy::AUTO) {
            return config.batch_strategy;
        }

        size_t batch_size = nodes.size();

        // Very small batches: per-node is simpler and has less overhead
        if (batch_size < 10) {
            return BatchStrategy::PER_NODE;
        }

        // Below leapfrog threshold: use per-node (existing behavior)
        if (batch_size < LEAPFROG_BATCH_THRESHOLD) {
            return BatchStrategy::PER_NODE;
        }

        // For larger batches, compare estimated costs
        if (total_edges == 0) {
            return BatchStrategy::PER_NODE;
        }

        // Estimate seek cost: B × log2(E) × overhead_factor
        double seek_cost = SeekBasedGnnSampler::estimate_seek_cost(
            batch_size,
            total_edges,
            config.seek_overhead_factor
        );

        // Estimate sweep cost: edges in the ID range
        // Heuristic: For batches >= threshold, assume sweep touches ~batch_size × avg_degree edges
        // This is a rough approximation; actual range depends on node ID distribution
        double avg_degree = static_cast<double>(total_edges) / std::max(1.0, static_cast<double>(batch_size));
        double sweep_cost = static_cast<double>(batch_size) * avg_degree;

        // Compare costs: use seek if it's cheaper
        if (seek_cost < sweep_cost) {
            return BatchStrategy::SEEK;
        }

        return BatchStrategy::SWEEP;
    }

    /**
     * @brief Sample up to `fanout` neighbors uniformly at random.
     *
     * Uses Fisher-Yates partial shuffle for efficiency when fanout < degree.
     */
    std::vector<std::pair<ObjectId, ObjectId>> sample_neighbors_uniform(
        ObjectId node_id,
        uint64_t fanout
    ) {
        const uint64_t tf = expand_profile_on() ? now_us_() : 0;
        topology->get_neighbors_into(node_id, config.orientation, nbr_scratch_);
        Neighbors& all_neighbors = nbr_scratch_;
        if (expand_profile_on()) {
            g_expand_profile.fetch_us.fetch_add(now_us_() - tf,
                                                std::memory_order_relaxed);
        }

        // ---- Self-loop ablation (env MDB_GNN_SAMPLE_SELF_LOOP=1, default OFF).
        //
        // GAP_REOPEN_2026-06-16: DiskGNN's papers100M graph prep does
        // dgl.add_self_loop (load_graph.py:24), so the node itself is one of
        // the (deg+1) candidates and is sampled w.p. ~min(1, fanout/(deg+1)) —
        // its features then enter the neighbor-MEAN. MDB normally excludes self
        // from the neighbor set (self handled separately by SAGEConv's
        // CONCAT(x_self, mean(N))). This flag injects the self as a candidate
        // BEFORE the empty-check + Fisher-Yates, faithfully emulating
        // add_self_loop (self competes for the fanout slots, and even a
        // degree-0 node gets a self-loop). Default OFF keeps cora bit-identical;
        // ON is the accuracy-gap ablation (re-sample + rebuild store + train).
        // text() and not flag(): the off-set here is {"0","false","off"},
        // compared case-sensitively, and it disagrees with the registry's
        // boolean rule on "off" (off here, ON there) and on "no" (on here, OFF
        // there). This switch decides whether the self enters the neighbour
        // MEAN, so an arm that silently flipped would be an accuracy result
        // about a different sampler. The raw value is declared; the rule below
        // is unchanged.
        static const bool kSampleSelfLoop = [] {
            const std::string e =
                Ablation::text("MDB_GNN_SAMPLE_SELF_LOOP", "0");
            return e != "0" && e != "false" && e != "off";
        }();
        if (kSampleSelfLoop) {
            all_neighbors.node_ids.push_back(node_id);
            all_neighbors.edge_ids.push_back(node_id);  // self-edge sentinel
        }

        std::vector<std::pair<ObjectId, ObjectId>> result;

        if (all_neighbors.node_ids.empty()) {
            return result;
        }

        size_t n = all_neighbors.node_ids.size();
        size_t k = std::min(static_cast<size_t>(fanout), n);

        result.reserve(k);

        if (k == n) {
            // Take all neighbors
            for (size_t i = 0; i < n; ++i) {
                result.emplace_back(all_neighbors.node_ids[i], all_neighbors.edge_ids[i]);
                // Count each visited neighbour for the frequency-based tier
                // assignment: a node's "frequency" is its total visit count
                // across the sampling run, matching DiskGNN's
                // `node_counts[v] += 1` in mega_batch_sampling.py:50.
                tally_(all_neighbors.node_ids[i]);
            }
        } else {
            // Fisher-Yates partial shuffle: sample k elements from n
            // We shuffle indices to preserve the node_id/edge_id pairing
            std::vector<size_t> indices(n);
            std::iota(indices.begin(), indices.end(), 0);

            for (size_t i = 0; i < k; ++i) {
                std::uniform_int_distribution<size_t> dist(i, n - 1);
                size_t j = dist(rng);
                std::swap(indices[i], indices[j]);
            }

            for (size_t i = 0; i < k; ++i) {
                size_t idx = indices[i];
                result.emplace_back(all_neighbors.node_ids[idx], all_neighbors.edge_ids[idx]);
                // Count each sampled neighbour for frequency-based tier
                // assignment (only the k actually visited ones, not the full
                // degree, mirroring the PER_NODE strategy below).
                tally_(all_neighbors.node_ids[idx]);
            }
        }

        return result;
    }

    /**
     * @brief Build the computational graph from sampled layers.
     *
     * Creates edges_per_layer with local indices mapping.
     */
    void build_edges(
        GraphSample& sample,
        const std::vector<std::unordered_map<uint64_t, std::vector<std::pair<ObjectId, ObjectId>>>>& sampled_edges
    ) {
        size_t num_layers = sample.nodes_per_layer.size();
        sample.edges_per_layer.resize(num_layers - 1);

        // Build node_id -> local_index mapping for each layer
        std::vector<std::unordered_map<uint64_t, int32_t>> layer_mappings(num_layers);

        for (size_t layer = 0; layer < num_layers; ++layer) {
            const auto& nodes = sample.nodes_per_layer[layer];
            for (size_t i = 0; i < nodes.size(); ++i) {
                layer_mappings[layer][nodes[i].id] = static_cast<int32_t>(i);
            }
        }

        // Build edges for each layer connection
        // edges_per_layer[k] connects layer k+1 (src) to layer k (dst)
        for (size_t k = 0; k < num_layers - 1; ++k) {
            auto& edges = sample.edges_per_layer[k];

            // sampled_edges[k] maps dst_node -> [(neighbor_node, edge_id), ...]
            // dst is in layer k, src (neighbors) are in layer k+1
            for (const auto& [dst_id, neighbor_edges] : sampled_edges[k]) {
                auto dst_it = layer_mappings[k].find(dst_id);
                if (dst_it == layer_mappings[k].end()) continue;
                int32_t dst_idx = dst_it->second;

                for (const auto& [src_node, edge_id] : neighbor_edges) {
                    auto src_it = layer_mappings[k + 1].find(src_node.id);
                    if (src_it == layer_mappings[k + 1].end()) continue;
                    int32_t src_idx = src_it->second;

                    edges.src_indices.push_back(src_idx);
                    edges.dst_indices.push_back(dst_idx);
                    edges.edge_ids.push_back(edge_id);
                }
            }
        }
    }

    /**
     * @brief Main sampling algorithm.
     */
    GraphSample do_sample(
        const std::vector<ObjectId>& seeds,
        uint64_t batch_id,
        SplitType split
    ) {
        GraphSample sample;
        sample.batch_id = batch_id;
        sample.split = split;

        if (seeds.empty() || config.fanouts.empty()) {
            return sample;
        }

        size_t K = config.fanouts.size();  // Number of layers

        // nodes_per_layer[0] = seeds, nodes_per_layer[k] = k-hop neighbors
        sample.nodes_per_layer.resize(K + 1);
        sample.nodes_per_layer[0] = seeds;

        // Track which edges were sampled at each layer
        // sampled_edges[k] = {dst_node_id -> [(src_node, edge_id), ...]}
        std::vector<std::unordered_map<uint64_t, std::vector<std::pair<ObjectId, ObjectId>>>> sampled_edges(K);

        // Get total edge count for strategy selection
        uint64_t total_edges = topology->get_edge_count();

        // Tally the seed nodes themselves for the frequency-based tier
        // assignment. Seeds are touched by every sample (their adjacency
        // drives the layer-0 expansion), so they belong in the access-count
        // vector even though `sample_neighbors_uniform` only counts the
        // neighbours it returns. Done before the layer loop so the seed
        // counts are deterministic w.r.t. the per-layer paths chosen.
        for (const ObjectId& seed_id : seeds) {
            tally_(seed_id);
        }

        // Sample layer by layer
        for (size_t k = 0; k < K; ++k) {
            uint64_t layer_fanout = config.fanouts[k];
            std::unordered_set<uint64_t> next_layer_set;
            const auto& current_layer = sample.nodes_per_layer[k];

            // Choose optimal strategy for this layer
            BatchStrategy strategy = use_leapfrog
                ? choose_strategy(current_layer, total_edges)
                : BatchStrategy::PER_NODE;

            BatchNeighbors batch_result;

            const bool prof = expand_profile_on();
            uint64_t   ts   = prof ? now_us_() : 0;

            switch (strategy) {
                case BatchStrategy::SWEEP:
                    // Coordinated B+Tree sweep (LeapfrogGnnSampler)
                    batch_result = leapfrog_sampler->sample_batch(
                        current_layer,
                        layer_fanout,
                        config.orientation
                    );
                    break;

                case BatchStrategy::SEEK:
                    // O(log E) seeks per node (SeekBasedGnnSampler)
                    batch_result = seek_sampler->sample_batch(
                        current_layer,
                        layer_fanout,
                        config.orientation
                    );
                    break;

                case BatchStrategy::AUTO:
                case BatchStrategy::PER_NODE:
                default:
                    // Per-node sampling (original algorithm)
                    for (const ObjectId& node_id : current_layer) {
                        auto neighbors = sample_neighbors_uniform(node_id, layer_fanout);
                        if (!neighbors.empty()) {
                            batch_result.neighbors[node_id.id] = std::move(neighbors);
                        }
                    }
                    break;
            }

            if (prof) {
                g_expand_profile.sample_us.fetch_add(now_us_() - ts,
                                                     std::memory_order_relaxed);
                ts = now_us_();
            }

            // Convert batch results to sampled_edges format
            for (const ObjectId& node_id : current_layer) {
                auto it = batch_result.neighbors.find(node_id.id);
                if (it != batch_result.neighbors.end() && !it->second.empty()) {
                    sampled_edges[k][node_id.id] = it->second;

                    for (const auto& [neighbor_node, edge_id] : it->second) {
                        next_layer_set.insert(neighbor_node.id);
                        // Tally each neighbour visited by SWEEP / SEEK
                        // strategies for the frequency-based tier assignment
                        // (PER_NODE already tallies inside
                        // sample_neighbors_uniform). Same semantic for all
                        // three strategies: count once per visit during this
                        // sampling run.
                        if (strategy == BatchStrategy::SWEEP ||
                            strategy == BatchStrategy::SEEK)
                        {
                            tally_(neighbor_node);
                        }
                    }
                }
            }

            // Defensive cap against unbounded growth. UNDIRECTED on
            // dense graphs can explode layer-2 to millions of nodes per
            // worker (papers100M fanout [10,15,20] N=20 silently SIGSEGV
            // pre-cap). Abort with an actionable error so the user can
            // reduce fanout or batch_size instead of crashing the server.
            if (config.max_layer_nodes > 0
                && next_layer_set.size() > config.max_layer_nodes)
            {
                throw std::runtime_error(
                    "BasicKHopSampler: layer " + std::to_string(k + 1)
                    + " grew to " + std::to_string(next_layer_set.size())
                    + " nodes (exceeds max_layer_nodes="
                    + std::to_string(config.max_layer_nodes) + "). Reduce "
                    + "fanout[" + std::to_string(k) + "]=" + std::to_string(layer_fanout)
                    + " or batch_size=" + std::to_string(config.batch_size)
                    + ", or raise max_layer_nodes if you have RAM headroom.");
            }

            // Convert set to vector for next layer
            sample.nodes_per_layer[k + 1].reserve(next_layer_set.size());
            for (uint64_t node_id : next_layer_set) {
                sample.nodes_per_layer[k + 1].emplace_back(node_id);
            }

            if (prof) {
                g_expand_profile.convert_us.fetch_add(now_us_() - ts,
                                                      std::memory_order_relaxed);
            }
        }

        // Build edge indices
        uint64_t te = expand_profile_on() ? now_us_() : 0;
        build_edges(sample, sampled_edges);
        if (expand_profile_on()) {
            g_expand_profile.edges_us.fetch_add(now_us_() - te,
                                                std::memory_order_relaxed);
            te = now_us_();
        }

        // Build all_unique_nodes
        sample.rebuild_unique_nodes();
        if (expand_profile_on()) {
            g_expand_profile.unique_us.fetch_add(now_us_() - te,
                                                 std::memory_order_relaxed);
        }

        return sample;
    }
};

// =============================================================================
// BasicKHopSampler Public Interface
// =============================================================================

BasicKHopSampler::BasicKHopSampler(GQL::ProjectionStorage& storage, const SamplingConfig& config)
    : impl_(std::make_unique<Impl>(storage, config))
{
}

BasicKHopSampler::BasicKHopSampler(
    GQL::ProjectionStorage& storage,
    const SamplingConfig& config,
    TopologyAccessor* shared_topology,
    uint32_t worker_offset)
    : impl_(std::make_unique<Impl>(storage, config, shared_topology, worker_offset))
{
}

BasicKHopSampler::~BasicKHopSampler() = default;

BasicKHopSampler::BasicKHopSampler(BasicKHopSampler&&) noexcept = default;
BasicKHopSampler& BasicKHopSampler::operator=(BasicKHopSampler&&) noexcept = default;

GraphSample BasicKHopSampler::sample(
    const std::vector<ObjectId>& seeds,
    uint64_t batch_id,
    SplitType split
) {
    return impl_->do_sample(seeds, batch_id, split);
}

GraphSample BasicKHopSampler::sample(const std::vector<ObjectId>& seeds) {
    return impl_->do_sample(seeds, 0, SplitType::TRAIN);
}

size_t BasicKHopSampler::num_layers() const {
    return impl_->config.fanouts.size();
}

uint64_t BasicKHopSampler::fanout(size_t layer) const {
    if (layer >= impl_->config.fanouts.size()) {
        return 0;
    }
    return impl_->config.fanouts[layer];
}

void BasicKHopSampler::set_random_seed(uint64_t seed) {
    impl_->rng.seed(seed);
    impl_->leapfrog_sampler->set_random_seed(seed);
}

uint64_t BasicKHopSampler::get_random_seed() const {
    return impl_->config.random_seed;
}

void BasicKHopSampler::set_use_leapfrog(bool enable) {
    impl_->use_leapfrog = enable;
}

bool BasicKHopSampler::get_use_leapfrog() const {
    return impl_->use_leapfrog;
}

const std::vector<uint64_t>& BasicKHopSampler::node_access_counts() const {
    return impl_->node_access_counts;
}

std::string BasicKHopSampler::dump_expand_profile() {
    if (!expand_profile_on()) return "";
    auto take = [](std::atomic<uint64_t>& a) {
        return a.exchange(0, std::memory_order_relaxed) / 1000.0;  // us -> ms
    };
    const double s = take(g_expand_profile.sample_us);
    const double f = take(g_expand_profile.fetch_us);
    const double c = take(g_expand_profile.convert_us);
    const double e = take(g_expand_profile.edges_us);
    const double u = take(g_expand_profile.unique_us);
    const double tot = s + c + e + u;
    auto pct = [&](double x) { return tot > 0 ? 100.0 * x / tot : 0.0; };
    char buf[640];
    std::snprintf(buf, sizeof(buf),
        "expand-profile (ms, summed over workers): sample=%.0f (%.1f%%) "
        "[of which fetch=%.0f (%.1f%%), reservoir+alloc=%.0f (%.1f%%)] "
        "convert+dedup=%.0f (%.1f%%) build_edges=%.0f (%.1f%%) "
        "unique=%.0f (%.1f%%) total=%.0f",
        s, pct(s), f, pct(f), s - f, pct(s - f), c, pct(c), e, pct(e),
        u, pct(u), tot);
    return std::string(buf);
}

TopologyAccessor& BasicKHopSampler::get_topology() {
    return *impl_->topology;
}

void BasicKHopSampler::reseed_for_batch(uint64_t batch_id) {
    // Mix `random_seed` with `batch_id` so each batch's RNG state is
    // deterministic regardless of which worker thread picks it up.
    // Plain XOR is fine here: `batch_id` is small and dense, so the
    // mt19937_64 warmup masks any low-bit correlation in the seed.
    const uint64_t batch_seed = impl_->config.random_seed ^ batch_id;
    impl_->rng.seed(batch_seed);
    impl_->leapfrog_sampler->set_random_seed(batch_seed);
    impl_->seek_sampler->set_random_seed(batch_seed);
}

void BasicKHopSampler::tally_nodes(const std::vector<ObjectId>& nodes) {
    for (const ObjectId& n : nodes) {
        impl_->tally_(n);
    }
}

void BasicKHopSampler::merge_counts_from(const std::vector<uint64_t>& other) {
    if (other.empty()) return;
    if (impl_->node_access_counts.size() < other.size()) {
        impl_->node_access_counts.resize(other.size(), 0);
    }
    for (std::size_t i = 0; i < other.size(); ++i) {
        impl_->node_access_counts[i] += other[i];
    }
}

void BasicKHopSampler::set_shared_access_counts(std::atomic<uint64_t>* base,
                                                std::size_t n) {
    impl_->shared_counts_  = base;
    impl_->shared_counts_n_ = (base != nullptr) ? n : 0;
}

void BasicKHopSampler::adopt_counts(std::vector<uint64_t> counts) {
    impl_->node_access_counts = std::move(counts);
}

BasicKHopSampler::Phase0Report BasicKHopSampler::phase0_report() const {
    Phase0Report out;
    out.triggered       = impl_->last_phase0_result.triggered;
    out.succeeded       = impl_->last_phase0_result.succeeded;
    out.walks_done      = impl_->last_phase0_result.walks_done;
    out.lookups_done    = impl_->last_phase0_result.lookups_done;
    out.elapsed_seconds = impl_->last_phase0_result.elapsed_seconds;
    return out;
}

} // namespace mdb::gnn
