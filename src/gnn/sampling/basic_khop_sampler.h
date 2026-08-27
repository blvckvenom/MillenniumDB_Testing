#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "graph_models/object_id.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sampling_config.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

class TopologyAccessor;

/**
 * @brief Basic k-hop neighborhood sampler for GNN mini-batches.
 *
 * Implements the standard GraphSAGE-style sampling algorithm:
 * 1. Start with seed nodes (layer 0)
 * 2. For each layer k, sample up to fanout[k] neighbors for each node
 * 3. Build computational graph with local indices
 *
 * ## Algorithm
 *
 * @code
 *   nodes[0] = seeds
 *   for k in 0..K-1:
 *       for each node in nodes[k]:
 *           neighbors = sample_neighbors(node, fanout[k])
 *           nodes[k+1] = union(nodes[k+1], neighbors)
 *       edges[k] = {(src, dst) | src in nodes[k+1], dst in nodes[k], edge exists}
 * @endcode
 *
 * ## Message Passing Direction
 *
 * By default (EdgeOrientation::REVERSE), messages flow FROM neighbors TO seeds:
 * - Layer K (input features) → Layer K-1 → ... → Layer 0 (output embeddings)
 * - This matches the standard GNN convention where node embeddings are computed
 *   by aggregating neighbor information.
 *
 * ## Limitations (Basic Implementation)
 *
 * - Samples each node independently (N tree traversals for N nodes)
 * - Loads all neighbors into memory before sampling
 * - No reservoir sampling for high-degree nodes
 *
 * For optimized sampling, see SortedBatchSampler and ReservoirSampler.
 *
 * @see GraphSample for output format
 * @see SamplingConfig for configuration
 * @see TopologyAccessor for neighbor retrieval
 */
class BasicKHopSampler {
public:
    /**
     * @brief Construct sampler for a projection.
     *
     * @param storage Reference to projection storage (must outlive sampler)
     * @param config Sampling configuration with fanouts and orientation
     */
    BasicKHopSampler(GQL::ProjectionStorage& storage, const SamplingConfig& config);

    /**
     * @brief Construct a worker sampler that shares a pre-built TopologyAccessor.
     *
     * Used by the parallel offline sampling worker pool. Each worker thread
     * builds its own `BasicKHopSampler` with private RNG + per-node access
     * counts, but borrows the topology (Four-Level Topology Store + adjacency
     * caches) from a primary instance. The primary owns the topology; workers
     * reference it through the raw pointer. The shared topology must outlive
     * every worker.
     *
     * Worker ctors skip the cold-start auto-profiler, `enable_four_level_store`,
     * and `prebuild_adjacency_cache` — those are assumed to have already
     * happened on the primary. RNG is seeded from `config.random_seed` plus
     * the worker_offset, but the recommended usage is to call
     * `reseed_for_batch(batch_id)` before each batch so output is invariant
     * to thread scheduling.
     *
     * @param storage           Reference to projection storage (read-only)
     * @param config            Sampling configuration (RNG seed + fanouts)
     * @param shared_topology   Non-owning pointer to a built TopologyAccessor
     *                          (typically `&primary.get_topology()`)
     * @param worker_offset     Worker index (>=1); used as additional RNG
     *                          seed material so workers start from distinct
     *                          but reproducible states.
     */
    BasicKHopSampler(
        GQL::ProjectionStorage& storage,
        const SamplingConfig& config,
        TopologyAccessor* shared_topology,
        uint32_t worker_offset);

    ~BasicKHopSampler();

    // Disable copy
    BasicKHopSampler(const BasicKHopSampler&) = delete;
    BasicKHopSampler& operator=(const BasicKHopSampler&) = delete;

    // Allow move
    BasicKHopSampler(BasicKHopSampler&&) noexcept;
    BasicKHopSampler& operator=(BasicKHopSampler&&) noexcept;

    // =========================================================================
    // Sampling Interface
    // =========================================================================

    /**
     * @brief Sample k-hop neighborhood for a batch of seeds.
     *
     * This is the main sampling method. Given seed nodes, it:
     * 1. Samples neighbors layer by layer according to fanouts
     * 2. Builds the computational graph with local indices
     * 3. Computes all unique nodes for feature fetching
     *
     * Following DiskGNN architecture, epoch information is NOT stored
     * in samples - epochs belong to the training layer.
     *
     * @param seeds Seed nodes (will become layer 0 in output)
     * @param batch_id Identifier for this batch
     * @param split Train/validation/test split type
     * @return GraphSample containing the sampled subgraph
     */
    GraphSample sample(
        const std::vector<ObjectId>& seeds,
        uint64_t batch_id,
        SplitType split
    );

    /**
     * @brief Sample with default metadata (batch_id=0, TRAIN).
     *
     * Convenience method for testing and simple use cases.
     */
    GraphSample sample(const std::vector<ObjectId>& seeds);

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Get number of layers (K = fanouts.size()).
     */
    size_t num_layers() const;

    /**
     * @brief Get fanout for a specific layer.
     */
    uint64_t fanout(size_t layer) const;

    /**
     * @brief Set random seed for reproducible sampling.
     */
    void set_random_seed(uint64_t seed);

    /**
     * @brief Get current random seed.
     */
    uint64_t get_random_seed() const;

    // =========================================================================
    // Optimization Options
    // =========================================================================

    /**
     * @brief Enable/disable Leapfrog optimization for batch sampling.
     *
     * When enabled, layers with more than LEAPFROG_BATCH_THRESHOLD nodes
     * use coordinated B+Tree iteration instead of per-node lookups.
     *
     * @param enable true to enable (default), false to use basic per-node sampling
     */
    void set_use_leapfrog(bool enable);

    /**
     * @brief Check if Leapfrog optimization is enabled.
     */
    bool get_use_leapfrog() const;

    // =========================================================================
    // Four-Level Topology Store — node access count tally (warm-start producer)
    // =========================================================================

    /**
     * @brief Per-node access counts accumulated across every `sample()` call
     *        on this instance. Indexed by dense `row_idx == ObjectId::get_value()`.
     *
     * Each entry tallies the number of times the node was accessed during
     * sampling — incremented once per visit as either a seed or a sampled
     * neighbor. The vector is empty until the first `sample()` call; it
     * grows up to the projection's node count on first access.
     *
     * Consumed by `OfflineSamplingEngine` to persist
     * `<projection_dir>/node_counts.bin`, which seeds
     * `TopologyFrequencyProfiler::compute_from_node_counts_` on the next
     * `gnn_offline_sample` run (warm start).
     */
    const std::vector<uint64_t>& node_access_counts() const;

    // =========================================================================
    // Expand-stage profiling (env MDB_GNN_EXPAND_PROFILE=1, default off)
    // =========================================================================

    /**
     * @brief One-line summary of the accumulated per-sub-cost expand timings,
     *        then reset the accumulators.
     *
     * Breaks the k-hop expand into its four sub-costs — the per-node sampling
     * loop, the convert-to-edges + next-layer dedup, the edge-index build, and
     * the unique-node rebuild — in milliseconds summed across all worker
     * threads. Returns an empty string when MDB_GNN_EXPAND_PROFILE is unset.
     * The offline sampling engine calls this once after the worker pool joins.
     * Profiling adds a few clock reads per batch (negligible) and is off by
     * default; it never changes the sampled output.
     */
    static std::string dump_expand_profile();

    // =========================================================================
    // Parallel offline sampling support (shared-memory worker pool)
    // =========================================================================

    /**
     * @brief Borrow this sampler's TopologyAccessor (non-owning).
     *
     * Used by the parallel offline sampling scheduler to share the
     * expensive Four-Level Topology Store (L1 RAM hash / L2 compact uint32
     * CSR / L3 mmap sidecar / L4 direct B+Tree) and adjacency caches across
     * worker sampler instances. The returned reference is valid for the
     * lifetime of this sampler.
     */
    TopologyAccessor& get_topology();

    /**
     * @brief Re-seed all internal RNGs deterministically for a batch.
     *
     * Call before `sample(batch_seeds, batch_id, split)` inside a parallel
     * sampling scheduler to make each batch's output invariant to which worker
     * thread picks it up. The new seed is derived as `config.random_seed XOR
     * batch_id` and is applied to the BasicKHopSampler's own `rng`, the
     * LeapfrogGnnSampler, and the SeekBasedGnnSampler.
     */
    void reseed_for_batch(uint64_t batch_id);

    /**
     * @brief Add one access count for each of `nodes` to the warm-start tally.
     *
     * Used by the GPU sampling path, which produces a GraphSample directly
     * without going through `sample_neighbors_uniform` (where the CPU path
     * tallies per visit). Tallying the sample's unique nodes keeps
     * `node_counts.bin` populated for the next warm-start. This counts a node
     * once per batch it appears in (not once per visit like the CPU path) — an
     * approximation acceptable for the frequency-based tier heuristic, since the
     * GPU path is not bit-identical to the CPU path anyway.
     */
    void tally_nodes(const std::vector<ObjectId>& nodes);

    /**
     * @brief Merge per-node access counts from a worker sampler.
     *
     * After all parallel workers finish sampling, the offline sampling
     * engine reduces their per-thread `node_access_counts()` vectors into
     * the primary's tally so the warm-start `node_counts.bin` file
     * reflects the full run's access pattern.
     */
    void merge_counts_from(const std::vector<uint64_t>& other);

    /**
     * @brief Install a SHARED atomic access-counts array.
     *
     * Replaces the per-worker N-sized `node_access_counts` vector (8 bytes per
     * node PER WORKER — close to 1 GB each at 100M-node scale) with a single
     * shared `std::atomic<uint64_t>` array of size `n`, owned by the
     * OfflineSamplingEngine and pointed-to by every worker sampler. `tally_`
     * then does a relaxed `fetch_add` into it instead of growing its private
     * vector, so total tally RAM is 8 bytes per node ONCE, regardless of the
     * number of parallel workers (with per-worker vectors, cost grew linearly
     * and capped worker count on memory-constrained hosts). The
     * array MUST be pre-sized to the projection's node count and
     * zero-initialized before sampling; it must outlive every sampler.
     * Correctness: the final per-node count is a commutative sum, so it is
     * identical regardless of thread interleaving (and `node_counts.bin` is
     * consumed by dict-equality). Passing `base == nullptr` reverts to the
     * private-vector path (legacy / single-thread). Does NOT affect the sample
     * output (batches.dat) — the tally is a side channel only.
     */
    void set_shared_access_counts(std::atomic<uint64_t>* base, std::size_t n);

    /**
     * @brief Adopt a materialized counts vector (replaces internal tally).
     *
     * After the parallel run, the offline sampling engine snapshots the
     * shared atomic array into a plain `std::vector<uint64_t>` and hands it
     * to the primary so the existing `node_access_counts()` accessor
     * (consumed by `node_counts_io::persist`) returns the full run's tally
     * without requiring a per-worker merge pass.
     */
    void adopt_counts(std::vector<uint64_t> counts);

    /**
     * @brief Telemetry from the cold-start random-walk topology profiler.
     *
     * Reports whether the cold-start auto-profiler (a degree-weighted
     * Vose-alias random-walk pass that writes `node_counts.bin` before the
     * Four-Level Topology Store is built) ran during this sampler's
     * construction, how many walks it performed, and how long it took.
     * Always populated (with `triggered=false` when the auto-profiler
     * decided not to run — typically because `node_counts.bin` already
     * existed or the user opted out via `auto_profile_on_cold_start=false`).
     */
    struct Phase0Report {
        bool        triggered     = false;
        bool        succeeded     = false;
        std::size_t walks_done    = 0;
        std::size_t lookups_done  = 0;
        double      elapsed_seconds = 0.0;
    };
    Phase0Report phase0_report() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
