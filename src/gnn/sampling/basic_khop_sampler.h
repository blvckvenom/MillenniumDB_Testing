#pragma once

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
 * For optimized sampling, see Phase 2B: SortedBatchSampler, ReservoirSampler.
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
     * Plan F (2026-05-11) — parallel sampling. Each worker thread builds its
     * own `BasicKHopSampler` with private RNG + per-node access counts, but
     * borrows the topology (FourLevelTopologyStore + adjacency caches) from
     * a primary instance. The primary owns the topology; workers reference
     * it through the raw pointer. The shared topology must outlive every
     * worker.
     *
     * Worker ctors skip Phase 0 auto-profile, `enable_four_level_store`,
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
    // Spec #13 Phase 5 — node access count tally (warm-start producer)
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
    // Plan F — parallel sampling support
    // =========================================================================

    /**
     * @brief Borrow this sampler's TopologyAccessor (non-owning).
     *
     * Plan F (2026-05-11) — used by the parallel scheduler to share the
     * expensive FourLevelTopologyStore + adjacency caches across worker
     * sampler instances. The returned reference is valid for the lifetime
     * of this sampler.
     */
    TopologyAccessor& get_topology();

    /**
     * @brief Re-seed all internal RNGs deterministically for a batch.
     *
     * Plan F (2026-05-11) — call before `sample(batch_seeds, batch_id, split)`
     * inside a parallel scheduler to make each batch's output invariant to
     * which worker thread picks it up. Seeds = `config.random_seed XOR
     * batch_id` mixing function, applied to the BasicKHopSampler's own
     * `rng`, the LeapfrogGnnSampler, and the SeekBasedGnnSampler.
     */
    void reseed_for_batch(uint64_t batch_id);

    /**
     * @brief Merge per-node access counts from a worker sampler.
     *
     * Plan F (2026-05-11) — after all workers finish sampling, the
     * scheduler reduces their per-thread `node_access_counts()` vectors
     * into the primary's tally so the warm-start `node_counts.bin`
     * reflects the entire run.
     */
    void merge_counts_from(const std::vector<uint64_t>& other);

    /**
     * @brief Plan E Phase 0 telemetry (2026-05-11).
     *
     * Reports whether the cold-start auto-profiler ran during this
     * sampler's construction, how many walks it performed, and how long
     * it took. Always populated (with `triggered=false` when the
     * auto-profiler decided not to run — typically because
     * `node_counts.bin` already existed or the user opted out via
     * `auto_profile_on_cold_start=false`).
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
