#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gnn/projection/edge_orientation.h"   // EdgeOrientation (lightweight header)
#include "gnn/projection/topology_accessor.h"  // SamplingStrategy + BatchStrategy (still depends on torch)

namespace mdb::gnn {

/**
 * @brief Configuration for offline sampling.
 *
 * Defines all parameters for the offline sampling process:
 * - Seed selection (which nodes to train on)
 * - Train/val/test splitting
 * - K-hop sampling (layers, fanouts)
 * - Reproducibility (random seed)
 *
 * Usage:
 * @code
 *   SamplingConfig config;
 *   config.projection_name = "my_graph";
 *   config.fanouts = {15, 10};  // 2-hop with fanout 15 then 10
 *   config.batch_size = 1024;
 *   config.validate();  // Throws if invalid
 * @endcode
 *
 * @see OfflineSamplingEngine for usage
 */
struct SamplingConfig {
    // =========================================================================
    // Projection Selection
    // =========================================================================

    std::string projection_name;  ///< Name of the projection to sample from

    // =========================================================================
    // Seed Selection
    // =========================================================================

    /**
     * @brief GQL query to select seed nodes.
     *
     * If empty, uses all nodes with the label_property defined.
     * Example: "MATCH (n:User) WHERE n.label IS NOT NULL RETURN id(n)"
     */
    std::optional<std::string> seed_query;

    /**
     * @brief Property name containing node labels for classification.
     *
     * Nodes without this property are excluded from training seeds.
     */
    std::string label_property = "label";

    // =========================================================================
    // Train/Val/Test Split
    // =========================================================================

    double train_ratio = 0.7;   ///< Fraction of seeds for training (default: 70%)
    double val_ratio = 0.15;    ///< Fraction of seeds for validation (default: 15%)
    double test_ratio = 0.15;   ///< Fraction of seeds for testing (default: 15%)

    /**
     * @brief Use predefined splits from splits.bin instead of random ratio-based splitting.
     *
     * When true, the sampler reads splits.bin from the projection directory and
     * assigns each seed node to train/val/test based on its stored split value
     * (0=TRAIN, 1=VAL, 2=TEST, 255=UNLABELED/skipped).
     * The trainRatio/validationRatio/testRatio fields are ignored.
     *
     * When false (default), ratio-based random splitting is used.
     */
    bool use_predefined_splits = false;

    // =========================================================================
    // Batching
    // =========================================================================

    uint64_t batch_size = 1024;  ///< Number of seed nodes per batch

    // =========================================================================
    // K-Hop Sampling
    // =========================================================================

    /**
     * @brief Number of neighbors to sample per layer.
     *
     * fanouts[0] = neighbors for 1-hop, fanouts[1] = neighbors for 2-hop, etc.
     * Number of layers K = fanouts.size()
     *
     * Example: {15, 10} means:
     *   - Sample 15 neighbors at 1-hop distance
     *   - Sample 10 neighbors at 2-hop distance
     */
    std::vector<uint64_t> fanouts = {15, 10};

    /**
     * @brief Edge orientation for neighbor traversal.
     *
     * - REVERSE: Messages flow from neighbors to seed (standard GNN convention)
     * - NATURAL: Messages flow from seed to neighbors
     * - UNDIRECTED: Traverse both directions
     */
    EdgeOrientation orientation = EdgeOrientation::REVERSE;

    /**
     * @brief Sampling strategy for neighbor selection.
     */
    SamplingStrategy strategy = SamplingStrategy::UNIFORM;

    /**
     * @brief Whether to sample with replacement.
     *
     * If true, same neighbor can be sampled multiple times.
     * If false (default), each neighbor appears at most once.
     */
    bool sample_with_replacement = false;

    // =========================================================================
    // Batch Strategy (B+Tree Traversal Optimization)
    // =========================================================================

    /**
     * @brief Strategy for batch neighbor sampling I/O pattern.
     *
     * - AUTO (default): Automatically choose between SWEEP and SEEK based on
     *   batch density and graph size
     * - SWEEP: Use coordinated B+Tree sweep (LeapfrogGnnSampler)
     * - SEEK: Use O(log E) seeks per node (SeekBasedGnnSampler)
     * - PER_NODE: Use TopologyAccessor per-node lookups
     *
     * @see BatchStrategy enum for performance comparison
     */
    BatchStrategy batch_strategy = BatchStrategy::AUTO;

    /**
     * @brief Overhead factor for seek cost estimation.
     *
     * Used in AUTO mode to decide between SWEEP and SEEK strategies.
     * Higher values favor SWEEP (more conservative about seek overhead).
     *
     * The cost model is: seek_cost = B × log2(E) × seek_overhead_factor
     *
     * Typical values:
     * - 1.5: Aggressive (prefers SEEK more often)
     * - 2.0: Balanced (default)
     * - 3.0: Conservative (prefers SWEEP more often)
     *
     * Tune based on your hardware characteristics:
     * - SSD storage: Use lower values (1.5-2.0)
     * - HDD storage: Use higher values (2.5-3.0)
     * - Large buffer pool: Use lower values
     */
    double seek_overhead_factor = 2.0;

    // =========================================================================
    // Reproducibility
    // =========================================================================

    static constexpr uint64_t DEFAULT_RANDOM_SEED = 42;
    uint64_t random_seed = DEFAULT_RANDOM_SEED;  ///< Random seed for deterministic sampling

    // =========================================================================
    // Memory Optimization
    // =========================================================================

    /**
     * @brief Degree threshold for reservoir sampling.
     *
     * Nodes with degree > this value use streaming reservoir sampling
     * instead of loading all neighbors into memory.
     */
    uint64_t reservoir_threshold = 10000;

    /**
     * @brief Use the in-memory projection adjacency cache (Spec #11).
     *
     * When true (default), the sampler full-scans `from_to_edge` +
     * `to_from_edge` ONCE at construction and resolves every subsequent
     * neighbour lookup in O(degree) from a hash map instead of paying an
     * O(page_tuples) range query against the live B+Tree per seed.
     *
     * Empirically replaces an 11-minute run on ogbn-products (62 M edges)
     * with a sub-30-second one — the same pattern that delivered ~700×
     * speed-up in EmbeddingWriter Phase B (commit 6521cc21).
     *
     * Memory cost: ~16 bytes × num_directed_entries. Both UNDIRECTED
     * directions cached together: ogbn-arxiv (~2.1 M directed) ≈ 34 MB,
     * ogbn-products (~124 M directed) ≈ 3 GB.
     *
     * Disable with `useAdjacencyCache: false` for memory-constrained
     * scenarios (papers100M scale needs partitioning before the cache fits
     * — see master plan §11) or when bypassing the cache for benchmarking.
     */
    bool use_adjacency_cache = true;

    // =========================================================================
    // Four-Level Topology Store (Spec #13)
    // =========================================================================

    /**
     * @brief Use the Four-Level Topology Store (Spec #13).
     *
     * When true, the sampler builds a tiered cache that partitions adjacency
     * across four tiers:
     *   - L1: RAM hash (hot nodes, ~5-20 ns/lookup)
     *   - L2: RAM compact CSR (warm nodes, ~50-200 ns/lookup)
     *   - L3: mmap-backed CSR sidecar (cold nodes, ~5-100 us/lookup)
     *   - L4: B+Tree direct (rare fallback)
     *
     * Designed to enable papers100M-scale sampling on commodity 32 GB RAM
     * hardware. Default true: strictly better than Spec #11 — matches it
     * for small graphs (everything fits in L1) and avoids OOM on graphs
     * larger than available RAM. Set false to opt out for A/B benchmarks
     * or to force pure Spec #11 behavior.
     *
     * Validation: setting this true while `use_adjacency_cache=false` is
     * an error — Spec #13 supersedes Spec #11 but does not bypass the cache
     * gate (D8 in the design doc).
     */
    bool use_four_level_topology_store = true;

    /**
     * @brief L1 cache budget in MiB. 0 means auto-detect from
     *        /proc/meminfo (25% of 70% of MemAvailable).
     */
    std::size_t l1_cache_mb = 0;

    /**
     * @brief L2 cache budget in MiB. 0 means auto-detect from
     *        /proc/meminfo (75% of 70% of MemAvailable).
     */
    std::size_t l2_cache_mb = 0;

    /**
     * @brief Use the mmap-backed L3 sidecar (Spec #4-B `topology_*.csr`
     *        files) when present. Requires `buildTopologySnapshot:true`
     *        at projection-build time. Silently ignored when the sidecar
     *        is absent or stale.
     */
    bool use_l3_mmap_sidecar = false;

    // =========================================================================
    // Output
    // =========================================================================

    /**
     * @brief Name for the sample storage.
     *
     * Samples are stored at: <db_folder>/samples/<sample_name>/
     */
    std::string sample_name;

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate configuration parameters.
     *
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (projection_name.empty()) {
            throw std::invalid_argument("projection_name cannot be empty");
        }

        if (sample_name.empty()) {
            throw std::invalid_argument("sample_name cannot be empty");
        }

        if (fanouts.empty()) {
            throw std::invalid_argument("fanouts cannot be empty (need at least 1 layer)");
        }

        for (size_t i = 0; i < fanouts.size(); ++i) {
            if (fanouts[i] == 0) {
                throw std::invalid_argument("fanout[" + std::to_string(i) + "] cannot be 0");
            }
        }

        if (batch_size == 0) {
            throw std::invalid_argument("batch_size must be > 0");
        }

        // Ratio validation only applies when not using predefined splits
        if (!use_predefined_splits) {
            double total_ratio = train_ratio + val_ratio + test_ratio;
            if (total_ratio < 0.999 || total_ratio > 1.001) {
                throw std::invalid_argument(
                    "train_ratio + val_ratio + test_ratio must equal 1.0 (got " +
                    std::to_string(total_ratio) + ")"
                );
            }

            if (train_ratio < 0.0 || val_ratio < 0.0 || test_ratio < 0.0) {
                throw std::invalid_argument("split ratios cannot be negative");
            }
        }

        // Spec #13 D8: useFourLevelTopologyStore implies useAdjacencyCache.
        // Both flags being false is OK (legacy fallback through sidecar /
        // BPT direct). Both true is OK (Spec #13 supersedes Spec #11
        // transparently).  The single illegal combination is
        // useFourLevelTopologyStore=true with useAdjacencyCache=false
        // because the user is asking for a tiered cache while explicitly
        // disabling the cache gate.
        if (use_four_level_topology_store && !use_adjacency_cache) {
            throw std::invalid_argument(
                "useFourLevelTopologyStore=true requires useAdjacencyCache=true. "
                "Set useAdjacencyCache:true (or omit it; default is true) when "
                "enabling the Four-Level Topology Store, OR disable both for "
                "the sidecar/BPT-direct path.");
        }

        // l1_cache_mb / l2_cache_mb are size_t and therefore non-negative
        // by construction; the GQL parser layer is responsible for
        // rejecting negative integer literals before they reach this
        // struct.
    }

    /**
     * @brief Get the number of GNN layers (K).
     */
    size_t num_layers() const {
        return fanouts.size();
    }

    /**
     * @brief Estimate maximum nodes per batch (upper bound).
     *
     * Actual count is typically lower due to overlap and degree variance.
     */
    uint64_t estimated_max_nodes_per_batch() const {
        uint64_t total = batch_size;  // Layer 0 (seeds)
        uint64_t layer_size = batch_size;

        for (uint64_t fanout : fanouts) {
            layer_size *= fanout;
            total += layer_size;
        }

        return total;
    }
};

} // namespace mdb::gnn
