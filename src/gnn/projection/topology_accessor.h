#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <torch/torch.h>

#include "gnn/projection/edge_orientation.h"
#include "gnn/projection/four_level_topology_store.h"  // FourLevelTopologyStore::Config
#include "graph_models/object_id.h"

namespace GQL {
class ProjectionStorage;
namespace Projection {
class TopologySnapshotReader;
}
}

namespace mdb::gnn {

/**
 * @brief Neighbor information for a single node.
 */
struct Neighbors {
    std::vector<ObjectId> node_ids;   ///< Neighbor node IDs
    std::vector<ObjectId> edge_ids;   ///< Corresponding edge IDs
};

/**
 * @brief Edge index in COO format for GNN message passing.
 */
struct EdgeIndex {
    torch::Tensor edge_index;         ///< [2, num_edges] source and target indices
    int64_t num_src_nodes;            ///< Number of source nodes
    int64_t num_dst_nodes;            ///< Number of destination nodes

    int64_t num_edges() const { return edge_index.size(1); }
};

/**
 * @brief Sampled subgraph for mini-batch training.
 */
struct SampledSubgraph {
    std::vector<ObjectId> src_nodes;              ///< Source layer nodes
    std::vector<ObjectId> dst_nodes;              ///< Destination layer nodes (seeds)
    EdgeIndex edge_index;                         ///< Edges between layers (local indices)

    std::unordered_map<uint64_t, int64_t> src_id_to_idx;  ///< ObjectId.id -> local index
    std::unordered_map<uint64_t, int64_t> dst_id_to_idx;  ///< ObjectId.id -> local index
};

// EdgeOrientation is now defined in `gnn/projection/edge_orientation.h` so
// lightweight consumers (sampling_config.h, four_level_topology_store.h)
// can use the enum without including this entire header. The include at
// the top of this file re-introduces the enum into this translation unit
// for backwards-compatible access via `mdb::gnn::EdgeOrientation`.

/**
 * @brief Sampling strategy for neighbor selection.
 *
 * Determines how neighbors are selected during k-hop sampling.
 */
enum class SamplingStrategy {
    UNIFORM,      ///< Uniform random sampling (GraphSAGE default)
    FULL,         ///< Return all neighbors (no sampling)
    IMPORTANCE,   ///< Degree-weighted sampling (higher degree = higher probability)
    RANDOM_WALK   ///< Random walk-based sampling (PinSAGE style)
};

/**
 * @brief Batch sampling strategy for B+Tree traversal.
 *
 * Controls how edges are accessed when sampling neighbors for a batch of nodes.
 * This affects I/O patterns and performance for different batch characteristics.
 *
 * ## Performance Comparison
 *
 * | Strategy | Sparse Batches | Dense Batches | Best For |
 * |----------|----------------|---------------|----------|
 * | AUTO     | Adaptive       | Adaptive      | Most cases |
 * | SWEEP    | O(E_range)     | O(E_range)    | Dense, sequential access |
 * | SEEK     | O(B × log E)   | O(B × log E)  | Sparse, large gaps |
 * | PER_NODE | O(B × log E)   | O(B × log E)  | Very small batches |
 *
 * Where:
 * - E_range = edges in the ID range [min_node, max_node]
 * - B = batch size
 * - E = total edges
 *
 * @see LeapfrogGnnSampler for SWEEP implementation
 * @see SeekBasedGnnSampler for SEEK implementation
 */
enum class BatchStrategy {
    AUTO,      ///< Automatically choose based on batch characteristics (recommended)
    SWEEP,     ///< Coordinated B+Tree sweep (LeapfrogGnnSampler)
    SEEK,      ///< Individual O(log E) seeks per node (SeekBasedGnnSampler)
    PER_NODE   ///< Per-node lookups via TopologyAccessor (best for <10 nodes)
};

/**
 * @brief Streaming iterator over projection nodes.
 *
 * Provides memory-efficient iteration over nodes without loading all into memory.
 * Uses the underlying B+tree index for sequential access.
 *
 * Usage patterns:
 * @code
 *   // Single node iteration
 *   NodeIterator iter(storage);
 *   while (auto node_id = iter.next()) {
 *       process(*node_id);
 *   }
 *
 *   // Batch iteration (more efficient)
 *   NodeIterator iter(storage);
 *   while (auto batch = iter.next_batch(1000)) {
 *       for (const auto& node_id : *batch) {
 *           process(node_id);
 *       }
 *   }
 * @endcode
 *
 * Memory: O(1) for single iteration, O(batch_size) for batch iteration
 * Performance: Sequential B+tree scan, optimal for full-graph processing
 *
 * @see TopologyAccessor for neighbor-based iteration
 */
class NodeIterator {
public:
    /**
     * @brief Construct iterator over all nodes in projection.
     * @param storage Reference to projection storage (must outlive iterator)
     */
    explicit NodeIterator(GQL::ProjectionStorage& storage);

    ~NodeIterator();

    // Disable copy (holds B+tree cursor state)
    NodeIterator(const NodeIterator&) = delete;
    NodeIterator& operator=(const NodeIterator&) = delete;

    // Allow move
    NodeIterator(NodeIterator&&) noexcept;
    NodeIterator& operator=(NodeIterator&&) noexcept;

    /**
     * @brief Get next node ID.
     * @return Next ObjectId, or std::nullopt if exhausted
     */
    std::optional<ObjectId> next();

    /**
     * @brief Get next batch of node IDs.
     *
     * More efficient than calling next() repeatedly due to reduced
     * function call overhead and potential memory locality benefits.
     *
     * @param batch_size Maximum nodes to retrieve
     * @return Vector of ObjectIds, or std::nullopt if exhausted
     *         May return fewer than batch_size nodes at end of iteration
     */
    std::optional<std::vector<ObjectId>> next_batch(size_t batch_size);

    /**
     * @brief Reset iterator to beginning.
     */
    void reset();

    /**
     * @brief Check if more nodes are available.
     */
    bool has_next() const;

    /**
     * @brief Get total node count (for progress tracking).
     */
    uint64_t total_count() const;

    /**
     * @brief Get count of nodes already iterated.
     */
    uint64_t iterated_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Provides neighbor traversal for GNN message passing.
 *
 * Enables efficient graph traversal through projection's edge indexes.
 * Supports:
 * - Outgoing neighbors (from → to)
 * - Incoming neighbors (to → from)
 * - Undirected neighbors (both directions)
 * - Neighbor sampling for mini-batch training
 *
 * Usage:
 * @code
 *   TopologyAccessor topology(projection_storage);
 *   auto neighbors = topology.get_out_neighbors(node_id);
 *   auto sampled = topology.sample_neighbors(seeds, fanout);
 * @endcode
 *
 * @see ProjectionStorage for underlying indexes
 * @see SampledSubgraph for sampling result
 */
class TopologyAccessor {
public:
    /**
     * @brief Construct accessor for a projection.
     * @param storage Reference to the projection storage (must outlive accessor)
     */
    explicit TopologyAccessor(GQL::ProjectionStorage& storage);

    ~TopologyAccessor();

    // Disable copy, allow move
    TopologyAccessor(const TopologyAccessor&) = delete;
    TopologyAccessor& operator=(const TopologyAccessor&) = delete;
    TopologyAccessor(TopologyAccessor&&) noexcept;
    TopologyAccessor& operator=(TopologyAccessor&&) noexcept;

    // =========================================================================
    // Single Node Neighbor Access
    // =========================================================================

    /**
     * @brief Get outgoing neighbors (node → neighbors).
     * @param node_id Source node
     * @return Neighbors reachable via outgoing edges
     */
    Neighbors get_out_neighbors(ObjectId node_id);

    /**
     * @brief Get incoming neighbors (neighbors → node).
     * @param node_id Target node
     * @return Neighbors with edges pointing to this node
     */
    Neighbors get_in_neighbors(ObjectId node_id);

    /**
     * @brief Get all neighbors (both directions for undirected traversal).
     * @param node_id Center node
     * @return All connected neighbors
     */
    Neighbors get_neighbors(ObjectId node_id);

    /**
     * @brief Get neighbors with explicit orientation control.
     *
     * This is the primary orientation-aware method for GNN message passing.
     *
     * @param node_id Center node
     * @param orientation How to traverse edges
     * @return Neighbors based on orientation:
     *         - NATURAL: outgoing neighbors (node → neighbors)
     *         - REVERSE: incoming neighbors (neighbors → node)
     *         - UNDIRECTED: all neighbors (deduplicated)
     */
    Neighbors get_neighbors(ObjectId node_id, EdgeOrientation orientation);

    /**
     * @brief Buffer-reusing variant of get_neighbors(node, orientation).
     *
     * Writes the neighbors into the caller-owned `out` (whose capacity is
     * retained across calls) instead of allocating a fresh Neighbors per call.
     * The content and order are byte-identical to get_neighbors(); only the
     * per-node allocation is avoided. Each parallel sampler worker passes its
     * own scratch buffer (single-threaded per worker), so no synchronisation is
     * needed. This is the hot path for k-hop sampling — the per-node Neighbors
     * allocation in the UNDIRECTED fetch was measured to be the dominant expand
     * sub-cost.
     */
    void get_neighbors_into(ObjectId node_id, EdgeOrientation orientation,
                            Neighbors& out);

    /// Buffer-reusing variants of get_out_neighbors / get_in_neighbors — see
    /// get_neighbors_into. Fill `out` (capacity retained); byte-identical result.
    void get_out_neighbors_into(ObjectId node_id, Neighbors& out);
    void get_in_neighbors_into(ObjectId node_id, Neighbors& out);

    // =========================================================================
    // Batch Neighbor Access
    // =========================================================================

    /**
     * @brief Get outgoing neighbors for multiple nodes.
     * @param node_ids Source nodes
     * @return Map of node_id -> Neighbors
     */
    std::unordered_map<uint64_t, Neighbors> get_batch_out_neighbors(
        const std::vector<ObjectId>& node_ids
    );

    /**
     * @brief Get incoming neighbors for multiple nodes.
     */
    std::unordered_map<uint64_t, Neighbors> get_batch_in_neighbors(
        const std::vector<ObjectId>& node_ids
    );

    /**
     * @brief Get neighbors for multiple nodes with orientation control.
     *
     * @param node_ids Source nodes
     * @param orientation How to traverse edges
     * @return Map of node_id -> Neighbors
     */
    std::unordered_map<uint64_t, Neighbors> get_batch_neighbors(
        const std::vector<ObjectId>& node_ids,
        EdgeOrientation orientation
    );

    // =========================================================================
    // Edge Index Construction
    // =========================================================================

    /**
     * @brief Build edge index from node set (all edges within set).
     *
     * Useful for building adjacency for a subgraph.
     *
     * @param node_ids Nodes to include
     * @return EdgeIndex with local indices
     */
    EdgeIndex build_edge_index(const std::vector<ObjectId>& node_ids);

    /**
     * @brief Build bipartite edge index (src → dst).
     *
     * Edges go from src_nodes to dst_nodes only.
     *
     * @param src_nodes Source layer nodes
     * @param dst_nodes Destination layer nodes
     * @return EdgeIndex for message passing
     */
    EdgeIndex build_bipartite_edge_index(
        const std::vector<ObjectId>& src_nodes,
        const std::vector<ObjectId>& dst_nodes
    );

    // =========================================================================
    // Neighbor Sampling
    // =========================================================================

    /**
     * @brief Sample neighbors for a batch of seed nodes.
     *
     * This is the primary interface for mini-batch GNN training.
     * Samples up to `fanout` neighbors for each seed node.
     *
     * @param seed_nodes Nodes to sample neighbors for
     * @param fanout Maximum neighbors per node (-1 for all)
     * @param strategy Sampling strategy
     * @param orientation Edge traversal direction (default: REVERSE for GNN message passing)
     * @return SampledSubgraph with nodes and edges
     *
     * @note Default REVERSE orientation matches typical GNN convention where
     *       messages flow from neighbors (src) to seeds (dst).
     */
    SampledSubgraph sample_neighbors(
        const std::vector<ObjectId>& seed_nodes,
        int64_t fanout,
        SamplingStrategy strategy = SamplingStrategy::UNIFORM,
        EdgeOrientation orientation = EdgeOrientation::REVERSE
    );

    /**
     * @brief Sample incoming neighbors for seed nodes (legacy method).
     *
     * Equivalent to sample_neighbors with EdgeOrientation::REVERSE.
     * Kept for backward compatibility.
     *
     * @deprecated Use sample_neighbors with explicit orientation instead.
     */
    SampledSubgraph sample_in_neighbors(
        const std::vector<ObjectId>& seed_nodes,
        int64_t fanout,
        SamplingStrategy strategy = SamplingStrategy::UNIFORM
    );

    /**
     * @brief Multi-layer neighbor sampling (k-hop).
     *
     * Samples k-hop neighborhood for GNN with k layers.
     *
     * @param seed_nodes Initial seed nodes
     * @param fanouts Fanout per layer [layer_0_fanout, layer_1_fanout, ...]
     * @param strategy Sampling strategy
     * @param orientation Edge traversal direction (default: REVERSE)
     * @return Vector of SampledSubgraph, one per layer
     */
    std::vector<SampledSubgraph> sample_khop_neighbors(
        const std::vector<ObjectId>& seed_nodes,
        const std::vector<int64_t>& fanouts,
        SamplingStrategy strategy = SamplingStrategy::UNIFORM,
        EdgeOrientation orientation = EdgeOrientation::REVERSE
    );

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get degree of a node (out-degree).
     */
    int64_t get_out_degree(ObjectId node_id);

    /**
     * @brief Get in-degree of a node.
     */
    int64_t get_in_degree(ObjectId node_id);

    /**
     * @brief Get total edge count.
     */
    uint64_t get_edge_count() const;

    /**
     * @brief Get total node count.
     */
    uint64_t get_node_count() const;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set random seed for sampling reproducibility.
     */
    void set_random_seed(uint64_t seed);

    /**
     * @brief Set target device for edge index tensors.
     */
    void set_target_device(torch::Device device);

    // =========================================================================
    // In-memory adjacency cache (one full B+Tree scan into hash map)
    // =========================================================================
    //
    // Opt-in performance path that materialises every projection edge into a
    // pair of in-memory `unordered_map<uint64_t, vector<AdjEntry>>` instances
    // (forward + reverse) by full-scanning the underlying B+Tree edge indexes
    // ONCE. Subsequent `get_out_neighbors` / `get_in_neighbors` /
    // `get_neighbors(UNDIRECTED)` calls then resolve in O(degree) via hash
    // lookups instead of paying O(page_tuples) per range query against the
    // live B+Tree.
    //
    // Memory cost: ~16 bytes per directed edge (uint64 neighbor + uint64
    // edge_id) plus hash-map overhead. ogbn-products (62 M edges, both
    // directions): ~3 GB — well inside a 31 GB commodity-RAM budget. Disabled
    // by default; callers must explicitly opt in.
    //
    // Mirrors the design of the EmbeddingWriter Phase B cache (commit
    // 6521cc21cf) but lives in TopologyAccessor so every consumer
    // (BasicKHopSampler, EmbeddingWriter, future gnn_predict, etc.) can
    // automatically benefit.

    /// Single adjacency entry: neighbor node id + edge id (raw uint64 form,
    /// without the 8-bit ObjectId type tag — added back when the cached
    /// entries are converted to ObjectIds at lookup time).
    struct AdjEntry {
        uint64_t node_id;
        uint64_t edge_id;
    };

    /// Enable the adjacency cache. Call before `prebuild_adjacency_cache(...)`
    /// or before any `get_*_neighbors` call you want to benefit. Subsequent
    /// disables are honoured (cache emptied, reverts to B+Tree path).
    void enable_adjacency_cache(bool enabled);

    /// Returns true if the cache is currently enabled.
    bool is_adjacency_cache_enabled() const;

    /// Eagerly populate the adjacency cache for the given orientation. For
    /// UNDIRECTED this scans both `from_to_edge` and `to_from_edge`; for
    /// NATURAL only `from_to_edge`; for REVERSE only `to_from_edge`.
    /// No-op if the corresponding direction is already built. Returns the
    /// number of milliseconds the build took (for instrumentation purposes).
    ///
    /// Pre-condition: `enable_adjacency_cache(true)` must have been called.
    /// If the cache is disabled this is a silent no-op returning 0 ms.
    uint64_t prebuild_adjacency_cache(EdgeOrientation orientation);

    /// Returns true if at least the side of the cache needed for `orientation`
    /// has been populated (i.e. fwd for NATURAL, rev for REVERSE, both for
    /// UNDIRECTED). Otherwise the accessor will fall back to the B+Tree path.
    bool is_adjacency_cache_built(EdgeOrientation orientation) const;

    /// Approximate resident-memory cost of the cache in bytes (entries +
    /// rough hash-map overhead). 0 when the cache is unbuilt or disabled.
    uint64_t get_adjacency_cache_size_bytes() const;

    /// Number of (src, neighbor, edge) triples currently held in the
    /// forward cache (count of directed entries).
    uint64_t get_adjacency_cache_fwd_entries() const;

    /// Number of (src, neighbor, edge) triples currently held in the
    /// reverse cache.
    uint64_t get_adjacency_cache_rev_entries() const;

    // =========================================================================
    // Four-Level Topology Store (frequency-tiered RAM/mmap/B+Tree cache)
    // =========================================================================
    //
    // When the four-level store is enabled, every neighbor lookup is
    // routed through it INSTEAD of the flat in-memory adjacency cache /
    // mmap CSR sidecar / B+Tree dispatch chain. The store is a
    // frequency-tiered cache — L1 RAM hash (hottest hubs), L2 RAM
    // compact uint32 CSR (warm nodes), L3 mmap CSR sidecar
    // (topology_{fwd,rev}.csr, cold nodes), L4 direct B+Tree (fallback
    // when no sidecar is present) — that targets commodity-RAM scenarios
    // on >100M-node graphs where the flat in-memory adjacency cache
    // (one unordered_map per direction holding all edges) would exhaust
    // available RAM.
    //
    // When unset (default), all existing dispatch logic is preserved
    // byte-for-byte — the regression contract every existing test
    // relies on.

    /**
     * @brief Build and enable the frequency-tiered Four-Level Topology Store.
     *
     * Constructs a `FourLevelTopologyStore` over the underlying
     * projection's B+Tree edge indexes, runs `build()` to populate
     * tiers, and registers the result as the dispatch target for
     * every subsequent `get_*_neighbors` call.
     *
     * Once enabled, the flat in-memory adjacency cache (which holds all
     * edges in a single unordered_map per direction) is bypassed: the
     * four-level store's own L1 RAM hash cache subsumes that role, and
     * populating both structures simultaneously would double-count RAM.
     * Calling `enable_adjacency_cache(false)` after this method is a no-op.
     *
     * Idempotent: calling twice throws `std::logic_error`. To swap
     * configurations, drop the accessor and recreate.
     *
     * Caller must include `gnn/projection/four_level_topology_store.h`
     * for the `Config` type. The forward-declared
     * `FourLevelTopologyStore` symbol above keeps this header
     * lightweight while letting the .cc file resolve the full type.
     *
     * @throws std::logic_error when called more than once on the same
     *         accessor.
     * @throws std::runtime_error on any underlying I/O / build
     *         failure.
     */
    void enable_four_level_store(
        const FourLevelTopologyStore::Config& config);

    /// Returns true iff `enable_four_level_store()` has run successfully.
    bool is_four_level_store_enabled() const;

    /// Returns true iff the four-level store is enabled AND its symmetric
    /// (pre-merged undirected) tier was populated — i.e. UNDIRECTED neighbor
    /// fetches resolve via a single pre-merged dispatch rather than the runtime
    /// out+in+merge.
    bool is_symmetric_topology_built() const;

    /// Resident bytes of the four-level store's in-RAM merged undirected slice
    /// (the GPU-UVA single-slice substrate); 0 when absent. For telemetry.
    std::size_t symmetric_ram_bytes() const;

    /// Resident bytes of the four-level store's in-RAM tiers (L1+L2, both
    /// directions + symmetric); 0 when no store. On the symmetric GPU path these
    /// directional tiers are superseded by the single pinned slice, so the
    /// engine treats this as freeable headroom for the GPU/CPU decision.
    std::size_t four_level_ram_used() const;

    /**
     * @brief Tier-2 compact CSR per direction, for SIZING the dynamic GPU/CPU
     *        sampling backend decision (`plan_sampling_backend`).
     *
     * Delegates to the Four-Level Topology Store when enabled; returns nullptr
     * otherwise (the GPU sampling path needs that in-RAM CSR to pin, so an
     * absent store means the decision falls back to the CPU backend). The
     * returned object is frozen/immutable post-build; callers read only
     * node_count()/edge_count() and must NOT cache its raw data pointers.
     */
    const L2CompactCsr* l2_fwd() const;
    const L2CompactCsr* l2_rev() const;

    /**
     * @brief The active L3 global topology sidecar reader per direction, for
     *        SIZING the GPU sampling-backend decision on the substrate actually
     *        pinned (the full graph), not the warm-tier L2. nullptr when absent.
     */
    const GQL::Projection::TopologySnapshotReader* l3_fwd() const;
    const GQL::Projection::TopologySnapshotReader* l3_rev() const;

    /**
     * @brief Pin the global topology CSR as a device-visible GPU substrate.
     *
     * Delegates to the Four-Level Topology Store's `enable_pinned_gpu_view`.
     * No-op when the store is absent, the plan chose the CPU backend, no GPU is
     * present, or the build has no CUDA. Mutates only the device-page
     * registration (the topology data stays read-only), so the sampling output
     * is unchanged when the view is unused.
     */
    void enable_pinned_gpu_view(const SamplingBackendPlan& plan);

    /// The registered GPU view, or nullptr when none was enabled / no store.
    const PinnedTopologyView* pinned_view() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
