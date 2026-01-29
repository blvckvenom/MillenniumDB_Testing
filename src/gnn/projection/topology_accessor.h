#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <torch/torch.h>

#include "graph_models/object_id.h"

namespace GQL {
class ProjectionStorage;
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

/**
 * @brief Edge orientation for neighbor traversal.
 *
 * Mirrors GQL::Procedures::Orientation but kept separate for module isolation.
 * Controls how edges are traversed during neighbor lookup and sampling.
 *
 * @see ISO/IEC 39075:2024 §4.3.5 (Undirected Edge Handling)
 */
enum class EdgeOrientation {
    NATURAL,     ///< Follow edge direction as stored (from → to)
    REVERSE,     ///< Reverse edge direction (to → from)
    UNDIRECTED   ///< Traverse both directions (bidirectional access)
};

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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
