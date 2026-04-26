#pragma once

#include <cstdint>
#include <functional>
#include <utility>

#include "graph_models/object_id.h"

template<std::size_t N>
class BPlusTree;

namespace GQL {

/**
 * @brief Performs direct B+Tree range scans for native projection.
 *
 * Provides low-level access to label_node, label_edge, and from_to_edge indexes
 * with callback-based iteration for memory efficiency.
 *
 * NOTE: This is a forward declaration header. The implementation will be provided
 * by another agent. This allows NativeProjectionBuilder to compile without the
 * full scanner implementation.
 */
class NativeScanner {
public:
    /**
     * @brief Constructs scanner with references to GQLModel indexes.
     *
     * @param label_node_idx Pointer to label_node B+Tree (non-owning)
     *                       Format: {label_id, node_id}
     * @param label_edge_idx Pointer to label_edge B+Tree (non-owning)
     *                       Format: {label_id, edge_id}
     * @param from_to_edge_idx Pointer to from_to_edge B+Tree (non-owning)
     *                         Format: {from, to, edge_id} - for DIRECTED edges
     * @param edge_from_to_idx Optional pointer to edge_from_to B+Tree (non-owning)
     *                         Format: {edge_id, from, to} - for DIRECTED edges
     *                         Much faster for edge endpoint lookup. If nullptr, falls back to scanning from_to_edge
     * @param n1_n2_edge_idx Pointer to n1_n2_edge B+Tree (non-owning)
     *                       Format: {n1, n2, edge_id} - for UNDIRECTED edges
     * @param edge_n1_n2_idx Optional pointer to edge_n1_n2 B+Tree (non-owning)
     *                       Format: {edge_id, n1, n2} - for UNDIRECTED edges
     *                       Much faster for edge endpoint lookup. If nullptr, falls back to scanning n1_n2_edge
     */
    NativeScanner(
        BPlusTree<2>* label_node_idx,
        BPlusTree<2>* label_edge_idx,
        BPlusTree<3>* from_to_edge_idx,
        BPlusTree<3>* edge_from_to_idx,
        BPlusTree<3>* n1_n2_edge_idx,
        BPlusTree<3>* edge_n1_n2_idx = nullptr
    );

    ~NativeScanner();

    /**
     * @brief Scans all nodes with the given label.
     *
     * Algorithm:
     *   min_key = [label_id, ObjectId::MIN]
     *   max_key = [label_id, ObjectId::MAX]
     *   for each record in range:
     *     callback(record[1])  // node_id
     *
     * Behavior is controlled by the env var MDB_PROJECTION_PARALLEL_NODE_SCAN
     * (default ON; "0"/"false"/"off" disables). When enabled, the
     * label-node id range is split into K disjoint sub-ranges, each scanned
     * in parallel by a TBB worker into a thread-local vector. After all
     * workers finish, the main thread invokes `callback` sequentially over
     * the per-partition vectors in ascending sub-range order. This preserves
     * the observed callback ordering of the legacy single-threaded path
     * (records are visited in B+Tree key order, partition-by-partition) and
     * keeps the callback contract single-threaded — callers do NOT need to
     * make their callback thread-safe.
     *
     * Partition count K is `min(hardware_concurrency, 16)` by default; can be
     * overridden via MDB_PROJECTION_NODE_SCAN_PARTITIONS (clamped [2, 64]).
     * For label-ranges with fewer than K records, falls back automatically
     * to the sequential path.
     *
     * @param label_id ObjectId of the label
     * @param callback Function called for each node_id found
     * @return Number of nodes scanned
     */
    uint64_t scan_label_node(
        ObjectId label_id,
        std::function<void(ObjectId)> callback
    );

    /**
     * @brief Scans all edges with the given type.
     *
     * Algorithm:
     *   min_key = [type_id, ObjectId::MIN]
     *   max_key = [type_id, ObjectId::MAX]
     *   for each record in range:
     *     callback(record[1])  // edge_id
     *
     * @param type_id ObjectId of the edge type
     * @param callback Function called for each edge_id found
     * @return Number of edges scanned
     */
    uint64_t scan_label_edge(
        ObjectId type_id,
        std::function<void(ObjectId)> callback
    );

    /**
     * @brief Gets the endpoints of an edge (handles both directed and undirected).
     *
     * Detects edge type by examining ObjectId mask:
     * - DIRECTED (0xe0): Uses edge_from_to or from_to_edge indexes
     * - UNDIRECTED (0xe4): Uses edge_n1_n2 or n1_n2_edge indexes
     *
     * @param edge_id ObjectId of the edge
     * @return Pair of (from_node, to_node) for directed, (n1, n2) for undirected
     * @throws std::runtime_error if edge not found
     */
    std::pair<ObjectId, ObjectId> get_edge_endpoints(ObjectId edge_id);

    /**
     * @brief Scans all edges with the given type AND returns endpoints in single pass.
     *
     * OPTIMIZATION: Combines scan_label_edge() + get_edge_endpoints() to eliminate
     * redundant B+Tree lookups. For each edge found, immediately looks up endpoints
     * from edge_from_to/edge_n1_n2 index while data is hot in cache.
     *
     * Algorithm:
     *   for each edge_id in label_edge where label == type_id:
     *     endpoints = lookup edge_id in edge_from_to or edge_n1_n2
     *     callback(edge_id, from_node, to_node)
     *
     * Performance: ~30-40% faster than scan_label_edge + get_edge_endpoints
     * due to reduced function call overhead and better cache locality.
     *
     * Behavior is controlled by the env var MDB_PROJECTION_PARALLEL_EDGE_SCAN
     * (default ON; "0"/"false"/"off" disables). When enabled, the
     * label-edge id range is split into K disjoint sub-ranges, each scanned
     * in parallel by a TBB worker into a thread-local vector of
     * (edge_id, from, to) triples. Per-edge endpoint lookups happen inside
     * each worker (the secondary edge_from_to / edge_n1_n2 trees are
     * thread-safe for concurrent reads). After all workers finish, the
     * main thread invokes `callback` sequentially over the per-partition
     * vectors in ascending sub-range order — preserving the observed
     * callback ordering of the legacy single-threaded path AND the
     * single-threaded callback contract (callers do NOT need to make
     * their callback thread-safe; in particular the
     * ParallelEdgeDetector aggregation state in native_projection_builder
     * is mutated only on the parent thread).
     *
     * Partition count K is `min(hardware_concurrency, 16)` by default; can
     * be overridden via MDB_PROJECTION_EDGE_SCAN_PARTITIONS (clamped [2, 64]).
     *
     * @param type_id ObjectId of the edge type
     * @param callback Function called for each (edge_id, from, to) tuple found
     * @return Number of edges scanned
     */
    uint64_t scan_label_edge_with_endpoints(
        ObjectId type_id,
        std::function<void(ObjectId, ObjectId, ObjectId)> callback
    );

    /**
     * @brief Counts edges with the given type without full enumeration.
     *
     * Uses B+Tree range scan to count edges efficiently.
     * Useful for estimating dataset size before choosing algorithm.
     *
     * @param type_id ObjectId of the edge type
     * @return Number of edges with this type
     */
    uint64_t count_edges_by_type(ObjectId type_id);

private:
    // Non-owning pointers to GQLModel indexes
    BPlusTree<2>* label_node_index;
    BPlusTree<2>* label_edge_index;

    // Directed edge indexes
    BPlusTree<3>* from_to_edge_index;  // {from, to, edge_id}
    BPlusTree<3>* edge_from_to_index;  // {edge_id, from, to} - Optional, O(log n) lookup

    // Undirected edge indexes
    BPlusTree<3>* n1_n2_edge_index;    // {n1, n2, edge_id}
    BPlusTree<3>* edge_n1_n2_index;    // {edge_id, n1, n2} - Optional, O(log n) lookup
};

} // namespace GQL
