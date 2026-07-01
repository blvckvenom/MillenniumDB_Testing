#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "graph_models/object_id.h"

namespace mdb::gnn {

/**
 * @brief Data split type for train/validation/test separation.
 *
 * Used to categorize samples during offline pre-computation.
 */
enum class SplitType {
    TRAIN,       ///< Training data (default: 70%)
    VALIDATION,  ///< Validation data (default: 15%)
    TEST         ///< Test data (default: 15%)
};

/**
 * @brief Edge connectivity for a single GNN layer.
 *
 * Represents the message-passing edges between two adjacent layers.
 * Uses local indices (relative to layer's node list) for efficient
 * sparse matrix construction.
 *
 * GNN Convention:
 * - src_indices: Indices in source layer (layer k+1, further from seeds)
 * - dst_indices: Indices in destination layer (layer k, closer to seeds)
 * - Messages flow from src -> dst (toward the seed nodes)
 *
 * Example for 2-hop GraphSAGE:
 * @code
 *   Layer 2 (input)  --edges_per_layer[1]-->  Layer 1  --edges_per_layer[0]-->  Layer 0 (seeds)
 * @endcode
 */
struct LayerEdges {
    std::vector<int32_t> src_indices;   ///< Source node indices (in layer k+1)
    std::vector<int32_t> dst_indices;   ///< Destination node indices (in layer k)
    std::vector<ObjectId> edge_ids;     ///< Original edge IDs (for edge feature lookup)

    /**
     * @brief Number of edges in this layer connection.
     */
    size_t size() const {
        return src_indices.size();
    }

    /**
     * @brief Check if edge data is consistent.
     */
    bool is_valid() const {
        return src_indices.size() == dst_indices.size() &&
               dst_indices.size() == edge_ids.size();
    }

    /**
     * @brief Reserve memory for expected number of edges.
     */
    void reserve(size_t n) {
        src_indices.reserve(n);
        dst_indices.reserve(n);
        edge_ids.reserve(n);
    }

    /**
     * @brief Clear all edge data.
     */
    void clear() {
        src_indices.clear();
        dst_indices.clear();
        edge_ids.clear();
    }
};

/**
 * @brief A pre-computed GNN mini-batch (computational graph).
 *
 * Represents all nodes and edges needed to compute embeddings for
 * a batch of seed nodes using K-hop neighborhood aggregation.
 *
 * ## Memory Layout
 *
 * The computational graph is stored in "message-passing order":
 * - `nodes_per_layer[0]` = seed nodes (output layer)
 * - `nodes_per_layer[K]` = K-hop neighbors (input layer)
 *
 * ## GNN Forward Pass
 *
 * During training, computation flows in reverse:
 * @code
 *   // Layer K (input features)
 *   H[K] = X[nodes_per_layer[K]]
 *
 *   // Message passing: K -> K-1 -> ... -> 0
 *   for (k = K-1; k >= 0; k--) {
 *       H[k] = aggregate(H[k+1], edges_per_layer[k])
 *   }
 *
 *   // H[0] contains embeddings for seed nodes
 * @endcode
 *
 * ## Serialization
 *
 * Binary format (little-endian):
 * - Magic: 4 bytes "GNSM"
 * - Version: 4 bytes
 * - batch_id, split
 * - num_layers, then for each layer:
 *   - num_nodes, node_ids
 * - For each layer connection:
 *   - num_edges, src_indices, dst_indices, edge_ids
 *
 * @see SamplingConfig for sampling parameters
 * @see OfflineSamplingEngine for batch generation
 */
struct GraphSample {
    // =========================================================================
    // Identification
    // =========================================================================

    uint64_t batch_id;   ///< Unique batch identifier within this sample set
    SplitType split;     ///< Train/validation/test partition

    // =========================================================================
    // Graph Structure
    // =========================================================================

    /**
     * @brief Nodes at each layer of the computational graph.
     *
     * - Index 0: Seed nodes (targets for embedding computation)
     * - Index K: K-hop neighbors (input features)
     *
     * Size: K+1 where K = number of GNN layers (sampling hops)
     */
    std::vector<std::vector<ObjectId>> nodes_per_layer;

    /**
     * @brief Edges connecting adjacent layers.
     *
     * - edges_per_layer[k] connects nodes_per_layer[k+1] to nodes_per_layer[k]
     * - Messages flow from higher layer index to lower
     *
     * Size: K (one less than nodes_per_layer)
     */
    std::vector<LayerEdges> edges_per_layer;

    // =========================================================================
    // Derived Data
    // =========================================================================

    /**
     * @brief All unique nodes across all layers (for feature fetching).
     *
     * Deduplicated union of all nodes_per_layer vectors.
     * Ordered by first appearance (layer 0 first).
     */
    std::vector<ObjectId> all_unique_nodes;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Total nodes across all layers (with duplicates).
     */
    size_t total_nodes() const {
        size_t sum = 0;
        for (const auto& layer : nodes_per_layer) {
            sum += layer.size();
        }
        return sum;
    }

    /**
     * @brief Total unique nodes (for feature loading).
     */
    size_t unique_node_count() const {
        return all_unique_nodes.size();
    }

    /**
     * @brief Total edges across all layer connections.
     */
    size_t total_edges() const {
        size_t sum = 0;
        for (const auto& layer : edges_per_layer) {
            sum += layer.size();
        }
        return sum;
    }

    /**
     * @brief Number of GNN layers (K).
     *
     * For K-hop sampling, this equals the number of hops.
     */
    size_t num_layers() const {
        return nodes_per_layer.empty() ? 0 : nodes_per_layer.size() - 1;
    }

    /**
     * @brief Number of seed nodes in this batch.
     */
    size_t batch_size() const {
        return nodes_per_layer.empty() ? 0 : nodes_per_layer[0].size();
    }

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate structural consistency.
     *
     * Checks:
     * - edges_per_layer.size() == nodes_per_layer.size() - 1
     * - All edge indices are within layer bounds
     * - All LayerEdges are internally consistent
     *
     * @throws std::runtime_error if validation fails
     */
    void validate() const {
        if (nodes_per_layer.empty()) {
            throw std::runtime_error("GraphSample: nodes_per_layer cannot be empty");
        }

        if (edges_per_layer.size() != nodes_per_layer.size() - 1) {
            throw std::runtime_error(
                "GraphSample: edges_per_layer.size() must equal nodes_per_layer.size() - 1"
            );
        }

        for (size_t k = 0; k < edges_per_layer.size(); ++k) {
            const auto& edges = edges_per_layer[k];

            if (!edges.is_valid()) {
                throw std::runtime_error(
                    "GraphSample: LayerEdges[" + std::to_string(k) + "] is inconsistent"
                );
            }

            int32_t src_layer_size = static_cast<int32_t>(nodes_per_layer[k + 1].size());
            int32_t dst_layer_size = static_cast<int32_t>(nodes_per_layer[k].size());

            for (size_t i = 0; i < edges.size(); ++i) {
                if (edges.src_indices[i] < 0 || edges.src_indices[i] >= src_layer_size) {
                    throw std::runtime_error(
                        "GraphSample: src_index out of bounds at layer " + std::to_string(k)
                    );
                }
                if (edges.dst_indices[i] < 0 || edges.dst_indices[i] >= dst_layer_size) {
                    throw std::runtime_error(
                        "GraphSample: dst_index out of bounds at layer " + std::to_string(k)
                    );
                }
            }
        }
    }

    // =========================================================================
    // Utility
    // =========================================================================

    /**
     * @brief Rebuild all_unique_nodes from nodes_per_layer.
     *
     * Call this after modifying nodes_per_layer to ensure
     * all_unique_nodes is consistent.
     */
    void rebuild_unique_nodes() {
        all_unique_nodes.clear();
        std::unordered_set<uint64_t> seen;

        for (const auto& layer : nodes_per_layer) {
            for (const auto& node : layer) {
                if (seen.insert(node.id).second) {
                    all_unique_nodes.push_back(node);
                }
            }
        }
    }

    /**
     * @brief Clear all data and reset to empty state.
     */
    void clear() {
        batch_id = 0;
        split = SplitType::TRAIN;
        nodes_per_layer.clear();
        edges_per_layer.clear();
        all_unique_nodes.clear();
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Magic number for binary format: "GNSM" (GNN Sample)
    static constexpr uint32_t MAGIC = 0x4D534E47;  // "GNSM" in little-endian
    /// v1: legacy format with epoch field; v2: element-by-element binary I/O;
    /// v3: bulk binary I/O (hot-path serialization optimization).
    /// Reader accepts v1/v2/v3; writer always emits VERSION.
    static constexpr uint32_t VERSION_V2 = 2;
    static constexpr uint32_t VERSION_V3 = 3;
    static constexpr uint32_t VERSION = VERSION_V3;  // current write version

    /**
     * @brief Serialize to binary stream.
     *
     * @param out Output stream (must be open in binary mode)
     * @throws std::runtime_error on write failure
     */
    void serialize(std::ostream& out) const;

    /**
     * @brief Deserialize from binary stream.
     *
     * @param in Input stream (must be open in binary mode)
     * @param skip_edge_ids When true, the per-layer edge_ids blocks are seeked
     *        past instead of read — edges_per_layer[k].edge_ids comes back empty.
     *        edge_ids are ~half the serialized bytes of a deep-fanout sample and
     *        are NOT consumed by the training / embedding read path (only
     *        src_indices/dst_indices are), so skipping them roughly halves the
     *        per-batch sample read I/O. Default false preserves a full,
     *        round-trippable deserialize for every other caller.
     * @param skip_edges When true, the per-layer src_indices + dst_indices blocks
     *        are seeked past (leaving those vectors empty); used when a baked
     *        computation-graph block supplies the edge structure at train time.
     *        Caller MUST supply that edge structure from another source when
     *        enabled, or the assembled graph is edgeless and training is wrong.
     * @return Deserialized GraphSample
     * @throws std::runtime_error on read failure or invalid format
     */
    static GraphSample deserialize(std::istream& in, bool skip_edge_ids = false,
                                   bool skip_edges = false);

    /**
     * @brief Read only the split field from a serialized GraphSample.
     *
     * Reads the header (magic, version, batch_id, and optionally the
     * legacy epoch for v1) to extract the SplitType without
     * deserializing the full sample. This is much faster for building
     * the split index during init_read_mode().
     *
     * @param in Input stream positioned at the start of a serialized GraphSample
     * @return The SplitType of the sample
     * @throws std::runtime_error on read failure or invalid format
     */
    static SplitType read_split(std::istream& in);
};

} // namespace mdb::gnn
