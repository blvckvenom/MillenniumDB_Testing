#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "graph_models/object_id.h"
#include "gnn/common/feature_matrix.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

/**
 * @brief Thread-safe cache statistics for feature accessor.
 *
 * Uses atomic counters for thread-safe updates without locks.
 */
struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<size_t> cached_nodes{0};
    std::atomic<size_t> cache_size_bytes{0};

    CacheStats() = default;

    // Copy constructor for returning stats snapshots
    CacheStats(const CacheStats& other)
        : hits(other.hits.load(std::memory_order_relaxed)),
          misses(other.misses.load(std::memory_order_relaxed)),
          cached_nodes(other.cached_nodes.load(std::memory_order_relaxed)),
          cache_size_bytes(other.cache_size_bytes.load(std::memory_order_relaxed)) {}

    CacheStats& operator=(const CacheStats& other) {
        if (this != &other) {
            hits.store(other.hits.load(std::memory_order_relaxed), std::memory_order_relaxed);
            misses.store(other.misses.load(std::memory_order_relaxed), std::memory_order_relaxed);
            cached_nodes.store(other.cached_nodes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            cache_size_bytes.store(other.cache_size_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    double hit_rate() const {
        uint64_t h = hits.load(std::memory_order_relaxed);
        uint64_t m = misses.load(std::memory_order_relaxed);
        uint64_t total = h + m;
        return (total > 0) ? static_cast<double>(h) / total : 0.0;
    }

    void reset() {
        hits.store(0, std::memory_order_relaxed);
        misses.store(0, std::memory_order_relaxed);
        cached_nodes.store(0, std::memory_order_relaxed);
        cache_size_bytes.store(0, std::memory_order_relaxed);
    }
};

/**
 * @brief Provides efficient access to tensor features from projection nodes.
 *
 * Bridges MillenniumDB's projection storage with LibTorch tensors.
 * Optimized for:
 * - Batch access (many nodes at once)
 * - Repeated access patterns (optional caching)
 * - Multiple properties per node
 *
 * Usage:
 * @code
 *   FeatureAccessor accessor(projection_storage);
 *   auto fm = accessor.get_batch_features(node_ids, "embedding");
 *   // fm.features is [N, D] tensor on GPU
 * @endcode
 *
 * @see ProjectionStorage for underlying storage
 * @see FeatureMatrix for result structure
 */
class FeatureAccessor {
public:
    /**
     * @brief Construct accessor for a projection.
     * @param storage Reference to the projection storage (must outlive accessor)
     */
    explicit FeatureAccessor(GQL::ProjectionStorage& storage);

    ~FeatureAccessor();

    // Disable copy, allow move
    FeatureAccessor(const FeatureAccessor&) = delete;
    FeatureAccessor& operator=(const FeatureAccessor&) = delete;
    FeatureAccessor(FeatureAccessor&&) noexcept;
    FeatureAccessor& operator=(FeatureAccessor&&) noexcept;

    // =========================================================================
    // Single Node Access
    // =========================================================================

    /**
     * @brief Get a single feature vector for a node.
     * @param node_id Node to get features for
     * @param property_name Name of the tensor property
     * @return Tensor of shape [feature_dim] on preferred device
     * @throws std::runtime_error if node or property not found
     */
    torch::Tensor get_feature(ObjectId node_id, const std::string& property_name);

    /**
     * @brief Get multiple properties for a single node.
     * @return Map of property name to tensor
     */
    std::unordered_map<std::string, torch::Tensor> get_features(
        ObjectId node_id,
        const std::vector<std::string>& property_names
    );

    // =========================================================================
    // Batch Access
    // =========================================================================

    /**
     * @brief Get features for multiple nodes (single property).
     *
     * This is the primary interface for GNN training.
     *
     * @param node_ids Nodes to fetch features for
     * @param property_name Property to extract
     * @return FeatureMatrix with [N, D] tensor and id mappings
     * @throws std::runtime_error if any node missing the property
     */
    FeatureMatrix get_batch_features(
        const std::vector<ObjectId>& node_ids,
        const std::string& property_name
    );

    /**
     * @brief Get multiple properties for multiple nodes.
     * @return Map of property name to FeatureMatrix
     */
    std::unordered_map<std::string, FeatureMatrix> get_batch_features(
        const std::vector<ObjectId>& node_ids,
        const std::vector<std::string>& property_names
    );

    // =========================================================================
    // Edge Feature Access
    // =========================================================================

    /**
     * @brief Get a single feature vector for an edge.
     * @param edge_id Edge to get features for
     * @param property_name Name of the tensor property
     * @return Tensor of shape [feature_dim] on preferred device
     * @throws std::runtime_error if edge or property not found
     */
    torch::Tensor get_edge_feature(ObjectId edge_id, const std::string& property_name);

    /**
     * @brief Get features for multiple edges (single property).
     *
     * Useful for GNN models that use edge features.
     *
     * @param edge_ids Edges to fetch features for
     * @param property_name Property to extract
     * @return FeatureMatrix with [N, D] tensor and id mappings
     * @throws std::runtime_error if any edge missing the property
     */
    FeatureMatrix get_edge_batch_features(
        const std::vector<ObjectId>& edge_ids,
        const std::string& property_name
    );

    // =========================================================================
    // Concatenated Features
    // =========================================================================

    /**
     * @brief Get concatenated features (multiple properties as single tensor).
     *
     * Useful when model expects single input tensor.
     *
     * @param node_ids Nodes to fetch
     * @param property_names Properties to concatenate (in order)
     * @return FeatureMatrix with [N, D1+D2+...] tensor
     */
    FeatureMatrix get_concatenated_features(
        const std::vector<ObjectId>& node_ids,
        const std::vector<std::string>& property_names
    );

    // =========================================================================
    // Metadata
    // =========================================================================

    /**
     * @brief Get dimension of a tensor property.
     *
     * Samples the first node to determine dimension.
     *
     * @param property_name Property to check
     * @return Feature dimension, or -1 if property not found
     */
    int64_t get_dimension(const std::string& property_name);

    /**
     * @brief Check if property exists and is a tensor.
     */
    bool has_tensor_property(const std::string& property_name);

    /**
     * @brief Get all nodes in the projection.
     */
    std::vector<ObjectId> get_all_node_ids() const;

    /**
     * @brief Get total number of nodes.
     */
    uint64_t get_node_count() const;

    // =========================================================================
    // Caching
    // =========================================================================

    /**
     * @brief Enable feature caching (useful for repeated access).
     * @param max_cached_nodes Maximum nodes to cache
     */
    void enable_cache(size_t max_cached_nodes);

    /**
     * @brief Disable and clear cache.
     */
    void disable_cache();

    /**
     * @brief Clear feature cache.
     */
    void clear_cache();

    /**
     * @brief Get cache statistics.
     */
    CacheStats get_cache_stats() const;

    // =========================================================================
    // Advanced Options
    // =========================================================================

    /**
     * @brief Set target device for returned tensors.
     * @param device torch::kCPU or torch::kCUDA
     */
    void set_target_device(torch::Device device);

    /**
     * @brief Get current target device.
     */
    torch::Device get_target_device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdb::gnn
