#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "graph_models/object_id.h"
#include "gnn/projection/feature_accessor.h"
#include "gnn/projection/topology_accessor.h"

namespace GQL {
class ProjectionStorage;
class ProjectionCatalog;
class ProjectionManager;
}

namespace mdb::gnn {

/**
 * @brief Configuration for GnnProjectionAdapter.
 */
struct GnnAdapterConfig {
    size_t feature_cache_size = 10000;     ///< Max nodes to cache for feature access
    bool enable_feature_cache = true;       ///< Enable feature caching
    uint64_t random_seed = 42;              ///< Seed for reproducible sampling
    torch::Device device = torch::kCPU;     ///< Target device for tensors
};

/**
 * @brief Metadata about a projection for GNN use.
 */
struct ProjectionMetadata {
    std::string name;                        ///< Projection name
    uint64_t node_count;                     ///< Total number of nodes
    uint64_t edge_count;                     ///< Total number of edges
    std::vector<std::string> node_labels;    ///< Available node labels
    std::vector<std::string> edge_labels;    ///< Available edge labels
    std::vector<std::string> tensor_properties;  ///< Properties that are tensors
};

/**
 * @brief Mini-batch data ready for GNN forward pass.
 *
 * Contains everything needed for one GNN training step:
 * - Node features for the batch
 * - Edge index for message passing
 * - Optional labels for supervised training
 */
struct GnnMiniBatch {
    torch::Tensor features;                  ///< [batch_size, feature_dim]
    torch::Tensor edge_index;                ///< [2, num_edges] in COO format
    std::vector<ObjectId> node_ids;          ///< Original node IDs
    std::optional<torch::Tensor> labels;     ///< Optional labels [batch_size]

    int64_t batch_size() const { return features.size(0); }
    int64_t feature_dim() const { return features.dim() > 1 ? features.size(1) : 0; }
    int64_t num_edges() const { return edge_index.size(1); }
};

/**
 * @brief Multi-layer mini-batch for k-hop GNN architectures.
 *
 * For GNNs like GraphSAGE that aggregate multi-hop neighborhoods.
 */
struct GnnMultiLayerBatch {
    std::vector<SampledSubgraph> layers;     ///< Layer-wise subgraphs (from seeds outward)
    torch::Tensor root_features;             ///< Features of seed nodes [batch_size, dim]
    std::vector<ObjectId> seed_nodes;        ///< Original seed node IDs
    std::optional<torch::Tensor> labels;     ///< Optional labels for seeds

    int64_t num_layers() const { return static_cast<int64_t>(layers.size()); }
    int64_t batch_size() const { return static_cast<int64_t>(seed_nodes.size()); }
};

/**
 * @brief Unified interface for GNN operations on MillenniumDB projections.
 *
 * GnnProjectionAdapter bridges MillenniumDB's projection storage with
 * the GNN training pipeline. It provides:
 *
 * - **Feature access**: Reading tensor properties from nodes
 * - **Topology access**: Graph traversal for message passing
 * - **Mini-batch construction**: Building GNN-ready data structures
 * - **Multi-hop sampling**: k-hop neighborhood sampling for GraphSAGE-style models
 *
 * Architecture:
 * ```
 *   ProjectionStorage
 *         │
 *    ┌────┴────┐
 *    │         │
 * FeatureAccessor  TopologyAccessor
 *    │         │
 *    └────┬────┘
 *         │
 * GnnProjectionAdapter
 *         │
 *    ┌────┴────┐
 *    │         │
 * MiniBatch  MultiLayerBatch
 * ```
 *
 * Usage:
 * @code
 *   // Get projection from catalog
 *   auto* storage = catalog.get_projection("my_graph");
 *
 *   // Create adapter
 *   GnnProjectionAdapter adapter(*storage);
 *
 *   // Single-layer mini-batch
 *   auto batch = adapter.create_mini_batch(node_ids, "embedding");
 *
 *   // Multi-layer sampling for GraphSAGE
 *   auto ml_batch = adapter.sample_multi_layer(seeds, {25, 10}, "embedding");
 * @endcode
 *
 * Thread Safety:
 * - Individual method calls are thread-safe
 * - Multiple adapters can share the same ProjectionStorage
 * - Caching is per-adapter (not shared)
 *
 * @see FeatureAccessor for tensor property access details
 * @see TopologyAccessor for graph traversal details
 * @see ProjectionStorage for underlying storage
 */
class GnnProjectionAdapter {
public:
    /**
     * @brief Construct adapter for a projection.
     * @param storage Reference to the projection storage (must outlive adapter)
     * @param config Optional configuration
     */
    explicit GnnProjectionAdapter(
        GQL::ProjectionStorage& storage,
        const GnnAdapterConfig& config = GnnAdapterConfig{}
    );

    ~GnnProjectionAdapter();

    // Disable copy, allow move
    GnnProjectionAdapter(const GnnProjectionAdapter&) = delete;
    GnnProjectionAdapter& operator=(const GnnProjectionAdapter&) = delete;
    GnnProjectionAdapter(GnnProjectionAdapter&&) noexcept;
    GnnProjectionAdapter& operator=(GnnProjectionAdapter&&) noexcept;

    // =========================================================================
    // Accessor Access (for advanced use cases)
    // =========================================================================

    /**
     * @brief Get the feature accessor for direct property access.
     */
    FeatureAccessor& features();
    const FeatureAccessor& features() const;

    /**
     * @brief Get the topology accessor for direct graph traversal.
     */
    TopologyAccessor& topology();
    const TopologyAccessor& topology() const;

    // =========================================================================
    // Metadata
    // =========================================================================

    /**
     * @brief Get basic projection metadata from storage.
     *
     * For full metadata (name, labels, properties), use
     * get_metadata(ProjectionCatalog&) overload.
     */
    ProjectionMetadata get_metadata() const;

    /**
     * @brief Get full projection metadata including catalog info.
     * @param catalog The projection catalog containing additional metadata
     */
    ProjectionMetadata get_metadata(const GQL::ProjectionCatalog& catalog) const;

    /**
     * @brief Get all node IDs in the projection.
     */
    std::vector<ObjectId> get_all_nodes() const;

    /**
     * @brief Get total node count.
     */
    uint64_t get_node_count() const;

    /**
     * @brief Get total edge count.
     */
    uint64_t get_edge_count() const;

    /**
     * @brief Check if a property exists and is a tensor.
     */
    bool has_tensor_property(const std::string& property_name) const;

    /**
     * @brief Get feature dimension for a property.
     * @return Dimension, or -1 if property not found
     */
    int64_t get_feature_dimension(const std::string& property_name);

    // =========================================================================
    // Single-Layer Mini-Batch Construction
    // =========================================================================

    /**
     * @brief Create a mini-batch from node IDs.
     *
     * Builds a GNN-ready batch with features and edge index.
     *
     * @param node_ids Nodes to include in batch
     * @param feature_property Name of the tensor property for features
     * @return GnnMiniBatch ready for forward pass
     * @throws std::runtime_error if property not found or missing
     */
    GnnMiniBatch create_mini_batch(
        const std::vector<ObjectId>& node_ids,
        const std::string& feature_property
    );

    /**
     * @brief Create a mini-batch with concatenated features.
     *
     * @param node_ids Nodes to include
     * @param feature_properties Properties to concatenate
     * @return GnnMiniBatch with concatenated feature vectors
     */
    GnnMiniBatch create_mini_batch(
        const std::vector<ObjectId>& node_ids,
        const std::vector<std::string>& feature_properties
    );

    /**
     * @brief Create a mini-batch with labels.
     *
     * @param node_ids Nodes to include
     * @param feature_property Feature property name
     * @param label_property Property containing labels
     * @return GnnMiniBatch with features and labels
     */
    GnnMiniBatch create_labeled_mini_batch(
        const std::vector<ObjectId>& node_ids,
        const std::string& feature_property,
        const std::string& label_property
    );

    // =========================================================================
    // Multi-Layer Sampling (for GraphSAGE, GAT, etc.)
    // =========================================================================

    /**
     * @brief Sample k-hop neighborhood for seed nodes.
     *
     * This is the primary interface for GraphSAGE-style training.
     * Creates a multi-layer batch with sampled neighborhoods.
     *
     * @param seed_nodes Nodes to sample neighborhoods for
     * @param fanouts Fanout per layer [layer_0_fanout, layer_1_fanout, ...]
     * @param feature_property Feature property name
     * @param strategy Sampling strategy
     * @return GnnMultiLayerBatch for multi-hop message passing
     *
     * @code
     *   // 2-layer GraphSAGE: 25 neighbors at layer 1, 10 at layer 2
     *   auto batch = adapter.sample_multi_layer(seeds, {25, 10}, "embedding");
     * @endcode
     */
    GnnMultiLayerBatch sample_multi_layer(
        const std::vector<ObjectId>& seed_nodes,
        const std::vector<int64_t>& fanouts,
        const std::string& feature_property,
        SamplingStrategy strategy = SamplingStrategy::UNIFORM
    );

    /**
     * @brief Sample k-hop neighborhood with labels.
     */
    GnnMultiLayerBatch sample_multi_layer_labeled(
        const std::vector<ObjectId>& seed_nodes,
        const std::vector<int64_t>& fanouts,
        const std::string& feature_property,
        const std::string& label_property,
        SamplingStrategy strategy = SamplingStrategy::UNIFORM
    );

    // =========================================================================
    // Batch Iteration Helpers
    // =========================================================================

    /**
     * @brief Create an iterator over mini-batches of all nodes.
     *
     * @param batch_size Number of nodes per batch
     * @param feature_property Feature property name
     * @param shuffle Whether to shuffle node order
     * @return Vector of mini-batches (for simplicity, full materialization)
     */
    std::vector<GnnMiniBatch> create_all_mini_batches(
        size_t batch_size,
        const std::string& feature_property,
        bool shuffle = true
    );

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set random seed for reproducible sampling.
     */
    void set_random_seed(uint64_t seed);

    /**
     * @brief Set target device for tensors.
     */
    void set_device(torch::Device device);

    /**
     * @brief Get current device.
     */
    torch::Device get_device() const;

    /**
     * @brief Clear feature cache.
     */
    void clear_cache();

    /**
     * @brief Get cache statistics.
     */
    CacheStats get_cache_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Handle that owns both storage and adapter for safe lifetime management.
 *
 * Since GnnProjectionAdapter holds a reference to ProjectionStorage,
 * this handle ensures proper ownership and lifetime management.
 *
 * Usage:
 * @code
 *   auto handle = GnnProjectionHandle::open("my_projection");
 *   if (handle) {
 *       auto& adapter = handle->adapter();
 *       // Use adapter...
 *   }
 * @endcode
 */
class GnnProjectionHandle {
public:
    /**
     * @brief Opens a projection by name.
     *
     * @param projection_name Name of the projection
     * @param config Optional adapter configuration
     * @return Handle, or nullptr if projection not found
     */
    static std::unique_ptr<GnnProjectionHandle> open(
        const std::string& projection_name,
        const GnnAdapterConfig& config = GnnAdapterConfig{}
    );

    ~GnnProjectionHandle();

    // Non-copyable, non-movable (due to internal references)
    GnnProjectionHandle(const GnnProjectionHandle&) = delete;
    GnnProjectionHandle& operator=(const GnnProjectionHandle&) = delete;

    /// @brief Get the GNN adapter
    GnnProjectionAdapter& adapter() { return *adapter_; }
    const GnnProjectionAdapter& adapter() const { return *adapter_; }

    /// @brief Get the underlying storage
    GQL::ProjectionStorage& storage() { return *storage_; }
    const GQL::ProjectionStorage& storage() const { return *storage_; }

    /// @brief Get the projection name
    const std::string& name() const { return projection_name_; }

private:
    GnnProjectionHandle(
        const std::string& name,
        std::unique_ptr<GQL::ProjectionStorage> storage,
        std::unique_ptr<GnnProjectionAdapter> adapter
    );

    std::string projection_name_;
    std::unique_ptr<GQL::ProjectionStorage> storage_;
    std::unique_ptr<GnnProjectionAdapter> adapter_;
};

} // namespace mdb::gnn
