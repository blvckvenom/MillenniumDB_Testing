#include "gnn_projection_adapter.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/projection_catalog.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "storage/index/bplus_tree/bplus_tree.h"

#include "gnn/core/cuda_context.h"

namespace mdb::gnn {

// ============================================================================
// GnnProjectionAdapter Implementation
// ============================================================================

struct GnnProjectionAdapter::Impl {
    GQL::ProjectionStorage& storage;
    FeatureAccessor feature_accessor;
    TopologyAccessor topology_accessor;
    GnnAdapterConfig config;
    std::mt19937_64 rng;

    explicit Impl(GQL::ProjectionStorage& storage_, const GnnAdapterConfig& config_)
        : storage(storage_),
          feature_accessor(storage_),
          topology_accessor(storage_),
          config(config_),
          rng(config_.random_seed) {

        // Configure feature accessor
        if (config_.enable_feature_cache) {
            feature_accessor.enable_cache(config_.feature_cache_size);
        }
        feature_accessor.set_target_device(config_.device);

        // Configure topology accessor
        topology_accessor.set_random_seed(config_.random_seed);
        topology_accessor.set_target_device(config_.device);
    }

    /**
     * @brief Read label property as tensor.
     *
     * Labels can be integers, which need conversion to tensor.
     */
    torch::Tensor get_labels(const std::vector<ObjectId>& node_ids, const std::string& label_property) {
        // Try to get as tensor property first
        try {
            auto fm = feature_accessor.get_batch_features(node_ids, label_property);
            // Labels should be 1D, squeeze if needed
            if (fm.features.dim() > 1 && fm.features.size(1) == 1) {
                return fm.features.squeeze(1);
            }
            return fm.features;
        } catch (const std::exception& e) {
            // Provide better error context while preserving original message
            throw std::runtime_error(
                "Failed to get label property '" + label_property + "': " + e.what()
            );
        }
    }

    /**
     * @brief Collect all unique node IDs from multi-layer sampling.
     */
    std::vector<ObjectId> collect_all_nodes(const std::vector<SampledSubgraph>& layers) {
        std::unordered_set<uint64_t> seen;
        std::vector<ObjectId> result;

        // Collect from all layers (src and dst nodes)
        for (const auto& layer : layers) {
            for (const auto& id : layer.src_nodes) {
                if (seen.insert(id.id).second) {
                    result.push_back(id);
                }
            }
            for (const auto& id : layer.dst_nodes) {
                if (seen.insert(id.id).second) {
                    result.push_back(id);
                }
            }
        }

        return result;
    }
};

// ============================================================================
// GnnProjectionAdapter Public Methods
// ============================================================================

GnnProjectionAdapter::GnnProjectionAdapter(
    GQL::ProjectionStorage& storage,
    const GnnAdapterConfig& config
) : impl_(std::make_unique<Impl>(storage, config)) {}

GnnProjectionAdapter::~GnnProjectionAdapter() = default;

GnnProjectionAdapter::GnnProjectionAdapter(GnnProjectionAdapter&&) noexcept = default;
GnnProjectionAdapter& GnnProjectionAdapter::operator=(GnnProjectionAdapter&&) noexcept = default;

// ----- Accessor Access -----

FeatureAccessor& GnnProjectionAdapter::features() {
    return impl_->feature_accessor;
}

const FeatureAccessor& GnnProjectionAdapter::features() const {
    return impl_->feature_accessor;
}

TopologyAccessor& GnnProjectionAdapter::topology() {
    return impl_->topology_accessor;
}

const TopologyAccessor& GnnProjectionAdapter::topology() const {
    return impl_->topology_accessor;
}

// ----- Metadata -----

ProjectionMetadata GnnProjectionAdapter::get_metadata() const {
    ProjectionMetadata meta;

    // Basic statistics are always available
    meta.node_count = impl_->storage.get_node_count();
    meta.edge_count = impl_->storage.get_edge_count();

    // Note: Full metadata (name, labels, properties) requires the ProjectionCatalog
    // which is not directly accessible from ProjectionStorage.
    // For now, we provide what's available from the storage itself.
    // To get full metadata, use the overload that accepts ProjectionCatalog.

    return meta;
}

ProjectionMetadata GnnProjectionAdapter::get_metadata(const GQL::ProjectionCatalog& catalog) const {
    ProjectionMetadata meta;

    meta.name = catalog.projection_name;
    meta.node_count = catalog.node_count;
    meta.edge_count = catalog.edge_count;

    // Note: Label and property names aren't directly available from the catalog
    // in a list format. The catalog stores counts and included property names.
    // For tensor_properties, we would need to sample nodes to detect tensors.

    // Copy included property names (these might be tensor properties)
    for (const auto& prop : catalog.included_node_properties) {
        if (impl_->feature_accessor.has_tensor_property(prop)) {
            meta.tensor_properties.push_back(prop);
        }
    }

    return meta;
}

std::vector<ObjectId> GnnProjectionAdapter::get_all_nodes() const {
    return impl_->storage.get_all_node_ids();
}

uint64_t GnnProjectionAdapter::get_node_count() const {
    return impl_->storage.get_node_count();
}

uint64_t GnnProjectionAdapter::get_edge_count() const {
    return impl_->storage.get_edge_count();
}

bool GnnProjectionAdapter::has_tensor_property(const std::string& property_name) const {
    return impl_->feature_accessor.has_tensor_property(property_name);
}

int64_t GnnProjectionAdapter::get_feature_dimension(const std::string& property_name) {
    return impl_->feature_accessor.get_dimension(property_name);
}

// ----- Single-Layer Mini-Batch -----

GnnMiniBatch GnnProjectionAdapter::create_mini_batch(
    const std::vector<ObjectId>& node_ids,
    const std::string& feature_property
) {
    GnnMiniBatch batch;

    // Get features
    auto fm = impl_->feature_accessor.get_batch_features(node_ids, feature_property);
    batch.features = std::move(fm.features);
    batch.node_ids = std::move(fm.node_ids);

    // Build edge index for this subgraph
    auto edge_idx = impl_->topology_accessor.build_edge_index(node_ids);
    batch.edge_index = std::move(edge_idx.edge_index);

    return batch;
}

GnnMiniBatch GnnProjectionAdapter::create_mini_batch(
    const std::vector<ObjectId>& node_ids,
    const std::vector<std::string>& feature_properties
) {
    GnnMiniBatch batch;

    // Get concatenated features
    auto fm = impl_->feature_accessor.get_concatenated_features(node_ids, feature_properties);
    batch.features = std::move(fm.features);
    batch.node_ids = std::move(fm.node_ids);

    // Build edge index
    auto edge_idx = impl_->topology_accessor.build_edge_index(node_ids);
    batch.edge_index = std::move(edge_idx.edge_index);

    return batch;
}

GnnMiniBatch GnnProjectionAdapter::create_labeled_mini_batch(
    const std::vector<ObjectId>& node_ids,
    const std::string& feature_property,
    const std::string& label_property
) {
    GnnMiniBatch batch = create_mini_batch(node_ids, feature_property);

    // Get labels
    batch.labels = impl_->get_labels(node_ids, label_property);

    return batch;
}

// ----- Multi-Layer Sampling -----

GnnMultiLayerBatch GnnProjectionAdapter::sample_multi_layer(
    const std::vector<ObjectId>& seed_nodes,
    const std::vector<int64_t>& fanouts,
    const std::string& feature_property,
    SamplingStrategy strategy
) {
    GnnMultiLayerBatch batch;
    batch.seed_nodes = seed_nodes;

    // Sample k-hop neighborhoods
    batch.layers = impl_->topology_accessor.sample_khop_neighbors(seed_nodes, fanouts, strategy);

    // Get features for ALL sampled nodes across ALL layers
    // In GraphSAGE-style GNNs, message passing aggregates from neighbors to seeds,
    // so every node that participates in aggregation needs its features.
    if (!batch.layers.empty()) {
        // Collect all unique node IDs from all layers
        std::vector<ObjectId> all_nodes = impl_->collect_all_nodes(batch.layers);
        auto fm = impl_->feature_accessor.get_batch_features(all_nodes, feature_property);
        batch.root_features = std::move(fm.features);
    } else {
        // No layers means no sampling happened (empty fanouts)
        auto fm = impl_->feature_accessor.get_batch_features(seed_nodes, feature_property);
        batch.root_features = std::move(fm.features);
    }

    return batch;
}

GnnMultiLayerBatch GnnProjectionAdapter::sample_multi_layer_labeled(
    const std::vector<ObjectId>& seed_nodes,
    const std::vector<int64_t>& fanouts,
    const std::string& feature_property,
    const std::string& label_property,
    SamplingStrategy strategy
) {
    GnnMultiLayerBatch batch = sample_multi_layer(seed_nodes, fanouts, feature_property, strategy);

    // Get labels for seed nodes only
    batch.labels = impl_->get_labels(seed_nodes, label_property);

    return batch;
}

// ----- Batch Iteration -----

std::vector<GnnMiniBatch> GnnProjectionAdapter::create_all_mini_batches(
    size_t batch_size,
    const std::string& feature_property,
    bool shuffle
) {
    std::vector<ObjectId> all_nodes = impl_->storage.get_all_node_ids();

    if (shuffle) {
        std::shuffle(all_nodes.begin(), all_nodes.end(), impl_->rng);
    }

    std::vector<GnnMiniBatch> batches;
    size_t num_batches = (all_nodes.size() + batch_size - 1) / batch_size;
    batches.reserve(num_batches);

    for (size_t i = 0; i < all_nodes.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, all_nodes.size());
        std::vector<ObjectId> batch_nodes(all_nodes.begin() + i, all_nodes.begin() + end);

        batches.push_back(create_mini_batch(batch_nodes, feature_property));
    }

    return batches;
}

// ----- Configuration -----

void GnnProjectionAdapter::set_random_seed(uint64_t seed) {
    impl_->rng.seed(seed);
    impl_->topology_accessor.set_random_seed(seed);
}

void GnnProjectionAdapter::set_device(torch::Device device) {
    impl_->config.device = device;
    impl_->feature_accessor.set_target_device(device);
    impl_->topology_accessor.set_target_device(device);
}

torch::Device GnnProjectionAdapter::get_device() const {
    return impl_->config.device;
}

void GnnProjectionAdapter::clear_cache() {
    impl_->feature_accessor.clear_cache();
}

CacheStats GnnProjectionAdapter::get_cache_stats() const {
    return impl_->feature_accessor.get_cache_stats();
}

// ============================================================================
// GnnProjectionHandle Implementation
// ============================================================================

GnnProjectionHandle::GnnProjectionHandle(
    const std::string& name,
    std::unique_ptr<GQL::ProjectionStorage> storage,
    std::unique_ptr<GnnProjectionAdapter> adapter
)
    : projection_name_(name),
      storage_(std::move(storage)),
      adapter_(std::move(adapter)) {}

GnnProjectionHandle::~GnnProjectionHandle() = default;

std::unique_ptr<GnnProjectionHandle> GnnProjectionHandle::open(
    const std::string& projection_name,
    const GnnAdapterConfig& config
) {
    // Use the singleton ProjectionManager
    auto& manager = GQL::ProjectionManager::get_instance();

    // Check if projection exists
    if (!manager.projection_exists(projection_name)) {
        return nullptr;
    }

    // Get projection directory and database folder
    std::string proj_dir = manager.get_projection_dir(projection_name);
    const std::string& db_folder = manager.get_db_folder();

    // Create storage (opens existing projection)
    auto storage = std::make_unique<GQL::ProjectionStorage>(proj_dir, db_folder);
    storage->open();

    // Create adapter with reference to storage
    auto adapter = std::make_unique<GnnProjectionAdapter>(*storage, config);

    // Use placement new to construct since constructor is private
    // Actually, we need a different approach - let's use a helper
    return std::unique_ptr<GnnProjectionHandle>(
        new GnnProjectionHandle(projection_name, std::move(storage), std::move(adapter))
    );
}

} // namespace mdb::gnn
