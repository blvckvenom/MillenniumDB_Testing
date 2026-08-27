#include "feature_accessor.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/common/datatypes/tensor/tensor.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "system/tensor_manager.h"

#include "gnn/core/cuda_context.h"

namespace mdb::gnn {

// ============================================================================
// LRU Cache Implementation
// ============================================================================

/**
 * @brief Thread-safe LRU (Least Recently Used) cache with configurable hash function.
 *
 * Uses shared_mutex for reader-writer locking: shared access for read-only lookups,
 * exclusive access for writes and LRU updates.
 *
 * @tparam K Key type
 * @tparam V Value type
 * @tparam Hash Hash function for keys (default: std::hash<K>)
 * @tparam KeyEqual Equality comparison for keys (default: std::equal_to<K>)
 */
template<typename K, typename V, typename Hash = std::hash<K>, typename KeyEqual = std::equal_to<K>>
class LRUCache {
public:
    explicit LRUCache(size_t capacity) : capacity_(capacity) {}

    /**
     * @brief Get value from cache (thread-safe).
     *
     * Tries exclusive lock first for full LRU update. If contended,
     * falls back to shared lock for safe read-only lookup (no LRU reorder).
     */
    std::optional<V> get(const K& key) {
        std::unique_lock<std::shared_mutex> ulock(mutex_, std::try_to_lock);
        if (ulock) {
            return get_with_lru_update(key);
        }
        // Under contention: shared lock for thread-safe read-only access
        std::shared_lock<std::shared_mutex> slock(mutex_);
        return get_without_lru_update(key);
    }

    /**
     * @brief Insert or update value in cache (thread-safe, exclusive lock).
     */
    void put(const K& key, V value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);

        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // Update existing
            it->second->second = std::move(value);
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }

        // Evict if full
        if (cache_map_.size() >= capacity_) {
            auto last = cache_list_.back();
            cache_map_.erase(last.first);
            cache_list_.pop_back();
        }

        // Insert new
        cache_list_.emplace_front(key, std::move(value));
        cache_map_[key] = cache_list_.begin();
    }

    /**
     * @brief Clear all entries (thread-safe, exclusive lock).
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_map_.clear();
        cache_list_.clear();
    }

    /**
     * @brief Get current cache size (thread-safe, shared lock).
     */
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_map_.size();
    }

private:
    /**
     * @brief Get with LRU update (caller must hold exclusive lock).
     */
    std::optional<V> get_with_lru_update(const K& key) {
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return std::nullopt;
        }
        // Move to front (most recently used)
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return it->second->second;
    }

    /**
     * @brief Get without LRU update (caller must hold at least shared lock).
     */
    std::optional<V> get_without_lru_update(const K& key) const {
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return std::nullopt;
        }
        return it->second->second;
    }

    size_t capacity_;
    std::list<std::pair<K, V>> cache_list_;
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator, Hash, KeyEqual> cache_map_;
    mutable std::shared_mutex mutex_;
};

// ============================================================================
// FeatureAccessor Implementation
// ============================================================================

struct FeatureAccessor::Impl {
    GQL::ProjectionStorage& storage;
    torch::Device target_device;

    // Cache: (node_id, property_name) -> tensor
    struct CacheKey {
        uint64_t node_id;
        std::string property_name;

        bool operator==(const CacheKey& other) const {
            return node_id == other.node_id && property_name == other.property_name;
        }
    };

    struct CacheKeyHash {
        size_t operator()(const CacheKey& k) const {
            return std::hash<uint64_t>{}(k.node_id) ^
                   (std::hash<std::string>{}(k.property_name) << 1);
        }
    };

    // Use LRUCache with custom hash function for CacheKey
    std::optional<LRUCache<CacheKey, torch::Tensor, CacheKeyHash>> cache;
    CacheStats cache_stats;

    // Dimension cache for properties
    std::unordered_map<std::string, int64_t> dimension_cache;

    explicit Impl(GQL::ProjectionStorage& storage_)
        : storage(storage_),
          target_device(CudaContext::instance().torch_device()) {}

    /**
     * @brief Convert property name string to node key ObjectId.
     *
     * Checks projection-specific keys first, then falls back to global catalog.
     * This allows projections to define synthetic properties like `_count`
     * that don't exist in the main database.
     */
    ObjectId get_node_key_id(const std::string& property_name) const {
        // Check projection-specific keys FIRST
        auto proj_key = storage.get_node_key_id(property_name);
        if (proj_key.has_value()) {
            return ObjectId(ObjectId::MASK_NODE_KEY | proj_key.value());
        }

        // Fallback to global GQL catalog
        auto& catalog = gql_model.catalog;
        auto it = catalog.node_keys2id.find(property_name);
        if (it == catalog.node_keys2id.end()) {
            return ObjectId::get_null();
        }
        // Combine with NODE_KEY mask
        return ObjectId(ObjectId::MASK_NODE_KEY | it->second);
    }

    /**
     * @brief Convert property name string to edge key ObjectId.
     *
     * Checks projection-specific keys first, then falls back to global catalog.
     */
    ObjectId get_edge_key_id(const std::string& property_name) const {
        // Check projection-specific keys FIRST
        auto proj_key = storage.get_edge_key_id(property_name);
        if (proj_key.has_value()) {
            return ObjectId(ObjectId::MASK_EDGE_KEY | proj_key.value());
        }

        // Fallback to global GQL catalog
        auto& catalog = gql_model.catalog;
        auto it = catalog.edge_keys2id.find(property_name);
        if (it == catalog.edge_keys2id.end()) {
            return ObjectId::get_null();
        }
        // Combine with EDGE_KEY mask
        return ObjectId(ObjectId::MASK_EDGE_KEY | it->second);
    }

    /**
     * @brief Get property value ObjectId for a node.
     *
     * Searches node_key_value_index for (node_id, key_id, *) and returns the value.
     */
    ObjectId get_node_property_value(ObjectId node_id, ObjectId key_id) const {
        auto* index = storage.get_node_key_value_index();
        if (!index) {
            return ObjectId::get_null();
        }

        // Search for (node_id, key_id, MIN) to (node_id, key_id, MAX)
        Record<3> min_record = {node_id.id, key_id.id, 0};
        Record<3> max_record = {node_id.id, key_id.id, UINT64_MAX};

        bool interruption_requested = false;
        auto it = index->get_range(&interruption_requested, min_record, max_record);

        const Record<3>* record = it.next();
        if (record == nullptr) {
            return ObjectId::get_null();
        }

        // Get the value (third element)
        return ObjectId(std::get<2>(*record));
    }

    /**
     * @brief Get property value ObjectId for an edge.
     *
     * Searches edge_key_value_index for (edge_id, key_id, *) and returns the value.
     */
    ObjectId get_edge_property_value(ObjectId edge_id, ObjectId key_id) const {
        auto* index = storage.get_edge_key_value_index();
        if (!index) {
            return ObjectId::get_null();
        }

        // Search for (edge_id, key_id, MIN) to (edge_id, key_id, MAX)
        Record<3> min_record = {edge_id.id, key_id.id, 0};
        Record<3> max_record = {edge_id.id, key_id.id, UINT64_MAX};

        bool interruption_requested = false;
        auto it = index->get_range(&interruption_requested, min_record, max_record);

        const Record<3>* record = it.next();
        if (record == nullptr) {
            return ObjectId::get_null();
        }

        // Get the value (third element)
        return ObjectId(std::get<2>(*record));
    }

    /**
     * @brief Convert MDB tensor ObjectId to torch::Tensor.
     *
     * Uses subtype mask comparison to correctly distinguish between
     * float and double tensors. The previous bitmask check was buggy:
     * (type & MASK_TENSOR_FLOAT) == MASK_TENSOR_FLOAT would match DOUBLE
     * types too since 0xB4 & 0xB0 = 0xB0.
     */
    torch::Tensor objectid_to_tensor(ObjectId tensor_oid) const {
        // Use subtype mask for correct discrimination
        auto subtype = tensor_oid.get_sub_type();

        // Check DOUBLE first (more specific) - subtype 0xB4
        if (subtype == ObjectId::MASK_TENSOR_DOUBLE) {
            auto mdb_tensor = tensor_manager.get_tensor<double>(tensor_oid);
            auto size = static_cast<int64_t>(mdb_tensor.size());

            auto options = torch::TensorOptions().dtype(torch::kFloat64);
            torch::Tensor result = torch::empty({size}, options);

            std::memcpy(result.data_ptr<double>(), mdb_tensor.data(), size * sizeof(double));

            // Clone to ensure we own the data (important for TMP tensors)
            return result.clone();
        }
        // Then check FLOAT - subtype 0xB0
        else if (subtype == ObjectId::MASK_TENSOR_FLOAT) {
            auto mdb_tensor = tensor_manager.get_tensor<float>(tensor_oid);
            auto size = static_cast<int64_t>(mdb_tensor.size());

            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor result = torch::empty({size}, options);

            std::memcpy(result.data_ptr<float>(), mdb_tensor.data(), size * sizeof(float));

            // Clone to ensure we own the data (important for TMP tensors)
            return result.clone();
        }

        throw std::runtime_error("ObjectId is not a tensor type: " +
            std::to_string(tensor_oid.id));
    }
};

// ============================================================================
// FeatureAccessor Public Methods
// ============================================================================

FeatureAccessor::FeatureAccessor(GQL::ProjectionStorage& storage)
    : impl_(std::make_unique<Impl>(storage)) {}

FeatureAccessor::~FeatureAccessor() = default;

FeatureAccessor::FeatureAccessor(FeatureAccessor&&) noexcept = default;
FeatureAccessor& FeatureAccessor::operator=(FeatureAccessor&&) noexcept = default;

// ----- Single Node Access -----

torch::Tensor FeatureAccessor::get_feature(ObjectId node_id, const std::string& property_name) {
    // Check cache first
    if (impl_->cache) {
        typename Impl::CacheKey key{node_id.id, property_name};
        if (auto cached = impl_->cache->get(key)) {
            impl_->cache_stats.hits.fetch_add(1, std::memory_order_relaxed);
            return cached->to(impl_->target_device);
        }
        impl_->cache_stats.misses.fetch_add(1, std::memory_order_relaxed);
    }

    // Get key ObjectId
    ObjectId key_id = impl_->get_node_key_id(property_name);
    if (key_id.is_null()) {
        throw std::runtime_error("Property '" + property_name + "' not found in catalog");
    }

    // Get property value
    ObjectId value_id = impl_->get_node_property_value(node_id, key_id);
    if (value_id.is_null()) {
        throw std::runtime_error(
            "Node " + std::to_string(node_id.id) +
            " missing property: " + property_name
        );
    }

    // Convert to tensor
    torch::Tensor tensor = impl_->objectid_to_tensor(value_id);

    // Update cache
    if (impl_->cache) {
        typename Impl::CacheKey key{node_id.id, property_name};
        impl_->cache->put(key, tensor.clone());
    }

    return tensor.to(impl_->target_device);
}

std::unordered_map<std::string, torch::Tensor> FeatureAccessor::get_features(
    ObjectId node_id,
    const std::vector<std::string>& property_names
) {
    std::unordered_map<std::string, torch::Tensor> result;
    result.reserve(property_names.size());

    for (const auto& name : property_names) {
        result[name] = get_feature(node_id, name);
    }

    return result;
}

// ----- Batch Access -----

/// Maximum tensor allocation size (4 GB)
static constexpr size_t MAX_TENSOR_BYTES = 4ULL * 1024 * 1024 * 1024;

TorchFeatureMatrix FeatureAccessor::get_batch_features(
    const std::vector<ObjectId>& node_ids,
    const std::string& property_name
) {
    if (node_ids.empty()) {
        int64_t dim = get_dimension(property_name);
        return TorchFeatureMatrix{
            torch::empty({0, dim > 0 ? dim : 1}, torch::kFloat32),
            {},
            {}
        };
    }

    // Get key ObjectId once
    ObjectId key_id = impl_->get_node_key_id(property_name);
    if (key_id.is_null()) {
        throw std::runtime_error("Property '" + property_name + "' not found in catalog");
    }

    int64_t N = static_cast<int64_t>(node_ids.size());
    int64_t D = get_dimension(property_name);

    // If dimension unknown, get from first tensor
    if (D <= 0) {
        ObjectId value_id = impl_->get_node_property_value(node_ids[0], key_id);
        if (value_id.is_null()) {
            throw std::runtime_error(
                "Node " + std::to_string(node_ids[0].id) +
                " missing property: " + property_name
            );
        }
        torch::Tensor first = impl_->objectid_to_tensor(value_id);
        D = first.size(0);
        impl_->dimension_cache[property_name] = D;
    }

    // Overflow validation (H2)
    if (N > 0 && D > 0) {
        // Check for multiplication overflow
        if (N > static_cast<int64_t>(MAX_TENSOR_BYTES / (static_cast<size_t>(D) * sizeof(float)))) {
            throw std::overflow_error(
                "Feature matrix too large: " + std::to_string(N) +
                " x " + std::to_string(D) + " exceeds 4GB limit"
            );
        }
    }

    // SINGLE allocation - performance optimization (H1)
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor features = torch::empty({N, D}, options);

    // Fill rows directly
    for (int64_t i = 0; i < N; ++i) {
        ObjectId node_id = node_ids[i];

        // Check cache
        torch::Tensor tensor;
        bool from_cache = false;

        if (impl_->cache) {
            typename Impl::CacheKey cache_key{node_id.id, property_name};
            if (auto cached = impl_->cache->get(cache_key)) {
                tensor = *cached;
                impl_->cache_stats.hits.fetch_add(1, std::memory_order_relaxed);
                from_cache = true;
            } else {
                impl_->cache_stats.misses.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (!from_cache) {
            // Get property value
            ObjectId value_id = impl_->get_node_property_value(node_id, key_id);
            if (value_id.is_null()) {
                throw std::runtime_error(
                    "Node " + std::to_string(node_id.id) +
                    " missing property: " + property_name
                );
            }

            tensor = impl_->objectid_to_tensor(value_id);

            // Update cache
            if (impl_->cache) {
                typename Impl::CacheKey cache_key{node_id.id, property_name};
                impl_->cache->put(cache_key, tensor.clone());
            }
        }

        // Dimension validation (H4)
        if (tensor.size(0) != D) {
            throw std::runtime_error(
                "Dimension mismatch at node " + std::to_string(node_id.id) +
                ": expected " + std::to_string(D) +
                ", got " + std::to_string(tensor.size(0))
            );
        }

        // Direct copy into pre-allocated matrix row
        features[i].copy_(tensor);
    }

    // Build ID mappings
    std::vector<ObjectId> result_ids = node_ids;
    std::unordered_map<uint64_t, int64_t> id_to_row;
    id_to_row.reserve(node_ids.size());
    for (int64_t i = 0; i < N; ++i) {
        id_to_row[node_ids[i].id] = i;
    }

    // Transfer to target device
    features = features.to(impl_->target_device);

    return TorchFeatureMatrix{
        std::move(features),
        std::move(result_ids),
        std::move(id_to_row)
    };
}

std::unordered_map<std::string, TorchFeatureMatrix> FeatureAccessor::get_batch_features(
    const std::vector<ObjectId>& node_ids,
    const std::vector<std::string>& property_names
) {
    std::unordered_map<std::string, TorchFeatureMatrix> result;
    result.reserve(property_names.size());

    for (const auto& name : property_names) {
        result.emplace(name, get_batch_features(node_ids, name));
    }

    return result;
}

// ----- Edge Feature Access -----

torch::Tensor FeatureAccessor::get_edge_feature(ObjectId edge_id, const std::string& property_name) {
    // Check cache first
    if (impl_->cache) {
        // Use negative node_id to distinguish edge cache entries
        typename Impl::CacheKey key{~edge_id.id, property_name};
        if (auto cached = impl_->cache->get(key)) {
            impl_->cache_stats.hits.fetch_add(1, std::memory_order_relaxed);
            return cached->to(impl_->target_device);
        }
        impl_->cache_stats.misses.fetch_add(1, std::memory_order_relaxed);
    }

    // Get key ObjectId
    ObjectId key_id = impl_->get_edge_key_id(property_name);
    if (key_id.is_null()) {
        throw std::runtime_error("Edge property '" + property_name + "' not found in catalog");
    }

    // Get property value
    ObjectId value_id = impl_->get_edge_property_value(edge_id, key_id);
    if (value_id.is_null()) {
        throw std::runtime_error(
            "Edge " + std::to_string(edge_id.id) +
            " missing property: " + property_name
        );
    }

    // Convert to tensor
    torch::Tensor tensor = impl_->objectid_to_tensor(value_id);

    // Update cache
    if (impl_->cache) {
        typename Impl::CacheKey key{~edge_id.id, property_name};
        impl_->cache->put(key, tensor.clone());
    }

    return tensor.to(impl_->target_device);
}

TorchFeatureMatrix FeatureAccessor::get_edge_batch_features(
    const std::vector<ObjectId>& edge_ids,
    const std::string& property_name
) {
    if (edge_ids.empty()) {
        return TorchFeatureMatrix{
            torch::empty({0, 1}, torch::kFloat32),
            {},
            {}
        };
    }

    // Get key ObjectId once
    ObjectId key_id = impl_->get_edge_key_id(property_name);
    if (key_id.is_null()) {
        throw std::runtime_error("Edge property '" + property_name + "' not found in catalog");
    }

    // Determine dimension from first edge
    ObjectId first_value = impl_->get_edge_property_value(edge_ids[0], key_id);
    if (first_value.is_null()) {
        throw std::runtime_error(
            "Edge " + std::to_string(edge_ids[0].id) +
            " missing property: " + property_name
        );
    }
    torch::Tensor first_tensor = impl_->objectid_to_tensor(first_value);
    int64_t D = first_tensor.size(0);

    int64_t N = static_cast<int64_t>(edge_ids.size());

    // Overflow validation
    if (N > 0 && D > 0) {
        if (N > static_cast<int64_t>(MAX_TENSOR_BYTES / (static_cast<size_t>(D) * sizeof(float)))) {
            throw std::overflow_error(
                "Edge feature matrix too large: " + std::to_string(N) +
                " x " + std::to_string(D) + " exceeds 4GB limit"
            );
        }
    }

    // Single allocation
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor features = torch::empty({N, D}, options);

    // Fill first row
    features[0].copy_(first_tensor);

    // Fill remaining rows
    for (int64_t i = 1; i < N; ++i) {
        ObjectId edge_id = edge_ids[i];

        // Check cache
        torch::Tensor tensor;
        bool from_cache = false;

        if (impl_->cache) {
            typename Impl::CacheKey cache_key{~edge_id.id, property_name};
            if (auto cached = impl_->cache->get(cache_key)) {
                tensor = *cached;
                impl_->cache_stats.hits.fetch_add(1, std::memory_order_relaxed);
                from_cache = true;
            } else {
                impl_->cache_stats.misses.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (!from_cache) {
            ObjectId value_id = impl_->get_edge_property_value(edge_id, key_id);
            if (value_id.is_null()) {
                throw std::runtime_error(
                    "Edge " + std::to_string(edge_id.id) +
                    " missing property: " + property_name
                );
            }

            tensor = impl_->objectid_to_tensor(value_id);

            if (impl_->cache) {
                typename Impl::CacheKey cache_key{~edge_id.id, property_name};
                impl_->cache->put(cache_key, tensor.clone());
            }
        }

        // Dimension validation
        if (tensor.size(0) != D) {
            throw std::runtime_error(
                "Dimension mismatch at edge " + std::to_string(edge_id.id) +
                ": expected " + std::to_string(D) +
                ", got " + std::to_string(tensor.size(0))
            );
        }

        features[i].copy_(tensor);
    }

    // Build ID mappings
    std::vector<ObjectId> result_ids = edge_ids;
    std::unordered_map<uint64_t, int64_t> id_to_row;
    id_to_row.reserve(edge_ids.size());
    for (int64_t i = 0; i < N; ++i) {
        id_to_row[edge_ids[i].id] = i;
    }

    features = features.to(impl_->target_device);

    return TorchFeatureMatrix{
        std::move(features),
        std::move(result_ids),
        std::move(id_to_row)
    };
}

// ----- Concatenated Features -----

TorchFeatureMatrix FeatureAccessor::get_concatenated_features(
    const std::vector<ObjectId>& node_ids,
    const std::vector<std::string>& property_names
) {
    if (property_names.empty()) {
        throw std::invalid_argument("property_names cannot be empty");
    }

    if (node_ids.empty()) {
        int64_t total_dim = 0;
        for (const auto& name : property_names) {
            int64_t dim = get_dimension(name);
            if (dim > 0) total_dim += dim;
        }
        return TorchFeatureMatrix{
            torch::empty({0, total_dim > 0 ? total_dim : 1}, torch::kFloat32),
            {},
            {}
        };
    }

    // Get feature matrices for each property
    std::vector<TorchFeatureMatrix> matrices;
    matrices.reserve(property_names.size());

    for (const auto& name : property_names) {
        matrices.push_back(get_batch_features(node_ids, name));
    }

    // Concatenate along feature dimension
    std::vector<torch::Tensor> feature_tensors;
    feature_tensors.reserve(matrices.size());

    for (auto& fm : matrices) {
        feature_tensors.push_back(fm.features);
    }

    torch::Tensor concatenated = torch::cat(feature_tensors, /*dim=*/1);

    return TorchFeatureMatrix{
        std::move(concatenated),
        std::move(matrices[0].node_ids),
        std::move(matrices[0].id_to_row)
    };
}

// ----- Metadata -----

int64_t FeatureAccessor::get_dimension(const std::string& property_name) {
    // Check cache
    auto it = impl_->dimension_cache.find(property_name);
    if (it != impl_->dimension_cache.end()) {
        return it->second;
    }

    // Sample from first node to get dimension
    auto node_ids = impl_->storage.get_all_node_ids();
    if (node_ids.empty()) {
        return -1;
    }

    try {
        torch::Tensor sample = get_feature(node_ids[0], property_name);
        int64_t dim = sample.size(0);
        impl_->dimension_cache[property_name] = dim;
        return dim;
    } catch (...) {
        return -1;
    }
}

bool FeatureAccessor::has_tensor_property(const std::string& property_name) {
    ObjectId key_id = impl_->get_node_key_id(property_name);
    if (key_id.is_null()) {
        return false;
    }

    // Check if at least one node has this property as a tensor
    auto node_ids = impl_->storage.get_all_node_ids();
    if (node_ids.empty()) {
        return false;
    }

    ObjectId value_id = impl_->get_node_property_value(node_ids[0], key_id);
    if (value_id.is_null()) {
        return false;
    }

    // Use subtype mask for correct tensor type detection
    auto subtype = value_id.get_sub_type();
    return subtype == ObjectId::MASK_TENSOR_FLOAT ||
           subtype == ObjectId::MASK_TENSOR_DOUBLE;
}

std::vector<ObjectId> FeatureAccessor::get_all_node_ids() const {
    return impl_->storage.get_all_node_ids();
}

uint64_t FeatureAccessor::get_node_count() const {
    return impl_->storage.get_node_count();
}

// ----- Caching -----

void FeatureAccessor::enable_cache(size_t max_cached_nodes) {
    impl_->cache.emplace(max_cached_nodes);
    impl_->cache_stats.reset();
}

void FeatureAccessor::disable_cache() {
    impl_->cache.reset();
}

void FeatureAccessor::clear_cache() {
    if (impl_->cache) {
        impl_->cache->clear();
    }
    impl_->cache_stats.reset();
}

CacheStats FeatureAccessor::get_cache_stats() const {
    CacheStats stats = impl_->cache_stats;  // Copy via copy constructor
    if (impl_->cache) {
        size_t cache_size = impl_->cache->size();
        stats.cached_nodes.store(cache_size, std::memory_order_relaxed);
        // Estimate size (rough approximation)
        stats.cache_size_bytes.store(cache_size * 128 * sizeof(float), std::memory_order_relaxed);
    }
    return stats;
}

// ----- Advanced Options -----

void FeatureAccessor::set_target_device(torch::Device device) {
    impl_->target_device = device;
}

torch::Device FeatureAccessor::get_target_device() const {
    return impl_->target_device;
}

} // namespace mdb::gnn
