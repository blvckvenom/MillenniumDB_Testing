#pragma once

/**
 * @file inmemory_gnn_tensor_store.h
 * @brief DEPRECATED — use FeatureMatrix instead.
 *
 * @deprecated Use feature_matrix.h and row_mapping.h instead.
 *
 * @brief In-memory implementation of GnnTensorStore.
 *
 * This implementation stores tensors entirely in memory using std::vector.
 * It's suitable for:
 *   - Unit testing
 *   - Small datasets that fit in RAM
 *   - Temporary tensor storage during computation
 *
 * For production use with large graphs, prefer FileGnnTensorStore which
 * uses memory-mapped files.
 */

#include "gnn_tensor_store.h"

#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace mdb::gnn {

/**
 * @brief In-memory implementation of GnnTensorStore.
 *
 * Thread-safe implementation using shared_mutex:
 *   - Multiple concurrent readers allowed
 *   - Writers have exclusive access
 */
class InMemoryGnnTensorStore : public GnnTensorStore {
public:
    InMemoryGnnTensorStore() = default;
    ~InMemoryGnnTensorStore() override = default;

    // Non-copyable, but movable
    InMemoryGnnTensorStore(const InMemoryGnnTensorStore&) = delete;
    InMemoryGnnTensorStore& operator=(const InMemoryGnnTensorStore&) = delete;
    InMemoryGnnTensorStore(InMemoryGnnTensorStore&&) = default;
    InMemoryGnnTensorStore& operator=(InMemoryGnnTensorStore&&) = default;

    // ========================================================================
    // Core Operations
    // ========================================================================

    bool store(const std::string& key,
               const void* data,
               const std::vector<int64_t>& shape,
               GnnDtype dtype) override;

    GnnTensorView load(const std::string& key) const override;

    bool remove(const std::string& key) override;

    bool exists(const std::string& key) const override;

    // ========================================================================
    // Metadata Operations
    // ========================================================================

    std::optional<GnnTensorMetadata> get_metadata(const std::string& key) const override;

    std::vector<std::string> list_keys() const override;

    size_t total_size() const override;

    size_t count() const override;

    // ========================================================================
    // Lifecycle Operations
    // ========================================================================

    void clear() override;

    void flush() override {
        // No-op for in-memory store
    }

private:
    /**
     * @brief Internal storage entry containing data and metadata.
     */
    struct StorageEntry {
        std::vector<uint8_t> data;    ///< Raw tensor data
        GnnTensorMetadata metadata;   ///< Tensor metadata
    };

    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, StorageEntry> storage_;
    size_t total_bytes_ = 0;
};

} // namespace mdb::gnn
