#pragma once

/**
 * @file gnn_tensor_store.h
 * @brief Abstract interface for GNN tensor storage.
 *
 * This file defines the GnnTensorStore interface - a storage system
 * COMPLETELY INDEPENDENT from MillenniumDB's existing tensor infrastructure.
 *
 * CRITICAL DESIGN DECISION:
 * The GNN system maintains its own tensor storage to avoid coupling with MDB's
 * internal tensor implementation (TensorManager, TensorsHash, ObjectId tensors).
 * This allows:
 *   1. Different access patterns (batch reads for training vs random access)
 *   2. Memory-mapped files for large embeddings
 *   3. Independent evolution of GNN and MDB tensor systems
 *   4. Clean separation of concerns
 *
 * Two implementations are provided:
 *   - InMemoryGnnTensorStore: For testing and small datasets
 *   - FileGnnTensorStore: For production use with memory-mapped files
 */

#include "gnn_dtype.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mdb::gnn {

/**
 * @brief Metadata for a stored tensor.
 *
 * This structure contains all information needed to interpret raw tensor data.
 */
struct GnnTensorMetadata {
    std::vector<int64_t> shape;  ///< Tensor dimensions (e.g., [1000, 128] for 1000 nodes with 128 features)
    GnnDtype dtype;              ///< Element data type
    size_t byte_size;            ///< Total size in bytes (computed from shape and dtype)

    /**
     * @brief Compute the total number of elements.
     */
    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * @brief Compute expected byte size from shape and dtype.
     */
    size_t compute_byte_size() const {
        return static_cast<size_t>(numel()) * dtype_size(dtype);
    }
};

/**
 * @brief A view into tensor data without ownership.
 *
 * This class provides a non-owning view into tensor storage.
 * The caller must ensure the underlying storage remains valid.
 */
class GnnTensorView {
public:
    GnnTensorView() = default;

    GnnTensorView(const void* data, GnnTensorMetadata metadata)
        : data_(data), metadata_(std::move(metadata)) {}

    /// @brief Returns pointer to raw data (nullptr if invalid)
    const void* data() const { return data_; }

    /// @brief Returns tensor metadata
    const GnnTensorMetadata& metadata() const { return metadata_; }

    /// @brief Returns tensor shape
    const std::vector<int64_t>& shape() const { return metadata_.shape; }

    /// @brief Returns tensor dtype
    GnnDtype dtype() const { return metadata_.dtype; }

    /// @brief Returns total byte size
    size_t byte_size() const { return metadata_.byte_size; }

    /// @brief Returns number of elements
    int64_t numel() const { return metadata_.numel(); }

    /// @brief Check if view is valid
    bool valid() const { return data_ != nullptr; }

    /// @brief Typed data access (no bounds checking)
    template<typename T>
    const T* data_as() const { return static_cast<const T*>(data_); }

private:
    const void* data_ = nullptr;
    GnnTensorMetadata metadata_;
};

/**
 * @brief Abstract interface for GNN tensor storage.
 *
 * This interface defines the contract for storing and retrieving tensors
 * used by the GNN subsystem. Implementations are free to use different
 * backing storage mechanisms (memory, files, etc.).
 *
 * Thread Safety:
 *   - Individual operations are thread-safe
 *   - Concurrent store() and load() to the same key has undefined behavior
 *   - Multiple concurrent load() to the same key is safe
 */
class GnnTensorStore {
public:
    virtual ~GnnTensorStore() = default;

    // ========================================================================
    // Core Operations
    // ========================================================================

    /**
     * @brief Store a tensor with the given key.
     *
     * @param key Unique identifier for the tensor (e.g., "node_features", "edge_index")
     * @param data Pointer to raw tensor data
     * @param shape Tensor dimensions
     * @param dtype Data type of tensor elements
     * @return true if successfully stored, false otherwise
     *
     * If a tensor with the same key already exists, it will be overwritten.
     */
    virtual bool store(const std::string& key,
                       const void* data,
                       const std::vector<int64_t>& shape,
                       GnnDtype dtype) = 0;

    /**
     * @brief Load a tensor by key, returning a view to its data.
     *
     * @param key The tensor identifier
     * @return GnnTensorView if found, invalid view if not found
     *
     * The returned view is valid only as long as:
     *   - The tensor is not removed
     *   - The store is not destroyed
     *   - The tensor is not overwritten
     */
    virtual GnnTensorView load(const std::string& key) const = 0;

    /**
     * @brief Remove a tensor from storage.
     *
     * @param key The tensor identifier
     * @return true if removed, false if not found
     */
    virtual bool remove(const std::string& key) = 0;

    /**
     * @brief Check if a tensor exists.
     *
     * @param key The tensor identifier
     * @return true if exists, false otherwise
     */
    virtual bool exists(const std::string& key) const = 0;

    // ========================================================================
    // Batch Operations (for efficient training data access)
    // ========================================================================

    /**
     * @brief Load multiple tensors at once.
     *
     * @param keys List of tensor identifiers
     * @return Vector of tensor views (invalid views for missing keys)
     *
     * This method allows implementations to optimize batch loading
     * (e.g., sequential disk reads, prefetching).
     */
    virtual std::vector<GnnTensorView> load_batch(const std::vector<std::string>& keys) const {
        // Default implementation: sequential loads
        std::vector<GnnTensorView> results;
        results.reserve(keys.size());
        for (const auto& key : keys) {
            results.push_back(load(key));
        }
        return results;
    }

    // ========================================================================
    // Metadata Operations
    // ========================================================================

    /**
     * @brief Get metadata for a tensor without loading data.
     *
     * @param key The tensor identifier
     * @return Metadata if found, nullopt otherwise
     */
    virtual std::optional<GnnTensorMetadata> get_metadata(const std::string& key) const = 0;

    /**
     * @brief List all stored tensor keys.
     *
     * @return Vector of all tensor identifiers
     */
    virtual std::vector<std::string> list_keys() const = 0;

    /**
     * @brief Get total storage size in bytes.
     *
     * @return Total bytes used by all stored tensors
     */
    virtual size_t total_size() const = 0;

    /**
     * @brief Get number of stored tensors.
     *
     * @return Count of stored tensors
     */
    virtual size_t count() const = 0;

    // ========================================================================
    // Lifecycle Operations
    // ========================================================================

    /**
     * @brief Clear all stored tensors.
     */
    virtual void clear() = 0;

    /**
     * @brief Flush any pending writes to persistent storage.
     *
     * For in-memory stores, this is a no-op.
     * For file-backed stores, this ensures data is written to disk.
     */
    virtual void flush() = 0;
};

} // namespace mdb::gnn
