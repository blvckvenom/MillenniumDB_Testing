#pragma once

/**
 * @file gnn_tensor_converter.h
 * @brief Converts between GnnTensorStore and torch::Tensor.
 *
 * This converter bridges the GNN tensor storage system with LibTorch,
 * allowing seamless conversion between our custom storage format and
 * PyTorch tensors used for training.
 *
 * IMPORTANT: This converter is the ONLY place where LibTorch should be
 * coupled with the GNN storage system. Keep storage operations independent
 * from torch::Tensor to maintain clean separation.
 *
 * Usage Examples:
 *
 *   // Store a torch tensor
 *   torch::Tensor features = torch::randn({1000, 128});
 *   GnnTensorConverter::store(store, "node_features", features);
 *
 *   // Load as torch tensor
 *   auto loaded = GnnTensorConverter::load(store, "node_features");
 *   if (loaded.defined()) {
 *       std::cout << "Shape: " << loaded.sizes() << std::endl;
 *   }
 *
 *   // Batch loading for training
 *   auto batch = GnnTensorConverter::load_batch(store,
 *       {"node_features", "edge_index", "labels"});
 */

#include "gnn_tensor_store.h"

#include <torch/torch.h>

#include <string>
#include <vector>

namespace mdb::gnn {

/**
 * @brief Converts between GnnTensorStore and torch::Tensor.
 *
 * This class provides static methods for converting between our custom
 * tensor storage and LibTorch tensors. All methods are stateless.
 */
class GnnTensorConverter {
public:
    // ========================================================================
    // Type Conversion
    // ========================================================================

    /**
     * @brief Convert GnnDtype to torch::ScalarType.
     *
     * @param dtype GNN data type
     * @return Corresponding torch scalar type
     * @throws std::invalid_argument if dtype is not supported
     */
    static torch::ScalarType to_torch_dtype(GnnDtype dtype);

    /**
     * @brief Convert torch::ScalarType to GnnDtype.
     *
     * @param torch_dtype Torch scalar type
     * @return Corresponding GNN data type
     * @throws std::invalid_argument if dtype is not supported
     */
    static GnnDtype from_torch_dtype(torch::ScalarType torch_dtype);

    // ========================================================================
    // Store Operations
    // ========================================================================

    /**
     * @brief Store a torch::Tensor in the GnnTensorStore.
     *
     * The tensor is converted to contiguous CPU memory before storing.
     * GPU tensors are automatically moved to CPU.
     *
     * @param store Target tensor store
     * @param key Tensor identifier
     * @param tensor Tensor to store
     * @return true if successfully stored, false otherwise
     */
    static bool store(GnnTensorStore& store,
                      const std::string& key,
                      const torch::Tensor& tensor);

    /**
     * @brief Store multiple tensors at once.
     *
     * @param store Target tensor store
     * @param tensors Map of key -> tensor pairs
     * @return Number of tensors successfully stored
     */
    static size_t store_batch(GnnTensorStore& store,
                              const std::vector<std::pair<std::string, torch::Tensor>>& tensors);

    // ========================================================================
    // Load Operations
    // ========================================================================

    /**
     * @brief Load a tensor from GnnTensorStore as torch::Tensor.
     *
     * The returned tensor shares memory with the store when possible
     * (zero-copy for contiguous data). The tensor is placed on CPU.
     *
     * @param store Source tensor store
     * @param key Tensor identifier
     * @param copy If true, always copy data (default: false for zero-copy when possible)
     * @return torch::Tensor if found, undefined tensor if not found
     *
     * IMPORTANT: If copy=false, the returned tensor's lifetime is tied to
     * the store. Do not use the tensor after the store is modified or destroyed.
     */
    static torch::Tensor load(const GnnTensorStore& store,
                              const std::string& key,
                              bool copy = false);

    /**
     * @brief Load a tensor and optionally move to a specific device.
     *
     * @param store Source tensor store
     * @param key Tensor identifier
     * @param device Target device (CPU, CUDA:0, etc.)
     * @return torch::Tensor if found, undefined tensor if not found
     */
    static torch::Tensor load_to_device(const GnnTensorStore& store,
                                        const std::string& key,
                                        torch::Device device);

    /**
     * @brief Load multiple tensors at once.
     *
     * @param store Source tensor store
     * @param keys List of tensor identifiers
     * @param copy If true, always copy data
     * @return Vector of tensors (undefined tensors for missing keys)
     */
    static std::vector<torch::Tensor> load_batch(const GnnTensorStore& store,
                                                  const std::vector<std::string>& keys,
                                                  bool copy = false);

    // ========================================================================
    // Utility Operations
    // ========================================================================

    /**
     * @brief Check if a tensor in the store is compatible with a torch tensor shape/dtype.
     *
     * @param store Tensor store to check
     * @param key Tensor identifier
     * @param expected_shape Expected shape (use -1 for dynamic dimensions)
     * @param expected_dtype Expected data type
     * @return true if compatible, false otherwise
     */
    static bool is_compatible(const GnnTensorStore& store,
                              const std::string& key,
                              const std::vector<int64_t>& expected_shape,
                              GnnDtype expected_dtype);

    /**
     * @brief Get tensor info string for debugging.
     *
     * @param store Tensor store to query
     * @param key Tensor identifier
     * @return Human-readable string describing the tensor, or "not found"
     */
    static std::string tensor_info(const GnnTensorStore& store,
                                   const std::string& key);
};

} // namespace mdb::gnn
