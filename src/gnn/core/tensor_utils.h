#pragma once

/**
 * @file tensor_utils.h
 * @brief LibTorch tensor utilities for GNN operations.
 *
 * This file provides utility functions for working with LibTorch tensors
 * in the GNN subsystem. For tensor storage, use GnnTensorStore and
 * GnnTensorConverter from gnn/storage/.
 *
 * IMPORTANT: This file does NOT provide tensor storage functionality.
 * Use mdb::gnn::GnnTensorStore for storing and loading tensors.
 *
 * @see gnn/storage/gnn_tensor_store.h for tensor storage
 * @see gnn/storage/gnn_tensor_converter.h for LibTorch conversion
 */

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "cuda_context.h"
#include "gnn/common/feature_matrix.h"

namespace mdb::gnn {

// ============================================================================
// Sparse Adjacency Structure
// ============================================================================

/**
 * @brief Sparse adjacency tensor in COO format.
 *
 * Represents a sparse matrix for GNN message passing.
 */
struct SparseAdjacency {
    torch::Tensor indices;           ///< [2, num_edges] - row and column indices (COO format)
    torch::Tensor values;            ///< [num_edges] - edge weights (usually 1.0)
    std::array<int64_t, 2> size;     ///< [num_src_nodes, num_dst_nodes]

    /**
     * @brief Get number of non-zero entries (edges).
     */
    int64_t nnz() const {
        return indices.defined() ? indices.size(1) : 0;
    }
};

// ============================================================================
// Feature Matrix Construction
// ============================================================================

/**
 * @brief Build a feature matrix from node IDs and feature tensors.
 *
 * @param node_ids Vector of node ObjectIds
 * @param features Vector of feature tensors (one per node, same dimension)
 * @param device Target device for the output tensor
 * @return FeatureMatrix with features and ID mappings
 */
FeatureMatrix build_feature_matrix(
    const std::vector<ObjectId>& node_ids,
    const std::vector<torch::Tensor>& features,
    torch::Device device = torch::kCPU
);

/**
 * @brief Stack multiple 1D tensors into a feature matrix.
 *
 * @param node_ids Vector of node ObjectIds
 * @param tensors Vector of 1D tensors to stack
 * @return FeatureMatrix with [N, D] features tensor
 */
FeatureMatrix stack_features(
    const std::vector<ObjectId>& node_ids,
    const std::vector<torch::Tensor>& tensors
);

// ============================================================================
// Sparse Adjacency Construction
// ============================================================================

/**
 * @brief Build a sparse adjacency tensor from edge list.
 *
 * @param edges Vector of (source_idx, target_idx) pairs using local indices
 * @param num_src_nodes Number of source nodes
 * @param num_dst_nodes Number of destination nodes
 * @param device Target device for output tensors
 * @return SparseAdjacency in COO format
 */
SparseAdjacency build_adjacency(
    const std::vector<std::pair<int64_t, int64_t>>& edges,
    int64_t num_src_nodes,
    int64_t num_dst_nodes,
    torch::Device device = torch::kCPU
);

/**
 * @brief Build a sparse adjacency tensor with edge weights.
 *
 * @param edges Vector of (source_idx, target_idx) pairs
 * @param weights Edge weights (must have same size as edges)
 * @param num_src_nodes Number of source nodes
 * @param num_dst_nodes Number of destination nodes
 * @param device Target device for output tensors
 * @return SparseAdjacency in COO format with custom weights
 */
SparseAdjacency build_adjacency_weighted(
    const std::vector<std::pair<int64_t, int64_t>>& edges,
    const std::vector<float>& weights,
    int64_t num_src_nodes,
    int64_t num_dst_nodes,
    torch::Device device = torch::kCPU
);

// ============================================================================
// Memory Optimization Utilities
// ============================================================================

/**
 * @brief Check if pinned memory allocation is available.
 *
 * Pinned memory requires CUDA to be initialized and working.
 * Use this to check before calling create_pinned_tensor() if you
 * need to know whether the result will actually be pinned.
 *
 * @return true if pinned memory can be allocated
 */
bool is_pinned_memory_available();

/**
 * @brief Create a pinned memory tensor for faster CPU-GPU transfer.
 *
 * Pinned (page-locked) memory enables asynchronous transfers and
 * higher transfer bandwidth.
 *
 * If CUDA is not available or pinned memory allocation fails,
 * this function returns a regular (non-pinned) CPU tensor instead.
 *
 * @param shape Shape of the tensor
 * @param dtype Data type
 * @return Pinned tensor on CPU (or regular CPU tensor if pinned unavailable)
 */
torch::Tensor create_pinned_tensor(
    const std::vector<int64_t>& shape,
    torch::Dtype dtype = torch::kFloat32
);

/**
 * @brief Transfer a tensor to GPU asynchronously.
 *
 * Requires the source tensor to be in pinned memory for best performance.
 *
 * @param tensor Source tensor (preferably pinned)
 * @param device Target device (default: CUDA if available)
 * @return Tensor on target device (transfer may still be in progress)
 */
torch::Tensor to_device_async(
    const torch::Tensor& tensor,
    torch::Device device
);

// ============================================================================
// Tensor Validation Utilities
// ============================================================================

/**
 * @brief Validate tensor shape matches expected dimensions.
 *
 * @param tensor Tensor to validate
 * @param expected_shape Expected shape (-1 for dynamic dimensions)
 * @param name Tensor name for error messages
 * @throws std::runtime_error if shape doesn't match
 */
void validate_shape(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& expected_shape,
    const std::string& name = "tensor"
);

/**
 * @brief Ensure tensor is contiguous and on specified device.
 *
 * @param tensor Input tensor
 * @param device Target device
 * @return Contiguous tensor on target device
 */
torch::Tensor ensure_contiguous(
    const torch::Tensor& tensor,
    torch::Device device
);

// ============================================================================
// Inline Utility Functions
// ============================================================================

/**
 * @brief Calculate total number of elements from shape.
 */
inline int64_t numel_from_shape(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    return numel;
}

/**
 * @brief Get the preferred device for GNN computation.
 *
 * Returns CUDA device if available, otherwise CPU.
 */
inline torch::Device get_preferred_device() {
    return CudaContext::instance().torch_device();
}

/**
 * @brief Get tensor options for the preferred device.
 */
inline torch::TensorOptions get_default_options(torch::Dtype dtype = torch::kFloat32) {
    return torch::TensorOptions()
        .dtype(dtype)
        .device(get_preferred_device());
}

} // namespace mdb::gnn
