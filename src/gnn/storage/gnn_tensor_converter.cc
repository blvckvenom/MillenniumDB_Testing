#include "gnn_tensor_converter.h"

#include <sstream>
#include <stdexcept>

namespace mdb::gnn {

// ============================================================================
// Type Conversion
// ============================================================================

torch::ScalarType GnnTensorConverter::to_torch_dtype(GnnDtype dtype) {
    switch (dtype) {
        case GnnDtype::FLOAT32: return torch::kFloat32;
        case GnnDtype::FLOAT64: return torch::kFloat64;
        case GnnDtype::INT32:   return torch::kInt32;
        case GnnDtype::INT64:   return torch::kInt64;
        case GnnDtype::UINT8:   return torch::kUInt8;
        case GnnDtype::BOOL:    return torch::kBool;
    }
    throw std::invalid_argument("Unknown GnnDtype for torch conversion");
}

GnnDtype GnnTensorConverter::from_torch_dtype(torch::ScalarType torch_dtype) {
    switch (torch_dtype) {
        case torch::kFloat32: return GnnDtype::FLOAT32;
        case torch::kFloat64: return GnnDtype::FLOAT64;
        case torch::kInt32:   return GnnDtype::INT32;
        case torch::kInt64:   return GnnDtype::INT64;
        case torch::kUInt8:   return GnnDtype::UINT8;
        case torch::kBool:    return GnnDtype::BOOL;
        default:
            throw std::invalid_argument(
                "Unsupported torch dtype for GNN storage: " +
                std::string(torch::toString(torch_dtype)));
    }
}

// ============================================================================
// Store Operations
// ============================================================================

bool GnnTensorConverter::store(GnnTensorStore& store,
                               const std::string& key,
                               const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return false;
    }

    // Ensure tensor is on CPU and contiguous
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();

    // Get dtype
    GnnDtype dtype;
    try {
        dtype = from_torch_dtype(cpu_tensor.scalar_type());
    } catch (const std::invalid_argument&) {
        return false;  // Unsupported dtype
    }

    // Get shape
    std::vector<int64_t> shape(cpu_tensor.sizes().begin(), cpu_tensor.sizes().end());

    // Store raw data
    return store.store(key, cpu_tensor.data_ptr(), shape, dtype);
}

size_t GnnTensorConverter::store_batch(
    GnnTensorStore& store,
    const std::vector<std::pair<std::string, torch::Tensor>>& tensors) {

    size_t success_count = 0;
    for (const auto& [key, tensor] : tensors) {
        if (GnnTensorConverter::store(store, key, tensor)) {
            ++success_count;
        }
    }
    return success_count;
}

// ============================================================================
// Load Operations
// ============================================================================

torch::Tensor GnnTensorConverter::load(const GnnTensorStore& store,
                                       const std::string& key,
                                       bool copy) {
    GnnTensorView view = store.load(key);
    if (!view.valid()) {
        return torch::Tensor();  // Undefined tensor
    }

    torch::ScalarType torch_dtype = to_torch_dtype(view.dtype());

    // Convert shape
    std::vector<int64_t> sizes(view.shape().begin(), view.shape().end());

    if (copy) {
        // Create new tensor and copy data
        auto options = torch::TensorOptions()
            .dtype(torch_dtype)
            .device(torch::kCPU);

        torch::Tensor result = torch::empty(sizes, options);
        std::memcpy(result.data_ptr(), view.data(), view.byte_size());
        return result;
    } else {
        // Zero-copy: wrap existing memory
        // CAUTION: The tensor's lifetime is tied to the store
        auto options = torch::TensorOptions()
            .dtype(torch_dtype)
            .device(torch::kCPU);

        // from_blob creates a tensor that references external memory
        // The deleter is empty because the store owns the memory
        torch::Tensor result = torch::from_blob(
            const_cast<void*>(view.data()),
            sizes,
            /*deleter=*/[](void*) {},  // No-op deleter
            options
        );

        return result;
    }
}

torch::Tensor GnnTensorConverter::load_to_device(const GnnTensorStore& store,
                                                  const std::string& key,
                                                  torch::Device device) {
    // Always copy when moving to device (especially for GPU)
    torch::Tensor cpu_tensor = load(store, key, /*copy=*/true);
    if (!cpu_tensor.defined()) {
        return cpu_tensor;
    }
    return cpu_tensor.to(device);
}

std::vector<torch::Tensor> GnnTensorConverter::load_batch(
    const GnnTensorStore& store,
    const std::vector<std::string>& keys,
    bool copy) {

    std::vector<torch::Tensor> results;
    results.reserve(keys.size());

    for (const auto& key : keys) {
        results.push_back(load(store, key, copy));
    }

    return results;
}

// ============================================================================
// Utility Operations
// ============================================================================

bool GnnTensorConverter::is_compatible(const GnnTensorStore& store,
                                       const std::string& key,
                                       const std::vector<int64_t>& expected_shape,
                                       GnnDtype expected_dtype) {
    auto metadata = store.get_metadata(key);
    if (!metadata) {
        return false;
    }

    // Check dtype
    if (metadata->dtype != expected_dtype) {
        return false;
    }

    // Check shape (dimension count must match)
    if (metadata->shape.size() != expected_shape.size()) {
        return false;
    }

    // Check each dimension (-1 means dynamic/don't care)
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (expected_shape[i] != -1 && expected_shape[i] != metadata->shape[i]) {
            return false;
        }
    }

    return true;
}

std::string GnnTensorConverter::tensor_info(const GnnTensorStore& store,
                                            const std::string& key) {
    auto metadata = store.get_metadata(key);
    if (!metadata) {
        return "not found";
    }

    std::ostringstream oss;
    oss << dtype_name(metadata->dtype) << "[";
    for (size_t i = 0; i < metadata->shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << metadata->shape[i];
    }
    oss << "] (" << metadata->byte_size << " bytes)";
    return oss.str();
}

} // namespace mdb::gnn
