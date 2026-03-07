#include "tensor_utils.h"

#include <cstring>
#include <stdexcept>

namespace mdb::gnn {

// ============================================================================
// Feature Matrix Construction
// ============================================================================

FeatureMatrix build_feature_matrix(
    const std::vector<ObjectId>& node_ids,
    const std::vector<torch::Tensor>& features,
    torch::Device device
) {
    if (node_ids.empty()) {
        return FeatureMatrix{
            torch::empty({0, 0}, torch::TensorOptions().device(device)),
            {},
            {}
        };
    }

    if (node_ids.size() != features.size()) {
        throw std::invalid_argument(
            "node_ids and features must have the same size"
        );
    }

    // Validate all features have same dimension
    int64_t feature_dim = features[0].numel();
    for (size_t i = 1; i < features.size(); ++i) {
        if (features[i].numel() != feature_dim) {
            throw std::runtime_error(
                "Feature " + std::to_string(i) + " has different dimension"
            );
        }
    }

    // Determine dtype from first tensor
    auto dtype = features[0].scalar_type();
    int64_t num_nodes = static_cast<int64_t>(node_ids.size());

    // Allocate output tensor on CPU first for efficient copying
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCPU);
    torch::Tensor output = torch::empty({num_nodes, feature_dim}, options);

    // Build ID to row mapping and copy data
    std::unordered_map<uint64_t, int64_t> id_to_row;
    char* output_ptr = static_cast<char*>(output.data_ptr());
    size_t bytes_per_row = static_cast<size_t>(feature_dim) * output.element_size();

    for (int64_t i = 0; i < num_nodes; ++i) {
        id_to_row[node_ids[static_cast<size_t>(i)].id] = i;

        // Ensure feature is contiguous and on CPU
        torch::Tensor feature = features[static_cast<size_t>(i)].contiguous();
        if (feature.is_cuda()) {
            feature = feature.to(torch::kCPU);
        }

        std::memcpy(
            output_ptr + static_cast<size_t>(i) * bytes_per_row,
            feature.data_ptr(),
            bytes_per_row
        );
    }

    // Move to target device
    if (device != torch::kCPU) {
        output = output.to(device);
    }

    return FeatureMatrix{
        std::move(output),
        std::vector<ObjectId>(node_ids),
        std::move(id_to_row)
    };
}

FeatureMatrix stack_features(
    const std::vector<ObjectId>& node_ids,
    const std::vector<torch::Tensor>& tensors
) {
    if (tensors.empty()) {
        return FeatureMatrix{
            torch::empty({0, 0}),
            {},
            {}
        };
    }

    // Stack tensors along dimension 0
    std::vector<torch::Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(0));
    }

    torch::Tensor stacked = torch::cat(unsqueezed, 0);

    // Build ID to row mapping
    std::unordered_map<uint64_t, int64_t> id_to_row;
    for (size_t i = 0; i < node_ids.size(); ++i) {
        id_to_row[node_ids[i].id] = static_cast<int64_t>(i);
    }

    return FeatureMatrix{
        std::move(stacked),
        std::vector<ObjectId>(node_ids),
        std::move(id_to_row)
    };
}

// ============================================================================
// Sparse Adjacency Construction
// ============================================================================

SparseAdjacency build_adjacency(
    const std::vector<std::pair<int64_t, int64_t>>& edges,
    int64_t num_src_nodes,
    int64_t num_dst_nodes,
    torch::Device device
) {
    int64_t num_edges = static_cast<int64_t>(edges.size());

    if (num_edges == 0) {
        return SparseAdjacency{
            torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(device)),
            torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
            {num_src_nodes, num_dst_nodes}
        };
    }

    // Build COO indices tensor [2, num_edges]
    auto indices = torch::empty(
        {2, num_edges},
        torch::TensorOptions().dtype(torch::kInt64)
    );
    auto indices_accessor = indices.accessor<int64_t, 2>();

    for (int64_t i = 0; i < num_edges; ++i) {
        indices_accessor[0][i] = edges[static_cast<size_t>(i)].first;   // source (row)
        indices_accessor[1][i] = edges[static_cast<size_t>(i)].second;  // target (col)
    }

    // All edges have weight 1.0
    auto values = torch::ones(
        {num_edges},
        torch::TensorOptions().dtype(torch::kFloat32)
    );

    // Move to target device
    if (device != torch::kCPU) {
        indices = indices.to(device);
        values = values.to(device);
    }

    return SparseAdjacency{
        std::move(indices),
        std::move(values),
        {num_src_nodes, num_dst_nodes}
    };
}

SparseAdjacency build_adjacency_weighted(
    const std::vector<std::pair<int64_t, int64_t>>& edges,
    const std::vector<float>& weights,
    int64_t num_src_nodes,
    int64_t num_dst_nodes,
    torch::Device device
) {
    if (edges.size() != weights.size()) {
        throw std::invalid_argument(
            "edges and weights must have the same size"
        );
    }

    int64_t num_edges = static_cast<int64_t>(edges.size());

    if (num_edges == 0) {
        return SparseAdjacency{
            torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(device)),
            torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
            {num_src_nodes, num_dst_nodes}
        };
    }

    // Build COO indices tensor [2, num_edges]
    auto indices = torch::empty(
        {2, num_edges},
        torch::TensorOptions().dtype(torch::kInt64)
    );
    auto indices_accessor = indices.accessor<int64_t, 2>();

    for (int64_t i = 0; i < num_edges; ++i) {
        indices_accessor[0][i] = edges[static_cast<size_t>(i)].first;   // source (row)
        indices_accessor[1][i] = edges[static_cast<size_t>(i)].second;  // target (col)
    }

    // Create values tensor from weights
    auto values = torch::from_blob(
        const_cast<float*>(weights.data()),
        {num_edges},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();  // Clone to own the data

    // Move to target device
    if (device != torch::kCPU) {
        indices = indices.to(device);
        values = values.to(device);
    }

    return SparseAdjacency{
        std::move(indices),
        std::move(values),
        {num_src_nodes, num_dst_nodes}
    };
}

// ============================================================================
// Memory Optimization Utilities
// ============================================================================

bool is_pinned_memory_available() {
    if (!CudaContext::instance().is_cuda_available()) {
        return false;
    }

    // Try a small allocation to verify pinned memory works
    try {
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .pinned_memory(true);
        auto test_tensor = torch::empty({1}, options);
        return test_tensor.is_pinned();
    } catch (const c10::Error&) {
        return false;
    }
}

torch::Tensor create_pinned_tensor(
    const std::vector<int64_t>& shape,
    torch::Dtype dtype
) {
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCPU);

    // Pinned memory requires CUDA to be available and properly initialized
    // Fall back to regular CPU tensor if pinned memory isn't supported
    if (CudaContext::instance().is_cuda_available()) {
        try {
            options = options.pinned_memory(true);
            return torch::empty(shape, options);
        } catch (const c10::Error&) {
            // Pinned memory allocator not available, fall back to regular CPU tensor
        }
    }

    // Return non-pinned CPU tensor as fallback
    return torch::empty(shape, options);
}

torch::Tensor to_device_async(
    const torch::Tensor& tensor,
    torch::Device device
) {
    if (tensor.device() == device) {
        return tensor;
    }

    return tensor.to(device, /*non_blocking=*/true);
}

// ============================================================================
// Tensor Validation Utilities
// ============================================================================

void validate_shape(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& expected_shape,
    const std::string& name
) {
    if (!tensor.defined()) {
        throw std::runtime_error(name + " is not defined");
    }

    if (tensor.dim() != static_cast<int64_t>(expected_shape.size())) {
        throw std::runtime_error(
            name + " has " + std::to_string(tensor.dim()) +
            " dimensions, expected " + std::to_string(expected_shape.size())
        );
    }

    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (expected_shape[i] != -1 && tensor.size(static_cast<int64_t>(i)) != expected_shape[i]) {
            throw std::runtime_error(
                name + " dimension " + std::to_string(i) +
                " has size " + std::to_string(tensor.size(static_cast<int64_t>(i))) +
                ", expected " + std::to_string(expected_shape[i])
            );
        }
    }
}

torch::Tensor ensure_contiguous(
    const torch::Tensor& tensor,
    torch::Device device
) {
    torch::Tensor result = tensor.contiguous();
    if (result.device() != device) {
        result = result.to(device);
    }
    return result;
}

} // namespace mdb::gnn
