#include "sparse_ops.h"

#include <algorithm>
#include <stdexcept>

#include "cuda_context.h"

namespace mdb::gnn::ops {

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

/**
 * @brief Infer dim_size from max index + 1.
 */
int64_t infer_dim_size(const torch::Tensor& idx) {
    if (idx.numel() == 0) {
        return 0;
    }
    return idx.max().item<int64_t>() + 1;
}

/**
 * @brief Validate scatter inputs.
 */
void validate_scatter_inputs(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size
) {
    if (src.dim() < 1) {
        throw std::invalid_argument("src must have at least 1 dimension");
    }
    if (idx.dim() != 1) {
        throw std::invalid_argument("idx must be 1-dimensional");
    }
    if (src.size(0) != idx.size(0)) {
        throw std::invalid_argument(
            "src.size(0) must equal idx.size(0)"
        );
    }
    if (dim_size <= 0) {
        throw std::invalid_argument("dim_size must be positive");
    }
}

} // anonymous namespace

// ============================================================================
// Scatter Operations
// ============================================================================

torch::Tensor scatter(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size,
    const std::string& reduce
) {
    if (reduce == "sum") {
        return scatter_sum(src, idx, dim_size);
    } else if (reduce == "mean") {
        return scatter_mean(src, idx, dim_size);
    } else if (reduce == "max") {
        return std::get<0>(scatter_max(src, idx, dim_size));
    } else if (reduce == "min") {
        return std::get<0>(scatter_min(src, idx, dim_size));
    } else {
        throw std::invalid_argument(
            "Unknown reduce operation: " + reduce +
            ". Supported: sum, mean, max, min"
        );
    }
}

torch::Tensor scatter_sum(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size
) {
    // Infer dim_size if not provided
    if (dim_size < 0) {
        dim_size = infer_dim_size(idx);
    }

    if (src.numel() == 0) {
        auto shape = src.sizes().vec();
        shape[0] = dim_size;
        return torch::zeros(shape, src.options());
    }

    validate_scatter_inputs(src, idx, dim_size);

    // Build output shape: [dim_size, ...rest of src shape]
    auto output_shape = src.sizes().vec();
    output_shape[0] = dim_size;

    auto out = torch::zeros(output_shape, src.options());

    // Use scatter_add_ for the operation
    // We need to expand idx to match src dimensions
    auto idx_expanded = idx.view({-1});
    for (int64_t d = 1; d < src.dim(); ++d) {
        idx_expanded = idx_expanded.unsqueeze(-1);
    }
    idx_expanded = idx_expanded.expand_as(src);

    out.scatter_add_(0, idx_expanded, src);

    return out;
}

torch::Tensor scatter_mean(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size
) {
    if (dim_size < 0) {
        dim_size = infer_dim_size(idx);
    }

    if (src.numel() == 0) {
        auto shape = src.sizes().vec();
        shape[0] = dim_size;
        return torch::zeros(shape, src.options());
    }

    validate_scatter_inputs(src, idx, dim_size);

    auto sum = scatter_sum(src, idx, dim_size);

    // Count occurrences of each index
    auto ones = torch::ones({idx.size(0)}, idx.options().dtype(src.scalar_type()));
    auto count = scatter_sum(ones, idx, dim_size);

    // Prevent division by zero
    count = count.clamp_min(1.0);

    // Broadcast count to match sum dimensions
    for (int64_t d = 1; d < sum.dim(); ++d) {
        count = count.unsqueeze(-1);
    }

    return sum / count;
}

std::tuple<torch::Tensor, torch::Tensor> scatter_max(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size
) {
    if (dim_size < 0) {
        dim_size = infer_dim_size(idx);
    }

    if (src.numel() == 0) {
        auto shape = src.sizes().vec();
        shape[0] = dim_size;
        auto out = torch::full(shape, -std::numeric_limits<float>::infinity(), src.options());
        auto argmax = torch::full({dim_size}, -1, idx.options());
        return std::make_tuple(out, argmax);
    }

    validate_scatter_inputs(src, idx, dim_size);

    // Build output shape
    auto output_shape = src.sizes().vec();
    output_shape[0] = dim_size;

    // -inf is the identity for max: untouched rows stay -inf rather than 0.
    auto out = torch::full(output_shape, -std::numeric_limits<float>::infinity(), src.options());

    // For argmax, we'll build on CPU then transfer
    auto argmax_cpu = torch::full({dim_size}, -1, torch::TensorOptions().dtype(torch::kInt64));

    // Expand idx for scatter
    auto idx_expanded = idx.view({-1});
    for (int64_t d = 1; d < src.dim(); ++d) {
        idx_expanded = idx_expanded.unsqueeze(-1);
    }
    idx_expanded = idx_expanded.expand_as(src);

    // Use scatter_reduce with max
    // Note: scatter_reduce is available in PyTorch 1.12+
    out.scatter_reduce_(0, idx_expanded, src, "amax", /*include_self=*/false);

    // Compute argmax by finding which source indices achieved the max
    // This is approximate for 2D+ tensors - we just track first dimension
    auto max_mask = (src == out.index_select(0, idx));
    if (max_mask.dim() > 1) {
        max_mask = max_mask.all(-1);  // Reduce to 1D
    }

    // Move to CPU for the argmax loop (item() calls are expensive on GPU)
    auto max_mask_cpu = max_mask.to(torch::kCPU);
    auto idx_cpu = idx.to(torch::kCPU);
    auto max_mask_accessor = max_mask_cpu.accessor<bool, 1>();
    auto idx_accessor = idx_cpu.accessor<int64_t, 1>();
    auto argmax_data = argmax_cpu.data_ptr<int64_t>();

    // Get first matching index for each target
    for (int64_t i = 0; i < idx_cpu.size(0); ++i) {
        if (max_mask_accessor[i]) {
            int64_t target = idx_accessor[i];
            if (argmax_data[target] < 0) {
                argmax_data[target] = i;
            }
        }
    }

    // Move argmax to original device
    auto argmax = argmax_cpu.to(idx.device());

    return std::make_tuple(out, argmax);
}

std::tuple<torch::Tensor, torch::Tensor> scatter_min(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size
) {
    if (dim_size < 0) {
        dim_size = infer_dim_size(idx);
    }

    if (src.numel() == 0) {
        auto shape = src.sizes().vec();
        shape[0] = dim_size;
        auto out = torch::full(shape, std::numeric_limits<float>::infinity(), src.options());
        auto argmin = torch::full({dim_size}, -1, idx.options());
        return std::make_tuple(out, argmin);
    }

    validate_scatter_inputs(src, idx, dim_size);

    auto output_shape = src.sizes().vec();
    output_shape[0] = dim_size;

    auto out = torch::full(output_shape, std::numeric_limits<float>::infinity(), src.options());

    // For argmin, we'll build on CPU then transfer
    auto argmin_cpu = torch::full({dim_size}, -1, torch::TensorOptions().dtype(torch::kInt64));

    auto idx_expanded = idx.view({-1});
    for (int64_t d = 1; d < src.dim(); ++d) {
        idx_expanded = idx_expanded.unsqueeze(-1);
    }
    idx_expanded = idx_expanded.expand_as(src);

    out.scatter_reduce_(0, idx_expanded, src, "amin", /*include_self=*/false);

    // Compute argmin (similar to argmax)
    auto min_mask = (src == out.index_select(0, idx));
    if (min_mask.dim() > 1) {
        min_mask = min_mask.all(-1);
    }

    // Move to CPU for the argmin loop (item() calls are expensive on GPU)
    auto min_mask_cpu = min_mask.to(torch::kCPU);
    auto idx_cpu = idx.to(torch::kCPU);
    auto min_mask_accessor = min_mask_cpu.accessor<bool, 1>();
    auto idx_accessor = idx_cpu.accessor<int64_t, 1>();
    auto argmin_data = argmin_cpu.data_ptr<int64_t>();

    for (int64_t i = 0; i < idx_cpu.size(0); ++i) {
        if (min_mask_accessor[i]) {
            int64_t target = idx_accessor[i];
            if (argmin_data[target] < 0) {
                argmin_data[target] = i;
            }
        }
    }

    // Move argmin to original device
    auto argmin = argmin_cpu.to(idx.device());

    return std::make_tuple(out, argmin);
}

// ============================================================================
// Gather Operations
// ============================================================================

torch::Tensor gather(
    const torch::Tensor& src,
    const torch::Tensor& idx
) {
    return src.index_select(0, idx);
}

// ============================================================================
// Sparse Matrix Operations
// ============================================================================

torch::Tensor spmm(
    const torch::Tensor& indices,
    const torch::Tensor& values,
    const torch::Tensor& dense,
    int64_t M
) {
    if (indices.size(0) != 2) {
        throw std::invalid_argument("indices must be [2, nnz]");
    }
    if (indices.size(1) != values.size(0)) {
        throw std::invalid_argument("indices and values must have same nnz");
    }

    // indices: [2, nnz] where [0,:]=rows, [1,:]=cols
    auto rows = indices[0];
    auto cols = indices[1];

    // Gather: get dense rows for each edge
    auto gathered = dense.index_select(0, cols);  // [nnz, D]

    // Scale by edge values
    auto scaled = gathered * values.unsqueeze(1);  // [nnz, D]

    // Scatter: aggregate to output nodes
    return scatter_sum(scaled, rows, M);
}

torch::Tensor spmm_unweighted(
    const torch::Tensor& indices,
    const torch::Tensor& dense,
    int64_t M
) {
    auto rows = indices[0];
    auto cols = indices[1];

    auto gathered = dense.index_select(0, cols);
    return scatter_sum(gathered, rows, M);
}

std::tuple<torch::Tensor, torch::Tensor, std::array<int64_t, 2>> sparse_transpose(
    const torch::Tensor& indices,
    const torch::Tensor& values,
    const std::array<int64_t, 2>& size
) {
    // Swap rows and columns
    auto transposed_indices = torch::stack({indices[1], indices[0]}, 0);

    return std::make_tuple(
        transposed_indices,
        values.clone(),
        std::array<int64_t, 2>{size[1], size[0]}
    );
}

// ============================================================================
// Segment Operations
// ============================================================================

torch::Tensor segment_reduce(
    const torch::Tensor& src,
    const torch::Tensor& segment_ids,
    int64_t num_segments,
    const std::string& reduce
) {
    // segment_reduce is essentially scatter with sorted indices
    return scatter(src, segment_ids, num_segments, reduce);
}

torch::Tensor segment_csr(
    const torch::Tensor& src,
    const torch::Tensor& indptr,
    const std::string& reduce
) {
    int64_t num_segments = indptr.size(0) - 1;

    if (num_segments <= 0) {
        auto shape = src.sizes().vec();
        shape[0] = 0;
        return torch::empty(shape, src.options());
    }

    // Move indptr to CPU for accessor (required for CUDA tensors)
    auto indptr_cpu = indptr.to(torch::kCPU);
    auto indptr_accessor = indptr_cpu.accessor<int64_t, 1>();

    // Build segment_ids on CPU, then move to source device
    auto segment_ids_cpu = torch::empty({src.size(0)}, torch::TensorOptions().dtype(torch::kInt64));
    auto segment_ids_data = segment_ids_cpu.data_ptr<int64_t>();

    for (int64_t seg = 0; seg < num_segments; ++seg) {
        int64_t start = indptr_accessor[seg];
        int64_t end = indptr_accessor[seg + 1];
        for (int64_t i = start; i < end; ++i) {
            segment_ids_data[i] = seg;
        }
    }

    // Move segment_ids to same device as src
    auto segment_ids = segment_ids_cpu.to(src.device());

    return segment_reduce(src, segment_ids, num_segments, reduce);
}

// ============================================================================
// Softmax Operations
// ============================================================================

torch::Tensor edge_softmax(
    const torch::Tensor& src,
    const torch::Tensor& index,
    int64_t num_nodes
) {
    // Numerical stability: subtract max per segment
    auto max_vals = std::get<0>(scatter_max(src, index, num_nodes));
    auto max_per_edge = max_vals.index_select(0, index);
    auto shifted = src - max_per_edge;

    // Exp
    auto exp_vals = shifted.exp();

    // Sum per segment
    auto sum_vals = scatter_sum(exp_vals, index, num_nodes);
    auto sum_per_edge = sum_vals.index_select(0, index);

    // Normalize with small epsilon for stability
    return exp_vals / (sum_per_edge + 1e-16);
}

torch::Tensor multihead_edge_softmax(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& edge_index
) {
    // query, key: [N, H, D]
    // edge_index: [2, E]

    auto src_idx = edge_index[0];
    auto dst_idx = edge_index[1];

    // Gather query and key for each edge
    auto q = query.index_select(0, dst_idx);  // [E, H, D]
    auto k = key.index_select(0, src_idx);    // [E, H, D]

    // Compute attention scores: (q * k).sum(-1) / sqrt(D)
    int64_t D = query.size(2);
    auto scores = (q * k).sum(-1) / std::sqrt(static_cast<float>(D));  // [E, H]

    // Apply softmax per head, per destination node
    int64_t num_nodes = query.size(0);
    int64_t num_heads = query.size(1);

    auto attention = torch::empty_like(scores);
    for (int64_t h = 0; h < num_heads; ++h) {
        attention.select(1, h) = edge_softmax(
            scores.select(1, h),
            dst_idx,
            num_nodes
        );
    }

    return attention;
}

// ============================================================================
// Message Passing Interface
// ============================================================================

torch::Tensor message_passing_simple(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    const std::string& aggr
) {
    auto src_idx = edge_index[0];
    auto dst_idx = edge_index[1];

    // Gather source features
    auto messages = x.index_select(0, src_idx);

    // Aggregate to destination
    return scatter(messages, dst_idx, x.size(0), aggr);
}

torch::Tensor message_passing_weighted(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    const torch::Tensor& edge_weight,
    const std::string& aggr
) {
    auto src_idx = edge_index[0];
    auto dst_idx = edge_index[1];

    // Gather source features
    auto messages = x.index_select(0, src_idx);

    // Weight messages
    auto weighted_messages = messages * edge_weight.unsqueeze(-1);

    // Aggregate to destination
    return scatter(weighted_messages, dst_idx, x.size(0), aggr);
}

// ============================================================================
// Degree Computation
// ============================================================================

torch::Tensor compute_degree(
    const torch::Tensor& edge_index,
    int64_t num_nodes,
    const std::string& direction
) {
    auto ones = torch::ones({edge_index.size(1)}, edge_index.options().dtype(torch::kFloat32));

    if (direction == "in") {
        // In-degree: count edges pointing TO each node (dst = edge_index[1])
        return scatter_sum(ones, edge_index[1], num_nodes);
    } else if (direction == "out") {
        // Out-degree: count edges FROM each node (src = edge_index[0])
        return scatter_sum(ones, edge_index[0], num_nodes);
    } else if (direction == "both") {
        auto in_deg = scatter_sum(ones, edge_index[1], num_nodes);
        auto out_deg = scatter_sum(ones, edge_index[0], num_nodes);
        return in_deg + out_deg;
    } else {
        throw std::invalid_argument(
            "Unknown direction: " + direction + ". Use 'in', 'out', or 'both'"
        );
    }
}

torch::Tensor symmetric_norm(
    const torch::Tensor& edge_index,
    int64_t num_nodes
) {
    auto deg = compute_degree(edge_index, num_nodes, "in");

    // D^{-1/2}
    auto deg_inv_sqrt = deg.pow(-0.5);
    deg_inv_sqrt = deg_inv_sqrt.masked_fill(deg_inv_sqrt.isinf(), 0.0);

    // For each edge (i, j), compute deg_inv_sqrt[i] * deg_inv_sqrt[j]
    auto src_idx = edge_index[0];
    auto dst_idx = edge_index[1];

    auto norm_src = deg_inv_sqrt.index_select(0, src_idx);
    auto norm_dst = deg_inv_sqrt.index_select(0, dst_idx);

    return norm_src * norm_dst;
}

// ============================================================================
// Utility Functions
// ============================================================================

torch::Tensor add_self_loops(
    const torch::Tensor& edge_index,
    int64_t num_nodes
) {
    // Create self-loop indices: [[0,1,2,...], [0,1,2,...]]
    auto self_loops = torch::arange(num_nodes, edge_index.options());
    auto self_loop_index = torch::stack({self_loops, self_loops}, 0);

    // Concatenate with existing edges
    return torch::cat({edge_index, self_loop_index}, 1);
}

torch::Tensor remove_self_loops(
    const torch::Tensor& edge_index
) {
    auto src = edge_index[0];
    auto dst = edge_index[1];

    // Mask: keep edges where src != dst
    auto mask = src != dst;

    // Apply mask
    auto new_src = src.masked_select(mask);
    auto new_dst = dst.masked_select(mask);

    return torch::stack({new_src, new_dst}, 0);
}

std::tuple<torch::Tensor, torch::Tensor> to_csr(
    const torch::Tensor& edge_index,
    int64_t num_nodes
) {
    // Sort by source
    auto [sorted_edge_index, sort_indices] = sort_edge_index(edge_index);

    auto src = sorted_edge_index[0];
    auto dst = sorted_edge_index[1];

    // Move src to CPU for efficient counting
    auto src_cpu = src.to(torch::kCPU);
    auto src_accessor = src_cpu.accessor<int64_t, 1>();

    // Build indptr on CPU
    auto indptr_cpu = torch::zeros({num_nodes + 1}, torch::TensorOptions().dtype(torch::kInt64));
    auto indptr_data = indptr_cpu.data_ptr<int64_t>();

    // Count edges per source
    for (int64_t i = 0; i < src_cpu.size(0); ++i) {
        indptr_data[src_accessor[i] + 1] += 1;
    }

    // Cumsum to get offsets
    indptr_cpu = indptr_cpu.cumsum(0);

    // Move to original device
    auto indptr = indptr_cpu.to(edge_index.device());

    return std::make_tuple(indptr, dst);
}

std::tuple<torch::Tensor, torch::Tensor> sort_edge_index(
    const torch::Tensor& edge_index
) {
    auto src = edge_index[0];

    // Sort by source
    auto [sorted_src, sort_indices] = src.sort();

    // Reorder edge_index
    auto sorted_edge_index = edge_index.index_select(1, sort_indices);

    return std::make_tuple(sorted_edge_index, sort_indices);
}

} // namespace mdb::gnn::ops
