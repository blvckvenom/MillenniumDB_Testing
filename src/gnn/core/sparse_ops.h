#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

namespace mdb::gnn::ops {

// ============================================================================
// Scatter Operations
// ============================================================================
//
// Scatter aggregates values from a source tensor into a destination tensor
// based on index. This is the core operation for GNN message aggregation.
//
// Visual explanation of scatter_sum:
//
//   src (source values):     [a, b, c, d, e, f]
//   idx (target indices):    [0, 1, 0, 2, 1, 0]
//
//   Result (scatter_sum):
//     dst[0] = a + c + f
//     dst[1] = b + e
//     dst[2] = d
//
// In GNN context:
//   src = messages from neighbors
//   idx = which node receives each message
//   dst = aggregated messages per node
//
// ============================================================================

/**
 * @brief Scatter operation with configurable reduction.
 *
 * Aggregates values from src into output based on idx:
 *   output[idx[i]] = reduce(output[idx[i]], src[i])
 *
 * @param src Source tensor [N, *] - values to scatter
 * @param idx Index tensor [N] - target indices for each source element
 * @param dim_size Size of output dimension (if -1, inferred from max(idx)+1)
 * @param reduce Reduction operation: "sum", "mean", "max", "min"
 * @return Aggregated tensor [dim_size, *]
 *
 * @note For "mean" reduction, zero counts are handled (return 0, not NaN)
 */
torch::Tensor scatter(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size = -1,
    const std::string& reduce = "sum"
);

/**
 * @brief Scatter sum: dst[idx[i]] += src[i]
 */
torch::Tensor scatter_sum(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size = -1
);

/**
 * @brief Scatter mean: dst[idx[i]] = mean(src[j] where idx[j] == idx[i])
 */
torch::Tensor scatter_mean(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size = -1
);

/**
 * @brief Scatter max: dst[idx[i]] = max(src[j] where idx[j] == idx[i])
 *
 * @return Tuple of (max_values, argmax_indices)
 */
std::tuple<torch::Tensor, torch::Tensor> scatter_max(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size = -1
);

/**
 * @brief Scatter min: dst[idx[i]] = min(src[j] where idx[j] == idx[i])
 *
 * @return Tuple of (min_values, argmin_indices)
 */
std::tuple<torch::Tensor, torch::Tensor> scatter_min(
    const torch::Tensor& src,
    const torch::Tensor& idx,
    int64_t dim_size = -1
);

// ============================================================================
// Gather Operations
// ============================================================================

/**
 * @brief Gather rows from source based on index.
 *
 * out[i] = src[idx[i]]
 *
 * @param src Source tensor [M, *]
 * @param idx Index tensor [N]
 * @return Gathered tensor [N, *]
 *
 * @note This is equivalent to torch::index_select(src, 0, idx)
 *       but provided for API consistency.
 */
torch::Tensor gather(
    const torch::Tensor& src,
    const torch::Tensor& idx
);

// ============================================================================
// Sparse Matrix Operations
// ============================================================================

/**
 * @brief Sparse-Dense Matrix Multiplication (SpMM).
 *
 * Computes: out = A @ B where A is sparse (COO) and B is dense.
 *
 * This is the core operation for GNN message passing:
 *   out[i] = sum_j(A[i,j] * B[j])
 *
 * @param indices Sparse indices [2, nnz] where [0,:]=rows, [1,:]=cols
 * @param values Sparse values [nnz] - edge weights
 * @param dense Dense matrix [N, D] - node features
 * @param M Number of rows in sparse matrix (output rows)
 * @return Result [M, D]
 */
torch::Tensor spmm(
    const torch::Tensor& indices,
    const torch::Tensor& values,
    const torch::Tensor& dense,
    int64_t M
);

/**
 * @brief SpMM with default edge weights of 1.0.
 *
 * @param indices Sparse indices [2, nnz]
 * @param dense Dense matrix [N, D]
 * @param M Number of output rows
 * @return Result [M, D]
 */
torch::Tensor spmm_unweighted(
    const torch::Tensor& indices,
    const torch::Tensor& dense,
    int64_t M
);

/**
 * @brief Sparse matrix transpose.
 *
 * Swaps rows and columns in a COO sparse tensor.
 *
 * @param indices Sparse indices [2, nnz]
 * @param values Sparse values [nnz]
 * @param size Original matrix size [M, N]
 * @return Tuple of (transposed_indices, values, new_size [N, M])
 */
std::tuple<torch::Tensor, torch::Tensor, std::array<int64_t, 2>> sparse_transpose(
    const torch::Tensor& indices,
    const torch::Tensor& values,
    const std::array<int64_t, 2>& size
);

// ============================================================================
// Segment Operations
// ============================================================================

/**
 * @brief Segment reduction over contiguous segments.
 *
 * Reduces values within segments defined by segment_ids.
 *
 * @param src Source tensor [N, *]
 * @param segment_ids Segment ID for each element [N] (must be sorted)
 * @param num_segments Number of segments (if -1, inferred from max+1)
 * @param reduce Reduction operation: "sum", "mean", "max", "min"
 * @return Reduced tensor [num_segments, *]
 */
torch::Tensor segment_reduce(
    const torch::Tensor& src,
    const torch::Tensor& segment_ids,
    int64_t num_segments = -1,
    const std::string& reduce = "sum"
);

/**
 * @brief Segment reduction using CSR-style offsets.
 *
 * More efficient when you have CSR row pointers (e.g., from cuSPARSE).
 *
 * @param src Source tensor [N, *]
 * @param indptr CSR row pointers [M+1] where segment i spans [indptr[i], indptr[i+1])
 * @param reduce Reduction operation
 * @return Reduced tensor [M, *]
 */
torch::Tensor segment_csr(
    const torch::Tensor& src,
    const torch::Tensor& indptr,
    const std::string& reduce = "sum"
);

// ============================================================================
// Softmax Operations
// ============================================================================

/**
 * @brief Softmax over edges grouped by target node.
 *
 * For attention mechanisms (GAT):
 *   attention[e] = softmax(score[e]) where softmax is computed over
 *   all edges pointing to the same target node.
 *
 * @param src Attention scores [E]
 * @param index Target node for each edge [E]
 * @param num_nodes Number of nodes N
 * @return Softmax attention weights [E]
 *
 * @note Numerically stable: subtracts max before exp.
 */
torch::Tensor edge_softmax(
    const torch::Tensor& src,
    const torch::Tensor& index,
    int64_t num_nodes
);

/**
 * @brief Softmax over edges with separate query and key.
 *
 * For multi-head attention:
 *   score[e] = (q[dst[e]] @ k[src[e]]) / sqrt(d)
 *   attention[e] = softmax(score[e]) over edges to same dst
 *
 * @param query Query tensor [N, H, D] - H heads, D dim per head
 * @param key Key tensor [N, H, D]
 * @param edge_index Edge indices [2, E]
 * @return Attention weights [E, H]
 */
torch::Tensor multihead_edge_softmax(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& edge_index
);

// ============================================================================
// Message Passing Interface
// ============================================================================

/**
 * @brief Message passing result.
 */
struct MessagePassingResult {
    torch::Tensor output;       ///< Updated node features [N, D_out]
    torch::Tensor attention;    ///< Attention weights [E] (optional, for GAT)
};

/**
 * @brief Simple message passing: aggregate neighbor features.
 *
 * Implements: h_i^{new} = AGG({h_j : j in N(i)})
 *
 * @param x Node features [N, D]
 * @param edge_index Edge indices [2, E] where [0,:]=src, [1,:]=dst
 * @param aggr Aggregation method: "sum", "mean", "max"
 * @return Aggregated features [N, D]
 */
torch::Tensor message_passing_simple(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    const std::string& aggr = "sum"
);

/**
 * @brief Message passing with edge weights.
 *
 * Implements: h_i^{new} = AGG({w_ij * h_j : j in N(i)})
 *
 * @param x Node features [N, D]
 * @param edge_index Edge indices [2, E]
 * @param edge_weight Edge weights [E]
 * @param aggr Aggregation method
 * @return Aggregated features [N, D]
 */
torch::Tensor message_passing_weighted(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    const torch::Tensor& edge_weight,
    const std::string& aggr = "sum"
);

// ============================================================================
// Degree Computation
// ============================================================================

/**
 * @brief Compute node degrees from edge index.
 *
 * @param edge_index Edge indices [2, E]
 * @param num_nodes Number of nodes
 * @param direction "in" for in-degree, "out" for out-degree, "both" for total
 * @return Degree tensor [N]
 */
torch::Tensor compute_degree(
    const torch::Tensor& edge_index,
    int64_t num_nodes,
    const std::string& direction = "in"
);

/**
 * @brief Compute symmetric normalization coefficients.
 *
 * For GCN-style normalization: D^{-1/2} A D^{-1/2}
 *
 * @param edge_index Edge indices [2, E]
 * @param num_nodes Number of nodes
 * @return Edge weights [E] for normalized adjacency
 */
torch::Tensor symmetric_norm(
    const torch::Tensor& edge_index,
    int64_t num_nodes
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Add self-loops to edge index.
 *
 * @param edge_index Edge indices [2, E]
 * @param num_nodes Number of nodes
 * @return Edge indices with self-loops [2, E+N]
 */
torch::Tensor add_self_loops(
    const torch::Tensor& edge_index,
    int64_t num_nodes
);

/**
 * @brief Remove self-loops from edge index.
 *
 * @param edge_index Edge indices [2, E]
 * @return Edge indices without self-loops [2, E']
 */
torch::Tensor remove_self_loops(
    const torch::Tensor& edge_index
);

/**
 * @brief Convert edge index to CSR format.
 *
 * @param edge_index Edge indices [2, E] (must be sorted by source)
 * @param num_nodes Number of nodes
 * @return Tuple of (indptr [N+1], indices [E])
 */
std::tuple<torch::Tensor, torch::Tensor> to_csr(
    const torch::Tensor& edge_index,
    int64_t num_nodes
);

/**
 * @brief Sort edges by source node.
 *
 * @param edge_index Edge indices [2, E]
 * @return Tuple of (sorted_edge_index, sort_indices)
 */
std::tuple<torch::Tensor, torch::Tensor> sort_edge_index(
    const torch::Tensor& edge_index
);

} // namespace mdb::gnn::ops
