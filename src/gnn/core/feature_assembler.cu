#include "gnn/core/feature_assembler.h"

#ifdef ENABLE_CUDA_ASSEMBLER

#include <cuda_runtime.h>

#include <c10/cuda/CUDAStream.h>

namespace mdb::gnn {

// ============================================================================
// CUDA Kernel
// ============================================================================
//
// Each block handles one output node (= one entry in the address table).
// Threads within the block cooperatively copy the feature_dim floats.
// Source is selected per-entry: level==0 reads from GPU HBM (~900 GB/s),
// level==1 reads from CPU pinned memory via UVA (~12 GB/s).
//

__global__ void assemble_kernel(
    float* __restrict__ output,           // [N, D] on GPU
    const float* __restrict__ gpu_data,   // [K1, D] on GPU (HBM)
    const float* __restrict__ cpu_data,   // CPU pinned (UVA)
    const int32_t* positions,             // [total_entries] output row indices
    const int32_t* source_indices,        // [total_entries] row index in source
    const uint8_t* source_level,          // 0=GPU, 1=CPU
    int64_t D,
    int64_t total_entries
) {
    int entry = blockIdx.x;
    if (entry >= total_entries) return;

    int out_row = positions[entry];
    int src_row = source_indices[entry];
    uint8_t level = source_level[entry];

    const float* src = (level == 0) ? gpu_data : cpu_data;

    // Cooperative copy across threads in block
    for (int feat = threadIdx.x; feat < D; feat += blockDim.x) {
        output[out_row * D + feat] = src[src_row * D + feat];
    }
}

// ============================================================================
// CUDA Assembly Path
// ============================================================================

torch::Tensor FeatureAssembler::assemble_cuda(
    int64_t total_nodes,
    const torch::Tensor& gpu_features,
    const std::vector<uint32_t>& gpu_positions,
    const float* cpu_data,
    int64_t cpu_count,
    const std::vector<uint32_t>& cpu_positions)
{
    int64_t D = feature_dim_;
    auto output = torch::zeros({total_nodes, D},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    if (gpu_positions.empty() && cpu_positions.empty()) {
        return output;
    }

    // Build host-side address tables
    int64_t total_entries = static_cast<int64_t>(gpu_positions.size())
                          + static_cast<int64_t>(cpu_positions.size());

    std::vector<uint32_t> h_positions(total_entries);
    std::vector<uint32_t> h_src_indices(total_entries);
    std::vector<uint8_t>  h_levels(total_entries);

    int64_t idx = 0;
    for (size_t i = 0; i < gpu_positions.size(); ++i, ++idx) {
        h_positions[idx]    = gpu_positions[i];
        h_src_indices[idx]  = static_cast<uint32_t>(i);
        h_levels[idx]       = 0;  // GPU source
    }
    for (size_t i = 0; i < cpu_positions.size(); ++i, ++idx) {
        h_positions[idx]    = cpu_positions[i];
        h_src_indices[idx]  = static_cast<uint32_t>(i);
        h_levels[idx]       = 1;  // CPU source (UVA)
    }

    // Upload address tables to GPU via LibTorch (handles allocation + H2D copy)
    auto d_positions = torch::tensor(
        std::vector<int32_t>(h_positions.begin(), h_positions.end()),
        torch::kInt32).to(torch::kCUDA);
    auto d_src_indices = torch::tensor(
        std::vector<int32_t>(h_src_indices.begin(), h_src_indices.end()),
        torch::kInt32).to(torch::kCUDA);

    // Upload level flags
    auto d_levels_cpu = torch::from_blob(
        h_levels.data(), {total_entries}, torch::kUInt8).clone();
    auto d_levels = d_levels_cpu.to(torch::kCUDA);

    // Kernel launch configuration
    int threads_per_block = std::min(static_cast<int64_t>(256), D);
    int blocks = static_cast<int>(total_entries);

    const float* gpu_data_ptr = (gpu_features.defined() && gpu_features.numel() > 0)
        ? gpu_features.data_ptr<float>() : nullptr;

    // Spec C3 stage 3 (2026-05-08): use the current CUDA stream rather than
    // the default stream so callers can drive this kernel onto a non-default
    // stream via c10::cuda::CUDAStreamGuard. With a guard set by the worker
    // thread, the kernel ends up on the worker's stream and can run
    // concurrently with model.forward+backward on a different stream.
    //
    // If no guard is set, getCurrentCUDAStream() returns the default stream,
    // preserving pre-2026-05-08 behaviour byte-for-byte.
    auto current_stream = c10::cuda::getCurrentCUDAStream();

    assemble_kernel<<<blocks, threads_per_block, 0, current_stream.stream()>>>(
        output.data_ptr<float>(),
        gpu_data_ptr,
        cpu_data,
        d_positions.data_ptr<int32_t>(),
        d_src_indices.data_ptr<int32_t>(),
        d_levels.data_ptr<uint8_t>(),
        D,
        total_entries
    );

    // Synchronize the current stream only — keeps the host-blocking semantics
    // of the legacy implementation while letting the stream itself be any
    // pool stream. Module 4 will add an async variant that returns without
    // syncing and lets the caller handle ordering via at::cuda::CUDAEvent.
    cudaStreamSynchronize(current_stream.stream());

    return output;
}

// ============================================================================
// Fused L2-direct kernel (3 sources)
// ============================================================================
//
// Like assemble_kernel but selects among THREE sources per entry:
//   level==0 -> GPU HBM (L1 gathered features, ~900 GB/s)
//   level==1 -> L2 pinned CPU cache base via UVA (read the row in place, no
//               host pre-copy into the combined buffer)
//   level==2 -> CPU pinned combined buffer via UVA (L3 + L4 rows)
// The output placement is identical to the 2-source kernel; only L2 rows are
// now read straight from the cache instead of from a pre-copied cpu_data slot.
//
__global__ void assemble_kernel_l2direct(
    float* __restrict__ output,           // [N, D] on GPU
    const float* __restrict__ gpu_data,   // [K1, D] on GPU (HBM)
    const float* __restrict__ l2_data,    // L2 pinned cache base (UVA)
    const float* __restrict__ cpu_data,   // L3+L4 combined pinned (UVA)
    const int32_t* positions,             // [total_entries] output row indices
    const int32_t* source_indices,        // [total_entries] row index in source
    const uint8_t* source_level,          // 0=GPU, 1=L2 cache, 2=CPU combined
    int64_t D,
    int64_t total_entries
) {
    int entry = blockIdx.x;
    if (entry >= total_entries) return;

    int out_row = positions[entry];
    int src_row = source_indices[entry];
    uint8_t level = source_level[entry];

    const float* src = (level == 0) ? gpu_data
                     : (level == 1) ? l2_data
                                    : cpu_data;

    for (int feat = threadIdx.x; feat < D; feat += blockDim.x) {
        output[out_row * D + feat] = src[src_row * D + feat];
    }
}

torch::Tensor FeatureAssembler::assemble_l2direct_cuda(
    int64_t total_nodes,
    const torch::Tensor& gpu_features,
    const std::vector<uint32_t>& gpu_positions,
    const float* l2_base,
    const std::vector<uint32_t>& l2_indices,
    const std::vector<uint32_t>& l2_positions,
    const float* cpu_data,
    int64_t cpu_count,
    const std::vector<uint32_t>& cpu_positions)
{
    int64_t D = feature_dim_;
    auto output = torch::zeros({total_nodes, D},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    int64_t total_entries = static_cast<int64_t>(gpu_positions.size())
                          + static_cast<int64_t>(l2_positions.size())
                          + static_cast<int64_t>(cpu_positions.size());
    if (total_entries == 0) {
        return output;
    }

    // Host-side address tables: L1 (level 0), L2 (level 1), L3+L4 (level 2).
    std::vector<int32_t> h_positions(total_entries);
    std::vector<int32_t> h_src_indices(total_entries);
    std::vector<uint8_t> h_levels(total_entries);

    int64_t idx = 0;
    for (size_t i = 0; i < gpu_positions.size(); ++i, ++idx) {
        h_positions[idx]   = static_cast<int32_t>(gpu_positions[i]);
        h_src_indices[idx] = static_cast<int32_t>(i);   // row in gpu_features
        h_levels[idx]      = 0;
    }
    for (size_t i = 0; i < l2_positions.size(); ++i, ++idx) {
        h_positions[idx]   = static_cast<int32_t>(l2_positions[i]);
        h_src_indices[idx] = static_cast<int32_t>(l2_indices[i]);  // cache slot
        h_levels[idx]      = 1;
    }
    for (size_t i = 0; i < cpu_positions.size(); ++i, ++idx) {
        h_positions[idx]   = static_cast<int32_t>(cpu_positions[i]);
        h_src_indices[idx] = static_cast<int32_t>(i);   // row in cpu_data
        h_levels[idx]      = 2;
    }

    auto d_positions = torch::tensor(h_positions, torch::kInt32).to(torch::kCUDA);
    auto d_src_indices = torch::tensor(h_src_indices, torch::kInt32).to(torch::kCUDA);
    auto d_levels = torch::from_blob(
        h_levels.data(), {total_entries}, torch::kUInt8).clone().to(torch::kCUDA);

    int threads_per_block = std::min(static_cast<int64_t>(256), D);
    int blocks = static_cast<int>(total_entries);

    const float* gpu_data_ptr = (gpu_features.defined() && gpu_features.numel() > 0)
        ? gpu_features.data_ptr<float>() : nullptr;

    auto current_stream = c10::cuda::getCurrentCUDAStream();
    assemble_kernel_l2direct<<<blocks, threads_per_block, 0, current_stream.stream()>>>(
        output.data_ptr<float>(),
        gpu_data_ptr,
        l2_base,
        cpu_data,
        d_positions.data_ptr<int32_t>(),
        d_src_indices.data_ptr<int32_t>(),
        d_levels.data_ptr<uint8_t>(),
        D,
        total_entries
    );
    cudaStreamSynchronize(current_stream.stream());

    return output;
}

} // namespace mdb::gnn

#endif // ENABLE_CUDA_ASSEMBLER
