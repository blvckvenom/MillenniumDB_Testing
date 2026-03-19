#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

namespace mdb::gnn {

/**
 * @brief Assembles features from multiple storage levels into one GPU tensor.
 *
 * When ENABLE_CUDA_ASSEMBLER is defined, uses a custom CUDA kernel where
 * each GPU thread reads from the appropriate source (GPU HBM or CPU pinned
 * via UVA) in a single pass. Otherwise falls back to LibTorch index_copy_.
 *
 * DiskGNN equivalent: the "Feature Assembler" stage in the training pipeline.
 */
class FeatureAssembler {
public:
    explicit FeatureAssembler(int64_t feature_dim);

    /**
     * Assemble complete feature tensor from partitioned sources.
     *
     * @param total_nodes Total nodes in output (= batch size)
     * @param gpu_features [K1, D] tensor on CUDA (L1 cache hits) -- may be empty
     * @param gpu_positions positions in output for GPU features
     * @param cpu_data pointer to CPU pinned memory (L2 + L3 + L4 features, contiguous)
     * @param cpu_count number of rows in cpu_data
     * @param cpu_positions positions in output for CPU features
     * @return [total_nodes, D] tensor on CUDA (or CPU if no GPU)
     */
    torch::Tensor assemble(
        int64_t total_nodes,
        const torch::Tensor& gpu_features,
        const std::vector<uint32_t>& gpu_positions,
        const float* cpu_data,
        int64_t cpu_count,
        const std::vector<uint32_t>& cpu_positions
    );

    /// Simplified interface: assemble from L1 tensor + CPU buffer.
    /// Handles the common case where L2/L3/L4 features are already combined.
    torch::Tensor assemble_simple(
        int64_t total_nodes,
        const torch::Tensor& l1_features,
        const std::vector<uint32_t>& l1_positions,
        const torch::Tensor& cpu_features,
        const std::vector<uint32_t>& cpu_positions
    );

    /// Fallback path using LibTorch index_copy_ (always available, even
    /// when ENABLE_CUDA_ASSEMBLER is defined -- exposed for testing).
    torch::Tensor assemble_fallback(
        int64_t total_nodes,
        const torch::Tensor& gpu_features,
        const std::vector<uint32_t>& gpu_positions,
        const float* cpu_data,
        int64_t cpu_count,
        const std::vector<uint32_t>& cpu_positions
    );

private:
    int64_t feature_dim_;

#ifdef ENABLE_CUDA_ASSEMBLER
    torch::Tensor assemble_cuda(
        int64_t total_nodes,
        const torch::Tensor& gpu_features,
        const std::vector<uint32_t>& gpu_positions,
        const float* cpu_data,
        int64_t cpu_count,
        const std::vector<uint32_t>& cpu_positions
    );
#endif
};

} // namespace mdb::gnn
