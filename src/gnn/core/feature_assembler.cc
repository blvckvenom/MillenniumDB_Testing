#include "gnn/core/feature_assembler.h"

#include <algorithm>
#include <stdexcept>

namespace mdb::gnn {

// ============================================================================
// Constructor
// ============================================================================

FeatureAssembler::FeatureAssembler(int64_t feature_dim)
    : feature_dim_(feature_dim) {
    if (feature_dim_ <= 0) {
        throw std::invalid_argument(
            "FeatureAssembler: feature_dim must be > 0, got " +
            std::to_string(feature_dim_));
    }
}

// ============================================================================
// Fallback Implementation (LibTorch index_copy_)
// ============================================================================

torch::Tensor FeatureAssembler::assemble_fallback(
    int64_t total_nodes,
    const torch::Tensor& gpu_features,
    const std::vector<uint32_t>& gpu_positions,
    const float* cpu_data,
    int64_t cpu_count,
    const std::vector<uint32_t>& cpu_positions)
{
    // Determine output device: if gpu_features lives on CUDA, output there
    auto device = (gpu_features.defined() && gpu_features.is_cuda())
        ? torch::kCUDA : torch::kCPU;
    auto output = torch::zeros({total_nodes, feature_dim_},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    if (total_nodes == 0) {
        return output;
    }

    // Place GPU features via index_copy_
    if (!gpu_positions.empty() && gpu_features.defined() && gpu_features.numel() > 0) {
        auto pos_tensor = torch::tensor(
            std::vector<int64_t>(gpu_positions.begin(), gpu_positions.end()),
            torch::kInt64).to(device);
        auto idx_tensor = torch::arange(
            static_cast<int64_t>(gpu_positions.size()),
            torch::kInt64).to(device);
        output.index_copy_(0, pos_tensor, gpu_features.index_select(0, idx_tensor));
    }

    // Place CPU features via index_copy_
    if (!cpu_positions.empty() && cpu_data != nullptr && cpu_count > 0) {
        auto cpu_tensor = torch::from_blob(
            const_cast<float*>(cpu_data),
            {cpu_count, feature_dim_},
            torch::kFloat32).clone();
        auto pos_tensor = torch::tensor(
            std::vector<int64_t>(cpu_positions.begin(), cpu_positions.end()),
            torch::kInt64);
        if (device == torch::kCUDA) {
            pos_tensor = pos_tensor.to(torch::kCUDA);
            cpu_tensor = cpu_tensor.to(torch::kCUDA);
        }
        auto idx_tensor = torch::arange(cpu_count, torch::kInt64).to(device);
        output.index_copy_(0, pos_tensor, cpu_tensor.index_select(0, idx_tensor));
    }

    return output;
}

// ============================================================================
// Simplified Interface
// ============================================================================

torch::Tensor FeatureAssembler::assemble_simple(
    int64_t total_nodes,
    const torch::Tensor& l1_features,
    const std::vector<uint32_t>& l1_positions,
    const torch::Tensor& cpu_features,
    const std::vector<uint32_t>& cpu_positions)
{
    const float* cpu_ptr = nullptr;
    int64_t cpu_count = 0;

    if (cpu_features.defined() && cpu_features.numel() > 0) {
        // Ensure contiguous CPU tensor for raw pointer access
        auto contig = cpu_features.contiguous().to(torch::kCPU);
        cpu_ptr = contig.data_ptr<float>();
        cpu_count = contig.size(0);
        return assemble(total_nodes, l1_features, l1_positions,
                        cpu_ptr, cpu_count, cpu_positions);
    }

    return assemble(total_nodes, l1_features, l1_positions,
                    nullptr, 0, cpu_positions);
}

// ============================================================================
// Main Dispatch
// ============================================================================

torch::Tensor FeatureAssembler::assemble(
    int64_t total_nodes,
    const torch::Tensor& gpu_features,
    const std::vector<uint32_t>& gpu_positions,
    const float* cpu_data,
    int64_t cpu_count,
    const std::vector<uint32_t>& cpu_positions)
{
#ifdef ENABLE_CUDA_ASSEMBLER
    if (torch::cuda::is_available()) {
        return assemble_cuda(total_nodes, gpu_features, gpu_positions,
                             cpu_data, cpu_count, cpu_positions);
    }
#endif
    return assemble_fallback(total_nodes, gpu_features, gpu_positions,
                             cpu_data, cpu_count, cpu_positions);
}

} // namespace mdb::gnn
