#include "gnn/core/feature_assembler.h"

#include <algorithm>
#include <stdexcept>

#ifdef ENABLE_CUDA_ASSEMBLER
#include <cuda_runtime.h>
#endif

namespace mdb::gnn {

#ifdef ENABLE_CUDA_ASSEMBLER
namespace {

// The assemble_cuda kernel dereferences cpu_data from device threads via
// UVA, which is only guaranteed-legal for page-locked (pinned), managed,
// or device memory. A pageable heap pointer (e.g. a caller whose pinned
// allocation failed) must take the index_copy_ fallback instead.
bool is_device_accessible_(const float* ptr) {
    cudaPointerAttributes attrs {};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        // Pre-CUDA-11 runtimes report unregistered host pointers as
        // cudaErrorInvalidValue; clear the sticky error and treat the
        // pointer as pageable.
        cudaGetLastError();
        return false;
    }
    return attrs.type == cudaMemoryTypeHost
        || attrs.type == cudaMemoryTypeDevice
        || attrs.type == cudaMemoryTypeManaged;
}

} // namespace
#endif

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
    // Bounds-check positions against total_nodes (I6)
    for (auto pos : gpu_positions) {
        if (pos >= static_cast<uint32_t>(total_nodes)) {
            throw std::out_of_range(
                "FeatureAssembler::assemble_fallback: gpu_position " +
                std::to_string(pos) + " >= total_nodes " + std::to_string(total_nodes));
        }
    }
    for (auto pos : cpu_positions) {
        if (pos >= static_cast<uint32_t>(total_nodes)) {
            throw std::out_of_range(
                "FeatureAssembler::assemble_fallback: cpu_position " +
                std::to_string(pos) + " >= total_nodes " + std::to_string(total_nodes));
        }
    }

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
    torch::Tensor contig_holder;  // must outlive the assemble() call

    if (cpu_features.defined() && cpu_features.numel() > 0) {
        // Ensure contiguous CPU tensor for raw pointer access
        contig_holder = cpu_features.contiguous().to(torch::kCPU);
#ifdef ENABLE_CUDA_ASSEMBLER
        // assemble() dispatches to the CUDA kernel when l1_features lives
        // on CUDA; the kernel reads cpu_ptr from device threads via UVA,
        // which requires page-locked memory.
        if (torch::cuda::is_available() &&
            l1_features.defined() && l1_features.is_cuda() &&
            !contig_holder.is_pinned()) {
            contig_holder = contig_holder.pin_memory();
        }
#endif
        cpu_ptr = contig_holder.data_ptr<float>();
        cpu_count = contig_holder.size(0);
    }

    return assemble(total_nodes, l1_features, l1_positions,
                    cpu_ptr, cpu_count, cpu_positions);
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
    // Bounds-check positions against total_nodes (I6)
    for (auto pos : gpu_positions) {
        if (pos >= static_cast<uint32_t>(total_nodes)) {
            throw std::out_of_range(
                "FeatureAssembler::assemble: gpu_position " +
                std::to_string(pos) + " >= total_nodes " + std::to_string(total_nodes));
        }
    }
    for (auto pos : cpu_positions) {
        if (pos >= static_cast<uint32_t>(total_nodes)) {
            throw std::out_of_range(
                "FeatureAssembler::assemble: cpu_position " +
                std::to_string(pos) + " >= total_nodes " + std::to_string(total_nodes));
        }
    }

#ifdef ENABLE_CUDA_ASSEMBLER
    // Only use the CUDA kernel when gpu_features is actually on CUDA.
    // Passing a CPU tensor would give the kernel a host pointer that
    // GPU threads cannot dereference (illegal memory access / SIGSEGV).
    // cpu_data has the same constraint: the kernel reads it from device
    // threads via UVA, which is only legal for pinned/managed memory, so
    // pageable pointers are routed to the fallback as well.
    if (torch::cuda::is_available() &&
        gpu_features.defined() && gpu_features.is_cuda() &&
        (cpu_positions.empty() || cpu_data == nullptr ||
         is_device_accessible_(cpu_data))) {
        return assemble_cuda(total_nodes, gpu_features, gpu_positions,
                             cpu_data, cpu_count, cpu_positions);
    }
#endif
    return assemble_fallback(total_nodes, gpu_features, gpu_positions,
                             cpu_data, cpu_count, cpu_positions);
}

// ============================================================================
// Fused L2-direct dispatch (CUDA-only)
// ============================================================================

torch::Tensor FeatureAssembler::assemble_l2direct(
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
    if (l2_indices.size() != l2_positions.size()) {
        throw std::invalid_argument(
            "FeatureAssembler::assemble_l2direct: l2_indices/l2_positions size mismatch");
    }
    // Bounds-check every output position against total_nodes (symmetric with assemble()).
    for (auto pos : gpu_positions) {
        if (pos >= static_cast<uint32_t>(total_nodes)) {
            throw std::out_of_range("FeatureAssembler::assemble_l2direct: gpu_position "
                + std::to_string(pos) + " >= total_nodes " + std::to_string(total_nodes));
        }
    }
    for (auto pos : l2_positions) {
        if (pos >= static_cast<uint32_t>(total_nodes)) {
            throw std::out_of_range("FeatureAssembler::assemble_l2direct: l2_position "
                + std::to_string(pos) + " >= total_nodes " + std::to_string(total_nodes));
        }
    }
    for (auto pos : cpu_positions) {
        if (pos >= static_cast<uint32_t>(total_nodes)) {
            throw std::out_of_range("FeatureAssembler::assemble_l2direct: cpu_position "
                + std::to_string(pos) + " >= total_nodes " + std::to_string(total_nodes));
        }
    }

#ifdef ENABLE_CUDA_ASSEMBLER
    // The kernel reads l2_base and cpu_data from device threads via UVA, which is
    // only legal for pinned/managed/device memory. The caller gates on a pinned
    // CPU cache, but re-verify here (cheap) so a pageable pointer can never reach
    // the kernel — throw so the four-level dispatcher falls back to legacy.
    if (torch::cuda::is_available() &&
        gpu_features.defined() && gpu_features.is_cuda() &&
        l2_base != nullptr && is_device_accessible_(l2_base) &&
        (cpu_positions.empty() || cpu_data == nullptr ||
         is_device_accessible_(cpu_data))) {
        return assemble_l2direct_cuda(total_nodes, gpu_features, gpu_positions,
                                      l2_base, l2_indices, l2_positions,
                                      cpu_data, cpu_count, cpu_positions);
    }
#endif
    throw std::runtime_error(
        "FeatureAssembler::assemble_l2direct: CUDA assembler path unavailable "
        "(requires ENABLE_CUDA_ASSEMBLER + CUDA gpu_features + pinned/UVA L2 base)");
}

} // namespace mdb::gnn
