#include "memory_pool.h"

#include <algorithm>
#include <cstring>

#include "cuda_context.h"

#ifdef GNN_CUDA_ENABLED
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace mdb::gnn {

// ============================================================================
// BatchMemoryEstimate Implementation
// ============================================================================

BatchMemoryEstimate BatchMemoryEstimate::compute(
    int64_t num_nodes,
    int64_t num_edges,
    int64_t feature_dim,
    int64_t hidden_dim,
    int num_layers,
    double safety_margin
) {
    BatchMemoryEstimate est;

    // Input features: [num_nodes, feature_dim] as float32
    est.feature_tensor = num_nodes * feature_dim * sizeof(float);

    // Adjacency in COO format:
    // - indices: [2, num_edges] as int64
    // - values: [num_edges] as float32
    est.adjacency_indices = 2 * num_edges * sizeof(int64_t);
    est.adjacency_values = num_edges * sizeof(float);

    // Activations per layer:
    // - Layer 0 (input): num_nodes * feature_dim
    // - Layers 1..K: num_nodes * hidden_dim each
    est.activations = num_nodes * feature_dim * sizeof(float);
    if (num_layers > 1) {
        est.activations += (num_layers - 1) * num_nodes * hidden_dim * sizeof(float);
    }

    // Gradients are roughly same size as activations during backward pass
    est.gradients = est.activations;

    // Total with safety margin
    size_t raw_total = est.feature_tensor + est.adjacency_indices +
                       est.adjacency_values + est.activations + est.gradients;
    est.total = static_cast<size_t>(raw_total * safety_margin);

    return est;
}

// ============================================================================
// GpuMemoryPool Implementation
// ============================================================================

GpuMemoryPool::GpuMemoryPool(const Config& config)
    : config_(config),
      pressure_callback_(nullptr) {
    // Initialize stats
    stats_.total_size = 0;
    stats_.allocated_size = 0;
    stats_.free_size = 0;
    stats_.num_allocations = 0;
    stats_.num_deallocations = 0;
    stats_.num_resets = 0;
    stats_.peak_usage = 0;
    stats_.fragmentation_bytes = 0;

    // Only create initial arena if CUDA is available
    if (CudaContext::instance().is_cuda_available() && config_.initial_size > 0) {
        try {
            arenas_.push_back(create_arena(config_.initial_size));
            stats_.total_size = config_.initial_size;
            stats_.free_size = config_.initial_size;
        } catch (const std::exception& e) {
            // Initial allocation failed - pool will try to allocate on demand
            arenas_.clear();
        }
    }
}

GpuMemoryPool::~GpuMemoryPool() {
    for (auto& arena : arenas_) {
        free_arena(arena);
    }
    arenas_.clear();
}

GpuMemoryPool::GpuMemoryPool(GpuMemoryPool&& other) noexcept
    : config_(std::move(other.config_)),
      arenas_(std::move(other.arenas_)),
      stats_(other.stats_),
      pressure_callback_(std::move(other.pressure_callback_)) {
    other.arenas_.clear();
    other.stats_ = MemoryPoolStats{};
}

GpuMemoryPool& GpuMemoryPool::operator=(GpuMemoryPool&& other) noexcept {
    if (this != &other) {
        // Free our arenas
        for (auto& arena : arenas_) {
            free_arena(arena);
        }

        config_ = std::move(other.config_);
        arenas_ = std::move(other.arenas_);
        stats_ = other.stats_;
        pressure_callback_ = std::move(other.pressure_callback_);

        other.arenas_.clear();
        other.stats_ = MemoryPoolStats{};
    }
    return *this;
}

GpuMemoryPool::Arena GpuMemoryPool::create_arena(size_t size) {
    Arena arena;
    arena.size = size;
    arena.offset = 0;
    arena.base_ptr = nullptr;

#ifdef GNN_CUDA_ENABLED
    // Only use CUDA allocation if CUDA is actually available at runtime
    if (CudaContext::instance().is_cuda_available()) {
        cudaError_t err = cudaMalloc(&arena.base_ptr, size);
        if (err != cudaSuccess) {
            throw CudaException(static_cast<int>(err), "cudaMalloc for arena");
        }
        return arena;
    }
#endif

    // CPU fallback when CUDA is not available or not compiled in
    arena.base_ptr = std::malloc(size);
    if (arena.base_ptr == nullptr) {
        throw std::bad_alloc();
    }

    return arena;
}

void GpuMemoryPool::free_arena(Arena& arena) {
    if (arena.base_ptr != nullptr) {
#ifdef GNN_CUDA_ENABLED
        // Only use cudaFree if CUDA is available (arena was allocated with cudaMalloc)
        if (CudaContext::instance().is_cuda_available()) {
            cudaFree(arena.base_ptr);
        } else {
            std::free(arena.base_ptr);
        }
#else
        std::free(arena.base_ptr);
#endif
        arena.base_ptr = nullptr;
        arena.size = 0;
        arena.offset = 0;
    }
}

size_t GpuMemoryPool::align_size(size_t size) const {
    return (size + config_.alignment - 1) & ~(config_.alignment - 1);
}

void* GpuMemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Align the size
    size_t aligned_size = align_size(size);

    // Try to allocate from existing arenas
    for (auto& arena : arenas_) {
        if (arena.offset + aligned_size <= arena.size) {
            void* ptr = static_cast<char*>(arena.base_ptr) + arena.offset;
            arena.offset += aligned_size;

            stats_.allocated_size += aligned_size;
            stats_.free_size -= aligned_size;
            stats_.num_allocations++;
            stats_.peak_usage = std::max(stats_.peak_usage, stats_.allocated_size);

            return ptr;
        }
    }

    // Need new arena
    if (!config_.allow_growth) {
        handle_pressure(aligned_size);
        throw std::bad_alloc();
    }

    // Check max size
    if (config_.max_size > 0 && stats_.total_size >= config_.max_size) {
        handle_pressure(aligned_size);
        throw std::bad_alloc();
    }

    // Calculate new arena size
    size_t new_arena_size = aligned_size;  // At minimum, fit the requested size
    if (!arenas_.empty()) {
        // Grow by growth_factor of current total size
        size_t growth_size = static_cast<size_t>(stats_.total_size * config_.growth_factor);
        new_arena_size = std::max(new_arena_size, growth_size);
    } else {
        // First arena after initial failure
        new_arena_size = std::max(new_arena_size, config_.initial_size);
    }

    // Respect max_size
    if (config_.max_size > 0) {
        size_t remaining = config_.max_size - stats_.total_size;
        new_arena_size = std::min(new_arena_size, remaining);
    }

    try {
        arenas_.push_back(create_arena(new_arena_size));
        stats_.total_size += new_arena_size;
        stats_.free_size += new_arena_size;
    } catch (...) {
        handle_pressure(aligned_size);
        throw;
    }

    // Retry allocation with new arena
    Arena& new_arena = arenas_.back();
    void* ptr = static_cast<char*>(new_arena.base_ptr) + new_arena.offset;
    new_arena.offset += aligned_size;

    stats_.allocated_size += aligned_size;
    stats_.free_size -= aligned_size;
    stats_.num_allocations++;
    stats_.peak_usage = std::max(stats_.peak_usage, stats_.allocated_size);

    return ptr;
}

void GpuMemoryPool::deallocate(void* ptr) {
    // In arena allocation, deallocation is a no-op
    // Memory is reclaimed only via reset()
    (void)ptr;

    std::lock_guard<std::mutex> lock(mutex_);
    stats_.num_deallocations++;
}

torch::Tensor GpuMemoryPool::allocate_tensor(
    const std::vector<int64_t>& shape,
    torch::Dtype dtype
) {
    // Calculate size needed
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    size_t size = numel * torch::elementSize(dtype);

    // Allocate from pool
    void* data_ptr = allocate(size);

    // Create tensor that uses this memory
    // We use a no-op deleter because the pool manages the memory
    auto deleter = [](void*) {};

    // Create tensor options - use CUDA only if available
    auto options = torch::TensorOptions()
        .dtype(dtype);

#ifdef GNN_CUDA_ENABLED
    if (CudaContext::instance().is_cuda_available()) {
        options = options.device(torch::kCUDA, CudaContext::instance().current_device());
    } else {
        options = options.device(torch::kCPU);
    }
#else
    options = options.device(torch::kCPU);
#endif

    return torch::from_blob(data_ptr, shape, deleter, options);
}

torch::Tensor GpuMemoryPool::allocate_tensor_zeros(
    const std::vector<int64_t>& shape,
    torch::Dtype dtype
) {
    auto tensor = allocate_tensor(shape, dtype);
    tensor.zero_();
    return tensor;
}

void GpuMemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Reset all arena offsets to 0
    for (auto& arena : arenas_) {
        arena.offset = 0;
    }

    stats_.allocated_size = 0;
    stats_.free_size = stats_.total_size;
    stats_.num_resets++;
}

void GpuMemoryPool::reserve(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if we already have enough capacity
    size_t current_capacity = 0;
    for (const auto& arena : arenas_) {
        current_capacity += arena.size;
    }

    if (current_capacity >= size) {
        return;
    }

    // Need to allocate more
    size_t additional_needed = size - current_capacity;

    try {
        arenas_.push_back(create_arena(additional_needed));
        stats_.total_size += additional_needed;
        stats_.free_size += additional_needed;
    } catch (...) {
        handle_pressure(additional_needed);
        throw;
    }
}

void GpuMemoryPool::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& arena : arenas_) {
        free_arena(arena);
    }
    arenas_.clear();

    stats_.total_size = 0;
    stats_.allocated_size = 0;
    stats_.free_size = 0;
}

void GpuMemoryPool::release_torch_cache() {
#ifdef GNN_CUDA_ENABLED
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
}

MemoryPoolStats GpuMemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

size_t GpuMemoryPool::capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.total_size;
}

size_t GpuMemoryPool::allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.allocated_size;
}

size_t GpuMemoryPool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.free_size;
}

torch::Tensor GpuMemoryPool::allocate_persistent(
    const std::vector<int64_t>& shape,
    torch::Dtype dtype
) {
    auto options = torch::TensorOptions().dtype(dtype);

#ifdef GNN_CUDA_ENABLED
    if (CudaContext::instance().is_cuda_available()) {
        options = options.device(torch::kCUDA, CudaContext::instance().current_device());
    }
#endif

    return torch::empty(shape, options);
}

void GpuMemoryPool::set_pressure_callback(PressureCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    pressure_callback_ = std::move(callback);
}

bool GpuMemoryPool::can_allocate(size_t size) const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t aligned_size = (size + config_.alignment - 1) & ~(config_.alignment - 1);

    // Check if any arena can fit this allocation
    for (const auto& arena : arenas_) {
        if (arena.offset + aligned_size <= arena.size) {
            return true;
        }
    }

    // Would need growth
    if (!config_.allow_growth) {
        return false;
    }

    if (config_.max_size > 0 && stats_.total_size + aligned_size > config_.max_size) {
        return false;
    }

    // Check if we have enough free memory for growth
#ifdef GNN_CUDA_ENABLED
    if (CudaContext::instance().is_cuda_available()) {
        size_t free_mem = CudaContext::instance().free_memory();
        return free_mem >= aligned_size;
    }
#endif

    // CPU fallback always "has memory" (relies on OS virtual memory)
    return true;
}

void GpuMemoryPool::handle_pressure(size_t requested_size) {
    if (pressure_callback_) {
        MemoryPressureAction action;

        // Determine appropriate action based on size ratio
        if (requested_size > stats_.total_size) {
            action = MemoryPressureAction::REDUCE_BATCH_SIZE;
        } else if (stats_.allocated_size > stats_.total_size * 0.9) {
            action = MemoryPressureAction::OFFLOAD_TO_CPU;
        } else {
            action = MemoryPressureAction::FAIL;
        }

        pressure_callback_(action, requested_size, stats_.free_size);
    }
}

} // namespace mdb::gnn
