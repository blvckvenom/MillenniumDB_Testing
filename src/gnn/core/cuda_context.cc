#include "cuda_context.h"

#include <algorithm>
#include <sstream>

#ifdef GNN_CUDA_ENABLED
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace mdb::gnn {

// ============================================================================
// CudaException Implementation
// ============================================================================

CudaException::CudaException(int error_code, const char* operation)
    : std::runtime_error(
          [error_code, operation]() {
              std::ostringstream oss;
              oss << "CUDA error in " << operation << ": ";
#ifdef GNN_CUDA_ENABLED
              oss << cudaGetErrorString(static_cast<cudaError_t>(error_code));
#else
              oss << "error code " << error_code;
#endif
              return oss.str();
          }()
      ),
      error_code_(error_code),
      operation_(operation) {}

// ============================================================================
// CudaContext Implementation
// ============================================================================

CudaContext& CudaContext::instance() {
    static CudaContext instance;
    return instance;
}

CudaContext::CudaContext()
    : cuda_available_(false),
      device_count_(0),
      device_id_(0) {
    initialize();
}

CudaContext::~CudaContext() {
#ifdef GNN_CUDA_ENABLED
    // Destroy managed streams
    for (auto stream : managed_streams_) {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }

    // Destroy purpose-specific streams
    for (int i = 0; i < 5; ++i) {
        if (streams_[i] != nullptr) {
            cudaStreamDestroy(streams_[i]);
        }
    }
#endif
}

void CudaContext::initialize() {
#ifdef GNN_CUDA_ENABLED
    // Check CUDA availability via LibTorch
    cuda_available_ = torch::cuda::is_available();

    if (!cuda_available_) {
        device_count_ = 0;
        return;
    }

    device_count_ = torch::cuda::device_count();

    if (device_count_ == 0) {
        cuda_available_ = false;
        return;
    }

    // Initialize streams array
    for (int i = 0; i < 5; ++i) {
        streams_[i] = nullptr;
    }

    // Select the best device by default
    select_device(DeviceSelectionStrategy::MOST_MEMORY);

#else
    cuda_available_ = false;
    device_count_ = 0;
#endif
}

// ============================================================================
// Device Queries
// ============================================================================

bool CudaContext::is_cuda_available() const {
    return cuda_available_;
}

int CudaContext::device_count() const {
    return device_count_;
}

std::vector<DeviceInfo> CudaContext::enumerate_devices() const {
    std::vector<DeviceInfo> devices;

#ifdef GNN_CUDA_ENABLED
    if (!cuda_available_) {
        return devices;
    }

    for (int i = 0; i < device_count_; ++i) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, i);

        DeviceInfo info;
        info.device_id = i;
        info.is_available = (err == cudaSuccess);

        if (info.is_available) {
            info.name = prop.name;
            info.total_memory = prop.totalGlobalMem;
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            info.multiprocessor_count = prop.multiProcessorCount;

            // Query free memory
            size_t free_mem, total_mem;
            cudaSetDevice(i);
            cudaMemGetInfo(&free_mem, &total_mem);
            info.free_memory = free_mem;
        } else {
            info.name = "Unknown";
            info.total_memory = 0;
            info.free_memory = 0;
            info.compute_capability_major = 0;
            info.compute_capability_minor = 0;
            info.multiprocessor_count = 0;
        }

        devices.push_back(info);
    }

    // Restore original device
    cudaSetDevice(device_id_);
#endif

    return devices;
}

// ============================================================================
// Current Device Management
// ============================================================================

int CudaContext::current_device() const {
    return device_id_;
}

void CudaContext::set_device(int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!cuda_available_) {
        return;
    }

    if (device_id < 0 || device_id >= device_count_) {
        throw std::invalid_argument(
            "Invalid device ID: " + std::to_string(device_id) +
            " (available: 0-" + std::to_string(device_count_ - 1) + ")"
        );
    }

    device_id_ = device_id;

#ifdef GNN_CUDA_ENABLED
    // Set for raw CUDA calls (also affects LibTorch)
    CUDA_CHECK(cudaSetDevice(device_id));
#endif
}

DeviceInfo CudaContext::current_device_info() const {
    auto devices = enumerate_devices();
    if (device_id_ < static_cast<int>(devices.size())) {
        return devices[device_id_];
    }

    // Return empty info if no device
    DeviceInfo info;
    info.device_id = -1;
    info.name = "No CUDA Device";
    info.total_memory = 0;
    info.free_memory = 0;
    info.compute_capability_major = 0;
    info.compute_capability_minor = 0;
    info.multiprocessor_count = 0;
    info.is_available = false;
    return info;
}

void CudaContext::select_device(DeviceSelectionStrategy strategy, int specific_id) {
    if (!cuda_available_ || device_count_ == 0) {
        return;
    }

    int selected_device = 0;

    switch (strategy) {
        case DeviceSelectionStrategy::FIRST_AVAILABLE:
            selected_device = 0;
            break;

        case DeviceSelectionStrategy::MOST_MEMORY: {
            auto devices = enumerate_devices();
            size_t max_free = 0;
            for (const auto& dev : devices) {
                if (dev.is_available && dev.free_memory > max_free) {
                    max_free = dev.free_memory;
                    selected_device = dev.device_id;
                }
            }
            break;
        }

        case DeviceSelectionStrategy::LEAST_UTILIZED: {
            auto devices = enumerate_devices();
            double min_util = 1.0;
            for (const auto& dev : devices) {
                if (dev.is_available && dev.total_memory > 0) {
                    double util = 1.0 - static_cast<double>(dev.free_memory) /
                                        static_cast<double>(dev.total_memory);
                    if (util < min_util) {
                        min_util = util;
                        selected_device = dev.device_id;
                    }
                }
            }
            break;
        }

        case DeviceSelectionStrategy::SPECIFIC:
            if (specific_id >= 0 && specific_id < device_count_) {
                selected_device = specific_id;
            }
            break;
    }

    set_device(selected_device);
}

// ============================================================================
// Memory Queries
// ============================================================================

size_t CudaContext::total_memory() const {
    if (!cuda_available_) {
        return 0;
    }

#ifdef GNN_CUDA_ENABLED
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return total_mem;
#else
    return 0;
#endif
}

size_t CudaContext::free_memory() const {
    if (!cuda_available_) {
        return 0;
    }

#ifdef GNN_CUDA_ENABLED
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
#else
    return 0;
#endif
}

MemoryStats CudaContext::get_memory_stats() const {
    MemoryStats stats;
    stats.total = 0;
    stats.free = 0;
    stats.used = 0;
    stats.allocated_by_us = 0;
    stats.cached_by_torch = 0;

    if (!cuda_available_) {
        return stats;
    }

#ifdef GNN_CUDA_ENABLED
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    stats.total = total_mem;
    stats.free = free_mem;
    stats.used = total_mem - free_mem;

    // LibTorch memory stats
    // Note: Detailed stats from CUDACachingAllocator require internal headers
    // For now, we report basic CUDA stats. cached_by_torch and allocated_by_us
    // are left as 0 (would need c10/cuda/CUDACachingAllocator.h which may not be public)
#endif

    return stats;
}

void CudaContext::release_torch_cache() {
#ifdef GNN_CUDA_ENABLED
    if (cuda_available_) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif
}

// ============================================================================
// Stream Management
// ============================================================================

#ifdef GNN_CUDA_ENABLED

cudaStream_t CudaContext::default_stream() const {
    return nullptr;  // CUDA default stream
}

cudaStream_t CudaContext::get_stream(StreamPurpose purpose) {
    std::lock_guard<std::mutex> lock(mutex_);

    int idx = static_cast<int>(purpose);
    if (idx < 0 || idx >= 5) {
        return nullptr;
    }

    // Lazy creation
    if (streams_[idx] == nullptr) {
        CUDA_CHECK(cudaStreamCreate(&streams_[idx]));
    }

    return streams_[idx];
}

void CudaContext::synchronize_stream(cudaStream_t stream) {
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void CudaContext::synchronize_all_streams() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (int i = 0; i < 5; ++i) {
        if (streams_[i] != nullptr) {
            cudaStreamSynchronize(streams_[i]);
        }
    }

    for (auto stream : managed_streams_) {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
    }
}

#endif  // GNN_CUDA_ENABLED

// ============================================================================
// LibTorch Integration
// ============================================================================

torch::Device CudaContext::torch_device() const {
    if (!cuda_available_) {
        return torch::kCPU;
    }
    return torch::Device(torch::kCUDA, device_id_);
}

torch::Device CudaContext::preferred_device() const {
    return torch_device();
}

torch::TensorOptions CudaContext::tensor_options(torch::Dtype dtype) const {
    return torch::TensorOptions()
        .dtype(dtype)
        .device(preferred_device());
}

// ============================================================================
// Error Handling
// ============================================================================

void CudaContext::check_error(const char* operation) {
#ifdef GNN_CUDA_ENABLED
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException(static_cast<int>(error), operation);
    }
#else
    (void)operation;
#endif
}

std::string CudaContext::last_error() const {
#ifdef GNN_CUDA_ENABLED
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return cudaGetErrorString(error);
    }
#endif
    return "";
}

} // namespace mdb::gnn
