#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <torch/torch.h>

#ifdef GNN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace mdb::gnn {

/**
 * Information about a CUDA device.
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;              // Total GPU memory in bytes
    size_t free_memory;               // Currently free memory in bytes
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    bool is_available;
};

/**
 * Memory statistics for GPU.
 */
struct MemoryStats {
    size_t total;            // Total GPU memory
    size_t free;             // Currently free
    size_t used;             // Total - Free
    size_t allocated_by_us;  // Memory allocated by our MemoryPool
    size_t cached_by_torch;  // Memory in LibTorch's caching allocator
};

/**
 * Device selection strategies for multi-GPU systems.
 */
enum class DeviceSelectionStrategy {
    FIRST_AVAILABLE,   // Use device 0
    MOST_MEMORY,       // Use device with most free memory
    LEAST_UTILIZED,    // Use device with lowest current utilization
    SPECIFIC           // Use a specific device ID
};

/**
 * Stream purposes for overlapping operations.
 */
enum class StreamPurpose {
    DEFAULT,          // Default stream
    DATA_TRANSFER,    // H2D and D2H transfers
    COMPUTE,          // Forward/backward pass
    FEATURE_LOAD,     // Feature loading pipeline
    GRAPH_LOAD        // Graph structure loading
};

/**
 * Custom exception for CUDA errors.
 */
class CudaException : public std::runtime_error {
public:
    CudaException(int error_code, const char* operation);

    int error_code() const { return error_code_; }
    const char* operation() const { return operation_; }

private:
    int error_code_;
    const char* operation_;
};

/**
 * Centralized CUDA context management for the GNN training system.
 *
 * Provides:
 *   - Device enumeration and selection
 *   - Memory capacity and usage queries
 *   - CUDA stream management for async operations
 *   - Error handling and recovery
 *   - Graceful CPU fallback when CUDA unavailable
 *
 * Usage:
 *   auto& ctx = CudaContext::instance();
 *   if (ctx.is_cuda_available()) {
 *       auto tensor = torch::randn({100, 100}, ctx.tensor_options());
 *   }
 */
class CudaContext {
public:
    // Singleton access
    static CudaContext& instance();

    // Prevent copying
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    // =========================================================================
    // Device Queries
    // =========================================================================

    /**
     * Check if CUDA is available on this system.
     */
    bool is_cuda_available() const;

    /**
     * Get the number of CUDA devices.
     */
    int device_count() const;

    /**
     * Enumerate all available CUDA devices with their information.
     */
    std::vector<DeviceInfo> enumerate_devices() const;

    // =========================================================================
    // Current Device Management
    // =========================================================================

    /**
     * Get the current device ID.
     */
    int current_device() const;

    /**
     * Set the current CUDA device.
     * Also updates LibTorch's device selection.
     */
    void set_device(int device_id);

    /**
     * Get information about the current device.
     */
    DeviceInfo current_device_info() const;

    /**
     * Select device using a strategy.
     */
    void select_device(DeviceSelectionStrategy strategy, int specific_id = -1);

    // =========================================================================
    // Memory Queries
    // =========================================================================

    /**
     * Get total memory on current device.
     */
    size_t total_memory() const;

    /**
     * Get free memory on current device.
     */
    size_t free_memory() const;

    /**
     * Get detailed memory statistics.
     */
    MemoryStats get_memory_stats() const;

    /**
     * Release LibTorch's cached memory.
     * Useful before querying actual free memory.
     */
    void release_torch_cache();

    // =========================================================================
    // Stream Management
    // =========================================================================

#ifdef GNN_CUDA_ENABLED
    /**
     * Get the default CUDA stream.
     */
    cudaStream_t default_stream() const;

    /**
     * Get or create a stream for a specific purpose.
     */
    cudaStream_t get_stream(StreamPurpose purpose);

    /**
     * Synchronize a stream (or all streams if nullptr).
     */
    void synchronize_stream(cudaStream_t stream = nullptr);

    /**
     * Synchronize all managed streams.
     */
    void synchronize_all_streams();
#endif

    // =========================================================================
    // LibTorch Integration
    // =========================================================================

    /**
     * Get the LibTorch device for the current CUDA device.
     * Returns CPU if CUDA is not available.
     */
    torch::Device torch_device() const;

    /**
     * Get the preferred device (CUDA if available, else CPU).
     */
    torch::Device preferred_device() const;

    /**
     * Get default tensor options for the current device.
     */
    torch::TensorOptions tensor_options(torch::Dtype dtype = torch::kFloat32) const;

    // =========================================================================
    // Error Handling
    // =========================================================================

    /**
     * Check for CUDA errors and throw if any.
     */
    void check_error(const char* operation);

    /**
     * Get the last error message (if any).
     */
    std::string last_error() const;

private:
    CudaContext();
    ~CudaContext();

    void initialize();

    bool cuda_available_;
    int device_count_;
    int device_id_;
    mutable std::mutex mutex_;

#ifdef GNN_CUDA_ENABLED
    std::vector<cudaStream_t> managed_streams_;
    cudaStream_t streams_[5];  // One per StreamPurpose
#endif
};

// ============================================================================
// Macros for CUDA error checking
// ============================================================================

#ifdef GNN_CUDA_ENABLED
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t error = (call);                                           \
        if (error != cudaSuccess) {                                           \
            throw mdb::gnn::CudaException(static_cast<int>(error), #call);   \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                     \
    do {                                                                       \
        cudaError_t error = cudaGetLastError();                               \
        if (error != cudaSuccess) {                                           \
            throw mdb::gnn::CudaException(static_cast<int>(error),           \
                                          "cudaGetLastError");                \
        }                                                                      \
    } while (0)
#else
#define CUDA_CHECK(call) (void)(call)
#define CUDA_CHECK_LAST() (void)0
#endif

} // namespace mdb::gnn
