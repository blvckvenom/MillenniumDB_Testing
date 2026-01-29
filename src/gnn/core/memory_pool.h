#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

#include <torch/torch.h>

namespace mdb::gnn {

/**
 * @brief Actions to take when GPU memory pressure is detected.
 */
enum class MemoryPressureAction {
    REDUCE_BATCH_SIZE,          ///< Suggest using smaller batches
    USE_GRADIENT_CHECKPOINTING, ///< Trade compute for memory
    OFFLOAD_TO_CPU,             ///< Move some data to CPU
    FAIL                        ///< Cannot proceed, throw exception
};

/**
 * @brief Memory statistics for the GPU memory pool.
 */
struct MemoryPoolStats {
    size_t total_size;          ///< Total pool size in bytes
    size_t allocated_size;      ///< Currently allocated bytes
    size_t free_size;           ///< Available bytes
    size_t num_allocations;     ///< Total allocation count
    size_t num_deallocations;   ///< Total deallocation count (for reference pools)
    size_t num_resets;          ///< Number of pool resets
    size_t peak_usage;          ///< Maximum bytes ever allocated
    size_t fragmentation_bytes; ///< Estimated fragmentation (arena pools: 0)
};

/**
 * @brief Memory estimation for a single GNN batch.
 *
 * Pre-compute memory requirements based on batch parameters to
 * configure the memory pool appropriately.
 */
struct BatchMemoryEstimate {
    size_t feature_tensor;      ///< num_nodes * feature_dim * sizeof(float)
    size_t adjacency_indices;   ///< 2 * num_edges * sizeof(int64_t)
    size_t adjacency_values;    ///< num_edges * sizeof(float)
    size_t activations;         ///< Intermediate activations (depends on model)
    size_t gradients;           ///< Same as activations during backward
    size_t total;               ///< Total with safety margin

    /**
     * @brief Compute memory estimate for a batch.
     *
     * @param num_nodes Number of nodes in the batch
     * @param num_edges Number of edges in the batch
     * @param feature_dim Input feature dimension
     * @param hidden_dim Hidden layer dimension
     * @param num_layers Number of GNN layers
     * @param safety_margin Multiplier for safety (default 1.2 = 20%)
     * @return BatchMemoryEstimate with all components
     */
    static BatchMemoryEstimate compute(
        int64_t num_nodes,
        int64_t num_edges,
        int64_t feature_dim,
        int64_t hidden_dim,
        int num_layers,
        double safety_margin = 1.2
    );
};

/**
 * @brief GPU memory pool using arena allocation.
 *
 * Provides efficient GPU memory management for GNN training with
 * predictable allocation patterns. Uses arena (bump) allocation
 * for O(1) allocations and O(1) reset.
 *
 * Design philosophy:
 * - Mini-batches have predictable lifetime: allocate all data,
 *   process the batch, then discard everything
 * - Arena allocation: just bump a pointer, no free list management
 * - Reset between batches: O(1) operation to reclaim all memory
 *
 * Usage:
 *   GpuMemoryPool pool(config);
 *   for (int batch = 0; batch < num_batches; ++batch) {
 *       auto features = pool.allocate_tensor({1000, 256}, torch::kFloat32);
 *       auto adj = pool.allocate_tensor({2, 5000}, torch::kInt64);
 *       // ... process batch ...
 *       pool.reset();  // Reclaim all memory for next batch
 *   }
 *
 * @see CudaContext for device management
 * @see BatchMemoryEstimate for sizing guidance
 */
class GpuMemoryPool {
public:
    /**
     * @brief Configuration for the memory pool.
     */
    struct Config {
        size_t initial_size = 256 * 1024 * 1024;  ///< Initial pool size (256 MB default)
        double growth_factor = 1.5;                ///< Growth multiplier when exhausted
        size_t max_size = 0;                       ///< Maximum pool size (0 = no limit)
        bool allow_growth = true;                  ///< Can pool grow beyond initial_size?
        double reserved_fraction = 0.9;            ///< Fraction of GPU memory to reserve
        size_t alignment = 256;                    ///< Memory alignment (256 for CUDA)
    };

    /**
     * @brief Construct a GPU memory pool.
     *
     * @param config Pool configuration
     * @throws CudaException if CUDA allocation fails
     */
    explicit GpuMemoryPool(const Config& config);

    /**
     * @brief Destructor - frees all GPU memory.
     */
    ~GpuMemoryPool();

    // Non-copyable
    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;

    // Movable
    GpuMemoryPool(GpuMemoryPool&& other) noexcept;
    GpuMemoryPool& operator=(GpuMemoryPool&& other) noexcept;

    // =========================================================================
    // Raw Allocation
    // =========================================================================

    /**
     * @brief Allocate raw GPU memory.
     *
     * Memory is aligned to config.alignment (256 bytes by default).
     *
     * @param size Number of bytes to allocate
     * @return Pointer to GPU memory
     * @throws std::bad_alloc if allocation fails
     */
    void* allocate(size_t size);

    /**
     * @brief Deallocate memory (no-op for arena allocation).
     *
     * In arena allocation, individual deallocations are no-ops.
     * Memory is only reclaimed via reset().
     *
     * @param ptr Pointer to deallocate (ignored)
     */
    void deallocate(void* ptr);

    // =========================================================================
    // Tensor Allocation
    // =========================================================================

    /**
     * @brief Allocate a CUDA tensor from the pool.
     *
     * The returned tensor uses memory from the pool. The tensor
     * does NOT own its memory - it will be invalidated after reset().
     *
     * @param shape Tensor shape
     * @param dtype Data type (default: float32)
     * @return CUDA tensor using pool memory
     * @throws std::bad_alloc if allocation fails
     */
    torch::Tensor allocate_tensor(
        const std::vector<int64_t>& shape,
        torch::Dtype dtype = torch::kFloat32
    );

    /**
     * @brief Allocate a CUDA tensor initialized to zero.
     *
     * @param shape Tensor shape
     * @param dtype Data type (default: float32)
     * @return Zero-initialized CUDA tensor using pool memory
     */
    torch::Tensor allocate_tensor_zeros(
        const std::vector<int64_t>& shape,
        torch::Dtype dtype = torch::kFloat32
    );

    // =========================================================================
    // Pool Management
    // =========================================================================

    /**
     * @brief Reset the pool, reclaiming all allocations.
     *
     * This is an O(1) operation that resets the arena offset to 0.
     * All tensors allocated from this pool become invalid after reset.
     */
    void reset();

    /**
     * @brief Reserve memory to avoid runtime allocation.
     *
     * Pre-allocates arenas to hold at least the specified size.
     *
     * @param size Minimum bytes to reserve
     */
    void reserve(size_t size);

    /**
     * @brief Release all pool memory back to CUDA.
     *
     * More aggressive than reset() - actually frees GPU memory.
     * Pool will reallocate on next use.
     */
    void release();

    /**
     * @brief Release LibTorch's cached memory.
     *
     * Useful before querying actual free GPU memory.
     */
    static void release_torch_cache();

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get current memory statistics.
     */
    MemoryPoolStats get_stats() const;

    /**
     * @brief Get total capacity of the pool.
     */
    size_t capacity() const;

    /**
     * @brief Get currently allocated bytes.
     */
    size_t allocated() const;

    /**
     * @brief Get available bytes without growing.
     */
    size_t available() const;

    // =========================================================================
    // Persistent Allocation (LibTorch managed)
    // =========================================================================

    /**
     * @brief Allocate a persistent tensor (not from pool).
     *
     * Use for tensors that outlive batch processing, like model weights.
     * These tensors are managed by LibTorch's allocator, not our pool.
     *
     * @param shape Tensor shape
     * @param dtype Data type
     * @return CUDA tensor managed by LibTorch
     */
    static torch::Tensor allocate_persistent(
        const std::vector<int64_t>& shape,
        torch::Dtype dtype = torch::kFloat32
    );

    // =========================================================================
    // Memory Pressure Handling
    // =========================================================================

    /**
     * @brief Callback type for memory pressure events.
     */
    using PressureCallback = std::function<void(MemoryPressureAction, size_t, size_t)>;

    /**
     * @brief Set callback for memory pressure events.
     *
     * The callback receives (action, requested_size, available_size).
     */
    void set_pressure_callback(PressureCallback callback);

    /**
     * @brief Check if allocation would succeed.
     *
     * @param size Bytes to check
     * @return true if allocation would succeed without growing
     */
    bool can_allocate(size_t size) const;

private:
    /**
     * @brief Internal arena structure.
     */
    struct Arena {
        void* base_ptr;     ///< Base pointer to GPU memory
        size_t size;        ///< Total arena size
        size_t offset;      ///< Current allocation offset

        Arena() : base_ptr(nullptr), size(0), offset(0) {}
        Arena(void* ptr, size_t s) : base_ptr(ptr), size(s), offset(0) {}

        size_t available() const { return size - offset; }
    };

    /**
     * @brief Create a new arena with the given size.
     */
    Arena create_arena(size_t size);

    /**
     * @brief Free an arena's memory.
     */
    void free_arena(Arena& arena);

    /**
     * @brief Handle memory pressure during allocation.
     */
    void handle_pressure(size_t requested_size);

    /**
     * @brief Align size to pool alignment.
     */
    size_t align_size(size_t size) const;

    Config config_;
    std::vector<Arena> arenas_;
    mutable std::mutex mutex_;
    MemoryPoolStats stats_;
    PressureCallback pressure_callback_;
};

/**
 * @brief RAII guard for pool-scoped allocations.
 *
 * Automatically resets the pool when the guard goes out of scope.
 *
 * Usage:
 *   {
 *       PoolScope scope(pool);
 *       auto tensor = pool.allocate_tensor({100, 256});
 *       // ... use tensor ...
 *   }  // pool.reset() called automatically
 */
class PoolScope {
public:
    explicit PoolScope(GpuMemoryPool& pool) : pool_(pool) {}
    ~PoolScope() { pool_.reset(); }

    // Non-copyable, non-movable
    PoolScope(const PoolScope&) = delete;
    PoolScope& operator=(const PoolScope&) = delete;

private:
    GpuMemoryPool& pool_;
};

} // namespace mdb::gnn
