#pragma once

/**
 * @file radix_partition_sort.h
 * @brief Memory-bounded parallel external sort via radix partitioning.
 *
 * RADIX backend for the projection B+Tree index build phase, selected at
 * runtime via MDB_PROJECTION_SORTER=radix (see sorter_dispatch.h).
 * Implemented in radix_partition_sort.cc; unit tests live in
 * src/tests/radix_partition_sort_test.cc, and golden-compare integration
 * against the CLASSIC backend in scripts/test_projection_radix.sh.
 *
 * Spec reference:
 *   docs/superpowers/specs/2026-04-21-radix-partition-sort-design.md §8.2
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "graph_models/gql/projection/counting_semaphore.h"
#include "graph_models/gql/projection/streaming_record_buffer.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/record.h"

namespace GQL {

/**
 * @brief Memory-bounded parallel external sort via radix partitioning.
 *
 * Phase 1: TBB parallel scan distributes records into per-thread partition
 *          files by top bits of record[0] (the primary key).
 * Phase 2: Each partition is sorted independently by one of num_workers
 *          concurrent workers, with malloc_trim(0) between partitions.
 * Phase 3: Sorted partitions are concatenated into B+Tree leaves (the
 *          radix prefix order guarantees global sorted order).
 */
template<std::size_t N>
class RadixPartitionSort {
public:
    struct Config {
        std::size_t partition_target_bytes = 256ULL * 1024 * 1024;  // 256 MB
        std::size_t min_partitions         = 8;
        std::size_t max_partitions         = 128;
        std::size_t worker_memory_budget   = 512ULL * 1024 * 1024;  // 512 MB
        std::size_t num_workers            = 0;   // 0 = auto (cores/mem)
        std::size_t num_scan_threads       = 0;   // 0 = auto (nproc/2)
        std::string scratch_dir;                   // partition file directory
        // Spec #5 T5.11b — leaf encoding for the B+Tree `.leaf` output.
        // BITSET (default) preserves pre-Spec-#5 byte-identical behavior;
        // DELTA_VARINT opts in to the delta+varint v2 encoding.
        BPT::LeafFormat leaf_format        = BPT::LeafFormat::BITSET;

        // Spec #8 T8.9 — when CSR_HYBRID, Phase 3 emits v3 CSR leaves via
        // BPTLeafCSRWriter<N> instead of the BITSET/DELTA_VARINT writers.
        // Caller gates this to N==3 edge indexes (FROM_TO_EDGE /
        // TO_FROM_EDGE) per design §3.6 D6. Default BTREE preserves
        // pre-Spec-#8 byte-identical RADIX output.
        BPT::GraphStorage graph_storage    = BPT::GraphStorage::BTREE;

        // Keep the intermediate `.sorted_part_*.bin` files after Phase 3
        // instead of removing them. The B+Tree (`.leaf` + `.dir`) remains
        // the authoritative output either way; this exists so tests can
        // inspect per-partition sort order post-hoc. Production callers
        // leave it false.
        bool keep_sorted_parts             = false;
    };

    explicit RadixPartitionSort(Config config);
    ~RadixPartitionSort();

    // No copy, no move (holds open file handles and scratch state).
    // Moves are explicitly deleted — a user-declared dtor + deleted copy ops
    // already suppress implicit move generation, but stating the invariant
    // at the language level blinds it against later dtor/copy refactors that
    // could silently re-enable moves and break the file-handle ownership.
    RadixPartitionSort(const RadixPartitionSort&)            = delete;
    RadixPartitionSort& operator=(const RadixPartitionSort&) = delete;
    RadixPartitionSort(RadixPartitionSort&&)                 = delete;
    RadixPartitionSort& operator=(RadixPartitionSort&&)      = delete;

    /// Phase 1. Partitions the input stream into `scratch_dir` files.
    /// @return number of partitions actually used (post adaptive clamp).
    std::size_t scan_and_partition(
        StreamingRecordBuffer<N>& input,
        std::uint64_t             estimated_count);

    /// Phases 2+3. Sort each partition in parallel; write B+Tree leaves.
    /// @return total unique records written.
    std::size_t sort_and_write(const std::string& output_base_path);

    /// Compute num_partitions for a given dataset size (testable independently).
    static std::size_t compute_num_partitions(
        std::size_t total_bytes,
        std::size_t partition_target_bytes,
        std::size_t min_partitions,
        std::size_t max_partitions);

    /// Compute num_workers based on cores, memory budget, and worker budget.
    static std::size_t compute_num_workers(
        std::size_t cores_available,
        std::size_t scan_threads,
        std::size_t memory_budget,
        std::size_t worker_memory_budget,
        std::size_t override_value);

    /// Task 4.2 telemetry: how many partitions Phase 2 sorted on the GPU vs
    /// the CPU. Populated during sort_and_write(); a single summary line is
    /// also emitted to stderr at the end of that call. On non-CUDA builds (or
    /// when the GPU path is disabled / planner-downgraded) every partition is
    /// counted as CPU.
    std::size_t gpu_partitions_sorted() const { return gpu_partitions_.load(); }
    std::size_t cpu_partitions_sorted() const { return cpu_partitions_.load(); }

    /// Task 5.1 test seam: the running maximum number of Phase 2 workers that
    /// were simultaneously inside the GPU-submission region. The GPU
    /// concurrency semaphore (sized from MDB_PROJECTION_RADIX_GPU_CONCURRENCY,
    /// default 1) bounds this to kGpuConcurrency; the unit test asserts the
    /// observed peak never exceeds that bound. On a non-CUDA build (or when
    /// no GPU is present and every partition routes to CPU) this stays 0.
    int gpu_peak_in_flight() const { return gpu_peak_in_flight_.load(); }

private:
    Config config_;
    std::size_t num_partitions_ = 0;
    std::vector<std::string> partition_paths_;  // per-partition merged files

    // Task 4.2: per-partition GPU/CPU sort tallies. Atomic because Phase 2
    // dispatches partitions across a tbb::parallel_for worker pool.
    std::atomic<std::size_t> gpu_partitions_{0};
    std::atomic<std::size_t> cpu_partitions_{0};

    // Task 5.1: bound how many Phase 2 workers submit to the single GPU at
    // once. Sized once in the ctor from MDB_PROJECTION_RADIX_GPU_CONCURRENCY
    // (default 1, clamped to [1, 8]); the remaining workers sort their
    // partitions on the CPU concurrently — that overlap is the "CPU and GPU
    // both busy" win. unique_ptr because CountingSemaphore is non-copyable
    // and the permit count is only known at construction time.
    int gpu_concurrency_ = 1;
    std::unique_ptr<CountingSemaphore> gpu_semaphore_;

    // Test seam: current / peak workers inside the GPU-submission region.
    // Incremented before acquiring the semaphore's effect is observable and
    // decremented on region exit; the peak is maintained via a CAS loop.
    std::atomic<int> gpu_in_flight_{0};
    std::atomic<int> gpu_peak_in_flight_{0};

    // Update gpu_peak_in_flight_ to max(current, candidate) via CAS.
    void bump_gpu_peak_(int candidate) {
        int prev = gpu_peak_in_flight_.load(std::memory_order_relaxed);
        while (candidate > prev &&
               !gpu_peak_in_flight_.compare_exchange_weak(
                   prev, candidate, std::memory_order_relaxed)) {
            // prev reloaded by compare_exchange_weak; retry.
        }
    }

    std::uint32_t radix_bucket(const Record<N>& r) const;
    void sort_partition_in_memory(std::size_t partition_idx,
                                  const std::string& sorted_output_path);
    void sort_partition_external(std::size_t partition_idx,
                                 const std::string& sorted_output_path);
};

// Forward declarations for explicit instantiations.
extern template class RadixPartitionSort<1>;
extern template class RadixPartitionSort<2>;
extern template class RadixPartitionSort<3>;

}  // namespace GQL
