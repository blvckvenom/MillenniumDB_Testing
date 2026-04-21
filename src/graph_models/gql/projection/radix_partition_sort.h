#pragma once

/**
 * @file radix_partition_sort.h
 * @brief Memory-bounded parallel external sort via radix partitioning.
 *
 * Skeleton header committed as part of the TDD RED cycle for the
 * Radix-Partition Sort campaign (M2 milestone).
 *
 * The implementation is added incrementally across Tasks 5-11. At the
 * current commit the declarations below are intentionally unbacked by
 * any `.cc` translation unit, so linking `radix_partition_sort_test`
 * yields unresolved-symbol errors. This is the expected RED state.
 *
 * Spec reference:
 *   docs/superpowers/specs/2026-04-21-radix-partition-sort-design.md §8.2
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "graph_models/gql/projection/streaming_record_buffer.h"
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

private:
    Config config_;
    std::size_t num_partitions_ = 0;
    std::vector<std::string> partition_paths_;  // per-partition merged files

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
