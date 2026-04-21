// src/graph_models/gql/projection/radix_partition_sort.cc
#include "graph_models/gql/projection/radix_partition_sort.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "graph_models/gql/projection/parallel_scan_partitioner.h"
#include "graph_models/gql/projection/partition_file.h"

namespace fs = std::filesystem;

namespace GQL {

template<std::size_t N>
std::size_t RadixPartitionSort<N>::compute_num_partitions(
    std::size_t total_bytes,
    std::size_t partition_target_bytes,
    std::size_t min_partitions,
    std::size_t max_partitions)
{
    if (partition_target_bytes == 0) return min_partitions;
    std::size_t raw = (total_bytes + partition_target_bytes - 1) / partition_target_bytes;
    return std::clamp(raw, min_partitions, max_partitions);
}

template<std::size_t N>
std::size_t RadixPartitionSort<N>::compute_num_workers(
    std::size_t cores_available,
    std::size_t scan_threads,
    std::size_t memory_budget,
    std::size_t worker_memory_budget,
    std::size_t override_value)
{
    if (override_value > 0) return override_value;
    std::size_t by_cores  = (cores_available > scan_threads)
                              ? (cores_available - scan_threads) : 1;
    std::size_t by_memory = (worker_memory_budget > 0)
                              ? (memory_budget / worker_memory_budget) : 1;
    return std::max<std::size_t>(1, std::min(by_cores, by_memory));
}

template<std::size_t N>
RadixPartitionSort<N>::RadixPartitionSort(Config config)
    : config_(std::move(config))
{
    if (config_.scratch_dir.empty()) {
        throw std::invalid_argument("RadixPartitionSort: scratch_dir is required");
    }
    fs::create_directories(config_.scratch_dir);
}

template<std::size_t N>
RadixPartitionSort<N>::~RadixPartitionSort() {
    // Exception-safe cleanup of any remaining scratch files.
    try { fs::remove_all(config_.scratch_dir); } catch (...) {}
}

template<std::size_t N>
std::uint32_t RadixPartitionSort<N>::radix_bucket(const Record<N>& r) const {
    // Top bits of record[0] → bucket index.
    if (num_partitions_ <= 1) return 0;
    // bit_width(x) = 64 - __builtin_clzll(x) for x > 0 (portable C++17 alt).
    int bit_width = 64 - __builtin_clzll(static_cast<unsigned long long>(num_partitions_ - 1));
    int shift = 64 - bit_width;
    return static_cast<std::uint32_t>(r[0] >> shift);
}

template<std::size_t N>
std::size_t RadixPartitionSort<N>::scan_and_partition(
    StreamingRecordBuffer<N>& input,
    std::uint64_t             estimated_count)
{
    std::size_t total_bytes = estimated_count * sizeof(Record<N>);
    num_partitions_ = compute_num_partitions(
        total_bytes, config_.partition_target_bytes,
        config_.min_partitions, config_.max_partitions);

    std::size_t scan_threads = (config_.num_scan_threads > 0)
        ? config_.num_scan_threads
        : std::max<std::size_t>(1, std::thread::hardware_concurrency() / 2);

    ParallelScanPartitioner<N> partitioner(
        num_partitions_, scan_threads, config_.scratch_dir,
        [this](const Record<N>& r) { return radix_bucket(r); });

    partitioner.run([&](auto emit) {
        // Single-threaded drain of the input buffer for now. The full
        // TBB-driven B+Tree scan integration lands in the projection
        // builder wiring task (Task 12); here we consume a streaming
        // buffer directly to satisfy the unit tests.
        input.begin_iteration();
        while (input.has_next()) {
            emit(input.next());
        }
    });

    partition_paths_ = partitioner.collect_merged_partition_paths();
    return num_partitions_;
}

template<std::size_t N>
std::size_t RadixPartitionSort<N>::sort_and_write(
    const std::string& output_base_path)
{
    // Stub — implemented in Task 8+.
    (void)output_base_path;
    return 0;
}

template<std::size_t N>
void RadixPartitionSort<N>::sort_partition_in_memory(
    std::size_t partition_idx, const std::string& sorted_output_path)
{
    (void)partition_idx; (void)sorted_output_path;  // Task 8.
}

template<std::size_t N>
void RadixPartitionSort<N>::sort_partition_external(
    std::size_t partition_idx, const std::string& sorted_output_path)
{
    (void)partition_idx; (void)sorted_output_path;  // Task 9.
}

template class RadixPartitionSort<1>;
template class RadixPartitionSort<2>;
template class RadixPartitionSort<3>;

}  // namespace GQL
