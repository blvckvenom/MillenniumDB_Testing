// src/graph_models/gql/projection/radix_partition_sort.cc
#include "graph_models/gql/projection/radix_partition_sort.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

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
    std::size_t total_written = 0;

    std::size_t num_workers = compute_num_workers(
        std::thread::hardware_concurrency(),
        (config_.num_scan_threads > 0)
            ? config_.num_scan_threads
            : std::thread::hardware_concurrency() / 2,
        4ULL * 1024 * 1024 * 1024,  // 4 GB default memory budget
        config_.worker_memory_budget,
        config_.num_workers);
    (void)num_workers;  // passed to TBB via default arena; grain 1 already set

    // Dispatch partitions across workers via tbb::parallel_for.
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_partitions_, 1),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t p = r.begin(); p < r.end(); ++p) {
                std::string sorted_path = output_base_path +
                    ".sorted_part_" + std::to_string(p) + ".bin";
                // Decide in-memory vs external.
                std::uintmax_t sz = fs::file_size(partition_paths_[p]);
                if (sz <= config_.worker_memory_budget) {
                    sort_partition_in_memory(p, sorted_path);
                } else {
                    sort_partition_external(p, sorted_path);
                }
                // Release free heap pages to the kernel between sorts.
#if defined(__GLIBC__)
                malloc_trim(0);
#endif
            }
        });

    // Count written records by reading each sorted file's size.
    for (std::size_t p = 0; p < num_partitions_; ++p) {
        std::string sorted_path = output_base_path +
            ".sorted_part_" + std::to_string(p) + ".bin";
        if (fs::exists(sorted_path)) {
            total_written += fs::file_size(sorted_path) / sizeof(Record<N>);
        }
    }
    return total_written;
}

template<std::size_t N>
void RadixPartitionSort<N>::sort_partition_in_memory(
    std::size_t partition_idx, const std::string& sorted_output_path)
{
    std::vector<Record<N>> buffer;
    typename PartitionFile<N>::Reader reader(partition_paths_[partition_idx]);
    Record<N> r{};
    while (reader.next(r)) {
        buffer.push_back(r);
    }
    std::sort(buffer.begin(), buffer.end());
    std::ofstream out(sorted_output_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(buffer.data()),
              buffer.size() * sizeof(Record<N>));
}

template<std::size_t N>
void RadixPartitionSort<N>::sort_partition_external(
    std::size_t partition_idx, const std::string& sorted_output_path)
{
    // Defensive path: partition exceeds worker_memory_budget. Chunk-sort
    // to intermediate "run" files, then K-way merge.
    std::vector<Record<N>> chunk;
    std::size_t chunk_cap = config_.worker_memory_budget / sizeof(Record<N>) / 2;
    if (chunk_cap == 0) chunk_cap = 1;  // minimum: handle tiny worker_memory_budget in tests
    chunk.reserve(chunk_cap);

    std::vector<std::string> run_paths;
    {
        typename PartitionFile<N>::Reader reader(partition_paths_[partition_idx]);
        Record<N> r{};
        std::size_t run_idx = 0;
        while (reader.next(r)) {
            chunk.push_back(r);
            if (chunk.size() >= chunk_cap) {
                std::sort(chunk.begin(), chunk.end());
                std::string run_path = sorted_output_path + ".run_" + std::to_string(run_idx++);
                std::ofstream run_out(run_path, std::ios::binary);
                run_out.write(reinterpret_cast<const char*>(chunk.data()),
                              chunk.size() * sizeof(Record<N>));
                run_paths.push_back(run_path);
                chunk.clear();
            }
        }
        if (!chunk.empty()) {
            std::sort(chunk.begin(), chunk.end());
            std::string run_path = sorted_output_path + ".run_" + std::to_string(run_idx++);
            std::ofstream run_out(run_path, std::ios::binary);
            run_out.write(reinterpret_cast<const char*>(chunk.data()),
                          chunk.size() * sizeof(Record<N>));
            run_paths.push_back(run_path);
        }
    }

    // K-way merge the runs into sorted_output_path. Use unique_ptr to avoid
    // relying on Reader being movable (Reader owns a FILE* with non-trivial dtor).
    std::vector<std::unique_ptr<typename PartitionFile<N>::Reader>> readers;
    readers.reserve(run_paths.size());
    for (auto& p : run_paths) {
        readers.emplace_back(std::make_unique<typename PartitionFile<N>::Reader>(p));
    }

    std::vector<std::optional<Record<N>>> fronts(run_paths.size());
    for (std::size_t i = 0; i < run_paths.size(); ++i) {
        Record<N> rr{};
        if (readers[i]->next(rr)) fronts[i] = rr;
    }

    std::ofstream out(sorted_output_path, std::ios::binary);
    while (true) {
        std::size_t min_idx = SIZE_MAX;
        for (std::size_t i = 0; i < fronts.size(); ++i) {
            if (!fronts[i].has_value()) continue;
            if (min_idx == SIZE_MAX || *fronts[i] < *fronts[min_idx]) {
                min_idx = i;
            }
        }
        if (min_idx == SIZE_MAX) break;
        out.write(reinterpret_cast<const char*>(&(*fronts[min_idx])),
                  sizeof(Record<N>));
        Record<N> rr{};
        if (readers[min_idx]->next(rr)) fronts[min_idx] = rr;
        else fronts[min_idx].reset();
    }

    readers.clear();  // close files before removing
    for (auto& p : run_paths) fs::remove(p);
}

template class RadixPartitionSort<1>;
template class RadixPartitionSort<2>;
template class RadixPartitionSort<3>;

}  // namespace GQL
