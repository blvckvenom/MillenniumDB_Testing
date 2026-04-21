// src/graph_models/gql/projection/parallel_scan_partitioner.cc
#include "graph_models/gql/projection/parallel_scan_partitioner.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <filesystem>
#include <fstream>
#include <mutex>

namespace fs = std::filesystem;

namespace GQL {

template<std::size_t N>
ParallelScanPartitioner<N>::ParallelScanPartitioner(
    std::size_t        num_partitions,
    std::size_t        num_scan_threads,
    const std::string& scratch_dir,
    std::function<std::uint32_t(const Record<N>&)> bucket_hash)
    : num_partitions_(num_partitions),
      num_scan_threads_(num_scan_threads),
      scratch_dir_(scratch_dir),
      bucket_hash_(std::move(bucket_hash))
{
    fs::create_directories(scratch_dir_);
    files_.resize(num_scan_threads_);
    for (std::size_t t = 0; t < num_scan_threads_; ++t) {
        fs::create_directories(fs::path(scratch_dir_) / ("thread_" + std::to_string(t)));
        files_[t].reserve(num_partitions_);
        for (std::size_t p = 0; p < num_partitions_; ++p) {
            std::string path = fs::path(scratch_dir_) /
                               ("thread_" + std::to_string(t)) /
                               ("part_" + std::to_string(p) + ".bin");
            files_[t].push_back(std::make_unique<PartitionFile<N>>(path));
        }
    }
}

template<std::size_t N>
ParallelScanPartitioner<N>::~ParallelScanPartitioner() {
    // Files auto-flush in their destructors.
}

template<std::size_t N>
void ParallelScanPartitioner<N>::run(
    std::function<void(
        std::function<void(const Record<N>&)>)> scan_fn)
{
    // NOTE: the caller's scan_fn is responsible for parallelizing itself
    // via TBB. Our job is to provide an emit() callback that routes each
    // record to the correct per-thread bucket. We use thread_local state
    // to avoid synchronization.
    std::mutex thread_slot_assign_mutex;
    std::size_t next_slot = 0;
    thread_local int my_slot_idx = -1;

    auto emit = [&](const Record<N>& r) {
        if (my_slot_idx < 0) {
            std::lock_guard<std::mutex> lk(thread_slot_assign_mutex);
            my_slot_idx = static_cast<int>(next_slot++);
            if (my_slot_idx >= static_cast<int>(num_scan_threads_)) {
                throw std::runtime_error("ParallelScanPartitioner: more threads than slots");
            }
        }
        std::uint32_t bucket = bucket_hash_(r) % num_partitions_;
        files_[my_slot_idx][bucket]->append(r);
    };

    scan_fn(emit);

    // Flush all partition files
    for (auto& thread_files : files_) {
        for (auto& f : thread_files) {
            f->flush();
        }
    }
}

template<std::size_t N>
std::vector<std::string>
ParallelScanPartitioner<N>::collect_merged_partition_paths() {
    // For each partition p, concatenate thread_*/part_p.bin into
    // scratch_dir_/partition_p.bin.
    std::vector<std::string> out;
    out.reserve(num_partitions_);
    for (std::size_t p = 0; p < num_partitions_; ++p) {
        std::string merged = fs::path(scratch_dir_) /
                             ("partition_" + std::to_string(p) + ".bin");
        std::ofstream sink(merged, std::ios::binary);
        for (std::size_t t = 0; t < num_scan_threads_; ++t) {
            std::string src = fs::path(scratch_dir_) /
                              ("thread_" + std::to_string(t)) /
                              ("part_" + std::to_string(p) + ".bin");
            std::ifstream in(src, std::ios::binary);
            sink << in.rdbuf();
            fs::remove(src);  // reclaim disk
        }
        out.push_back(merged);
    }
    // Cleanup per-thread dirs
    for (std::size_t t = 0; t < num_scan_threads_; ++t) {
        fs::remove_all(fs::path(scratch_dir_) / ("thread_" + std::to_string(t)));
    }
    return out;
}

template class ParallelScanPartitioner<1>;
template class ParallelScanPartitioner<2>;
template class ParallelScanPartitioner<3>;

}  // namespace GQL
