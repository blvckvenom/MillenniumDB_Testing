// src/graph_models/gql/projection/parallel_scan_partitioner.h
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "graph_models/gql/projection/partition_file.h"
#include "storage/index/record.h"

namespace GQL {

template<std::size_t N>
class ParallelScanPartitioner {
public:
    ParallelScanPartitioner(
        std::size_t        num_partitions,
        std::size_t        num_scan_threads,
        const std::string& scratch_dir,
        std::function<std::uint32_t(const Record<N>&)> bucket_hash);

    ~ParallelScanPartitioner();

    /// Drive the scan: scan_fn receives an emit(record) callback and is
    /// expected to call it for each record to partition. The callback is
    /// thread-local; calling it is safe from inside any TBB task.
    void run(std::function<void(
        std::function<void(const Record<N>&)>)> scan_fn);

    /// After run(), returns the merged per-partition file list. Each
    /// partition is the concatenation of thread_*/part_p.bin files.
    std::vector<std::string> collect_merged_partition_paths();

private:
    std::size_t num_partitions_;
    std::size_t num_scan_threads_;
    std::string scratch_dir_;
    std::function<std::uint32_t(const Record<N>&)> bucket_hash_;

    // Per-(thread, partition) output files. Created lazily in run().
    std::vector<std::vector<std::unique_ptr<PartitionFile<N>>>> files_;
};

extern template class ParallelScanPartitioner<1>;
extern template class ParallelScanPartitioner<2>;
extern template class ParallelScanPartitioner<3>;

}  // namespace GQL
