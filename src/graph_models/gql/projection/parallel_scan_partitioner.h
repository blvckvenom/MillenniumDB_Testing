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

    /// Spec #25 (Option C — CPU TBB) — chunk-parallel partition fill.
    ///
    /// Drives the scan by repeatedly pulling fixed-size chunks of records
    /// from `scan_fn` (single-threaded producer) and dispatching the
    /// partitioning of each chunk across TBB workers. Each worker is
    /// assigned a stable per-thread output slot (recycled across chunks),
    /// so writes go to `files_[my_slot][bucket]` without locking.
    ///
    /// `scan_fn(out_chunk, max_records)` must:
    ///  - append up to `max_records` Records into `out_chunk` (it may
    ///    append fewer; an empty result terminates the drain).
    ///  - never block on I/O outside its own logic; this method invokes
    ///    it in a hot loop on the calling thread.
    ///
    /// Output is byte-identical (modulo the order of records WITHIN each
    /// per-thread `part_p.bin` file, which Phase 2 sorts away) to a
    /// sequential `run()` call over the same input stream.
    ///
    /// Tuning: `chunk_size_records` defaults to 64 K (~1.5 MB at N=3,
    /// fits in L2 cache and amortizes TBB task dispatch overhead).
    void run_parallel_chunked(
        std::function<std::size_t(
            std::vector<Record<N>>& /*out_chunk*/,
            std::size_t             /*max_records*/)> scan_fn,
        std::size_t chunk_size_records = 65536);

    /// After run() / run_parallel_chunked(), returns the merged per-partition
    /// file list. Each partition is the concatenation of thread_*/part_p.bin
    /// files.
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
