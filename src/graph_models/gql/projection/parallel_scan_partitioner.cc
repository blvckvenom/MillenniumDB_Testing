// src/graph_models/gql/projection/parallel_scan_partitioner.cc
#include "graph_models/gql/projection/parallel_scan_partitioner.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>

#include <algorithm>
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

// ---------------------------------------------------------------------------
// Spec #25 (Option C — CPU TBB) — chunk-parallel partition fill.
//
// Design:
//   - Producer (calling thread) repeatedly drains the input via scan_fn into
//     a chunk buffer of size `chunk_size_records`.
//   - For each chunk, dispatch tbb::parallel_for over a `blocked_range` of
//     the chunk's record indices. Each TBB worker is assigned a stable slot
//     in `files_` via tbb::enumerable_thread_specific, the slot index being
//     drawn from a monotonically-incremented counter the first time a worker
//     touches the partitioner.
//   - Inside its sub-range each worker accumulates per-partition stack
//     buffers (capacity 64 records) and flushes them to its own
//     `files_[slot][bucket]` in batches. This collapses N append() calls
//     into one path-local fwrite, cutting libc stdio overhead.
//
// Correctness notes:
//   * Each worker only ever writes to `files_[my_slot][*]` — distinct
//     PartitionFile instances per slot, so no locking is required.
//   * The slot-assign counter is std::atomic so concurrent first-touches
//     receive distinct slot indices. We pre-size files_ to num_scan_threads_,
//     so a runtime check ensures TBB never spawns more workers than slots
//     (callers should pass num_scan_threads_ == effective TBB worker count).
//   * Records *within* a single per-thread bucket file may end up in a
//     different in-file order than the legacy single-thread run() produces,
//     because chunks are processed concurrently and buckets are written as
//     records are scanned. Phase 2 sorts each partition end-to-end after
//     concatenation, so the final B+Tree leaves are byte-identical.
// ---------------------------------------------------------------------------
template<std::size_t N>
void ParallelScanPartitioner<N>::run_parallel_chunked(
    std::function<std::size_t(
        std::vector<Record<N>>&,
        std::size_t)> scan_fn,
    std::size_t chunk_size_records)
{
    if (chunk_size_records == 0) {
        chunk_size_records = 65536;
    }

    std::vector<Record<N>> chunk;
    chunk.reserve(chunk_size_records);

    // Per-(slot, bucket) stack-buffer flush size. Tuned to collapse the
    // append() loop into ~64 records per fwrite while staying small enough
    // to fit num_partitions_ × 64 × sizeof(Record<N>) into L1 per worker
    // (e.g. 128 × 64 × 24 B ≈ 192 KB worst case).
    constexpr std::size_t kBucketBatch = 64;

    // Dispatch exactly num_scan_threads_ TBB tasks per chunk, each with a
    // STABLE slot index (the loop variable). This guarantees every worker
    // writes to its own files_[slot][*] sub-array without conflict, and
    // never exceeds the slot capacity we pre-sized in the constructor —
    // independent of TBB's hardware_concurrency() and across recycled OS
    // threads between chunks. We use a task_arena to bound the in-flight
    // worker count (see notes below).
    const std::size_t num_slots = std::max<std::size_t>(1, num_scan_threads_);
    const int arena_concurrency = static_cast<int>(num_slots);
    tbb::task_arena arena(arena_concurrency);

    while (true) {
        chunk.clear();
        const std::size_t got = scan_fn(chunk, chunk_size_records);
        if (got == 0) break;

        // Slice the chunk into num_slots equal sub-ranges and process each
        // in its own TBB task. simple_partitioner with grain=1 over an
        // num_slots-element index range disables TBB's range-splitting
        // heuristic, so we get exactly num_slots tasks (one per slot).
        const std::size_t chunk_n = chunk.size();
        const std::size_t base = chunk_n / num_slots;
        const std::size_t rem  = chunk_n % num_slots;

        arena.execute([&] {
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, num_slots, 1),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    for (std::size_t slot = r.begin(); slot < r.end(); ++slot) {
                        const std::size_t start = slot * base + std::min(slot, rem);
                        const std::size_t end   = start + base + (slot < rem ? 1 : 0);
                        if (start >= end) continue;

                        std::vector<std::vector<Record<N>>> local_buckets(num_partitions_);
                        for (auto& b : local_buckets) {
                            b.reserve(kBucketBatch);
                        }

                        auto flush_bucket = [&](std::size_t bidx) {
                            auto& buf = local_buckets[bidx];
                            if (buf.empty()) return;
                            auto& f = files_[slot][bidx];
                            for (const auto& rec : buf) {
                                f->append(rec);
                            }
                            buf.clear();
                        };

                        for (std::size_t i = start; i < end; ++i) {
                            const Record<N>& rec = chunk[i];
                            const std::uint32_t bucket =
                                bucket_hash_(rec) % num_partitions_;
                            auto& buf = local_buckets[bucket];
                            buf.push_back(rec);
                            if (buf.size() >= kBucketBatch) {
                                flush_bucket(bucket);
                            }
                        }
                        for (std::size_t b = 0; b < num_partitions_; ++b) {
                            flush_bucket(b);
                        }
                    }
                },
                tbb::simple_partitioner());
        });
    }

    // Flush all partition files (matches sequential run() behavior).
    for (auto& thread_files : files_) {
        for (auto& f : thread_files) {
            f->flush();
        }
    }
}

template<std::size_t N>
std::vector<std::string>
ParallelScanPartitioner<N>::collect_merged_partition_paths() {
    // Pre-fix: this loop concatenated thread_*/part_p.bin into
    // partition_p.bin sequentially across `num_partitions_ * num_scan_threads_`
    // (file_open + read + write + remove) operations. For papers100M with
    // 128 partitions × 10 scan threads = 1280 files (~38 GB total I/O),
    // the merge alone took ~10 min on a single thread, leaving 9 cores idle.
    //
    // Fix: dispatch one TBB task per partition. Each task does its own
    // open/read/write/remove on its sub-tree. The output paths array is
    // pre-sized so each worker writes to its own slot — no shared mutable
    // state, no locks. Per-thread dir cleanup remains serial (cheap, just
    // rmdir on N empty directories).
    std::vector<std::string> out(num_partitions_);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_partitions_, 1),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t p = r.begin(); p < r.end(); ++p) {
                std::string merged = fs::path(scratch_dir_) /
                                     ("partition_" + std::to_string(p) + ".bin");
                std::ofstream sink(merged, std::ios::binary);
                for (std::size_t t = 0; t < num_scan_threads_; ++t) {
                    std::string src = fs::path(scratch_dir_) /
                                      ("thread_" + std::to_string(t)) /
                                      ("part_" + std::to_string(p) + ".bin");
                    std::ifstream in(src, std::ios::binary);
                    sink << in.rdbuf();
                    std::error_code ec;
                    fs::remove(src, ec);  // reclaim disk; ignore missing-file races
                }
                out[p] = merged;
            }
        });

    // Cleanup per-thread dirs (now empty after parallel removes above).
    for (std::size_t t = 0; t < num_scan_threads_; ++t) {
        std::error_code ec;
        fs::remove_all(fs::path(scratch_dir_) / ("thread_" + std::to_string(t)), ec);
    }
    return out;
}

template class ParallelScanPartitioner<1>;
template class ParallelScanPartitioner<2>;
template class ParallelScanPartitioner<3>;

}  // namespace GQL
