#pragma once

/**
 * @file external_edge_sort.h
 * @brief External merge-sort for EdgeAggregationRecords.
 *
 * Implements a memory-bounded external sort algorithm for edge aggregation.
 * Based on the proven DiskVector pattern from src/import/disk_vector.h,
 * adapted for EdgeAggregationRecord (40 bytes, 5 uint64_t fields).
 *
 * ## Algorithm
 *
 * 1. **Run Formation**: Sort each input spill file in memory
 *    - Read up to buffer_size bytes
 *    - Sort using std::sort with EdgeAggregationRecord::operator<
 *    - Write sorted "run" back to disk
 *
 * 2. **K-Way Merge**: Merge all runs using priority queue
 *    - One block per run in memory
 *    - Pop minimum, advance corresponding run
 *    - Stream output without full materialization
 *
 * ## Complexity
 *
 * - Time: O(N log N) where N = total records
 * - Space: O(R × B) where R = run count, B = block size
 *   - For 123M edges at 256 MB buffer: R ≈ 20 runs, peak ≈ 320 MB
 * - I/O: ~3 passes over data (collection, sort, merge)
 *
 * @see disk_vector.h for original pattern
 * @see edge_aggregation_record.h for record format
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

// Parallel execution for std::sort and std::for_each (requires TBB on GCC/Clang)
#ifdef HAS_TBB
#include <execution>
#endif

// Radix sort for O(N) complexity on integer keys (vs O(N log N) for comparison sort)
#include "ska_sort.hpp"

#include "graph_models/gql/projection/edge_aggregation_record.h"
#include "graph_models/gql/projection/parallel_merge.h"
#include "misc/available_ram.h"
#include "storage/async_io/async_io.h"
#include "storage/index/record.h"
#include "storage/page/page.h"

#ifdef MDB_GPU_ENABLED
#include "gpu/sort/gpu_sort.h"
#endif

namespace GQL {

/**
 * @brief High-performance sort for EdgeAggregationRecord using radix sort.
 *
 * Uses ska_sort (radix sort) which achieves O(N) time complexity for integer keys,
 * compared to O(N log N) for comparison-based sorts. For 61M records, this can be
 * 5-10× faster than std::sort.
 *
 * @param begin Iterator to first element
 * @param end Iterator past last element
 */
template<typename Iterator>
inline void radix_sort_edge_records(Iterator begin, Iterator end) {
    ska_sort(begin, end, [](const EdgeAggregationRecord& rec) {
        return rec.get_sort_key();
    });
}

/**
 * @brief Comparator for K-way merge priority queue.
 *
 * Min-heap: returns true if lhs > rhs (so smallest is at top).
 * Pair format: (record, run_index)
 */
struct RunRecordComparator {
    bool operator()(
        const std::pair<EdgeAggregationRecord, size_t>& lhs,
        const std::pair<EdgeAggregationRecord, size_t>& rhs
    ) const {
        // For min-heap, return true if lhs should come AFTER rhs
        return !(lhs.first < rhs.first);
    }
};

/**
 * @brief External merge-sort for EdgeAggregationRecords.
 *
 * Sorts records from multiple spill files into a single sorted stream.
 * Uses K-way merge with bounded memory.
 *
 * ## Usage
 *
 * ```cpp
 * ExternalEdgeSort sorter(temp_dir);  // adaptive default
 * // or: ExternalEdgeSort sorter(temp_dir, 512 * 1024 * 1024);  // explicit override
 *
 * // Add spill files from StreamingRecordBuffer
 * sorter.add_run("path/to/spill_0", record_count_0);
 * sorter.add_run("path/to/spill_1", record_count_1);
 *
 * // Stream sorted output
 * sorter.stream_sorted([](const EdgeAggregationRecord& rec) {
 *     // Process in sorted order
 * });
 * ```
 */
class ExternalEdgeSort {
public:
    /// @brief Record size in bytes (5 × 8 = 40 bytes)
    static constexpr size_t RECORD_SIZE = 5 * sizeof(uint64_t);

    /// @brief Records per buffer page
    static constexpr size_t RECORDS_PER_PAGE = Page::SIZE / RECORD_SIZE;

    /**
     * @brief Constructs external sorter.
     *
     * @param temp_dir Directory for temporary sorted files
     * @param buffer_size Maximum memory for sort buffers. Use 0 (default)
     *                    to enable the adaptive calculation from
     *                    /proc/meminfo MemAvailable. Any non-zero value is
     *                    treated as an explicit override (e.g., tests).
     */
    ExternalEdgeSort(const std::string& temp_dir, size_t buffer_size = 0)
        : temp_dir_(temp_dir)
        , buffer_size_(0)
        , block_size_(Page::SIZE * RECORD_SIZE)  // One page worth of records
        , total_records_(0)
    {
        if (buffer_size == 0) {
            AdaptiveBufferResult r = compute_adaptive_sort_buffer_result();
            buffer_size_ = r.bytes;
            const char* tag = (r.source == AdaptiveBufferResult::ADAPTIVE) ? "adaptive"
                            : (r.source == AdaptiveBufferResult::ENV)      ? "env"
                                                                           : "env_invalid";
            std::cout << "[sort] buffer=" << (buffer_size_ / (1ULL << 20))
                      << " MB (source=" << tag << ")\n";
        } else {
            buffer_size_ = buffer_size;
            std::cout << "[sort] buffer=" << (buffer_size_ / (1ULL << 20))
                      << " MB (source=explicit)\n";
        }
        // Ensure temp directory exists
        std::filesystem::create_directories(temp_dir_);
    }

    ~ExternalEdgeSort() {
        cleanup();
    }

    // Non-copyable
    ExternalEdgeSort(const ExternalEdgeSort&) = delete;
    ExternalEdgeSort& operator=(const ExternalEdgeSort&) = delete;

    /**
     * @brief Adds an unsorted run file to be sorted and merged.
     *
     * @param file_path Path to spill file (raw binary records)
     * @param record_count Number of records in the file
     */
    void add_run(const std::string& file_path, size_t record_count) {
        run_files_.push_back(file_path);
        run_record_counts_.push_back(record_count);
        total_records_ += record_count;
    }

    /**
     * @brief Adds memory records directly (for small datasets).
     *
     * @param records Vector of records to sort
     */
    void add_memory_records(std::vector<EdgeAggregationRecord>&& records) {
        memory_records_ = std::move(records);
        total_records_ += memory_records_.size();
    }

    /**
     * @brief Returns total number of records to sort.
     */
    size_t total_records() const { return total_records_; }

    /**
     * @brief Checks if all data fits in memory (no disk I/O needed).
     */
    bool fits_in_memory() const {
        return (total_records_ * RECORD_SIZE) <= buffer_size_;
    }

    /// Mutable access to in-memory records
    std::vector<EdgeAggregationRecord>& memory_records() { return memory_records_; }

    /// Run file paths for spill files
    const std::vector<std::string>& run_files() const { return run_files_; }

    /// Record counts per run file
    const std::vector<size_t>& run_record_counts() const { return run_record_counts_; }

    /**
     * @brief Enables parallel I/O mode for improved performance.
     *
     * When enabled, uses async I/O with prefetching for the merge phase.
     * Recommended for datasets with 10M+ records on NVMe storage.
     *
     * ## Performance Model
     *
     * Sequential mode (default):
     * ```
     * Time: |--read-0--|--process--|--read-1--|--process--|...
     * ```
     *
     * Parallel mode:
     * ```
     * Time: |--read-0--|--process+prefetch-1--|--process+prefetch-2--|...
     * ```
     *
     * Expected improvement: 2-4× for I/O-bound workloads.
     *
     * @param num_threads Number of I/O threads (for preadv fallback)
     * @param queue_depth Maximum concurrent I/O operations (for io_uring)
     */
    void set_parallel_mode(size_t num_threads = 4, size_t queue_depth = 32) {
        parallel_mode_ = true;
        async_io_ = Storage::AsyncIO::create(queue_depth, true);
        parallel_block_size_ = 8 * 1024 * 1024;  // 8 MB per run buffer
        (void)num_threads;  // Used internally by preadv backend

        std::cout << "[ExternalEdgeSort] Parallel mode enabled with "
                  << async_io_->backend_name() << " backend" << std::endl;
    }

    /**
     * @brief Checks if parallel mode is enabled.
     */
    bool is_parallel_mode() const { return parallel_mode_; }

    /**
     * @brief Streams sorted records to callback function.
     *
     * For in-memory data: sorts and iterates.
     * For disk data: performs external merge-sort with streaming output.
     * If parallel mode is enabled, uses async prefetching.
     *
     * @param callback Function called for each record in sorted order
     */
    template<typename Callback>
    void stream_sorted(Callback&& callback) {
        if (total_records_ == 0) {
            return;
        }

#ifdef MDB_GPU_ENABLED
        if (!std::getenv("MDB_FORCE_CPU_SORT")) {
            // Verify binary compatibility: EdgeAggregationRecord (5 contiguous uint64_t)
            // must match Record<5> (std::array<uint64_t, 5>) in size so that spill files
            // written as EdgeAggregationRecords can be read as Record<5>.
            static_assert(sizeof(EdgeAggregationRecord) == 5 * sizeof(uint64_t),
                          "EdgeAggregationRecord must be 5 contiguous uint64_t for GPU sort");

            auto resources = mdb::gpu::detect_resources();
            // Use 5 for num_fields since we sort Record<5> (all fields).
            // 5-pass sort is a valid refinement of the 3-field sort key:
            // records with the same (from, to, type) group are still adjacent,
            // just sub-sorted by edge_id and property_bits.
            auto plan = mdb::gpu::plan_sort(total_records_, 5, resources);

            if (plan.strategy != mdb::gpu::SortStrategy::EXTERNAL_SORT) {
                // Convert EdgeAggregationRecord -> Record<5>
                std::vector<Record<5>> record5_vec;
                record5_vec.reserve(memory_records_.size());
                for (auto& rec : memory_records_) {
                    record5_vec.push_back(rec.to_array());
                }
                memory_records_.clear();

                // Spill files are already binary-compatible: EdgeAggregationRecord
                // fields are laid out as 5 contiguous uint64_t, matching Record<5>

                std::function<void(const Record<5>&)> gpu_callback =
                    [&callback](const Record<5>& r) {
                        callback(EdgeAggregationRecord::from_array(r));
                    };

                bool used = mdb::gpu::sort_and_stream<5>(
                    record5_vec, run_files_, run_record_counts_,
                    total_records_, gpu_callback, resources);
                if (used) return;

                // GPU declined -- restore memory records for CPU fallback
                memory_records_.reserve(record5_vec.size());
                for (auto& r : record5_vec) {
                    memory_records_.push_back(EdgeAggregationRecord::from_array(r));
                }
            }
        }
#endif

        // Existing fallback unchanged
        if (fits_in_memory()) {
            stream_in_memory(std::forward<Callback>(callback));
        } else if (parallel_mode_) {
            stream_external_parallel(std::forward<Callback>(callback));
        } else {
            stream_external(std::forward<Callback>(callback));
        }
    }

    /**
     * @brief Streams sorted records using parallel I/O with prefetching.
     *
     * Uses ParallelMerge with double-buffering and async I/O.
     * Requires set_parallel_mode() to have been called first.
     *
     * @param callback Function called for each record in sorted order
     */
    template<typename Callback>
    void stream_sorted_parallel(Callback&& callback) {
        if (!parallel_mode_) {
            // Fallback to sequential if parallel not enabled
            stream_sorted(std::forward<Callback>(callback));
            return;
        }

        stream_sorted(std::forward<Callback>(callback));
    }

private:
    /**
     * @brief In-memory sort and stream (for small datasets).
     */
    template<typename Callback>
    void stream_in_memory(Callback&& callback) {
        // Collect all records into memory
        std::vector<EdgeAggregationRecord> all_records;
        all_records.reserve(total_records_);

        // Add any pre-loaded memory records
        for (auto& rec : memory_records_) {
            all_records.push_back(std::move(rec));
        }
        memory_records_.clear();

        // Read from run files
        for (size_t i = 0; i < run_files_.size(); ++i) {
            read_file_to_vector(run_files_[i], run_record_counts_[i], all_records);
        }

        // Sort in memory using radix sort (O(N) vs O(N log N) for comparison sort)
        radix_sort_edge_records(all_records.begin(), all_records.end());

        // Stream to callback
        for (const auto& rec : all_records) {
            callback(rec);
        }
    }

    /**
     * @brief External merge-sort with parallel I/O and prefetching.
     *
     * Uses ParallelMerge for double-buffered async I/O during the
     * merge phase, significantly reducing I/O wait time.
     */
    template<typename Callback>
    void stream_external_parallel(Callback&& callback) {
        // Phase 1: Sort each run file (same as sequential)
        std::vector<std::string> sorted_run_files;
        sort_run_files(sorted_run_files);

        // Handle pre-loaded memory records if any
        if (!memory_records_.empty()) {
            radix_sort_edge_records(memory_records_.begin(), memory_records_.end());
            std::string mem_run_path = temp_dir_ + "/sorted_run_mem";
            write_vector_to_file(memory_records_, mem_run_path);
            sorted_run_files.push_back(mem_run_path);
            memory_records_.clear();
        }

        if (sorted_run_files.empty()) {
            return;
        }

        // Phase 2: Parallel K-way merge with async prefetching
        if (sorted_run_files.size() == 1) {
            // Single run - just stream it (no merge needed)
            stream_single_file(sorted_run_files[0], callback);
        } else {
            // Build run descriptors for ParallelMerge
            std::vector<ParallelMerge::RunDescriptor> run_descriptors;
            run_descriptors.reserve(sorted_run_files.size());

            for (const auto& path : sorted_run_files) {
                // Get file size to calculate record count
                std::ifstream f(path, std::ios::binary | std::ios::ate);
                if (!f) {
                    throw std::runtime_error("Failed to open sorted run: " + path);
                }
                size_t record_count = static_cast<size_t>(f.tellg()) / RECORD_SIZE;
                run_descriptors.push_back({path, record_count});
            }

            std::cout << "[ExternalEdgeSort] Starting parallel merge of "
                      << run_descriptors.size() << " runs with async I/O" << std::endl;

            // Create and run parallel merge
            ParallelMerge merger(run_descriptors, async_io_.get(), parallel_block_size_);
            merger.stream_merged(std::forward<Callback>(callback));
        }

        // Cleanup sorted run files
        for (const auto& path : sorted_run_files) {
            std::filesystem::remove(path);
        }
    }

    /**
     * @brief External merge-sort with streaming output.
     *
     * Phase 1: Sort each run file in-place
     * Phase 2: K-way merge using priority queue
     */
    template<typename Callback>
    void stream_external(Callback&& callback) {
        // Phase 1: Sort each run file
        std::vector<std::string> sorted_run_files;
        sort_run_files(sorted_run_files);

        // Handle pre-loaded memory records if any
        if (!memory_records_.empty()) {
            radix_sort_edge_records(memory_records_.begin(), memory_records_.end());
            std::string mem_run_path = temp_dir_ + "/sorted_run_mem";
            write_vector_to_file(memory_records_, mem_run_path);
            sorted_run_files.push_back(mem_run_path);
            memory_records_.clear();
        }

        // Phase 2: K-way merge
        if (sorted_run_files.size() == 1) {
            // Single run - just stream it
            stream_single_file(sorted_run_files[0], callback);
        } else {
            // Multiple runs - merge
            merge_runs(sorted_run_files, callback);
        }

        // Cleanup sorted run files
        for (const auto& path : sorted_run_files) {
            std::filesystem::remove(path);
        }
    }

    /**
     * @brief Sorts each input run file in-place.
     *
     * When TBB is available, sorts all run files in parallel for significant
     * speedup on multi-core systems. Each parallel task gets its own buffer
     * to avoid contention.
     */
    void sort_run_files(std::vector<std::string>& sorted_run_files) {
        if (run_files_.empty()) {
            return;
        }

        size_t max_records_in_buffer = buffer_size_ / RECORD_SIZE;

        // Pre-allocate output paths (thread-safe since size is fixed)
        sorted_run_files.resize(run_files_.size());
        for (size_t i = 0; i < run_files_.size(); ++i) {
            sorted_run_files[i] = temp_dir_ + "/sorted_run_" + std::to_string(i);
        }

#ifdef HAS_TBB
        // Parallel: sort all run files concurrently
        // Each thread gets its own buffer to avoid contention
        std::vector<size_t> indices(run_files_.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            const std::string& input_path = run_files_[i];
            size_t record_count = run_record_counts_[i];
            const std::string& sorted_path = sorted_run_files[i];

            if (record_count > max_records_in_buffer) {
                throw std::runtime_error(
                    "Single run file exceeds buffer size. Consider increasing buffer_size "
                    "or implementing multi-pass external sort."
                );
            }

            // Each thread gets its own buffer
            std::vector<EdgeAggregationRecord> buffer;
            buffer.reserve(record_count);
            read_file_to_vector(input_path, record_count, buffer);

            // Sort using radix sort (O(N) complexity)
            radix_sort_edge_records(buffer.begin(), buffer.end());

            write_vector_to_file(buffer, sorted_path);
        });
#else
        // Sequential fallback: reuse single buffer across iterations
        std::vector<EdgeAggregationRecord> buffer;
        buffer.reserve(max_records_in_buffer);

        for (size_t i = 0; i < run_files_.size(); ++i) {
            const std::string& input_path = run_files_[i];
            size_t record_count = run_record_counts_[i];
            const std::string& sorted_path = sorted_run_files[i];

            if (record_count <= max_records_in_buffer) {
                buffer.clear();
                read_file_to_vector(input_path, record_count, buffer);
                radix_sort_edge_records(buffer.begin(), buffer.end());
                write_vector_to_file(buffer, sorted_path);
            } else {
                throw std::runtime_error(
                    "Single run file exceeds buffer size. Consider increasing buffer_size "
                    "or implementing multi-pass external sort."
                );
            }
        }
#endif
    }

    /**
     * @brief K-way merge of sorted run files.
     */
    template<typename Callback>
    void merge_runs(const std::vector<std::string>& sorted_run_files, Callback&& callback) {
        size_t num_runs = sorted_run_files.size();

        // Calculate block size per run (divide buffer evenly)
        size_t block_records = (buffer_size_ / RECORD_SIZE) / (num_runs + 1);  // +1 for output
        if (block_records == 0) block_records = RECORDS_PER_PAGE;

        // Open all run files
        std::vector<std::ifstream> run_streams(num_runs);
        std::vector<std::vector<EdgeAggregationRecord>> run_buffers(num_runs);
        std::vector<size_t> run_buffer_pos(num_runs, 0);
        std::vector<size_t> run_remaining(num_runs);

        for (size_t i = 0; i < num_runs; ++i) {
            run_streams[i].open(sorted_run_files[i], std::ios::binary);
            if (!run_streams[i]) {
                throw std::runtime_error("Failed to open sorted run: " + sorted_run_files[i]);
            }

            // Get file size to calculate record count
            run_streams[i].seekg(0, std::ios::end);
            run_remaining[i] = static_cast<size_t>(run_streams[i].tellg()) / RECORD_SIZE;
            run_streams[i].seekg(0, std::ios::beg);

            run_buffers[i].reserve(block_records);
        }

        // Priority queue for K-way merge: (record, run_index)
        std::priority_queue<
            std::pair<EdgeAggregationRecord, size_t>,
            std::vector<std::pair<EdgeAggregationRecord, size_t>>,
            RunRecordComparator
        > pq;

        // Load first block from each run and push first record to queue
        for (size_t i = 0; i < num_runs; ++i) {
            if (run_remaining[i] > 0) {
                refill_run_buffer(run_streams[i], run_buffers[i], run_remaining[i], block_records);
                run_buffer_pos[i] = 0;

                if (!run_buffers[i].empty()) {
                    pq.push({run_buffers[i][0], i});
                    run_buffer_pos[i] = 1;
                }
            }
        }

        // Merge loop
        while (!pq.empty()) {
            auto [record, run_idx] = pq.top();
            pq.pop();

            // Output this record
            callback(record);

            // Try to get next record from same run
            if (run_buffer_pos[run_idx] < run_buffers[run_idx].size()) {
                // More records in current buffer
                pq.push({run_buffers[run_idx][run_buffer_pos[run_idx]], run_idx});
                run_buffer_pos[run_idx]++;
            } else if (run_remaining[run_idx] > 0) {
                // Need to refill buffer from disk
                refill_run_buffer(run_streams[run_idx], run_buffers[run_idx],
                                  run_remaining[run_idx], block_records);
                run_buffer_pos[run_idx] = 0;

                if (!run_buffers[run_idx].empty()) {
                    pq.push({run_buffers[run_idx][0], run_idx});
                    run_buffer_pos[run_idx] = 1;
                }
            }
            // else: run exhausted, don't push anything
        }

        // Close all streams
        for (auto& stream : run_streams) {
            stream.close();
        }
    }

    /**
     * @brief Refills a run buffer from disk.
     */
    void refill_run_buffer(
        std::ifstream& stream,
        std::vector<EdgeAggregationRecord>& buffer,
        size_t& remaining,
        size_t max_records
    ) {
        buffer.clear();

        size_t to_read = std::min(remaining, max_records);
        if (to_read == 0) return;

        buffer.resize(to_read);

        // Read raw bytes and interpret as records
        stream.read(reinterpret_cast<char*>(buffer.data()), to_read * RECORD_SIZE);
        size_t actually_read = static_cast<size_t>(stream.gcount()) / RECORD_SIZE;

        buffer.resize(actually_read);
        remaining -= actually_read;
    }

    /**
     * @brief Streams a single sorted file (no merge needed).
     */
    template<typename Callback>
    void stream_single_file(const std::string& path, Callback&& callback) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("Failed to open sorted file: " + path);
        }

        std::vector<EdgeAggregationRecord> buffer;
        size_t block_records = buffer_size_ / RECORD_SIZE;
        buffer.reserve(block_records);

        while (stream) {
            buffer.resize(block_records);
            stream.read(reinterpret_cast<char*>(buffer.data()), block_records * RECORD_SIZE);
            size_t actually_read = static_cast<size_t>(stream.gcount()) / RECORD_SIZE;
            buffer.resize(actually_read);

            for (const auto& rec : buffer) {
                callback(rec);
            }

            if (actually_read < block_records) break;
        }
    }

    /**
     * @brief Reads a file into a vector of records.
     */
    void read_file_to_vector(
        const std::string& path,
        size_t record_count,
        std::vector<EdgeAggregationRecord>& vec
    ) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("Failed to open file for reading: " + path);
        }

        size_t start_size = vec.size();
        vec.resize(start_size + record_count);

        stream.read(
            reinterpret_cast<char*>(vec.data() + start_size),
            record_count * RECORD_SIZE
        );
    }

    /**
     * @brief Writes a vector of records to a file.
     */
    void write_vector_to_file(
        const std::vector<EdgeAggregationRecord>& vec,
        const std::string& path
    ) {
        std::ofstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }

        stream.write(
            reinterpret_cast<const char*>(vec.data()),
            vec.size() * RECORD_SIZE
        );
    }

    /**
     * @brief Cleans up temporary files.
     */
    void cleanup() {
        // Note: sorted_run files are cleaned in stream_external()
        // Original run files are managed by StreamingRecordBuffer
    }

    std::string temp_dir_;
    size_t buffer_size_;
    size_t block_size_;
    size_t total_records_;

    std::vector<std::string> run_files_;
    std::vector<size_t> run_record_counts_;
    std::vector<EdgeAggregationRecord> memory_records_;

    // Parallel mode configuration
    bool parallel_mode_ = false;                          ///< Whether parallel I/O is enabled
    std::unique_ptr<Storage::AsyncIO> async_io_;          ///< Async I/O backend
    size_t parallel_block_size_ = 8 * 1024 * 1024;        ///< Block size for parallel reads (8 MB)
};

} // namespace GQL
