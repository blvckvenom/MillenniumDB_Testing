#pragma once

/**
 * @file external_record_sort.h
 * @brief External merge-sort for Record<N> types.
 *
 * Implements a memory-bounded external sort algorithm for B+tree index building.
 * Based on the proven ExternalEdgeSort pattern, generalized to work with any
 * Record<N> type (where Record<N> = std::array<uint64_t, N>).
 *
 * ## Algorithm
 *
 * 1. **Run Formation**: Sort each input spill file in memory
 *    - Read up to buffer_size bytes
 *    - Sort using std::sort with lexicographic comparison (std::array::operator<)
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
 * - Space: O(R × B) where R = run count, B = block size per run
 *   - For 100M records at 256 MB buffer: R ≈ 20 runs, peak ≈ 320 MB
 * - I/O: ~3 passes over data (collection, sort, merge)
 *
 * @see external_edge_sort.h for the EdgeAggregationRecord-specific version
 * @see streaming_record_buffer.h for record collection
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

#include "storage/index/record.h"
#include "storage/page/page.h"

#ifdef MDB_GPU_ENABLED
#include "gpu/sort/gpu_sort.h"
#endif

namespace GQL {

/**
 * @brief External merge-sort for Record<N> types.
 *
 * Sorts records from multiple spill files into a single sorted stream.
 * Uses K-way merge with bounded memory, enabling index building for
 * arbitrarily large datasets.
 *
 * ## Usage
 *
 * ```cpp
 * ExternalRecordSort<3> sorter(temp_dir, 256 * 1024 * 1024);
 *
 * // Add spill files from StreamingRecordBuffer
 * sorter.add_run("path/to/spill_0", record_count_0);
 * sorter.add_run("path/to/spill_1", record_count_1);
 *
 * // Optionally add memory records directly
 * sorter.add_memory_records(std::move(in_memory_vector));
 *
 * // Stream sorted output with inline deduplication
 * Record<3> prev;
 * bool has_prev = false;
 * sorter.stream_sorted([&](const Record<3>& rec) {
 *     if (!has_prev || rec != prev) {
 *         write_to_index(rec);
 *         prev = rec;
 *         has_prev = true;
 *     }
 * });
 * ```
 *
 * @tparam N Number of uint64_t values per record (typically 1, 2, or 3)
 */
template<std::size_t N>
class ExternalRecordSort {
public:
    /// @brief Record size in bytes (N × 8 bytes)
    static constexpr size_t RECORD_SIZE = N * sizeof(uint64_t);

    /// @brief Records that fit in one page (for efficient I/O)
    static constexpr size_t RECORDS_PER_PAGE = Page::SIZE / RECORD_SIZE;

    /// @brief Default merge buffer size (256 MB)
    static constexpr size_t DEFAULT_BUFFER_SIZE = 256 * 1024 * 1024;

    /**
     * @brief Constructs external sorter.
     *
     * @param temp_dir Directory for temporary sorted files
     * @param buffer_size Maximum memory for sort buffers (default 256 MB)
     */
    ExternalRecordSort(const std::string& temp_dir, size_t buffer_size = DEFAULT_BUFFER_SIZE)
        : temp_dir_(temp_dir)
        , buffer_size_(buffer_size)
        , total_records_(0)
    {
        // Ensure temp directory exists
        std::filesystem::create_directories(temp_dir_);
    }

    ~ExternalRecordSort() {
        cleanup();
    }

    // Non-copyable
    ExternalRecordSort(const ExternalRecordSort&) = delete;
    ExternalRecordSort& operator=(const ExternalRecordSort&) = delete;

    // Movable
    ExternalRecordSort(ExternalRecordSort&&) = default;
    ExternalRecordSort& operator=(ExternalRecordSort&&) = default;

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
     * @brief Adds memory records directly (avoids disk round-trip for small datasets).
     *
     * Takes ownership of the vector to avoid copying.
     *
     * @param records Vector of records to include in sort (moved)
     */
    void add_memory_records(std::vector<Record<N>>&& records) {
        total_records_ += records.size();
        memory_records_ = std::move(records);
    }

    /**
     * @brief Returns total number of records to sort.
     */
    size_t total_records() const { return total_records_; }

    /**
     * @brief Checks if all data fits in memory (no external sort needed).
     */
    bool fits_in_memory() const {
        return (total_records_ * RECORD_SIZE) <= buffer_size_;
    }

    /// Mutable access to in-memory records (sort_and_stream may move them)
    std::vector<Record<N>>& memory_records() { return memory_records_; }

    /// Run file paths for spill files
    const std::vector<std::string>& run_files() const { return run_files_; }

    /// Record counts per run file
    const std::vector<size_t>& run_record_counts() const { return run_record_counts_; }

    /**
     * @brief Streams sorted records to callback function.
     *
     * For in-memory data: sorts and iterates directly.
     * For disk data: performs external merge-sort with streaming output.
     *
     * Records are delivered in sorted order (lexicographic comparison).
     * Deduplication should be done in the callback if needed.
     *
     * @param callback Function called for each record in sorted order
     */
    template<typename Callback>
    void stream_sorted(Callback&& callback) {
        if (total_records_ == 0) {
            return;
        }

#ifdef MDB_GPU_ENABLED
        {
            auto resources = mdb::gpu::detect_resources();
            auto plan = mdb::gpu::plan_sort(total_records_, N, resources);

            if (plan.strategy != mdb::gpu::SortStrategy::EXTERNAL_SORT) {
                std::function<void(const Record<N>&)> gpu_callback =
                    [&callback](const Record<N>& r) { callback(r); };
                bool used = mdb::gpu::sort_and_stream<N>(
                    memory_records_, run_files_, run_record_counts_,
                    total_records_, gpu_callback, resources);
                if (used) return;
            }
        }
#endif

        // Existing fallback unchanged
        if (fits_in_memory()) {
            stream_in_memory(std::forward<Callback>(callback));
        } else {
            stream_external(std::forward<Callback>(callback));
        }
    }

private:
    /**
     * @brief Comparator for K-way merge priority queue.
     *
     * Min-heap: returns true if lhs > rhs (so smallest is at top).
     * Pair format: (record, run_index)
     */
    struct RunRecordComparator {
        bool operator()(
            const std::pair<Record<N>, size_t>& lhs,
            const std::pair<Record<N>, size_t>& rhs
        ) const {
            // For min-heap, return true if lhs should come AFTER rhs
            return !(lhs.first < rhs.first);
        }
    };

    /**
     * @brief In-memory sort and stream (for small datasets).
     */
    template<typename Callback>
    void stream_in_memory(Callback&& callback) {
        // Collect all records into memory
        std::vector<Record<N>> all_records;
        all_records.reserve(total_records_);

        // Add any pre-loaded memory records
        for (auto& rec : memory_records_) {
            all_records.push_back(std::move(rec));
        }
        memory_records_.clear();
        memory_records_.shrink_to_fit();

        // Read from run files
        for (size_t i = 0; i < run_files_.size(); ++i) {
            read_file_to_vector(run_files_[i], run_record_counts_[i], all_records);
        }

        // Sort in memory (lexicographic comparison via std::array::operator<)
#ifdef HAS_TBB
        std::sort(std::execution::par_unseq, all_records.begin(), all_records.end());
#else
        std::sort(all_records.begin(), all_records.end());
#endif

        // Stream to callback
        for (const auto& rec : all_records) {
            callback(rec);
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
#ifdef HAS_TBB
            std::sort(std::execution::par_unseq, memory_records_.begin(), memory_records_.end());
#else
            std::sort(memory_records_.begin(), memory_records_.end());
#endif
            std::string mem_run_path = temp_dir_ + "/sorted_run_mem";
            write_vector_to_file(memory_records_, mem_run_path);
            sorted_run_files.push_back(mem_run_path);
            memory_records_.clear();
            memory_records_.shrink_to_fit();
        }

        // Phase 2: K-way merge
        if (sorted_run_files.empty()) {
            return;
        } else if (sorted_run_files.size() == 1) {
            // Single run - just stream it (no merge needed)
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
     * @brief Sorts each input run file and writes to sorted run files.
     *
     * Reads each spill file entirely into memory (up to buffer_size),
     * sorts it, and writes back as a sorted run.
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
                    "Single run file (" + std::to_string(record_count) + " records, " +
                    std::to_string(record_count * RECORD_SIZE / (1024*1024)) + " MB) exceeds buffer size (" +
                    std::to_string(buffer_size_ / (1024*1024)) + " MB). Consider increasing buffer_size "
                    "or reducing the streaming buffer threshold."
                );
            }

            // Each thread gets its own buffer
            std::vector<Record<N>> buffer;
            buffer.reserve(record_count);
            read_file_to_vector(input_path, record_count, buffer);

            // Sort using parallel algorithm (TBB handles nested parallelism)
            std::sort(std::execution::par_unseq, buffer.begin(), buffer.end());

            write_vector_to_file(buffer, sorted_path);
        });
#else
        // Sequential fallback: reuse single buffer across iterations
        std::vector<Record<N>> buffer;
        buffer.reserve(max_records_in_buffer);

        for (size_t i = 0; i < run_files_.size(); ++i) {
            const std::string& input_path = run_files_[i];
            size_t record_count = run_record_counts_[i];
            const std::string& sorted_path = sorted_run_files[i];

            if (record_count <= max_records_in_buffer) {
                buffer.clear();
                read_file_to_vector(input_path, record_count, buffer);
                std::sort(buffer.begin(), buffer.end());
                write_vector_to_file(buffer, sorted_path);
            } else {
                throw std::runtime_error(
                    "Single run file (" + std::to_string(record_count) + " records, " +
                    std::to_string(record_count * RECORD_SIZE / (1024*1024)) + " MB) exceeds buffer size (" +
                    std::to_string(buffer_size_ / (1024*1024)) + " MB). Consider increasing buffer_size "
                    "or reducing the streaming buffer threshold."
                );
            }
        }
#endif
    }

    /**
     * @brief K-way merge of sorted run files using priority queue.
     *
     * Maintains one block of records from each run in memory.
     * Uses min-heap to always output the smallest record next.
     */
    template<typename Callback>
    void merge_runs(const std::vector<std::string>& sorted_run_files, Callback&& callback) {
        size_t num_runs = sorted_run_files.size();

        // Calculate block size per run (divide buffer evenly among runs)
        // Reserve some space for priority queue and overhead
        size_t block_records = (buffer_size_ / RECORD_SIZE) / (num_runs + 1);
        if (block_records == 0) {
            block_records = RECORDS_PER_PAGE;
        }

        // Open all run files and initialize buffers
        std::vector<std::ifstream> run_streams(num_runs);
        std::vector<std::vector<Record<N>>> run_buffers(num_runs);
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
            std::pair<Record<N>, size_t>,
            std::vector<std::pair<Record<N>, size_t>>,
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

        // Merge loop: pop minimum, output, refill if needed
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
     *
     * @param stream Input file stream (positioned at next unread record)
     * @param buffer Buffer to fill
     * @param remaining Records remaining in file (updated by reference)
     * @param max_records Maximum records to read
     */
    void refill_run_buffer(
        std::ifstream& stream,
        std::vector<Record<N>>& buffer,
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

        std::vector<Record<N>> buffer;
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
        std::vector<Record<N>>& vec
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
        const std::vector<Record<N>>& vec,
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
     *
     * Note: sorted_run files are cleaned in stream_external().
     * Original run files are managed by StreamingRecordBuffer.
     */
    void cleanup() {
        // Sorted run files are already cleaned up after streaming
        // Just clear our tracking vectors
        run_files_.clear();
        run_record_counts_.clear();
    }

    // Configuration
    std::string temp_dir_;
    size_t buffer_size_;
    size_t total_records_;

    // Run tracking
    std::vector<std::string> run_files_;
    std::vector<size_t> run_record_counts_;
    std::vector<Record<N>> memory_records_;
};

} // namespace GQL
