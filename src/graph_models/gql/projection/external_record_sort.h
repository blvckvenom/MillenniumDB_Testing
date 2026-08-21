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
 * ## Memory model — what `MDB_SORT_BUFFER_MB` bounds (Task 3.2)
 *
 * `buffer_size_` is the single memory knob, sourced from `MDB_SORT_BUFFER_MB`
 * (or the adaptive `MemAvailable * 3/4`, floored at 256 MB) via
 * `compute_adaptive_sort_buffer_result`. It bounds, BY CONSTRUCTION, each
 * distinct resident buffer used by the external sort:
 *
 *   (a) Per-run sort buffer — `sort_run_files` reads at most
 *       `max_records_in_buffer = buffer_size_ / RECORD_SIZE` records of one run
 *       into memory at a time (a run larger than this throws rather than
 *       silently over-allocating).
 *   (b) K-way merge blocks — `merge_runs` allocates
 *       `block_records = (buffer_size_ / RECORD_SIZE) / (num_runs + 1)` per run,
 *       so the sum of all merge blocks is ≤ `buffer_size_`.
 *   (c) In-memory remainder — `add_memory_records()` may receive a vector of
 *       ANY size. `stream_external` does NOT hold it fully resident: an
 *       oversized `memory_records_` is sorted-and-flushed in `≤ buffer_size_`
 *       chunks into additional run files (see `flush_memory_records_bounded_`),
 *       so the resident tail never exceeds one buffer. (The production
 *       `run_classic` driver only ever hands a ≤ 64 MB `STREAMING_BUFFER_THRESHOLD`
 *       remainder, well under the 256 MB floor, so it already took the single-
 *       chunk fast path before this guard existed — the guard makes the API
 *       contract robust to any direct caller, not just the disciplined one.)
 *
 * Orthogonally, the Phase-1 run-sort fan-out is bounded by a
 * `tbb::task_arena(hardware_concurrency()/2)` (see the MEMORY BOUND comment in
 * `sort_run_files`): without it, `std::for_each(par)` over hundreds of runs
 * keeps one ~run-sized buffer per in-flight task. Putting (a)+(b)+(c) and the
 * arena together, peak RSS ≈ `buffer_size_ + arena_workers × run_bytes`, both
 * terms gated by the two knobs — independent of total dataset size.
 *
 * @see external_edge_sort.h for the EdgeAggregationRecord-specific version
 * @see streaming_record_buffer.h for record collection
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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
#include <thread>
#include <tbb/task_arena.h>
#endif

#include "graph_models/gql/projection/spill_codec.h"
#include "misc/ablation_registry.h"
#include "misc/available_ram.h"
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
 * ExternalRecordSort<3> sorter(temp_dir);  // adaptive default
 * // or: ExternalRecordSort<3> sorter(temp_dir, 512 * 1024 * 1024);  // explicit override
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

    /**
     * @brief Constructs external sorter.
     *
     * @param temp_dir Directory for temporary sorted files
     * @param buffer_size Maximum memory for sort buffers. Use 0 (default)
     *                    to enable the adaptive calculation from
     *                    /proc/meminfo MemAvailable. Any non-zero value is
     *                    treated as an explicit override (e.g., tests).
     */
    ExternalRecordSort(const std::string& temp_dir, size_t buffer_size = 0)
        : temp_dir_(temp_dir)
        , buffer_size_(0)
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
     * @brief Returns the active memory budget (bytes).
     *
     * This is the single bound described in the file-header memory model:
     * the per-run sort buffer (a), the sum of K-way merge blocks (b), and the
     * resident in-memory remainder (c) are each ≤ this value.
     */
    size_t buffer_size() const { return buffer_size_; }

    /**
     * @brief Bytes currently held resident in heap-owned record buffers.
     *
     * Counts the `memory_records_` tail (the only long-lived resident buffer
     * outside the transient per-run / per-merge-block buffers, which are
     * already bounded by `buffer_size_` by construction). After
     * `stream_sorted` drains, `memory_records_` has been flushed in bounded
     * chunks, so this is 0; before draining it reflects whatever the caller
     * handed `add_memory_records`. Used by the regression test to assert the
     * "resident ≤ buffer" invariant holds post-sort.
     */
    size_t resident_memory_bytes() const {
        return memory_records_.size() * RECORD_SIZE;
    }

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
        // The DECISION below stays a presence test, unchanged: setting
        // MDB_FORCE_CPU_SORT=0 has always forced the CPU path, and flag() would
        // read that as "off" — a different switch. text() only puts the value
        // the process saw into the log, which is what was missing.
        static const std::string forced_cpu_sort =
            Ablation::text("MDB_FORCE_CPU_SORT", "(unset)");
        (void) forced_cpu_sort;
        if (!std::getenv("MDB_FORCE_CPU_SORT")) {
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
#else
        // No CUDA in this build: there is no GPU sort for this switch to force
        // off, so it is declared inert instead of passing for an effective arm.
        static const bool cpu_sort_switch_declared = [] {
            Ablation::inert("MDB_FORCE_CPU_SORT", "built without CUDA");
            return true;
        }();
        (void) cpu_sort_switch_declared;
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

        // Handle pre-loaded memory records if any. The in-memory remainder is
        // bounded by `buffer_size_` BY CONSTRUCTION here: if it exceeds one
        // buffer it is sorted and flushed in `<= buffer_size_` chunks into
        // additional run files instead of being held fully resident through the
        // merge (see the file-header memory model, item (c)).
        if (!memory_records_.empty()) {
            flush_memory_records_bounded_(sorted_run_files);
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
     * @brief Sorts and flushes `memory_records_` into bounded sorted runs.
     *
     * The remainder handed to `add_memory_records()` may be larger than one
     * sort buffer. To keep peak RSS bounded by `buffer_size_` (file-header
     * memory model item (c)), this:
     *
     *   - Fast path (remainder ≤ one buffer): sorts in place and writes a
     *     SINGLE run named `sorted_run_mem` — byte-identical to the historical
     *     behavior, so production builds (≤ 64 MB remainder) and the
     *     golden-compare are unaffected.
     *   - Bounded path (remainder > one buffer): partitions into
     *     `≤ max_records_in_buffer`-sized chunks, sorts each in its own buffer,
     *     and writes them as `sorted_run_mem_0 .. sorted_run_mem_k`. Each chunk
     *     is moved out of `memory_records_` before sorting, so the resident set
     *     never exceeds one buffer's worth of records. The merge step then
     *     folds these in like any other sorted run.
     *
     * In both cases `memory_records_` is emptied (and its capacity released)
     * before returning, so `resident_memory_bytes()` is 0 afterward.
     */
    void flush_memory_records_bounded_(std::vector<std::string>& sorted_run_files) {
        const size_t max_records_in_buffer =
            buffer_size_ / RECORD_SIZE > 0 ? buffer_size_ / RECORD_SIZE : 1;

        if (memory_records_.size() <= max_records_in_buffer) {
            // Fast path: single run, byte-identical to the legacy code path.
#ifdef HAS_TBB
            std::sort(std::execution::par_unseq,
                      memory_records_.begin(), memory_records_.end());
#else
            std::sort(memory_records_.begin(), memory_records_.end());
#endif
            std::string mem_run_path = temp_dir_ + "/sorted_run_mem";
            write_vector_to_file(memory_records_, mem_run_path);
            sorted_run_files.push_back(mem_run_path);
            memory_records_.clear();
            memory_records_.shrink_to_fit();
            return;
        }

        // Bounded path: split the oversized remainder into buffer-sized chunks.
        // Consume from the BACK so the source vector can release its tail each
        // iteration (POD Record<N> elements: a trailing resize() drops them and
        // a periodic shrink_to_fit() returns capacity to the allocator). Chunk
        // order is irrelevant — each chunk becomes an independent sorted run and
        // the K-way merge restores global order. Net resident records during the
        // loop ≈ remaining source tail (shrinking) + one ≤ buffer chunk.
        std::vector<Record<N>> chunk;
        chunk.reserve(max_records_in_buffer);
        size_t chunk_idx = 0;
        while (!memory_records_.empty()) {
            const size_t take =
                std::min(max_records_in_buffer, memory_records_.size());
            const size_t first = memory_records_.size() - take;

            chunk.assign(
                std::make_move_iterator(memory_records_.begin() + first),
                std::make_move_iterator(memory_records_.end()));

            // Release the consumed tail so the source shrinks as we go. resize()
            // alone keeps capacity; shrink_to_fit() periodically reclaims it
            // (every chunk would over-reallocate, so we only shrink when the
            // freed capacity is large relative to what remains).
            memory_records_.resize(first);
            if (memory_records_.capacity() > 2 * (first + max_records_in_buffer)) {
                memory_records_.shrink_to_fit();
            }

            // Sequential inner sort: each chunk is already ≤ one buffer, and a
            // nested par_unseq sort would spawn extra transient buffers that
            // break the single-chunk-resident bound.
            std::sort(chunk.begin(), chunk.end());

            std::string mem_run_path =
                temp_dir_ + "/sorted_run_mem_" + std::to_string(chunk_idx++);
            write_vector_to_file(chunk, mem_run_path);
            sorted_run_files.push_back(mem_run_path);

            chunk.clear();
        }
        memory_records_.clear();
        memory_records_.shrink_to_fit();
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

        // MEMORY BOUND (2026-06-15): cap the run-sort fan-out with a task_arena.
        // Without it, std::for_each(par) over hundreds of spill runs lets TBB keep
        // one ~run-sized buffer per IN-FLIGHT task and over-subscribes far beyond
        // the core count, so peak RSS = O(in-flight tasks x run bytes) -- NOT
        // bounded by buffer_size_/MDB_SORT_BUFFER_MB (which only gates the K-way
        // merge below). On papers100M (606 x ~64 MB runs) this spiked ~18 GB and
        // OOM-killed the build on a 30 GB shared host. Bounding the arena to
        // ~cores/2 caps it to a few hundred MB. Byte-identical: same sort, same
        // records, same writer -- only concurrency changes.
        const unsigned hwt_ = std::thread::hardware_concurrency();
        const int sort_arena_workers = static_cast<int>(hwt_ >= 2 ? hwt_ / 2 : 1);
        tbb::task_arena sort_arena(sort_arena_workers);
        sort_arena.execute([&]() {
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

            // Sequential inner sort: the outer task_arena already parallelizes
            // across runs; a nested par_unseq sort would spawn extra TBB tasks
            // (and transient buffers) inside each capped worker, breaking the bound.
            std::sort(buffer.begin(), buffer.end());

            write_vector_to_file(buffer, sorted_path);
        });
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
    // Reads a StreamingRecordBuffer spill file into the tail of `vec`.
    //
    // Uses SpillReader so the function is transparent to compression:
    //   * new compressed spills (LZ4, post-2026-04-20) — read via LZ4 stream
    //   * legacy headerless spills (pre-change)         — fall-through raw read
    //
    // Looping is required because SpillReader::read() may return short reads
    // at decompression block boundaries (in particular for LZ4). We stop
    // only when EOF is reached or we've filled the requested record_count.
    //
    // INVARIANT: the caller must pass record_count equal to the count
    // originally written by StreamingRecordBuffer (tracked in
    // records_per_spill_). This is not inferable from compressed file size.
    void read_file_to_vector(
        const std::string& path,
        size_t record_count,
        std::vector<Record<N>>& vec
    ) {
        if (record_count == 0) return;

        size_t start_size = vec.size();
        vec.resize(start_size + record_count);
        uint8_t* dst = reinterpret_cast<uint8_t*>(vec.data() + start_size);
        const size_t wanted_bytes = record_count * RECORD_SIZE;

        SpillReader reader(path);
        size_t got = 0;
        while (got < wanted_bytes && !reader.eof()) {
            size_t n = reader.read(dst + got, wanted_bytes - got);
            if (n == 0) break;
            got += n;
        }

        if (got != wanted_bytes) {
            throw std::runtime_error(
                "read_file_to_vector: short read from '" + path +
                "' (got " + std::to_string(got) +
                " bytes, expected " + std::to_string(wanted_bytes) + ")");
        }
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
