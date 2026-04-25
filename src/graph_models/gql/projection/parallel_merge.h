#pragma once

/**
 * @file parallel_merge.h
 * @brief Parallel K-way merge with async prefetching.
 *
 * ## Algorithm Overview
 *
 * This implements a double-buffered K-way merge that overlaps I/O with
 * computation. While processing buffer A, we prefetch buffer B in the
 * background using async I/O.
 *
 * ## Performance Model
 *
 * Traditional blocking merge:
 * ```
 * Time: |--read-0--|--process--|--read-1--|--process--|--read-2--|...
 *       Total = N × (read_time + process_time)
 * ```
 *
 * Double-buffered async merge:
 * ```
 * Time: |--read-0--|--process+prefetch-1--|--process+prefetch-2--|...
 *       Total ≈ read_time + N × max(read_time, process_time)
 * ```
 *
 * For I/O-bound workloads (external sort), this can yield 2-4× speedup.
 *
 * ## Memory Model
 *
 * - 2 buffers per run: ~8 MB each (16 MB per run)
 * - For 20 runs: ~320 MB total (fits in L3/RAM hierarchy)
 * - Bounded memory regardless of data size
 *
 * ## Usage
 *
 * ```cpp
 * ParallelMerge merger(run_descriptors, async_io.get(), temp_dir);
 * merger.stream_merged([](const EdgeAggregationRecord& rec) {
 *     // Process in sorted order
 * });
 * ```
 *
 * @see external_edge_sort.h for integration point
 * @see async_io.h for I/O backend interface
 */

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <queue>
#include <string>
#include <unistd.h>
#include <vector>

#include "graph_models/gql/projection/edge_aggregation_record.h"
#include "storage/async_io/async_io.h"
#include "storage/page/page.h"

namespace GQL {

/**
 * @brief Comparator for priority queue (min-heap).
 *
 * Returns true if lhs > rhs so that smallest is at top of heap.
 */
struct ParallelMergeComparator {
    bool operator()(
        const std::pair<EdgeAggregationRecord, size_t>& lhs,
        const std::pair<EdgeAggregationRecord, size_t>& rhs
    ) const {
        return !(lhs.first < rhs.first);
    }
};

/**
 * @brief Parallel K-way merge with async prefetching.
 *
 * Maintains double-buffers per run for I/O overlap:
 * - While processing buffer A, prefetch into buffer B
 * - Swap when buffer A is exhausted
 */
class ParallelMerge {
public:
    /// @brief Record size in bytes (5 × 8 = 40 bytes)
    static constexpr size_t RECORD_SIZE = 5 * sizeof(uint64_t);

    /// @brief Default block size per run buffer (8 MB)
    static constexpr size_t DEFAULT_BLOCK_SIZE = 8 * 1024 * 1024;

    /// @brief Descriptor for a sorted run file.
    struct RunDescriptor {
        std::string file_path;    ///< Path to the sorted run file
        size_t total_records;     ///< Total number of records in the run
    };

    /**
     * @brief Constructs parallel merge for given run files.
     *
     * @param runs Vector of run descriptors
     * @param async_io Async I/O backend (may be nullptr for sync fallback)
     * @param block_size Bytes per buffer (default 8 MB)
     */
    ParallelMerge(
        const std::vector<RunDescriptor>& runs,
        Storage::AsyncIO* async_io,
        size_t block_size = DEFAULT_BLOCK_SIZE
    );

    /**
     * @brief Destructor - closes all file descriptors.
     */
    ~ParallelMerge();

    // Non-copyable
    ParallelMerge(const ParallelMerge&) = delete;
    ParallelMerge& operator=(const ParallelMerge&) = delete;

    /**
     * @brief Streams merged output to callback function.
     *
     * Records are delivered in sorted order according to
     * EdgeAggregationRecord::operator<.
     *
     * @param callback Function called for each record in sorted order
     */
    template<typename Callback>
    void stream_merged(Callback&& callback);

    /**
     * @brief Returns total records across all runs.
     */
    size_t total_records() const { return total_records_; }

    /**
     * @brief Returns number of runs being merged.
     */
    size_t num_runs() const { return runs_.size(); }

private:
    /**
     * @brief Per-run state for double-buffered reading.
     */
    struct RunState {
        int fd;                                        ///< File descriptor
        size_t records_remaining;                      ///< Records left to read
        off_t next_read_offset;                        ///< File offset for next read

        // Double buffers
        std::vector<EdgeAggregationRecord> buffer_a;
        std::vector<EdgeAggregationRecord> buffer_b;

        // Current buffer state
        std::vector<EdgeAggregationRecord>* active_buffer;
        std::vector<EdgeAggregationRecord>* prefetch_buffer;
        size_t buffer_pos;                             ///< Position in active buffer
        bool prefetch_pending;                         ///< Is a prefetch in flight?
        bool exhausted;                                ///< No more records in this run
    };

    /**
     * @brief Initializes all run states and issues initial prefetches.
     */
    void initialize_runs();

    /**
     * @brief Issues async prefetch for a run's prefetch buffer.
     */
    void issue_prefetch(size_t run_idx);

    /**
     * @brief Waits for prefetch to complete and swaps buffers.
     */
    void swap_buffers(size_t run_idx);

    /**
     * @brief Gets next record from a run, handling buffer switches.
     *
     * @return Pointer to record, or nullptr if run exhausted
     */
    EdgeAggregationRecord* get_next_record(size_t run_idx);

    /**
     * @brief Synchronous buffer fill (fallback when async_io is null).
     */
    void fill_buffer_sync(size_t run_idx, std::vector<EdgeAggregationRecord>& buffer);

    std::vector<RunDescriptor> runs_;
    std::vector<RunState> run_states_;
    Storage::AsyncIO* async_io_;
    size_t block_size_;
    size_t records_per_block_;
    size_t total_records_;
};

// Template implementation (must be in header)
template<typename Callback>
void ParallelMerge::stream_merged(Callback&& callback) {
    if (runs_.empty()) {
        return;
    }

    // Initialize all runs and issue initial prefetches
    initialize_runs();

    // Priority queue for K-way merge: (record, run_index)
    std::priority_queue<
        std::pair<EdgeAggregationRecord, size_t>,
        std::vector<std::pair<EdgeAggregationRecord, size_t>>,
        ParallelMergeComparator
    > pq;

    // Load first record from each non-exhausted run
    for (size_t i = 0; i < run_states_.size(); ++i) {
        EdgeAggregationRecord* rec = get_next_record(i);
        if (rec) {
            pq.push({*rec, i});
        }
    }

    // Main merge loop
    while (!pq.empty()) {
        auto [record, run_idx] = pq.top();
        pq.pop();

        // Output the record
        callback(record);

        // Get next record from the same run
        EdgeAggregationRecord* next_rec = get_next_record(run_idx);
        if (next_rec) {
            pq.push({*next_rec, run_idx});
        }
    }

    // Close all file descriptors
    for (auto& state : run_states_) {
        if (state.fd >= 0) {
            close(state.fd);
            state.fd = -1;
        }
    }
}

} // namespace GQL
