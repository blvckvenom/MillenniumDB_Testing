#pragma once

/**
 * @file preadv_backend.h
 * @brief Thread pool fallback using preadv/pwritev for async I/O.
 *
 * ## Design Rationale
 *
 * When io_uring is unavailable (macOS, older Linux, BSDs), this backend
 * provides async I/O semantics using a thread pool with vectored I/O.
 *
 * ## Performance Characteristics
 *
 * - **IOPS**: ~50K-100K with 4 threads on NVMe
 * - **Latency**: Higher than io_uring due to thread scheduling
 * - **Throughput**: 1-3 GB/s depending on thread count
 *
 * ## Thread Pool Architecture
 *
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                      Thread Pool                             │
 * │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
 * │  │ Worker 1│  │ Worker 2│  │ Worker 3│  │ Worker 4│         │
 * │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘         │
 * │       │            │            │            │               │
 * │       ▼            ▼            ▼            ▼               │
 * │  ┌──────────────────────────────────────────────────────┐   │
 * │  │               Work Queue (Thread-Safe)                │   │
 * │  │  [Request 0] [Request 1] [Request 2] ...             │   │
 * │  └──────────────────────────────────────────────────────┘   │
 * │                            │                                 │
 * │                            ▼                                 │
 * │  ┌──────────────────────────────────────────────────────┐   │
 * │  │             Completion Queue (Thread-Safe)            │   │
 * │  │  [(Request*, result), (Request*, result), ...]        │   │
 * │  └──────────────────────────────────────────────────────┘   │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * ## Why preadv Instead of pread?
 *
 * While this implementation uses single-buffer reads, preadv is used
 * because it's the most portable POSIX async-capable I/O primitive.
 * Future versions could batch adjacent reads into single preadv calls.
 *
 * @see async_io.h for interface documentation
 */

#include "storage/async_io/async_io.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Storage {

/**
 * @brief Thread pool fallback for async I/O using preadv/pwritev.
 *
 * Spawns worker threads that execute blocking I/O operations.
 * Provides async semantics via completion queue.
 */
class PreadvBackend : public AsyncIO {
public:
    /**
     * @brief Constructs preadv backend with thread pool.
     *
     * @param num_threads Number of worker threads (default: 4)
     * @param queue_depth Maximum pending operations (default: 32)
     */
    PreadvBackend(size_t num_threads = 4, size_t queue_depth = 32);

    /**
     * @brief Destructor - signals shutdown and joins all threads.
     */
    ~PreadvBackend() override;

    void submit_reads(const std::vector<IORequest>& requests) override;
    void submit_writes(const std::vector<IORequest>& requests) override;
    size_t wait_completions(CompletionCallback callback) override;
    void drain() override;

    size_t pending_count() const override { return pending_count_.load(); }
    size_t queue_depth() const override { return queue_depth_; }
    const char* backend_name() const override { return "preadv"; }

private:
    /**
     * @brief Work item for the thread pool.
     */
    struct WorkItem {
        IORequest request;
        bool is_write;
    };

    /**
     * @brief Completion result from a worker thread.
     */
    struct CompletionItem {
        IORequest request;
        ssize_t result;
    };

    /**
     * @brief Worker thread main loop.
     */
    void worker_loop();

    /**
     * @brief Executes a single I/O operation (blocking).
     */
    ssize_t execute_io(const WorkItem& work);

    // Thread pool
    std::vector<std::thread> workers_;
    std::atomic<bool> shutdown_{false};

    // Work queue (producer: submit_*, consumer: workers)
    std::queue<WorkItem> work_queue_;
    std::mutex work_mutex_;
    std::condition_variable work_cv_;

    // Completion queue (producer: workers, consumer: wait_completions)
    std::queue<CompletionItem> completion_queue_;
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;

    // Tracking
    std::atomic<size_t> pending_count_{0};
    size_t queue_depth_;
};

} // namespace Storage
