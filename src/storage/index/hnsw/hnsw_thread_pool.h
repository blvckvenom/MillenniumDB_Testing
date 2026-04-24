#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace HNSW {

/**
 * @brief Simple thread pool for parallel HNSW construction.
 *
 * Provides a parallel_for interface for data-parallel workloads.
 * Threads are created once and reused for all tasks.
 */
class ThreadPool {
public:
    /**
     * @brief Create a thread pool with the specified number of workers.
     * @param num_threads Number of worker threads to create
     */
    explicit ThreadPool(size_t num_threads);

    /**
     * @brief Destructor - signals workers to stop and joins all threads.
     */
    ~ThreadPool();

    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /**
     * @brief Execute a function in parallel over a range [start, end).
     *
     * The range is divided among worker threads. This call blocks until
     * all iterations complete.
     *
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @param fn Function to execute for each index: void(size_t index)
     */
    void parallel_for(size_t start, size_t end, std::function<void(size_t)> fn);

    /**
     * @brief Get the number of worker threads.
     */
    size_t num_threads() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex mutex_;
    std::condition_variable cv_;           // Signals workers when tasks available
    std::condition_variable done_cv_;      // Signals when all tasks complete

    std::atomic<bool> stop_ { false };
    std::atomic<size_t> active_ { 0 };     // Number of workers currently executing tasks
    std::atomic<size_t> pending_ { 0 };    // Number of tasks waiting + active

    void worker_thread();
};

} // namespace HNSW
