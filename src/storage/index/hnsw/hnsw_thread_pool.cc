#include "hnsw_thread_pool.h"

#include <algorithm>

namespace HNSW {

ThreadPool::ThreadPool(size_t num_threads)
{
    // Ensure at least 1 thread
    num_threads = std::max(size_t(1), num_threads);

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool()
{
    // Signal all workers to stop
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();

    // Wait for all workers to finish
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_thread()
{
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait for task or stop signal
            cv_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });

            // Exit if stopped and no more tasks
            if (stop_ && tasks_.empty()) {
                return;
            }

            // Get next task
            task = std::move(tasks_.front());
            tasks_.pop();
            ++active_;
        }

        // Execute task outside lock — catch exceptions to prevent worker death
        try {
            task();
        } catch (const std::exception& e) {
            // Task exception is captured by std::packaged_task's future if applicable.
            // Log but don't propagate — worker must survive to process more tasks.
            (void)e;  // Avoid unused variable warning
        } catch (...) {
            // Unknown exception — worker survives anyway
        }

        // Signal completion
        --active_;
        --pending_;
        done_cv_.notify_all();
    }
}

void ThreadPool::parallel_for(size_t start, size_t end, std::function<void(size_t)> fn)
{
    if (start >= end) {
        return;
    }

    const size_t total = end - start;
    const size_t num_workers = workers_.size();

    // For small ranges, just run sequentially
    if (total <= num_workers || num_workers == 1) {
        for (size_t i = start; i < end; ++i) {
            fn(i);
        }
        return;
    }

    // Divide work into chunks
    const size_t chunk_size = (total + num_workers - 1) / num_workers;

    // Use a shared atomic counter for robust completion tracking
    // This avoids race conditions between task completion and wait notification
    auto remaining = std::make_shared<std::atomic<size_t>>(0);

    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (size_t t = 0; t < num_workers; ++t) {
            const size_t chunk_start = start + t * chunk_size;
            const size_t chunk_end = std::min(chunk_start + chunk_size, end);

            if (chunk_start >= end) {
                break;
            }

            ++(*remaining);  // Increment BEFORE pushing task
            ++pending_;
            tasks_.push([fn, chunk_start, chunk_end, remaining, this] {
                for (size_t i = chunk_start; i < chunk_end; ++i) {
                    fn(i);
                }
                // Decrement remaining AFTER task completes
                // Use fetch_sub to atomically decrement and get previous value
                if (remaining->fetch_sub(1) == 1) {
                    // This was the last task - notify waiters
                    done_cv_.notify_all();
                }
            });
        }
    }

    // Wake up all workers
    cv_.notify_all();

    // Wait for all tasks to complete using the shared counter
    while (remaining->load() > 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        // Double-check under lock to avoid spurious wakeups
        if (remaining->load() > 0) {
            done_cv_.wait_for(lock, std::chrono::milliseconds(1), [&remaining] {
                return remaining->load() == 0;
            });
        }
    }
}

} // namespace HNSW
