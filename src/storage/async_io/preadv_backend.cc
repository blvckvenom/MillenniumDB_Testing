/**
 * @file preadv_backend.cc
 * @brief Thread pool implementation using preadv/pwritev.
 */

#include "storage/async_io/preadv_backend.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sys/uio.h>
#include <unistd.h>

namespace Storage {

PreadvBackend::PreadvBackend(size_t num_threads, size_t queue_depth)
    : queue_depth_(queue_depth)
{
    // Start worker threads
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&PreadvBackend::worker_loop, this);
    }
}

PreadvBackend::~PreadvBackend() {
    // Signal shutdown
    shutdown_.store(true);

    // Wake up all workers
    work_cv_.notify_all();

    // Join all threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void PreadvBackend::submit_reads(const std::vector<IORequest>& requests) {
    if (requests.empty()) {
        return;
    }

    // Check capacity
    if (pending_count_.load() + requests.size() > queue_depth_) {
        throw std::runtime_error("preadv queue full - too many pending operations");
    }

    // Add to work queue
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        for (const auto& req : requests) {
            work_queue_.push({req, false});
        }
    }

    pending_count_.fetch_add(requests.size());

    // Wake up workers
    work_cv_.notify_all();
}

void PreadvBackend::submit_writes(const std::vector<IORequest>& requests) {
    if (requests.empty()) {
        return;
    }

    // Check capacity
    if (pending_count_.load() + requests.size() > queue_depth_) {
        throw std::runtime_error("preadv queue full - too many pending operations");
    }

    // Add to work queue
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        for (const auto& req : requests) {
            work_queue_.push({req, true});
        }
    }

    pending_count_.fetch_add(requests.size());

    // Wake up workers
    work_cv_.notify_all();
}

size_t PreadvBackend::wait_completions(CompletionCallback callback) {
    if (pending_count_.load() == 0) {
        return 0;
    }

    size_t completed = 0;

    // Wait for at least one completion
    {
        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this]() {
            return !completion_queue_.empty() || shutdown_.load();
        });

        // Process all available completions
        while (!completion_queue_.empty()) {
            auto completion = std::move(completion_queue_.front());
            completion_queue_.pop();
            lock.unlock();

            // Invoke callback outside the lock
            IORequest req_copy = completion.request;
            callback(&req_copy, completion.result);
            pending_count_.fetch_sub(1);
            completed++;

            lock.lock();
        }
    }

    return completed;
}

void PreadvBackend::drain() {
    // Wait for all pending operations
    while (pending_count_.load() > 0) {
        wait_completions([](IORequest*, ssize_t) {
            // Discard results
        });
    }
}

void PreadvBackend::worker_loop() {
    while (!shutdown_.load()) {
        WorkItem work;

        // Wait for work
        {
            std::unique_lock<std::mutex> lock(work_mutex_);
            work_cv_.wait(lock, [this]() {
                return !work_queue_.empty() || shutdown_.load();
            });

            if (shutdown_.load() && work_queue_.empty()) {
                return;  // Clean shutdown
            }

            if (work_queue_.empty()) {
                continue;
            }

            work = std::move(work_queue_.front());
            work_queue_.pop();
        }

        // Execute the I/O operation (outside the lock)
        ssize_t result = execute_io(work);

        // Post completion
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            completion_queue_.push({work.request, result});
        }
        completion_cv_.notify_one();
    }
}

ssize_t PreadvBackend::execute_io(const WorkItem& work) {
    struct iovec iov;
    iov.iov_base = work.request.buffer;
    iov.iov_len = work.request.size;

    ssize_t result;
    if (work.is_write) {
        result = pwritev(work.request.fd, &iov, 1, work.request.offset);
    } else {
        result = preadv(work.request.fd, &iov, 1, work.request.offset);
    }

    if (result < 0) {
        return -errno;  // Return negative errno on error
    }

    return result;
}

} // namespace Storage
