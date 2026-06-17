#pragma once

/**
 * @file counting_semaphore.h
 * @brief Minimal C++17 counting semaphore (mutex + condvar + counter).
 *
 * The project compiles under -std=c++17 (CMAKE_CXX_STANDARD 17), so the
 * C++20 <semaphore> header (std::counting_semaphore) is unavailable. This
 * tiny helper provides just the acquire()/release() + RAII-guard surface
 * that RadixPartitionSort needs to bound how many Phase 2 workers submit to
 * the GPU concurrently (Task 5.1). It is intentionally not a general-purpose
 * synchronization primitive — only the operations exercised here are
 * implemented.
 */

#include <condition_variable>
#include <mutex>

namespace GQL {

/// A counting semaphore: at most `count` concurrent acquirers proceed; the
/// rest block in acquire() until a holder calls release().
class CountingSemaphore {
public:
    /// @param initial number of permits (>= 1; clamped to 1 by the caller).
    explicit CountingSemaphore(int initial) : count_(initial) {}

    CountingSemaphore(const CountingSemaphore&)            = delete;
    CountingSemaphore& operator=(const CountingSemaphore&) = delete;

    /// Block until a permit is available, then take it.
    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return count_ > 0; });
        --count_;
    }

    /// Return a permit and wake one waiter.
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++count_;
        }
        cv_.notify_one();
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    int                     count_;
};

/// RAII guard: acquires on construction, releases on destruction (every
/// scope exit, including exceptions). Used around the GPU-submission region
/// so a permit is never leaked on a throw or early return.
class CountingSemaphoreGuard {
public:
    explicit CountingSemaphoreGuard(CountingSemaphore& sem) : sem_(sem) {
        sem_.acquire();
    }
    ~CountingSemaphoreGuard() { sem_.release(); }

    CountingSemaphoreGuard(const CountingSemaphoreGuard&)            = delete;
    CountingSemaphoreGuard& operator=(const CountingSemaphoreGuard&) = delete;

private:
    CountingSemaphore& sem_;
};

}  // namespace GQL
