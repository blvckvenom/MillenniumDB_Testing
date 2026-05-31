#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <utility>

namespace mdb::gnn {

/**
 * @brief Bounded single-producer/single-consumer queue with exception
 *        propagation.
 *
 * Used to overlap I/O with compute in `create_reordered` (both chunked
 * and external-sort paths) and in the L4 packed_slim worker. A worker
 * thread runs the producer side (gather + pack a chunk), while the
 * main thread runs the consumer side (pwrite the chunk to disk). The
 * queue's bounded capacity provides natural backpressure: the producer
 * stops when the consumer can't keep up, and vice versa.
 *
 * Thread-safety: exactly one producer thread and one consumer thread.
 * Multiple producers or consumers are NOT supported (would need a
 * different data structure with stricter ordering guarantees).
 */
template <typename T>
class ChunkPipeline {
public:
    explicit ChunkPipeline(std::size_t capacity)
        : capacity_(capacity) {}

    ChunkPipeline(const ChunkPipeline&) = delete;
    ChunkPipeline& operator=(const ChunkPipeline&) = delete;

    /// Push an item; blocks if the queue is full.
    /// Throws if the consumer already saw an error.
    void push(T value) {
        std::unique_lock<std::mutex> lk(mu_);
        not_full_.wait(lk, [this] {
            return queue_.size() < capacity_ || closed_ || error_;
        });
        if (error_) std::rethrow_exception(error_);
        if (closed_) return;  // pushes after close() are silently dropped
        queue_.push_back(std::move(value));
        lk.unlock();
        not_empty_.notify_one();
    }

    /// Pop the next item. Returns nullopt when the queue is closed
    /// AND empty. Re-throws producer errors set via set_error() only
    /// after all queued items have been drained — callers always see
    /// pre-error data first, then the error, then nullopt.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait(lk, [this] {
            return !queue_.empty() || closed_ || error_;
        });
        if (!queue_.empty()) {
            T v = std::move(queue_.front());
            queue_.pop_front();
            lk.unlock();
            not_full_.notify_one();
            return v;
        }
        if (error_) std::rethrow_exception(error_);
        return std::nullopt;
    }

    /// Signal that no more items will be pushed.
    void close() {
        std::unique_lock<std::mutex> lk(mu_);
        closed_ = true;
        lk.unlock();
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    /// Record a producer-side exception. The next pop() rethrows it.
    void set_error(std::exception_ptr e) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!error_) error_ = std::move(e);
        closed_ = true;
        lk.unlock();
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    const std::size_t        capacity_;
    std::mutex               mu_;
    std::condition_variable  not_full_;
    std::condition_variable  not_empty_;
    std::deque<T>            queue_;
    bool                     closed_ = false;
    std::exception_ptr       error_  = nullptr;
};

}  // namespace mdb::gnn
