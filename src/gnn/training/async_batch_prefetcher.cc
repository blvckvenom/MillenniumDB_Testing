#include "gnn/training/async_batch_prefetcher.h"

#include <optional>
#include <stdexcept>
#include <utility>

#ifdef ENABLE_CUDA_ASSEMBLER
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace mdb::gnn {

AsyncBatchPrefetcher::AsyncBatchPrefetcher(BatchAssembler& assembler,
                                           size_t queue_size,
                                           bool use_cuda_streams)
    : assembler_(assembler)
    , queue_size_(queue_size)
    , use_cuda_streams_(use_cuda_streams)
{
    if (queue_size_ == 0) {
        throw std::invalid_argument(
            "AsyncBatchPrefetcher: queue_size must be > 0");
    }
    worker_ = std::thread(&AsyncBatchPrefetcher::worker_loop, this);
}

AsyncBatchPrefetcher::~AsyncBatchPrefetcher() {
    shutdown();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void AsyncBatchPrefetcher::prefetch(uint64_t batch_id) {
    std::unique_lock<std::mutex> lk(mu_);
    space_cv_.wait(lk, [&] {
        return shutdown_requested_ || in_flight_count_ < queue_size_;
    });
    if (shutdown_requested_) {
        throw std::runtime_error(
            "AsyncBatchPrefetcher::prefetch: shutdown was already requested");
    }
    req_queue_.push(batch_id);
    ++in_flight_count_;
    item_cv_.notify_one();
}

MiniBatch AsyncBatchPrefetcher::next() {
    std::unique_lock<std::mutex> lk(mu_);
    item_cv_.wait(lk, [&] {
        // We can return when:
        //   - resp queue has an item, OR
        //   - the worker raised an exception we must propagate, OR
        //   - shutdown was requested AND nothing is in flight (caller drained).
        return !resp_queue_.empty()
            || (err_ && req_queue_.empty())
            || (shutdown_requested_ && in_flight_count_ == 0);
    });

    if (!resp_queue_.empty()) {
        MiniBatch out = std::move(resp_queue_.front());
        resp_queue_.pop();
        // in_flight_count_ stays — already decremented when worker pushed
        // to resp; actually we keep it incremented through resp lifetime so
        // that next() consuming reflects in space_cv_ properly.
        --in_flight_count_;
        space_cv_.notify_one();
        return out;
    }

    if (err_) {
        // Rethrow the first exception caught by the worker.
        std::exception_ptr to_throw = err_;
        err_ = nullptr;
        std::rethrow_exception(to_throw);
    }

    // Nothing left to deliver.
    throw std::runtime_error(
        "AsyncBatchPrefetcher::next: no more batches "
        "(shutdown after all consumed)");
}

void AsyncBatchPrefetcher::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (shutdown_requested_) return;
        shutdown_requested_ = true;
    }
    // Wake any waiters.
    space_cv_.notify_all();
    item_cv_.notify_all();
}

size_t AsyncBatchPrefetcher::in_flight() const {
    std::lock_guard<std::mutex> lk(mu_);
    return in_flight_count_;
}

bool AsyncBatchPrefetcher::is_shutdown() const {
    std::lock_guard<std::mutex> lk(mu_);
    return shutdown_requested_;
}

void AsyncBatchPrefetcher::worker_loop() {
    // Spec C3 stage 3: when use_cuda_streams_ is true, the worker keeps a
    // single pool stream for its lifetime. All assemblies run under
    // CUDAStreamGuard(worker_stream); we record a CUDAEvent into the
    // produced MiniBatch so the consumer can sync via event.block().
    //
    // Acquired lazily here (not in constructor) to keep CUDA dependencies
    // off the constructor path when use_cuda_streams_ is false.
#ifdef ENABLE_CUDA_ASSEMBLER
    std::optional<c10::cuda::CUDAStream> worker_stream;
    if (use_cuda_streams_) {
        worker_stream.emplace(c10::cuda::getStreamFromPool());
    }
#endif

    while (true) {
        uint64_t bid = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            item_cv_.wait(lk, [&] {
                return shutdown_requested_ || !req_queue_.empty();
            });
            if (req_queue_.empty()) {
                // Shutdown with empty queue: terminate.
                // Notify any waiting next() so it can return / throw.
                item_cv_.notify_all();
                return;
            }
            bid = req_queue_.front();
            req_queue_.pop();
        }

        // Assemble OUTSIDE the lock — this is the expensive work.
        try {
            MiniBatch batch;
#ifdef ENABLE_CUDA_ASSEMBLER
            if (use_cuda_streams_ && worker_stream.has_value()) {
                c10::cuda::CUDAStreamGuard guard(*worker_stream);
                batch = assembler_.assemble(bid);
                // Record the event AFTER assemble so the consumer's
                // event.block() correctly waits for assemble's GPU kernels
                // (assemble_kernel + any .to(device) issued inside).
                batch.ready_event.record(*worker_stream);
            } else {
                batch = assembler_.assemble(bid);
            }
#else
            batch = assembler_.assemble(bid);
#endif
            std::lock_guard<std::mutex> lk(mu_);
            resp_queue_.push(std::move(batch));
            item_cv_.notify_one();
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu_);
            if (!err_) err_ = std::current_exception();
            // Even on error we count this as no longer in-flight to avoid
            // deadlocking next(); the caller will see the exception.
            // Note: in_flight_count_ tracks (queued + assembling + resp);
            // we already decremented from req_queue_, so resp side is null.
            --in_flight_count_;
            space_cv_.notify_all();
            item_cv_.notify_all();
            // Continue the loop — subsequent batches may still succeed,
            // but next() will rethrow err_ before serving them.
        }
    }
}

} // namespace mdb::gnn
