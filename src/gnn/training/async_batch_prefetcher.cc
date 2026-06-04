#include "gnn/training/async_batch_prefetcher.h"

#include "gnn/storage/four_level_store.h"  // FourLevelStore::bind_worker_id

#include <algorithm>
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
                                           bool use_cuda_streams,
                                           unsigned num_workers)
    : assembler_(assembler)
    , queue_size_(queue_size)
    , use_cuda_streams_(use_cuda_streams)
{
    if (queue_size_ == 0) {
        throw std::invalid_argument(
            "AsyncBatchPrefetcher: queue_size must be > 0");
    }
    if (num_workers == 0) {
        throw std::invalid_argument(
            "AsyncBatchPrefetcher: num_workers must be >= 1");
    }
    // Cap num_workers at queue_size: at most queue_size batches can be in
    // flight, so extra workers would always be idle on req_queue_ cv.
    const unsigned effective = std::min<unsigned>(
        num_workers, static_cast<unsigned>(queue_size_));

    workers_.reserve(effective);
    for (unsigned i = 0; i < effective; ++i) {
        workers_.emplace_back(&AsyncBatchPrefetcher::worker_loop, this, i);
    }
}

AsyncBatchPrefetcher::~AsyncBatchPrefetcher() {
    shutdown();
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
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
    const uint64_t pos = next_submit_pos_++;
    req_queue_.push(Request{pos, batch_id});
    ++in_flight_count_;
    // Wake exactly one worker — only one work item was added.
    item_cv_.notify_one();
}

MiniBatch AsyncBatchPrefetcher::next() {
    std::unique_lock<std::mutex> lk(mu_);
    // We can return when:
    //   - the result at next_consume_pos_ is ready (resp_map_ or err_map_), OR
    //   - shutdown was requested AND nothing is in flight (caller drained).
    item_cv_.wait(lk, [&] {
        if (shutdown_requested_ && in_flight_count_ == 0) return true;
        if (resp_map_.find(next_consume_pos_) != resp_map_.end()) return true;
        if (err_map_.find(next_consume_pos_) != err_map_.end()) return true;
        return false;
    });

    // Check the error slot for this position first — a worker that threw
    // already decremented in_flight_count_ on its way out.
    auto eit = err_map_.find(next_consume_pos_);
    if (eit != err_map_.end()) {
        std::exception_ptr ep = eit->second;
        err_map_.erase(eit);
        ++next_consume_pos_;
        // Wake any blocked prefetchers (the worker already decremented
        // in_flight_count_, but space_cv_ may not have been observed yet
        // by the producer; this is a defensive wake).
        space_cv_.notify_all();
        std::rethrow_exception(ep);
    }

    auto it = resp_map_.find(next_consume_pos_);
    if (it != resp_map_.end()) {
        MiniBatch out = std::move(it->second);
        resp_map_.erase(it);
        ++next_consume_pos_;
        --in_flight_count_;
        space_cv_.notify_one();
        return out;
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
    // Wake all waiters: any worker stuck on item_cv_ AND a consumer in next()
    // waiting for a position that will never arrive.
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

void AsyncBatchPrefetcher::worker_loop(unsigned worker_idx) {
    // Round 3B-mw (2026-06-01): bind this thread's worker id so the
    // FourLevelStore hot path routes to this worker's PRIVATE DirectIoReader
    // + pinned staging buffer (no cross-worker race on feature content).
    // Harmless in FeatureMatrix-fallback mode (the id is simply unread there).
    FourLevelStore::bind_worker_id(worker_idx);

    // Spec C3 stage 3: when use_cuda_streams_ is true, each worker keeps a
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
        Request req{0, 0};
        {
            std::unique_lock<std::mutex> lk(mu_);
            item_cv_.wait(lk, [&] {
                return shutdown_requested_ || !req_queue_.empty();
            });
            if (req_queue_.empty()) {
                // Shutdown with empty queue: terminate.
                // Notify any waiting next() so it can return / throw, and
                // any sibling worker so the entire pool drains.
                item_cv_.notify_all();
                return;
            }
            req = req_queue_.front();
            req_queue_.pop();
        }

        // Assemble OUTSIDE the lock — this is the expensive work.
        try {
            MiniBatch batch;
#ifdef ENABLE_CUDA_ASSEMBLER
            if (use_cuda_streams_ && worker_stream.has_value()) {
                c10::cuda::CUDAStreamGuard guard(*worker_stream);
                batch = assembler_.assemble(req.batch_id);
                // Record the event AFTER assemble so the consumer's
                // event.block() correctly waits for assemble's GPU kernels
                // (assemble_kernel + any .to(device) issued inside).
                batch.ready_event.record(*worker_stream);
            } else {
                batch = assembler_.assemble(req.batch_id);
            }
#else
            batch = assembler_.assemble(req.batch_id);
#endif
            std::lock_guard<std::mutex> lk(mu_);
            resp_map_.emplace(req.position, std::move(batch));
            // Wake all consumers — only the one waiting for this exact
            // position will proceed, but we don't know which condition
            // variable wait predicate is satisfied for each waiter, so
            // notify_all is the safe option (consumer count is 1 in
            // practice for the training loop, so cost is negligible).
            item_cv_.notify_all();
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu_);
            err_map_.emplace(req.position, std::current_exception());
            // Decrement in_flight on error so backpressure releases.
            --in_flight_count_;
            space_cv_.notify_all();
            item_cv_.notify_all();
            // Continue the loop — subsequent batches may still succeed,
            // but next() will rethrow this position's exception before
            // serving any successful batch at a LATER position (FIFO).
        }
    }
}

} // namespace mdb::gnn
