#pragma once

// Spec C3 stage 1 (2026-05-07): single-worker async prefetcher.
//
// Hides the per-batch assemble cost (sample read + feature gather +
// host→device transfer, measured at 65.7% of train wall-time on
// papers100M Stage 0 baseline) behind the model forward+backward
// compute on the GPU. Mirrors the producer-consumer queue pattern from
// DiskGNN SIGMOD'25 §5.3 with queue size 2 (paper §6 default).
//
// Thread model:
//   producer thread   → prefetch(batch_id): enqueue assembly request
//   single worker     → BatchAssembler::assemble() in background
//   consumer thread   → next(): blocking dequeue of assembled MiniBatch
//
// Bounded total in-flight = queue_size. prefetch() blocks (backpressure)
// when the bound is reached, preventing memory blow-up if the consumer
// stalls.
//
// In Stage 1 the producer and consumer are the same training thread:
// it calls prefetch(b+1) and next() interleaved with model.forward / .backward.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <queue>
#include <thread>

#include "gnn/training/batch_assembler.h"
#include "gnn/training/mini_batch.h"

#ifdef ENABLE_CUDA_ASSEMBLER
#include <c10/cuda/CUDAStream.h>
#endif

namespace mdb::gnn {

class AsyncBatchPrefetcher {
public:
    /// `assembler` MUST outlive this object. `queue_size` bounds outstanding
    /// (queued + being assembled + completed-not-yet-consumed) batches.
    /// queue_size=2 matches DiskGNN SIGMOD'25 §6 ("the sizes of all shared
    /// queues are set to 2").
    ///
    /// `use_cuda_streams` (Stage 3, default false): when true, the worker
    /// thread acquires its own pool stream, runs assembly under
    /// CUDAStreamGuard, and records a CUDAEvent into MiniBatch.ready_event
    /// after each assembly. Consumers MUST call ready_event.block() on
    /// their training stream before reading GPU tensors.
    /// Has no effect when ENABLE_CUDA_ASSEMBLER is undefined.
    explicit AsyncBatchPrefetcher(BatchAssembler& assembler,
                                  size_t queue_size = 2,
                                  bool use_cuda_streams = false);

    /// Destructor calls shutdown() then joins the worker. Always succeeds.
    ~AsyncBatchPrefetcher();

    // Non-copyable, non-movable: owns a thread tied to internal queue state.
    AsyncBatchPrefetcher(const AsyncBatchPrefetcher&) = delete;
    AsyncBatchPrefetcher& operator=(const AsyncBatchPrefetcher&) = delete;
    AsyncBatchPrefetcher(AsyncBatchPrefetcher&&) = delete;
    AsyncBatchPrefetcher& operator=(AsyncBatchPrefetcher&&) = delete;

    /// Submit `batch_id` for async assembly. Blocks if in-flight count ==
    /// queue_size (backpressure). Throws std::runtime_error if shutdown
    /// was already requested (the worker is no longer accepting work).
    void prefetch(uint64_t batch_id);

    /// Block until the next assembled MiniBatch is ready, then return it.
    /// Returned MiniBatches are in submission order (FIFO).
    ///
    /// Throws std::runtime_error if there are no more batches to deliver
    /// (shutdown was called AND every prefetched batch has been consumed).
    /// If the worker thread caught an exception during assembly, that
    /// exception is rethrown here.
    MiniBatch next();

    /// Stop accepting new prefetch requests. The worker drains its req
    /// queue, assembles every still-queued batch into resp queue, then
    /// exits. Outstanding next() calls still receive their MiniBatches.
    /// Idempotent. Called automatically by destructor.
    void shutdown();

    /// Diagnostics. Cheap, lock-protected.
    size_t in_flight() const;
    bool   is_shutdown() const;

private:
    void worker_loop();

    BatchAssembler& assembler_;
    const size_t    queue_size_;
    const bool      use_cuda_streams_;

    mutable std::mutex      mu_;
    std::condition_variable space_cv_;  // notified when in_flight decreases
    std::condition_variable item_cv_;   // notified when resp_queue gains item

    std::queue<uint64_t>  req_queue_;
    std::queue<MiniBatch> resp_queue_;

    // First exception caught inside worker_loop. Rethrown by next().
    std::exception_ptr err_;

    // Total outstanding = req + assembling + resp. Bounded by queue_size_.
    size_t in_flight_count_ = 0;

    bool shutdown_requested_ = false;

    std::thread worker_;
};

} // namespace mdb::gnn
