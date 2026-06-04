#pragma once

// Spec C3 stage 1 (2026-05-07): async batch prefetcher.
//
// Hides the per-batch assemble cost (sample read + feature gather +
// host→device transfer, measured at 65.7% of train wall-time on
// papers100M Stage 0 baseline) behind the model forward+backward
// compute on the GPU. Mirrors the producer-consumer queue pattern from
// DiskGNN SIGMOD'25 §5.3 with queue size 2 (paper §6 default).
//
// Thread model:
//   producer thread   → prefetch(batch_id): enqueue assembly request
//   N worker threads  → BatchAssembler::assemble() in background
//   consumer thread   → next(): blocking dequeue of assembled MiniBatch
//
// Bounded total in-flight = queue_size. prefetch() blocks (backpressure)
// when the bound is reached, preventing memory blow-up if the consumer
// stalls.
//
// In Stage 1 the producer and consumer are the same training thread:
// it calls prefetch(b+1) and next() interleaved with model.forward / .backward.
//
// Round 3B (2026-05-15): the prefetcher can now spawn `num_workers` worker
// threads. Order is preserved at the consumer: each prefetch() is assigned
// a monotonically-increasing "submission position", and next() retrieves
// the result for the next-expected position (waiting if the matching
// worker hasn't finished yet). Default num_workers=1 → byte-identical to
// the original single-worker behavior.
//
// Multi-worker thread-safety requirements (see also docs and runtime
// checks in TrainingLoop):
//   - SampleStorage::read_sample is thread-safe (mmap or fresh-ifstream).
//   - BatchAssembler.assemble_from_sample is reentrant: it builds local
//     unordered_maps and tensors; no shared mutable state.
//   - In FeatureMatrix fallback mode (BatchAssembler with FeatureMatrix
//     ctor) features are read concurrent-safe via extract_rows.
//   - In FourLevelStore full mode the load_batch_features path has
//     SHARED state that is NOT YET multi-worker-safe:
//        a) DirectIoReader holds 4 io_uring rings without inter-call
//           locking (header marks "NOT thread-safe").
//        b) pinned_ptr_ is a single shared host-pinned buffer; concurrent
//           memcpy → assembler->assemble() races on the buffer contents
//           because the CUDA kernel reads it asynchronously.
//   The training loop guards against this by enforcing num_workers==1
//   when the BatchAssembler is in FourLevelStore mode (see TrainingLoop
//   construction). num_workers>1 with FourLevelStore would silently
//   corrupt feature tensors (no test would catch it at the API surface).

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

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
    /// `use_cuda_streams` (Stage 3, default false): when true, each worker
    /// thread acquires its own pool stream, runs assembly under
    /// CUDAStreamGuard, and records a CUDAEvent into MiniBatch.ready_event
    /// after each assembly. Consumers MUST call ready_event.block() on
    /// their training stream before reading GPU tensors.
    /// Has no effect when ENABLE_CUDA_ASSEMBLER is undefined.
    ///
    /// `num_workers` (Round 3B, default 1): number of background worker
    /// threads consuming the request queue in parallel. The consumer
    /// re-orders results so next() returns batches in submission order.
    /// Values > queue_size are useless (in_flight is capped by queue_size).
    /// 0 is rejected. See the file-level comment for multi-worker
    /// thread-safety constraints (FourLevelStore is not safe).
    explicit AsyncBatchPrefetcher(BatchAssembler& assembler,
                                  size_t queue_size = 2,
                                  bool use_cuda_streams = false,
                                  unsigned num_workers = 1);

    /// Destructor calls shutdown() then joins all workers. Always succeeds.
    ~AsyncBatchPrefetcher();

    // Non-copyable, non-movable: owns threads tied to internal queue state.
    AsyncBatchPrefetcher(const AsyncBatchPrefetcher&) = delete;
    AsyncBatchPrefetcher& operator=(const AsyncBatchPrefetcher&) = delete;
    AsyncBatchPrefetcher(AsyncBatchPrefetcher&&) = delete;
    AsyncBatchPrefetcher& operator=(AsyncBatchPrefetcher&&) = delete;

    /// Submit `batch_id` for async assembly. Blocks if in-flight count ==
    /// queue_size (backpressure). Throws std::runtime_error if shutdown
    /// was already requested (the workers are no longer accepting work).
    void prefetch(uint64_t batch_id);

    /// Block until the next assembled MiniBatch is ready, then return it.
    /// Returned MiniBatches are in submission order (FIFO).
    ///
    /// Throws std::runtime_error if there are no more batches to deliver
    /// (shutdown was called AND every prefetched batch has been consumed).
    /// If a worker thread caught an exception while assembling the
    /// next-in-line batch, that exception is rethrown here.
    MiniBatch next();

    /// Stop accepting new prefetch requests. Workers drain their req
    /// queue, assemble every still-queued batch into resp, then exit.
    /// Outstanding next() calls still receive their MiniBatches.
    /// Idempotent. Called automatically by destructor.
    void shutdown();

    /// Diagnostics. Cheap, lock-protected.
    size_t in_flight() const;
    bool   is_shutdown() const;

    /// Number of worker threads spawned (post-construction; reflects any
    /// clamp applied during construction).
    unsigned num_workers() const { return static_cast<unsigned>(workers_.size()); }

private:
    // Round 3B-mw (2026-06-01): each worker thread is assigned a stable index
    // 0..num_workers-1 at spawn. It binds that index via
    // FourLevelStore::bind_worker_id() at thread start so the FourLevelStore
    // hot path selects this worker's private DirectIoReader + pinned buffer.
    void worker_loop(unsigned worker_idx);

    BatchAssembler& assembler_;
    const size_t    queue_size_;
    const bool      use_cuda_streams_;

    mutable std::mutex      mu_;
    std::condition_variable space_cv_;  // notified when in_flight decreases
    std::condition_variable item_cv_;   // notified when resp_map_ gains item OR new req available

    // Round 3B: submission-position-keyed request queue. Each entry is
    // (submission_position, batch_id). Workers pop in FIFO order off
    // req_queue_, perform assembly, and write the result into resp_map_
    // under the submission_position key. Consumer reads resp_map_ at
    // next_consume_pos_ in strict order.
    struct Request {
        uint64_t position;
        uint64_t batch_id;
    };
    std::queue<Request>                req_queue_;
    std::map<uint64_t, MiniBatch>      resp_map_;
    // Per-position exception slot. Populated by worker if assembly throws.
    std::map<uint64_t, std::exception_ptr> err_map_;

    // Monotonic position counters. next_submit_pos_ is incremented by
    // prefetch(); next_consume_pos_ is incremented by next().
    uint64_t next_submit_pos_  = 0;
    uint64_t next_consume_pos_ = 0;

    // Total outstanding = req + assembling + resp. Bounded by queue_size_.
    size_t in_flight_count_ = 0;

    bool shutdown_requested_ = false;

    std::vector<std::thread> workers_;
};

} // namespace mdb::gnn
