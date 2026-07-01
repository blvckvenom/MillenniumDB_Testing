#pragma once

// Async batch prefetcher — training pipeline overlap.
//
// Hides the per-batch assemble cost (sample read + feature gather +
// host→device transfer, measured at roughly two-thirds of train
// wall-time on a papers100M-scale baseline) behind the model
// forward+backward compute on the GPU.
// Mirrors the producer-consumer queue pattern from DiskGNN
// SIGMOD'25 §5.3 with queue size 2 (paper §6 default).
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
// In single-worker mode the producer and consumer are the same training
// thread: it calls prefetch(b+1) and next() interleaved with
// model.forward / .backward.
//
// Multi-worker extension: the prefetcher can spawn `num_workers` worker
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
//   - In FourLevelStore full mode (where each worker gets a private
//     DirectIoReader + pinned host buffer) num_workers>1 is safe ONLY IF the
//     per-worker IO slots were provisioned BEFORE this
//     prefetcher is constructed, via
//     BatchAssembler::prepare_feature_store_workers(num_workers) →
//     FourLevelStore::prepare_worker_io(num_workers). Each worker thread
//     then binds its id (bind_worker_id, see worker_loop) and the hot path
//     routes it to a PRIVATE DirectIoReader (io_uring rings are not
//     thread-safe) + a PRIVATE pinned staging buffer.
//     Worker ids BEYOND the provisioned slots silently fall back to the
//     SHARED primary reader/buffer (FourLevelStore::
//     l3_reader_for_current_worker_) and would race on feature content —
//     silent corruption, invisible at the API surface. TrainingLoop
//     provisions the slots up front and clamps to num_workers=1 when
//     provisioning fails; any new call site MUST do the same.

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
    /// `use_cuda_streams` (default false): when true, splits the assembly
    /// kernel and the model forward+backward onto separate CUDA streams to
    /// allow overlap. Each worker thread acquires its own pool stream, runs
    /// assembly under CUDAStreamGuard, and records a CUDAEvent into
    /// MiniBatch.ready_event after each assembly. Consumers MUST call
    /// ready_event.block() on their training stream before reading GPU tensors.
    /// Has no effect when ENABLE_CUDA_ASSEMBLER is undefined.
    ///
    /// `num_workers` (default 1): number of background worker threads
    /// consuming the request queue in parallel. The consumer re-orders
    /// results so next() returns batches in submission order. Values >
    /// queue_size are useless (in_flight is capped by queue_size). 0 is
    /// rejected. See the file-level comment for multi-worker thread-safety
    /// constraints (FourLevelStore mode requires
    /// prepare_feature_store_workers(num_workers) BEFORE construction).
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
    /// Rethrows a recorded worker-thread failure (see worker_failure_)
    /// instead of accepting work a dead pool may never assemble.
    void prefetch(uint64_t batch_id);

    /// Block until the next assembled MiniBatch is ready, then return it.
    /// Returned MiniBatches are in submission order (FIFO).
    ///
    /// Throws std::runtime_error if there are no more batches to deliver
    /// (shutdown was called AND every prefetched batch has been consumed).
    /// If a worker thread caught an exception while assembling the
    /// next-in-line batch, that exception is rethrown here. A worker-thread
    /// failure outside per-batch assembly (worker_failure_) is rethrown
    /// once no completed batch is deliverable.
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
    // Each worker thread is assigned a stable index 0..num_workers-1 at
    // spawn. It binds that index via FourLevelStore::bind_worker_id() at
    // thread start so the FourLevelStore hot path selects this worker's
    // private DirectIoReader + pinned buffer.
    //
    // worker_loop is a catch-all shell around worker_loop_impl: an exception
    // escaping a std::thread body calls std::terminate (whole-process death),
    // so anything thrown outside the per-request try is recorded into
    // worker_failure_ and rethrown by next()/prefetch() instead.
    void worker_loop(unsigned worker_idx);
    void worker_loop_impl(unsigned worker_idx);

    BatchAssembler& assembler_;
    const size_t    queue_size_;
    const bool      use_cuda_streams_;

    mutable std::mutex      mu_;
    std::condition_variable space_cv_;  // notified when in_flight decreases
    std::condition_variable item_cv_;   // notified when resp_map_ gains item OR new req available

    // Submission-position-keyed request queue for multi-worker ordering.
    // Each entry is (submission_position, batch_id). Workers pop in FIFO
    // order off req_queue_, perform assembly, and write the result into
    // resp_map_ under the submission_position key. Consumer reads resp_map_
    // at next_consume_pos_ in strict order.
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

    // First exception that escaped a worker thread OUTSIDE the per-request
    // try (e.g. thread-startup failure). Rethrown by next() when no result
    // is deliverable and by prefetch() before accepting new work, so the
    // consumer sees a clean error instead of a hang or std::terminate.
    std::exception_ptr worker_failure_;

    std::vector<std::thread> workers_;
};

} // namespace mdb::gnn
