#pragma once

// Spec C3 stage 3 (started 2026-05-08): CUDA stream helpers for pipeline
// overlap of assemble_kernel with model forward+backward on the GPU.
//
// Background (DiskGNN SIGMOD'25 §5.3, §6):
//   "we run the model trainer and feature assembler on separate CUDA streams
//    to improve GPU utilization"
//
// The pattern:
//   1. Worker thread gets `assemble_stream` from the c10 pool
//   2. assemble_kernel<<<..., assemble_stream>>>(...)
//   3. Worker records a `ready_event` on assemble_stream after launch
//   4. Main thread receives the MiniBatch + ready_event from the queue
//   5. Main thread blocks `train_stream` on ready_event (non-blocking on host)
//   6. Main thread launches model.forward() on train_stream
//   7. The two streams run concurrently on GPU — overlap appears when the
//      GPU has spare SMs (paper says SAGE/GAT graph samples don't saturate
//      GPU, so spare exists).
//
// LibTorch provides the primitives:
//   - c10::cuda::CUDAStream      (pooled wrapper around cudaStream_t)
//   - c10::cuda::CUDAStreamGuard (RAII — set + restore current stream)
//   - at::cuda::CUDAEvent        (RAII — owns cudaEvent_t, move-only)
//
// Files using this header should #ifdef ENABLE_CUDA_ASSEMBLER (the same
// compile flag we already use for assemble_kernel) since we depend on
// CUDA-specific symbols.

#ifdef ENABLE_CUDA_ASSEMBLER

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace mdb::gnn {

/// Bundle of (stream, event) used to signal "this stream's work is done"
/// from one thread to another. The event is created lazily on first record.
///
/// Usage (worker side, recording):
///   StreamSignal sig{c10::cuda::getStreamFromPool()};
///   {
///       c10::cuda::CUDAStreamGuard guard(sig.stream);
///       my_kernel<<<..., sig.stream.stream()>>>(...);
///       sig.event.record(sig.stream);
///   }
///   // sig is now movable to another thread.
///
/// Usage (consumer side, waiting):
///   auto train_stream = c10::cuda::getStreamFromPool();
///   sig.event.block(train_stream);   // train_stream waits without blocking host
///   {
///       c10::cuda::CUDAStreamGuard guard(train_stream);
///       output = model.forward(features);
///   }
struct StreamSignal {
    c10::cuda::CUDAStream stream;
    at::cuda::CUDAEvent   event;

    // CUDAStream is value type; CUDAEvent is move-only.
    StreamSignal(StreamSignal&&) noexcept = default;
    StreamSignal& operator=(StreamSignal&&) noexcept = default;
    StreamSignal(const StreamSignal&) = delete;
    StreamSignal& operator=(const StreamSignal&) = delete;

    explicit StreamSignal(c10::cuda::CUDAStream s) : stream(std::move(s)) {}
};

/// Convenience: acquire a fresh stream from the pool. The pool is global
/// + lock-protected; safe to call concurrently from worker + main threads.
/// @param high_priority  If true, request a high-priority stream so kernel
///                       launches preempt lower-priority work (useful for
///                       the train_stream so model forward isn't starved).
inline c10::cuda::CUDAStream
acquire_pool_stream(bool high_priority = false) {
    return c10::cuda::getStreamFromPool(high_priority);
}

} // namespace mdb::gnn

#endif // ENABLE_CUDA_ASSEMBLER
