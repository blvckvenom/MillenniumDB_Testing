// gpu_membership.cu — Parallel binary-search membership kernel
//
// GPU acceleration for building the edge-keep bitmap (deciding which edges
// to retain based on whether both endpoints survive the node scan).
//
// Sister kernel to gpu_filter.cu's bitset filter: instead of testing
// membership against a flat uint64 bitset (which only works when node IDs
// are dense in [0, N_max)), this kernel tests membership against a
// SORTED uint64 array via per-thread binary search. That fits the
// projection's `collected_nodes_` representation, which stores full 64-bit
// ObjectIds (type-prefixed) and is sorted+uniqued by
// ProjectionStorage::finalize_node_scan() before precompute_edge_filter_
// runs.
//
// API: edge_keep_membership_gpu() takes:
//   - d_from[E], d_to[E]:       per-edge endpoint ObjectIds
//   - d_sorted_nodes[N]:        sorted uint64 ObjectIds (dedup'd)
//   - d_keep_flags[E] (output): 1 if BOTH endpoints in d_sorted_nodes,
//                               0 otherwise
//
// The host (`EdgeKeepBitmap` batcher in
// src/graph_models/gql/projection/edge_keep_bitmap_gpu.cc) sets bits in
// the per-orientation bitmap based on the keep_flags array and the
// captured `edge_id` values. Compaction into SoA output arrays (the job
// of CUB::DeviceSelect in gpu_filter.cu) is NOT needed here because the
// downstream consumers (`scan_edges_impl_serialized_`) read the bitmap
// keyed by edge counter, not by position in a compacted array.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

namespace mdb::gpu {

// ---------------------------------------------------------------------------
// Error-handling macro: returns false on failure (no CUDA exit())
// ---------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);               \
            return false;                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Device-side binary search on a sorted uint64 array. Returns true iff
// `key` is present in `arr[0..n)`. Standard textbook lower_bound, branch
// elision-friendly. Avg O(log n) global-memory loads.
// ---------------------------------------------------------------------------
__device__ inline bool device_contains(const uint64_t* __restrict__ arr,
                                       uint64_t                     n,
                                       uint64_t                     key)
{
    uint64_t lo = 0;
    uint64_t hi = n;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        uint64_t v   = arr[mid];
        if (v < key)       lo = mid + 1;
        else if (v > key)  hi = mid;
        else               return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Kernel: per-edge endpoint membership test.
// flags[i] = 1 iff (from_ids[i] in sorted_nodes) AND (to_ids[i] in sorted_nodes).
// ---------------------------------------------------------------------------
__global__ void edge_keep_membership_kernel(
    const uint64_t* __restrict__ from_ids,        // [E] full ObjectId.id values
    const uint64_t* __restrict__ to_ids,          // [E] full ObjectId.id values
    const uint64_t* __restrict__ sorted_nodes,    // [N] sorted uint64 ObjectIds
    uint64_t                     num_nodes,
    uint8_t*       __restrict__ keep_flags,       // [E] output (0 or 1)
    uint64_t                     num_edges)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_edges;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    {
        const uint64_t f = from_ids[i];
        const uint64_t t = to_ids[i];

        const bool from_ok = device_contains(sorted_nodes, num_nodes, f);
        // Short-circuit: skip the second search when the first already failed.
        const bool to_ok   = from_ok ? device_contains(sorted_nodes, num_nodes, t) : false;

        keep_flags[i] = (from_ok && to_ok) ? 1u : 0u;
    }
}

// ---------------------------------------------------------------------------
// Host-callable wrapper. All device pointers must be allocated by caller;
// keep_flags must point to E bytes of device memory.
//
// This wrapper synchronizes before returning so the caller can safely
// cudaMemcpy keep_flags back to host. For pipelined use, callers can
// instead launch the kernel directly (kernel definition is intentionally
// public-via-extern but we expose only the wrapper to keep the API
// surface small).
// ---------------------------------------------------------------------------
bool edge_keep_membership_gpu(
    const uint64_t* d_from,
    const uint64_t* d_to,
    const uint64_t* d_sorted_nodes,
    uint64_t        num_nodes,
    uint8_t*        d_keep_flags,
    uint64_t        num_edges)
{
    if (num_edges == 0) {
        return true;
    }

    constexpr int block_size = 256;
    int grid_size = static_cast<int>(
        // 65535 grid limit (well below CC 7.x+ limits but matches gpu_filter.cu's
        // conservative cap so a single consumer-GPU config doesn't dispatch
        // absurd grids; the for-loop in the kernel handles any leftover work).
        std::min(static_cast<uint64_t>((num_edges + block_size - 1) / block_size),
                 static_cast<uint64_t>(65535)));

    edge_keep_membership_kernel<<<grid_size, block_size>>>(
        d_from, d_to,
        d_sorted_nodes, num_nodes,
        d_keep_flags, num_edges);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return true;
}

#undef CHECK_CUDA

} // namespace mdb::gpu
