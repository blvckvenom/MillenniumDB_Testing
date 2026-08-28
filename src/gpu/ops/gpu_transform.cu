// gpu_transform.cu — UNDIRECTED expand kernel
//
// Doubles directed edges for UNDIRECTED projections: each input edge
// (from, to, edge_id) produces both (from, to, edge_id) and (to, from, edge_id)
// in the output arrays.  Output layout is interleaved: forward at [2*i],
// reverse at [2*i+1].

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

namespace mdb::gpu {

// ---------------------------------------------------------------------------
// Error-handling macro: returns false instead of calling exit()
// ---------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            return false;                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: expand each directed edge into forward + reverse
// ---------------------------------------------------------------------------
__global__ void expand_undirected_kernel(
    const uint32_t* from_vals,       // [E] input
    const uint32_t* to_vals,         // [E] input
    const uint32_t* edge_vals,       // [E] input
    uint32_t*       out_from_vals,   // [2E] output
    uint32_t*       out_to_vals,     // [2E] output
    uint32_t*       out_edge_vals,   // [2E] output (duplicated)
    uint64_t        num_edges
) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_edges;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    {
        // Forward: (from, to)
        out_from_vals[i * 2]     = from_vals[i];
        out_to_vals[i * 2]       = to_vals[i];
        out_edge_vals[i * 2]     = edge_vals[i];

        // Reverse: (to, from) — same edge_val
        out_from_vals[i * 2 + 1] = to_vals[i];
        out_to_vals[i * 2 + 1]   = from_vals[i];
        out_edge_vals[i * 2 + 1] = edge_vals[i];
    }
}

// ---------------------------------------------------------------------------
// Host-callable wrapper
// ---------------------------------------------------------------------------
bool expand_undirected_gpu(
    const uint32_t* d_from, const uint32_t* d_to, const uint32_t* d_edge,
    uint32_t* d_out_from, uint32_t* d_out_to, uint32_t* d_out_edge,
    uint64_t num_edges
) {
    // Handle empty input
    if (num_edges == 0) {
        return true;
    }

    constexpr int block_size = 256;
    int grid_size = static_cast<int>(
        std::min(static_cast<uint64_t>((num_edges + block_size - 1) / block_size),
                 static_cast<uint64_t>(65535)));

    expand_undirected_kernel<<<grid_size, block_size>>>(
        d_from, d_to, d_edge,
        d_out_from, d_out_to, d_out_edge,
        num_edges);
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to ensure kernel completion before caller reads output
    CHECK_CUDA(cudaDeviceSynchronize());

    return true;
}

#undef CHECK_CUDA

} // namespace mdb::gpu
