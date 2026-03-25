// gpu_filter.cu — Bitset filter kernel + CUB compaction for edge membership
//
// Given SoA edge arrays (from, to, edge_type) and bitset arrays for node/edge
// membership, computes per-edge pass/fail flags and uses cub::DeviceSelect::Flagged
// to compact surviving edges into contiguous output arrays.

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

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
// Kernel: per-edge bitset membership test
// ---------------------------------------------------------------------------
__global__ void filter_edges_kernel(
    const uint32_t* from_vals,       // [E] — SoA: from node VALUES
    const uint32_t* to_vals,         // [E] — SoA: to node VALUES
    const uint32_t* edge_vals,       // [E] — SoA: edge VALUES
    const uint64_t* node_bitset,     // [ceil(N/64)] — bit = node included
    const uint64_t* edge_bitset,     // [ceil(E_max/64)] — bit = edge type valid
    uint8_t*        flags,           // [E] — output: 1=pass, 0=exclude
    uint64_t        num_edges
) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_edges;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    {
        uint32_t fv = from_vals[i];
        uint32_t tv = to_vals[i];
        uint32_t ev = edge_vals[i];

        bool from_ok = (node_bitset[fv / 64] >> (fv % 64)) & 1;
        bool to_ok   = (node_bitset[tv / 64] >> (tv % 64)) & 1;
        bool edge_ok = (edge_bitset[ev / 64] >> (ev % 64)) & 1;

        flags[i] = (from_ok && to_ok && edge_ok) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Host-callable wrapper: filter + CUB compaction
// ---------------------------------------------------------------------------
bool filter_edges_gpu(
    const uint32_t* d_from, const uint32_t* d_to, const uint32_t* d_edge,
    const uint64_t* d_node_bitset, const uint64_t* d_edge_bitset,
    uint32_t* d_out_from, uint32_t* d_out_to, uint32_t* d_out_edge,
    uint64_t num_edges, uint64_t* num_surviving
) {
    // Handle empty input
    if (num_edges == 0) {
        *num_surviving = 0;
        return true;
    }

    // -----------------------------------------------------------------------
    // 1. Allocate flags array and compute per-edge pass/fail
    // -----------------------------------------------------------------------
    uint8_t* d_flags = nullptr;
    CHECK_CUDA(cudaMalloc(&d_flags, num_edges * sizeof(uint8_t)));

    constexpr int block_size = 256;
    int grid_size = static_cast<int>(
        std::min(static_cast<uint64_t>((num_edges + block_size - 1) / block_size),
                 static_cast<uint64_t>(65535)));

    filter_edges_kernel<<<grid_size, block_size>>>(
        d_from, d_to, d_edge,
        d_node_bitset, d_edge_bitset,
        d_flags, num_edges);
    CHECK_CUDA(cudaGetLastError());

    // -----------------------------------------------------------------------
    // 2. CUB DeviceSelect::Flagged — two-call pattern for each SoA column
    //    All three columns share the same flags; we compact each separately.
    // -----------------------------------------------------------------------
    // First: determine required temp storage size
    void*  d_temp     = nullptr;
    size_t temp_bytes = 0;

    // d_num_selected_out lives on device
    uint64_t* d_num_selected = nullptr;
    CHECK_CUDA(cudaMalloc(&d_num_selected, sizeof(uint64_t)));

    // Size query (nullptr for d_temp)
    CHECK_CUDA(cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        d_from, d_flags, d_out_from, d_num_selected,
        static_cast<int>(num_edges)));

    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));

    // Compact from_vals
    CHECK_CUDA(cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        d_from, d_flags, d_out_from, d_num_selected,
        static_cast<int>(num_edges)));

    // Compact to_vals (reuse same temp; size is sufficient)
    CHECK_CUDA(cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        d_to, d_flags, d_out_to, d_num_selected,
        static_cast<int>(num_edges)));

    // Compact edge_vals
    CHECK_CUDA(cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        d_edge, d_flags, d_out_edge, d_num_selected,
        static_cast<int>(num_edges)));

    // -----------------------------------------------------------------------
    // 3. Copy back the number of surviving edges
    // -----------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(num_surviving, d_num_selected, sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // 4. Cleanup
    // -----------------------------------------------------------------------
    cudaFree(d_flags);
    cudaFree(d_temp);
    cudaFree(d_num_selected);

    return true;
}

#undef CHECK_CUDA

} // namespace mdb::gpu
