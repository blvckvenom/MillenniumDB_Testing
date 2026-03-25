// gpu_radix_sort.cu — Multi-pass stable CUB RadixSort for Record<N>
//
// N-pass sort using CUB DeviceRadixSort::SortPairs on uint32 keys.
// Each pass sorts by one field (least-significant first), leveraging CUB's
// stability guarantee to preserve the ordering from previous passes.

#include "gpu/sort/gpu_radix_sort.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

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
// Gather kernel: reorder keys by sorted indices for the next sort pass
// ---------------------------------------------------------------------------
__global__ void gather_keys_kernel(
    const uint32_t* field_values,   // [num_items] — values for current field
    const uint32_t* indices,        // [num_items] — permutation from prior pass
    uint32_t*       gathered_keys,  // [num_items] — output reordered keys
    uint64_t        num_items
) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_items;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    {
        gathered_keys[i] = field_values[indices[i]];
    }
}

// ---------------------------------------------------------------------------
// RAII helper to free a collection of device pointers on scope exit
// ---------------------------------------------------------------------------
struct DeviceMemory {
    std::vector<void*> ptrs;

    void track(void* p) { ptrs.push_back(p); }

    void free_all() {
        for (void* p : ptrs) {
            if (p) cudaFree(p);
        }
        ptrs.clear();
    }

    ~DeviceMemory() { free_all(); }
};

// ---------------------------------------------------------------------------
// Core implementation (not templated — works on raw pointers)
// ---------------------------------------------------------------------------
static bool gpu_radix_sort_impl(
    const uint32_t* const* h_fields,    // [num_passes] host field arrays
    uint32_t               num_passes,
    uint64_t               num_items,
    uint32_t*              h_sorted_indices  // [num_items] output
) {
    DeviceMemory dmem;

    // CUB DeviceRadixSort takes int num_items; guard against silent overflow on
    // large-VRAM GPUs (H100/H200 with 80+ GB) where the caller could legitimately
    // pass more than INT_MAX records. Return false to trigger CPU fallback.
    if (num_items > static_cast<uint64_t>(INT_MAX)) {
        fprintf(stderr, "GPU sort: num_items %llu exceeds CUB int limit (%d)\n",
                (unsigned long long)num_items, INT_MAX);
        return false;
    }

    // -----------------------------------------------------------------------
    // 1. Allocate device field arrays and upload
    // -----------------------------------------------------------------------
    std::vector<uint32_t*> d_fields(num_passes, nullptr);
    size_t field_bytes = num_items * sizeof(uint32_t);

    for (uint32_t f = 0; f < num_passes; f++) {
        CHECK_CUDA(cudaMalloc(&d_fields[f], field_bytes));
        dmem.track(d_fields[f]);
        CHECK_CUDA(cudaMemcpy(d_fields[f], h_fields[f], field_bytes,
                              cudaMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // 2. Allocate sort double-buffers (keys + values) and temp storage
    // -----------------------------------------------------------------------
    uint32_t* d_keys_in  = nullptr;
    uint32_t* d_keys_out = nullptr;
    uint32_t* d_vals_in  = nullptr;
    uint32_t* d_vals_out = nullptr;

    CHECK_CUDA(cudaMalloc(&d_keys_in,  field_bytes));  dmem.track(d_keys_in);
    CHECK_CUDA(cudaMalloc(&d_keys_out, field_bytes));  dmem.track(d_keys_out);
    CHECK_CUDA(cudaMalloc(&d_vals_in,  field_bytes));  dmem.track(d_vals_in);
    CHECK_CUDA(cudaMalloc(&d_vals_out, field_bytes));  dmem.track(d_vals_out);

    // CUB temp storage size (two-call pattern with nullptr)
    void*  d_temp      = nullptr;
    size_t temp_bytes  = 0;
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        static_cast<int>(num_items)));

    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
    dmem.track(d_temp);

    // Kernel launch configuration for gather
    int block_size = 256;
    int grid_size  = static_cast<int>(
        std::min(static_cast<uint64_t>((num_items + block_size - 1) / block_size),
                 static_cast<uint64_t>(65535)));

    // -----------------------------------------------------------------------
    // 3. N passes: least-significant field (N-1) to most-significant (0)
    // -----------------------------------------------------------------------
    for (uint32_t pass = 0; pass < num_passes; pass++) {
        // Field index: sort from least-significant to most-significant
        uint32_t field_idx = num_passes - 1 - pass;

        if (pass == 0) {
            // First pass: keys = field[N-1], values = iota(0..num_items-1)
            CHECK_CUDA(cudaMemcpy(d_keys_in, d_fields[field_idx], field_bytes,
                                  cudaMemcpyDeviceToDevice));

            // Initialize indices 0..num_items-1 on host and upload
            // (cheaper than a kernel for the first pass only)
            std::vector<uint32_t> h_iota(num_items);
            std::iota(h_iota.begin(), h_iota.end(), 0u);
            CHECK_CUDA(cudaMemcpy(d_vals_in, h_iota.data(), field_bytes,
                                  cudaMemcpyHostToDevice));
        } else {
            // Subsequent passes: gather keys for current field by sorted indices
            // The sorted indices from the previous pass are in d_vals_out
            gather_keys_kernel<<<grid_size, block_size>>>(
                d_fields[field_idx], d_vals_out, d_keys_in, num_items);
            CHECK_CUDA(cudaGetLastError());

            // Both ops on default stream: kernel launch and D2D memcpy are serialized by CUDA.
            // gather reads d_vals_out, memcpy reads d_vals_out — both reads, no race.

            // Sorted indices become the new values
            CHECK_CUDA(cudaMemcpy(d_vals_in, d_vals_out, field_bytes,
                                  cudaMemcpyDeviceToDevice));
        }

        // Sort pairs: keys_in -> keys_out, vals_in -> vals_out
        CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
            d_temp, temp_bytes,
            d_keys_in, d_keys_out,
            d_vals_in, d_vals_out,
            static_cast<int>(num_items)));
    }

    // -----------------------------------------------------------------------
    // 4. Download final sorted indices
    // -----------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(h_sorted_indices, d_vals_out, field_bytes,
                          cudaMemcpyDeviceToHost));

    return true;
}

// ---------------------------------------------------------------------------
// Templated public API
// ---------------------------------------------------------------------------
template<std::size_t N>
bool execute_gpu_radix_sort(
    std::vector<Record<N>>&                all_records,
    std::function<void(const Record<N>&)>& callback,
    uint32_t                               num_passes
) {
    uint64_t num_items = all_records.size();

    // Clamp passes to record width
    if (num_passes > N) {
        num_passes = static_cast<uint32_t>(N);
    }

    if (num_items == 0 || num_passes == 0) {
        // Nothing to sort — stream records as-is
        for (const auto& rec : all_records) {
            callback(rec);
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // Step 1: AoS -> SoA conversion on CPU
    //   Extract lower 32 bits of each uint64 field (ObjectId VALUE bits)
    // -----------------------------------------------------------------------
    std::vector<std::vector<uint32_t>> h_fields(num_passes);
    for (uint32_t f = 0; f < num_passes; f++) {
        h_fields[f].resize(num_items);
    }

    for (uint64_t i = 0; i < num_items; i++) {
        for (uint32_t f = 0; f < num_passes; f++) {
            h_fields[f][i] = static_cast<uint32_t>(
                all_records[i][f] & 0x00FFFFFFFFFFFFFFULL);
        }
    }

    // Build pointer array for the impl function
    std::vector<const uint32_t*> field_ptrs(num_passes);
    for (uint32_t f = 0; f < num_passes; f++) {
        field_ptrs[f] = h_fields[f].data();
    }

    // -----------------------------------------------------------------------
    // Step 2: GPU sort (returns sorted indices)
    // -----------------------------------------------------------------------
    std::vector<uint32_t> sorted_indices(num_items);

    bool ok = gpu_radix_sort_impl(
        field_ptrs.data(), num_passes, num_items, sorted_indices.data());

    if (!ok) {
        return false;  // CUDA error — caller should fall back to CPU
    }

    // -----------------------------------------------------------------------
    // Step 3: Scatter back — reorder original records by sorted indices
    //   and stream through the callback
    // -----------------------------------------------------------------------
    for (uint64_t i = 0; i < num_items; i++) {
        callback(all_records[sorted_indices[i]]);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Explicit instantiations for the three Record widths used by MillenniumDB
// ---------------------------------------------------------------------------
template bool execute_gpu_radix_sort<1>(
    std::vector<Record<1>>&,
    std::function<void(const Record<1>&)>&,
    uint32_t);

template bool execute_gpu_radix_sort<2>(
    std::vector<Record<2>>&,
    std::function<void(const Record<2>&)>&,
    uint32_t);

template bool execute_gpu_radix_sort<3>(
    std::vector<Record<3>>&,
    std::function<void(const Record<3>&)>&,
    uint32_t);

} // namespace mdb::gpu
