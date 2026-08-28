// benchmark_gpu_kernels.cu — Instrumented GPU sort pipeline with cudaEvent timing
//
// Provides a host-callable function that runs the multi-pass CUB RadixSort
// pipeline with per-phase cudaEvent measurements for H2D, each sort pass,
// and D2H.  Compiled as CUDA, linked into the benchmark_gpu_sort executable.

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace mdb::gpu::bench {

struct GpuTimings {
    float    h2d_ms          = 0;
    float    sort_pass_ms[8] = {};
    float    d2h_ms          = 0;
    uint32_t num_passes      = 0;
    bool     success         = false;
};

// ---------------------------------------------------------------------------
// Gather kernel (benchmark-local copy to avoid cross-TU CUDA linking issues)
// ---------------------------------------------------------------------------
__global__ void bench_gather_keys(
    const uint32_t* field_values,
    const uint32_t* indices,
    uint32_t*       gathered_keys,
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
// Error-handling macro: logs and jumps to cleanup
// ---------------------------------------------------------------------------
#define BENCH_CHECK_CUDA(call)                                                \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            goto fail;                                                        \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Instrumented GPU sort pipeline
// ---------------------------------------------------------------------------
GpuTimings gpu_sort_instrumented(
    const uint32_t* const* h_fields,
    uint32_t               num_passes,
    uint64_t               num_items,
    uint32_t*              h_sorted_indices
) {
    GpuTimings t;
    t.num_passes = num_passes;

    if (num_items == 0 || num_passes == 0 ||
        num_items > static_cast<uint64_t>(INT_MAX))
    {
        return t;
    }

    size_t field_bytes = num_items * sizeof(uint32_t);

    // Device allocations
    std::vector<uint32_t*> d_fields(num_passes, nullptr);
    uint32_t* d_keys_in  = nullptr;
    uint32_t* d_keys_out = nullptr;
    uint32_t* d_vals_in  = nullptr;
    uint32_t* d_vals_out = nullptr;
    void*     d_temp     = nullptr;

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop  = nullptr;

    auto cleanup = [&]() {
        for (auto* p : d_fields) { if (p) cudaFree(p); }
        if (d_keys_in)  cudaFree(d_keys_in);
        if (d_keys_out) cudaFree(d_keys_out);
        if (d_vals_in)  cudaFree(d_vals_in);
        if (d_vals_out) cudaFree(d_vals_out);
        if (d_temp)     cudaFree(d_temp);
        if (ev_start)   cudaEventDestroy(ev_start);
        if (ev_stop)    cudaEventDestroy(ev_stop);
    };

    // Allocate device field arrays
    for (uint32_t f = 0; f < num_passes; f++) {
        BENCH_CHECK_CUDA(cudaMalloc(&d_fields[f], field_bytes));
    }
    BENCH_CHECK_CUDA(cudaMalloc(&d_keys_in,  field_bytes));
    BENCH_CHECK_CUDA(cudaMalloc(&d_keys_out, field_bytes));
    BENCH_CHECK_CUDA(cudaMalloc(&d_vals_in,  field_bytes));
    BENCH_CHECK_CUDA(cudaMalloc(&d_vals_out, field_bytes));

    // CUB temp storage (DoubleBuffer variant — O(P) temp)
    {
        cub::DoubleBuffer<uint32_t> db_keys_probe(d_keys_in, d_keys_out);
        cub::DoubleBuffer<uint32_t> db_vals_probe(d_vals_in, d_vals_out);
        size_t temp_bytes_probe = 0;
        BENCH_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes_probe, db_keys_probe, db_vals_probe,
            static_cast<int>(num_items)));
        BENCH_CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes_probe));
    }

    // Create timing events
    BENCH_CHECK_CUDA(cudaEventCreate(&ev_start));
    BENCH_CHECK_CUDA(cudaEventCreate(&ev_stop));

    // -------------------------------------------------------------------
    // Phase: H2D Transfer
    // -------------------------------------------------------------------
    {
        BENCH_CHECK_CUDA(cudaEventRecord(ev_start));
        for (uint32_t f = 0; f < num_passes; f++) {
            BENCH_CHECK_CUDA(cudaMemcpy(
                d_fields[f], h_fields[f], field_bytes, cudaMemcpyHostToDevice));
        }
        BENCH_CHECK_CUDA(cudaEventRecord(ev_stop));
        BENCH_CHECK_CUDA(cudaEventSynchronize(ev_stop));
        BENCH_CHECK_CUDA(cudaEventElapsedTime(&t.h2d_ms, ev_start, ev_stop));
    }

    // Kernel launch configuration for gather
    {
        int block = 256;
        int grid  = static_cast<int>(
            std::min(static_cast<uint64_t>((num_items + block - 1) / block),
                     static_cast<uint64_t>(65535)));

        cub::DoubleBuffer<uint32_t> db_keys(d_keys_in, d_keys_out);
        cub::DoubleBuffer<uint32_t> db_vals(d_vals_in, d_vals_out);

        // Re-query CUB temp size with the actual buffers
        size_t temp_bytes = 0;
        BENCH_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes, db_keys, db_vals, static_cast<int>(num_items)));

        // -------------------------------------------------------------------
        // Phase: Sort Passes (each individually timed)
        // -------------------------------------------------------------------
        for (uint32_t pass = 0; pass < num_passes; pass++) {
            uint32_t field_idx = num_passes - 1 - pass;

            if (pass == 0) {
                BENCH_CHECK_CUDA(cudaMemcpy(
                    d_keys_in, d_fields[field_idx], field_bytes,
                    cudaMemcpyDeviceToDevice));

                // Initialize iota indices on host and upload
                std::vector<uint32_t> h_iota(num_items);
                std::iota(h_iota.begin(), h_iota.end(), 0u);
                BENCH_CHECK_CUDA(cudaMemcpy(
                    d_vals_in, h_iota.data(), field_bytes,
                    cudaMemcpyHostToDevice));
            } else {
                bench_gather_keys<<<grid, block>>>(
                    d_fields[field_idx], db_vals.Current(),
                    d_keys_in, num_items);
                BENCH_CHECK_CUDA(cudaGetLastError());
                BENCH_CHECK_CUDA(cudaDeviceSynchronize());

                BENCH_CHECK_CUDA(cudaMemcpy(
                    d_vals_in, db_vals.Current(), field_bytes,
                    cudaMemcpyDeviceToDevice));
            }

            // Reset DoubleBuffer selectors: Current() == d_keys_in / d_vals_in
            db_keys.selector = 0;
            db_vals.selector = 0;

            // Timed CUB sort
            BENCH_CHECK_CUDA(cudaEventRecord(ev_start));
            BENCH_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                d_temp, temp_bytes, db_keys, db_vals,
                static_cast<int>(num_items)));
            BENCH_CHECK_CUDA(cudaEventRecord(ev_stop));
            BENCH_CHECK_CUDA(cudaEventSynchronize(ev_stop));
            BENCH_CHECK_CUDA(cudaEventElapsedTime(
                &t.sort_pass_ms[pass], ev_start, ev_stop));
        }

        // -------------------------------------------------------------------
        // Phase: D2H Transfer
        // -------------------------------------------------------------------
        BENCH_CHECK_CUDA(cudaEventRecord(ev_start));
        BENCH_CHECK_CUDA(cudaMemcpy(
            h_sorted_indices, db_vals.Current(), field_bytes,
            cudaMemcpyDeviceToHost));
        BENCH_CHECK_CUDA(cudaEventRecord(ev_stop));
        BENCH_CHECK_CUDA(cudaEventSynchronize(ev_stop));
        BENCH_CHECK_CUDA(cudaEventElapsedTime(&t.d2h_ms, ev_start, ev_stop));
    }

    cleanup();
    t.success = true;
    return t;

fail:
    cleanup();
    return t;
}

#undef BENCH_CHECK_CUDA

} // namespace mdb::gpu::bench
