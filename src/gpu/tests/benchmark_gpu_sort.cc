// benchmark_gpu_sort.cc — standalone sort microbenchmark (NOT GTest)
//
// Measures sorting throughput for Record<3> across multiple strategies
// (CPU sequential, CPU parallel/TBB, GPU radix) and data sizes.
//
// Output: CSV to stdout.  System info to stderr.
// Usage:  ./benchmark_gpu_sort 2>/dev/null | column -t -s,

#include "gpu/gpu_device.h"
#include "gpu/resource_planner.h"
#include "storage/index/record.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#ifdef HAS_TBB
#include <execution>
#endif

// ---------------------------------------------------------------------------
// GPU instrumented pipeline (defined in benchmark_gpu_kernels.cu)
// ---------------------------------------------------------------------------
#ifdef MDB_GPU_ENABLED
namespace mdb::gpu::bench {
struct GpuTimings {
    float    h2d_ms;
    float    sort_pass_ms[8];
    float    d2h_ms;
    uint32_t num_passes;
    bool     success;
};
GpuTimings gpu_sort_instrumented(
    const uint32_t* const*, uint32_t, uint64_t, uint32_t*);
} // namespace mdb::gpu::bench
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int WARMUP_ITERATIONS  = 1;
static constexpr int MEASURE_ITERATIONS = 5;

static const std::vector<uint64_t> SIZES = {
    100'000,
    500'000,
    1'000'000,
    5'000'000,
    10'000'000,
    50'000'000,
    100'000'000,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Generate N records with 3 fields using a deterministic PRNG.
static std::vector<Record<3>> generate_data(uint64_t N) {
    std::mt19937 rng(42);
    std::vector<Record<3>> data(N);
    for (uint64_t i = 0; i < N; ++i) {
        data[i][0] = rng() % (N / 10);
        data[i][1] = rng() % (N / 10);
        data[i][2] = rng() % N;
    }
    return data;
}

// Return median of a vector (mutates input).
static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Emit one CSV row.
static void emit_csv(uint64_t     N,
                     const char*  strategy,
                     const char*  label,
                     double       median_ms,
                     int          iterations) {
    std::printf("%llu,3,%s,%s,%.3f,%d\n",
                static_cast<unsigned long long>(N),
                strategy,
                label,
                median_ms,
                iterations);
}

// ---------------------------------------------------------------------------
// System info (stderr)
// ---------------------------------------------------------------------------
static void print_system_info() {
    auto res = mdb::gpu::detect_resources();

    std::fprintf(stderr, "=== Benchmark System Info ===\n");
    if (res.has_gpu) {
        std::fprintf(stderr, "GPU:  device %d, CC %d.%d, VRAM %.1f GB (%.1f GB free)\n",
                     res.gpu.device_id,
                     res.gpu.compute_capability / 10,
                     res.gpu.compute_capability % 10,
                     res.gpu.total_vram / 1e9,
                     res.gpu.free_vram  / 1e9);
    } else {
        std::fprintf(stderr, "GPU:  not available\n");
    }
    std::fprintf(stderr, "RAM:  %.1f GB available\n", res.ram_available / 1e9);
    std::fprintf(stderr, "TBB:  %s\n", res.has_tbb ? "yes" : "no");
    std::fprintf(stderr, "Warmup: %d, Measured: %d\n", WARMUP_ITERATIONS, MEASURE_ITERATIONS);
    std::fprintf(stderr, "============================\n\n");
}

// ---------------------------------------------------------------------------
// CPU Sequential benchmark
// ---------------------------------------------------------------------------
static void bench_cpu_sequential(uint64_t N,
                                 const std::vector<Record<3>>& original,
                                 const std::vector<Record<3>>& reference) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        auto copy = original;
        std::sort(copy.begin(), copy.end());
    }

    // Measure
    std::vector<double> times;
    times.reserve(MEASURE_ITERATIONS);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i) {
        auto copy = original;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::sort(copy.begin(), copy.end());
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);

        // Correctness check
        if (copy != reference) {
            std::fprintf(stderr, "ERROR: CPU_SEQUENTIAL produced wrong result for N=%llu\n",
                         static_cast<unsigned long long>(N));
        }
    }

    emit_csv(N, "CPU_SEQUENTIAL", "std::sort", median(times), MEASURE_ITERATIONS);
}

// ---------------------------------------------------------------------------
// CPU Parallel benchmark (TBB)
// ---------------------------------------------------------------------------
#ifdef HAS_TBB
static void bench_cpu_parallel(uint64_t N,
                               const std::vector<Record<3>>& original,
                               const std::vector<Record<3>>& reference) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        auto copy = original;
        std::sort(std::execution::par_unseq, copy.begin(), copy.end());
    }

    // Measure
    std::vector<double> times;
    times.reserve(MEASURE_ITERATIONS);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i) {
        auto copy = original;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::sort(std::execution::par_unseq, copy.begin(), copy.end());
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);

        // Correctness check
        if (copy != reference) {
            std::fprintf(stderr, "ERROR: CPU_PARALLEL produced wrong result for N=%llu\n",
                         static_cast<unsigned long long>(N));
        }
    }

    emit_csv(N, "CPU_PARALLEL", "par_unseq", median(times), MEASURE_ITERATIONS);
}
#endif

// ---------------------------------------------------------------------------
// GPU Full benchmark (per-phase timing)
// ---------------------------------------------------------------------------
#ifdef MDB_GPU_ENABLED
static void bench_gpu_full(uint64_t N,
                           const std::vector<Record<3>>& original,
                           const std::vector<Record<3>>& reference) {
    constexpr uint32_t NUM_FIELDS = 3;

    auto res = mdb::gpu::detect_resources();
    if (!res.has_gpu) {
        std::fprintf(stderr, "GPU_FULL: skipped (no GPU)\n");
        return;
    }

    // Check VRAM: need ~(NUM_FIELDS + 4) * N * 4 bytes + CUB temp
    size_t estimated_vram = static_cast<size_t>(NUM_FIELDS + 4) * N * sizeof(uint32_t)
                            + 64 * 1024 * 1024;  // 64 MB headroom for CUB temp
    if (estimated_vram > res.gpu.free_vram) {
        std::fprintf(stderr, "GPU_FULL: skipped N=%llu (need %.1f MB, have %.1f MB free)\n",
                     static_cast<unsigned long long>(N),
                     estimated_vram / 1e6,
                     res.gpu.free_vram / 1e6);
        return;
    }

    // CUB SortPairs takes int num_items
    if (N > static_cast<uint64_t>(INT_MAX)) {
        std::fprintf(stderr, "GPU_FULL: skipped N=%llu (exceeds CUB int limit)\n",
                     static_cast<unsigned long long>(N));
        return;
    }

    // Warmup
    for (int w = 0; w < WARMUP_ITERATIONS; ++w) {
        // AoS -> SoA
        std::vector<std::vector<uint32_t>> h_fields(NUM_FIELDS);
        for (uint32_t f = 0; f < NUM_FIELDS; f++) {
            h_fields[f].resize(N);
        }
        for (uint64_t i = 0; i < N; i++) {
            for (uint32_t f = 0; f < NUM_FIELDS; f++) {
                h_fields[f][i] = static_cast<uint32_t>(original[i][f]);
            }
        }
        std::vector<const uint32_t*> ptrs(NUM_FIELDS);
        for (uint32_t f = 0; f < NUM_FIELDS; f++) ptrs[f] = h_fields[f].data();

        std::vector<uint32_t> sorted_idx(N);
        mdb::gpu::bench::gpu_sort_instrumented(
            ptrs.data(), NUM_FIELDS, N, sorted_idx.data());
    }

    // Measure
    std::vector<double> times_aos2soa;
    std::vector<double> times_h2d;
    std::vector<double> times_sort_total;
    std::vector<std::vector<double>> times_sort_pass(NUM_FIELDS);
    std::vector<double> times_d2h;
    std::vector<double> times_scatter;
    std::vector<double> times_total;

    times_aos2soa.reserve(MEASURE_ITERATIONS);
    times_h2d.reserve(MEASURE_ITERATIONS);
    times_sort_total.reserve(MEASURE_ITERATIONS);
    for (auto& v : times_sort_pass) v.reserve(MEASURE_ITERATIONS);
    times_d2h.reserve(MEASURE_ITERATIONS);
    times_scatter.reserve(MEASURE_ITERATIONS);
    times_total.reserve(MEASURE_ITERATIONS);

    for (int iter = 0; iter < MEASURE_ITERATIONS; ++iter) {
        // Phase 1: AoS -> SoA (chrono, CPU operation)
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<uint32_t>> h_fields(NUM_FIELDS);
        for (uint32_t f = 0; f < NUM_FIELDS; f++) {
            h_fields[f].resize(N);
        }
        for (uint64_t i = 0; i < N; i++) {
            for (uint32_t f = 0; f < NUM_FIELDS; f++) {
                h_fields[f][i] = static_cast<uint32_t>(original[i][f]);
            }
        }
        std::vector<const uint32_t*> ptrs(NUM_FIELDS);
        for (uint32_t f = 0; f < NUM_FIELDS; f++) ptrs[f] = h_fields[f].data();

        auto t1 = std::chrono::high_resolution_clock::now();
        double aos2soa_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_aos2soa.push_back(aos2soa_ms);

        // Phase 2: GPU sort (cudaEvent timing inside)
        std::vector<uint32_t> sorted_idx(N);
        auto gpu_t = mdb::gpu::bench::gpu_sort_instrumented(
            ptrs.data(), NUM_FIELDS, N, sorted_idx.data());

        if (!gpu_t.success) {
            std::fprintf(stderr, "ERROR: GPU_FULL sort failed for N=%llu iter=%d\n",
                         static_cast<unsigned long long>(N), iter);
            return;
        }

        times_h2d.push_back(gpu_t.h2d_ms);
        double sort_sum = 0;
        for (uint32_t p = 0; p < gpu_t.num_passes; p++) {
            times_sort_pass[p].push_back(gpu_t.sort_pass_ms[p]);
            sort_sum += gpu_t.sort_pass_ms[p];
        }
        times_sort_total.push_back(sort_sum);
        times_d2h.push_back(gpu_t.d2h_ms);

        // Phase 3: Scatter (chrono, CPU operation)
        auto t2 = std::chrono::high_resolution_clock::now();

        std::vector<Record<3>> result(N);
        for (uint64_t i = 0; i < N; i++) {
            result[i] = original[sorted_idx[i]];
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        double scatter_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        times_scatter.push_back(scatter_ms);

        // Total = AoS2SoA + H2D + sort + D2H + scatter
        double total = aos2soa_ms + gpu_t.h2d_ms + sort_sum
                       + gpu_t.d2h_ms + scatter_ms;
        times_total.push_back(total);

        // Correctness check
        if (result != reference) {
            std::fprintf(stderr, "ERROR: GPU_FULL produced wrong result for N=%llu iter=%d\n",
                         static_cast<unsigned long long>(N), iter);
        }
    }

    // Emit per-phase CSV rows
    emit_csv(N, "GPU_FULL", "total",     median(times_total),      MEASURE_ITERATIONS);
    emit_csv(N, "GPU_FULL", "aos2soa",   median(times_aos2soa),    MEASURE_ITERATIONS);
    emit_csv(N, "GPU_FULL", "h2d",       median(times_h2d),        MEASURE_ITERATIONS);
    emit_csv(N, "GPU_FULL", "sort_total", median(times_sort_total), MEASURE_ITERATIONS);
    for (uint32_t p = 0; p < NUM_FIELDS; p++) {
        char label[32];
        std::snprintf(label, sizeof(label), "sort_pass_%u", p);
        emit_csv(N, "GPU_FULL", label, median(times_sort_pass[p]), MEASURE_ITERATIONS);
    }
    emit_csv(N, "GPU_FULL", "d2h",       median(times_d2h),        MEASURE_ITERATIONS);
    emit_csv(N, "GPU_FULL", "scatter",   median(times_scatter),    MEASURE_ITERATIONS);
}
#endif

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    print_system_info();

    // CSV header
    std::printf("records,fields,strategy,label,median_ms,iterations\n");

    for (uint64_t N : SIZES) {
        std::fprintf(stderr, "N = %llu ...\n", static_cast<unsigned long long>(N));

        // Generate data once per size
        auto original = generate_data(N);

        // Build reference (sorted copy for correctness verification)
        auto reference = original;
        std::sort(reference.begin(), reference.end());

        // CPU Sequential
        bench_cpu_sequential(N, original, reference);

        // CPU Parallel (TBB)
#ifdef HAS_TBB
        bench_cpu_parallel(N, original, reference);
#endif

        // GPU Full (per-phase timing)
#ifdef MDB_GPU_ENABLED
        bench_gpu_full(N, original, reference);
#endif

        std::fflush(stdout);
    }

    return 0;
}
