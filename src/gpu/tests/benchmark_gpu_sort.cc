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

        std::fflush(stdout);
    }

    return 0;
}
