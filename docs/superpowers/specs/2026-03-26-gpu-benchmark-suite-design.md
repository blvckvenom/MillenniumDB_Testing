# GPU Adaptive Scheduler — Benchmark Suite Design

**Date:** 2026-03-26
**Status:** Approved Design
**Depends on:** GPU Adaptive Scheduler (20 commits, implemented 2026-03-25/26)
**Output:** CSV benchmark data + theoretical analysis document

---

## 1. Purpose

Provide empirical performance measurements and theoretical analysis for the GPU Adaptive Scheduler, producing data for the thesis performance chapter.

**Three deliverables:**
1. **Sort Microbenchmark** — isolated sort timing across strategies and sizes
2. **graph_project Instrumentation** — phase breakdown of real projection pipeline
3. **Theoretical Analysis Document** — Work/Span/Amdahl/Gustafson with measured data

## 2. Component A: Sort Microbenchmark

### 2.1 File

`src/gpu/tests/benchmark_gpu_sort.cc` — standalone executable, NOT a GTest.

### 2.2 Measurement Protocol

```
For each size N in {100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000}:
  For Record<3> (from_node, to_node, edge_id):
    1. Generate N records with std::mt19937(seed=42)
       - field[0] = rng() % (N/10)    // ~10 edges per source node
       - field[1] = rng() % (N/10)    // ~10 edges per target node
       - field[2] = rng() % N         // unique edge ids

    2. Create reference copy, sort with std::sort → reference_sorted

    3. Measure CPU_SEQUENTIAL:
       - Copy data, measure std::sort
       - 1 warmup + 5 iterations, report median of 5
       - Verify output == reference_sorted

    4. Measure CPU_PARALLEL (if TBB available):
       - Copy data, measure std::sort(std::execution::par_unseq)
       - 1 warmup + 5 iterations, report median of 5
       - Verify output == reference_sorted

    5. Measure GPU_FULL (if GPU available AND data fits in VRAM):
       - Use an INSTRUMENTED sort pipeline (not sort_and_stream directly)
         The benchmark implements its own sort loop calling gpu_radix_sort_impl
         sub-phases directly, with timing hooks between each phase.
       - Sub-phase breakdown with CORRECT timing method per phase:
         a) aos_to_soa:    AoS → SoA conversion          [chrono — CPU operation]
         b) h2d_transfer:  cudaMemcpy H2D for field arrays [cudaEvent — GPU operation]
         c) sort_pass_1:   CUB DeviceRadixSort (field[2])  [cudaEvent — GPU operation]
         d) sort_pass_2:   gather + CUB (field[1])         [cudaEvent — GPU operation]
         e) sort_pass_3:   gather + CUB (field[0])         [cudaEvent — GPU operation]
         f) d2h_transfer:  cudaMemcpy D2H for indices      [cudaEvent — GPU operation]
         g) scatter_back:  reorder records by indices       [chrono — CPU operation]
         h) total:         end-to-end (chrono wall-clock)  [chrono — overall]
       - 1 warmup + 5 iterations, report median of 5
       - Verify output == reference_sorted

    6. Measure GPU_CHUNKED (only for N >= 10M):
       - Force chunked mode with fake SystemResources (free_vram = 20 MB)
       - Additional metric: merge_time (K-way merge on CPU) [chrono]
       - 1 warmup + 5 iterations, report median of 5
       - Verify output == reference_sorted
```

### 2.3 GPU Timing

**Two timing methods** — chosen per sub-phase based on where the work runs:

**GPU operations (h2d, sort passes, d2h):** Use `cudaEvent` pairs for sub-millisecond precision:
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// ... GPU operation ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

**CPU operations (aos_to_soa, scatter_back, total):** Use `std::chrono::high_resolution_clock`:
```cpp
auto t0 = std::chrono::high_resolution_clock::now();
// ... CPU operation ...
auto t1 = std::chrono::high_resolution_clock::now();
double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
```

IMPORTANT: Do NOT use cudaEvent for CPU-only phases — cudaEventRecord timestamps the GPU stream, not CPU execution. cudaEvent around a CPU phase would report ~0 ms.

### 2.4 Output Format

CSV to stdout (redirect to file):

```csv
size,width,strategy,metric,value_ms,iteration
100000,3,CPU_SEQUENTIAL,total,12.5,1
100000,3,CPU_SEQUENTIAL,total,12.3,2
100000,3,CPU_SEQUENTIAL,total,12.4,3
100000,3,GPU_FULL,aos_to_soa,0.8,1
100000,3,GPU_FULL,h2d_transfer,0.3,1
100000,3,GPU_FULL,sort_pass_1,0.5,1
100000,3,GPU_FULL,sort_pass_2,0.6,1
100000,3,GPU_FULL,sort_pass_3,0.5,1
100000,3,GPU_FULL,d2h_transfer,0.2,1
100000,3,GPU_FULL,scatter_back,0.4,1
100000,3,GPU_FULL,total,3.3,1
```

### 2.5 Implementation Constraints

- NO dependency on MillenniumDB beyond `storage/index/record.h` and `src/gpu/`
- Data generation is self-contained (no DB import needed)
- Correctness verification after EVERY measurement (fail-fast if sort is wrong)
- cudaEvent timing requires CUDA; for CPU-only builds, only CPU strategies are measured
- CMake: `add_executable(benchmark_gpu_sort ...)` NOT registered with `add_test` (manual run only)
- Build alongside mdb_gpu: `target_link_libraries(benchmark_gpu_sort mdb_gpu)`
- The benchmark implements its own instrumented sort pipeline (AoS→SoA → upload → per-pass sort → download → scatter) rather than calling `sort_and_stream()`, to insert timing hooks between phases. This requires exposing lower-level functions from `gpu_radix_sort.cu` via a benchmark-specific header or inlining the pipeline logic. See Section 2.7 for the approach.

### 2.6 Sizes and Expected Behavior on GTX 1660 Super

| Size N | GPU VRAM needed (4N+16 formula) | CPU RAM needed (N×3×8) | GPU_FULL? | GPU_CHUNKED? |
|--------|--------------------------------|----------------------|-----------|-------------|
| 100K | 2.8 MB | 2.4 MB | Yes | Skip |
| 500K | 14 MB | 12 MB | Yes | Skip |
| 1M | 28 MB | 24 MB | Yes | Skip |
| 5M | 140 MB | 120 MB | Yes | Skip |
| 10M | 280 MB | 240 MB | Yes | Force test |
| 50M | 1.4 GB | 1.2 GB | Yes | Force test |
| 100M | 2.8 GB | 2.4 GB | Yes | Force test |

Note: GPU VRAM per record = (4×3+16) = 28 bytes. CPU per record = 3×8 = 24 bytes. These differ because the GPU uses SoA uint32 fields + sort buffers, while CPU stores full uint64 fields.

GPU_FULL capacity: 4.64 GB / 28 = 165.7M records. All sizes fit.
GPU_CHUNKED is forced with artificial VRAM limit for correctness validation.

### 2.7 GPU Sub-Phase Instrumentation Approach

The production `execute_gpu_radix_sort<N>()` is monolithic — no timing hooks. The benchmark needs per-phase timing. Approach:

**The benchmark implements its own instrumented sort pipeline in `benchmark_gpu_sort.cc`**, reusing the same GPU primitives (cudaMalloc, CUB SortPairs, gather_keys_kernel) but with cudaEvent/chrono calls between phases. This is NOT a copy of the production code — it is a benchmark-specific pipeline that:

1. Performs AoS→SoA on CPU (timed with chrono)
2. Uploads to GPU (timed with cudaEvent)
3. Runs each sort pass separately (timed with cudaEvent per pass)
4. Downloads indices (timed with cudaEvent)
5. Scatters on CPU (timed with chrono)

The `gather_keys_kernel` is declared in `gpu_radix_sort.cuh` and callable from the benchmark. CUB DeviceRadixSort is header-only. The benchmark includes `<cub/cub.cuh>` directly and allocates its own device memory.

This approach avoids modifying the production GPU sort code while enabling fine-grained timing.

---

## 3. Component B: graph_project Instrumentation

### 3.1 Files Modified

- `src/graph_models/gql/projection/native_projection_builder.h` — add `ProjectionTimers` struct
- `src/graph_models/gql/projection/native_projection_builder.cc` — add timer calls around high-level phases
- `src/graph_models/gql/projection/projection_storage.cc` — add timer calls inside `build_all_indexes_bulk()` to separate sort_ms from btree_write_ms (the builder calls `storage->flush()` which encapsulates both)
- `src/graph_models/gql/projection/external_record_sort.h` — pass timers reference to `stream_sorted()` so sort time can be measured inside the sort decision point

### 3.2 Timer Struct

```cpp
struct ProjectionTimers {
    double node_scan_ms     = 0;
    double edge_scan_ms     = 0;
    double property_ms      = 0;
    double sort_ms          = 0;
    double btree_write_ms   = 0;
    double aggregation_ms   = 0;
    double metadata_ms      = 0;
    double total_ms         = 0;

    bool enabled = false;

    void print(const std::string& projection_name, uint64_t edge_count) const;
};
```

### 3.3 Activation

```cpp
// At projection start:
ProjectionTimers timers;
timers.enabled = (std::getenv("MDB_BENCHMARK") != nullptr);
```

When disabled: zero overhead (no chrono calls). The struct is stack-allocated and never used.

### 3.4 Instrumentation Pattern

```cpp
auto t0 = timers.enabled ? std::chrono::high_resolution_clock::now()
                         : std::chrono::high_resolution_clock::time_point{};

// ... existing phase code (unchanged) ...

if (timers.enabled) {
    auto t1 = std::chrono::high_resolution_clock::now();
    timers.sort_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}
```

### 3.5 Output

Printed to stderr at the end of `build_projection()`:

```
[BENCHMARK] graph_project 'my_projection' — 338541 edges
[BENCHMARK]   node_scan:       120.3 ms  ( 8.2%)
[BENCHMARK]   edge_scan:       245.1 ms  (16.7%)
[BENCHMARK]   property:         89.4 ms  ( 6.1%)
[BENCHMARK]   sort:            832.7 ms  (56.8%)  ← Amdahl P
[BENCHMARK]   btree_write:     142.0 ms  ( 9.7%)
[BENCHMARK]   aggregation:      28.5 ms  ( 1.9%)
[BENCHMARK]   metadata:          7.2 ms  ( 0.5%)
[BENCHMARK]   total:          1465.2 ms
[BENCHMARK]   sort_fraction:  0.568
```

### 3.6 Usage

```bash
# Start server with benchmark enabled
MDB_BENCHMARK=1 ./build/Release/bin/mdb server data/dbs/gql/my_db

# In another terminal, run a projection query
# The benchmark output appears on stderr of the server process
```

### 3.7 Constraints

- Zero overhead when `MDB_BENCHMARK` is not set
- No changes to projection logic — only wrapping existing phases with timers
- Output to stderr (not stdout, which is used for query results)
- Timer granularity: milliseconds (sufficient for phases that take 10ms+)
- Works with both GPU and CPU sort paths (measures whichever is selected)

### 3.8 CPU Baseline Protocol for Amdahl Validation

To compute S_measured for graph_project (not just isolated sort), need T_total_cpu and T_total_gpu:

**CPU baseline**: Run `graph_project` with GPU disabled by setting fake resources:
```bash
MDB_BENCHMARK=1 MDB_FORCE_CPU_SORT=1 ./build/Release/bin/mdb server data/dbs/gql/my_db
```

When `MDB_FORCE_CPU_SORT=1` is set, the GPU path in `stream_sorted()` is skipped (even if GPU is available), forcing the existing CPU sort. This produces T_total_cpu.

**GPU measurement**: Run the same `graph_project` without `MDB_FORCE_CPU_SORT`:
```bash
MDB_BENCHMARK=1 ./build/Release/bin/mdb server data/dbs/gql/my_db
```

This produces T_total_gpu. Then: `S_measured = T_total_cpu / T_total_gpu`.

The `MDB_FORCE_CPU_SORT` env var check is added alongside the existing `MDB_GPU_ENABLED` guard in `external_record_sort.h` and `external_edge_sort.h`.

---

## 4. Component C: Theoretical Analysis Document

### 4.1 File

`docs/research/2026-03-26-gpu-scheduler-performance-analysis.md`

### 4.2 Sections

#### 4.2.1 Work Law

Total operations for each algorithm:

| Algorithm | Work W(n) | Operations |
|-----------|-----------|------------|
| std::sort (introsort) | O(n log n) | comparisons (3 uint64 per Record<3>) |
| TBB parallel sort | O(n log n) | same work, distributed across cores |
| CUB RadixSort (3-pass) | O(3 × n × 8) = O(24n) | 8 radix passes per 32-bit key (CUB default 4-bit radix width), 3 multi-pass sort passes |
| External merge-sort | O(n log n) + O(n) I/O | comparison sort + disk read/write |

Analysis of constant factors: std::sort comparison ≈ 3 branches, CUB scatter ≈ 1 global memory transaction. GPU memory transaction latency ≈ 400 cycles but throughput ≈ 900 GB/s.

#### 4.2.2 Span Law (Critical Path)

For GPU_FULL pipeline on Record<3>, N records:

```
T∞ = T_aos_to_soa + T_h2d + 3×T_sort_pass + T_d2h + T_scatter

where:
  T_aos_to_soa  = N × 3 × (extract + cast)      ≈ O(N) CPU cycles
  T_h2d         = N × 12 bytes / BW_pcie          (12 bytes = 3 fields × 4)
  T_sort_pass   = O(N / P_gpu) + O(d)            (P_gpu = 1280 CUDA cores for 1660 Super)
  T_d2h         = N × 4 bytes / BW_pcie           (4 bytes = uint32 index)
  T_scatter     = N × (random read + sequential write) ≈ O(N) CPU cycles
```

Identify bottleneck: compute (GPU sort) vs transfer (PCIe) vs CPU (scatter).

For GTX 1660 Super: PCIe 3.0 ≈ 12 GB/s, GPU memory BW ≈ 336 GB/s.

#### 4.2.3 Amdahl's Law

```
S_total = 1 / ((1 - P) + P / S_sort)

P = sort_fraction (MEASURED from Component B)
S_sort = T_cpu_sort / T_gpu_sort (MEASURED from Component A)
```

Table with measured P and S_sort at different sizes. Graph showing S_total curve with asymptote at 1/(1-P).

#### 4.2.4 Gustafson's Law (Scaled Workload Analysis)

In standard Gustafson's Law, N represents processor count. Here we adapt it to a **fixed-hardware, scaled-workload** analysis (Gustafson-Barsis reformulation) where the problem size grows while the hardware (GTX 1660 Super) stays fixed:

```
S_scaled(s) = s - α(s - 1)

where:
  s = sort speedup factor (T_cpu_sort / T_gpu_sort), measured from Component A
  α = sequential fraction measured in the PARALLEL (GPU) execution
    = (T_total_gpu - T_sort_gpu) / T_total_gpu

Note: α is measured in the GPU run (not the CPU run) — this is the key
difference from Amdahl's P which is measured in the sequential run.
```

Analysis: as graph size grows, sort dominates more → α decreases → scaled speedup improves. Extrapolation from measured data at example sizes to papers100M scale.

#### 4.2.5 Predicted vs Measured Speedup

Final validation table:

| Size | P (measured) | S_sort (measured) | S_predicted (Amdahl) | S_measured | Error % |
|------|-------------|-------------------|---------------------|------------|---------|

This is the key thesis result — demonstrates whether the theoretical model accurately predicts real performance.

### 4.3 Data Sources

- Sort speedups: from Component A CSV
- Phase fractions: from Component B stderr output
- Hardware specs: GTX 1660 Super (6 GB, 1280 cores, PCIe 3.0), CPU from /proc/cpuinfo
- Theoretical constants: from CUDA documentation and CUB source analysis

---

## 5. Files Summary

### New Files
| File | Responsibility |
|------|---------------|
| `src/gpu/tests/benchmark_gpu_sort.cc` | Sort microbenchmark executable |
| `docs/research/2026-03-26-gpu-scheduler-performance-analysis.md` | Theoretical analysis |

### Modified Files
| File | Change |
|------|--------|
| `src/gpu/CMakeLists.txt` | Add benchmark_gpu_sort executable |
| `src/graph_models/gql/projection/native_projection_builder.h` | Add ProjectionTimers struct |
| `src/graph_models/gql/projection/native_projection_builder.cc` | Add timer instrumentation around phases |
| `src/graph_models/gql/projection/projection_storage.cc` | Timer calls inside build_all_indexes_bulk() |
| `src/graph_models/gql/projection/external_record_sort.h` | Pass timers ref, add MDB_FORCE_CPU_SORT check |
| `src/graph_models/gql/projection/external_edge_sort.h` | Add MDB_FORCE_CPU_SORT check |

---

## 6. Exclusions

NOT included in this design (by explicit decision):

- Roofline model (requires Nsight Compute profiling)
- Filter/transform kernel benchmarks (not integrated in pipeline)
- Neo4j GDS comparison (different platform, not apples-to-apples)
- Record<1>, <2>, <5> widths (Record<3> is dominant case)
- Google Benchmark framework (unnecessary dependency)
- GPU_CHUNKED with real VRAM exhaustion (all test sizes fit in GTX 1660 Super)
