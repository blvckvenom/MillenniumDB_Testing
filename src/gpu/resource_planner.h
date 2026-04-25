#pragma once

#include "gpu/gpu_device.h"
#include "storage/index/record.h"

#include <cstdint>
#include <cstdio>

namespace mdb::gpu {

enum class SortStrategy {
    GPU_FULL,
    GPU_CHUNKED,
    CPU_PARALLEL,
    CPU_SEQUENTIAL,
    EXTERNAL_SORT
};

struct SortPlan {
    SortStrategy strategy            = SortStrategy::EXTERNAL_SORT;
    size_t       gpu_memory_needed   = 0;
    size_t       cpu_memory_needed   = 0;
    uint32_t     num_chunks          = 0;
    uint32_t     num_passes          = 0;
    uint32_t     records_per_chunk   = 0;
};

struct PlannerConfig {
    float    cpu_ram_safety   = 0.70f;
    uint64_t min_records_gpu  = 500000;
    size_t   min_chunk_vram   = 256 * 1024 * 1024;
};

SortPlan plan_sort(
    uint64_t               num_records,
    uint32_t               num_fields,
    const SystemResources& resources,
    const PlannerConfig&   config = {}
);

// Memory-safety ceiling for the GPU sort path.
//
// The CLASSIC monolithic projection sort can hand sort_and_stream the entire
// index as one in-memory vector (38.7 GB on papers100M), which the GPU code
// path would try to stage through host RAM and CUB device buffers — OOMing
// long before the planner's per-chunk VRAM math protects it. The RADIX
// per-partition path, by contrast, only ever submits one partition's worth of
// records (bounded by worker_memory_budget, default 512 MB) so it is always
// far under this ceiling.
//
// Call this AFTER plan_sort and BEFORE acting on plan.strategy: it downgrades
// GPU_FULL / GPU_CHUNKED to a CPU strategy when the full record vector would
// exceed 2 GB, leaving CPU / EXTERNAL plans untouched. Record<N> is
// std::array<uint64_t, N>, so sizeof(Record<N>) == 8*N is the exact per-record
// host footprint.
template <std::size_t N>
inline void enforce_gpu_dataset_ceiling(SortPlan& plan, uint64_t total_records,
                                        const SystemResources& resources) {
    constexpr size_t kMaxGpuDatasetBytes = 2ULL << 30;  // 2 GB
    const size_t bytes = static_cast<size_t>(total_records) * sizeof(Record<N>);
    if ((plan.strategy == SortStrategy::GPU_FULL ||
         plan.strategy == SortStrategy::GPU_CHUNKED) && bytes > kMaxGpuDatasetBytes) {
        std::fprintf(stderr,
                     "gpu_sort: dataset %.2f GB exceeds GPU safety ceiling (2 GB); "
                     "GPU path is per-partition only -- falling back to CPU.\n",
                     bytes / (1024.0 * 1024.0 * 1024.0));
        plan.strategy = resources.has_tbb ? SortStrategy::CPU_PARALLEL
                                          : SortStrategy::CPU_SEQUENTIAL;
    }
}

} // namespace mdb::gpu
