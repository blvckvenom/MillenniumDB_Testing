#pragma once

#include "gpu/gpu_device.h"
#include <cstdint>

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

} // namespace mdb::gpu
