#include "gpu/resource_planner.h"
#include <algorithm>
#include <cmath>

namespace mdb::gpu {

SortPlan plan_sort(
    uint64_t               num_records,
    uint32_t               num_fields,
    const SystemResources& resources,
    const PlannerConfig&   config
) {
    SortPlan plan;
    plan.num_passes = num_fields;

    // Per-record GPU memory: N field arrays (uint32 each) + 4 sort buffers (uint32 each)
    size_t bytes_per_record_gpu = static_cast<size_t>(num_fields) * 4 + 16;  // (4N + 16)
    size_t sort_data_size = num_records * bytes_per_record_gpu;

    // Full record size for CPU in-memory sort
    size_t full_record_bytes = num_records * num_fields * sizeof(uint64_t);

    // Rule 1: Small dataset — CPU always faster (GPU kernel launch overhead dominates)
    if (num_records < config.min_records_gpu) {
        plan.strategy = SortStrategy::CPU_SEQUENTIAL;
        plan.cpu_memory_needed = full_record_bytes;
        return plan;
    }

    // Rule 2: GPU_FULL — all sort data fits in VRAM
    if (resources.has_gpu) {
        size_t usable_vram = resources.gpu.free_vram;  // already has safety factor applied

        if (sort_data_size <= usable_vram) {
            plan.strategy = SortStrategy::GPU_FULL;
            plan.gpu_memory_needed = sort_data_size;
            plan.cpu_memory_needed = full_record_bytes;  // need records in CPU for AoS->SoA
            plan.num_chunks = 1;
            plan.records_per_chunk = static_cast<uint32_t>(
                std::min(num_records, static_cast<uint64_t>(UINT32_MAX)));
            return plan;
        }

        // Rule 3: GPU_CHUNKED — sort in chunks
        if (usable_vram >= config.min_chunk_vram) {
            uint64_t records_per_chunk = usable_vram / bytes_per_record_gpu;
            uint32_t chunks = static_cast<uint32_t>(
                (num_records + records_per_chunk - 1) / records_per_chunk);

            plan.strategy = SortStrategy::GPU_CHUNKED;
            plan.gpu_memory_needed = usable_vram;
            plan.cpu_memory_needed = full_record_bytes;
            plan.num_chunks = chunks;
            plan.records_per_chunk = static_cast<uint32_t>(
                std::min(records_per_chunk, static_cast<uint64_t>(UINT32_MAX)));
            return plan;
        }
    }

    // Rule 4: CPU_PARALLEL — fits in RAM
    size_t usable_ram = static_cast<size_t>(
        static_cast<double>(resources.ram_available) * config.cpu_ram_safety);

    if (full_record_bytes <= usable_ram) {
        plan.strategy = resources.has_tbb
            ? SortStrategy::CPU_PARALLEL
            : SortStrategy::CPU_SEQUENTIAL;
        plan.cpu_memory_needed = full_record_bytes;
        return plan;
    }

    // Rule 5: Fallback — external sort (current behavior)
    plan.strategy = SortStrategy::EXTERNAL_SORT;
    return plan;
}

} // namespace mdb::gpu
