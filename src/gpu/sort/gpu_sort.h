#pragma once

#include "gpu/gpu_device.h"
#include "gpu/resource_planner.h"
#include "storage/index/record.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mdb::gpu {

/// Sort records and stream them in lexicographic order through the callback.
///
/// Integrates with plan_sort() to choose the best strategy:
///   - CPU_SEQUENTIAL / CPU_PARALLEL: handled here
///   - GPU_FULL: CUB RadixSort on GPU (falls back to CPU on CUDA error)
///   - GPU_CHUNKED: CPU fallback (Task 6 adds chunked GPU sort)
///   - EXTERNAL_SORT: returns false — caller must use ExternalRecordSort
///
/// @param memory_records  Records already in memory (moved from, cleared after)
/// @param spill_files     Paths to binary spill files (N * sizeof(uint64_t) per record)
/// @param spill_counts    Number of records in each spill file
/// @param total_records   Sum of memory_records.size() + all spill_counts
/// @param callback        Called once per record in sorted order
/// @param resources       System resource snapshot
/// @param config          Planner tuning knobs
/// @return true if sort was handled; false if caller should fall back to external sort
template<std::size_t N>
bool sort_and_stream(
    std::vector<Record<N>>&               memory_records,
    const std::vector<std::string>&       spill_files,
    const std::vector<size_t>&            spill_counts,
    uint64_t                              total_records,
    std::function<void(const Record<N>&)> callback,
    const SystemResources&                resources,
    const PlannerConfig&                  config = {}
);

} // namespace mdb::gpu
