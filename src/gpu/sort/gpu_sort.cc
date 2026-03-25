#include "gpu/sort/gpu_sort.h"

#include <algorithm>
#include <cassert>
#include <fstream>

#ifdef HAS_TBB
#include <execution>
#endif

#ifdef MDB_GPU_ENABLED
#include "gpu/sort/gpu_radix_sort.cuh"
#endif

namespace mdb::gpu {

namespace {

/// Read a binary spill file into the tail of `out`.
/// Each record is N contiguous uint64_t values (raw, no header).
template<std::size_t N>
void read_spill_file(const std::string& path, size_t count, std::vector<Record<N>>& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("gpu sort: cannot open spill file: " + path);
    }
    size_t start = out.size();
    out.resize(start + count);
    size_t expected_bytes = count * N * sizeof(uint64_t);
    file.read(reinterpret_cast<char*>(out.data() + start),
              static_cast<std::streamsize>(expected_bytes));
    if (static_cast<size_t>(file.gcount()) != expected_bytes) {
        out.resize(start);  // roll back
        throw std::runtime_error("gpu sort: truncated spill file: " + path);
    }
}

/// Sort in-place and stream every record through the callback.
template<std::size_t N>
bool execute_cpu_sort(
    std::vector<Record<N>>&                all_records,
    std::function<void(const Record<N>&)>& callback,
    bool                                   use_parallel
) {
#ifdef HAS_TBB
    if (use_parallel) {
        std::sort(std::execution::par_unseq, all_records.begin(), all_records.end());
    } else {
        std::sort(all_records.begin(), all_records.end());
    }
#else
    (void)use_parallel;
    std::sort(all_records.begin(), all_records.end());
#endif

    for (const auto& rec : all_records) {
        callback(rec);
    }
    return true;
}

} // anonymous namespace


template<std::size_t N>
bool sort_and_stream(
    std::vector<Record<N>>&               memory_records,
    const std::vector<std::string>&       spill_files,
    const std::vector<size_t>&            spill_counts,
    uint64_t                              total_records,
    std::function<void(const Record<N>&)> callback,
    const SystemResources&                resources,
    const PlannerConfig&                  config
) {
#ifndef NDEBUG
    {
        size_t computed_total = memory_records.size();
        for (size_t c : spill_counts) computed_total += c;
        assert(computed_total == total_records && "total_records mismatch with actual record counts");
    }
#endif
    auto plan = plan_sort(total_records, N, resources, config);

    // EXTERNAL_SORT: signal caller to use its own external merge-sort
    if (plan.strategy == SortStrategy::EXTERNAL_SORT) {
        return false;
    }

    // Collect all records into a single in-memory vector
    std::vector<Record<N>> all_records;
    all_records.reserve(total_records);

    for (auto& rec : memory_records) {
        all_records.push_back(std::move(rec));
    }
    memory_records.clear();

    for (size_t i = 0; i < spill_files.size(); i++) {
        read_spill_file<N>(spill_files[i], spill_counts[i], all_records);
    }

    switch (plan.strategy) {
        case SortStrategy::CPU_SEQUENTIAL:
            return execute_cpu_sort<N>(all_records, callback, /*use_parallel=*/false);

        case SortStrategy::CPU_PARALLEL:
            return execute_cpu_sort<N>(all_records, callback, /*use_parallel=*/true);

#ifdef MDB_GPU_ENABLED
        case SortStrategy::GPU_FULL: {
            bool ok = execute_gpu_radix_sort<N>(all_records, callback, plan.num_passes);
            if (!ok) {
                // GPU error — fall back to best CPU sort
                return execute_cpu_sort<N>(all_records, callback, resources.has_tbb);
            }
            return true;
        }
        case SortStrategy::GPU_CHUNKED: {
            bool ok = execute_gpu_chunked_sort<N>(
                all_records, callback, plan.num_passes,
                plan.num_chunks, plan.records_per_chunk);
            if (!ok) {
                // GPU error — fall back to best CPU sort
                return execute_cpu_sort<N>(all_records, callback, resources.has_tbb);
            }
            return true;
        }
#else
        case SortStrategy::GPU_FULL:
        case SortStrategy::GPU_CHUNKED:
            // CUDA not compiled in; fall back to best available CPU sort
            return execute_cpu_sort<N>(all_records, callback, resources.has_tbb);
#endif

        case SortStrategy::EXTERNAL_SORT:
        default:
            return false;
    }
}

// Explicit instantiations for the three Record widths used by MillenniumDB
template bool sort_and_stream<1>(
    std::vector<Record<1>>&, const std::vector<std::string>&,
    const std::vector<size_t>&, uint64_t,
    std::function<void(const Record<1>&)>,
    const SystemResources&, const PlannerConfig&);

template bool sort_and_stream<2>(
    std::vector<Record<2>>&, const std::vector<std::string>&,
    const std::vector<size_t>&, uint64_t,
    std::function<void(const Record<2>&)>,
    const SystemResources&, const PlannerConfig&);

template bool sort_and_stream<3>(
    std::vector<Record<3>>&, const std::vector<std::string>&,
    const std::vector<size_t>&, uint64_t,
    std::function<void(const Record<3>&)>,
    const SystemResources&, const PlannerConfig&);

template bool sort_and_stream<5>(
    std::vector<Record<5>>&, const std::vector<std::string>&,
    const std::vector<size_t>&, uint64_t,
    std::function<void(const Record<5>&)>,
    const SystemResources&, const PlannerConfig&);

} // namespace mdb::gpu
