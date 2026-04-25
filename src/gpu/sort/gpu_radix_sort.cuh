// gpu_radix_sort.cuh — internal header for GPU radix sort (CUDA only)
//
// Not part of the public API.  Only included by gpu_sort.cc and
// gpu_radix_sort.cu, both of which live inside the mdb_gpu static library.
#pragma once

#include "storage/index/record.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mdb::gpu {

/// Multi-pass CUB RadixSort for Record<N>.
///
/// Algorithm:
///   1. AoS -> SoA: extract lower 32 bits from each uint64 field
///   2. Upload N field arrays to GPU
///   3. N passes (least-significant to most-significant field):
///      - Pass 1: keys = field[N-1], values = iota(0..num_records-1)
///      - Pass 2..N: gather keys by sorted indices, then sort (stable)
///   4. Scatter back: reorder records by final sorted indices on CPU
///
/// Returns true on success; false on any CUDA error (caller can fall back
/// to CPU sort).
template<std::size_t N>
bool execute_gpu_radix_sort(
    std::vector<Record<N>>&                all_records,
    std::function<void(const Record<N>&)>& callback,
    uint32_t                               num_passes
);

/// Chunked GPU sort for datasets exceeding VRAM.
///
/// Algorithm:
///   1. Divide records into chunks that fit in VRAM
///   2. Sort each chunk on GPU using multi-pass RadixSort
///   3. Write each sorted chunk to a temporary file
///   4. K-way merge on CPU using a min-heap priority queue
///
/// Returns true on success; false on any CUDA error (caller can fall back
/// to CPU sort).  Temporary files are cleaned up on both success and failure.
template<std::size_t N>
bool execute_gpu_chunked_sort(
    std::vector<Record<N>>&                all_records,
    std::function<void(const Record<N>&)>& callback,
    uint32_t                               num_passes,
    uint32_t                               num_chunks,
    uint32_t                               records_per_chunk,
    const std::string&                     temp_dir = "/tmp"
);

} // namespace mdb::gpu
