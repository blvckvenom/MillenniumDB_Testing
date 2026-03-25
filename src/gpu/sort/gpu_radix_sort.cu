// gpu_radix_sort.cu — Multi-pass stable CUB RadixSort for Record<N>
//
// N-pass sort using CUB DeviceRadixSort::SortPairs on uint32 keys.
// Each pass sorts by one field (least-significant first), leveraging CUB's
// stability guarantee to preserve the ordering from previous passes.
//
// Also provides chunked GPU sort for datasets exceeding VRAM: sort each
// chunk on GPU, write to temp file, then K-way merge on CPU.

#include "gpu/sort/gpu_radix_sort.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace mdb::gpu {

// ---------------------------------------------------------------------------
// Error-handling macro: returns false instead of calling exit()
// ---------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            return false;                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Gather kernel: reorder keys by sorted indices for the next sort pass
// ---------------------------------------------------------------------------
__global__ void gather_keys_kernel(
    const uint32_t* field_values,   // [num_items] — values for current field
    const uint32_t* indices,        // [num_items] — permutation from prior pass
    uint32_t*       gathered_keys,  // [num_items] — output reordered keys
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
// RAII helper to free a collection of device pointers on scope exit
// ---------------------------------------------------------------------------
struct DeviceMemory {
    std::vector<void*> ptrs;

    void track(void* p) { ptrs.push_back(p); }

    void free_all() {
        for (void* p : ptrs) {
            if (p) cudaFree(p);
        }
        ptrs.clear();
    }

    ~DeviceMemory() { free_all(); }
};

// ---------------------------------------------------------------------------
// Core implementation (not templated — works on raw pointers)
// ---------------------------------------------------------------------------
static bool gpu_radix_sort_impl(
    const uint32_t* const* h_fields,    // [num_passes] host field arrays
    uint32_t               num_passes,
    uint64_t               num_items,
    uint32_t*              h_sorted_indices  // [num_items] output
) {
    DeviceMemory dmem;

    // CUB DeviceRadixSort takes int num_items; guard against silent overflow on
    // large-VRAM GPUs (H100/H200 with 80+ GB) where the caller could legitimately
    // pass more than INT_MAX records. Return false to trigger CPU fallback.
    if (num_items > static_cast<uint64_t>(INT_MAX)) {
        fprintf(stderr, "GPU sort: num_items %llu exceeds CUB int limit (%d)\n",
                (unsigned long long)num_items, INT_MAX);
        return false;
    }

    // -----------------------------------------------------------------------
    // 1. Allocate device field arrays and upload
    // -----------------------------------------------------------------------
    std::vector<uint32_t*> d_fields(num_passes, nullptr);
    size_t field_bytes = num_items * sizeof(uint32_t);

    for (uint32_t f = 0; f < num_passes; f++) {
        CHECK_CUDA(cudaMalloc(&d_fields[f], field_bytes));
        dmem.track(d_fields[f]);
        CHECK_CUDA(cudaMemcpy(d_fields[f], h_fields[f], field_bytes,
                              cudaMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // 2. Allocate sort double-buffers (keys + values) and temp storage
    // -----------------------------------------------------------------------
    uint32_t* d_keys_in  = nullptr;
    uint32_t* d_keys_out = nullptr;
    uint32_t* d_vals_in  = nullptr;
    uint32_t* d_vals_out = nullptr;

    CHECK_CUDA(cudaMalloc(&d_keys_in,  field_bytes));  dmem.track(d_keys_in);
    CHECK_CUDA(cudaMalloc(&d_keys_out, field_bytes));  dmem.track(d_keys_out);
    CHECK_CUDA(cudaMalloc(&d_vals_in,  field_bytes));  dmem.track(d_vals_in);
    CHECK_CUDA(cudaMalloc(&d_vals_out, field_bytes));  dmem.track(d_vals_out);

    // CUB temp storage size (two-call pattern with nullptr)
    void*  d_temp      = nullptr;
    size_t temp_bytes  = 0;
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        static_cast<int>(num_items)));

    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
    dmem.track(d_temp);

    // Kernel launch configuration for gather
    int block_size = 256;
    int grid_size  = static_cast<int>(
        std::min(static_cast<uint64_t>((num_items + block_size - 1) / block_size),
                 static_cast<uint64_t>(65535)));

    // -----------------------------------------------------------------------
    // 3. N passes: least-significant field (N-1) to most-significant (0)
    // -----------------------------------------------------------------------
    for (uint32_t pass = 0; pass < num_passes; pass++) {
        // Field index: sort from least-significant to most-significant
        uint32_t field_idx = num_passes - 1 - pass;

        if (pass == 0) {
            // First pass: keys = field[N-1], values = iota(0..num_items-1)
            CHECK_CUDA(cudaMemcpy(d_keys_in, d_fields[field_idx], field_bytes,
                                  cudaMemcpyDeviceToDevice));

            // Initialize indices 0..num_items-1 on host and upload
            // (cheaper than a kernel for the first pass only)
            std::vector<uint32_t> h_iota(num_items);
            std::iota(h_iota.begin(), h_iota.end(), 0u);
            CHECK_CUDA(cudaMemcpy(d_vals_in, h_iota.data(), field_bytes,
                                  cudaMemcpyHostToDevice));
        } else {
            // Subsequent passes: gather keys for current field by sorted indices
            // The sorted indices from the previous pass are in d_vals_out
            gather_keys_kernel<<<grid_size, block_size>>>(
                d_fields[field_idx], d_vals_out, d_keys_in, num_items);
            CHECK_CUDA(cudaGetLastError());

            // Both ops on default stream: kernel launch and D2D memcpy are serialized by CUDA.
            // gather reads d_vals_out, memcpy reads d_vals_out — both reads, no race.

            // Sorted indices become the new values
            CHECK_CUDA(cudaMemcpy(d_vals_in, d_vals_out, field_bytes,
                                  cudaMemcpyDeviceToDevice));
        }

        // Sort pairs: keys_in -> keys_out, vals_in -> vals_out
        CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
            d_temp, temp_bytes,
            d_keys_in, d_keys_out,
            d_vals_in, d_vals_out,
            static_cast<int>(num_items)));
    }

    // -----------------------------------------------------------------------
    // 4. Download final sorted indices
    // -----------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(h_sorted_indices, d_vals_out, field_bytes,
                          cudaMemcpyDeviceToHost));

    return true;
}

// ---------------------------------------------------------------------------
// Templated public API
// ---------------------------------------------------------------------------
template<std::size_t N>
bool execute_gpu_radix_sort(
    std::vector<Record<N>>&                all_records,
    std::function<void(const Record<N>&)>& callback,
    uint32_t                               num_passes
) {
    uint64_t num_items = all_records.size();

    // Clamp passes to record width
    if (num_passes > N) {
        num_passes = static_cast<uint32_t>(N);
    }

    if (num_items == 0 || num_passes == 0) {
        // Nothing to sort — stream records as-is
        for (const auto& rec : all_records) {
            callback(rec);
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // Step 1: AoS -> SoA conversion on CPU
    //   Extract lower 32 bits of each uint64 field (ObjectId VALUE bits)
    // -----------------------------------------------------------------------
    std::vector<std::vector<uint32_t>> h_fields(num_passes);
    for (uint32_t f = 0; f < num_passes; f++) {
        h_fields[f].resize(num_items);
    }

    for (uint64_t i = 0; i < num_items; i++) {
        for (uint32_t f = 0; f < num_passes; f++) {
            h_fields[f][i] = static_cast<uint32_t>(
                all_records[i][f] & 0x00FFFFFFFFFFFFFFULL);
        }
    }

    // Build pointer array for the impl function
    std::vector<const uint32_t*> field_ptrs(num_passes);
    for (uint32_t f = 0; f < num_passes; f++) {
        field_ptrs[f] = h_fields[f].data();
    }

    // -----------------------------------------------------------------------
    // Step 2: GPU sort (returns sorted indices)
    // -----------------------------------------------------------------------
    std::vector<uint32_t> sorted_indices(num_items);

    bool ok = gpu_radix_sort_impl(
        field_ptrs.data(), num_passes, num_items, sorted_indices.data());

    if (!ok) {
        return false;  // CUDA error — caller should fall back to CPU
    }

    // -----------------------------------------------------------------------
    // Step 3: Scatter back — reorder original records by sorted indices
    //   and stream through the callback
    // -----------------------------------------------------------------------
    for (uint64_t i = 0; i < num_items; i++) {
        callback(all_records[sorted_indices[i]]);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Chunked GPU sort: sort chunks on GPU, K-way merge on CPU
// ---------------------------------------------------------------------------

namespace /* anonymous */ {

/// Record size in bytes for Record<N>.
template<std::size_t N>
constexpr size_t RECORD_BYTES = N * sizeof(uint64_t);

/// Number of records to buffer per run during K-way merge I/O.
constexpr size_t MERGE_BLOCK_RECORDS = 4096;

/// Write a vector of sorted records to a binary file.
template<std::size_t N>
bool write_sorted_chunk(const std::vector<Record<N>>& records, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "GPU chunked sort: cannot create temp file: %s\n", path.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char*>(records.data()),
              static_cast<std::streamsize>(records.size() * RECORD_BYTES<N>));
    return out.good();
}

/// Comparator for K-way merge min-heap: (record, run_index).
/// Returns true when lhs should come AFTER rhs (min-heap convention).
template<std::size_t N>
struct ChunkMergeComparator {
    bool operator()(
        const std::pair<Record<N>, size_t>& lhs,
        const std::pair<Record<N>, size_t>& rhs
    ) const {
        return !(lhs.first < rhs.first);
    }
};

/// Refill a run buffer from an open file stream.
template<std::size_t N>
void refill_buffer(
    std::ifstream&          stream,
    std::vector<Record<N>>& buffer,
    size_t&                 remaining,
    size_t                  max_records
) {
    buffer.clear();
    size_t to_read = std::min(remaining, max_records);
    if (to_read == 0) return;

    buffer.resize(to_read);
    stream.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(to_read * RECORD_BYTES<N>));
    size_t actually_read = static_cast<size_t>(stream.gcount()) / RECORD_BYTES<N>;
    buffer.resize(actually_read);
    remaining -= actually_read;
}

/// K-way merge of sorted chunk files via min-heap priority queue.
/// Reads records in blocks (MERGE_BLOCK_RECORDS per refill) for I/O efficiency.
template<std::size_t N>
void kway_merge(
    const std::vector<std::string>&        sorted_files,
    const std::vector<size_t>&             chunk_record_counts,
    std::function<void(const Record<N>&)>& callback
) {
    size_t num_runs = sorted_files.size();

    // Open all chunk files and initialize per-run state
    std::vector<std::ifstream>          streams(num_runs);
    std::vector<std::vector<Record<N>>> buffers(num_runs);
    std::vector<size_t>                 buf_pos(num_runs, 0);
    std::vector<size_t>                 remaining(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        streams[i].open(sorted_files[i], std::ios::binary);
        if (!streams[i]) {
            fprintf(stderr, "GPU chunked sort: cannot open chunk file: %s\n",
                    sorted_files[i].c_str());
            return;
        }
        remaining[i] = chunk_record_counts[i];
        buffers[i].reserve(MERGE_BLOCK_RECORDS);
    }

    // Priority queue: (record, run_index) min-heap
    std::priority_queue<
        std::pair<Record<N>, size_t>,
        std::vector<std::pair<Record<N>, size_t>>,
        ChunkMergeComparator<N>
    > pq;

    // Seed the heap with the first record from each run
    for (size_t i = 0; i < num_runs; ++i) {
        if (remaining[i] > 0) {
            refill_buffer<N>(streams[i], buffers[i], remaining[i], MERGE_BLOCK_RECORDS);
            buf_pos[i] = 0;
            if (!buffers[i].empty()) {
                pq.push({buffers[i][0], i});
                buf_pos[i] = 1;
            }
        }
    }

    // Merge loop: pop minimum, emit via callback, refill from same run
    while (!pq.empty()) {
        auto [record, run_idx] = pq.top();
        pq.pop();

        callback(record);

        if (buf_pos[run_idx] < buffers[run_idx].size()) {
            // More records in the current buffer
            pq.push({buffers[run_idx][buf_pos[run_idx]], run_idx});
            buf_pos[run_idx]++;
        } else if (remaining[run_idx] > 0) {
            // Refill buffer from disk
            refill_buffer<N>(streams[run_idx], buffers[run_idx],
                             remaining[run_idx], MERGE_BLOCK_RECORDS);
            buf_pos[run_idx] = 0;
            if (!buffers[run_idx].empty()) {
                pq.push({buffers[run_idx][0], run_idx});
                buf_pos[run_idx] = 1;
            }
        }
        // else: run exhausted
    }

    for (auto& s : streams) {
        s.close();
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API: chunked GPU sort
// ---------------------------------------------------------------------------
template<std::size_t N>
bool execute_gpu_chunked_sort(
    std::vector<Record<N>>&                all_records,
    std::function<void(const Record<N>&)>& callback,
    uint32_t                               num_passes,
    uint32_t                               num_chunks,
    uint32_t                               records_per_chunk,
    const std::string&                     temp_dir
) {
    std::vector<std::string> sorted_chunk_files;
    std::vector<size_t>      chunk_record_counts;
    sorted_chunk_files.reserve(num_chunks);
    chunk_record_counts.reserve(num_chunks);

    // Ensure temp directory exists
    std::filesystem::create_directories(temp_dir);

    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        uint64_t start = static_cast<uint64_t>(chunk) * records_per_chunk;
        uint64_t end   = std::min(start + static_cast<uint64_t>(records_per_chunk),
                                  static_cast<uint64_t>(all_records.size()));
        if (start >= static_cast<uint64_t>(all_records.size())) break;
        uint64_t chunk_size = end - start;

        // Extract chunk records
        std::vector<Record<N>> chunk_records(
            all_records.begin() + static_cast<ptrdiff_t>(start),
            all_records.begin() + static_cast<ptrdiff_t>(end));

        // Sort chunk on GPU using existing multi-pass radix sort
        std::vector<Record<N>> sorted_chunk;
        sorted_chunk.reserve(chunk_size);

        std::function<void(const Record<N>&)> collect =
            [&sorted_chunk](const Record<N>& r) { sorted_chunk.push_back(r); };

        bool ok = execute_gpu_radix_sort<N>(chunk_records, collect, num_passes);
        if (!ok) {
            // GPU error — clean up any temp files already written
            for (const auto& path : sorted_chunk_files) {
                std::filesystem::remove(path);
            }
            return false;  // Caller will fall back to CPU
        }

        // Write sorted chunk to temp file
        std::string chunk_path = temp_dir + "/gpu_chunk_" + std::to_string(chunk);
        if (!write_sorted_chunk<N>(sorted_chunk, chunk_path)) {
            for (const auto& path : sorted_chunk_files) {
                std::filesystem::remove(path);
            }
            return false;
        }
        sorted_chunk_files.push_back(chunk_path);
        chunk_record_counts.push_back(static_cast<size_t>(sorted_chunk.size()));
    }

    // Free the original records (no longer needed)
    all_records.clear();
    all_records.shrink_to_fit();

    // K-way merge sorted chunk files
    if (sorted_chunk_files.size() == 1) {
        // Single chunk — just read it back and stream
        std::ifstream in(sorted_chunk_files[0], std::ios::binary);
        if (in) {
            std::vector<Record<N>> buf(MERGE_BLOCK_RECORDS);
            size_t remaining = chunk_record_counts[0];
            while (remaining > 0) {
                size_t to_read = std::min(remaining, MERGE_BLOCK_RECORDS);
                buf.resize(to_read);
                in.read(reinterpret_cast<char*>(buf.data()),
                        static_cast<std::streamsize>(to_read * RECORD_BYTES<N>));
                size_t got = static_cast<size_t>(in.gcount()) / RECORD_BYTES<N>;
                for (size_t i = 0; i < got; i++) {
                    callback(buf[i]);
                }
                remaining -= got;
                if (got < to_read) break;
            }
        }
    } else {
        kway_merge<N>(sorted_chunk_files, chunk_record_counts, callback);
    }

    // Cleanup temp files
    for (const auto& path : sorted_chunk_files) {
        std::filesystem::remove(path);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Explicit instantiations for the three Record widths used by MillenniumDB
// ---------------------------------------------------------------------------
template bool execute_gpu_radix_sort<1>(
    std::vector<Record<1>>&,
    std::function<void(const Record<1>&)>&,
    uint32_t);

template bool execute_gpu_radix_sort<2>(
    std::vector<Record<2>>&,
    std::function<void(const Record<2>&)>&,
    uint32_t);

template bool execute_gpu_radix_sort<3>(
    std::vector<Record<3>>&,
    std::function<void(const Record<3>&)>&,
    uint32_t);

template bool execute_gpu_chunked_sort<1>(
    std::vector<Record<1>>&,
    std::function<void(const Record<1>&)>&,
    uint32_t, uint32_t, uint32_t, const std::string&);

template bool execute_gpu_chunked_sort<2>(
    std::vector<Record<2>>&,
    std::function<void(const Record<2>&)>&,
    uint32_t, uint32_t, uint32_t, const std::string&);

template bool execute_gpu_chunked_sort<3>(
    std::vector<Record<3>>&,
    std::function<void(const Record<3>&)>&,
    uint32_t, uint32_t, uint32_t, const std::string&);

} // namespace mdb::gpu
