#include "gpu/sort/gpu_sort.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#ifdef HAS_TBB
#include <execution>
#endif

#ifdef HAS_LZ4
#include <lz4frame.h>
#endif

#ifdef MDB_GPU_ENABLED
#include "gpu/sort/gpu_radix_sort.cuh"
#endif

namespace mdb::gpu {

namespace {

// -----------------------------------------------------------------------------
// Spill file reader compatible with SpillCodec format.
//
// Spills produced by StreamingRecordBuffer may be compressed with LZ4 and
// carry an 8-byte header (magic 'GSPL' + version + compression_type +
// reserved). Legacy headerless spills are still supported via magic-byte
// detection. This reader is a LIGHTWEIGHT DUPLICATE of the read half of
// src/graph_models/gql/projection/spill_codec.{h,cc}; kept independent so
// the mdb_gpu library stays isolated from graph_models (per the "zero
// MillenniumDB dependencies" rule declared at the top of
// src/gpu/CMakeLists.txt).
//
// If the SpillCodec format ever evolves, update BOTH this copy and the
// canonical implementation. The unit tests in spill_codec_test.cc guard
// round-trip fidelity; a future test should also round-trip through this
// reader to detect divergence early.
// -----------------------------------------------------------------------------

constexpr uint32_t SPILL_MAGIC       = 0x4C505347u; // 'GSPL' little-endian
constexpr uint8_t  SPILL_VERSION     = 1u;
constexpr size_t   SPILL_HEADER_SIZE = 8u;
constexpr uint8_t  SPILL_COMP_NONE   = 0u;
constexpr uint8_t  SPILL_COMP_LZ4    = 1u;

/// Read a binary spill file into the tail of `out`.
/// Each record is N contiguous uint64_t values, either raw (legacy /
/// header compression=NONE) or LZ4-compressed behind an 8-byte header.
template<std::size_t N>
void read_spill_file(const std::string& path, size_t count, std::vector<Record<N>>& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("gpu sort: cannot open spill file: " + path);
    }

    // Detect SpillCodec header. A legacy file has no magic; fall through to
    // the raw read path without seeking past the first bytes.
    uint8_t hdr[SPILL_HEADER_SIZE];
    file.read(reinterpret_cast<char*>(hdr), SPILL_HEADER_SIZE);
    const std::streamsize hgot = file.gcount();

    bool    header_ok  = false;
    uint8_t compression = SPILL_COMP_NONE;

    if (hgot == static_cast<std::streamsize>(SPILL_HEADER_SIZE)) {
        const uint32_t magic = static_cast<uint32_t>(hdr[0])
                             | (static_cast<uint32_t>(hdr[1]) << 8)
                             | (static_cast<uint32_t>(hdr[2]) << 16)
                             | (static_cast<uint32_t>(hdr[3]) << 24);
        const uint8_t ver = hdr[4];
        const uint8_t ct  = hdr[5];
        if (magic == SPILL_MAGIC && ver == SPILL_VERSION
            && (ct == SPILL_COMP_NONE || ct == SPILL_COMP_LZ4))
        {
            header_ok   = true;
            compression = ct;
        }
    }

    if (!header_ok) {
        // Legacy or corrupt header: rewind and treat as raw.
        file.clear();
        file.seekg(0, std::ios::beg);
        if (!file) {
            throw std::runtime_error("gpu sort: cannot rewind spill file: " + path);
        }
    }

    const size_t start           = out.size();
    const size_t expected_bytes  = count * N * sizeof(uint64_t);
    out.resize(start + count);
    uint8_t* dst = reinterpret_cast<uint8_t*>(out.data() + start);

    if (compression == SPILL_COMP_NONE) {
        file.read(reinterpret_cast<char*>(dst),
                  static_cast<std::streamsize>(expected_bytes));
        if (static_cast<size_t>(file.gcount()) != expected_bytes) {
            out.resize(start); // roll back
            throw std::runtime_error("gpu sort: truncated spill file: " + path);
        }
        return;
    }

    // LZ4-compressed path.
#ifdef HAS_LZ4
    LZ4F_dctx*       ctx = nullptr;
    LZ4F_errorCode_t err = LZ4F_createDecompressionContext(&ctx, LZ4F_VERSION);
    if (LZ4F_isError(err)) {
        out.resize(start);
        throw std::runtime_error(
            std::string("gpu sort: LZ4F_createDecompressionContext failed: ") +
            LZ4F_getErrorName(err));
    }

    constexpr size_t IN_BUF_SIZE = 64 * 1024;
    std::vector<uint8_t> in_buf(IN_BUF_SIZE);
    size_t               total_out = 0;
    bool                 eof_reached = false;

    while (total_out < expected_bytes) {
        file.read(reinterpret_cast<char*>(in_buf.data()),
                  static_cast<std::streamsize>(in_buf.size()));
        std::streamsize got = file.gcount();
        if (got <= 0) {
            eof_reached = true;
            got = 0;
        }

        size_t src_pos = 0;
        const size_t src_end = static_cast<size_t>(got);
        do {
            size_t src_size = src_end - src_pos;
            size_t dst_size = expected_bytes - total_out;
            size_t hint = LZ4F_decompress(ctx,
                                          dst + total_out, &dst_size,
                                          in_buf.data() + src_pos, &src_size,
                                          nullptr);
            if (LZ4F_isError(hint)) {
                LZ4F_freeDecompressionContext(ctx);
                out.resize(start);
                throw std::runtime_error(
                    std::string("gpu sort: LZ4F_decompress failed: ") +
                    LZ4F_getErrorName(hint));
            }
            src_pos   += src_size;
            total_out += dst_size;

            if (dst_size == 0 && src_size == 0) {
                // No forward progress; break to avoid infinite loop.
                break;
            }
            if (hint == 0 && total_out == expected_bytes) {
                // Frame complete and we have everything we asked for.
                break;
            }
        } while (src_pos < src_end && total_out < expected_bytes);

        if (eof_reached) break;
    }

    LZ4F_freeDecompressionContext(ctx);

    if (total_out != expected_bytes) {
        out.resize(start);
        throw std::runtime_error(
            "gpu sort: LZ4 decompress truncated (got " + std::to_string(total_out)
            + " of " + std::to_string(expected_bytes) + " bytes) in " + path);
    }
#else
    // Compressed spill encountered but the build lacks LZ4 support.
    out.resize(start);
    throw std::runtime_error(
        "gpu sort: encountered LZ4-compressed spill but this build has no LZ4 "
        "support (rebuild with liblz4-dev). File: " + path);
#endif
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

    // Memory-safety gate: downgrade a GPU plan to CPU when the full record
    // vector exceeds the 2 GB ceiling. Protects the CLASSIC monolithic path
    // from OOMing the GPU on a multi-GB host vector; a no-op for the RADIX
    // per-partition path (always well under 2 GB).
    enforce_gpu_dataset_ceiling<N>(plan, total_records, resources);

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
