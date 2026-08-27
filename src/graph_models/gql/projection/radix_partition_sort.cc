// src/graph_models/gql/projection/radix_partition_sort.cc
#include "graph_models/gql/projection/radix_partition_sort.h"

#include <algorithm>
#include <bitset>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "graph_models/gql/projection/leaf_compression.h"
#include "graph_models/gql/projection/parallel_scan_partitioner.h"
#include "graph_models/gql/projection/partition_file.h"
#include "misc/ablation_registry.h"
#include "misc/available_ram.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/bplus_tree/bpt_mem_import.h"
#include "storage/page/page.h"

#ifdef MDB_GPU_ENABLED
#include "gpu/sort/gpu_sort.h"
#endif

namespace fs = std::filesystem;

namespace GQL {

namespace {

// Phase 2 emits its `.sorted_part_*.bin` files through std::ofstream, which
// swallows I/O failures by default: a short write (e.g. ENOSPC) would leave
// a truncated file that the Phase 3 reader consumes as a clean EOF,
// producing a structurally-valid B+Tree silently missing records. Every
// Phase 2 output stream therefore goes through these helpers, which arm the
// stream's exception mask and convert failures into std::runtime_error with
// errno context.
std::ofstream open_checked_output(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error(
            "RadixPartitionSort: cannot open " + path + " for writing: "
            + std::strerror(errno));
    }
    out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    return out;
}

[[noreturn]] void throw_write_error(const std::string& path) {
    throw std::runtime_error(
        "RadixPartitionSort: write to " + path + " failed: "
        + std::strerror(errno));
}

// close() flushes the stdio buffer; with the exception mask armed a flush
// failure surfaces here instead of being dropped by the ofstream destructor.
void close_checked_output(std::ofstream& out, const std::string& path) {
    try {
        out.close();
    } catch (const std::ios_base::failure&) {
        throw_write_error(path);
    }
}

#ifdef MDB_GPU_ENABLED
// mdb::gpu::execute_gpu_radix_sort sorts by the low 32 bits of each field's
// 56-bit ObjectId counter and drops the 8-bit type prefix, so its output
// matches the full 64-bit CPU ordering only when every counter fits in
// 32 bits and each field's type prefix is constant across the partition.
// Records violating either condition would silently mis-sort on the GPU;
// callers must route such partitions to the CPU std::sort path instead.
template<std::size_t N>
bool gpu_sort_preconditions_hold(const std::vector<Record<N>>& records) {
    if (records.empty()) return true;
    constexpr std::uint64_t kCounterMask = 0x00FFFFFFFFFFFFFFULL;
    for (std::size_t f = 0; f < N; ++f) {
        const std::uint64_t prefix = records[0][f] >> 56;
        for (const auto& rec : records) {
            if ((rec[f] & kCounterMask) > 0xFFFFFFFFULL
                || (rec[f] >> 56) != prefix) {
                return false;
            }
        }
    }
    return true;
}
#endif  // MDB_GPU_ENABLED

}  // namespace

// ---------------------------------------------------------------------------
// Phase 3: concatenate sorted per-partition files into B+Tree leaves.
//
// The radix-prefix property established by Phase 1 guarantees that the set of
// sorted partitions, when read in partition order, forms a globally sorted
// sequence. We therefore need no merge step: we stream records across the
// partitions, accumulate them into leaf-page-sized buffers, and hand each
// full page to BPTLeafWriter / BPTDirWriter in the exact same format used by
// ProjectionStorage::build_index_streaming (projection_storage.cc:895-987).
//
// Inline dedup at write time preserves the "no duplicates" invariant that
// ExternalRecordSort's single-pass writer delivers to the classic backend.
// ---------------------------------------------------------------------------
// BITSET-backed concatenation (V1 leaf format, pre-delta-varint-compression era).
// Preserves byte-identical output for indexes that do not use delta+LEB128-varint
// leaf encoding (the B+Tree leaf compression scheme where record 0 is stored as
// full LEB128 varints and records 1..k-1 are stored as zigzag-delta LEB128 varints).
template<std::size_t N>
static std::size_t write_btree_from_sorted_partitions_bitset_(
    const std::vector<std::string>& sorted_partition_paths,
    const std::string& base_path)
{
    BPTLeafWriter<N> leaf_writer(base_path + ".leaf");
    BPTDirWriter<N>  dir_writer(base_path + ".dir");

    // Are all partitions empty? Handle the empty-index case.
    bool any_nonempty = false;
    for (const auto& p : sorted_partition_paths) {
        if (fs::exists(p) && fs::file_size(p) > 0) {
            any_nonempty = true;
            break;
        }
    }
    if (!any_nonempty) {
        leaf_writer.make_empty();
        return 0;
    }

    // Leaf page capacity matches build_index_streaming exactly.
    constexpr std::size_t max_records_per_leaf =
        (Page::SIZE - 2 * sizeof(uint32_t) - N) / (sizeof(uint64_t) * N);

    // Scratch buffer: one leaf page. Compression packs more records per page,
    // so size to the Page::SIZE compressed-payload upper bound.
    const std::size_t max_buffer_size = Page::SIZE;
    auto leaf_buffer = std::make_unique<char[]>(max_buffer_size);

    std::vector<Record<N>> page_records;
    page_records.reserve(max_records_per_leaf);

    Record<N>   prev_record{};
    bool        has_prev         = false;
    std::size_t unique_count     = 0;
    std::size_t leaf_page_number = 0;

    const bool compress_leaves = !GQL::leaf_compression_disabled();
    std::bitset<N * 8> no_compression;  // all zeros ⇒ no compression applied

    // Running redundant-byte bitset over buffered page records (see the
    // identical logic in ProjectionStorage::build_index_streaming).
    std::bitset<N * 8> running_bitset;
    running_bitset.set();
    auto page_overflows_with = [&](const Record<N>& candidate) -> bool {
        std::bitset<N * 8> bs = running_bitset;
        const unsigned char* first =
            reinterpret_cast<const unsigned char*>(&page_records[0]);
        const unsigned char* cand =
            reinterpret_cast<const unsigned char*>(&candidate);
        for (std::size_t b = 0; b < N * 8; ++b) {
            if (bs[b] && cand[b] != first[b]) bs.set(b, false);
        }
        const std::size_t rc = bs.count();
        const std::size_t n  = page_records.size() + 1;
        return 2 * sizeof(uint32_t) + N + rc
             + n * (sizeof(uint64_t) * N - rc) > Page::SIZE;
    };

    auto write_leaf_page = [&](bool is_last_page) {
        if (page_records.empty()) return;

        const uint32_t next_page = is_last_page
            ? 0u
            : static_cast<uint32_t>(leaf_page_number + 1);

        // Skip directory entry for the first leaf (B+tree convention).
        if (leaf_page_number > 0) {
            dir_writer.bulk_insert(
                &page_records[0],
                0,
                static_cast<int32_t>(leaf_page_number));
        }

        // Format: [N-byte bitset] + [redundant bytes] + [non-redundant
        // per-record bytes] — the exact layout BPTLeafV1 decodes, identical
        // to ProjectionStorage::build_index_streaming so the radix/classic/GPU
        // backends stay byte-identical (golden-compare gate). Default
        // compresses; MDB_PROJECTION_NO_LEAF_COMPRESSION restores raw bytes.
        std::bitset<N * 8> page_bitset =
            compress_leaves
                ? GQL::compute_redundant_bitset<N>(
                      page_records.data(),
                      static_cast<uint32_t>(page_records.size()))
                : no_compression;

        GQL::pack_compressed_page<N>(
            page_records.data(),
            static_cast<uint32_t>(page_records.size()),
            page_bitset,
            leaf_buffer.get());

        leaf_writer.process_block(
            leaf_buffer.get(),
            static_cast<uint32_t>(page_records.size()),
            page_bitset,
            next_page);

        ++leaf_page_number;
        page_records.clear();
        running_bitset.set();  // fresh page: a single record is all-redundant
    };

    // Stream across all partitions in order. Each partition is already sorted,
    // and radix partitioning ensures partition_i < partition_{i+1} key-wise,
    // so the concatenation is globally sorted.
    //
    // Bulk-read partition records into a 4096-record batch to amortize
    // per-record fread() overhead. Without batching, each record routes
    // through a function-pointer chain into libc fread() for a 24-byte read;
    // batching amortizes that cost ~4000× for large partitions.
    constexpr std::size_t kReadBatch = 4096;
    std::vector<Record<N>> batch(kReadBatch);

    for (const auto& path : sorted_partition_paths) {
        if (!fs::exists(path)) continue;
        typename PartitionFile<N>::Reader reader(path);
        while (true) {
            std::size_t got = reader.read_batch(batch.data(), kReadBatch);
            if (got == 0) break;
            for (std::size_t i = 0; i < got; ++i) {
                const Record<N>& r = batch[i];
                if (has_prev && r == prev_record) {
                    continue;  // dedup
                }
                prev_record = r;
                has_prev    = true;
                ++unique_count;

                if (compress_leaves) {
                    if (!page_records.empty() && page_overflows_with(r)) {
                        write_leaf_page(/*is_last_page=*/false);
                    }
                    if (page_records.empty()) {
                        running_bitset.set();
                    } else {
                        const unsigned char* first =
                            reinterpret_cast<const unsigned char*>(&page_records[0]);
                        const unsigned char* rec =
                            reinterpret_cast<const unsigned char*>(&r);
                        for (std::size_t b = 0; b < N * 8; ++b) {
                            if (running_bitset[b] && rec[b] != first[b]) {
                                running_bitset.set(b, false);
                            }
                        }
                    }
                    page_records.push_back(r);
                } else {
                    page_records.push_back(r);
                    if (page_records.size() >= max_records_per_leaf) {
                        write_leaf_page(/*is_last_page=*/false);
                    }
                }
            }
            if (got < kReadBatch) break;
        }
    }

    // Flush the final partial page (marks next_page=0 as terminator).
    write_leaf_page(/*is_last_page=*/true);

    return unique_count;
}

// DELTA_VARINT-backed concatenation (V2 leaf format, delta+LEB128-varint
// B+Tree leaf compression). Streams records one at a time into BPTLeafV2Writer,
// which handles page boundaries internally (variable per-record bytes via
// zigzag deltas). Directory entries are emitted on page boundary crossings.
template<std::size_t N>
static std::size_t write_btree_from_sorted_partitions_delta_varint_(
    const std::vector<std::string>& sorted_partition_paths,
    const std::string& base_path)
{
    BPTLeafV2Writer<N> leaf_writer(base_path + ".leaf");
    BPTDirWriter<N>    dir_writer(base_path + ".dir");

    // Empty-index: finalize writes one empty v2 leaf page.
    bool any_nonempty = false;
    for (const auto& p : sorted_partition_paths) {
        if (fs::exists(p) && fs::file_size(p) > 0) {
            any_nonempty = true;
            break;
        }
    }
    if (!any_nonempty) {
        leaf_writer.make_empty();
        return 0;
    }

    Record<N>   prev_record{};
    bool        has_prev     = false;
    std::size_t unique_count = 0;
    std::size_t page_count   = 0;   // number of completed page boundary crossings

    // Bulk-read partition records into a 4096-record batch (~96 KB at N=3).
    // The original per-record path called PartitionFile::Reader::next() per
    // record, which routes each call through a function-pointer chain into
    // libc fread() for a 24-byte read — significant overhead for billions
    // of records. Bulk-read amortizes the call cost ~4000×.
    constexpr std::size_t kReadBatch = 4096;
    std::vector<Record<N>> batch(kReadBatch);

    for (const auto& path : sorted_partition_paths) {
        if (!fs::exists(path)) continue;
        typename PartitionFile<N>::Reader reader(path);
        while (true) {
            std::size_t got = reader.read_batch(batch.data(), kReadBatch);
            if (got == 0) break;
            for (std::size_t i = 0; i < got; ++i) {
                const Record<N>& r = batch[i];
                if (has_prev && r == prev_record) {
                    continue;  // dedup
                }
                prev_record = r;
                has_prev    = true;
                ++unique_count;

                const bool started_new_page = leaf_writer.append_record(r);
                if (started_new_page) {
                    ++page_count;
                    dir_writer.bulk_insert(
                        &r,
                        0,
                        static_cast<int32_t>(leaf_writer.current_page_index()));
                }
            }
            if (got < kReadBatch) break;  // partial read = EOF
        }
    }

    // Finalize: flush the tail page with next_leaf = 0.
    leaf_writer.finalize();

    return unique_count;
}

// CSR_HYBRID concatenation (v3 leaf format: edge-index B+Tree leaves ARE
// the CSR layout). Streams records from globally-ordered partition files into
// BPTLeafCSRWriter, which groups by record[0] (src) and emits chain-head +
// optional continuation pages. A trivial root .dir (key_count=0) is emitted;
// lookups route to leaf 0 and walk the next_leaf chain. Correctness-first MVP:
// src-keyed directory routing is deferred until scan benchmarks show the
// leaf-chain walk is a bottleneck.
//
// N == 3 is the only instantiation that produces sensible CSR output
// (src, dst, edge_id). The writer drops edge_id (the single-stream v3
// payload does not encode it) — on read, v3 returns 0
// for position 2, which GnnProjectionAdapter / TopologyAccessor callers
// tolerate.
template<std::size_t N>
static std::size_t write_btree_from_sorted_partitions_csr_(
    const std::vector<std::string>& sorted_partition_paths,
    const std::string& base_path)
{
    if constexpr (N == 3) {
        // Enable edge_id stream emission on edge indexes (N == 3 only;
        // property indexes route through the non-CSR sibling). The CSR-hybrid
        // layout stores edge_ids in a stream parallel to the dst stream so that
        // count(e) queries and edge-id lookups remain correct. See
        // sorter_dispatch.cc build_index_csr_from_sorter_ for the full rationale.
        BPTLeafCSRWriter<N> leaf_writer(base_path + ".leaf",
                                        /*emit_edge_ids=*/true);
        // Dtor emits the root dir page.
        BPTDirWriter<N>     dir_writer(base_path + ".dir");

        bool any_nonempty = false;
        for (const auto& p : sorted_partition_paths) {
            if (fs::exists(p) && fs::file_size(p) > 0) {
                any_nonempty = true;
                break;
            }
        }
        if (!any_nonempty) {
            leaf_writer.make_empty();
            return 0;
        }

        Record<N>   prev_record{};
        bool        has_prev     = false;
        std::size_t unique_count = 0;

        // Bulk-read into batches to amortize per-record fread() call overhead.
        constexpr std::size_t kReadBatch = 4096;
        std::vector<Record<N>> batch(kReadBatch);

        for (const auto& path : sorted_partition_paths) {
            if (!fs::exists(path)) continue;
            typename PartitionFile<N>::Reader reader(path);
            while (true) {
                std::size_t got = reader.read_batch(batch.data(), kReadBatch);
                if (got == 0) break;
                for (std::size_t i = 0; i < got; ++i) {
                    const Record<N>& r = batch[i];
                    if (has_prev && r == prev_record) continue;
                    prev_record = r;
                    has_prev    = true;
                    ++unique_count;
                    leaf_writer.append(r);
                }
                if (got < kReadBatch) break;
            }
        }

        leaf_writer.flush_finalize();
        return unique_count;
    } else {
        // CSR_HYBRID (edge-index B+Tree leaves as CSR) is scoped to edge indexes
        // (N==3). Reaching here is a caller-side invariant violation — radix
        // config should have kept graph_storage at BTREE for N==2 indexes.
        // Return 0 as a defensive fallback; the upstream ProjectionStorage
        // dispatch enforces the scope at config time.
        (void)sorted_partition_paths;
        (void)base_path;
        return 0;
    }
}

// Dispatch shim. Default leaf_format is BITSET to preserve the behavior of
// callers that do not specify delta+LEB128-varint B+Tree leaf compression.
template<std::size_t N>
std::size_t write_btree_from_sorted_partitions(
    const std::vector<std::string>& sorted_partition_paths,
    const std::string& base_path,
    BPT::LeafFormat leaf_format,
    BPT::GraphStorage graph_storage)
{
    // CSR_HYBRID (edge-index B+Tree leaves as CSR layout) supersedes leaf_format
    // for its scope (edge indexes, N==3). The helper itself gates on N via constexpr.
    if (graph_storage == BPT::GraphStorage::CSR_HYBRID) {
        return write_btree_from_sorted_partitions_csr_<N>(
            sorted_partition_paths, base_path);
    }
    if (leaf_format == BPT::LeafFormat::DELTA_VARINT) {
        return write_btree_from_sorted_partitions_delta_varint_<N>(
            sorted_partition_paths, base_path);
    }
    return write_btree_from_sorted_partitions_bitset_<N>(
        sorted_partition_paths, base_path);
}

template<std::size_t N>
std::size_t RadixPartitionSort<N>::compute_num_partitions(
    std::size_t total_bytes,
    std::size_t partition_target_bytes,
    std::size_t min_partitions,
    std::size_t max_partitions)
{
    if (partition_target_bytes == 0) return min_partitions;
    std::size_t raw = (total_bytes + partition_target_bytes - 1) / partition_target_bytes;
    return std::clamp(raw, min_partitions, max_partitions);
}

template<std::size_t N>
std::size_t RadixPartitionSort<N>::compute_num_workers(
    std::size_t cores_available,
    std::size_t scan_threads,
    std::size_t memory_budget,
    std::size_t worker_memory_budget,
    std::size_t override_value)
{
    if (override_value > 0) return override_value;
    std::size_t by_cores  = (cores_available > scan_threads)
                              ? (cores_available - scan_threads) : 1;
    std::size_t by_memory = (worker_memory_budget > 0)
                              ? (memory_budget / worker_memory_budget) : 1;
    return std::max<std::size_t>(1, std::min(by_cores, by_memory));
}

template<std::size_t N>
RadixPartitionSort<N>::RadixPartitionSort(Config config)
    : config_(std::move(config))
{
    if (config_.scratch_dir.empty()) {
        throw std::invalid_argument("RadixPartitionSort: scratch_dir is required");
    }
    fs::create_directories(config_.scratch_dir);

    // Size the GPU-submission semaphore once, from the environment.
    // Default 1 (serialize all GPU submits — the single device sorts one
    // partition at a time while the other workers run CPU sorts in parallel).
    // Clamp to [1, 8] to keep a bound on contention even on a misconfigured
    // env value. Read once: the count is fixed for this sort's lifetime.
    gpu_concurrency_ = 1;
    if (const char* env = std::getenv("MDB_PROJECTION_RADIX_GPU_CONCURRENCY")) {
        try {
            long parsed = std::stol(env);
            if (parsed < 1) parsed = 1;
            if (parsed > 8) parsed = 8;
            gpu_concurrency_ = static_cast<int>(parsed);
        } catch (...) {
            gpu_concurrency_ = 1;  // leave default on parse failure
        }
    }
    gpu_semaphore_ = std::make_unique<CountingSemaphore>(gpu_concurrency_);
}

template<std::size_t N>
RadixPartitionSort<N>::~RadixPartitionSort() {
    // Exception-safe cleanup of any remaining scratch files.
    try { fs::remove_all(config_.scratch_dir); } catch (...) {}
}

template<std::size_t N>
std::uint32_t RadixPartitionSort<N>::radix_bucket(const Record<N>& r) const {
    // Bucket from the high COUNTER bits of record[0], skipping the 8-bit
    // ObjectId type prefix.
    //
    // Pre-fix bug: a naïve `r[0] >> (64 - bit_width)` reads the TOP bits of
    // the id, but ObjectIds in this codebase reserve bits 56-63 for the
    // type prefix (MASK_NODE = 0xD4, MASK_DIRECTED_EDGE = 0xE0,
    // MASK_UNDIRECTED_EDGE = 0xE4, etc. — see object_id.h). All records in a
    // given index share the same type byte, so the top bits are constant
    // and every record collapses to a single bucket. Phase 2 sort then runs
    // on a single populated partition, defeating the parallel design.
    //
    // Fix: mask off the type prefix and bucket on the next-most-significant
    // bits of the 56-bit counter. This is still an ORDER-PRESERVING radix
    // (records that bucket to b are all < records that bucket to b+1 in
    // sort key order *within the index's homogeneous type space*), so the
    // Phase 3 concatenation invariant ("walk partitions in bucket order =
    // globally sorted output") still holds. For indexes where r[0] is a
    // label_id with very few distinct values (e.g. label_node, label_edge),
    // this fix doesn't help — partitioning is fundamentally bounded by the
    // distinct-value count there. The vast majority of edge / node indexes
    // benefit fully.
    if (num_partitions_ <= 1) return 0;
    int bit_width = 64 - __builtin_clzll(static_cast<unsigned long long>(num_partitions_ - 1));
    // Strip the 8-bit type prefix; counter occupies bits 0-55. Shift so the
    // top `bit_width` bits of the 56-bit counter become the bucket index.
    constexpr std::uint64_t kCounterMask = 0x00FFFFFFFFFFFFFFULL;
    std::uint64_t counter_only = r[0] & kCounterMask;
    int shift = 56 - bit_width;
    if (shift < 0) shift = 0;  // safety for num_partitions > 2^56 (unreachable)
    return static_cast<std::uint32_t>(counter_only >> shift);
}

template<std::size_t N>
std::size_t RadixPartitionSort<N>::scan_and_partition(
    StreamingRecordBuffer<N>& input,
    std::uint64_t             estimated_count)
{
    std::size_t total_bytes = estimated_count * sizeof(Record<N>);
    num_partitions_ = compute_num_partitions(
        total_bytes, config_.partition_target_bytes,
        config_.min_partitions, config_.max_partitions);

    std::size_t scan_threads = (config_.num_scan_threads > 0)
        ? config_.num_scan_threads
        : std::max<std::size_t>(1, std::thread::hardware_concurrency() / 2);

    ParallelScanPartitioner<N> partitioner(
        num_partitions_, scan_threads, config_.scratch_dir,
        [this](const Record<N>& r) { return radix_bucket(r); });

    // Chunk-parallel Phase 1 partition fill via TBB (CPU parallelism option).
    // Default ON; opt-out with MDB_PROJECTION_RADIX_PHASE1_PARALLEL=0 for
    // A/B benchmarking and bisecting regressions. The single-thread fallback
    // preserves sequential-fill behavior bit-for-bit.
    bool parallel_phase1 = true;
    if (const char* env = std::getenv("MDB_PROJECTION_RADIX_PHASE1_PARALLEL")) {
        if (std::string(env) == "0") parallel_phase1 = false;
    }
    // Tiny inputs don't amortize TBB dispatch — fall back to sequential.
    // Threshold matches the default chunk size (64 K records).
    constexpr std::uint64_t kParallelMinRecords = 65536;
    if (parallel_phase1 && estimated_count >= kParallelMinRecords && scan_threads > 1) {
        input.begin_iteration();
        partitioner.run_parallel_chunked(
            [&](std::vector<Record<N>>& out_chunk, std::size_t max_records)
                -> std::size_t
            {
                std::size_t pulled = 0;
                while (pulled < max_records && input.has_next()) {
                    out_chunk.push_back(input.next());
                    ++pulled;
                }
                return pulled;
            });
    } else {
        partitioner.run([&](auto emit) {
            // Sequential drain of the input buffer (legacy path).
            input.begin_iteration();
            while (input.has_next()) {
                emit(input.next());
            }
        });
    }

    partition_paths_ = partitioner.collect_merged_partition_paths();

    // How the records actually landed across the partitions.
    //
    // WHY THIS LINE EXISTS. The whole point of this backend is that Phase 2
    // sorts P partitions concurrently; if the bucket function routes everything
    // into one, Phase 2 has a single unit of work and the backend degrades into
    // a single-threaded external sort that also paid to write P files first.
    // That degradation is INVISIBLE from the outside: the output is still
    // correct and still sorted, because one sorted partition IS the sorted
    // whole. It was in fact happening on every graph in this project, and the
    // only reason anyone noticed is that a measurement went looking.
    //
    // Emitted here rather than in Phase 2 because the distribution is a
    // property of the PARTITIONING, and reporting it next to the sort would
    // conflate "the bucket function is broken" with "the sorter is slow".
    {
        std::uint64_t nonempty = 0, total_bytes = 0, max_bytes = 0;
        for (const auto& p : partition_paths_) {
            std::uintmax_t sz = 0;
            try { sz = fs::file_size(p); } catch (...) { sz = 0; }
            if (sz > 0) ++nonempty;
            total_bytes += sz;
            if (sz > max_bytes) max_bytes = sz;
        }
        // Ratio of the largest partition to the perfectly balanced share. 1.0
        // is ideal; P means total collapse into one bucket.
        const double desbalance =
            (total_bytes > 0 && num_partitions_ > 0)
                ? static_cast<double>(max_bytes) * num_partitions_ / total_bytes
                : 0.0;
        std::cerr << "[RADIX] partition balance: nonempty=" << nonempty
                  << "/" << num_partitions_
                  << " max_bytes=" << max_bytes
                  << " total_bytes=" << total_bytes
                  << " imbalance=" << desbalance
                  << (nonempty <= 1 && num_partitions_ > 1
                          ? "  COLLAPSED: Phase 2 has one unit of work"
                          : "")
                  << "\n";
    }

    return num_partitions_;
}

template<std::size_t N>
std::size_t RadixPartitionSort<N>::sort_and_write(
    const std::string& output_base_path)
{
    std::size_t total_written = 0;

    // Use adaptive memory budget instead of a hardcoded constant. An earlier
    // hardcoded 4 GB budget capped Phase 2 worker count to
    // `min(cores - scan_threads, 4 GB / 512 MB) = min(20-10, 8) = 8`
    // workers on a 20-core host. Adaptive sizing reads MemAvailable at
    // sort-phase entry, after the scan phase has released its streaming
    // buffers. compute_adaptive_sort_buffer respects the MDB_SORT_BUFFER_MB
    // env override too, so operators can pin a value for benchmarking.
    std::size_t adaptive_memory_budget = compute_adaptive_sort_buffer();
    std::size_t num_workers = compute_num_workers(
        std::thread::hardware_concurrency(),
        (config_.num_scan_threads > 0)
            ? config_.num_scan_threads
            : std::thread::hardware_concurrency() / 2,
        adaptive_memory_budget,
        config_.worker_memory_budget,
        config_.num_workers);
    // Bound Phase 2 concurrency to the memory-capped worker count. Each
    // in-flight partition sort buffers up to worker_memory_budget bytes, so
    // dispatching on the default arena (up to hardware_concurrency workers)
    // would multiply transient RSS far beyond the
    // O(num_workers × worker_memory_budget) bound that compute_num_workers
    // derives (peak-RSS design constraint for the radix partition sort backend).
    tbb::task_arena arena(static_cast<int>(num_workers));

    // Dispatch partitions across workers via tbb::parallel_for.
    arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, num_partitions_, 1),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t p = r.begin(); p < r.end(); ++p) {
                    std::string sorted_path = output_base_path +
                        ".sorted_part_" + std::to_string(p) + ".bin";
                    // Decide in-memory vs external.
                    std::uintmax_t sz = fs::file_size(partition_paths_[p]);
                    if (sz <= config_.worker_memory_budget) {
                        sort_partition_in_memory(p, sorted_path);
                    } else {
                        sort_partition_external(p, sorted_path);
                    }
                    // Consistency cross-check: Phase 2 sorts without
                    // deduplicating, so the sorted output must carry exactly
                    // the bytes of its input partition. A mismatch means a
                    // short read or short write slipped past the per-call
                    // checks; failing here beats handing Phase 3 a truncated
                    // partition it would consume as a clean EOF.
                    std::uintmax_t sorted_sz = fs::file_size(sorted_path);
                    if (sorted_sz != sz) {
                        throw std::runtime_error(
                            "RadixPartitionSort: sorted partition "
                            + sorted_path + " holds "
                            + std::to_string(sorted_sz)
                            + " bytes but its input partition holds "
                            + std::to_string(sz) + " bytes");
                    }
                    // Release free heap pages to the kernel between sorts.
#if defined(__GLIBC__)
                    malloc_trim(0);
#endif
                }
            });
    });

    // Phase 3: concatenate sorted partitions into B+Tree leaves.
    // The radix-prefix property guarantees global sorted order without a
    // merge step; write_btree_from_sorted_partitions streams records
    // straight into BPTLeafWriter/BPTDirWriter and returns the post-dedup
    // unique count (previously approximated by file-size summation).
    std::vector<std::string> sorted_paths;
    sorted_paths.reserve(num_partitions_);
    for (std::size_t p = 0; p < num_partitions_; ++p) {
        std::string sorted_path = output_base_path +
            ".sorted_part_" + std::to_string(p) + ".bin";
        if (fs::exists(sorted_path)) {
            sorted_paths.push_back(sorted_path);
        }
    }
    total_written = write_btree_from_sorted_partitions<N>(
        sorted_paths, output_base_path,
        config_.leaf_format, config_.graph_storage);

    // Cleanup intermediate per-partition sorted files — the B+Tree
    // (`.leaf` + `.dir`) is now the authoritative output. Tests may set
    // keep_sorted_parts to inspect per-partition sort order post-hoc.
    if (!config_.keep_sorted_parts) {
        for (const auto& p : sorted_paths) {
            try { fs::remove(p); } catch (...) { /* best-effort cleanup */ }
        }
    }

    // One-line summary of where Phase 2 sorted each partition.
    // Proves the GPU per-partition path actually ran (gpu>0) vs everything
    // routing to CPU (gpu==0, e.g. partitions below the planner's
    // min_records_gpu threshold or a non-CUDA build).
    std::cerr << "[RADIX] partitions sorted: gpu=" << gpu_partitions_.load()
              << " cpu=" << cpu_partitions_.load() << "\n";

    return total_written;
}

// GPU path for RADIX Phase 2 per-partition sort.
//
// When MillenniumDB is built with CUDA (MDB_GPU_ENABLED) and a CUDA device is
// available at runtime, route each in-memory partition through
// mdb::gpu::sort_and_stream<N> which selects between GPU_FULL,
// GPU_CHUNKED, and CPU strategies via the existing resource_planner.
//
// Disabled when:
//   - the build lacks CUDA (no MDB_GPU_ENABLED define), or
//   - env var MDB_PROJECTION_RADIX_GPU is set to "0" (A/B benchmark switch),
//   - env var MDB_FORCE_CPU_SORT is set (matches external_record_sort.h:58
//     precedent), or
//   - the planner downgrades to a CPU strategy (small dataset, no free VRAM,
//     etc.) — in that case the wrapper sorts on CPU itself, so we still
//     stream straight to the output writer without a separate std::sort.
//
// MDB_PROJECTION_RADIX_GPU_MIN_RECORDS overrides PlannerConfig.min_records_gpu
// so the GPU path is reachable on small synthetic datasets during testing
// (default 500 K, see resource_planner.h:27).
//
// Output equivalence note: mdb::gpu::execute_gpu_radix_sort extracts only
// the lower 32 bits of each ObjectId value (gpu_radix_sort.cu, mask
// 0x00FFFFFFFFFFFFFFULL truncated to uint32_t). For B+Tree records this is
// safe when (a) every record in a given index shares the same top-8-bit
// type prefix, so masking it is a no-op for ordering, and (b) the value
// field fits in 32 bits. gpu_sort_preconditions_hold verifies both before
// the GPU branch engages; partitions that violate either condition (e.g.
// counters past 4 B objects, or property values whose type bytes differ
// across records) take the in-process std::sort path below instead of
// silently sorting by truncated keys.
template<std::size_t N>
void RadixPartitionSort<N>::sort_partition_in_memory(
    std::size_t partition_idx, const std::string& sorted_output_path)
{
    std::vector<Record<N>> buffer;
    {
        // Bulk-read into batches, then bulk-append. This amortizes fread
        // overhead and uses vector::insert (memcpy under the hood) instead
        // of per-element push_back checks.
        constexpr std::size_t kReadBatch = 4096;
        std::vector<Record<N>> batch(kReadBatch);
        typename PartitionFile<N>::Reader reader(partition_paths_[partition_idx]);
        while (true) {
            std::size_t got = reader.read_batch(batch.data(), kReadBatch);
            if (got == 0) break;
            buffer.insert(buffer.end(), batch.begin(), batch.begin() + got);
            if (got < kReadBatch) break;
        }
    }

#ifdef MDB_GPU_ENABLED
    {
        const char* gpu_off  = std::getenv("MDB_PROJECTION_RADIX_GPU");
        const char* cpu_only = std::getenv("MDB_FORCE_CPU_SORT");
        const bool radix_gpu_disabled =
            (gpu_off != nullptr && std::string(gpu_off) == "0") ||
            cpu_only != nullptr;

        bool gpu_eligible = !radix_gpu_disabled && !buffer.empty();
        if (gpu_eligible && !gpu_sort_preconditions_hold<N>(buffer)) {
            gpu_eligible = false;
            std::fprintf(stderr,
                         "RadixPartitionSort: partition %zu has keys outside "
                         "the GPU sort's 32-bit range; using CPU sort\n",
                         partition_idx);
        }
        if (gpu_eligible) {
            // Bound concurrent GPU submissions. Acquire the
            // semaphore immediately before the GPU-submission region and
            // release it on every exit/exception path via the RAII guard;
            // the other Phase 2 workers sort their partitions on the CPU
            // concurrently (they never reach this block). The in_flight /
            // peak counters are maintained around the same region so the
            // unit test can assert the observed peak never exceeds
            // gpu_concurrency_. Inc/dec straddle the guard so the peak
            // reflects everyone who held (or is waiting just past) a permit.
            CountingSemaphoreGuard gpu_guard(*gpu_semaphore_);
            int now = gpu_in_flight_.fetch_add(1, std::memory_order_relaxed) + 1;
            bump_gpu_peak_(now);
            struct InFlightExit {
                std::atomic<int>& in_flight;
                ~InFlightExit() { in_flight.fetch_sub(1, std::memory_order_relaxed); }
            } in_flight_exit{gpu_in_flight_};

            std::ofstream out = open_checked_output(sorted_output_path);
            std::function<void(const Record<N>&)> emit =
                [&out, &sorted_output_path](const Record<N>& rec) {
                    try {
                        out.write(reinterpret_cast<const char*>(&rec),
                                  sizeof(Record<N>));
                    } catch (const std::ios_base::failure&) {
                        throw_write_error(sorted_output_path);
                    }
                };

            mdb::gpu::PlannerConfig pcfg;
            if (const char* min_rec = std::getenv("MDB_PROJECTION_RADIX_GPU_MIN_RECORDS")) {
                try {
                    pcfg.min_records_gpu = std::stoull(min_rec);
                } catch (...) {
                    // Leave default on parse failure.
                }
            }

            auto resources = mdb::gpu::detect_resources();
            std::vector<std::string> empty_spill_files;
            std::vector<std::size_t> empty_spill_counts;

            // Determine whether the GPU will *actually* sort this
            // partition (vs sort_and_stream internally downgrading to CPU for a
            // sub-threshold or oversized partition) by replaying the same plan
            // the wrapper computes. Without this the telemetry would count a
            // planner CPU-downgrade as a GPU sort and falsely report gpu>0 on
            // small inputs (e.g. cora partitions below min_records_gpu).
            auto plan = mdb::gpu::plan_sort(
                static_cast<std::uint64_t>(buffer.size()), N, resources, pcfg);
            mdb::gpu::enforce_gpu_dataset_ceiling<N>(
                plan, static_cast<std::uint64_t>(buffer.size()), resources);
            const bool gpu_strategy =
                (plan.strategy == mdb::gpu::SortStrategy::GPU_FULL ||
                 plan.strategy == mdb::gpu::SortStrategy::GPU_CHUNKED);

            const bool used = mdb::gpu::sort_and_stream<N>(
                buffer, empty_spill_files, empty_spill_counts,
                static_cast<std::uint64_t>(buffer.size()),
                emit, resources, pcfg);

            if (used) {
                // sort_and_stream moved out of `buffer` and either sorted on
                // GPU or on CPU (planner downgrade); either way the records are
                // now in `sorted_output_path`. Tally by the actual strategy.
                if (gpu_strategy) {
                    gpu_partitions_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    cpu_partitions_.fetch_add(1, std::memory_order_relaxed);
                }
                close_checked_output(out, sorted_output_path);
                return;
            }
            // EXTERNAL_SORT decision (planner returned false): fall through
            // to the in-process std::sort path below. `buffer` contents are
            // preserved on a `false` return per the wrapper contract.
        }
    }
#else
    // The three switches above steer a block this build does not contain. An
    // arm that sets one of them here runs the same CPU sort as the arm that
    // does not, so they are declared inert rather than left looking effective.
    // Resolved once: this runs per partition, on every worker.
    static const bool gpu_sort_switches_declared = [] {
        Ablation::inert("MDB_PROJECTION_RADIX_GPU", "built without CUDA");
        Ablation::inert("MDB_PROJECTION_RADIX_GPU_MIN_RECORDS",
                        "built without CUDA");
        Ablation::inert("MDB_FORCE_CPU_SORT", "built without CUDA");
        return true;
    }();
    (void) gpu_sort_switches_declared;
#endif  // MDB_GPU_ENABLED

    // CPU std::sort fallback path: GPU disabled, build lacks CUDA, partition
    // preconditions failed, or planner returned EXTERNAL_SORT. Empty
    // partitions are not counted (telemetry tracks non-empty partitions only).
    if (!buffer.empty()) {
        cpu_partitions_.fetch_add(1, std::memory_order_relaxed);
    }
    std::sort(buffer.begin(), buffer.end());
    std::ofstream out = open_checked_output(sorted_output_path);
    try {
        out.write(reinterpret_cast<const char*>(buffer.data()),
                  buffer.size() * sizeof(Record<N>));
    } catch (const std::ios_base::failure&) {
        throw_write_error(sorted_output_path);
    }
    close_checked_output(out, sorted_output_path);
}

template<std::size_t N>
void RadixPartitionSort<N>::sort_partition_external(
    std::size_t partition_idx, const std::string& sorted_output_path)
{
    // Defensive path: partition exceeds worker_memory_budget. Chunk-sort
    // to intermediate "run" files, then K-way merge. Always CPU — too large
    // for the in-memory GPU buffer path.
    cpu_partitions_.fetch_add(1, std::memory_order_relaxed);
    std::vector<Record<N>> chunk;
    std::size_t chunk_cap = config_.worker_memory_budget / sizeof(Record<N>) / 2;
    if (chunk_cap == 0) chunk_cap = 1;  // minimum: handle tiny worker_memory_budget in tests
    chunk.reserve(chunk_cap);

    std::vector<std::string> run_paths;
    {
        // Bulk-read instead of per-record fread to amortize call overhead.
        // This path handles partitions that exceed worker_memory_budget; with
        // an accurate estimated_count most partitions hit the in-memory
        // sort path instead, so this code is exercised only on degenerate inputs.
        typename PartitionFile<N>::Reader reader(partition_paths_[partition_idx]);
        constexpr std::size_t kReadBatch = 4096;
        std::vector<Record<N>> batch(kReadBatch);
        std::size_t run_idx = 0;
        bool eof = false;
        while (!eof) {
            std::size_t got = reader.read_batch(batch.data(), kReadBatch);
            if (got == 0) break;
            if (got < kReadBatch) eof = true;
            for (std::size_t i = 0; i < got; ++i) {
                chunk.push_back(batch[i]);
                if (chunk.size() >= chunk_cap) {
                    std::sort(chunk.begin(), chunk.end());
                    std::string run_path = sorted_output_path + ".run_" + std::to_string(run_idx++);
                    std::ofstream run_out = open_checked_output(run_path);
                    try {
                        run_out.write(reinterpret_cast<const char*>(chunk.data()),
                                      chunk.size() * sizeof(Record<N>));
                    } catch (const std::ios_base::failure&) {
                        throw_write_error(run_path);
                    }
                    close_checked_output(run_out, run_path);
                    run_paths.push_back(run_path);
                    chunk.clear();
                }
            }
        }
        if (!chunk.empty()) {
            std::sort(chunk.begin(), chunk.end());
            std::string run_path = sorted_output_path + ".run_" + std::to_string(run_idx++);
            std::ofstream run_out = open_checked_output(run_path);
            try {
                run_out.write(reinterpret_cast<const char*>(chunk.data()),
                              chunk.size() * sizeof(Record<N>));
            } catch (const std::ios_base::failure&) {
                throw_write_error(run_path);
            }
            close_checked_output(run_out, run_path);
            run_paths.push_back(run_path);
        }
    }

    // K-way merge the runs into sorted_output_path. Use unique_ptr to avoid
    // relying on Reader being movable (Reader owns a FILE* with non-trivial dtor).
    std::vector<std::unique_ptr<typename PartitionFile<N>::Reader>> readers;
    readers.reserve(run_paths.size());
    for (auto& p : run_paths) {
        readers.emplace_back(std::make_unique<typename PartitionFile<N>::Reader>(p));
    }

    std::vector<std::optional<Record<N>>> fronts(run_paths.size());
    for (std::size_t i = 0; i < run_paths.size(); ++i) {
        Record<N> rr{};
        if (readers[i]->next(rr)) fronts[i] = rr;
    }

    std::ofstream out = open_checked_output(sorted_output_path);
    try {
        while (true) {
            std::size_t min_idx = SIZE_MAX;
            for (std::size_t i = 0; i < fronts.size(); ++i) {
                if (!fronts[i].has_value()) continue;
                if (min_idx == SIZE_MAX || *fronts[i] < *fronts[min_idx]) {
                    min_idx = i;
                }
            }
            if (min_idx == SIZE_MAX) break;
            out.write(reinterpret_cast<const char*>(&(*fronts[min_idx])),
                      sizeof(Record<N>));
            Record<N> rr{};
            if (readers[min_idx]->next(rr)) fronts[min_idx] = rr;
            else fronts[min_idx].reset();
        }
        out.close();
    } catch (const std::ios_base::failure&) {
        throw_write_error(sorted_output_path);
    }

    readers.clear();  // close files before removing
    for (auto& p : run_paths) fs::remove(p);
}

template class RadixPartitionSort<1>;
template class RadixPartitionSort<2>;
template class RadixPartitionSort<3>;

// Explicit instantiations for the free-function Phase 3 helper.
template std::size_t write_btree_from_sorted_partitions<1>(
    const std::vector<std::string>&, const std::string&,
    BPT::LeafFormat, BPT::GraphStorage);
template std::size_t write_btree_from_sorted_partitions<2>(
    const std::vector<std::string>&, const std::string&,
    BPT::LeafFormat, BPT::GraphStorage);
template std::size_t write_btree_from_sorted_partitions<3>(
    const std::vector<std::string>&, const std::string&,
    BPT::LeafFormat, BPT::GraphStorage);

}  // namespace GQL
