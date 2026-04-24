// src/graph_models/gql/projection/radix_partition_sort.cc
#include "graph_models/gql/projection/radix_partition_sort.h"

#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "graph_models/gql/projection/parallel_scan_partitioner.h"
#include "graph_models/gql/projection/partition_file.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/bplus_tree/bpt_mem_import.h"
#include "storage/page/page.h"

namespace fs = std::filesystem;

namespace GQL {

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
// BITSET-backed concatenation (V1 leaf format, pre-Spec-#5).
// Preserves pre-Spec-#5 byte-identical output.
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

    // Scratch buffer: one leaf page worth of "bitset + record bytes".
    constexpr std::size_t max_buffer_size =
        N + max_records_per_leaf * sizeof(uint64_t) * N;
    auto leaf_buffer = std::make_unique<char[]>(max_buffer_size);

    std::vector<Record<N>> page_records;
    page_records.reserve(max_records_per_leaf);

    Record<N>   prev_record{};
    bool        has_prev         = false;
    std::size_t unique_count     = 0;
    std::size_t leaf_page_number = 0;

    std::bitset<N * 8> no_compression;  // all zeros ⇒ no compression applied

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

        // Format: [N-byte bitset (zeros)] + [packed record bytes].
        unsigned long bits_ul = no_compression.to_ulong();
        std::memcpy(leaf_buffer.get(), &bits_ul, N);
        std::memcpy(leaf_buffer.get() + N,
                    reinterpret_cast<const char*>(page_records.data()),
                    page_records.size() * sizeof(uint64_t) * N);

        leaf_writer.process_block(
            leaf_buffer.get(),
            static_cast<uint32_t>(page_records.size()),
            no_compression,
            next_page);

        ++leaf_page_number;
        page_records.clear();
    };

    // Stream across all partitions in order. Each partition is already sorted,
    // and radix partitioning ensures partition_i < partition_{i+1} key-wise,
    // so the concatenation is globally sorted.
    for (const auto& path : sorted_partition_paths) {
        if (!fs::exists(path)) continue;
        typename PartitionFile<N>::Reader reader(path);
        Record<N> r{};
        while (reader.next(r)) {
            if (has_prev && r == prev_record) {
                continue;  // dedup
            }
            prev_record = r;
            has_prev    = true;
            ++unique_count;

            page_records.push_back(r);
            if (page_records.size() >= max_records_per_leaf) {
                write_leaf_page(/*is_last_page=*/false);
            }
        }
    }

    // Flush the final partial page (marks next_page=0 as terminator).
    write_leaf_page(/*is_last_page=*/true);

    return unique_count;
}

// DELTA_VARINT-backed concatenation (V2 leaf format, Spec #5).
// Streams records one at a time into BPTLeafV2Writer, which handles
// page boundaries internally (variable per-record bytes via zigzag
// deltas). Directory entries are emitted on page boundary crossings.
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

    for (const auto& path : sorted_partition_paths) {
        if (!fs::exists(path)) continue;
        typename PartitionFile<N>::Reader reader(path);
        Record<N> r{};
        while (reader.next(r)) {
            if (has_prev && r == prev_record) {
                continue;  // dedup
            }
            prev_record = r;
            has_prev    = true;
            ++unique_count;

            const bool started_new_page = leaf_writer.append_record(r);
            if (started_new_page) {
                // `r` is now the first record of page index
                // current_page_index(). B+Tree convention: the very first
                // leaf (index 0) does not get a dir entry. Every boundary
                // crossing happens at page_count >= 1, so we always emit.
                ++page_count;
                dir_writer.bulk_insert(
                    &r,
                    0,
                    static_cast<int32_t>(leaf_writer.current_page_index()));
            }
        }
    }

    // Finalize: flush the tail page with next_leaf = 0.
    leaf_writer.finalize();

    return unique_count;
}

// CSR_HYBRID concatenation (v3 leaf format, Spec #8). Streams records
// from globally-ordered partition files into BPTLeafCSRWriter, which
// groups by record[0] (src) and emits chain-head + optional continuation
// pages. A trivial root .dir (key_count=0) is emitted; lookups route to
// leaf 0 and walk the next_leaf chain. Correctness-first MVP: src-keyed
// directory routing is deferred to T8.12's Gate D bench optimization.
//
// N == 3 is the only instantiation that produces sensible CSR output
// (src, dst, edge_id). The writer drops edge_id (design §3.4 notes the
// single-stream v3 payload does not encode it) — on read, v3 returns 0
// for position 2, which GnnProjectionAdapter / TopologyAccessor callers
// tolerate.
template<std::size_t N>
static std::size_t write_btree_from_sorted_partitions_csr_(
    const std::vector<std::string>& sorted_partition_paths,
    const std::string& base_path)
{
    if constexpr (N == 3) {
        // Spec #8-B task #1: enable edge_id stream emission on edge
        // indexes (N == 3 only; property indexes route through the
        // non-CSR sibling). See sorter_dispatch.cc build_index_csr_from_sorter_
        // for the full rationale.
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

        for (const auto& path : sorted_partition_paths) {
            if (!fs::exists(path)) continue;
            typename PartitionFile<N>::Reader reader(path);
            Record<N> r{};
            while (reader.next(r)) {
                if (has_prev && r == prev_record) continue;
                prev_record = r;
                has_prev    = true;
                ++unique_count;
                leaf_writer.append(r);
            }
        }

        leaf_writer.flush_finalize();
        return unique_count;
    } else {
        // Spec #8 scopes CSR_HYBRID to edge indexes (N==3). Reaching here
        // is a caller-side invariant violation — radix config should have
        // kept graph_storage at BTREE for N==2 indexes. Return 0 as a
        // defensive fallback; the upstream ProjectionStorage dispatch
        // enforces the scope at config time.
        (void)sorted_partition_paths;
        (void)base_path;
        return 0;
    }
}

// Dispatch shim. Default leaf_format is BITSET to preserve pre-Spec-#5
// behavior for any caller not yet plumbed through T5.11b.
template<std::size_t N>
std::size_t write_btree_from_sorted_partitions(
    const std::vector<std::string>& sorted_partition_paths,
    const std::string& base_path,
    BPT::LeafFormat leaf_format,
    BPT::GraphStorage graph_storage)
{
    // Spec #8 T8.9 — CSR_HYBRID supersedes leaf_format for its scope
    // (edge indexes, N==3). The helper itself gates on N via constexpr.
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
}

template<std::size_t N>
RadixPartitionSort<N>::~RadixPartitionSort() {
    // Exception-safe cleanup of any remaining scratch files.
    try { fs::remove_all(config_.scratch_dir); } catch (...) {}
}

template<std::size_t N>
std::uint32_t RadixPartitionSort<N>::radix_bucket(const Record<N>& r) const {
    // Top bits of record[0] → bucket index.
    if (num_partitions_ <= 1) return 0;
    // bit_width(x) = 64 - __builtin_clzll(x) for x > 0 (portable C++17 alt).
    int bit_width = 64 - __builtin_clzll(static_cast<unsigned long long>(num_partitions_ - 1));
    int shift = 64 - bit_width;
    return static_cast<std::uint32_t>(r[0] >> shift);
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

    partitioner.run([&](auto emit) {
        // Single-threaded drain of the input buffer for now. The full
        // TBB-driven B+Tree scan integration lands in the projection
        // builder wiring task (Task 12); here we consume a streaming
        // buffer directly to satisfy the unit tests.
        input.begin_iteration();
        while (input.has_next()) {
            emit(input.next());
        }
    });

    partition_paths_ = partitioner.collect_merged_partition_paths();
    return num_partitions_;
}

template<std::size_t N>
std::size_t RadixPartitionSort<N>::sort_and_write(
    const std::string& output_base_path)
{
    std::size_t total_written = 0;

    std::size_t num_workers = compute_num_workers(
        std::thread::hardware_concurrency(),
        (config_.num_scan_threads > 0)
            ? config_.num_scan_threads
            : std::thread::hardware_concurrency() / 2,
        4ULL * 1024 * 1024 * 1024,  // 4 GB default memory budget
        config_.worker_memory_budget,
        config_.num_workers);
    (void)num_workers;  // passed to TBB via default arena; grain 1 already set

    // Dispatch partitions across workers via tbb::parallel_for.
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
                // Release free heap pages to the kernel between sorts.
#if defined(__GLIBC__)
                malloc_trim(0);
#endif
            }
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
    // (`.leaf` + `.dir`) is now the authoritative output.
    for (const auto& p : sorted_paths) {
        try { fs::remove(p); } catch (...) { /* best-effort cleanup */ }
    }

    return total_written;
}

template<std::size_t N>
void RadixPartitionSort<N>::sort_partition_in_memory(
    std::size_t partition_idx, const std::string& sorted_output_path)
{
    std::vector<Record<N>> buffer;
    typename PartitionFile<N>::Reader reader(partition_paths_[partition_idx]);
    Record<N> r{};
    while (reader.next(r)) {
        buffer.push_back(r);
    }
    std::sort(buffer.begin(), buffer.end());
    std::ofstream out(sorted_output_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(buffer.data()),
              buffer.size() * sizeof(Record<N>));
}

template<std::size_t N>
void RadixPartitionSort<N>::sort_partition_external(
    std::size_t partition_idx, const std::string& sorted_output_path)
{
    // Defensive path: partition exceeds worker_memory_budget. Chunk-sort
    // to intermediate "run" files, then K-way merge.
    std::vector<Record<N>> chunk;
    std::size_t chunk_cap = config_.worker_memory_budget / sizeof(Record<N>) / 2;
    if (chunk_cap == 0) chunk_cap = 1;  // minimum: handle tiny worker_memory_budget in tests
    chunk.reserve(chunk_cap);

    std::vector<std::string> run_paths;
    {
        typename PartitionFile<N>::Reader reader(partition_paths_[partition_idx]);
        Record<N> r{};
        std::size_t run_idx = 0;
        while (reader.next(r)) {
            chunk.push_back(r);
            if (chunk.size() >= chunk_cap) {
                std::sort(chunk.begin(), chunk.end());
                std::string run_path = sorted_output_path + ".run_" + std::to_string(run_idx++);
                std::ofstream run_out(run_path, std::ios::binary);
                run_out.write(reinterpret_cast<const char*>(chunk.data()),
                              chunk.size() * sizeof(Record<N>));
                run_paths.push_back(run_path);
                chunk.clear();
            }
        }
        if (!chunk.empty()) {
            std::sort(chunk.begin(), chunk.end());
            std::string run_path = sorted_output_path + ".run_" + std::to_string(run_idx++);
            std::ofstream run_out(run_path, std::ios::binary);
            run_out.write(reinterpret_cast<const char*>(chunk.data()),
                          chunk.size() * sizeof(Record<N>));
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

    std::ofstream out(sorted_output_path, std::ios::binary);
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
