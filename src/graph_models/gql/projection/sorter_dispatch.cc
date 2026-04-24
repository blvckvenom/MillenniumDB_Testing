/**
 * @file sorter_dispatch.cc
 * @brief Implementation of the projection sorter dispatch facade.
 *
 * See sorter_dispatch.h for design rationale, including the signature
 * adaptation versus the original plan (we accept a `build_from_sorter`
 * callback rather than a non-existent static API).
 */

#include "graph_models/gql/projection/sorter_dispatch.h"

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "graph_models/gql/projection/external_record_sort.h"
#include "graph_models/gql/projection/radix_partition_sort.h"
#include "graph_models/gql/projection/streaming_record_buffer.h"

namespace GQL {

namespace {

// Cached backend, set once on first call.
SorterBackend cached_backend_ = SorterBackend::CLASSIC;
std::once_flag cached_backend_flag_;

void init_cached_backend() {
    const char* env = std::getenv("MDB_PROJECTION_SORTER");
    if (env == nullptr) {
        cached_backend_ = SorterBackend::CLASSIC;
        return;
    }
    const std::string v(env);
    if (v == "radix") {
        cached_backend_ = SorterBackend::RADIX;
    } else {
        // Unknown value → safe default.
        cached_backend_ = SorterBackend::CLASSIC;
    }
}

/**
 * @brief Shared CLASSIC-backend body.
 *
 * Mirrors the existing `build_index_with_streaming_sort` helper in
 * `projection_storage.cc`. Kept here so the dispatch facade can own the
 * full sort → build → trim lifecycle once callers migrate to it (wiring
 * happens in a later task; Task 1 only introduces the facade).
 */
template<std::size_t N>
std::size_t run_classic(
    StreamingRecordBuffer<N>&    input_stream,
    const std::string&           index_base_path,
    const BuildFromSorterFn<N>&  build_from_sorter,
    const std::string&           sort_temp_dir
) {
    // Finalize the buffer (flush any remaining memory to disk if spilled).
    input_stream.finalize();

    std::size_t count = 0;
    {
        // Inner scope: ensure `sorter` (and any large std::vector buffers
        // it owned during Phase 1 in-memory sort / Phase 2 K-way merge)
        // is destructed before we call malloc_trim below. Without the
        // scope, the sorter would stay alive until function exit, AFTER
        // the trim call.
        ExternalRecordSort<N> sorter(sort_temp_dir);

        // Add spill files directly to sorter (no memory copy).
        const auto& spill_paths  = input_stream.get_spill_paths();
        const auto& spill_counts = input_stream.get_spill_counts();
        for (std::size_t i = 0; i < spill_paths.size(); ++i) {
            sorter.add_run(spill_paths[i], spill_counts[i]);
        }

        // Add in-memory records (moved, not copied).
        if (input_stream.memory_buffer_size() > 0) {
            sorter.add_memory_records(input_stream.take_memory_buffer());
        }

        // Build index with streaming sort (caller-supplied B+tree writer).
        count = build_from_sorter(sorter, index_base_path);
    }  // sorter destructs here, releasing std::vector storage it owned.

    // Clear buffer (removes spill files).
    input_stream.clear();

    // Release retained heap pages to the kernel so the NEXT index build
    // starts from a low RSS baseline. Without this, glibc keeps freed
    // chunks in its free-lists; a 10-index projection on 100M+ nodes can
    // pile multiple GB of inaccessible heap before hitting the
    // virtual-memory ceiling.
#if defined(__GLIBC__)
    malloc_trim(0);
#endif

    return count;
}

}  // namespace

SorterBackend get_sorter_backend() {
    std::call_once(cached_backend_flag_, init_cached_backend);
    return cached_backend_;
}

template<std::size_t N>
std::size_t sort_and_build_index(
    StreamingRecordBuffer<N>&    input_stream,
    const std::string&           index_base_path,
    std::uint64_t                estimated_count,
    const BuildFromSorterFn<N>&  build_from_sorter,
    const std::string&           sort_temp_dir,
    BPT::LeafFormat              leaf_format
) {
    switch (get_sorter_backend()) {
        case SorterBackend::CLASSIC: {
            (void)estimated_count;  // Unused in CLASSIC.
            // CLASSIC does not directly honor `leaf_format` in this layer —
            // the caller's `build_from_sorter` callback closes over the
            // ProjectionStorage instance and reads `requested_leaf_format`
            // from there when it calls build_index_streaming. We accept the
            // parameter for signature symmetry and to satisfy the unit-
            // level API contract (T5.11b unit tests pass BITSET explicitly
            // to pin default behavior).
            (void)leaf_format;
            return run_classic<N>(
                input_stream, index_base_path, build_from_sorter, sort_temp_dir);
        }
        case SorterBackend::RADIX: {
            // RADIX backend pipeline (Tasks 5-11, Spec #5 T5.11b):
            //   Phase 1: scan + partition by top-bits of record[0]
            //   Phase 2: parallel per-partition sort (in-memory or external fallback)
            //   Phase 3: concatenate sorted partitions into BPT{V1,V2}LeafWriter
            //            / BPTDirWriter depending on `leaf_format`.
            // The `build_from_sorter` callback is unused here because RADIX's own
            // write_btree_from_sorted_partitions<N> performs the B+Tree write
            // directly (mirroring the page-level process_block pattern used by
            // ProjectionStorage::build_index_streaming). Under BITSET, the
            // resulting .leaf / .dir files are bit-identical to the CLASSIC
            // backend's output on the same input (validated by Task 13's
            // golden-compare script). Under DELTA_VARINT, the .leaf file starts
            // with the v2 header (byte 0 = 0x02).
            (void)build_from_sorter;

            typename RadixPartitionSort<N>::Config cfg;
            cfg.scratch_dir =
                (std::filesystem::path(sort_temp_dir) / ".radix_scratch").string();
            cfg.leaf_format = leaf_format;
            // Reuse sort_temp_dir's subdirectory so each projection has its own
            // scratch area — build_all_indexes_bulk passes the same
            // sort_temp_dir across all 14 index builds, but `.radix_scratch`
            // is recreated per call (ctor creates it; dtor removes it), so
            // there is no cross-call collision.

            RadixPartitionSort<N> sorter(cfg);
            sorter.scan_and_partition(input_stream, estimated_count);

            // Mirror the post-sort cleanup the CLASSIC path does:
            //   - input_stream.clear() releases buffer state (including spill
            //     files) so subsequent index builds start from a fresh state;
            //   - malloc_trim(0) releases glibc-retained heap between index
            //     builds to bound RSS growth across the 14 index sequence.
            std::size_t written = sorter.sort_and_write(index_base_path);
            input_stream.clear();
#if defined(__GLIBC__)
            malloc_trim(0);
#endif
            return written;
        }
    }
    return 0;  // unreachable
}

// Explicit instantiations.
template std::size_t sort_and_build_index<1>(
    StreamingRecordBuffer<1>&, const std::string&, std::uint64_t,
    const BuildFromSorterFn<1>&, const std::string&, BPT::LeafFormat);
template std::size_t sort_and_build_index<2>(
    StreamingRecordBuffer<2>&, const std::string&, std::uint64_t,
    const BuildFromSorterFn<2>&, const std::string&, BPT::LeafFormat);
template std::size_t sort_and_build_index<3>(
    StreamingRecordBuffer<3>&, const std::string&, std::uint64_t,
    const BuildFromSorterFn<3>&, const std::string&, BPT::LeafFormat);

}  // namespace GQL
