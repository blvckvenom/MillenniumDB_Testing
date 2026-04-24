#pragma once

/**
 * @file sorter_dispatch.h
 * @brief Thin dispatch facade that selects a sorting backend for projection
 *        B+tree index building.
 *
 * Introduced for the Radix-Partition Sort campaign. Reads the environment
 * variable MDB_PROJECTION_SORTER once on first call:
 *
 *   - unset (or any unrecognized value)  → SorterBackend::CLASSIC  (default)
 *   - "radix"                            → SorterBackend::RADIX    (wired later)
 *
 * In the M1 scaffolding step, the RADIX case falls through to CLASSIC so that
 * the facade is a no-op behavioral change. The new RadixPartitionSort backend
 * is wired into the RADIX case in a later task (Task 12) once it passes unit
 * tests.
 *
 * ## Signature adaptation (recorded here for future reviewers)
 *
 * The design plan originally specified a hypothetical static API
 * `ExternalRecordSort<N>::build_index_from_stream(buffer, base_path)` which
 * does NOT exist in this codebase. The real flow (see
 * `ProjectionStorage::build_all_indexes_bulk`) is:
 *
 *   1. Drain a `StreamingRecordBuffer<N>` (spill paths + in-memory records)
 *      into an `ExternalRecordSort<N>` instance.
 *   2. Call a member B+tree builder (`ProjectionStorage::build_index_streaming`)
 *      that consumes `sorter.stream_sorted(...)` and writes `.leaf` / `.dir`
 *      pages with inline deduplication.
 *
 * To keep the facade a minimal, honest dispatch layer without duplicating the
 * B+tree-build logic, `sort_and_build_index` accepts a `build_from_sorter`
 * callback that performs step 2. Callers already hold a reference to a
 * `ProjectionStorage` instance, so they pass a lambda like:
 *
 *   [this](ExternalRecordSort<N>& s, const std::string& p) {
 *       return this->build_index_streaming<N>(s, p);
 *   }
 *
 * This mirrors the existing pattern in `projection_storage.cc`. A later task
 * can, if desired, promote `build_index_streaming` to a free function and drop
 * the callback parameter.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "graph_models/gql/projection/external_record_sort.h"
#include "graph_models/gql/projection/streaming_record_buffer.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"

namespace GQL {

enum class SorterBackend {
    CLASSIC,  ///< Existing ExternalRecordSort path (default).
    RADIX,    ///< New RadixPartitionSort path (M5+). Selected via env var.
};

/**
 * @brief Reads the MDB_PROJECTION_SORTER env var.
 *        Cached after first call. Unknown values fall back to CLASSIC.
 *        Thread-safe (std::call_once).
 */
SorterBackend get_sorter_backend();

/**
 * @brief Callback used by `sort_and_build_index` to perform the B+tree build
 *        step once records are streamed out of the selected sorter backend.
 *
 * @tparam N Record arity.
 */
template<std::size_t N>
using BuildFromSorterFn =
    std::function<std::size_t(ExternalRecordSort<N>&, const std::string&)>;

/**
 * @brief Unified facade for sorting records and building a B+Tree index.
 *
 * Replaces direct `ExternalRecordSort<N>` construction at the caller site.
 * In CLASSIC mode, performs the same work as the pre-refactor
 * `build_index_with_streaming_sort` helper: finalize the buffer, transfer
 * spills + memory into an `ExternalRecordSort<N>`, run the caller-supplied
 * `build_from_sorter` (typically a wrapper over
 * `ProjectionStorage::build_index_streaming`), then release the sorter and
 * trim glibc heap on supported platforms.
 *
 * @tparam N                Record arity (1, 2, or 3 for the projection subsystem).
 * @param  input_stream     StreamingRecordBuffer to consume (drained into
 *                          the sorter). Its spill files and memory buffer
 *                          are moved into the sorter and cleared.
 * @param  index_base_path  Path prefix for `.leaf` and `.dir` output files.
 * @param  estimated_count  Reserved for the RADIX backend (partition-count
 *                          adaptive calculation, wired in Task 12). Unused
 *                          by the CLASSIC backend.
 * @param  build_from_sorter Callback that consumes the populated sorter and
 *                          writes the B+tree index. Typically a lambda
 *                          forwarding to `ProjectionStorage::build_index_streaming`.
 * @param  sort_temp_dir    Directory for the sorter's temporary files.
 *                          Usually `<projection_dir>/sort_tmp`.
 * @param  leaf_format      Spec #5 T5.11b. Selects the on-disk leaf encoding.
 *                          BITSET (default) preserves pre-Spec-#5 byte-
 *                          identical behavior; DELTA_VARINT opts into the
 *                          zigzag-delta + LEB128 varint v2 encoding. The
 *                          RADIX backend propagates this into
 *                          RadixPartitionSort<N>::Config.leaf_format; the
 *                          CLASSIC backend relies on the caller-supplied
 *                          `build_from_sorter` callback picking up the same
 *                          per-projection preset from ProjectionStorage
 *                          (the callback closes over `this` and reads
 *                          `requested_leaf_format` directly, so the value
 *                          does not need to be re-threaded through the
 *                          callback signature).
 * @return Total unique records written to the B+Tree (return value of the
 *         caller's build callback).
 */
template<std::size_t N>
std::size_t sort_and_build_index(
    StreamingRecordBuffer<N>&    input_stream,
    const std::string&           index_base_path,
    std::uint64_t                estimated_count,
    const BuildFromSorterFn<N>&  build_from_sorter,
    const std::string&           sort_temp_dir,
    BPT::LeafFormat              leaf_format = BPT::LeafFormat::BITSET
);

// Explicit instantiations (declared here, defined in .cc).
extern template std::size_t sort_and_build_index<1>(
    StreamingRecordBuffer<1>&, const std::string&, std::uint64_t,
    const BuildFromSorterFn<1>&, const std::string&, BPT::LeafFormat);
extern template std::size_t sort_and_build_index<2>(
    StreamingRecordBuffer<2>&, const std::string&, std::uint64_t,
    const BuildFromSorterFn<2>&, const std::string&, BPT::LeafFormat);
extern template std::size_t sort_and_build_index<3>(
    StreamingRecordBuffer<3>&, const std::string&, std::uint64_t,
    const BuildFromSorterFn<3>&, const std::string&, BPT::LeafFormat);

}  // namespace GQL
