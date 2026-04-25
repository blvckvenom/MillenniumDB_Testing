// src/graph_models/gql/projection/batched_edge_aggregator.h
//
// Batch-shaped edge aggregator (CPU + future GPU).
//
// The legacy `StreamingEdgeAggregator` (streaming_aggregator.h) consumes one
// record at a time via an inner-loop callback, threading per-group state
// through `process()` calls. That shape composes well with the
// `ExternalEdgeSort::stream_sorted` callback API but is the wrong shape for
// data-parallel reduce-by-key on a GPU: CUB primitives
// (`cub::DeviceReduce::ReduceByKey`, `cub::DeviceSegmentedReduce`) operate
// over a contiguous SoA buffer of already-sorted records, emitting one
// representative record per group.
//
// `BatchedEdgeAggregator` exposes that contiguous-buffer contract:
//
//   1. Caller hands it a SORTED span of EdgeAggregationRecord (lexicographic
//      by (from_node, to_node, type_id)).
//   2. It walks the span once, identifies group boundaries, and emits one
//      representative record per group via the same `AggregationEmitCallback`
//      as `StreamingEdgeAggregator` (so call-sites can swap implementations
//      without touching downstream property-write logic).
//   3. The reduce dispatch is selected by `BatchedEdgeAggregator::Backend`
//      (auto / cpu / gpu). The auto path consults
//      `MDB_PROJECTION_AGGREGATION_GPU` (default off) and the runtime
//      `mdb::gpu::detect_resources()` probe, mirroring the resolution order
//      of `EdgeKeepBitmapGpuBatcher::gpu_path_available()`.
//   4. The GPU path is currently a CPU-equivalent stub — this first stage
//      establishes the data-flow contract; a later GPU-acceleration stage
//      will replace the stub body with a `cub::DeviceRadixSort` (in case the
//      caller-supplied span is not strictly sorted by the 3-field key) +
//      `cub::DeviceReduce::ReduceByKey` pair. The stub returns
//      bit-identical output to the CPU path so call-sites can wire the
//      GPU dispatch today and pick up acceleration when that later stage
//      lands without any consumer changes.
//
// The class supports the same five aggregation strategies as the streaming
// path (SINGLE/MIN/MAX/SUM/COUNT). SINGLE preserves the existing
// throw-on-duplicate semantics — when invoked under SINGLE mode, encountering
// more than one record per (from, to, type) group raises a QueryException
// matching the legacy `EdgeAggregator::process_edge` SINGLE branch text, so
// existing test assertions over the SINGLE error message remain valid.
//
// Integration note (the wiring below is NOT yet done; only unit tests
// exercise this class today):
//   `ExternalEdgeSort::stream_sorted` is the natural wiring point. When the
//   in-memory branch (`fits_in_memory()` true) runs, the existing per-record
//   callback can be replaced by a single
//   `BatchedEdgeAggregator::aggregate(span)` call. The external-sort branch
//   stays on the streaming path because the records arrive lazily through a
//   k-way merge and never form a contiguous span. That asymmetry is fine:
//   the in-memory branch is the one that hits multi-billion-record graphs
//   with VRAM-resident data (papers100M scale when using the four-level
//   topology store, which keeps hot hubs in a RAM hash / compact uint32
//   CSR / mmap sidecar / direct B+Tree tiering and returns neighbor slices
//   in O(1) for the frequently-accessed nodes) — the case GPU
//   acceleration is meant for.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

#include "graph_models/gql/projection/edge_aggregation_record.h"
#include "graph_models/gql/projection/streaming_aggregator.h"  // AggregationEmitCallback, Aggregation

namespace GQL {

/**
 * @brief Reduce-by-key aggregator over a sorted contiguous span.
 *
 * Designed as the GPU-friendly sibling of `StreamingEdgeAggregator`.
 * Unlike the streaming aggregator, which consumes one record per
 * `process()` call and mutates per-group state across calls, this class
 * processes an entire pre-sorted buffer in a single `aggregate()` call.
 *
 * Memory per group state: 64 bytes (same as streaming). Output: one
 * `emit()` callback per unique group.
 */
class BatchedEdgeAggregator {
public:
    /**
     * @brief Backend dispatch strategy.
     *
     * AUTO consults the env var + runtime CUDA probe. CPU and GPU force
     * a specific path (used by tests to pin behaviour for parity checks).
     */
    enum class Backend { AUTO, CPU, GPU };

    /**
     * @brief Constructs the aggregator.
     *
     * @param strategy Aggregation rule (COUNT/SUM/MIN/MAX/SINGLE)
     * @param emit Callback invoked once per unique (from, to, type) group
     * @param backend Dispatch strategy (default AUTO)
     */
    BatchedEdgeAggregator(Aggregation              strategy,
                          AggregationEmitCallback  emit,
                          Backend                  backend = Backend::AUTO)
        : strategy_(strategy)
        , emit_(std::move(emit))
        , backend_(backend)
    {}

    /**
     * @brief Reduce a sorted span into one emit() per group.
     *
     * Pre-condition: `records[0..n)` is non-strictly sorted by
     * (from_node, to_node, type_id). Records with the same key MUST be
     * adjacent — the caller (typically `radix_sort_edge_records()` or
     * `cub::DeviceRadixSort`) is responsible for satisfying this. A
     * debug-mode `assert()` guards the invariant on the CPU path; the
     * GPU path will skip the check (sort step on the device guarantees
     * it).
     *
     * @param records Pointer to first record
     * @param n       Number of records
     * @return true on success, false on backend dispatch failure (the
     *         CPU path never returns false; the GPU stub may return
     *         false in v2 if cudaMalloc fails).
     */
    bool aggregate(const EdgeAggregationRecord* records, std::size_t n);

    /// Convenience overload for STL containers / spans.
    template<typename Container>
    bool aggregate(const Container& c) {
        return aggregate(c.data(), c.size());
    }

    /**
     * @brief Returns the resolved backend used by the most recent call.
     *
     * Useful for tests that want to assert "GPU path actually fired".
     * Returns the requested backend pre-aggregate(); after aggregate()
     * returns, reflects the path that ran (which may differ from
     * the requested AUTO if the GPU probe disabled it or fell back).
     */
    Backend last_backend_used() const { return last_backend_used_; }

    /**
     * @brief Static helper: probe whether GPU path is available right now.
     *
     * Resolution order:
     *   1. MDB_GPU_ENABLED build flag — undefined => false.
     *   2. MDB_PROJECTION_AGGREGATION_GPU env var — "1" enables, anything
     *      else (including unset, "0", empty) disables. Mirrors
     *      MDB_PROJECTION_BITMAP_GPU's polarity but defaults OFF — the
     *      GPU aggregation body is a stub today, so the AUTO default stays
     *      on CPU until the real GPU reduce-by-key path lands.
     *   3. Runtime CUDA device probe (`mdb::gpu::detect_resources()`).
     *
     * Result is cached on first call.
     */
    static bool gpu_path_available();

private:
    bool aggregate_cpu_(const EdgeAggregationRecord* records, std::size_t n);
    bool aggregate_gpu_(const EdgeAggregationRecord* records, std::size_t n);

    Aggregation             strategy_;
    AggregationEmitCallback emit_;
    Backend                 backend_;
    Backend                 last_backend_used_ = Backend::CPU;
};

} // namespace GQL
