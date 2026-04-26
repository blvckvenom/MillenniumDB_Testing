// src/graph_models/gql/projection/batched_edge_aggregator.cc
//
// Spec #26 — implementation of BatchedEdgeAggregator (see header for design).
//
// CPU path: linear scan over a sorted EdgeAggregationRecord buffer, tracking
// per-group state in 64 stack-resident bytes; emits one record per unique
// (from_node, to_node, type_id) group via the user-supplied callback. The
// reduction semantics match `StreamingEdgeAggregator::aggregate_into_group`
// exactly so a side-by-side test (see batched_edge_aggregator_test.cc) can
// assert bit-identical output.
//
// GPU path: a CPU-equivalent stub today. The contract is intentional: the
// stub returns the same output as the CPU path so any consumer that wires
// `Backend::AUTO` (or `Backend::GPU` once enabled) sees no behavioural
// change pre-v2. Spec #26 v2 swaps the stub body for `cub::DeviceRadixSort`
// + `cub::DeviceReduce::ReduceByKey` (or `DeviceSegmentedReduce` once
// segment offsets are computed) without touching consumers.

#include "graph_models/gql/projection/batched_edge_aggregator.h"

#include <cassert>
#include <cstdio>
#include <limits>

#include "gpu/gpu_device.h"
#include "query/exceptions.h"

namespace GQL {

// ---------------------------------------------------------------------------
// gpu_path_available(): cached probe of build flag + env var + CUDA device.
// ---------------------------------------------------------------------------
//
// Mirrors EdgeKeepBitmapGpuBatcher::gpu_path_available() for consistency,
// with one polarity flip: this env var defaults OFF because the GPU body is
// a stub today. Once Spec #26 v2 lands, the natural follow-up flips the
// default to ON (matching MDB_PROJECTION_BITMAP_GPU's "1" = on-by-default
// convention).
bool BatchedEdgeAggregator::gpu_path_available() {
#ifndef MDB_GPU_ENABLED
    return false;
#else
    static const bool cached = []() {
        const char* env = std::getenv("MDB_PROJECTION_AGGREGATION_GPU");
        if (env == nullptr || env[0] != '1' || env[1] != '\0') {
            // Default-off + any non-"1" value disables. We deliberately do
            // NOT treat unset == on (unlike MDB_PROJECTION_BITMAP_GPU) so
            // the stub doesn't masquerade as a working GPU path during
            // Spec #26 v1 rollout.
            return false;
        }
        const auto res = mdb::gpu::detect_resources();
        return res.has_gpu;
    }();
    return cached;
#endif
}

// ---------------------------------------------------------------------------
// aggregate(): top-level dispatch. Resolves backend then delegates.
// ---------------------------------------------------------------------------
bool BatchedEdgeAggregator::aggregate(const EdgeAggregationRecord* records,
                                      std::size_t                   n)
{
    if (n == 0) {
        last_backend_used_ = Backend::CPU;
        return true;
    }

    Backend resolved = backend_;
    if (resolved == Backend::AUTO) {
        resolved = gpu_path_available() ? Backend::GPU : Backend::CPU;
    }

    if (resolved == Backend::GPU) {
        last_backend_used_ = Backend::GPU;
        if (aggregate_gpu_(records, n)) {
            return true;
        }
        // GPU path declined — fall through to CPU. This mirrors the
        // EdgeKeepBitmapGpuBatcher fallback contract: transient cudaMalloc
        // failures under VRAM pressure should NOT abort the projection
        // build; the CPU path always works.
        last_backend_used_ = Backend::CPU;
        return aggregate_cpu_(records, n);
    }

    last_backend_used_ = Backend::CPU;
    return aggregate_cpu_(records, n);
}

// ---------------------------------------------------------------------------
// aggregate_cpu_(): single linear pass over the sorted buffer.
// ---------------------------------------------------------------------------
//
// Reduction state: 64 bytes on stack, no heap allocation. We deliberately
// duplicate the per-strategy switch from StreamingEdgeAggregator rather
// than constructing one and feeding records through process(): the two
// classes serve different shapes (callback-driven vs span-driven), and
// inlining the switch here lets the compiler vectorise the COUNT path
// (the most common one, since wildcard graph_project defaults to it).
bool BatchedEdgeAggregator::aggregate_cpu_(
    const EdgeAggregationRecord* records,
    std::size_t                   n)
{
    // Bootstrap with the first record.
    uint64_t cur_from   = records[0].from_node;
    uint64_t cur_to     = records[0].to_node;
    uint64_t cur_type   = records[0].type_id;
    uint64_t count      = 1;
    uint64_t rep_edge   = records[0].edge_id;
    double   sum_value  = unpack_bits_to_double(records[0].property_bits);
    double   min_value  = sum_value;
    double   max_value  = sum_value;

    auto emit_current = [&]() {
        double agg_value = 0.0;
        switch (strategy_) {
            case Aggregation::COUNT:  agg_value = static_cast<double>(count); break;
            case Aggregation::SUM:    agg_value = sum_value; break;
            case Aggregation::MIN:    agg_value = min_value; break;
            case Aggregation::MAX:    agg_value = max_value; break;
            case Aggregation::SINGLE: agg_value = 0.0; break;
        }
        emit_(rep_edge, count, agg_value);
    };

    for (std::size_t i = 1; i < n; ++i) {
        const EdgeAggregationRecord& rec = records[i];

        // Debug-mode invariant: caller MUST hand us a sorted buffer.
        // Records with the same key MUST be adjacent. Violations corrupt
        // group counts (a duplicate group would emit twice).
        assert((rec.from_node > cur_from) ||
               (rec.from_node == cur_from && rec.to_node > cur_to) ||
               (rec.from_node == cur_from && rec.to_node == cur_to &&
                rec.type_id >= cur_type));

        const bool same_group = (rec.from_node == cur_from &&
                                 rec.to_node   == cur_to &&
                                 rec.type_id   == cur_type);

        if (!same_group) {
            emit_current();

            // Start a new group.
            cur_from = rec.from_node;
            cur_to   = rec.to_node;
            cur_type = rec.type_id;
            count    = 1;
            rep_edge = rec.edge_id;

            const double v = unpack_bits_to_double(rec.property_bits);
            sum_value = v;
            min_value = v;
            max_value = v;
            continue;
        }

        // Same group — accumulate per strategy.
        ++count;

        switch (strategy_) {
            case Aggregation::COUNT:
                // Just counting; property irrelevant.
                break;

            case Aggregation::SUM:
                sum_value += unpack_bits_to_double(rec.property_bits);
                break;

            case Aggregation::MIN: {
                const double v = unpack_bits_to_double(rec.property_bits);
                if (v < min_value) {
                    min_value = v;
                    rep_edge  = rec.edge_id;
                }
                break;
            }

            case Aggregation::MAX: {
                const double v = unpack_bits_to_double(rec.property_bits);
                if (v > max_value) {
                    max_value = v;
                    rep_edge  = rec.edge_id;
                }
                break;
            }

            case Aggregation::SINGLE:
                // Match the legacy EdgeAggregator::process_edge SINGLE
                // branch text so existing test assertions over the error
                // message remain valid (cf.
                // native_projection_builder.cc:367).
                throw ::QueryException(
                    "Parallel edges detected but aggregation is SINGLE (the default).\n"
                    "SINGLE refuses to drop data silently — choose how to collapse parallels:\n"
                    "  aggregation: 'COUNT'  -> emit one edge with synthetic `_count`\n"
                    "  aggregation: 'SUM'    -> sum `aggregationProperty` across parallels\n"
                    "  aggregation: 'MIN'    -> keep edge with smallest property value\n"
                    "  aggregation: 'MAX'    -> keep edge with largest property value\n"
                    "\n"
                    "Example:\n"
                    "  CALL graph_project('g', nodes, edges, {aggregation: 'COUNT'})\n"
                );
        }
    }

    // Emit the trailing group.
    emit_current();
    return true;
}

// ---------------------------------------------------------------------------
// aggregate_gpu_(): Spec #26 v1 stub. CPU-equivalent output.
// ---------------------------------------------------------------------------
//
// The stub deliberately re-uses aggregate_cpu_() to guarantee bit-identical
// output. Spec #26 v2 will replace this body with:
//
//   1. cudaMalloc + cudaMemcpy of the SoA layout (5 parallel uint64
//      arrays). EdgeAggregationRecord's static_assert (5 contiguous
//      uint64_t — already enforced by external_edge_sort.h) makes this
//      a single H2D transfer.
//   2. cub::DeviceRadixSort::SortKeys (or SortPairs) over a packed
//      128-bit key (from << 64 | to) with type_id as a secondary
//      sort pass — radix sort is stable so two passes give the
//      lexicographic order required by the reduce-by-key step.
//   3. cub::DeviceReduce::ReduceByKey on the sorted stream with a
//      strategy-dependent op (cuda::std::plus for SUM/COUNT,
//      thrust::minimum for MIN, thrust::maximum for MAX).
//      Representative-edge tracking (MIN/MAX path) needs a custom op
//      that compares (value, edge_id) pairs lexicographically by
//      value with edge_id as the tie-breaker payload — this is the
//      one non-trivial bit and the main reason the v2 wiring is
//      deferred to a separate spec.
//   4. cudaMemcpy of the unique-key array + reduced values back to
//      host, then a single host loop that fires emit() per group.
//
// The stub returns true (matching the CPU path's success contract) so
// AUTO callers can already exercise the GPU dispatch wiring; tests can
// pin Backend::GPU to assert the dispatch fires (last_backend_used_).
bool BatchedEdgeAggregator::aggregate_gpu_(
    const EdgeAggregationRecord* records,
    std::size_t                   n)
{
    return aggregate_cpu_(records, n);
}

} // namespace GQL
