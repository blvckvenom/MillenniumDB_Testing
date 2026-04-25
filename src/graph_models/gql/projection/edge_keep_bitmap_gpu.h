// src/graph_models/gql/projection/edge_keep_bitmap_gpu.h
//
// GPU-accelerated batch membership filter feeding EdgeFilter.
//
// `precompute_edge_filter_` (Phase B of the SERIALIZED scan pipeline)
// historically applied a per-edge has_node() membership test via two
// std::binary_search probes against ProjectionStorage::collected_nodes_.
// On papers100M-scale graphs (~1.6 B edges, 56 M unique paper nodes) that
// inner loop dominates Phase B wall clock: 2 × 1.6 B × log2(56 M) ≈ 8.3 B
// branch-heavy comparisons on a single CPU thread.
//
// Membership testing is embarrassingly parallel and bandwidth-light, so
// the same work fits the existing src/gpu/ops/ kernel pattern (sister to
// `gpu_filter.cu`'s bitset filter). EdgeKeepBitmapGpuBatcher buffers
// (edge_id, from, to) triples emitted by NativeScanner::scan_label_edge_
// _with_endpoints, and on flush dispatches one of:
//
//   1. GPU path (default when MDB_GPU_ENABLED is defined, a CUDA device is
//      visible, and the buffered batch is large enough to amortize PCIe).
//      Calls edge_keep_membership_gpu() (gpu/ops/gpu_membership.cu) which
//      runs N parallel binary searches against the sorted node array.
//
//   2. CPU fallback. Sequential for-loop calling
//      ProjectionStorage::has_node() per edge. Identical semantics to the
//      original inline lambda in precompute_edge_filter_ that this class
//      replaced — used as the correctness baseline for unit tests and
//      selected automatically when GPU is disabled / unavailable /
//      unprofitable.
//
// Either path produces the same `keep_flags[i]` array. The batcher then
// walks the captured edge_ids and calls EdgeFilter::set_kept(edge_id) on
// the survivors. EdgeFilter does the orientation routing.
//
// Env var: MDB_PROJECTION_BITMAP_GPU=0 disables the GPU path even when
// available (A/B benchmarking + safety valve). Any other value (including
// unset) keeps GPU enabled when the build supports it.
//
// Thread-safety: instances are not internally synchronised. Phase B is
// single-threaded today (the edge-scan loop in precompute_edge_filter_ is
// not parallelised) so the batcher is owned by a single scan loop. If
// Phase B parallelises across edge types in the future,
// each worker should hold its own batcher (cheap — no GPU resources are
// allocated until the first flush) and merge bits into the shared
// EdgeFilter post-finalize.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph_models/gql/projection/edge_filter.h"
#include "graph_models/object_id.h"

namespace GQL {

class ProjectionStorage;  // forward decl — full definition not needed in header

class EdgeKeepBitmapGpuBatcher {
public:
    /// Tunables. Defaults tuned on a PCIe Gen5 host with a consumer GPU:
    ///   - 1 M edges per flush ≈ 24 MB SoA, fits a single H2D burst at
    ///     PCIe Gen5 in <1 ms.
    ///   - min_edges_for_gpu = 64 K is the empirical break-even point on
    ///     RTX 5070 Ti (PCIe round-trip + kernel launch ≈ 200 µs; 64 K
    ///     edges × ~1 µs CPU per edge = 64 ms saved). Below the threshold
    ///     CPU wins on small graphs (cora_gnn ≈ 5 K edges).
    struct Config {
        std::size_t batch_capacity     = 1u << 20;  // 1 048 576 edges
        std::size_t min_edges_for_gpu  = 1u << 16;  // 65 536 — break-even
    };

    /// Construct a batcher writing into `filter`. The `storage` reference
    /// is used both for the GPU node-array snapshot and for the CPU
    /// fallback's per-edge has_node(). Both references must outlive the
    /// batcher (lifetime of one Phase B call in practice).
    EdgeKeepBitmapGpuBatcher(EdgeFilter&              filter,
                             const ProjectionStorage& storage,
                             Config                   cfg);

    /// Default-config overload — equivalent to passing `Config{}`.
    /// (Defining the default in-class is awkward because Config has
    /// member-default-initializers, which the compiler can't yet see at
    /// the point the constructor signature is parsed.)
    EdgeKeepBitmapGpuBatcher(EdgeFilter&              filter,
                             const ProjectionStorage& storage);

    /// Append one edge to the buffer. Triggers an automatic flush when
    /// the buffer reaches `cfg.batch_capacity`.
    void add(ObjectId edge_id, ObjectId from_node, ObjectId to_node);

    /// Drain any remaining buffered edges. Idempotent. Caller must invoke
    /// before reading the EdgeFilter (or the filter's finalize()).
    void flush();

    /// True iff the most recent flush() routed work to the GPU.
    /// Useful for tests + benchmarking (NOT a correctness signal).
    bool last_flush_used_gpu() const noexcept { return last_flush_used_gpu_; }

    /// Aggregate stats accumulated across all flushes (lifetime of the
    /// batcher). Useful for verbose logging from precompute_edge_filter_.
    struct Stats {
        std::size_t total_edges       = 0;
        std::size_t total_kept        = 0;
        std::size_t flushes_on_gpu    = 0;
        std::size_t flushes_on_cpu    = 0;
    };
    Stats stats() const noexcept { return stats_; }

    /// Returns true iff the build defines MDB_GPU_ENABLED, a CUDA device
    /// is visible, and the env var override is not set to "0". Cached on
    /// first call. Public so callers (and tests) can introspect.
    static bool gpu_path_available();

private:
    EdgeFilter&              filter_;
    const ProjectionStorage& storage_;
    Config                   cfg_;

    // Per-batch SoA buffers. We reserve to capacity once at construction
    // time so steady-state add() is amortized O(1) without reallocation.
    std::vector<uint64_t> edge_ids_;
    std::vector<uint64_t> from_ids_;
    std::vector<uint64_t> to_ids_;
    std::vector<uint8_t>  keep_flags_;  // sized to batch_capacity once

    bool  last_flush_used_gpu_ = false;
    Stats stats_;

    // Internal: dispatch the appropriate path. n is the current valid
    // length of edge_ids_/from_ids_/to_ids_ (≤ batch_capacity).
    void flush_n_(std::size_t n);

    // CPU implementation. Always available, identical semantics to the
    // historic inline lambda. Must be O(N log M) for N edges and M nodes.
    void run_cpu_(std::size_t n);

#ifdef MDB_GPU_ENABLED
    // GPU implementation. Returns false on any CUDA failure so the caller
    // can fall back to CPU, ensuring Phase B never aborts a projection due
    // to a transient GPU error.
    bool run_gpu_(std::size_t n);
#endif
};

}  // namespace GQL
