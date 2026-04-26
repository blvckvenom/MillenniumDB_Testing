// src/graph_models/gql/projection/edge_keep_bitmap_gpu.cc
//
// Spec #27 — implementation of EdgeKeepBitmapGpuBatcher (see header for
// rationale + design).

#include "graph_models/gql/projection/edge_keep_bitmap_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "graph_models/gql/projection/projection_storage.h"
#include "gpu/gpu_device.h"

#ifdef MDB_GPU_ENABLED
#include <cuda_runtime.h>

namespace mdb::gpu {
// Forward declaration of the wrapper defined in src/gpu/ops/gpu_membership.cu.
// We do not expose a public header from src/gpu/ops/ because the existing
// kernels (gpu_filter.cu / gpu_transform.cu) follow the same convention —
// callers (test_gpu_ops.cc, edge_keep_bitmap_gpu.cc) forward-declare
// locally so the GPU op surface stays opt-in per consumer.
bool edge_keep_membership_gpu(
    const uint64_t* d_from,
    const uint64_t* d_to,
    const uint64_t* d_sorted_nodes,
    uint64_t        num_nodes,
    uint8_t*        d_keep_flags,
    uint64_t        num_edges);
}  // namespace mdb::gpu
#endif

namespace GQL {

// ---------------------------------------------------------------------------
// gpu_path_available(): cached probe of build flag + runtime device + env var.
// ---------------------------------------------------------------------------
//
// Resolution order (first decisive answer wins):
//   1. MDB_GPU_ENABLED build flag — if undefined, GPU path is impossible.
//   2. MDB_PROJECTION_BITMAP_GPU env var — set to "0" disables (A/B switch).
//   3. mdb::gpu::detect_resources().has_gpu — runtime CUDA device probe.
//
// The result is cached so subsequent flush()es don't re-probe the env or
// the driver. cuda{Get,Set}Device() at the rate of one call per flush
// would still be cheap (~µs) but pointless — neither input changes during
// a single Phase B run.
bool EdgeKeepBitmapGpuBatcher::gpu_path_available() {
#ifndef MDB_GPU_ENABLED
    return false;
#else
    static const bool cached = []() {
        if (const char* env = std::getenv("MDB_PROJECTION_BITMAP_GPU")) {
            // Only "0" (the documented disable value) is treated as an opt-out.
            // Any other value — including empty string — keeps GPU on, matching
            // the convention of MDB_PROJECTION_SORTER (CLAUDE.md L240).
            if (env[0] == '0' && env[1] == '\0') {
                return false;
            }
        }
        const auto res = mdb::gpu::detect_resources();
        return res.has_gpu;
    }();
    return cached;
#endif
}

// ---------------------------------------------------------------------------
// Constructor. Pre-reserves the SoA buffers to batch_capacity so add() is
// O(1) amortized with no allocator pressure during Phase B (which is the
// hottest CPU loop in the projection pipeline on papers100M).
// ---------------------------------------------------------------------------
EdgeKeepBitmapGpuBatcher::EdgeKeepBitmapGpuBatcher(
    EdgeFilter&              filter,
    const ProjectionStorage& storage,
    Config                   cfg)
    : filter_(filter)
    , storage_(storage)
    , cfg_(cfg)
{
    edge_ids_.reserve(cfg_.batch_capacity);
    from_ids_.reserve(cfg_.batch_capacity);
    to_ids_.reserve(cfg_.batch_capacity);
    keep_flags_.assign(cfg_.batch_capacity, 0);
}

EdgeKeepBitmapGpuBatcher::EdgeKeepBitmapGpuBatcher(
    EdgeFilter&              filter,
    const ProjectionStorage& storage)
    : EdgeKeepBitmapGpuBatcher(filter, storage, Config{})
{}

// ---------------------------------------------------------------------------
// add() — buffer one edge; auto-flush when full.
// ---------------------------------------------------------------------------
void EdgeKeepBitmapGpuBatcher::add(ObjectId edge_id,
                                   ObjectId from_node,
                                   ObjectId to_node) {
    edge_ids_.push_back(edge_id.id);
    from_ids_.push_back(from_node.id);
    to_ids_.push_back(to_node.id);

    if (edge_ids_.size() >= cfg_.batch_capacity) {
        flush_n_(edge_ids_.size());
        // Reset SoA buffers in place — capacity is preserved by clear().
        edge_ids_.clear();
        from_ids_.clear();
        to_ids_.clear();
    }
}

void EdgeKeepBitmapGpuBatcher::flush() {
    const std::size_t n = edge_ids_.size();
    if (n == 0) return;
    flush_n_(n);
    edge_ids_.clear();
    from_ids_.clear();
    to_ids_.clear();
}

// ---------------------------------------------------------------------------
// flush_n_() — single dispatch point. Picks GPU or CPU per heuristic, then
// scatters surviving edges into the EdgeFilter.
//
// Heuristic: GPU if (a) gpu_path_available() and (b) the batch is large
// enough to amortize PCIe round-trip + kernel launch. Tiny graphs
// (cora_gnn: 5 K edges total) always go through CPU even when GPU is
// healthy — H2D + kernel + D2H overhead is ~200 µs on celebi, vs ~5 ms
// for the entire CPU loop on 5 K edges, so GPU is a regression at that
// scale. The threshold lives in cfg_.min_edges_for_gpu so tests can
// override it (see edge_keep_bitmap_gpu_test.cc).
// ---------------------------------------------------------------------------
void EdgeKeepBitmapGpuBatcher::flush_n_(std::size_t n) {
    last_flush_used_gpu_ = false;

#ifdef MDB_GPU_ENABLED
    if (gpu_path_available() && n >= cfg_.min_edges_for_gpu) {
        if (run_gpu_(n)) {
            last_flush_used_gpu_ = true;
            ++stats_.flushes_on_gpu;
            // Scatter keep_flags_ into the EdgeFilter. Single-threaded
            // because EdgeFilter::set_kept is not internally synchronised.
            for (std::size_t i = 0; i < n; ++i) {
                if (keep_flags_[i]) {
                    filter_.set_kept(ObjectId(edge_ids_[i]));
                    ++stats_.total_kept;
                }
            }
            stats_.total_edges += n;
            return;
        }
        // GPU path threw — fall through to CPU. We do NOT cache the
        // failure as "GPU broken" because cudaMalloc can transiently fail
        // under VRAM pressure (e.g. concurrent GNN training). The next
        // flush gets a fresh attempt.
    }
#endif

    run_cpu_(n);
    ++stats_.flushes_on_cpu;
    stats_.total_edges += n;
}

// ---------------------------------------------------------------------------
// run_cpu_(): identical semantics to the historic inline lambda inside
// precompute_edge_filter_. Runs the two has_node() probes per edge in
// order; ProjectionStorage::has_node() does std::binary_search against
// collected_nodes_ (sorted by Phase A's finalize_node_scan).
// ---------------------------------------------------------------------------
void EdgeKeepBitmapGpuBatcher::run_cpu_(std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const ObjectId from_oid(from_ids_[i]);
        const ObjectId to_oid(to_ids_[i]);
        const bool has_from = storage_.has_node(from_oid);
        const bool has_to   = has_from && storage_.has_node(to_oid);
        if (has_from && has_to) {
            filter_.set_kept(ObjectId(edge_ids_[i]));
            ++stats_.total_kept;
        }
    }
}

#ifdef MDB_GPU_ENABLED
// ---------------------------------------------------------------------------
// run_gpu_(): SoA H2D + kernel + D2H. Returns false on any CUDA failure;
// the caller (flush_n_) silently falls back to the CPU path.
//
// Memory plan (per flush):
//   - d_from / d_to:       n × uint64 each  → 2 × 8 × n bytes
//   - d_sorted_nodes:      M × uint64       (uploaded once per batcher? No —
//                          we re-upload per flush for now since
//                          collected_nodes_ may grow across flushes if
//                          finalize_node_scan() is called between batches.
//                          In Phase B, finalize_node_scan has already been
//                          called by Phase A, so the array is stable, and
//                          re-uploading is just wasted bandwidth.
//                          OPTIMISATION DEFERRED to a follow-up spec — the
//                          simple version is correct and the bandwidth cost
//                          is small relative to the kernel work.)
//   - d_keep_flags:        n × uint8         → n bytes
//
// For papers100M (1.6 B edges, 56 M nodes): one flush of 1 M edges is
// 16 MB H2D for from+to, 448 MB H2D for sorted_nodes (re-uploaded), 1 MB
// D2H. Total per flush ≈ 465 MB at 64 GB/s = ~7 ms transfer + kernel
// (binary search × 1 M × log2(56 M) ≈ 26 M ops, fits well in <1 ms on
// 5070 Ti). 1.6 B / 1 M = 1 600 flushes × 8 ms = 13 s of GPU work for
// the entire Phase B. Compare to CPU's ~5-15 min on the same hardware.
// ---------------------------------------------------------------------------
bool EdgeKeepBitmapGpuBatcher::run_gpu_(std::size_t n) {
    // Snapshot the sorted node array. has_node() requires
    // collected_nodes_sorted_ at this point — Spec #2 invariant I1
    // (Phase A's finalize_node_scan runs strictly before Phase B).
    const std::vector<uint64_t>& nodes = storage_.collected_nodes();
    if (nodes.empty()) {
        // Empty projection: nothing keeps. Set all flags to 0 in-place
        // (no kernel needed) and let the caller scatter (which will be
        // a no-op).
        std::memset(keep_flags_.data(), 0, n);
        return true;
    }

    uint64_t* d_from         = nullptr;
    uint64_t* d_to           = nullptr;
    uint64_t* d_sorted_nodes = nullptr;
    uint8_t*  d_keep_flags   = nullptr;

    auto cleanup = [&]() {
        if (d_from)         cudaFree(d_from);
        if (d_to)           cudaFree(d_to);
        if (d_sorted_nodes) cudaFree(d_sorted_nodes);
        if (d_keep_flags)   cudaFree(d_keep_flags);
    };

    auto cuda_check = [&](cudaError_t err, const char* what) {
        if (err == cudaSuccess) return true;
        std::fprintf(stderr,
                     "EdgeKeepBitmapGpuBatcher: CUDA %s failed (%s); "
                     "falling back to CPU path for this flush\n",
                     what, cudaGetErrorString(err));
        cleanup();
        return false;
    };

    if (!cuda_check(cudaMalloc(&d_from, n * sizeof(uint64_t)), "cudaMalloc(d_from)"))
        return false;
    if (!cuda_check(cudaMalloc(&d_to, n * sizeof(uint64_t)), "cudaMalloc(d_to)"))
        return false;
    if (!cuda_check(cudaMalloc(&d_sorted_nodes, nodes.size() * sizeof(uint64_t)),
                    "cudaMalloc(d_sorted_nodes)"))
        return false;
    if (!cuda_check(cudaMalloc(&d_keep_flags, n * sizeof(uint8_t)),
                    "cudaMalloc(d_keep_flags)"))
        return false;

    if (!cuda_check(cudaMemcpy(d_from, from_ids_.data(),
                               n * sizeof(uint64_t), cudaMemcpyHostToDevice),
                    "cudaMemcpy(from H2D)"))
        return false;
    if (!cuda_check(cudaMemcpy(d_to, to_ids_.data(),
                               n * sizeof(uint64_t), cudaMemcpyHostToDevice),
                    "cudaMemcpy(to H2D)"))
        return false;
    if (!cuda_check(cudaMemcpy(d_sorted_nodes, nodes.data(),
                               nodes.size() * sizeof(uint64_t),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(sorted_nodes H2D)"))
        return false;

    if (!mdb::gpu::edge_keep_membership_gpu(
            d_from, d_to, d_sorted_nodes,
            static_cast<uint64_t>(nodes.size()),
            d_keep_flags,
            static_cast<uint64_t>(n))) {
        std::fprintf(stderr,
                     "EdgeKeepBitmapGpuBatcher: kernel launch failed; "
                     "falling back to CPU path for this flush\n");
        cleanup();
        return false;
    }

    if (!cuda_check(cudaMemcpy(keep_flags_.data(), d_keep_flags,
                               n * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(keep_flags D2H)"))
        return false;

    cleanup();
    return true;
}
#endif  // MDB_GPU_ENABLED

}  // namespace GQL
