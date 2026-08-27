#pragma once

// sampling_backend_plan.h
//
// Decides, at offline-sampling construction time, whether the neighbor-candidate
// fetch for k-hop sampling should run on the GPU (walking the uint32 adjacency
// CSR pinned in host RAM via cudaHostRegister/UVA, or copied to VRAM) or on the
// proven out-of-core CPU path (the per-node sampler over the hierarchical
// topology store, spilling to the mmap sidecar / B+Tree for the cold tail).
//
// `plan_sampling_backend` is a PURE function: it makes no CUDA calls and caches
// no pointers; it only reasons about sizes (those of the compact in-RAM CSR)
// and the hardware resources reported by `mdb::gpu::detect_resources()`. It is
// the sibling of `plan_sort()` (src/gpu/resource_planner.h), which already makes
// the GPU/CPU decision for projection sorting from the same `SystemResources`.
// This makes it unit-testable in CI without a GPU, by feeding it a synthetic
// `SystemResources`.

#include <cstddef>
#include <cstdint>
#include <string>

#include "gnn/projection/edge_orientation.h"
#include "gpu/gpu_device.h"

namespace mdb::gnn {

// Dimensions of one direction of the global topology CSR that the GPU pins (the
// narrow uint32 `topology_*.csr` sidecar). The decision MUST be sized against
// THIS substrate — the one `enable_pinned_gpu_view` actually registers — and
// not against the warm in-RAM tier. The SUSTAINED RAM cost depends on the
// consumption mode:
//  - Full pin (`tiled_mmap=false`): `cudaHostRegister` over the whole array
//    makes `(n_rows+1)*8 + n_edges*4` bytes non-swappable for the entire run —
//    the RAM gate must count the full global byte size.
//  - Baked slice consumed tiled from mmap (`tiled_mmap=true`, the lean
//    symmetric path): only ROW_PTR (`(n_rows+1)*8`) and a bounded staging
//    window are copied+pinned; COL_IDX is served from the mmap's page cache,
//    which is reclaimable and already counted in MemAvailable. Charging the
//    full CSR to the gate in this mode overstates the sustained cost by an
//    order of magnitude on graphs whose COL_IDX dominates, and demotes later
//    runs in the same server session to CPU: the first run's page cache lowers
//    MemAvailable even though those pages are reclaimable.
// `present` is false when that direction has no pinnable narrow sidecar.
struct DirCsrDims {
    std::uint64_t n_rows     = 0;      // global N (nodes)
    std::uint64_t n_edges    = 0;      // edges of that direction (whole graph)
    bool          present    = false;
    bool          tiled_mmap = false;  // true => locked cost = ROW_PTR + window
};

// Sampling backend chosen by the hardware-driven decision.
enum class SamplingBackend {
    CPU_OUT_OF_CORE,  // per-node sampler over the hierarchical store (proven path)
    GPU_UVA,          // GPU walks the RAM-pinned CSR over PCIe (zero-copy)
    GPU_VRAM_COPY,    // CSR fits in VRAM: copy to device, faster gathers
};

// User choice (override of the automatic decision).
enum class SamplingBackendChoice {
    AUTO,        // decide from hardware
    FORCE_CPU,   // force the CPU path (bit-reproducible reference)
    FORCE_GPU,   // require GPU (hard error if no capable GPU)
};

// Which graph directions the GPU serves. For UNDIRECTED, one direction may fit
// in RAM while the other falls back to the host; the neighbor union stays
// complete.
enum class GpuDirections {
    NONE,
    FORWARD_ONLY,
    REVERSE_ONLY,
    BOTH,
};

// Decision parameters (first approximations copied from the sort planner's
// philosophy; tune with A/B measurements before freezing defaults).
struct SamplingBackendConfig {
    // The CSR + the dense global->row map must fit in this fraction of
    // available RAM. 0.60 (stricter than the sort planner's 0.70) because
    // sampling also keeps the feature store and the label/split stores
    // resident, and (under UVA) the pinned pages live for the whole run.
    double   ram_headroom_factor    = 0.60;
    // CUDA compute-capability floor (Volta+, the floor of mdb_gnn_core).
    int      min_compute_capability = 70;
    // GPU is the DEFAULT sampling backend whenever a capable GPU exists and the
    // CSR fits in VRAM (any graph size), with automatic fallback to CPU. This
    // floor used to be 2'000'000 (the CPU path is sub-second on small graphs
    // and the PCIe pin/copy dominated); lowered to 1 so that small graphs take
    // the device-resident GPU path by default. The compute-capability and VRAM
    // gates (+ the CPU fallback) protect correctness.
    std::uint64_t min_edges_for_gpu = 1;
    // For UNDIRECTED, allow accelerating on GPU only the direction that fits.
    bool     allow_single_direction = true;
    // Absolute VRAM reserve (CUDA context + per-batch scratch buffers) that
    // must remain free IN ADDITION to the CSR to pick the device-resident path.
    // The CSR + this reserve must fit in the raw free VRAM (without the blanket
    // derate).
    std::size_t vram_abs_headroom_bytes = 768ull * 1024 * 1024;  // ~0.75 GiB
    // Bytes of the pinned staging window that the tiled-mmap path keeps
    // resident in addition to ROW_PTR. Must cover the store's default window
    // (64 Mi edges x 4 B = 256 MiB; the actual window only grows beyond that if
    // MDB_GNN_TILE_WINDOW_EDGES raises it or a maximum degree exceeds it).
    std::size_t tiled_stage_bytes = 256ull * 1024 * 1024;
};

// Decision result. `reason` is human-readable and logged once. The sizes are
// exposed for the procedure's yields/telemetry.
struct SamplingBackendPlan {
    SamplingBackend backend              = SamplingBackend::CPU_OUT_OF_CORE;
    GpuDirections   directions           = GpuDirections::NONE;
    // Set by the ENGINE (not the pure planner) when the symmetric pre-merged
    // undirected slice is served: the plan then carries directions==FORWARD_ONLY
    // and the pinned view is the single undirected CSR (never BOTH — that would
    // double-count an already-merged list). plan_sampling_backend leaves false.
    bool            use_symmetric        = false;
    std::string     reason;
    std::size_t     fwd_csr_bytes        = 0;  // (n_rows+1)*8 + n_edges*4, fwd dir
    std::size_t     rev_csr_bytes        = 0;  // (n_rows+1)*8 + n_edges*4, rev dir
    std::size_t     node_map_bytes       = 0;  // 0 with the global sidecar (dense
                                               // row-indexed, no map needed)
    std::size_t     estimated_vram_bytes = 0;  // device-resident bytes if GPU
};

// Decides the backend from the system resources and the CSR sizes.
//
// PURE: no CUDA calls, no cached pointers. Sizing only.
//
// @param res          system resources (from detect_resources()).
// @param orientation  sampling orientation (NATURAL/REVERSE/UNDIRECTED).
// @param fwd_dims     dimensions of the natural-direction global sidecar.
// @param rev_dims     dimensions of the reverse-direction global sidecar.
// @param choice       user override (AUTO/FORCE_CPU/FORCE_GPU).
// @param cfg          decision thresholds.
SamplingBackendPlan plan_sampling_backend(
    const mdb::gpu::SystemResources& res,
    EdgeOrientation                  orientation,
    DirCsrDims                       fwd_dims,
    DirCsrDims                       rev_dims,
    SamplingBackendChoice            choice = SamplingBackendChoice::AUTO,
    const SamplingBackendConfig&     cfg    = {});

// Text helpers for logs/yields.
const char* to_string(SamplingBackend backend) noexcept;
const char* to_string(GpuDirections directions) noexcept;

} // namespace mdb::gnn
