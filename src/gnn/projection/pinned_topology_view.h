#pragma once

// pinned_topology_view.h
//
// Registers the uint32 CSR adjacency arrays (already resident in host RAM) as
// GPU-visible via cudaHostRegister/UVA, WITHOUT duplicating them: the GPU walks
// them over PCIe (zero-copy), like an out-of-core sampler that pins its graph
// in host RAM and reads it from the device. This is the substrate the k-hop
// sampling kernel consumes; this class only registers/unregisters the host
// pages and exposes the device-visible pointers.
//
// Possible substrates for the arrays (the caller decides which one to pass):
//   - Global topology sidecar (ROW_PTR[N+1] uint64 + COL_IDX[M] uint32):
//     covers ALL nodes indexed by dense row, correctness-complete.
//   - Tier-2 compact CSR (L2CompactCsr): only the warm fraction.
// In both cases the descriptor is the same (host pointers + sizes + tag), so
// this class is substrate-agnostic.
//
// Without a GPU at runtime (or in a build compiled without CUDA) this is a
// complete no-op: `build_and_register` registers nothing, `is_registered()`
// stays false and the device pointers stay null — the CPU sampling path is
// entirely unchanged and its output is byte-identical.

#include <cstddef>
#include <cstdint>

namespace mdb::gnn {

/**
 * @brief Device view of one CSR direction (forward or reverse).
 *
 * The pointers are device-visible over the SAME host pages (not copies; the
 * zero-copy mapping makes the GPU walk host RAM over PCIe). `dst_type_tag` is
 * the pre-shifted ObjectId type tag (`tag << 56`) that the kernel ORs back
 * when materializing an output ObjectId, because `d_col_idx` stores
 * tag-stripped uint32 ordinals.
 */
struct PinnedDirView {
    const uint64_t* d_row_ptr    = nullptr;  // device ptr, length n_rows + 1
    const uint32_t* d_col_idx    = nullptr;  // device ptr, length n_edges
    uint64_t        dst_type_tag = 0;        // tag << 56
    std::size_t     n_rows       = 0;        // N (global sidecar) or N_L2 (compact)
    std::size_t     n_edges      = 0;

    // --- Tiled mode (lean symmetric path) -----------------------------------
    // In tiled mode COL_IDX is NOT registered whole: ROW_PTR is pinned in full
    // and COL_IDX is streamed in node-aligned windows through a single reusable
    // pinned buffer. `tiled==false` => the fields above behave EXACTLY like the
    // non-tiled path (d_col_idx is the whole device-visible COL_IDX) and the
    // fields below stay unused.
    const uint64_t* h_row_ptr      = nullptr;  // host ROW_PTR (per-window degree scan)
    const uint32_t* h_col_src      = nullptr;  // host COL_IDX source (mmap/heap base); NEVER registered
    uint32_t*       h_col_window   = nullptr;  // pinned mapped staging buffer (host ptr)
    uint32_t*       d_col_window   = nullptr;  // its device pointer
    std::size_t     window_cap_edges = 0;      // buffer capacity (in edges)
    bool            tiled          = false;

    // --- Resident mode (whole CSR in VRAM) ----------------------------------
    // When the graph fits in VRAM, ROW_PTR and COL_IDX are copied into DEVICE
    // memory of their own (cudaMalloc) and d_row_ptr/d_col_idx point there: the
    // kernel reads from HBM (~hundreds of GB/s) instead of host pages over PCIe
    // (UVA). resident and tiled are mutually exclusive (resident => tiled=false,
    // d_col_idx non-null).
    bool            resident       = false;
};

/**
 * @brief Host descriptor of one direction: pointers to CSR arrays already in RAM.
 *
 * `row_ptr`/`col_idx` == nullptr => that direction is not registered (e.g.
 * NATURAL passes forward only, or a node has no edges in one direction). The
 * pointers must be stable for the whole lifetime of the view (post-freeze /
 * post-mmap); registering an array that later relocates leaves dangling device
 * pointers.
 */
struct HostCsrArrays {
    const uint64_t* row_ptr      = nullptr;  // host, length n_rows + 1
    const uint32_t* col_idx      = nullptr;  // host, length n_edges
    std::size_t     n_rows       = 0;
    std::size_t     n_edges      = 0;
    uint64_t        dst_type_tag = 0;        // tag << 56
};

/**
 * @brief Registers/unregisters the CSR host pages as device-visible.
 *
 * Neither copyable nor movable: it manages page registrations with the CUDA
 * runtime, where a double-unregister would be an error. The owner keeps it
 * behind a `std::unique_ptr` (moving the pointer transfers ownership without
 * moving the object). Construct only over the FINAL arrays
 * (post-freeze/post-mmap).
 */
class PinnedTopologyView {
public:
    PinnedTopologyView() = default;
    ~PinnedTopologyView();

    PinnedTopologyView(const PinnedTopologyView&)            = delete;
    PinnedTopologyView& operator=(const PinnedTopologyView&) = delete;
    PinnedTopologyView(PinnedTopologyView&&)                 = delete;
    PinnedTopologyView& operator=(PinnedTopologyView&&)      = delete;

    /**
     * @brief Registers forward (when it carries pointers) and reverse (optional).
     *
     * Silent no-op (`is_registered()` stays false, device pointers null) when
     * the build has no CUDA or no capable GPU is present at runtime — the CPU
     * path is unchanged. With a GPU present, registers each region with
     * `cudaHostRegisterMapped` and obtains the device pointers via
     * `cudaHostGetDevicePointer`.
     *
     * @throws std::logic_error if a registration is already active (call
     *         `release()` first; re-registering already-pinned pages is a
     *         caller bug).
     * @throws CudaException if a CUDA call fails; any partial registration is
     *         undone before propagating.
     */
    void build_and_register(const HostCsrArrays& fwd, const HostCsrArrays& rev);

    /**
     * @brief Tiled variant: pins the whole ROW_PTR but NOT COL_IDX; streams
     *        COL_IDX in node-aligned windows through a reusable pinned buffer.
     *
     * Intended for the lean symmetric path on graphs whose COL_IDX is a large
     * fraction of host RAM (e.g. ~13 GB on a ~111 M-node / ~3.3 B-edge
     * symmetric graph): pinning the whole COL_IDX as a pinned host copy is
     * exactly what triggers the OOM. Here ROW_PTR (an order of magnitude
     * smaller) is pinned whole (cudaHostRegisterMapped) and, per active
     * direction, ONE pinned mapped buffer of `window_cap_edges` edges is
     * allocated (cudaHostAllocMapped); COL_IDX is copied window by window with
     * `map_col_window`. Leaves `fwd()->d_col_idx == nullptr`: the caller feeds
     * COL_IDX per launch via `map_col_window`. Silent no-op without GPU/CUDA
     * (same as build_and_register).
     *
     * @param window_cap_edges window buffer capacity in edges (must be >= the
     *        degree of the highest-degree node, so no single-node window can
     *        exceed the buffer).
     * @throws std::logic_error if a registration is already active.
     * @throws CudaException if a CUDA call fails (partial registration undone).
     */
    void build_and_register_tiled(const HostCsrArrays& fwd,
                                  const HostCsrArrays& rev,
                                  std::size_t          window_cap_edges);

    /**
     * @brief Stages col_idx[edge_lo, edge_hi) of `dir` into its pinned buffer
     *        and returns the device pointer to read from.
     *
     * For a non-tiled direction returns `dir.d_col_idx` untouched (edge_lo/
     * edge_hi are ignored): the whole COL_IDX is already device-visible. For a
     * tiled one, copies the window into the pinned buffer (host->pinned-host
     * memcpy, device-visible through the mapping) and returns
     * `dir.d_col_window`. The buffer is SINGLE and reusable: each call
     * overwrites it, so the previous window's kernel must have synchronized
     * before re-mapping (sample_layer_on_device_ does this with
     * cudaDeviceSynchronize).
     *
     * @throws std::logic_error if the window exceeds the buffer capacity or the
     *         tiled pointers are unset.
     */
    const uint32_t* map_col_window(const PinnedDirView& dir,
                                   std::uint64_t        edge_lo,
                                   std::uint64_t        edge_hi) const;

    /**
     * @brief Resident variant: copies the WHOLE CSR into device memory
     *        (cudaMalloc + cudaMemcpy) and lets the kernel read it from HBM.
     *
     * For graphs that fit in VRAM: the fastest path (reads from HBM, not over
     * PCIe via UVA, and stages no windows). A single upload amortized at start.
     * Leaves `resident=true, tiled=false` and d_row_ptr/d_col_idx pointing at
     * the owned device buffers. Silent no-op without GPU/CUDA. The kernel's
     * output is byte-identical to build_and_register's (the RNG does not depend
     * on where col_idx lives).
     *
     * @throws std::logic_error if a registration is already active.
     * @throws CudaException if a CUDA call fails (partial buffers freed).
     */
    void build_and_register_resident(const HostCsrArrays& fwd,
                                     const HostCsrArrays& rev);

    /**
     * @brief Unregisters all regions. Idempotent and noexcept.
     */
    void release() noexcept;

    bool is_registered() const noexcept { return registered_; }

    /// nullptr if that direction was not registered.
    const PinnedDirView* fwd() const noexcept { return fwd_active_ ? &fwd_ : nullptr; }
    const PinnedDirView* rev() const noexcept { return rev_active_ ? &rev_ : nullptr; }

private:
    // Original HOST pointers per direction, retained for `cudaHostUnregister`
    // (which takes the host pointer, not the device one).
    struct HostRegistration {
        void* row_ptr     = nullptr;
        void* col_idx     = nullptr;
        void* col_window  = nullptr;  // cudaHostAlloc buffer (tiled); cudaFreeHost in release()
        void* d_row_owned = nullptr;  // cudaMalloc ROW_PTR (resident); cudaFree in release()
        void* d_col_owned = nullptr;  // cudaMalloc COL_IDX (resident); cudaFree in release()
    };

    // Registers one direction. Fills `reg` incrementally (after each successful
    // `cudaHostRegister`) so `release()` can undo partial registrations if a
    // later call throws.
    void register_dir_(const HostCsrArrays& src,
                       PinnedDirView&       out,
                       bool&                active_flag,
                       HostRegistration&    reg);

    // Tiled variant of register_dir_: pins the whole ROW_PTR, allocates the
    // pinned mapped window buffer, leaves d_col_idx null and sets out's tiled
    // fields.
    void register_dir_tiled_(const HostCsrArrays& src,
                             PinnedDirView&       out,
                             bool&                active_flag,
                             HostRegistration&    reg,
                             std::size_t          window_cap_edges);

    // Resident variant of register_dir_: cudaMallocs ROW_PTR+COL_IDX on the
    // device and cudaMemcpys from the mmap/heap source; sets resident and reg's
    // d_*_owned fields.
    void register_dir_resident_(const HostCsrArrays& src,
                                PinnedDirView&       out,
                                bool&                active_flag,
                                HostRegistration&    reg);

    bool             registered_ = false;
    bool             fwd_active_ = false;
    bool             rev_active_ = false;
    PinnedDirView    fwd_;
    PinnedDirView    rev_;
    HostRegistration fwd_reg_;
    HostRegistration rev_reg_;
};

}  // namespace mdb::gnn
