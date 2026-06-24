#include "gnn/projection/pinned_topology_view.h"

#include <cstring>
#include <stdexcept>

#ifdef GNN_CUDA_ENABLED
#include <cuda_runtime.h>

#include "gnn/core/cuda_context.h"  // CUDA_CHECK / CudaException
#endif

namespace mdb::gnn {

#ifdef GNN_CUDA_ENABLED
namespace {

// True iff at least one CUDA device is visible at runtime. A build compiled with
// CUDA can still run on a host without a GPU (or with the driver absent); in that
// case registration must be a clean no-op rather than an error.
bool gpu_present_() noexcept {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error so later CUDA calls are sane
        return false;
    }
    return count > 0;
}

}  // namespace
#endif

PinnedTopologyView::~PinnedTopologyView() {
    release();
}

void PinnedTopologyView::build_and_register(const HostCsrArrays& fwd,
                                            const HostCsrArrays& rev) {
    if (registered_) {
        throw std::logic_error(
            "PinnedTopologyView::build_and_register called while already "
            "registered; call release() first");
    }
#ifdef GNN_CUDA_ENABLED
    if (!gpu_present_()) {
        return;  // no-op: no capable GPU at runtime
    }
    try {
        register_dir_(fwd, fwd_, fwd_active_, fwd_reg_);
        register_dir_(rev, rev_, rev_active_, rev_reg_);
    } catch (...) {
        release();  // undo any partial registration before propagating
        throw;
    }
    registered_ = fwd_active_ || rev_active_;
#else
    (void)fwd;
    (void)rev;  // no-op: compiled without CUDA
#endif
}

void PinnedTopologyView::build_and_register_tiled(const HostCsrArrays& fwd,
                                                  const HostCsrArrays& rev,
                                                  std::size_t window_cap_edges) {
    if (registered_) {
        throw std::logic_error(
            "PinnedTopologyView::build_and_register_tiled called while already "
            "registered; call release() first");
    }
#ifdef GNN_CUDA_ENABLED
    if (!gpu_present_()) {
        return;  // no-op: no capable GPU at runtime
    }
    try {
        register_dir_tiled_(fwd, fwd_, fwd_active_, fwd_reg_, window_cap_edges);
        register_dir_tiled_(rev, rev_, rev_active_, rev_reg_, window_cap_edges);
    } catch (...) {
        release();  // undo any partial registration before propagating
        throw;
    }
    registered_ = fwd_active_ || rev_active_;
#else
    (void)fwd;
    (void)rev;
    (void)window_cap_edges;  // no-op: compiled without CUDA
#endif
}

void PinnedTopologyView::build_and_register_resident(const HostCsrArrays& fwd,
                                                     const HostCsrArrays& rev) {
    if (registered_) {
        throw std::logic_error(
            "PinnedTopologyView::build_and_register_resident called while already "
            "registered; call release() first");
    }
#ifdef GNN_CUDA_ENABLED
    if (!gpu_present_()) {
        return;  // no-op: no capable GPU at runtime
    }
    try {
        register_dir_resident_(fwd, fwd_, fwd_active_, fwd_reg_);
        register_dir_resident_(rev, rev_, rev_active_, rev_reg_);
    } catch (...) {
        release();  // free any partial device buffers before propagating
        throw;
    }
    registered_ = fwd_active_ || rev_active_;
#else
    (void)fwd;
    (void)rev;  // no-op: compiled without CUDA
#endif
}

const uint32_t* PinnedTopologyView::map_col_window(const PinnedDirView& dir,
                                                   std::uint64_t edge_lo,
                                                   std::uint64_t edge_hi) const {
    if (!dir.tiled) {
        return dir.d_col_idx;  // no-tiled: COL_IDX entero ya device-visible
    }
    const std::uint64_t span = edge_hi - edge_lo;
    if (span > dir.window_cap_edges || dir.h_col_window == nullptr
        || dir.h_col_src == nullptr) {
        throw std::logic_error(
            "PinnedTopologyView::map_col_window: window span exceeds buffer "
            "capacity or tiled buffers are unset");
    }
    std::memcpy(dir.h_col_window, dir.h_col_src + edge_lo,
                static_cast<std::size_t>(span) * sizeof(uint32_t));
    return dir.d_col_window;
}

#ifdef GNN_CUDA_ENABLED
void PinnedTopologyView::register_dir_(const HostCsrArrays& src,
                                       PinnedDirView&       out,
                                       bool&                active_flag,
                                       HostRegistration&    reg) {
    active_flag = false;
    if (src.row_ptr == nullptr || src.col_idx == nullptr) {
        return;  // direction not present (e.g. NATURAL passes no reverse)
    }

    auto* host_row = const_cast<uint64_t*>(src.row_ptr);
    auto* host_col = const_cast<uint32_t*>(src.col_idx);

    // Register each host region as device-mapped, stashing the host pointer into
    // `reg` immediately after each success so a throw on the second call still
    // lets release() unregister the first.
    CUDA_CHECK(cudaHostRegister(host_row, (src.n_rows + 1) * sizeof(uint64_t),
                                cudaHostRegisterMapped));
    reg.row_ptr = host_row;
    CUDA_CHECK(cudaHostRegister(host_col, src.n_edges * sizeof(uint32_t),
                                cudaHostRegisterMapped));
    reg.col_idx = host_col;

    void* d_row = nullptr;
    void* d_col = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_row, host_row, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_col, host_col, 0));

    out.d_row_ptr    = static_cast<const uint64_t*>(d_row);
    out.d_col_idx    = static_cast<const uint32_t*>(d_col);
    out.dst_type_tag = src.dst_type_tag;
    out.n_rows       = src.n_rows;
    out.n_edges      = src.n_edges;
    active_flag      = true;
}

void PinnedTopologyView::register_dir_tiled_(const HostCsrArrays& src,
                                             PinnedDirView&       out,
                                             bool&                active_flag,
                                             HostRegistration&    reg,
                                             std::size_t          window_cap_edges) {
    active_flag = false;
    if (src.row_ptr == nullptr || src.col_idx == nullptr) {
        return;  // direction not present
    }

    auto* host_row = const_cast<uint64_t*>(src.row_ptr);

    // Pin ROW_PTR whole ((N+1)*8 ~= 0.9 GB on papers100M). COL_IDX is NOT
    // registered: it is streamed window-by-window through the reusable pinned
    // staging buffer below, so its ~12.9 GB never coexist as a pinned host copy.
    CUDA_CHECK(cudaHostRegister(host_row, (src.n_rows + 1) * sizeof(uint64_t),
                                cudaHostRegisterMapped));
    reg.row_ptr = host_row;

    void* d_row = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_row, host_row, 0));

    // One reusable pinned, device-mapped staging buffer sized to the window cap.
    const std::size_t cap = window_cap_edges > 0 ? window_cap_edges : 1;
    void* h_win = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_win, cap * sizeof(uint32_t), cudaHostAllocMapped));
    reg.col_window = h_win;  // freed by release() via cudaFreeHost
    void* d_win = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_win, h_win, 0));

    out.d_row_ptr        = static_cast<const uint64_t*>(d_row);
    out.d_col_idx        = nullptr;  // windowed: fed per-launch via map_col_window
    out.dst_type_tag     = src.dst_type_tag;
    out.n_rows           = src.n_rows;
    out.n_edges          = src.n_edges;  // FULL edge count (window math)
    out.h_row_ptr        = src.row_ptr;
    out.h_col_src        = src.col_idx;  // mmap/heap base (NEVER registered)
    out.h_col_window     = static_cast<uint32_t*>(h_win);
    out.d_col_window     = static_cast<uint32_t*>(d_win);
    out.window_cap_edges = cap;
    out.tiled            = true;
    active_flag          = true;
}

void PinnedTopologyView::register_dir_resident_(const HostCsrArrays& src,
                                                PinnedDirView&       out,
                                                bool&                active_flag,
                                                HostRegistration&    reg) {
    active_flag = false;
    if (src.row_ptr == nullptr || src.col_idx == nullptr) {
        return;  // direction not present
    }

    const std::size_t row_bytes = (src.n_rows + 1) * sizeof(uint64_t);
    const std::size_t col_bytes = src.n_edges * sizeof(uint32_t);

    // Allocate the CSR in DEVICE memory and stream it up from the (mmap/heap)
    // source ONCE, so the kernel reads adjacency from HBM instead of host pages
    // over PCIe. Stash each device buffer in `reg` right after cudaMalloc so a
    // throw on a later call still lets release() cudaFree the partial allocation.
    void* d_row = nullptr;
    void* d_col = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row, row_bytes));
    reg.d_row_owned = d_row;
    CUDA_CHECK(cudaMalloc(&d_col, col_bytes));
    reg.d_col_owned = d_col;

    // ROW_PTR whole; COL_IDX in bounded chunks so the driver's bounce buffer for
    // the pageable (mmap) source stays small. No madvise on the source: it may be
    // a heap buffer (MADV_DONTNEED zero-fills anonymous pages), and the faulted
    // mmap pages are reclaimable + this upload is one-time.
    CUDA_CHECK(cudaMemcpy(d_row, src.row_ptr, row_bytes, cudaMemcpyHostToDevice));
    {
        auto* cdst = reinterpret_cast<uint8_t*>(d_col);
        const auto* csrc = reinterpret_cast<const uint8_t*>(src.col_idx);
        const std::size_t CHUNK = std::size_t(256) << 20;  // 256 MB
        for (std::size_t off = 0; off < col_bytes; off += CHUNK) {
            const std::size_t len = (col_bytes - off < CHUNK) ? (col_bytes - off)
                                                              : CHUNK;
            CUDA_CHECK(cudaMemcpy(cdst + off, csrc + off, len,
                                  cudaMemcpyHostToDevice));
        }
    }

    out.d_row_ptr    = static_cast<const uint64_t*>(d_row);
    out.d_col_idx    = static_cast<const uint32_t*>(d_col);
    out.dst_type_tag = src.dst_type_tag;
    out.n_rows       = src.n_rows;
    out.n_edges      = src.n_edges;
    out.resident     = true;
    out.tiled        = false;
    active_flag      = true;
}
#else
void PinnedTopologyView::register_dir_(const HostCsrArrays&,
                                       PinnedDirView&,
                                       bool&,
                                       HostRegistration&) {}
void PinnedTopologyView::register_dir_tiled_(const HostCsrArrays&,
                                             PinnedDirView&,
                                             bool&,
                                             HostRegistration&,
                                             std::size_t) {}
void PinnedTopologyView::register_dir_resident_(const HostCsrArrays&,
                                                PinnedDirView&,
                                                bool&,
                                                HostRegistration&) {}
#endif

void PinnedTopologyView::release() noexcept {
#ifdef GNN_CUDA_ENABLED
    auto unregister = [](void* host_ptr) noexcept {
        if (host_ptr != nullptr) {
            cudaError_t e = cudaHostUnregister(host_ptr);
            (void)e;
            cudaGetLastError();  // swallow — release() is noexcept / best-effort
        }
    };
    unregister(fwd_reg_.row_ptr);
    unregister(fwd_reg_.col_idx);
    unregister(rev_reg_.row_ptr);
    unregister(rev_reg_.col_idx);
    auto free_window = [](void* host_ptr) noexcept {
        if (host_ptr != nullptr) {
            cudaError_t e = cudaFreeHost(host_ptr);
            (void)e;
            cudaGetLastError();  // swallow — release() is noexcept / best-effort
        }
    };
    free_window(fwd_reg_.col_window);
    free_window(rev_reg_.col_window);
    auto free_device = [](void* dev_ptr) noexcept {
        if (dev_ptr != nullptr) {
            cudaError_t e = cudaFree(dev_ptr);
            (void)e;
            cudaGetLastError();  // swallow — release() is noexcept / best-effort
        }
    };
    free_device(fwd_reg_.d_row_owned);
    free_device(fwd_reg_.d_col_owned);
    free_device(rev_reg_.d_row_owned);
    free_device(rev_reg_.d_col_owned);
#endif
    fwd_reg_    = {};
    rev_reg_    = {};
    fwd_        = {};
    rev_        = {};
    fwd_active_ = false;
    rev_active_ = false;
    registered_ = false;
}

}  // namespace mdb::gnn
