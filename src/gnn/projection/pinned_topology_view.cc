#include "gnn/projection/pinned_topology_view.h"

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
#else
void PinnedTopologyView::register_dir_(const HostCsrArrays&,
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
