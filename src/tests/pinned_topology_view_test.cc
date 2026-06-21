// pinned_topology_view_test.cc
//
// Unit tests for PinnedTopologyView (Phase 2 of the dynamic GPU-UVA sampling
// path). The view registers host-resident uint32 CSR arrays as device-visible
// (cudaHostRegister) so a future k-hop kernel can walk them over PCIe.
//
// These tests are CI-friendly: they pass with OR without a GPU. When no capable
// GPU is present at runtime, build_and_register is a documented no-op
// (is_registered() == false, fwd()/rev() == nullptr) and the device==host
// memcpy checks are skipped. When a GPU IS present they additionally verify the
// device pointers read back the exact host bytes.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/pinned_topology_view.h"

#ifdef GNN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace mdb::gnn {
namespace {

// A small synthetic CSR for one direction. Owns the backing vectors so the
// registered pages stay valid for the test's lifetime.
//
// The two backing vectors `reserve` a large capacity up front so glibc serves
// each from its own mmap'd, page-aligned region (allocations >= 128 KiB are
// mmap'd). Two tiny adjacent heap vectors would otherwise share a 4 KiB page,
// and the second cudaHostRegister (which registers whole pages) would fail with
// cudaErrorHostMemoryAlreadyRegistered. The real substrate (an mmap'd sidecar)
// is already page-aligned, so this guard only matters for the synthetic test.
struct SyntheticCsr {
    std::vector<uint64_t> row_ptr;
    std::vector<uint32_t> col_idx;
    uint64_t              tag = 0;

    SyntheticCsr() {
        row_ptr.reserve(64 * 1024);  // 512 KiB -> own mmap region
        col_idx.reserve(64 * 1024);  // 256 KiB -> own mmap region
    }

    HostCsrArrays arrays() const {
        HostCsrArrays a;
        a.row_ptr      = row_ptr.data();
        a.col_idx      = col_idx.data();
        a.n_rows       = row_ptr.empty() ? 0 : row_ptr.size() - 1;
        a.n_edges      = col_idx.size();
        a.dst_type_tag = tag;
        return a;
    }
};

// 3 nodes, degrees {2,1,3} => 6 edges. row_ptr = [0,2,3,6].
SyntheticCsr make_fwd() {
    SyntheticCsr c;
    c.row_ptr = {0, 2, 3, 6};
    c.col_idx = {1, 2, 0, 0, 1, 2};
    c.tag     = static_cast<uint64_t>(0xD4) << 56;  // a plausible ObjectId tag
    return c;
}

// A distinct reverse direction so BOTH-direction tests can tell them apart.
SyntheticCsr make_rev() {
    SyntheticCsr c;
    c.row_ptr = {0, 1, 3, 4};
    c.col_idx = {2, 0, 1, 0};
    c.tag     = static_cast<uint64_t>(0xD4) << 56;
    return c;
}

#ifdef GNN_CUDA_ENABLED
bool gpu_present() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

// Copy `n` uint32 from a device pointer back to host and compare to expected.
void expect_device_u32_equals(const uint32_t* d_ptr,
                              const std::vector<uint32_t>& expected) {
    std::vector<uint32_t> host(expected.size(), 0xFFFFFFFFu);
    ASSERT_EQ(cudaMemcpy(host.data(), d_ptr, expected.size() * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(host, expected);
}

void expect_device_u64_equals(const uint64_t* d_ptr,
                              const std::vector<uint64_t>& expected) {
    std::vector<uint64_t> host(expected.size(), 0);
    ASSERT_EQ(cudaMemcpy(host.data(), d_ptr, expected.size() * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(host, expected);
}
#else
bool gpu_present() { return false; }
#endif

// ---------------------------------------------------------------------------

TEST(PinnedTopologyView, ForwardOnly_RegistersOrNoOps) {
    SyntheticCsr fwd = make_fwd();
    PinnedTopologyView view;
    view.build_and_register(fwd.arrays(), HostCsrArrays{});

    if (!view.is_registered()) {
        // No GPU at runtime: documented no-op.
        EXPECT_EQ(view.fwd(), nullptr);
        EXPECT_EQ(view.rev(), nullptr);
        return;
    }
    ASSERT_NE(view.fwd(), nullptr);
    EXPECT_EQ(view.rev(), nullptr);  // reverse not supplied
    EXPECT_EQ(view.fwd()->n_rows, 3u);
    EXPECT_EQ(view.fwd()->n_edges, 6u);
    EXPECT_EQ(view.fwd()->dst_type_tag, static_cast<uint64_t>(0xD4) << 56);
    ASSERT_NE(view.fwd()->d_row_ptr, nullptr);
    ASSERT_NE(view.fwd()->d_col_idx, nullptr);
}

TEST(PinnedTopologyView, DeviceSpanEqualsHost) {
    if (!gpu_present()) {
        GTEST_SKIP() << "no GPU at runtime; device==host check is GPU-only";
    }
    SyntheticCsr fwd = make_fwd();
    PinnedTopologyView view;
    view.build_and_register(fwd.arrays(), HostCsrArrays{});
    ASSERT_TRUE(view.is_registered());
    ASSERT_NE(view.fwd(), nullptr);
#ifdef GNN_CUDA_ENABLED
    expect_device_u64_equals(view.fwd()->d_row_ptr, fwd.row_ptr);
    expect_device_u32_equals(view.fwd()->d_col_idx, fwd.col_idx);
#endif
}

TEST(PinnedTopologyView, Undirected_BothDirections) {
    SyntheticCsr fwd = make_fwd();
    SyntheticCsr rev = make_rev();
    PinnedTopologyView view;
    view.build_and_register(fwd.arrays(), rev.arrays());

    if (!view.is_registered()) {
        EXPECT_EQ(view.fwd(), nullptr);
        EXPECT_EQ(view.rev(), nullptr);
        return;
    }
    ASSERT_NE(view.fwd(), nullptr);
    ASSERT_NE(view.rev(), nullptr);
    EXPECT_EQ(view.fwd()->n_edges, 6u);
    EXPECT_EQ(view.rev()->n_edges, 4u);
    EXPECT_EQ(view.rev()->n_rows, 3u);
#ifdef GNN_CUDA_ENABLED
    if (gpu_present()) {
        expect_device_u32_equals(view.fwd()->d_col_idx, fwd.col_idx);
        expect_device_u32_equals(view.rev()->d_col_idx, rev.col_idx);
        expect_device_u64_equals(view.rev()->d_row_ptr, rev.row_ptr);
    }
#endif
}

TEST(PinnedTopologyView, Release_IsIdempotent) {
    SyntheticCsr fwd = make_fwd();
    PinnedTopologyView view;
    view.build_and_register(fwd.arrays(), HostCsrArrays{});
    view.release();
    EXPECT_FALSE(view.is_registered());
    EXPECT_EQ(view.fwd(), nullptr);
    EXPECT_EQ(view.rev(), nullptr);
    // Second release is a clean no-op.
    view.release();
    EXPECT_FALSE(view.is_registered());
}

TEST(PinnedTopologyView, RegisterAfterRelease_Roundtrips) {
    SyntheticCsr fwd = make_fwd();
    PinnedTopologyView view;
    view.build_and_register(fwd.arrays(), HostCsrArrays{});
    const bool first = view.is_registered();
    view.release();
    // Re-registering after release must succeed (not throw).
    view.build_and_register(fwd.arrays(), HostCsrArrays{});
    EXPECT_EQ(view.is_registered(), first);
}

TEST(PinnedTopologyView, DoubleRegister_Throws) {
    if (!gpu_present()) {
        GTEST_SKIP() << "no GPU: build_and_register no-ops so it never registers";
    }
    SyntheticCsr fwd = make_fwd();
    PinnedTopologyView view;
    view.build_and_register(fwd.arrays(), HostCsrArrays{});
    ASSERT_TRUE(view.is_registered());
    EXPECT_THROW(view.build_and_register(fwd.arrays(), HostCsrArrays{}),
                 std::logic_error);
}

TEST(PinnedTopologyView, EmptyArrays_NoOp) {
    PinnedTopologyView view;
    view.build_and_register(HostCsrArrays{}, HostCsrArrays{});
    // Null host arrays register nothing even when a GPU is present.
    EXPECT_FALSE(view.is_registered());
    EXPECT_EQ(view.fwd(), nullptr);
    EXPECT_EQ(view.rev(), nullptr);
}

}  // namespace
}  // namespace mdb::gnn
