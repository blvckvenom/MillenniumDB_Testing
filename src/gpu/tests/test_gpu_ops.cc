// test_gpu_ops.cc — Tests for GPU bitset filter and UNDIRECTED expand kernels
//
// All GPU tests require CUDA and a visible device.  When compiled without
// MDB_GPU_ENABLED or run on a machine with no GPU the tests GTEST_SKIP.

#include "gpu/gpu_device.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#ifdef MDB_GPU_ENABLED
#include <cuda_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Forward declarations of the host-callable wrappers (defined in .cu files)
// ---------------------------------------------------------------------------
namespace mdb::gpu {
#ifdef MDB_GPU_ENABLED
bool filter_edges_gpu(
    const uint32_t* d_from, const uint32_t* d_to, const uint32_t* d_edge,
    const uint64_t* d_node_bitset, const uint64_t* d_edge_bitset,
    uint32_t* d_out_from, uint32_t* d_out_to, uint32_t* d_out_edge,
    uint64_t num_edges, uint64_t* num_surviving);

bool expand_undirected_gpu(
    const uint32_t* d_from, const uint32_t* d_to, const uint32_t* d_edge,
    uint32_t* d_out_from, uint32_t* d_out_to, uint32_t* d_out_edge,
    uint64_t num_edges);
#endif
} // namespace mdb::gpu

// ===========================================================================
// Helper: RAII guard for device memory
// ===========================================================================
#ifdef MDB_GPU_ENABLED
struct CudaPtr {
    void* ptr = nullptr;
    CudaPtr() = default;
    explicit CudaPtr(size_t bytes) { cudaMalloc(&ptr, bytes); }
    ~CudaPtr() { if (ptr) cudaFree(ptr); }

    // Non-copyable
    CudaPtr(const CudaPtr&) = delete;
    CudaPtr& operator=(const CudaPtr&) = delete;

    // Movable (for returning from upload())
    CudaPtr(CudaPtr&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    CudaPtr& operator=(CudaPtr&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    template<typename T>
    T* as() { return static_cast<T*>(ptr); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(ptr); }
};

/// Upload host vector to a newly-allocated device buffer.
template<typename T>
CudaPtr upload(const std::vector<T>& h_vec) {
    CudaPtr d(h_vec.size() * sizeof(T));
    cudaMemcpy(d.ptr, h_vec.data(), h_vec.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

/// Download device buffer into a host vector.
template<typename T>
std::vector<T> download(const CudaPtr& d, size_t count) {
    std::vector<T> h(count);
    cudaMemcpy(h.data(), d.ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
#endif

// ===========================================================================
// Helper: set a bit in a uint64_t bitset
// ===========================================================================
static void set_bit(std::vector<uint64_t>& bitset, uint32_t val) {
    size_t word = val / 64;
    if (word >= bitset.size()) {
        bitset.resize(word + 1, 0);
    }
    bitset[word] |= (uint64_t(1) << (val % 64));
}

// ===========================================================================
// Bitset Filter Tests
// ===========================================================================

class GpuFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifndef MDB_GPU_ENABLED
        GTEST_SKIP() << "Compiled without CUDA (MDB_GPU_ENABLED not defined)";
#else
        auto res = mdb::gpu::detect_resources();
        if (!res.has_gpu) {
            GTEST_SKIP() << "No GPU available";
        }
#endif
    }
};

TEST_F(GpuFilterTest, BasicFilter) {
#ifdef MDB_GPU_ENABLED
    // 5 edges; only edges where both endpoints AND edge type are in bitsets pass
    //
    //   edge 0: from=0 to=1 edge=0  -> all in bitset -> pass
    //   edge 1: from=2 to=3 edge=1  -> node 3 NOT in bitset -> fail
    //   edge 2: from=0 to=2 edge=2  -> edge-type 2 NOT in bitset -> fail
    //   edge 3: from=1 to=2 edge=0  -> all in bitset -> pass
    //   edge 4: from=2 to=0 edge=1  -> all in bitset -> pass
    //
    std::vector<uint32_t> from_vals = {0, 2, 0, 1, 2};
    std::vector<uint32_t> to_vals   = {1, 3, 2, 2, 0};
    std::vector<uint32_t> edge_vals = {0, 1, 2, 0, 1};

    // Node bitset: include nodes {0, 1, 2} but NOT 3
    std::vector<uint64_t> node_bitset(1, 0);
    set_bit(node_bitset, 0);
    set_bit(node_bitset, 1);
    set_bit(node_bitset, 2);

    // Edge bitset: include edge types {0, 1} but NOT 2
    std::vector<uint64_t> edge_bitset(1, 0);
    set_bit(edge_bitset, 0);
    set_bit(edge_bitset, 1);

    uint64_t E = from_vals.size();

    // Upload
    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);
    auto d_nbs  = upload(node_bitset);
    auto d_ebs  = upload(edge_bitset);

    // Output buffers (max size = E)
    CudaPtr d_out_from(E * sizeof(uint32_t));
    CudaPtr d_out_to(E * sizeof(uint32_t));
    CudaPtr d_out_edge(E * sizeof(uint32_t));

    uint64_t num_surviving = 0;
    bool ok = mdb::gpu::filter_edges_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_nbs.as<uint64_t>(), d_ebs.as<uint64_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        E, &num_surviving);

    ASSERT_TRUE(ok);
    ASSERT_EQ(num_surviving, 3u);

    auto h_out_from = download<uint32_t>(d_out_from, num_surviving);
    auto h_out_to   = download<uint32_t>(d_out_to,   num_surviving);
    auto h_out_edge = download<uint32_t>(d_out_edge, num_surviving);

    // Surviving edges (order preserved by CUB Flagged):
    //   edge 0: (0, 1, 0)
    //   edge 3: (1, 2, 0)
    //   edge 4: (2, 0, 1)
    EXPECT_EQ(h_out_from[0], 0u); EXPECT_EQ(h_out_to[0], 1u); EXPECT_EQ(h_out_edge[0], 0u);
    EXPECT_EQ(h_out_from[1], 1u); EXPECT_EQ(h_out_to[1], 2u); EXPECT_EQ(h_out_edge[1], 0u);
    EXPECT_EQ(h_out_from[2], 2u); EXPECT_EQ(h_out_to[2], 0u); EXPECT_EQ(h_out_edge[2], 1u);
#endif
}

TEST_F(GpuFilterTest, EmptyInput) {
#ifdef MDB_GPU_ENABLED
    // Zero edges — should succeed with 0 surviving
    std::vector<uint64_t> node_bitset(1, 0);
    std::vector<uint64_t> edge_bitset(1, 0);
    auto d_nbs = upload(node_bitset);
    auto d_ebs = upload(edge_bitset);

    // Allocate minimal output buffers (1 element; won't be written)
    CudaPtr d_out_from(sizeof(uint32_t));
    CudaPtr d_out_to(sizeof(uint32_t));
    CudaPtr d_out_edge(sizeof(uint32_t));

    uint64_t num_surviving = 999;
    bool ok = mdb::gpu::filter_edges_gpu(
        nullptr, nullptr, nullptr,
        d_nbs.as<uint64_t>(), d_ebs.as<uint64_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        0, &num_surviving);

    ASSERT_TRUE(ok);
    EXPECT_EQ(num_surviving, 0u);
#endif
}

TEST_F(GpuFilterTest, AllPass) {
#ifdef MDB_GPU_ENABLED
    // All edges pass the filter
    std::vector<uint32_t> from_vals = {0, 1, 2};
    std::vector<uint32_t> to_vals   = {1, 2, 0};
    std::vector<uint32_t> edge_vals = {0, 1, 0};

    std::vector<uint64_t> node_bitset(1, 0);
    set_bit(node_bitset, 0);
    set_bit(node_bitset, 1);
    set_bit(node_bitset, 2);

    std::vector<uint64_t> edge_bitset(1, 0);
    set_bit(edge_bitset, 0);
    set_bit(edge_bitset, 1);

    uint64_t E = from_vals.size();

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);
    auto d_nbs  = upload(node_bitset);
    auto d_ebs  = upload(edge_bitset);

    CudaPtr d_out_from(E * sizeof(uint32_t));
    CudaPtr d_out_to(E * sizeof(uint32_t));
    CudaPtr d_out_edge(E * sizeof(uint32_t));

    uint64_t num_surviving = 0;
    bool ok = mdb::gpu::filter_edges_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_nbs.as<uint64_t>(), d_ebs.as<uint64_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        E, &num_surviving);

    ASSERT_TRUE(ok);
    EXPECT_EQ(num_surviving, 3u);

    auto h_out_from = download<uint32_t>(d_out_from, num_surviving);
    auto h_out_to   = download<uint32_t>(d_out_to,   num_surviving);
    auto h_out_edge = download<uint32_t>(d_out_edge, num_surviving);

    for (uint64_t i = 0; i < num_surviving; i++) {
        EXPECT_EQ(h_out_from[i], from_vals[i]);
        EXPECT_EQ(h_out_to[i],   to_vals[i]);
        EXPECT_EQ(h_out_edge[i], edge_vals[i]);
    }
#endif
}

TEST_F(GpuFilterTest, AllFail) {
#ifdef MDB_GPU_ENABLED
    // No nodes in bitset → all edges fail
    std::vector<uint32_t> from_vals = {0, 1, 2};
    std::vector<uint32_t> to_vals   = {1, 2, 0};
    std::vector<uint32_t> edge_vals = {0, 1, 0};

    std::vector<uint64_t> node_bitset(1, 0);  // empty — no bits set
    std::vector<uint64_t> edge_bitset(1, 0);
    set_bit(edge_bitset, 0);
    set_bit(edge_bitset, 1);

    uint64_t E = from_vals.size();

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);
    auto d_nbs  = upload(node_bitset);
    auto d_ebs  = upload(edge_bitset);

    CudaPtr d_out_from(E * sizeof(uint32_t));
    CudaPtr d_out_to(E * sizeof(uint32_t));
    CudaPtr d_out_edge(E * sizeof(uint32_t));

    uint64_t num_surviving = 999;
    bool ok = mdb::gpu::filter_edges_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_nbs.as<uint64_t>(), d_ebs.as<uint64_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        E, &num_surviving);

    ASSERT_TRUE(ok);
    EXPECT_EQ(num_surviving, 0u);
#endif
}

TEST_F(GpuFilterTest, LargeBitsetValues) {
#ifdef MDB_GPU_ENABLED
    // Node/edge values that span multiple uint64_t words in the bitset
    // Node IDs: 0, 100, 200  (words 0, 1, 3)
    // Edge type: 130          (word 2)
    std::vector<uint32_t> from_vals = {0,   100};
    std::vector<uint32_t> to_vals   = {100, 200};
    std::vector<uint32_t> edge_vals = {130, 130};

    std::vector<uint64_t> node_bitset(4, 0);
    set_bit(node_bitset, 0);
    set_bit(node_bitset, 100);
    set_bit(node_bitset, 200);

    std::vector<uint64_t> edge_bitset(3, 0);
    set_bit(edge_bitset, 130);

    uint64_t E = from_vals.size();

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);
    auto d_nbs  = upload(node_bitset);
    auto d_ebs  = upload(edge_bitset);

    CudaPtr d_out_from(E * sizeof(uint32_t));
    CudaPtr d_out_to(E * sizeof(uint32_t));
    CudaPtr d_out_edge(E * sizeof(uint32_t));

    uint64_t num_surviving = 0;
    bool ok = mdb::gpu::filter_edges_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_nbs.as<uint64_t>(), d_ebs.as<uint64_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        E, &num_surviving);

    ASSERT_TRUE(ok);
    EXPECT_EQ(num_surviving, 2u);

    auto h_out_from = download<uint32_t>(d_out_from, num_surviving);
    auto h_out_to   = download<uint32_t>(d_out_to,   num_surviving);

    EXPECT_EQ(h_out_from[0], 0u);   EXPECT_EQ(h_out_to[0], 100u);
    EXPECT_EQ(h_out_from[1], 100u); EXPECT_EQ(h_out_to[1], 200u);
#endif
}

TEST_F(GpuFilterTest, SingleEdgePass) {
#ifdef MDB_GPU_ENABLED
    std::vector<uint32_t> from_vals = {5};
    std::vector<uint32_t> to_vals   = {10};
    std::vector<uint32_t> edge_vals = {3};

    std::vector<uint64_t> node_bitset(1, 0);
    set_bit(node_bitset, 5);
    set_bit(node_bitset, 10);

    std::vector<uint64_t> edge_bitset(1, 0);
    set_bit(edge_bitset, 3);

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);
    auto d_nbs  = upload(node_bitset);
    auto d_ebs  = upload(edge_bitset);

    CudaPtr d_out_from(sizeof(uint32_t));
    CudaPtr d_out_to(sizeof(uint32_t));
    CudaPtr d_out_edge(sizeof(uint32_t));

    uint64_t num_surviving = 0;
    bool ok = mdb::gpu::filter_edges_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_nbs.as<uint64_t>(), d_ebs.as<uint64_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        1, &num_surviving);

    ASSERT_TRUE(ok);
    EXPECT_EQ(num_surviving, 1u);

    auto h_out_from = download<uint32_t>(d_out_from, 1);
    EXPECT_EQ(h_out_from[0], 5u);
#endif
}

TEST_F(GpuFilterTest, SingleEdgeFail) {
#ifdef MDB_GPU_ENABLED
    std::vector<uint32_t> from_vals = {5};
    std::vector<uint32_t> to_vals   = {10};
    std::vector<uint32_t> edge_vals = {3};

    std::vector<uint64_t> node_bitset(1, 0);
    set_bit(node_bitset, 5);
    // node 10 NOT in bitset

    std::vector<uint64_t> edge_bitset(1, 0);
    set_bit(edge_bitset, 3);

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);
    auto d_nbs  = upload(node_bitset);
    auto d_ebs  = upload(edge_bitset);

    CudaPtr d_out_from(sizeof(uint32_t));
    CudaPtr d_out_to(sizeof(uint32_t));
    CudaPtr d_out_edge(sizeof(uint32_t));

    uint64_t num_surviving = 999;
    bool ok = mdb::gpu::filter_edges_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_nbs.as<uint64_t>(), d_ebs.as<uint64_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        1, &num_surviving);

    ASSERT_TRUE(ok);
    EXPECT_EQ(num_surviving, 0u);
#endif
}

// ===========================================================================
// UNDIRECTED Expand Tests
// ===========================================================================

class GpuExpandTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifndef MDB_GPU_ENABLED
        GTEST_SKIP() << "Compiled without CUDA (MDB_GPU_ENABLED not defined)";
#else
        auto res = mdb::gpu::detect_resources();
        if (!res.has_gpu) {
            GTEST_SKIP() << "No GPU available";
        }
#endif
    }
};

TEST_F(GpuExpandTest, BasicExpand) {
#ifdef MDB_GPU_ENABLED
    // 3 edges → 6 after expansion
    std::vector<uint32_t> from_vals = {0, 1, 2};
    std::vector<uint32_t> to_vals   = {1, 2, 0};
    std::vector<uint32_t> edge_vals = {10, 20, 30};

    uint64_t E = from_vals.size();

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);

    CudaPtr d_out_from(2 * E * sizeof(uint32_t));
    CudaPtr d_out_to(2 * E * sizeof(uint32_t));
    CudaPtr d_out_edge(2 * E * sizeof(uint32_t));

    bool ok = mdb::gpu::expand_undirected_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        E);

    ASSERT_TRUE(ok);

    auto h_out_from = download<uint32_t>(d_out_from, 2 * E);
    auto h_out_to   = download<uint32_t>(d_out_to,   2 * E);
    auto h_out_edge = download<uint32_t>(d_out_edge, 2 * E);

    // Edge 0 forward: (0, 1, 10)
    EXPECT_EQ(h_out_from[0], 0u);  EXPECT_EQ(h_out_to[0], 1u);  EXPECT_EQ(h_out_edge[0], 10u);
    // Edge 0 reverse: (1, 0, 10)
    EXPECT_EQ(h_out_from[1], 1u);  EXPECT_EQ(h_out_to[1], 0u);  EXPECT_EQ(h_out_edge[1], 10u);
    // Edge 1 forward: (1, 2, 20)
    EXPECT_EQ(h_out_from[2], 1u);  EXPECT_EQ(h_out_to[2], 2u);  EXPECT_EQ(h_out_edge[2], 20u);
    // Edge 1 reverse: (2, 1, 20)
    EXPECT_EQ(h_out_from[3], 2u);  EXPECT_EQ(h_out_to[3], 1u);  EXPECT_EQ(h_out_edge[3], 20u);
    // Edge 2 forward: (2, 0, 30)
    EXPECT_EQ(h_out_from[4], 2u);  EXPECT_EQ(h_out_to[4], 0u);  EXPECT_EQ(h_out_edge[4], 30u);
    // Edge 2 reverse: (0, 2, 30)
    EXPECT_EQ(h_out_from[5], 0u);  EXPECT_EQ(h_out_to[5], 2u);  EXPECT_EQ(h_out_edge[5], 30u);
#endif
}

TEST_F(GpuExpandTest, EmptyInput) {
#ifdef MDB_GPU_ENABLED
    // Zero edges — should succeed, no output
    CudaPtr d_out_from(sizeof(uint32_t));
    CudaPtr d_out_to(sizeof(uint32_t));
    CudaPtr d_out_edge(sizeof(uint32_t));

    bool ok = mdb::gpu::expand_undirected_gpu(
        nullptr, nullptr, nullptr,
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        0);

    EXPECT_TRUE(ok);
#endif
}

TEST_F(GpuExpandTest, SingleEdge) {
#ifdef MDB_GPU_ENABLED
    std::vector<uint32_t> from_vals = {42};
    std::vector<uint32_t> to_vals   = {99};
    std::vector<uint32_t> edge_vals = {7};

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);

    CudaPtr d_out_from(2 * sizeof(uint32_t));
    CudaPtr d_out_to(2 * sizeof(uint32_t));
    CudaPtr d_out_edge(2 * sizeof(uint32_t));

    bool ok = mdb::gpu::expand_undirected_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        1);

    ASSERT_TRUE(ok);

    auto h_out_from = download<uint32_t>(d_out_from, 2);
    auto h_out_to   = download<uint32_t>(d_out_to,   2);
    auto h_out_edge = download<uint32_t>(d_out_edge, 2);

    // Forward: (42, 99, 7)
    EXPECT_EQ(h_out_from[0], 42u); EXPECT_EQ(h_out_to[0], 99u); EXPECT_EQ(h_out_edge[0], 7u);
    // Reverse: (99, 42, 7)
    EXPECT_EQ(h_out_from[1], 99u); EXPECT_EQ(h_out_to[1], 42u); EXPECT_EQ(h_out_edge[1], 7u);
#endif
}

TEST_F(GpuExpandTest, SelfLoop) {
#ifdef MDB_GPU_ENABLED
    // A self-loop (from == to) still produces two entries (both identical)
    std::vector<uint32_t> from_vals = {5};
    std::vector<uint32_t> to_vals   = {5};
    std::vector<uint32_t> edge_vals = {0};

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);

    CudaPtr d_out_from(2 * sizeof(uint32_t));
    CudaPtr d_out_to(2 * sizeof(uint32_t));
    CudaPtr d_out_edge(2 * sizeof(uint32_t));

    bool ok = mdb::gpu::expand_undirected_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        1);

    ASSERT_TRUE(ok);

    auto h_out_from = download<uint32_t>(d_out_from, 2);
    auto h_out_to   = download<uint32_t>(d_out_to,   2);
    auto h_out_edge = download<uint32_t>(d_out_edge, 2);

    // Both forward and reverse are (5, 5, 0) for a self-loop
    EXPECT_EQ(h_out_from[0], 5u); EXPECT_EQ(h_out_to[0], 5u); EXPECT_EQ(h_out_edge[0], 0u);
    EXPECT_EQ(h_out_from[1], 5u); EXPECT_EQ(h_out_to[1], 5u); EXPECT_EQ(h_out_edge[1], 0u);
#endif
}

TEST_F(GpuExpandTest, EdgeValPreserved) {
#ifdef MDB_GPU_ENABLED
    // Verify that the edge_val is the same for forward and reverse entries
    std::vector<uint32_t> from_vals = {0, 10};
    std::vector<uint32_t> to_vals   = {1, 20};
    std::vector<uint32_t> edge_vals = {100, 200};

    uint64_t E = from_vals.size();

    auto d_from = upload(from_vals);
    auto d_to   = upload(to_vals);
    auto d_edge = upload(edge_vals);

    CudaPtr d_out_from(2 * E * sizeof(uint32_t));
    CudaPtr d_out_to(2 * E * sizeof(uint32_t));
    CudaPtr d_out_edge(2 * E * sizeof(uint32_t));

    bool ok = mdb::gpu::expand_undirected_gpu(
        d_from.as<uint32_t>(), d_to.as<uint32_t>(), d_edge.as<uint32_t>(),
        d_out_from.as<uint32_t>(), d_out_to.as<uint32_t>(), d_out_edge.as<uint32_t>(),
        E);

    ASSERT_TRUE(ok);

    auto h_out_edge = download<uint32_t>(d_out_edge, 2 * E);

    // Forward and reverse for edge 0 share the same edge_val
    EXPECT_EQ(h_out_edge[0], 100u);
    EXPECT_EQ(h_out_edge[1], 100u);
    // Forward and reverse for edge 1 share the same edge_val
    EXPECT_EQ(h_out_edge[2], 200u);
    EXPECT_EQ(h_out_edge[3], 200u);
#endif
}
