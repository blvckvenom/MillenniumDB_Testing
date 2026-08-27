// Unit tests for plan_sampling_backend (the pure GPU/CPU sampling decision).
//
// The decision is sized against the global topology sidecar (what the GPU
// actually pins). These tests pass synthetic dimensions (DirCsrDims) plus
// synthetic system resources to exercise the decision table without a real
// GPU and without building large CSRs. Runs in CI without CUDA: the function
// is pure sizing arithmetic.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "gnn/projection/edge_orientation.h"
#include "gnn/sampling/sampling_backend_plan.h"
#include "gpu/gpu_device.h"

using namespace mdb::gnn;

namespace {

// Dimensions of one direction: n_rows nodes, n_edges edges.
// Pinned csr_bytes = (n_rows+1)*8 + n_edges*4.
DirCsrDims dims(std::uint64_t n_rows, std::uint64_t n_edges) {
    return DirCsrDims{n_rows, n_edges, /*present=*/true};
}

mdb::gpu::SystemResources make_res(bool has_gpu, int cc,
                                   std::size_t free_vram, std::size_t ram) {
    mdb::gpu::SystemResources r;
    r.has_gpu                  = has_gpu;
    r.gpu.compute_capability   = cc;
    r.gpu.free_vram            = free_vram;
    r.gpu.raw_free_vram        = free_vram;  // tests: raw == derated (no derating)
    r.gpu.total_vram           = free_vram;
    r.ram_available            = ram;
    return r;
}

// Config with a configurable edge-count threshold.
SamplingBackendConfig cfg_min_edges(std::uint64_t min_edges) {
    SamplingBackendConfig c;
    c.min_edges_for_gpu = min_edges;
    return c;
}

}  // namespace

TEST(SamplingBackendPlan, NoGpuFallsToCpu) {
    auto res  = make_res(/*has_gpu=*/false, /*cc=*/0, /*vram=*/0, /*ram=*/100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_NE(plan.reason.find("no capable GPU"), std::string::npos);
}

TEST(SamplingBackendPlan, ComputeCapabilityTooLowFallsToCpu) {
    auto res  = make_res(true, /*cc=*/60, 100'000'000, 100'000'000);  // cc 60 < 70
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_NE(plan.reason.find("no capable GPU"), std::string::npos);
}

TEST(SamplingBackendPlan, WorkloadTooSmallFallsToCpu) {
    auto res  = make_res(true, 80, 100'000'000, 100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1'000'000));  // threshold 1M > 5000
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_NE(plan.reason.find("workload too small"), std::string::npos);
}

TEST(SamplingBackendPlan, CsrExceedsRamHeadroomFallsToCpu) {
    // csr_bytes = 101*8 + 5000*4 = 20808; headroom = 0.60*30000 = 18000 < 20808.
    auto res  = make_res(true, 80, 100'000'000, /*ram=*/30'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_NE(plan.reason.find("RAM headroom"), std::string::npos);
}

TEST(SamplingBackendPlan, FitsRamAndVramPicksVramCopy) {
    // needed = 20808 + abs headroom (~0.75 GiB) <= raw_free_vram (2 GB) => VRAM_COPY.
    auto res  = make_res(true, 80, /*vram=*/2'000'000'000, /*ram=*/100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::GPU_VRAM_COPY);
    EXPECT_EQ(plan.directions, GpuDirections::FORWARD_ONLY);
    EXPECT_GT(plan.estimated_vram_bytes, 0u);
}

TEST(SamplingBackendPlan, FitsRawVramEvenIfDeratedRejects) {
    // The CSR + headroom fit the RAW free VRAM (2 GB) even though the derated
    // free_vram (1 MB) would reject it: the VRAM admission check must key on
    // raw free VRAM, not the derated figure.
    mdb::gpu::SystemResources r;
    r.has_gpu                = true;
    r.gpu.compute_capability = 80;
    r.gpu.free_vram          = 1'000'000;       // derated (would reject)
    r.gpu.raw_free_vram      = 2'000'000'000;   // raw (admits CSR + headroom)
    r.gpu.total_vram         = 2'000'000'000;
    r.ram_available          = 100'000'000;
    auto plan = plan_sampling_backend(r, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::GPU_VRAM_COPY);
}

TEST(SamplingBackendPlan, RawVramTooTightForHeadroomPicksUva) {
    // raw free is above the CSR (20808) but below CSR + abs headroom => UVA.
    mdb::gpu::SystemResources r;
    r.has_gpu                = true;
    r.gpu.compute_capability = 80;
    r.gpu.free_vram          = 30'000;
    r.gpu.raw_free_vram      = 30'000;          // > 20808 but < 20808 + ~0.75 GiB
    r.gpu.total_vram         = 30'000;
    r.ram_available          = 100'000'000;
    auto plan = plan_sampling_backend(r, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::GPU_UVA);
}

TEST(SamplingBackendPlan, FitsRamButNotVramPicksUva) {
    // small vram (10 KB) < device_resident (~20.8 KB), ample ram => UVA.
    auto res  = make_res(true, 80, /*vram=*/10'000, /*ram=*/100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::GPU_UVA);
    EXPECT_EQ(plan.directions, GpuDirections::FORWARD_ONLY);
}

TEST(SamplingBackendPlan, UndirectedBothFit) {
    auto res  = make_res(true, 80, 100'000'000, 100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::UNDIRECTED,
                                      dims(100, 5000), dims(100, 5000),
                                      SamplingBackendChoice::AUTO, cfg_min_edges(1000));
    EXPECT_NE(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_EQ(plan.directions, GpuDirections::BOTH);
}

TEST(SamplingBackendPlan, UndirectedOnlyOneDirectionFits) {
    // small fwd: csr = 808 + 2000*4 = 8808; large rev: 808 + 20000*4 = 80808.
    // headroom = 0.60*100000 = 60000 => only fwd fits (rev does not, both do not).
    auto res  = make_res(true, 80, /*vram=*/100'000'000, /*ram=*/100'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::UNDIRECTED,
                                      dims(100, 2000), dims(100, 20000),
                                      SamplingBackendChoice::AUTO, cfg_min_edges(1000));
    EXPECT_NE(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_EQ(plan.directions, GpuDirections::FORWARD_ONLY);
}

TEST(SamplingBackendPlan, AbsentSidecarFallsToCpu) {
    // No pinnable sidecar in either direction (present=false) => zero edges =>
    // workload too small => CPU. This is a projection built without the
    // narrow-width sidecar.
    auto res  = make_res(true, 80, 100'000'000, 100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::UNDIRECTED, DirCsrDims{},
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
}

TEST(SamplingBackendPlan, ForceCpuSkipsGates) {
    auto res  = make_res(true, 80, 100'000'000, 100'000'000);  // capable GPU
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::FORCE_CPU,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_NE(plan.reason.find("forced CPU"), std::string::npos);
}

TEST(SamplingBackendPlan, ForceGpuWithoutGpuFlagsHardError) {
    auto res  = make_res(/*has_gpu=*/false, 0, 0, 100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::FORCE_GPU,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_EQ(plan.reason.rfind("ERROR:", 0), 0u);  // starts with "ERROR:"
}

// ---------------------------------------------------------------------------
// Regression, 2026-07-03: the baked symmetric slice is consumed TILED from
// the mmap (only ROW_PTR plus a staging window stay page-locked; COL_IDX
// lives in reclaimable page cache). The RAM admission check must charge that
// locked cost, not the full CSR — charged the full CSR, the second sample of
// a single server session (MemAvailable down to ~21 GB after the first)
// fell to CPU_OUT_OF_CORE even though the VRAM was free. The figures below
// are the EXACT ones from the run that exposed it: a ~13.8 GB symmetric CSR
// (111 M nodes, 3.2 B edges) against ~21 GB of MemAvailable.
// ---------------------------------------------------------------------------

TEST(SamplingBackendPlan, TiledMmapSliceNotChargedFullCsr_Papers100mRun2) {
    // headroom = 0.60 * 21'082'181'632 = 12'649'308'979 < csr 13'800'978'504,
    // but locked = row_ptr 888'479'656 + window 268'435'456 = 1'156'915'112.
    auto res = make_res(/*has_gpu=*/true, /*cc=*/120,
                        /*vram=*/16'176'250'880, /*ram=*/21'082'181'632);
    DirCsrDims sym{/*n_rows=*/111'059'956, /*n_edges=*/3'228'124'712,
                   /*present=*/true, /*tiled_mmap=*/true};
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, sym,
                                      DirCsrDims{}, SamplingBackendChoice::AUTO);
    EXPECT_EQ(plan.backend, SamplingBackend::GPU_VRAM_COPY)
        << "reason: " << plan.reason;
    EXPECT_EQ(plan.directions, GpuDirections::FORWARD_ONLY);
}

TEST(SamplingBackendPlan, FullPinStillChargedFullCsr) {
    // Same figures WITHOUT tiled_mmap: a full pin really does retain the whole
    // CSR => the conservative check must keep sending it to CPU.
    auto res = make_res(true, 120, 16'176'250'880, 21'082'181'632);
    DirCsrDims full{111'059'956, 3'228'124'712, /*present=*/true,
                    /*tiled_mmap=*/false};
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, full,
                                      DirCsrDims{}, SamplingBackendChoice::AUTO);
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE)
        << "reason: " << plan.reason;
    EXPECT_NE(plan.reason.find("locked"), std::string::npos);
}

TEST(SamplingBackendPlan, TiledMmapStillCpuWhenRamTrulyTiny) {
    // With tiny RAM not even ROW_PTR + window fit => CPU.
    auto res = make_res(true, 120, 16'176'250'880, /*ram=*/1'000'000'000);
    DirCsrDims sym{111'059'956, 3'228'124'712, true, /*tiled_mmap=*/true};
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, sym,
                                      DirCsrDims{}, SamplingBackendChoice::AUTO);
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE)
        << "reason: " << plan.reason;
}
