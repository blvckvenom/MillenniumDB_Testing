// Unit tests para plan_sampling_backend (la decision pura GPU/CPU del muestreo).
//
// La decision se dimensiona sobre el sidecar global de topologia (lo que la GPU
// realmente pinea). Estos tests pasan dimensiones sinteticas (DirCsrDims) +
// recursos de sistema sinteticos para ejercitar la tabla de decision sin GPU real
// ni construir CSRs grandes. Corre en CI sin CUDA: la funcion es pura sizing.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "gnn/projection/edge_orientation.h"
#include "gnn/sampling/sampling_backend_plan.h"
#include "gpu/gpu_device.h"

using namespace mdb::gnn;

namespace {

// Dimensiones de una direccion: n_rows nodos, n_edges aristas.
// csr_bytes pineados = (n_rows+1)*8 + n_edges*4.
DirCsrDims dims(std::uint64_t n_rows, std::uint64_t n_edges) {
    return DirCsrDims{n_rows, n_edges, /*present=*/true};
}

mdb::gpu::SystemResources make_res(bool has_gpu, int cc,
                                   std::size_t free_vram, std::size_t ram) {
    mdb::gpu::SystemResources r;
    r.has_gpu                  = has_gpu;
    r.gpu.compute_capability   = cc;
    r.gpu.free_vram            = free_vram;
    r.gpu.total_vram           = free_vram;
    r.ram_available            = ram;
    return r;
}

// cfg con umbral de aristas configurable.
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
                                      cfg_min_edges(1'000'000));  // umbral 1M > 5000
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
    // needed = 20808 <= headroom (60M) y <= free_vram (100M) => VRAM_COPY.
    auto res  = make_res(true, 80, /*vram=*/100'000'000, /*ram=*/100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::NATURAL, dims(100, 5000),
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::GPU_VRAM_COPY);
    EXPECT_EQ(plan.directions, GpuDirections::FORWARD_ONLY);
    EXPECT_GT(plan.estimated_vram_bytes, 0u);
}

TEST(SamplingBackendPlan, FitsRamButNotVramPicksUva) {
    // vram chica (10 KB) < device_resident (~20.8 KB), ram amplia => UVA.
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
    // fwd chico: csr = 808 + 2000*4 = 8808; rev grande: 808 + 20000*4 = 80808.
    // headroom = 0.60*100000 = 60000 => solo fwd cabe (rev no, both no).
    auto res  = make_res(true, 80, /*vram=*/100'000'000, /*ram=*/100'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::UNDIRECTED,
                                      dims(100, 2000), dims(100, 20000),
                                      SamplingBackendChoice::AUTO, cfg_min_edges(1000));
    EXPECT_NE(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
    EXPECT_EQ(plan.directions, GpuDirections::FORWARD_ONLY);
}

TEST(SamplingBackendPlan, AbsentSidecarFallsToCpu) {
    // Sin sidecar pineable en ninguna direccion (present=false) => sin aristas =>
    // workload too small => CPU. Es el caso de una proyeccion sin sidecar narrow.
    auto res  = make_res(true, 80, 100'000'000, 100'000'000);
    auto plan = plan_sampling_backend(res, EdgeOrientation::UNDIRECTED, DirCsrDims{},
                                      DirCsrDims{}, SamplingBackendChoice::AUTO,
                                      cfg_min_edges(1000));
    EXPECT_EQ(plan.backend, SamplingBackend::CPU_OUT_OF_CORE);
}

TEST(SamplingBackendPlan, ForceCpuSkipsGates) {
    auto res  = make_res(true, 80, 100'000'000, 100'000'000);  // GPU capaz
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
    EXPECT_EQ(plan.reason.rfind("ERROR:", 0), 0u);  // empieza con "ERROR:"
}
