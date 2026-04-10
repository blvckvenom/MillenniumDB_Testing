#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "gnn/core/feature_assembler.h"

using namespace mdb::gnn;

// =============================================================================
// Basic Fallback Assembly Tests
// =============================================================================

TEST(FeatureAssemblerTest, FallbackAssemblesCorrectly) {
    constexpr int64_t N = 5, D = 4;
    FeatureAssembler assembler(D);

    // 2 nodes from "GPU" (actually CPU tensor for this test)
    auto gpu_features = torch::tensor({{1.f, 2.f, 3.f, 4.f},
                                       {5.f, 6.f, 7.f, 8.f}});
    std::vector<uint32_t> gpu_pos = {0, 3};

    // 3 nodes from CPU buffer
    float cpu_data[] = {10.f,  20.f,  30.f,  40.f,
                        50.f,  60.f,  70.f,  80.f,
                        90.f, 100.f, 110.f, 120.f};
    std::vector<uint32_t> cpu_pos = {1, 2, 4};

    auto output = assembler.assemble(N, gpu_features, gpu_pos,
                                     cpu_data, 3, cpu_pos);

    EXPECT_EQ(output.size(0), N);
    EXPECT_EQ(output.size(1), D);

    auto acc = output.accessor<float, 2>();
    // Position 0: gpu row 0 -> [1, 2, 3, 4]
    EXPECT_FLOAT_EQ(acc[0][0], 1.f);
    EXPECT_FLOAT_EQ(acc[0][1], 2.f);
    EXPECT_FLOAT_EQ(acc[0][2], 3.f);
    EXPECT_FLOAT_EQ(acc[0][3], 4.f);
    // Position 3: gpu row 1 -> [5, 6, 7, 8]
    EXPECT_FLOAT_EQ(acc[3][0], 5.f);
    EXPECT_FLOAT_EQ(acc[3][1], 6.f);
    EXPECT_FLOAT_EQ(acc[3][2], 7.f);
    EXPECT_FLOAT_EQ(acc[3][3], 8.f);
    // Position 1: cpu row 0 -> [10, 20, 30, 40]
    EXPECT_FLOAT_EQ(acc[1][0], 10.f);
    EXPECT_FLOAT_EQ(acc[1][1], 20.f);
    EXPECT_FLOAT_EQ(acc[1][2], 30.f);
    EXPECT_FLOAT_EQ(acc[1][3], 40.f);
    // Position 2: cpu row 1 -> [50, 60, 70, 80]
    EXPECT_FLOAT_EQ(acc[2][0], 50.f);
    EXPECT_FLOAT_EQ(acc[2][1], 60.f);
    EXPECT_FLOAT_EQ(acc[2][2], 70.f);
    EXPECT_FLOAT_EQ(acc[2][3], 80.f);
    // Position 4: cpu row 2 -> [90, 100, 110, 120]
    EXPECT_FLOAT_EQ(acc[4][0], 90.f);
    EXPECT_FLOAT_EQ(acc[4][1], 100.f);
    EXPECT_FLOAT_EQ(acc[4][2], 110.f);
    EXPECT_FLOAT_EQ(acc[4][3], 120.f);
}

// =============================================================================
// Edge Cases: Empty Partitions
// =============================================================================

TEST(FeatureAssemblerTest, EmptyGpuFeatures) {
    FeatureAssembler assembler(4);
    auto empty_gpu = torch::empty({0, 4});
    float cpu_data[] = {1.f, 2.f, 3.f, 4.f};
    auto output = assembler.assemble(1, empty_gpu, {}, cpu_data, 1, {0});

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 4);
    EXPECT_FLOAT_EQ(output[0][0].item<float>(), 1.f);
    EXPECT_FLOAT_EQ(output[0][1].item<float>(), 2.f);
    EXPECT_FLOAT_EQ(output[0][2].item<float>(), 3.f);
    EXPECT_FLOAT_EQ(output[0][3].item<float>(), 4.f);
}

TEST(FeatureAssemblerTest, EmptyCpuFeatures) {
    FeatureAssembler assembler(4);
    auto gpu = torch::tensor({{1.f, 2.f, 3.f, 4.f}});
    auto output = assembler.assemble(1, gpu, {0}, nullptr, 0, {});

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 4);
    EXPECT_FLOAT_EQ(output[0][0].item<float>(), 1.f);
    EXPECT_FLOAT_EQ(output[0][1].item<float>(), 2.f);
    EXPECT_FLOAT_EQ(output[0][2].item<float>(), 3.f);
    EXPECT_FLOAT_EQ(output[0][3].item<float>(), 4.f);
}

TEST(FeatureAssemblerTest, AllEmpty) {
    FeatureAssembler assembler(4);
    auto output = assembler.assemble(0, torch::empty({0, 4}), {},
                                     nullptr, 0, {});
    EXPECT_EQ(output.size(0), 0);
    EXPECT_EQ(output.size(1), 4);
}

// =============================================================================
// Constructor Validation
// =============================================================================

TEST(FeatureAssemblerTest, InvalidFeatureDimThrows) {
    EXPECT_THROW(FeatureAssembler(0), std::invalid_argument);
    EXPECT_THROW(FeatureAssembler(-1), std::invalid_argument);
}

TEST(FeatureAssemblerTest, ValidFeatureDimDoesNotThrow) {
    EXPECT_NO_THROW(FeatureAssembler(1));
    EXPECT_NO_THROW(FeatureAssembler(128));
    EXPECT_NO_THROW(FeatureAssembler(4096));
}

// =============================================================================
// Single-Element Partitions
// =============================================================================

TEST(FeatureAssemblerTest, SingleGpuNode) {
    constexpr int64_t D = 3;
    FeatureAssembler assembler(D);
    auto gpu = torch::tensor({{10.f, 20.f, 30.f}});
    auto output = assembler.assemble(1, gpu, {0}, nullptr, 0, {});

    auto acc = output.accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc[0][0], 10.f);
    EXPECT_FLOAT_EQ(acc[0][1], 20.f);
    EXPECT_FLOAT_EQ(acc[0][2], 30.f);
}

TEST(FeatureAssemblerTest, SingleCpuNode) {
    constexpr int64_t D = 3;
    FeatureAssembler assembler(D);
    float cpu_data[] = {10.f, 20.f, 30.f};
    auto output = assembler.assemble(1, torch::empty({0, D}), {},
                                     cpu_data, 1, {0});

    auto acc = output.accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc[0][0], 10.f);
    EXPECT_FLOAT_EQ(acc[0][1], 20.f);
    EXPECT_FLOAT_EQ(acc[0][2], 30.f);
}

// =============================================================================
// Large Feature Dimension
// =============================================================================

TEST(FeatureAssemblerTest, LargeFeatureDim) {
    constexpr int64_t N = 4, D = 512;
    FeatureAssembler assembler(D);

    // 2 GPU nodes, 2 CPU nodes
    auto gpu = torch::arange(2 * D, torch::kFloat32).reshape({2, D});
    std::vector<uint32_t> gpu_pos = {0, 2};

    std::vector<float> cpu_data(2 * D);
    std::iota(cpu_data.begin(), cpu_data.end(), 1000.f);
    std::vector<uint32_t> cpu_pos = {1, 3};

    auto output = assembler.assemble(N, gpu, gpu_pos,
                                     cpu_data.data(), 2, cpu_pos);

    ASSERT_EQ(output.size(0), N);
    ASSERT_EQ(output.size(1), D);

    auto acc = output.accessor<float, 2>();
    // gpu row 0 -> position 0: first element is 0.f
    EXPECT_FLOAT_EQ(acc[0][0], 0.f);
    // gpu row 1 -> position 2: first element is D (512.f)
    EXPECT_FLOAT_EQ(acc[2][0], static_cast<float>(D));
    // cpu row 0 -> position 1: first element is 1000.f
    EXPECT_FLOAT_EQ(acc[1][0], 1000.f);
    // cpu row 1 -> position 3: first element is 1000 + D
    EXPECT_FLOAT_EQ(acc[3][0], 1000.f + static_cast<float>(D));
}

// =============================================================================
// Non-Contiguous Position Indices
// =============================================================================

TEST(FeatureAssemblerTest, SparsePositions) {
    constexpr int64_t D = 2;
    FeatureAssembler assembler(D);

    // 10 total nodes, but only fill positions 1, 5, 9
    auto gpu = torch::tensor({{1.f, 2.f}});
    float cpu_data[] = {3.f, 4.f, 5.f, 6.f};

    auto output = assembler.assemble(10, gpu, {5},
                                     cpu_data, 2, {1, 9});

    auto acc = output.accessor<float, 2>();
    // Unfilled positions should be zero
    EXPECT_FLOAT_EQ(acc[0][0], 0.f);
    EXPECT_FLOAT_EQ(acc[0][1], 0.f);
    // Position 5: gpu
    EXPECT_FLOAT_EQ(acc[5][0], 1.f);
    EXPECT_FLOAT_EQ(acc[5][1], 2.f);
    // Position 1: cpu row 0
    EXPECT_FLOAT_EQ(acc[1][0], 3.f);
    EXPECT_FLOAT_EQ(acc[1][1], 4.f);
    // Position 9: cpu row 1
    EXPECT_FLOAT_EQ(acc[9][0], 5.f);
    EXPECT_FLOAT_EQ(acc[9][1], 6.f);
    // Spot-check another unfilled position
    EXPECT_FLOAT_EQ(acc[7][0], 0.f);
}

// =============================================================================
// assemble_simple Interface
// =============================================================================

TEST(FeatureAssemblerTest, AssembleSimpleMatchesAssemble) {
    constexpr int64_t N = 4, D = 3;
    FeatureAssembler assembler(D);

    auto gpu = torch::tensor({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    std::vector<uint32_t> gpu_pos = {0, 2};

    auto cpu = torch::tensor({{7.f, 8.f, 9.f}, {10.f, 11.f, 12.f}});
    std::vector<uint32_t> cpu_pos = {1, 3};

    // Use assemble_simple (takes torch::Tensor for CPU features)
    auto simple_out = assembler.assemble_simple(N, gpu, gpu_pos, cpu, cpu_pos);

    // Use assemble (takes raw float* for CPU features)
    auto cpu_contig = cpu.contiguous();
    auto raw_out = assembler.assemble(N, gpu, gpu_pos,
                                      cpu_contig.data_ptr<float>(), 2, cpu_pos);

    auto diff = (simple_out - raw_out).abs().max().item<float>();
    EXPECT_FLOAT_EQ(diff, 0.f);
}

TEST(FeatureAssemblerTest, AssembleSimpleEmptyCpu) {
    constexpr int64_t D = 3;
    FeatureAssembler assembler(D);

    auto gpu = torch::tensor({{1.f, 2.f, 3.f}});
    auto empty_cpu = torch::empty({0, D});

    auto output = assembler.assemble_simple(1, gpu, {0}, empty_cpu, {});
    EXPECT_FLOAT_EQ(output[0][0].item<float>(), 1.f);
}

// =============================================================================
// Fallback Explicitly Called (Exposed for Testing)
// =============================================================================

TEST(FeatureAssemblerTest, FallbackDirectCall) {
    constexpr int64_t N = 3, D = 2;
    FeatureAssembler assembler(D);

    auto gpu = torch::tensor({{1.f, 2.f}});
    float cpu_data[] = {3.f, 4.f, 5.f, 6.f};

    auto output = assembler.assemble_fallback(N, gpu, {0},
                                              cpu_data, 2, {1, 2});

    auto acc = output.accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc[0][0], 1.f);
    EXPECT_FLOAT_EQ(acc[0][1], 2.f);
    EXPECT_FLOAT_EQ(acc[1][0], 3.f);
    EXPECT_FLOAT_EQ(acc[1][1], 4.f);
    EXPECT_FLOAT_EQ(acc[2][0], 5.f);
    EXPECT_FLOAT_EQ(acc[2][1], 6.f);
}

// =============================================================================
// Interleaved GPU/CPU Positions
// =============================================================================

TEST(FeatureAssemblerTest, InterleavedPositions) {
    constexpr int64_t N = 6, D = 2;
    FeatureAssembler assembler(D);

    // GPU features at even positions, CPU at odd positions
    auto gpu = torch::tensor({{10.f, 11.f}, {20.f, 21.f}, {30.f, 31.f}});
    std::vector<uint32_t> gpu_pos = {0, 2, 4};

    float cpu_data[] = {40.f, 41.f, 50.f, 51.f, 60.f, 61.f};
    std::vector<uint32_t> cpu_pos = {1, 3, 5};

    auto output = assembler.assemble(N, gpu, gpu_pos, cpu_data, 3, cpu_pos);

    auto acc = output.accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc[0][0], 10.f);
    EXPECT_FLOAT_EQ(acc[1][0], 40.f);
    EXPECT_FLOAT_EQ(acc[2][0], 20.f);
    EXPECT_FLOAT_EQ(acc[3][0], 50.f);
    EXPECT_FLOAT_EQ(acc[4][0], 30.f);
    EXPECT_FLOAT_EQ(acc[5][0], 60.f);
}

// =============================================================================
// Many Nodes (Stress Test for Thread Cooperation)
// =============================================================================

TEST(FeatureAssemblerTest, ManyNodes) {
    constexpr int64_t N = 1000, D = 64;
    constexpr int64_t K_GPU = 200, K_CPU = 800;
    FeatureAssembler assembler(D);

    // GPU: positions 0..199
    auto gpu = torch::arange(K_GPU * D, torch::kFloat32).reshape({K_GPU, D});
    std::vector<uint32_t> gpu_pos(K_GPU);
    std::iota(gpu_pos.begin(), gpu_pos.end(), 0);

    // CPU: positions 200..999
    std::vector<float> cpu_data(K_CPU * D);
    for (int64_t i = 0; i < K_CPU * D; ++i) {
        cpu_data[i] = static_cast<float>(10000 + i);
    }
    std::vector<uint32_t> cpu_pos(K_CPU);
    std::iota(cpu_pos.begin(), cpu_pos.end(), static_cast<uint32_t>(K_GPU));

    auto output = assembler.assemble(N, gpu, gpu_pos,
                                     cpu_data.data(), K_CPU, cpu_pos);

    ASSERT_EQ(output.size(0), N);
    ASSERT_EQ(output.size(1), D);

    auto acc = output.accessor<float, 2>();

    // Check a few GPU positions
    EXPECT_FLOAT_EQ(acc[0][0], 0.f);
    EXPECT_FLOAT_EQ(acc[199][0], 199.f * D);

    // Check a few CPU positions
    EXPECT_FLOAT_EQ(acc[200][0], 10000.f);
    EXPECT_FLOAT_EQ(acc[999][0], 10000.f + (K_CPU - 1) * D);
}

// =============================================================================
// CUDA Path (only runs when CUDA is available)
// =============================================================================

// =============================================================================
// Large Dimension Tiling (D=2048 exceeds typical CUDA block size of 256)
// CUDA kernel should tile: for (feat = threadIdx.x; feat < D; feat += blockDim.x)
// =============================================================================

TEST(FeatureAssemblerTest, LargeDimensionTiling) {
    constexpr int64_t N = 4, D = 2048;
    FeatureAssembler assembler(D);

    auto gpu = torch::randn({2, D});
    std::vector<uint32_t> gpu_pos = {0, 2};

    std::vector<float> cpu_data(2 * D);
    for (auto& v : cpu_data) v = 1.0f;
    std::vector<uint32_t> cpu_pos = {1, 3};

    auto output = assembler.assemble(N, gpu, gpu_pos, cpu_data.data(), 2, cpu_pos);
    EXPECT_EQ(output.size(0), N);
    EXPECT_EQ(output.size(1), D);

    // Verify CPU rows are all 1.0
    auto acc = output.accessor<float, 2>();
    for (int d = 0; d < D; ++d) {
        EXPECT_FLOAT_EQ(acc[1][d], 1.0f) << "CPU row 1, dim " << d;
        EXPECT_FLOAT_EQ(acc[3][d], 1.0f) << "CPU row 3, dim " << d;
    }

    // Verify GPU rows match the random tensor
    auto gpu_acc = gpu.accessor<float, 2>();
    for (int d = 0; d < D; ++d) {
        EXPECT_FLOAT_EQ(acc[0][d], gpu_acc[0][d]) << "GPU row 0, dim " << d;
        EXPECT_FLOAT_EQ(acc[2][d], gpu_acc[1][d]) << "GPU row 2, dim " << d;
    }
}

// =============================================================================
// Position Bounds Validation (throws std::out_of_range)
// =============================================================================

TEST(FeatureAssemblerTest, GpuPositionOutOfBoundsThrows) {
    FeatureAssembler assembler(4);
    auto gpu = torch::randn({1, 4});
    // Position 5 >= total_nodes 3 -> should throw
    EXPECT_THROW(
        assembler.assemble(3, gpu, {5}, nullptr, 0, {}),
        std::out_of_range);
}

TEST(FeatureAssemblerTest, CpuPositionOutOfBoundsThrows) {
    FeatureAssembler assembler(4);
    float data[] = {1, 2, 3, 4};
    // Position 5 >= total_nodes 2 -> should throw
    EXPECT_THROW(
        assembler.assemble(2, torch::empty({0, 4}), {}, data, 1, {5}),
        std::out_of_range);
}

// =============================================================================
// CUDA Path (only runs when CUDA is available)
// =============================================================================

#ifdef ENABLE_CUDA_ASSEMBLER
TEST(FeatureAssemblerTest, CudaKernelMatchesFallback) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping CUDA kernel test";
    }

    constexpr int64_t N = 10, D = 8;
    FeatureAssembler assembler(D);

    // Create random test data on GPU
    auto gpu_features = torch::randn({3, D}).to(torch::kCUDA);
    std::vector<uint32_t> gpu_pos = {0, 4, 7};

    // CPU features (7 rows for positions 1,2,3,5,6,8,9)
    std::vector<float> cpu_data(7 * D);
    std::srand(42);
    for (auto& v : cpu_data) {
        v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    std::vector<uint32_t> cpu_pos = {1, 2, 3, 5, 6, 8, 9};

    // Run CUDA path
    auto cuda_result = assembler.assemble(N, gpu_features, gpu_pos,
                                          cpu_data.data(), 7, cpu_pos);

    // Run fallback on CPU
    auto fb_result = assembler.assemble_fallback(N, gpu_features.cpu(), gpu_pos,
                                                 cpu_data.data(), 7, cpu_pos);

    // Compare: the results should match within float precision
    auto diff = (cuda_result.cpu() - fb_result).abs().max().item<float>();
    EXPECT_LT(diff, 1e-5f);
}
#endif

