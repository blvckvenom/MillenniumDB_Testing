#include "gpu/resource_planner.h"
#include <gtest/gtest.h>

using namespace mdb::gpu;

TEST(ResourcePlannerTest, SmallDatasetChoosesCpuSequential) {
    SystemResources res;
    res.has_gpu = true;
    res.gpu.free_vram = 4000000000;  // 4 GB
    res.ram_available = 64000000000;  // 64 GB
    res.has_tbb = true;

    auto plan = plan_sort(1000, 3, res);  // 1K records, Record<3>
    EXPECT_EQ(plan.strategy, SortStrategy::CPU_SEQUENTIAL);
}

TEST(ResourcePlannerTest, MediumDatasetFitsGpuFull) {
    SystemResources res;
    res.has_gpu = true;
    res.gpu.free_vram = 4000000000;  // 4 GB
    res.ram_available = 64000000000;
    res.has_tbb = true;

    // 50M records × 28 bytes = 1.4 GB < 4 GB → GPU_FULL
    auto plan = plan_sort(50000000, 3, res);
    EXPECT_EQ(plan.strategy, SortStrategy::GPU_FULL);
    EXPECT_EQ(plan.num_chunks, 1u);
    EXPECT_EQ(plan.num_passes, 3u);
}

TEST(ResourcePlannerTest, LargeDatasetGpuChunked) {
    SystemResources res;
    res.has_gpu = true;
    res.gpu.free_vram = 4000000000;  // 4 GB
    res.ram_available = 64000000000;
    res.has_tbb = true;

    // 500M records × 28 bytes = 14 GB > 4 GB → GPU_CHUNKED
    auto plan = plan_sort(500000000, 3, res);
    EXPECT_EQ(plan.strategy, SortStrategy::GPU_CHUNKED);
    EXPECT_GT(plan.num_chunks, 1u);
    EXPECT_EQ(plan.num_passes, 3u);
}

TEST(ResourcePlannerTest, NoGpuFallsToCpuParallel) {
    SystemResources res;
    res.has_gpu = false;
    res.ram_available = 64000000000;
    res.has_tbb = true;

    auto plan = plan_sort(50000000, 3, res);
    EXPECT_EQ(plan.strategy, SortStrategy::CPU_PARALLEL);
}

TEST(ResourcePlannerTest, NoGpuNoTbbFallsToCpuSequential) {
    SystemResources res;
    res.has_gpu = false;
    res.ram_available = 64000000000;
    res.has_tbb = false;

    auto plan = plan_sort(50000000, 3, res);
    EXPECT_EQ(plan.strategy, SortStrategy::CPU_SEQUENTIAL);
}

TEST(ResourcePlannerTest, InsufficientRamFallsToExternal) {
    SystemResources res;
    res.has_gpu = false;
    res.ram_available = 1000000000;  // 1 GB
    res.has_tbb = true;

    // 500M records × 24 bytes = 12 GB > 1 GB × 0.7 → EXTERNAL_SORT
    auto plan = plan_sort(500000000, 3, res);
    EXPECT_EQ(plan.strategy, SortStrategy::EXTERNAL_SORT);
}

TEST(ResourcePlannerTest, Record2Uses24BytesPerRecord) {
    SystemResources res;
    res.has_gpu = true;
    res.gpu.free_vram = 4000000000;
    res.ram_available = 64000000000;
    res.has_tbb = true;

    // Record<2>: 24 bytes/record. 100M × 24 = 2.4 GB < 4 GB → GPU_FULL
    auto plan = plan_sort(100000000, 2, res);
    EXPECT_EQ(plan.strategy, SortStrategy::GPU_FULL);
    EXPECT_EQ(plan.num_passes, 2u);
}
