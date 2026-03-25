#include "gpu/resource_planner.h"
#include "gpu/sort/gpu_sort.h"
#include "storage/index/record.h"

#include <algorithm>
#include <random>

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

// ---------------------------------------------------------------------------
// sort_and_stream tests
// ---------------------------------------------------------------------------

TEST(GpuSortTest, CpuSequentialSortsCorrectly) {
    std::mt19937 rng(42);
    std::vector<Record<3>> records(10000);
    for (auto& r : records) {
        r[0] = rng() % 1000;
        r[1] = rng() % 1000;
        r[2] = rng() % 10000;
    }

    auto expected = records;
    std::sort(expected.begin(), expected.end());

    std::vector<Record<3>> result;
    result.reserve(records.size());

    SystemResources res;
    res.has_gpu = false;
    res.ram_available = 1ULL << 30;
    res.has_tbb = false;

    bool ok = sort_and_stream<3>(
        records, {}, {}, records.size(),
        [&result](const Record<3>& r) { result.push_back(r); },
        res);

    EXPECT_TRUE(ok);
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(GpuSortTest, GpuFullSortsCorrectly) {
    auto res = mdb::gpu::detect_resources();
    if (!res.has_gpu) {
        GTEST_SKIP() << "No GPU available";
    }

    std::mt19937 rng(42);
    std::vector<Record<3>> records(1000000);  // 1M records
    for (auto& r : records) {
        r[0] = rng() % 100000;
        r[1] = rng() % 100000;
        r[2] = rng() % 1000000;
    }

    auto expected = records;
    std::sort(expected.begin(), expected.end());

    std::vector<Record<3>> result;
    result.reserve(records.size());

    mdb::gpu::PlannerConfig config;
    config.min_records_gpu = 100;  // Low threshold to force GPU

    bool ok = mdb::gpu::sort_and_stream<3>(
        records, {}, {}, records.size(),
        [&result](const Record<3>& r) { result.push_back(r); },
        res, config);

    EXPECT_TRUE(ok);
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < std::min(result.size(), expected.size()); i++) {
        EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(GpuSortTest, SortPreservesEdgeIdOrder) {
    // Records with same (from, to) must be ordered by edge_id
    std::vector<Record<3>> records = {
        {5, 3, 101},
        {5, 3, 100},
        {5, 3, 102},
        {5, 4, 50},
        {5, 3, 99},
    };

    std::vector<Record<3>> result;
    SystemResources res;
    res.has_gpu = false;
    res.ram_available = 1ULL << 30;
    res.has_tbb = false;

    sort_and_stream<3>(
        records, {}, {}, records.size(),
        [&result](const Record<3>& r) { result.push_back(r); },
        res);

    ASSERT_EQ(result.size(), 5u);
    EXPECT_EQ(result[0], (Record<3>{5, 3, 99}));
    EXPECT_EQ(result[1], (Record<3>{5, 3, 100}));
    EXPECT_EQ(result[2], (Record<3>{5, 3, 101}));
    EXPECT_EQ(result[3], (Record<3>{5, 3, 102}));
    EXPECT_EQ(result[4], (Record<3>{5, 4, 50}));
}
