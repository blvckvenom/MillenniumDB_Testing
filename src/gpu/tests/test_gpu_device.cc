#include "gpu/gpu_device.h"

#include <gtest/gtest.h>

TEST(GpuDeviceTest, DetectResourcesReturnsValidRAM) {
    auto res = mdb::gpu::detect_resources();
    // RAM should be > 0 on any system with /proc/meminfo
    EXPECT_GT(res.ram_available, 0u);
}

TEST(GpuDeviceTest, DetectResourcesGpuFieldConsistent) {
    auto res = mdb::gpu::detect_resources();
    if (res.has_gpu) {
        EXPECT_GT(res.gpu.total_vram, 0u);
        EXPECT_GT(res.gpu.free_vram, 0u);
        EXPECT_LE(res.gpu.free_vram, res.gpu.total_vram);
        EXPECT_GE(res.gpu.compute_capability, 70);
    } else {
        EXPECT_EQ(res.gpu.total_vram, 0u);
        EXPECT_EQ(res.gpu.free_vram, 0u);
    }
}

TEST(GpuDeviceTest, RefreshGpuVramReturnsConsistent) {
    auto res = mdb::gpu::detect_resources();
    if (res.has_gpu) {
        size_t refreshed = mdb::gpu::refresh_gpu_free_vram();
        EXPECT_GT(refreshed, 0u);
        EXPECT_LE(refreshed, res.gpu.total_vram);
    }
}

TEST(GpuDeviceTest, RefreshGpuVramWithoutGpuReturnsZero) {
    auto res = mdb::gpu::detect_resources();
    if (!res.has_gpu) {
        size_t refreshed = mdb::gpu::refresh_gpu_free_vram();
        EXPECT_EQ(refreshed, 0u);
    }
}

TEST(GpuDeviceTest, DetectResourcesIdempotent) {
    auto res1 = mdb::gpu::detect_resources();
    auto res2 = mdb::gpu::detect_resources();
    // Structural fields must be stable across calls
    EXPECT_EQ(res1.has_gpu, res2.has_gpu);
    EXPECT_EQ(res1.gpu.device_id, res2.gpu.device_id);
    EXPECT_EQ(res1.gpu.total_vram, res2.gpu.total_vram);
    EXPECT_EQ(res1.gpu.compute_capability, res2.gpu.compute_capability);
    EXPECT_EQ(res1.has_tbb, res2.has_tbb);
}

TEST(GpuDeviceTest, DefaultGpuInfoIsZero) {
    mdb::gpu::GpuInfo info;
    EXPECT_EQ(info.device_id, -1);
    EXPECT_EQ(info.total_vram, 0u);
    EXPECT_EQ(info.free_vram, 0u);
    EXPECT_EQ(info.compute_capability, 0);
}

TEST(GpuDeviceTest, DefaultSystemResourcesIsEmpty) {
    mdb::gpu::SystemResources res;
    EXPECT_FALSE(res.has_gpu);
    EXPECT_EQ(res.ram_available, 0u);
    EXPECT_FALSE(res.has_tbb);
}
