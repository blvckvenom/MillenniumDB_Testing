#pragma once

#include <cstddef>

namespace mdb::gpu {

struct GpuInfo {
    int    device_id          = -1;
    size_t total_vram         = 0;
    size_t free_vram          = 0;  // safety-derated free (sort/resource planner)
    // Raw cudaMemGetInfo free, NO safety derate. For consumers (e.g. the GNN
    // sampling backend) that compute their own phase-specific absolute headroom
    // and must reason about the true free VRAM, not the blanket-derated value.
    size_t raw_free_vram      = 0;
    int    compute_capability = 0;  // major*10 + minor
};

struct SystemResources {
    bool    has_gpu       = false;
    GpuInfo gpu;
    size_t  ram_available = 0;
    bool    has_tbb       = false;
};

// Query system resources: CPU RAM, GPU VRAM + properties.
// Cheap enough to call once per sort operation.
SystemResources detect_resources();

// Re-query GPU free VRAM (e.g. between sort chunks).
// Returns 0 when no GPU is available.
size_t refresh_gpu_free_vram();

} // namespace mdb::gpu
