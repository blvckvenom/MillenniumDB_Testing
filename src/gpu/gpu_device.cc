#include "gpu/gpu_device.h"

#include <cstdio>
#include <fstream>
#include <string>

#ifdef MDB_GPU_ENABLED
#include <cuda_runtime.h>
#endif

namespace mdb::gpu {

// Reserve 20% of VRAM as safety margin to avoid OOM on the GPU.
static constexpr double VRAM_SAFETY_FACTOR = 0.80;

// Parse MemAvailable from /proc/meminfo (Linux).
static size_t parse_memavailable() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return 0;
    }

    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            size_t kb = 0;
            if (std::sscanf(line.c_str(), "MemAvailable: %zu kB", &kb) == 1) {
                return kb * 1024;
            }
            return 0;
        }
    }
    return 0;
}

SystemResources detect_resources() {
    SystemResources res;

    // CPU RAM
    res.ram_available = parse_memavailable();

    // TBB (compile-time flag set by root CMakeLists.txt)
#ifdef HAS_TBB
    res.has_tbb = true;
#endif

    // GPU (only when built with CUDA support)
#ifdef MDB_GPU_ENABLED
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            res.has_gpu                = true;
            res.gpu.device_id          = 0;
            res.gpu.total_vram         = prop.totalGlobalMem;
            res.gpu.compute_capability = prop.major * 10 + prop.minor;

            size_t free_bytes  = 0;
            size_t total_bytes = 0;
            int prev_device    = 0;
            cudaGetDevice(&prev_device);
            cudaSetDevice(0);
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                res.gpu.raw_free_vram = free_bytes;
                res.gpu.free_vram = static_cast<size_t>(
                    static_cast<double>(free_bytes) * VRAM_SAFETY_FACTOR);
            }
            cudaSetDevice(prev_device);
        }
    }
#endif

    return res;
}

size_t refresh_gpu_free_vram() {
#ifdef MDB_GPU_ENABLED
    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(0);
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    size_t result      = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        result = static_cast<size_t>(
            static_cast<double>(free_bytes) * VRAM_SAFETY_FACTOR);
    }
    cudaSetDevice(prev_device);
    return result;
#endif
    return 0;
}

} // namespace mdb::gpu
