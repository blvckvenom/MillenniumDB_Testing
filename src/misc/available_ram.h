#pragma once

// Adaptive sort buffer helper for ExternalRecordSort and ExternalEdgeSort.
// Convention: no namespace (matches src/misc/total_ram.h sibling).
// See docs/superpowers/specs/2026-04-14-adaptive-ram-buffer-design.md

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

/// Minimum buffer size. Matches the historical 256 MB hardcoded constant.
inline constexpr size_t DEFAULT_SORT_BUFFER_MIN = 256ULL * 1024 * 1024;

/// Result of adaptive computation with provenance for logging.
struct AdaptiveBufferResult {
    size_t bytes;
    enum Source { ADAPTIVE, ENV, ENV_INVALID } source;
};

// Declarations

inline uint64_t get_mem_available_from(const char* path);
inline uint64_t get_mem_available();
inline AdaptiveBufferResult compute_adaptive_sort_buffer_core(
    uint64_t mem_available, const char* env_value, size_t floor);
inline AdaptiveBufferResult compute_adaptive_sort_buffer_result(
    size_t floor = DEFAULT_SORT_BUFFER_MIN);
inline size_t compute_adaptive_sort_buffer(
    size_t floor = DEFAULT_SORT_BUFFER_MIN);

// Definitions

inline uint64_t get_mem_available_from(const char* path) {
    std::ifstream meminfo(path);
    if (!meminfo.is_open()) return 0;

    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            size_t kb = 0;
            if (std::sscanf(line.c_str(), "MemAvailable: %zu kB", &kb) == 1) {
                return static_cast<uint64_t>(kb) * 1024;
            }
            return 0;
        }
    }
    return 0;
}

inline uint64_t get_mem_available() {
    return get_mem_available_from("/proc/meminfo");
}

inline AdaptiveBufferResult compute_adaptive_sort_buffer_core(
    uint64_t /*mem_available*/, const char* /*env_value*/, size_t floor)
{
    return {floor, AdaptiveBufferResult::ADAPTIVE};
}

inline AdaptiveBufferResult compute_adaptive_sort_buffer_result(size_t floor) {
    return compute_adaptive_sort_buffer_core(
        get_mem_available(), std::getenv("MDB_SORT_BUFFER_MB"), floor);
}

inline size_t compute_adaptive_sort_buffer(size_t floor) {
    return compute_adaptive_sort_buffer_result(floor).bytes;
}
