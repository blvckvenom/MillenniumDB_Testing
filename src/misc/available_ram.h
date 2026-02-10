#pragma once

// Adaptive sort buffer helper for ExternalRecordSort and ExternalEdgeSort.
// Convention: no namespace (matches src/misc/total_ram.h sibling).
//
// Sizing policy: buffer = max(256 MB floor, MemAvailable * 3/4), overridable
// with the MDB_SORT_BUFFER_MB environment variable (integer megabytes).
//
// Why MemAvailable and not total RAM: the sort runs mid-session, after buffer
// pools and other subsystems already hold memory, so total RAM overstates
// what can safely be taken. The Linux kernel documentation ("The /proc
// Filesystem", section "meminfo", https://docs.kernel.org/filesystems/proc.html)
// defines MemAvailable as "an estimate of how much memory is available for
// starting new applications, without swapping" — the budget an in-RAM sort
// can claim. The kernel does not prescribe how much of that budget a process
// should take: the 3/4 ratio is our choice, matching the factor this project
// already applies to import-time buffer-pool sizing (see
// src/import/*/default_config.h), and the 256 MB floor preserves the
// historical fixed buffer size so no workload regresses and a mis-set
// override (e.g. MDB_SORT_BUFFER_MB=1) cannot shrink the buffer below what
// the in-memory sort path needs.
//
// The value is recomputed at each sorter construction (one /proc/meminfo
// read, well under a millisecond) so later sorts observe RAM freed or
// claimed by other processes in between; there is deliberately no caching.

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
    uint64_t mem_available, const char* env_value, size_t floor)
{
    // Path 1: env var override
    if (env_value != nullptr) {
        // Reject leading sign (strtoull accepts '-' via wrap-around; '+' is
        // technically valid but we force callers to pass bare positive ints).
        const char* scan = env_value;
        while (std::isspace(static_cast<unsigned char>(*scan))) ++scan;
        bool sign_prefix = (*scan == '-' || *scan == '+');

        errno = 0;
        char* end = nullptr;
        unsigned long long mb = std::strtoull(env_value, &end, 10);
        bool valid = (end != env_value)
                  && (*end == '\0' ||
                      std::isspace(static_cast<unsigned char>(*end)))
                  && (errno == 0)
                  && (mb > 0)
                  && !sign_prefix;
        if (valid) {
            uint64_t bytes64 = static_cast<uint64_t>(mb) * 1024ULL * 1024ULL;
            size_t bytes = (bytes64 > static_cast<uint64_t>(SIZE_MAX))
                ? SIZE_MAX
                : static_cast<size_t>(bytes64);
            if (bytes < floor) bytes = floor;
            return {bytes, AdaptiveBufferResult::ENV};
        }
        // Fall through to adaptive with ENV_INVALID tag
    }

    // Path 2: adaptive calculation
    uint64_t adaptive64 = (mem_available > (UINT64_MAX / 3))
        ? (mem_available / 4) * 3
        : (mem_available * 3) / 4;
    size_t adaptive = (adaptive64 > static_cast<uint64_t>(SIZE_MAX))
        ? SIZE_MAX
        : static_cast<size_t>(adaptive64);
    if (adaptive < floor) adaptive = floor;

    // Tag: ENV_INVALID if env var was present but rejected, else ADAPTIVE.
    AdaptiveBufferResult::Source src =
        (env_value != nullptr)
            ? AdaptiveBufferResult::ENV_INVALID
            : AdaptiveBufferResult::ADAPTIVE;
    return {adaptive, src};
}

inline AdaptiveBufferResult compute_adaptive_sort_buffer_result(size_t floor) {
    return compute_adaptive_sort_buffer_core(
        get_mem_available(), std::getenv("MDB_SORT_BUFFER_MB"), floor);
}

inline size_t compute_adaptive_sort_buffer(size_t floor) {
    return compute_adaptive_sort_buffer_result(floor).bytes;
}
