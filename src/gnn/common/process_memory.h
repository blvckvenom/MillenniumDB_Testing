#pragma once

// process_memory.h — in-process resident-set-size (RSS) probes.
//
// Header-only, Linux-only (reads /proc/self/status). All functions are
// best-effort: they return 0 (or false) when the procfs entry is unavailable,
// never throw, and are safe to call from any thread.
//
// Motivation: the GNN offline-sampling engine has phases whose peak RAM is the
// real constraint on commodity hosts (e.g. the symmetric-CSR merge + GPU pin
// transiently holds the merged slice AND the directional tiers). Without an
// in-process counter the only way to observe peak RSS was an external monitor.
// These probes let the engine surface the peak (VmHWM) and the current resident
// size (VmRSS) as telemetry, and reset the kernel's high-water mark between
// phases so a per-phase peak can be measured.

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace mdb::gnn {

namespace detail {

// Read a "Key:\t  <number> kB" line from /proc/self/status and return the value
// in BYTES. Returns 0 if the file or key is unavailable.
inline std::size_t read_proc_status_bytes_(const char* key) {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (f == nullptr) return 0;
    const std::size_t klen = std::strlen(key);
    char line[256];
    std::size_t value_kb = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        if (std::strncmp(line, key, klen) == 0) {
            const char* p = line + klen;
            // Skip to the first digit (past ':' and whitespace).
            while (*p != '\0' && (*p < '0' || *p > '9')) ++p;
            unsigned long long v = 0;
            while (*p >= '0' && *p <= '9') {
                v = v * 10ULL + static_cast<unsigned long long>(*p - '0');
                ++p;
            }
            value_kb = static_cast<std::size_t>(v);
            break;
        }
    }
    std::fclose(f);
    return value_kb * 1024ULL;
}

}  // namespace detail

/// Current resident set size of this process in bytes (VmRSS). 0 if unavailable.
inline std::size_t current_rss_bytes() {
    return detail::read_proc_status_bytes_("VmRSS:");
}

/// Peak resident set size since process start — or since the last
/// reset_peak_rss() — in bytes (VmHWM). 0 if unavailable.
inline std::size_t peak_rss_bytes() {
    return detail::read_proc_status_bytes_("VmHWM:");
}

/// Best-effort reset of the kernel's peak-RSS high-water mark (VmHWM) down to
/// the current VmRSS, by writing "5" to /proc/self/clear_refs (the
/// CLEAR_REFS_MM_HIWATER_RSS op, Linux >= 4.0). After this, peak_rss_bytes()
/// tracks the high-water mark of the NEXT phase only. Returns false when the
/// procfs entry is unavailable (e.g. older kernels / sandboxes); callers should
/// treat a per-phase peak as "since process start" in that case.
inline bool reset_peak_rss() {
    std::FILE* f = std::fopen("/proc/self/clear_refs", "w");
    if (f == nullptr) return false;
    const bool ok = (std::fputs("5\n", f) >= 0);
    std::fclose(f);
    return ok;
}

}  // namespace mdb::gnn
