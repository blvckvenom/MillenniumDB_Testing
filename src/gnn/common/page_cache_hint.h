#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include "misc/ablation_registry.h"

namespace mdb::gnn {

namespace pch_detail {
inline bool disabled() {
    // The static is what keeps the registry's mutex off the hot path: the hints
    // fire once per written chunk, per packed-slim file and per consumed
    // batches.dat range, so only the first call may pay for resolving anything.
    static const bool cached = []{
        // "1", "true" and "yes" are the only spellings this switch has ever
        // acted on, so they stay the accepted set and the meaning of a run does
        // not shift. What changes is that any OTHER value is now reported as
        // unrecognised instead of quietly reading as off, which is the case
        // where an ablation arm looks like it disabled the hints and did not.
        return Ablation::choice("MDB_GNN_NO_FADVISE", "0", {"1", "true", "yes"}) != "0";
    }();
    return cached;
}
}  // namespace pch_detail

/**
 * @brief Tell the kernel that we're done with [fd:offset .. offset+len)
 *        and the pages can be evicted from the page cache.
 *
 * Wraps posix_fadvise(2) with POSIX_FADV_DONTNEED. Returns 0 on
 * success, errno on failure. Never throws — callers always invoke
 * this on "we're done with this region" boundaries; an error must
 * not break the surrounding I/O. Disabled by env var
 * MDB_GNN_NO_FADVISE=1 (for ablation studies).
 *
 * Motivation: the build phases stream artifacts (reordered feature
 * matrix, serialized batches, tier caches) whose combined size can be
 * several times the host's RAM — at papers100M scale, roughly 5x on a
 * 30 GB host. Without explicit hints the kernel keeps
 * already-consumed pages in the page cache until LRU eviction forces
 * them out — by which point productive pages have been swapped.
 * Issuing DONTNEED after each region is consumed keeps the active
 * working set bounded to whatever is currently being read or written.
 */
inline int fadvise_dontneed(int fd, off_t offset, off_t len) {
    if (pch_detail::disabled()) return 0;
    // len == 0 means "from offset to EOF" per posix_fadvise(2);
    // only short-circuit on truly invalid negative lengths.
    if (len < 0) return 0;
#if defined(__linux__) && defined(POSIX_FADV_DONTNEED)
    int rc = ::posix_fadvise(fd, offset, len, POSIX_FADV_DONTNEED);
    return rc;  // posix_fadvise returns errno value directly (not via errno)
#else
    (void)fd; (void)offset; (void)len;
    return 0;
#endif
}

/**
 * @brief Tell the kernel that the mmap'd region [ptr .. ptr+len) is no
 *        longer needed; pages may be evicted.
 *
 * Wraps madvise(2) with MADV_DONTNEED. Returns 0 on success, errno on
 * failure. Same never-throw contract as fadvise_dontneed. The address
 * and length are page-aligned inward so we never advise pages we don't
 * own.
 */
inline int madvise_dontneed(void* ptr, std::size_t len) {
    if (pch_detail::disabled()) return 0;
    if (ptr == nullptr || len == 0) return 0;
#if defined(__linux__) && defined(MADV_DONTNEED)
    const long page = ::sysconf(_SC_PAGESIZE);
    const auto p   = reinterpret_cast<std::uintptr_t>(ptr);
    const auto end = p + len;
    const auto aligned_p   = (p + page - 1) & ~static_cast<std::uintptr_t>(page - 1);
    const auto aligned_end = end & ~static_cast<std::uintptr_t>(page - 1);
    if (aligned_end <= aligned_p) return 0;
    int rc = ::madvise(reinterpret_cast<void*>(aligned_p),
                       aligned_end - aligned_p, MADV_DONTNEED);
    return (rc == 0) ? 0 : errno;
#else
    (void)ptr; (void)len;
    return 0;
#endif
}

}  // namespace mdb::gnn
