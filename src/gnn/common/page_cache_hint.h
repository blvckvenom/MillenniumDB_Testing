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

namespace mdb::gnn {

namespace pch_detail {
inline bool disabled() {
    static const bool cached = []{
        const char* env = std::getenv("MDB_GNN_NO_FADVISE");
        return env != nullptr && (std::strcmp(env, "1") == 0 ||
                                  std::strcmp(env, "true") == 0 ||
                                  std::strcmp(env, "yes") == 0);
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
 * Fix #22 motivation: papers100M's 56 GB reordered.fmat + 87 GB
 * batches.dat + ~8 GB caches exceed the 30 GB host RAM. Without
 * explicit hints the kernel keeps already-consumed pages in cache
 * until LRU eviction forces them out — by which point productive
 * pages have been swapped. With DONTNEED the working set stays
 * bounded to whatever we're actively reading or writing.
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
