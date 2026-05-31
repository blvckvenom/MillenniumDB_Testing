// Tests for posix_fadvise/madvise convenience wrappers used by Fix #22.
//
// These helpers must remain safe and idempotent: callers always invoke
// them on regions they're *about to stop using*, so an error must NEVER
// abort the surrounding I/O.

#include "gnn/common/page_cache_hint.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace fs = std::filesystem;
using mdb::gnn::fadvise_dontneed;
using mdb::gnn::madvise_dontneed;

namespace {

fs::path make_temp_file(size_t bytes) {
    auto p = fs::temp_directory_path() /
             ("mdb_pchint_" + std::to_string(::getpid()) + "_" +
              std::to_string(std::rand()));
    std::ofstream f(p, std::ios::binary);
    std::vector<char> buf(bytes, 'x');
    f.write(buf.data(), buf.size());
    f.close();
    return p;
}

} // namespace

// 1. fadvise_dontneed returns 0 on a valid fd.
TEST(PageCacheHint, FadviseValidFd) {
    auto p = make_temp_file(4096);
    int fd = ::open(p.c_str(), O_RDONLY);
    ASSERT_GT(fd, 0);
    EXPECT_EQ(fadvise_dontneed(fd, 0, 4096), 0);
    ::close(fd);
    fs::remove(p);
}

// 2. fadvise_dontneed on an invalid fd returns errno (does NOT throw).
TEST(PageCacheHint, FadviseClosedFdReturnsErrno) {
    int rc = fadvise_dontneed(-1, 0, 4096);
    EXPECT_NE(rc, 0);  // EBADF or similar
}

// 3. madvise_dontneed on a freshly mmap'd region returns 0.
TEST(PageCacheHint, MadviseValidRegion) {
    auto p = make_temp_file(4096);
    int fd = ::open(p.c_str(), O_RDONLY);
    void* m = ::mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT_NE(m, MAP_FAILED);
    EXPECT_EQ(madvise_dontneed(m, 4096), 0);
    ::munmap(m, 4096);
    ::close(fd);
    fs::remove(p);
}

// 4. The pch_detail::disabled() cache makes setenv() ineffective
//    after the first call. Test the short-circuit paths that hold
//    regardless of env state: null ptr / zero-byte mmap. Note: we
//    cannot assert fadvise_dontneed(-1, 0, 0) == 0 here because
//    len == 0 now means "from offset to EOF" per posix_fadvise(2)
//    and reaches the syscall (which returns EBADF for fd=-1).
TEST(PageCacheHint, EnvDisableEffectIfRunFirst) {
    // The pch_detail::disabled() cache makes setenv() ineffective
    // after the first call. Test the short-circuit paths that hold
    // regardless of env state: null ptr / zero-byte mmap.
    EXPECT_EQ(madvise_dontneed(nullptr, 0), 0);
}
