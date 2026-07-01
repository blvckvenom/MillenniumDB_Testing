// src/gnn/storage/consolidated_slim_reader.cc
#include "gnn/storage/consolidated_slim_reader.h"

#include <cerrno>
#include <unistd.h>

namespace mdb::gnn {

bool validate_consolidated_header(const ConsolidatedSlimHeader& h,
                                  uint64_t expected_feature_dim,
                                  uint8_t  expected_dtype,
                                  uint64_t expected_perm_fp,
                                  uint64_t expected_meta_sha)
{
    if (!h.is_valid())                              return false;
    if (h.feature_dim != expected_feature_dim)      return false;
    if (h.dtype       != expected_dtype)            return false;
    if (expected_perm_fp != 0 && h.perm_fingerprint != expected_perm_fp)   return false;
    if (expected_meta_sha != 0 && h.meta_sha256_head != expected_meta_sha) return false;
    return true;
}

bool pread_exact(int fd, void* dst, size_t len, uint64_t offset)
{
    if (fd < 0) return false;
    char*  p         = static_cast<char*>(dst);
    size_t remaining = len;
    off_t  off       = static_cast<off_t>(offset);
    while (remaining > 0) {
        ssize_t n = ::pread(fd, p, remaining, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;  // premature EOF
        p         += n;
        off       += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
}

} // namespace mdb::gnn
