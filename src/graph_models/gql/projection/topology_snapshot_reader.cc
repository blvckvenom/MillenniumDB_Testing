#include "graph_models/gql/projection/topology_snapshot_reader.h"

#include <fcntl.h>       // open, O_RDONLY
#include <sys/mman.h>    // mmap, munmap, madvise, MAP_PRIVATE, MADV_RANDOM
#include <sys/stat.h>    // fstat
#include <unistd.h>      // close

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace GQL::Projection {

namespace {

// Output filename for each direction (§5.1), mirrors topology_snapshot_writer.cc.
const char* output_basename_for(TopologySnapshotReader::Direction d) {
    switch (d) {
    case TopologySnapshotReader::Direction::FORWARD: return "topology_fwd.csr";
    case TopologySnapshotReader::Direction::REVERSE: return "topology_rev.csr";
    }
    return "topology_fwd.csr";  // unreachable; keeps compiler happy
}

// Single-line diagnostic, keyed by the full path so log scrubbers can filter.
// Intentionally not a full logging framework — the projection module doesn't
// have one, and "TopologySnapshotReader: <path>: <reason>" is enough grep
// bait for post-hoc triage.
void warn(const std::filesystem::path& path, const std::string& reason) {
    std::cerr << "TopologySnapshotReader: " << path.string()
              << ": " << reason << std::endl;
}

// Upper bound on `num_nodes` for which we perform the O(N) monotonicity
// scan at open time. Above this, we trust the writer invariant — the scan
// cost would measurably hurt `papers100M`-scale open latency (N=111M →
// hundreds of MB of linear reads just to validate ROW_PTR).
constexpr uint64_t kMonotonicityScanNodeLimit = 10'000'000;

}  // namespace

// ---------------------------------------------------------------------------
// open() — the whole validation pipeline lives here. Every failure path
// emits a warn() + returns a reader with has_data_ == false.
// ---------------------------------------------------------------------------

TopologySnapshotReader TopologySnapshotReader::open(
    const std::filesystem::path& projection_dir,
    Direction                    dir) {
    TopologySnapshotReader reader;
    const std::filesystem::path path = projection_dir / output_basename_for(dir);

    // Step 1 — file absent is the normal "sidecar not built" path. No log.
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return reader;
    }

    // Step 2 — open RDONLY. Failure is rare (permission denied / racy
    // unlink), so log it.
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        warn(path, "open(O_RDONLY) failed (errno=" + std::to_string(errno) + ")");
        return reader;
    }

    // Step 3 — fstat → file size. Must be ≥ 64 bytes for a header to exist.
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        warn(path, "fstat failed (errno=" + std::to_string(errno) + ")");
        ::close(fd);
        return reader;
    }
    if (st.st_size < static_cast<off_t>(kTopologySnapshotHeaderSize)) {
        warn(path, "file too small to hold a 64-byte header (size="
                   + std::to_string(st.st_size) + ")");
        ::close(fd);
        return reader;
    }
    const std::size_t file_size = static_cast<std::size_t>(st.st_size);

    // Step 4 — parse header via the T4.3 validator (magic / version /
    // id_width). We read via pread here rather than reading through the
    // eventual mmap so that a header-only failure does not leave an mmap
    // mapped + discarded. One syscall is cheap.
    uint8_t hdr_buf[kTopologySnapshotHeaderSize];
    {
        ssize_t got = 0;
        std::size_t off = 0;
        while (off < kTopologySnapshotHeaderSize) {
            got = ::pread(fd, hdr_buf + off,
                          kTopologySnapshotHeaderSize - off,
                          static_cast<off_t>(off));
            if (got < 0) {
                if (errno == EINTR) continue;
                warn(path, "pread(header) failed (errno="
                           + std::to_string(errno) + ")");
                ::close(fd);
                return reader;
            }
            if (got == 0) {
                warn(path, "short read on header");
                ::close(fd);
                return reader;
            }
            off += static_cast<std::size_t>(got);
        }
    }

    TopologySnapshotHeader header;
    try {
        header = parse_topology_snapshot_header(hdr_buf);
    } catch (const TopologySnapshotFormatError& e) {
        warn(path, std::string("header validation rejected: ") + e.what());
        ::close(fd);
        return reader;
    }

    // Step 5 — file-size invariant (§5.2 step 4).
    // expected = 64 + 8 * (N + 1) + 8 * M * (has_edge_ids ? 2 : 1)
    const bool has_edge_ids_flag =
        (header.flags & TopologySnapshotFlags::kHasEdgeIds) != 0;
    const uint64_t N = header.num_nodes;
    const uint64_t M = header.num_edges;
    const uint64_t expected =
        static_cast<uint64_t>(kTopologySnapshotHeaderSize)
        + sizeof(uint64_t) * (N + 1)
        + sizeof(uint64_t) * M * (has_edge_ids_flag ? 2 : 1);
    if (expected != static_cast<uint64_t>(file_size)) {
        warn(path, "file size mismatch: expected=" + std::to_string(expected)
                   + ", actual=" + std::to_string(file_size)
                   + " (N=" + std::to_string(N) + ", M=" + std::to_string(M)
                   + ", has_edge_ids=" + (has_edge_ids_flag ? "1" : "0") + ")");
        ::close(fd);
        return reader;
    }

    // Step 6 — mmap MAP_PRIVATE for the full file. The ROW_PTR / COL_IDX
    // sections will be accessed via typed pointers cast from `map_base`.
    void* map_base = ::mmap(nullptr, file_size, PROT_READ,
                            MAP_PRIVATE, fd, 0);
    if (map_base == MAP_FAILED) {
        warn(path, "mmap failed (errno=" + std::to_string(errno) + ")");
        ::close(fd);
        return reader;
    }

    // Step 7 — madvise(MADV_RANDOM). Sampling walks the CSR in access
    // patterns dictated by the seed set, not linear traversal, so kernel
    // readahead just pollutes the page cache. Best-effort; ignore failure.
    if (::madvise(map_base, file_size, MADV_RANDOM) != 0) {
        // Not fatal. Not even a warning — some filesystems/kernels return
        // EINVAL for MADV_RANDOM on certain mount options, and we don't
        // want to log spuriously on every open.
    }

    // Step 8 — ROW_PTR structural invariants (§5.2 step 5).
    // ROW_PTR lives immediately after the 64-byte header.
    auto* map_bytes = static_cast<const uint8_t*>(map_base);
    const uint64_t* row_ptr = reinterpret_cast<const uint64_t*>(
        map_bytes + kTopologySnapshotHeaderSize);

    if (N > 0) {
        if (row_ptr[0] != 0) {
            warn(path, "ROW_PTR[0] != 0 (got "
                       + std::to_string(row_ptr[0]) + ")");
            ::munmap(map_base, file_size);
            ::close(fd);
            return reader;
        }
        if (row_ptr[N] != M) {
            warn(path, "ROW_PTR[N] != M (ROW_PTR[" + std::to_string(N) + "]="
                       + std::to_string(row_ptr[N])
                       + ", M=" + std::to_string(M) + ")");
            ::munmap(map_base, file_size);
            ::close(fd);
            return reader;
        }
        if (N <= kMonotonicityScanNodeLimit) {
            for (uint64_t i = 0; i < N; ++i) {
                if (row_ptr[i] > row_ptr[i + 1]) {
                    warn(path, "ROW_PTR not monotonic at i="
                               + std::to_string(i)
                               + " (row_ptr[i]="
                               + std::to_string(row_ptr[i])
                               + ", row_ptr[i+1]="
                               + std::to_string(row_ptr[i + 1]) + ")");
                    ::munmap(map_base, file_size);
                    ::close(fd);
                    return reader;
                }
            }
        } else {
            std::cerr << "TopologySnapshotReader: " << path.string()
                      << ": monotonicity check skipped for N=" << N
                      << " (> " << kMonotonicityScanNodeLimit
                      << "), trusting writer invariant" << std::endl;
        }
    } else {
        // N == 0: a degenerate-but-valid projection. ROW_PTR[0] must be 0
        // and M must be 0. The file-size check above already pinned M via
        // the `expected` formula, so if N=0 passes that, M is also 0.
        if (row_ptr[0] != 0) {
            warn(path, "ROW_PTR[0] != 0 for N=0 graph");
            ::munmap(map_base, file_size);
            ::close(fd);
            return reader;
        }
    }

    // All checks passed — commit to the reader.
    reader.header_    = header;
    reader.map_base_  = map_base;
    reader.file_size_ = file_size;
    reader.fd_        = fd;

    reader.row_ptr_ = row_ptr;
    reader.col_idx_ = reinterpret_cast<const uint64_t*>(
        map_bytes + kTopologySnapshotHeaderSize + sizeof(uint64_t) * (N + 1));
    reader.edge_ids_ = has_edge_ids_flag
        ? reinterpret_cast<const uint64_t*>(
              map_bytes + kTopologySnapshotHeaderSize
                        + sizeof(uint64_t) * (N + 1)
                        + sizeof(uint64_t) * M)
        : nullptr;
    reader.has_data_ = true;
    return reader;
}

// ---------------------------------------------------------------------------
// Destructor + move
// ---------------------------------------------------------------------------

void TopologySnapshotReader::release_resources_() noexcept {
    if (map_base_ != nullptr && file_size_ > 0) {
        ::munmap(map_base_, file_size_);
    }
    map_base_  = nullptr;
    file_size_ = 0;
    row_ptr_   = nullptr;
    col_idx_   = nullptr;
    edge_ids_  = nullptr;
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    has_data_ = false;
    header_   = {};
}

TopologySnapshotReader::~TopologySnapshotReader() {
    release_resources_();
}

TopologySnapshotReader::TopologySnapshotReader(
    TopologySnapshotReader&& other) noexcept
    : has_data_  (other.has_data_)
    , header_    (other.header_)
    , map_base_  (other.map_base_)
    , file_size_ (other.file_size_)
    , row_ptr_   (other.row_ptr_)
    , col_idx_   (other.col_idx_)
    , edge_ids_  (other.edge_ids_)
    , fd_        (other.fd_)
{
    // Detach `other` so its destructor is a no-op.
    other.has_data_  = false;
    other.header_    = {};
    other.map_base_  = nullptr;
    other.file_size_ = 0;
    other.row_ptr_   = nullptr;
    other.col_idx_   = nullptr;
    other.edge_ids_  = nullptr;
    other.fd_        = -1;
}

TopologySnapshotReader& TopologySnapshotReader::operator=(
    TopologySnapshotReader&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    release_resources_();

    has_data_  = other.has_data_;
    header_    = other.header_;
    map_base_  = other.map_base_;
    file_size_ = other.file_size_;
    row_ptr_   = other.row_ptr_;
    col_idx_   = other.col_idx_;
    edge_ids_  = other.edge_ids_;
    fd_        = other.fd_;

    other.has_data_  = false;
    other.header_    = {};
    other.map_base_  = nullptr;
    other.file_size_ = 0;
    other.row_ptr_   = nullptr;
    other.col_idx_   = nullptr;
    other.edge_ids_  = nullptr;
    other.fd_        = -1;
    return *this;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

ConstU64Span TopologySnapshotReader::neighbors(uint64_t node_idx) const {
    if (!has_data_) {
        throw std::out_of_range(
            "TopologySnapshotReader::neighbors: reader has no data "
            "(call has_data() first)");
    }
    if (node_idx >= header_.num_nodes) {
        throw std::out_of_range(
            "TopologySnapshotReader::neighbors: node_idx "
            + std::to_string(node_idx) + " >= num_nodes "
            + std::to_string(header_.num_nodes));
    }
    const uint64_t start = row_ptr_[node_idx];
    const uint64_t end   = row_ptr_[node_idx + 1];
    return ConstU64Span(col_idx_ + start,
                        static_cast<std::size_t>(end - start));
}

ConstU64Span TopologySnapshotReader::edge_ids(uint64_t node_idx) const {
    if (!has_data_) {
        throw std::out_of_range(
            "TopologySnapshotReader::edge_ids: reader has no data "
            "(call has_data() first)");
    }
    if (node_idx >= header_.num_nodes) {
        throw std::out_of_range(
            "TopologySnapshotReader::edge_ids: node_idx "
            + std::to_string(node_idx) + " >= num_nodes "
            + std::to_string(header_.num_nodes));
    }
    if (edge_ids_ == nullptr) {
        return {};  // No EDGE_IDS section in this file — spec §4.3 contract.
    }
    const uint64_t start = row_ptr_[node_idx];
    const uint64_t end   = row_ptr_[node_idx + 1];
    return ConstU64Span(edge_ids_ + start,
                        static_cast<std::size_t>(end - start));
}

// ---------------------------------------------------------------------------
// verify_source_sha256 — T4.5 stub; T4.10 wires in the streaming SHA-256.
// ---------------------------------------------------------------------------

bool TopologySnapshotReader::verify_source_sha256(
    const std::filesystem::path& source_leaf_path) const {
    // Intentionally unused in T4.5 — T4.10 replaces this body with an
    // EVP_DigestUpdate streaming pass over source_leaf_path, compared
    // against header_.source_sha256.
    (void)source_leaf_path;
    return true;
}

}  // namespace GQL::Projection
