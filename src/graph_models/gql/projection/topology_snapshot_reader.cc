#include "graph_models/gql/projection/topology_snapshot_reader.h"

#include <fcntl.h>       // open, O_RDONLY
#include <sys/mman.h>    // mmap, munmap, madvise, MAP_PRIVATE, MADV_RANDOM
#include <sys/stat.h>    // fstat
#include <unistd.h>      // close

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <openssl/evp.h>

namespace GQL::Projection {

namespace {

// Output filename for each direction; mirrors topology_snapshot_writer.cc.
const char* output_basename_for(TopologySnapshotReader::Direction d) {
    switch (d) {
    case TopologySnapshotReader::Direction::FORWARD: return "topology_fwd.csr";
    case TopologySnapshotReader::Direction::REVERSE: return "topology_rev.csr";
    }
    return "topology_fwd.csr";  // unreachable; keeps compiler happy
}

// Source `.leaf` basename for each direction. Mirrors the matching
// helper in topology_snapshot_writer.cc so producer and consumer agree on
// *which* file's bytes feed the SHA-256 — keeps the producer/consumer hash
// chain symmetric.
const char* source_basename_for(TopologySnapshotReader::Direction d) {
    switch (d) {
    case TopologySnapshotReader::Direction::FORWARD: return "from_to_edge.leaf";
    case TopologySnapshotReader::Direction::REVERSE: return "to_from_edge.leaf";
    }
    return "from_to_edge.leaf";  // unreachable
}

// Symmetric sidecar filename. Unlike the directional files it has no single
// source — its combined digest chains the fixed-order pair below.
const char* sym_output_basename() { return "topology_sym.csr"; }

// Process-level memoization of the source-.leaf SHA-256, keyed by
// (path, mtime_ns, size). The four-level sample path hashes the SAME .leaf
// TWICE per run — once in the TopologyAccessor ctor (opening fwd_csr_/rev_csr_)
// and once in FourLevelTopologyStore::open_l3_sidecars_ — measured at 174s
// (cold) + 83s (warm) = 257s on papers100M (72.4 GB .leaf), ~67% of the whole
// sample stage. The second pass is pure redundancy: same file, same mtime/size
// → identical digest. This cache makes it an O(1) lookup, and the (mtime,size)
// key keeps the staleness gate correct (any .leaf edit changes the key →
// recompute). Thread-safe for the parallel sampler workers (though both passes
// happen single-threaded during setup today).
struct ShaCacheEntry {
    int64_t                 mtime_ns = 0;
    int64_t                 size     = 0;
    std::array<uint8_t, 32> digest{};
};
std::mutex                                   g_sha_cache_mutex;
std::map<std::string, ShaCacheEntry>         g_sha_cache;

// Stream SHA-256 over `path` using EVP. Memoized (see above) + reads via a
// raw fd with POSIX_FADV_SEQUENTIAL so the cold first pass gets aggressive
// kernel readahead instead of the ~415 MB/s the ifstream path was limited to.
//
// Returns true + fills `out` on success. Returns false on any I/O,
// allocation, or digest-API failure — the caller treats that as a
// mismatch (conservative: an unverifiable sidecar must not be trusted).
bool compute_sha256_64k(const std::filesystem::path& path,
                        std::array<uint8_t, 32>&     out) {
    // stat for the cache key (mtime + size). On failure fall through to the
    // open() below, which will also fail and return false.
    struct stat st{};
    int64_t key_mtime = 0, key_size = -1;
    if (::stat(path.c_str(), &st) == 0) {
        key_mtime = static_cast<int64_t>(st.st_mtime) * 1000000000LL
                  + static_cast<int64_t>(st.st_mtim.tv_nsec);
        key_size  = static_cast<int64_t>(st.st_size);
        std::lock_guard<std::mutex> lk(g_sha_cache_mutex);
        auto it = g_sha_cache.find(path.string());
        if (it != g_sha_cache.end()
            && it->second.mtime_ns == key_mtime
            && it->second.size == key_size) {
            out = it->second.digest;  // cache hit — skip the re-hash
            return true;
        }
    }

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    // Cold sequential scan of a multi-GB .leaf — ask the kernel for large
    // readahead. Best-effort; ignore failure.
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        ::close(fd);
        return false;
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        ::close(fd);
        return false;
    }

    // 256 KiB read buffer (vs the old 64 KiB ifstream chunk) — fewer syscalls
    // on a 72 GB file; the SHA digest is byte-identical regardless of chunking.
    constexpr std::size_t BUF = 256 * 1024;
    std::array<char, BUF> buf{};
    bool ok = true;
    while (true) {
        ssize_t n = ::read(fd, buf.data(), BUF);
        if (n < 0) {
            if (errno == EINTR) continue;
            ok = false;
            break;
        }
        if (n == 0) break;  // EOF
        if (EVP_DigestUpdate(ctx, buf.data(), static_cast<std::size_t>(n)) != 1) {
            ok = false;
            break;
        }
    }
    ::close(fd);
    if (!ok) {
        EVP_MD_CTX_free(ctx);
        return false;
    }

    unsigned int len = 0;
    if (EVP_DigestFinal_ex(ctx, out.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        return false;
    }
    EVP_MD_CTX_free(ctx);
    if (len != 32) {
        return false;
    }

    // Populate the cache so the redundant second pass over the same .leaf is
    // an O(1) hit.
    if (key_size >= 0) {
        std::lock_guard<std::mutex> lk(g_sha_cache_mutex);
        g_sha_cache[path.string()] = ShaCacheEntry{key_mtime, key_size, out};
    }
    return len == 32;
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

    // Step 4 — parse header (validates magic bytes, format version, and
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

    // Step 5 — file-size invariant: verify the file is exactly the size the
    // header metadata implies.
    // ROW_PTR is always uint64; COL_IDX / EDGE_IDS use the header's id_width
    // (8 = full tagged ObjectId; 4 = tag-stripped uint32, where the 8-bit
    // ObjectId type tag is stripped at write time and reconstructed on read
    // by OR-ing the per-section type tag stored in the header, losslessly
    // halving the topology file size for large graphs).
    // expected = 64 + 8 * (N + 1) + W * M * (has_edge_ids ? 2 : 1)
    const bool has_edge_ids_flag =
        (header.flags & TopologySnapshotFlags::kHasEdgeIds) != 0;
    const uint64_t N = header.num_nodes;
    const uint64_t M = header.num_edges;
    const uint64_t W = static_cast<uint64_t>(header.id_width);  // 4 or 8 (parse-validated)
    const uint64_t expected =
        static_cast<uint64_t>(kTopologySnapshotHeaderSize)
        + sizeof(uint64_t) * (N + 1)
        + W * M * (has_edge_ids_flag ? 2 : 1);
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

    // Step 8 — ROW_PTR structural invariants: ROW_PTR[0]==0, ROW_PTR[N]==M,
    // and monotonically non-decreasing (verified up to kMonotonicityScanNodeLimit).
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

    // All structural checks passed — provisionally commit to the reader
    // so verify_source_sha256() can see header_ and has_data_ = true.
    reader.header_    = header;
    reader.map_base_  = map_base;
    reader.file_size_ = file_size;
    reader.fd_        = fd;

    reader.row_ptr_ = row_ptr;
    // COL_IDX starts right after ROW_PTR (uint64[N+1]); EDGE_IDS (if present)
    // right after COL_IDX. Section element width is W (4 or 8). The base
    // offsets are 8-aligned (header is 64, ROW_PTR is 8*(N+1)), so the uint32
    // reinterpret is naturally aligned; EDGE_IDS at col_base + 4*M is at least
    // 4-aligned, which is sufficient for uint32 reads.
    const uint8_t* col_base = map_bytes + kTopologySnapshotHeaderSize
                                        + sizeof(uint64_t) * (N + 1);
    const uint8_t* eid_base = col_base + W * M;
    if (header.id_width == kTopologySnapshotIdWidthNarrow) {
        reader.col_idx32_  = reinterpret_cast<const uint32_t*>(col_base);
        reader.edge_ids32_ = has_edge_ids_flag
            ? reinterpret_cast<const uint32_t*>(eid_base)
            : nullptr;
    } else {
        reader.col_idx_  = reinterpret_cast<const uint64_t*>(col_base);
        reader.edge_ids_ = has_edge_ids_flag
            ? reinterpret_cast<const uint64_t*>(eid_base)
            : nullptr;
    }
    reader.has_data_ = true;

    // Step 9 — staleness gate: re-hash the source `.leaf` with SHA-256 and
    // compare the result against the digest the writer embedded in the sidecar
    // header at build time. If the .leaf has changed since the sidecar was
    // written (e.g. the projection was rebuilt), the digest will differ and
    // the sidecar must be considered stale and unsafe to use.
    // On mismatch (or any I/O / OpenSSL failure) emit a one-line warning
    // and fall back to the B+Tree by wiping this reader's state, so the
    // caller sees has_data() == false — identical to the "file absent"
    // contract. Missing source file or unreadable source both collapse
    // to "mismatch" via compute_sha256_64k() returning false.
    //
    // Opt-out: MDB_GNN_TRUST_SIDECAR=1 skips the re-hash entirely. The SHA-256
    // streams the full source .leaf (72.4 GB on papers100M GNN_MINIMAL, ~81s /
    // ~21% of a four-level sample build, recomputed EVERY open) purely as a
    // staleness gate. When the caller KNOWS the projection's .leaf is unchanged
    // since the sidecar was built (the common repeated-sampling workflow), this
    // trades the safety net for the time. Default OFF (verify) — a stale
    // sidecar silently producing wrong topology corrupts every downstream
    // sample, so trust must be explicit.
    {
        const char* trust = std::getenv("MDB_GNN_TRUST_SIDECAR");
        if (trust && (trust[0] == '1' || trust[0] == 't' || trust[0] == 'T')) {
            std::cerr << "TopologySnapshotReader: " << path.string()
                      << ": MDB_GNN_TRUST_SIDECAR set — skipping source .leaf "
                         "SHA-256 staleness check (caller asserts freshness)\n";
            return reader;
        }
    }
    const std::filesystem::path source_leaf_path =
        projection_dir / source_basename_for(dir);
    if (!reader.verify_source_sha256(source_leaf_path)) {
        warn(path,
             "source SHA-256 mismatch for " + source_leaf_path.string()
             + ", falling back to B+Tree");
        reader.release_resources_();
        return reader;
    }
    return reader;
}

// ---------------------------------------------------------------------------
// open_symmetric() — same validation pipeline as open(), three differences:
//   (1) reads topology_sym.csr,
//   (2) parses with parse_topology_snapshot_sym_header (magic "TOPOSYM1"),
//   (3) staleness gate is the COMBINED two-source digest over
//       {from_to_edge.leaf, to_from_edge.leaf} (verify_combined_sha256).
// The body is intentionally a faithful copy of open() rather than a shared
// helper: open() is the validated directional path (papers100M-proven) and
// keeping the symmetric opener structurally identical-but-separate avoids any
// risk of regressing it. Every failure path emits warn() + has_data_==false.
// ---------------------------------------------------------------------------

TopologySnapshotReader TopologySnapshotReader::open_symmetric(
    const std::filesystem::path& projection_dir) {
    TopologySnapshotReader reader;
    const std::filesystem::path path = projection_dir / sym_output_basename();

    // Step 1 — absent is the normal "sidecar not built" path. No log.
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return reader;
    }

    // Step 2 — open RDONLY.
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        warn(path, "open(O_RDONLY) failed (errno=" + std::to_string(errno) + ")");
        return reader;
    }

    // Step 3 — fstat → file size ≥ 64.
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

    // Step 4 — read + parse the SYMMETRIC header (validates magic "TOPOSYM1",
    // version, id_width).
    uint8_t hdr_buf[kTopologySnapshotHeaderSize];
    {
        std::size_t off = 0;
        while (off < kTopologySnapshotHeaderSize) {
            ssize_t got = ::pread(fd, hdr_buf + off,
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
        header = parse_topology_snapshot_sym_header(hdr_buf);
    } catch (const TopologySnapshotFormatError& e) {
        warn(path, std::string("sym header validation rejected: ") + e.what());
        ::close(fd);
        return reader;
    }

    // Step 5 — file-size invariant.
    const bool has_edge_ids_flag =
        (header.flags & TopologySnapshotFlags::kHasEdgeIds) != 0;
    const uint64_t N = header.num_nodes;
    const uint64_t M = header.num_edges;
    const uint64_t W = static_cast<uint64_t>(header.id_width);
    const uint64_t expected =
        static_cast<uint64_t>(kTopologySnapshotHeaderSize)
        + sizeof(uint64_t) * (N + 1)
        + W * M * (has_edge_ids_flag ? 2 : 1);
    if (expected != static_cast<uint64_t>(file_size)) {
        warn(path, "file size mismatch: expected=" + std::to_string(expected)
                   + ", actual=" + std::to_string(file_size)
                   + " (N=" + std::to_string(N) + ", M=" + std::to_string(M)
                   + ", has_edge_ids=" + (has_edge_ids_flag ? "1" : "0") + ")");
        ::close(fd);
        return reader;
    }

    // Step 6 — mmap MAP_PRIVATE.
    void* map_base = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map_base == MAP_FAILED) {
        warn(path, "mmap failed (errno=" + std::to_string(errno) + ")");
        ::close(fd);
        return reader;
    }

    // Step 7 — MADV_RANDOM (best-effort).
    if (::madvise(map_base, file_size, MADV_RANDOM) != 0) {
        // Not fatal; some mount options reject it — don't log spuriously.
    }

    // Step 8 — ROW_PTR structural invariants.
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
        if (row_ptr[0] != 0) {
            warn(path, "ROW_PTR[0] != 0 for N=0 graph");
            ::munmap(map_base, file_size);
            ::close(fd);
            return reader;
        }
    }

    // Provisionally commit so verify_combined_sha256() sees header_/has_data_.
    reader.header_    = header;
    reader.map_base_  = map_base;
    reader.file_size_ = file_size;
    reader.fd_        = fd;

    reader.row_ptr_ = row_ptr;
    const uint8_t* col_base = map_bytes + kTopologySnapshotHeaderSize
                                        + sizeof(uint64_t) * (N + 1);
    const uint8_t* eid_base = col_base + W * M;
    if (header.id_width == kTopologySnapshotIdWidthNarrow) {
        reader.col_idx32_  = reinterpret_cast<const uint32_t*>(col_base);
        reader.edge_ids32_ = has_edge_ids_flag
            ? reinterpret_cast<const uint32_t*>(eid_base)
            : nullptr;
    } else {
        reader.col_idx_  = reinterpret_cast<const uint64_t*>(col_base);
        reader.edge_ids_ = has_edge_ids_flag
            ? reinterpret_cast<const uint64_t*>(eid_base)
            : nullptr;
    }
    reader.has_data_ = true;

    // Step 9 — two-source staleness gate (combined chained digest over BOTH
    // .leaf streams). MDB_GNN_TRUST_SIDECAR opts out, same as open().
    {
        const char* trust = std::getenv("MDB_GNN_TRUST_SIDECAR");
        if (trust && (trust[0] == '1' || trust[0] == 't' || trust[0] == 'T')) {
            std::cerr << "TopologySnapshotReader: " << path.string()
                      << ": MDB_GNN_TRUST_SIDECAR set — skipping combined "
                         "two-source SHA-256 staleness check\n";
            return reader;
        }
    }
    const std::vector<std::filesystem::path> sources = {
        projection_dir / "from_to_edge.leaf",
        projection_dir / "to_from_edge.leaf"
    };
    if (!reader.verify_combined_sha256(sources)) {
        warn(path, "combined two-source SHA-256 mismatch, falling back to B+Tree");
        reader.release_resources_();
        return reader;
    }
    return reader;
}

// ---------------------------------------------------------------------------
// Destructor + move
// ---------------------------------------------------------------------------

void TopologySnapshotReader::release_resources_() noexcept {
    if (map_base_ != nullptr && file_size_ > 0) {
        ::munmap(map_base_, file_size_);
    }
    map_base_   = nullptr;
    file_size_  = 0;
    row_ptr_    = nullptr;
    col_idx_    = nullptr;
    edge_ids_   = nullptr;
    col_idx32_  = nullptr;
    edge_ids32_ = nullptr;
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
    : has_data_   (other.has_data_)
    , header_     (other.header_)
    , map_base_   (other.map_base_)
    , file_size_  (other.file_size_)
    , row_ptr_    (other.row_ptr_)
    , col_idx_    (other.col_idx_)
    , edge_ids_   (other.edge_ids_)
    , col_idx32_  (other.col_idx32_)
    , edge_ids32_ (other.edge_ids32_)
    , fd_         (other.fd_)
{
    // Detach `other` so its destructor is a no-op.
    other.has_data_   = false;
    other.header_     = {};
    other.map_base_   = nullptr;
    other.file_size_  = 0;
    other.row_ptr_    = nullptr;
    other.col_idx_    = nullptr;
    other.edge_ids_   = nullptr;
    other.col_idx32_  = nullptr;
    other.edge_ids32_ = nullptr;
    other.fd_         = -1;
}

TopologySnapshotReader& TopologySnapshotReader::operator=(
    TopologySnapshotReader&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    release_resources_();

    has_data_   = other.has_data_;
    header_     = other.header_;
    map_base_   = other.map_base_;
    file_size_  = other.file_size_;
    row_ptr_    = other.row_ptr_;
    col_idx_    = other.col_idx_;
    edge_ids_   = other.edge_ids_;
    col_idx32_  = other.col_idx32_;
    edge_ids32_ = other.edge_ids32_;
    fd_         = other.fd_;

    other.has_data_   = false;
    other.header_     = {};
    other.map_base_   = nullptr;
    other.file_size_  = 0;
    other.row_ptr_    = nullptr;
    other.col_idx_    = nullptr;
    other.edge_ids_   = nullptr;
    other.col_idx32_  = nullptr;
    other.edge_ids32_ = nullptr;
    other.fd_         = -1;
    return *this;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

uint64_t TopologySnapshotReader::degree(uint64_t node_idx) const {
    if (!has_data_) {
        throw std::out_of_range(
            "TopologySnapshotReader::degree: reader has no data "
            "(call has_data() first)");
    }
    if (node_idx >= header_.num_nodes) {
        throw std::out_of_range(
            "TopologySnapshotReader::degree: node_idx "
            + std::to_string(node_idx) + " >= num_nodes "
            + std::to_string(header_.num_nodes));
    }
    // ROW_PTR is uint64 for both id widths, so degree is width-agnostic.
    return row_ptr_[node_idx + 1] - row_ptr_[node_idx];
}

ConstU64Span TopologySnapshotReader::neighbors(uint64_t node_idx) const {
    if (!has_data_) {
        throw std::out_of_range(
            "TopologySnapshotReader::neighbors: reader has no data "
            "(call has_data() first)");
    }
    if (header_.id_width != kTopologySnapshotIdWidth) {
        // Narrow (uint32) layout: a uint64 span would be a wrong
        // reinterpret-cast of tag-stripped uint32 ordinals. Direct callers
        // to the width-agnostic copy_neighbors() (or raw col_idx32_row()).
        throw TopologySnapshotFormatError(
            "TopologySnapshotReader::neighbors: uint64 span unavailable for "
            "id_width=4 (use copy_neighbors() or col_idx32_row())");
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
    if (header_.id_width != kTopologySnapshotIdWidth) {
        throw TopologySnapshotFormatError(
            "TopologySnapshotReader::edge_ids: uint64 span unavailable for "
            "id_width=4 (use copy_edge_ids() or edge_ids32_row())");
    }
    if (node_idx >= header_.num_nodes) {
        throw std::out_of_range(
            "TopologySnapshotReader::edge_ids: node_idx "
            + std::to_string(node_idx) + " >= num_nodes "
            + std::to_string(header_.num_nodes));
    }
    if (edge_ids_ == nullptr) {
        return {};  // No EDGE_IDS section in this file — empty span, not an error.
    }
    const uint64_t start = row_ptr_[node_idx];
    const uint64_t end   = row_ptr_[node_idx + 1];
    return ConstU64Span(edge_ids_ + start,
                        static_cast<std::size_t>(end - start));
}

// ---------------------------------------------------------------------------
// Width-agnostic copy accessors (work for id_width ∈ {4, 8})
// ---------------------------------------------------------------------------
//
// For id_width==8 the stored values are full tagged ObjectIds → straight
// memcpy. For id_width==4 each stored uint32 is the tag-stripped ordinal;
// we widen it and OR the per-section type tag (carried in the header) back
// into the top byte, reproducing the EXACT tagged ObjectId the uint64 layout
// would have stored. This is the losslessness contract that lets the cora
// byte-identical batches.dat gate pass for both widths.

void TopologySnapshotReader::copy_neighbors(
    uint64_t node_idx, std::vector<uint64_t>& out) const {
    if (!has_data_) {
        throw std::out_of_range(
            "TopologySnapshotReader::copy_neighbors: reader has no data "
            "(call has_data() first)");
    }
    if (node_idx >= header_.num_nodes) {
        throw std::out_of_range(
            "TopologySnapshotReader::copy_neighbors: node_idx "
            + std::to_string(node_idx) + " >= num_nodes "
            + std::to_string(header_.num_nodes));
    }
    const uint64_t start = row_ptr_[node_idx];
    const uint64_t end   = row_ptr_[node_idx + 1];
    const std::size_t deg = static_cast<std::size_t>(end - start);
    out.reserve(out.size() + deg);
    if (header_.id_width == kTopologySnapshotIdWidthNarrow) {
        const uint64_t tag =
            static_cast<uint64_t>(header_.dst_type_tag) << 56;
        for (uint64_t i = start; i < end; ++i) {
            out.push_back(tag | static_cast<uint64_t>(col_idx32_[i]));
        }
    } else {
        out.insert(out.end(), col_idx_ + start, col_idx_ + end);
    }
}

void TopologySnapshotReader::copy_edge_ids(
    uint64_t node_idx, std::vector<uint64_t>& out) const {
    if (!has_data_) {
        throw std::out_of_range(
            "TopologySnapshotReader::copy_edge_ids: reader has no data "
            "(call has_data() first)");
    }
    if (node_idx >= header_.num_nodes) {
        throw std::out_of_range(
            "TopologySnapshotReader::copy_edge_ids: node_idx "
            + std::to_string(node_idx) + " >= num_nodes "
            + std::to_string(header_.num_nodes));
    }
    const bool has_eids =
        (header_.flags & TopologySnapshotFlags::kHasEdgeIds) != 0;
    if (!has_eids) {
        return;  // No EDGE_IDS section — append nothing (matches edge_ids()).
    }
    const uint64_t start = row_ptr_[node_idx];
    const uint64_t end   = row_ptr_[node_idx + 1];
    const std::size_t deg = static_cast<std::size_t>(end - start);
    out.reserve(out.size() + deg);
    if (header_.id_width == kTopologySnapshotIdWidthNarrow) {
        const uint64_t tag =
            static_cast<uint64_t>(header_.edge_type_tag) << 56;
        for (uint64_t i = start; i < end; ++i) {
            out.push_back(tag | static_cast<uint64_t>(edge_ids32_[i]));
        }
    } else {
        out.insert(out.end(), edge_ids_ + start, edge_ids_ + end);
    }
}

const uint32_t* TopologySnapshotReader::col_idx32_row(uint64_t node_idx) const {
    if (!has_data_ || header_.id_width != kTopologySnapshotIdWidthNarrow
        || col_idx32_ == nullptr) {
        return nullptr;
    }
    if (node_idx >= header_.num_nodes) {
        throw std::out_of_range(
            "TopologySnapshotReader::col_idx32_row: node_idx "
            + std::to_string(node_idx) + " >= num_nodes "
            + std::to_string(header_.num_nodes));
    }
    return col_idx32_ + row_ptr_[node_idx];
}

const uint32_t* TopologySnapshotReader::edge_ids32_row(uint64_t node_idx) const {
    if (!has_data_ || header_.id_width != kTopologySnapshotIdWidthNarrow
        || edge_ids32_ == nullptr) {
        return nullptr;
    }
    if (node_idx >= header_.num_nodes) {
        throw std::out_of_range(
            "TopologySnapshotReader::edge_ids32_row: node_idx "
            + std::to_string(node_idx) + " >= num_nodes "
            + std::to_string(header_.num_nodes));
    }
    return edge_ids32_ + row_ptr_[node_idx];
}

void TopologySnapshotReader::advise_access(bool sequential) const noexcept {
    if (!has_data_ || map_base_ == nullptr || file_size_ == 0) {
        return;
    }
    // Best-effort perf hint only — never affects correctness, so failures
    // (EINVAL on some mount options) are silently ignored, matching the
    // open()-time MADV_RANDOM policy.
    ::madvise(map_base_, file_size_, sequential ? MADV_SEQUENTIAL : MADV_RANDOM);
}

void TopologySnapshotReader::prefetch_rows(uint64_t start_row,
                                           uint64_t end_row) const noexcept {
    if (!has_data_ || map_base_ == nullptr || row_ptr_ == nullptr) {
        return;
    }
    if (end_row > header_.num_nodes) end_row = header_.num_nodes;
    if (start_row >= end_row) return;

    const uint64_t elem_start = row_ptr_[start_row];
    const uint64_t elem_end   = row_ptr_[end_row];
    if (elem_end <= elem_start) return;

    const bool narrow = header_.id_width == kTopologySnapshotIdWidthNarrow;
    const std::size_t width = narrow ? sizeof(uint32_t) : sizeof(uint64_t);
    const char* base = narrow ? reinterpret_cast<const char*>(col_idx32_)
                              : reinterpret_cast<const char*>(col_idx_);
    if (base == nullptr) return;

    char* ptr = const_cast<char*>(base) + elem_start * width;
    const std::size_t len = static_cast<std::size_t>(elem_end - elem_start) * width;
    ::madvise(ptr, len, MADV_WILLNEED);  // best-effort; ignore failures
}

// ---------------------------------------------------------------------------
// verify_source_sha256 — streaming SHA-256 staleness gate.
// ---------------------------------------------------------------------------
//
// Streams the source .leaf file through SHA-256 and compares the result
// against the digest embedded in the CSR sidecar header at write time.
// Returns true iff the digest matches, confirming the sidecar is fresh.
// Any failure mode (reader has no data, source file absent, I/O error,
// OpenSSL context allocation) returns false and is treated by callers as
// equivalent to a mismatch — conservative by design: an unverifiable
// sidecar must not be trusted.

bool TopologySnapshotReader::verify_source_sha256(
    const std::filesystem::path& source_leaf_path) const {
    if (!has_data_) {
        return false;  // nothing to verify against
    }
    std::array<uint8_t, 32> actual{};
    if (!compute_sha256_64k(source_leaf_path, actual)) {
        return false;
    }
    return std::memcmp(actual.data(), header_.source_sha256, 32) == 0;
}

// ---------------------------------------------------------------------------
// verify_combined_sha256 — chained two-source staleness gate (symmetric).
// ---------------------------------------------------------------------------
//
// Streams the source .leaf files through ONE SHA-256 context in list order
// (matching the writer's sha256_of_files_chained) and compares to the combined
// digest the symmetric writer embedded in header_.source_sha256. The per-file
// compute_sha256_64k cache cannot express a chained multi-file digest, so this
// re-streams both files; the staleness-gate cost is paid only at open time.
// Conservative: reader-has-no-data / unreadable source / order or content
// mismatch all return false (untrusted).

bool TopologySnapshotReader::verify_combined_sha256(
    const std::vector<std::filesystem::path>& source_leaf_paths) const {
    if (!has_data_) {
        return false;
    }
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return false;
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return false;
    }
    constexpr std::size_t BUF = 256 * 1024;
    std::array<char, BUF> buf{};
    for (const auto& path : source_leaf_paths) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            EVP_MD_CTX_free(ctx);
            return false;  // unreadable source → untrusted
        }
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        bool ok = true;
        while (true) {
            ssize_t n = ::read(fd, buf.data(), BUF);
            if (n < 0) { if (errno == EINTR) continue; ok = false; break; }
            if (n == 0) break;
            if (EVP_DigestUpdate(ctx, buf.data(),
                                 static_cast<std::size_t>(n)) != 1) {
                ok = false; break;
            }
        }
        ::close(fd);
        if (!ok) { EVP_MD_CTX_free(ctx); return false; }
    }
    std::array<uint8_t, 32> actual{};
    unsigned int len = 0;
    if (EVP_DigestFinal_ex(ctx, actual.data(), &len) != 1 || len != 32) {
        EVP_MD_CTX_free(ctx);
        return false;
    }
    EVP_MD_CTX_free(ctx);
    return std::memcmp(actual.data(), header_.source_sha256, 32) == 0;
}

}  // namespace GQL::Projection
