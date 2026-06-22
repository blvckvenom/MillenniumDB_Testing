#include "graph_models/gql/projection/topology_snapshot_writer.h"

#include <fcntl.h>        // open, O_WRONLY, O_CREAT, O_EXCL, O_TRUNC
#include <sys/stat.h>     // mode constants
#include <unistd.h>       // write, pwrite, fsync, close

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <openssl/evp.h>

namespace GQL::Projection {

namespace {

// Source `.leaf` basename for each direction (§3.7).
const char* source_basename_for(TopologySnapshotWriter::Direction d) {
    switch (d) {
    case TopologySnapshotWriter::Direction::FORWARD: return "from_to_edge.leaf";
    case TopologySnapshotWriter::Direction::REVERSE: return "to_from_edge.leaf";
    }
    return "from_to_edge.leaf";  // unreachable
}

// Output filename for each direction (§5.1).
const char* output_basename_for(TopologySnapshotWriter::Direction d) {
    switch (d) {
    case TopologySnapshotWriter::Direction::FORWARD: return "topology_fwd.csr";
    case TopologySnapshotWriter::Direction::REVERSE: return "topology_rev.csr";
    }
    return "topology_fwd.csr";
}

// Mirrors `fsync_dir_impl` from gnn/output/model_checkpoint.cc — opens the
// directory read-only and fsyncs it so the rename is durable.
void fsync_directory(const std::filesystem::path& dir) {
    int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "TopologySnapshotWriter: cannot open directory for fsync: "
            + dir.string() + " (errno=" + std::to_string(errno) + ")");
    }
    if (::fsync(fd) != 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(
            "TopologySnapshotWriter: fsync failed on " + dir.string()
            + " (errno=" + std::to_string(e) + ")");
    }
    ::close(fd);
}

// Stream SHA-256 over `path` using EVP. Mirrors
// ModelCheckpoint::compute_gnn_meta_hash (model_checkpoint.cc:75-115).
std::array<uint8_t, 32> sha256_of_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "TopologySnapshotWriter: cannot open source .leaf for SHA-256: "
            + path.string());
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("TopologySnapshotWriter: EVP_MD_CTX_new failed");
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error(
            "TopologySnapshotWriter: EVP_DigestInit_ex(SHA-256) failed");
    }

    // 64 KiB amortizes syscall overhead on multi-GB papers100M-scale .leaf
    // files (~37 GB under GNN_MINIMAL) — 16× fewer read() calls than the
    // 4 KiB reference pattern from model_checkpoint.cc at no extra memory
    // cost on this hot one-shot code path.
    constexpr std::size_t BUF = 64 * 1024;
    std::array<char, BUF> buf;
    while (f.read(buf.data(), BUF) || f.gcount() > 0) {
        if (EVP_DigestUpdate(ctx, buf.data(), static_cast<std::size_t>(f.gcount())) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error(
                "TopologySnapshotWriter: EVP_DigestUpdate failed");
        }
    }

    std::array<uint8_t, 32> digest{};
    unsigned int len = 0;
    if (EVP_DigestFinal_ex(ctx, digest.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error(
            "TopologySnapshotWriter: EVP_DigestFinal_ex failed");
    }
    EVP_MD_CTX_free(ctx);

    if (len != 32) {
        throw std::runtime_error(
            "TopologySnapshotWriter: SHA-256 produced unexpected digest length "
            + std::to_string(len));
    }
    return digest;
}

// Stream SHA-256 over a fixed-order list of files, chaining all bytes through
// one EVP context. The result is the symmetric sidecar's combined two-source
// digest. Empty/absent paths in the list are skipped by the caller's guard.
std::array<uint8_t, 32> sha256_of_files_chained(
    const std::vector<std::filesystem::path>& paths) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("TopologySnapshotWriter: EVP_MD_CTX_new failed");
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error(
            "TopologySnapshotWriter: EVP_DigestInit_ex(SHA-256) failed");
    }
    constexpr std::size_t BUF = 64 * 1024;
    std::array<char, BUF> buf;
    for (const auto& path : paths) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error(
                "TopologySnapshotWriter: cannot open source .leaf for combined "
                "SHA-256: " + path.string());
        }
        while (f.read(buf.data(), BUF) || f.gcount() > 0) {
            if (EVP_DigestUpdate(ctx, buf.data(),
                                 static_cast<std::size_t>(f.gcount())) != 1) {
                EVP_MD_CTX_free(ctx);
                throw std::runtime_error(
                    "TopologySnapshotWriter: EVP_DigestUpdate failed (chained)");
            }
        }
    }
    std::array<uint8_t, 32> digest{};
    unsigned int len = 0;
    if (EVP_DigestFinal_ex(ctx, digest.data(), &len) != 1 || len != 32) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error(
            "TopologySnapshotWriter: EVP_DigestFinal_ex failed (chained)");
    }
    EVP_MD_CTX_free(ctx);
    return digest;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TopologySnapshotWriter::TopologySnapshotWriter(
    const std::filesystem::path& projection_dir,
    Direction                    dir,
    uint64_t                     num_nodes,
    std::vector<uint64_t>        degrees,
    bool                         include_edge_ids)
    : projection_dir_   (projection_dir)
    , direction_        (dir)
    , num_nodes_        (num_nodes)
    , num_edges_        (0)
    , include_edge_ids_ (include_edge_ids)
    , source_leaf_path_ (projection_dir / source_basename_for(dir))
    , final_path_       (projection_dir / output_basename_for(dir))
    , tmp_path_         (final_path_.string() + ".tmp")
{
    if (degrees.size() != num_nodes_) {
        throw std::invalid_argument(
            "TopologySnapshotWriter: degrees.size() ("
            + std::to_string(degrees.size())
            + ") != num_nodes (" + std::to_string(num_nodes_) + ")");
    }

    // Build row_ptr as prefix sum of degrees. row_ptr has length N+1 and
    // encodes invariant row_ptr[0]=0, row_ptr[N]=M (§5.1).
    row_ptr_.resize(static_cast<std::size_t>(num_nodes_) + 1);
    row_ptr_[0] = 0;
    uint64_t running = 0;
    for (uint64_t i = 0; i < num_nodes_; ++i) {
        running += degrees[i];
        row_ptr_[i + 1] = running;
    }
    num_edges_ = running;

    // uint32 topology sidecar eligibility. Opt-in via the `MDB_GNN_TOPOLOGY_UINT32`
    // env var; default OFF keeps the legacy uint64 layout byte-identical. The
    // narrow layout stores tag-stripped ordinals (node id & VALUE_MASK, edge id
    // & VALUE_MASK) as uint32, which is lossless because ObjectId encodes the
    // 8-bit type tag in the top byte — stripping it and storing only the lower
    // 56-bit value payload halves on-disk topology size (~27 GB vs ~54 GB for
    // papers100M). The tag is captured once per section into the header and
    // reconstructed by the reader via OR with the constant type tag. Lossless
    // only when every stripped ordinal fits in uint32 — node ordinals < num_nodes,
    // edge ordinals < num_edges, both < 2^32. ROW_PTR stays uint64 (offsets can
    // exceed 2^32 for M > 4B).
    {
        bool want_narrow = false;
        if (const char* e = std::getenv("MDB_GNN_TOPOLOGY_UINT32")) {
            want_narrow = (e[0] == '1' || e[0] == 't' || e[0] == 'T');
        }
        const uint64_t kU32 = uint64_t{1} << 32;
        const bool node_fits = num_nodes_ < kU32;
        const bool edge_fits = !include_edge_ids_ || num_edges_ < kU32;
        id_width_ = (want_narrow && node_fits && edge_fits)
                        ? kTopologySnapshotIdWidthNarrow
                        : kTopologySnapshotIdWidth;
    }

    write_cursor_.assign(static_cast<std::size_t>(num_nodes_), 0);

    // Reserve 1 MiB of staging space for each per-section coalescing
    // buffer. Append_edge() pushes words here instead of issuing a
    // pwrite per call; the buffer flushes when full and once at
    // finalize(). Keeps peak RAM at ~2 MiB regardless of graph size.
    col_idx_buf_.reserve(kCoalesceBytes / sizeof(uint64_t));
    if (include_edge_ids_) {
        edge_ids_buf_.reserve(kCoalesceBytes / sizeof(uint64_t));
    }
    if (id_width_ == kTopologySnapshotIdWidthNarrow) {
        col_idx_buf32_.reserve(kCoalesceBytes / sizeof(uint64_t));
        if (include_edge_ids_) {
            edge_ids_buf32_.reserve(kCoalesceBytes / sizeof(uint64_t));
        }
    }

    // Section offsets. Fixed once N, M, and id_width_ are known. ROW_PTR is
    // uint64; COL_IDX / EDGE_IDS use element_size_() (4 or 8).
    col_idx_offset_  = static_cast<uint64_t>(kTopologySnapshotHeaderSize)
                     + sizeof(uint64_t) * (num_nodes_ + 1);
    edge_ids_offset_ = col_idx_offset_
                     + element_size_() * num_edges_;
    const uint64_t file_size =
        col_idx_offset_
        + element_size_() * num_edges_ * (include_edge_ids_ ? 2 : 1);

    // Open .tmp with O_EXCL so a concurrent writer targeting the same file
    // is rejected with EEXIST rather than racing on the content.
    out_fd_ = ::open(tmp_path_.c_str(),
                     O_WRONLY | O_CREAT | O_EXCL | O_TRUNC,
                     S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (out_fd_ < 0) {
        int e = errno;
        throw std::runtime_error(
            "TopologySnapshotWriter: cannot create " + tmp_path_.string()
            + " (errno=" + std::to_string(e)
            + (e == EEXIST ? " / another writer already holds this file" : "")
            + ")");
    }

    // If any step between here and end-of-ctor throws, release the fd and
    // remove the .tmp so a retry doesn't hit O_EXCL on the next attempt.
    // The destructor does not run on a partially constructed object.
    try {
        // Truncate to the full expected size so subsequent pwrites always
        // land inside the file (they do not extend it themselves).
        if (::ftruncate(out_fd_, static_cast<off_t>(file_size)) != 0) {
            int e = errno;
            throw std::runtime_error(
                "TopologySnapshotWriter: ftruncate failed on " + tmp_path_.string()
                + " to " + std::to_string(file_size) + " bytes"
                + " (errno=" + std::to_string(e) + ")");
        }

        // Step 1 (§5.3): 64-byte zero placeholder. Rewritten in finalize().
        std::array<uint8_t, kTopologySnapshotHeaderSize> zero_header{};
        pwrite_all(zero_header.data(), zero_header.size(), 0);

        // Step 2 (§5.3 extended): write ROW_PTR immediately. The body layout
        // is fixed — ROW_PTR is right after the header — so we can commit it
        // now. COL_IDX and (optionally) EDGE_IDS are filled in by
        // append_edge() via pwrite at the per-edge offsets implied by row_ptr_.
        pwrite_all(row_ptr_.data(),
                   row_ptr_.size() * sizeof(uint64_t),
                   kTopologySnapshotHeaderSize);
    } catch (...) {
        ::close(out_fd_);
        out_fd_ = -1;
        std::error_code ec;
        std::filesystem::remove(tmp_path_, ec);
        throw;
    }
}

// ---------------------------------------------------------------------------
// Constructor — explicit basename + chained multi-source hash (symmetric CSR)
// ---------------------------------------------------------------------------
//
// Body is byte-identical to the Direction ctor above (degrees check, row_ptr
// prefix sum, uint32 eligibility, coalescing-buffer reserves, section offsets,
// O_EXCL open / ftruncate / placeholder-header / ROW_PTR prewrite). Only the
// init-list differs: final_path_ is the explicit basename, the source set is
// the chained `source_leaf_paths_` list, and the two symmetric-mode members
// are set. The source hashing itself happens in finalize().

TopologySnapshotWriter::TopologySnapshotWriter(
    const std::filesystem::path&             projection_dir,
    std::string                              output_basename,
    std::vector<std::filesystem::path>       source_leaf_paths,
    uint64_t                                 num_nodes,
    std::vector<uint64_t>                    degrees,
    bool                                     include_edge_ids,
    bool                                     symmetric_format)
    : projection_dir_    (projection_dir)
    , direction_         (Direction::FORWARD)   // unused in this mode
    , num_nodes_         (num_nodes)
    , num_edges_         (0)
    , include_edge_ids_  (include_edge_ids)
    , source_leaf_path_  ()                      // unused; combined hash uses the list
    , final_path_        (projection_dir / output_basename)
    , tmp_path_          (final_path_.string() + ".tmp")
    , source_leaf_paths_ (std::move(source_leaf_paths))
    , symmetric_format_  (symmetric_format)
{
    if (degrees.size() != num_nodes_) {
        throw std::invalid_argument(
            "TopologySnapshotWriter: degrees.size() ("
            + std::to_string(degrees.size())
            + ") != num_nodes (" + std::to_string(num_nodes_) + ")");
    }

    // Build row_ptr as prefix sum of degrees. row_ptr has length N+1 and
    // encodes invariant row_ptr[0]=0, row_ptr[N]=M (§5.1).
    row_ptr_.resize(static_cast<std::size_t>(num_nodes_) + 1);
    row_ptr_[0] = 0;
    uint64_t running = 0;
    for (uint64_t i = 0; i < num_nodes_; ++i) {
        running += degrees[i];
        row_ptr_[i + 1] = running;
    }
    num_edges_ = running;

    // uint32 topology sidecar eligibility — identical opt-in rules as the
    // Direction ctor (MDB_GNN_TOPOLOGY_UINT32 + node/edge ids fit uint32).
    {
        bool want_narrow = false;
        if (const char* e = std::getenv("MDB_GNN_TOPOLOGY_UINT32")) {
            want_narrow = (e[0] == '1' || e[0] == 't' || e[0] == 'T');
        }
        const uint64_t kU32 = uint64_t{1} << 32;
        const bool node_fits = num_nodes_ < kU32;
        const bool edge_fits = !include_edge_ids_ || num_edges_ < kU32;
        id_width_ = (want_narrow && node_fits && edge_fits)
                        ? kTopologySnapshotIdWidthNarrow
                        : kTopologySnapshotIdWidth;
    }

    write_cursor_.assign(static_cast<std::size_t>(num_nodes_), 0);

    col_idx_buf_.reserve(kCoalesceBytes / sizeof(uint64_t));
    if (include_edge_ids_) {
        edge_ids_buf_.reserve(kCoalesceBytes / sizeof(uint64_t));
    }
    if (id_width_ == kTopologySnapshotIdWidthNarrow) {
        col_idx_buf32_.reserve(kCoalesceBytes / sizeof(uint64_t));
        if (include_edge_ids_) {
            edge_ids_buf32_.reserve(kCoalesceBytes / sizeof(uint64_t));
        }
    }

    // Section offsets. Fixed once N, M, and id_width_ are known. ROW_PTR is
    // uint64; COL_IDX / EDGE_IDS use element_size_() (4 or 8).
    col_idx_offset_  = static_cast<uint64_t>(kTopologySnapshotHeaderSize)
                     + sizeof(uint64_t) * (num_nodes_ + 1);
    edge_ids_offset_ = col_idx_offset_
                     + element_size_() * num_edges_;
    const uint64_t file_size =
        col_idx_offset_
        + element_size_() * num_edges_ * (include_edge_ids_ ? 2 : 1);

    out_fd_ = ::open(tmp_path_.c_str(),
                     O_WRONLY | O_CREAT | O_EXCL | O_TRUNC,
                     S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (out_fd_ < 0) {
        int e = errno;
        throw std::runtime_error(
            "TopologySnapshotWriter: cannot create " + tmp_path_.string()
            + " (errno=" + std::to_string(e)
            + (e == EEXIST ? " / another writer already holds this file" : "")
            + ")");
    }

    try {
        if (::ftruncate(out_fd_, static_cast<off_t>(file_size)) != 0) {
            int e = errno;
            throw std::runtime_error(
                "TopologySnapshotWriter: ftruncate failed on " + tmp_path_.string()
                + " to " + std::to_string(file_size) + " bytes"
                + " (errno=" + std::to_string(e) + ")");
        }

        std::array<uint8_t, kTopologySnapshotHeaderSize> zero_header{};
        pwrite_all(zero_header.data(), zero_header.size(), 0);

        pwrite_all(row_ptr_.data(),
                   row_ptr_.size() * sizeof(uint64_t),
                   kTopologySnapshotHeaderSize);
    } catch (...) {
        ::close(out_fd_);
        out_fd_ = -1;
        std::error_code ec;
        std::filesystem::remove(tmp_path_, ec);
        throw;
    }
}

// ---------------------------------------------------------------------------
// Destructor — best-effort cleanup of an abandoned .tmp
// ---------------------------------------------------------------------------

TopologySnapshotWriter::~TopologySnapshotWriter() {
    if (out_fd_ >= 0) {
        ::close(out_fd_);
        out_fd_ = -1;
    }
    if (!finalized_) {
        std::error_code ec;
        std::filesystem::remove(tmp_path_, ec);  // ignore errors
    }
}

// ---------------------------------------------------------------------------
// append_edge
// ---------------------------------------------------------------------------

void TopologySnapshotWriter::append_edge(ObjectId src, ObjectId dst, ObjectId edge_id) {
    if (finalized_) {
        throw std::logic_error(
            "TopologySnapshotWriter: append_edge after finalize");
    }

    uint64_t src_idx = src.id;
    // Stored COL_IDX value. Wide: the raw tagged ObjectId (legacy). Narrow:
    // the tag-stripped ordinal, with the constant type tag captured into the
    // header. capture_tag_ asserts a single tag per section.
    uint64_t dst_idx = dst.id;
    if (id_width_ == kTopologySnapshotIdWidthNarrow) {
        capture_tag_(dst_tag_, static_cast<uint8_t>(dst.id >> 56), "dst");
        dst_idx = dst.id & kTopologySnapshotValueMask;
        if (dst_idx > 0xFFFFFFFFULL) {
            throw std::runtime_error(
                "TopologySnapshotWriter: dst ordinal " + std::to_string(dst_idx)
                + " exceeds uint32 under narrow id_width");
        }
    }

    // Always-on invariant checks. Release-build silent corruption of the
    // CSR body would not be caught by the reader's SHA-256 (that hashes the
    // source .leaf, not the CSR). Wrong sampling output is a thesis-grade
    // correctness bug; the extra branch cost per edge is negligible vs the
    // 8-byte pwrite syscall that follows.
    if (src_idx >= num_nodes_) {
        throw std::runtime_error(
            "TopologySnapshotWriter: src_idx " + std::to_string(src_idx)
            + " out of range (num_nodes=" + std::to_string(num_nodes_) + ")");
    }
    if (src_idx < last_src_idx_) {
        throw std::runtime_error(
            "TopologySnapshotWriter: edges must arrive in src-monotonic order "
            "(src_idx=" + std::to_string(src_idx)
            + ", last_src_idx=" + std::to_string(last_src_idx_) + ")");
    }
    const std::size_t src_slot = static_cast<std::size_t>(src_idx);
    const uint64_t declared_degree =
        row_ptr_[src_slot + 1] - row_ptr_[src_slot];
    if (write_cursor_[src_slot] >= declared_degree) {
        throw std::runtime_error(
            "TopologySnapshotWriter: more edges than declared degree "
            "for src_idx=" + std::to_string(src_idx)
            + " (declared=" + std::to_string(declared_degree) + ")");
    }

    // Index of this edge within COL_IDX: row_ptr[src] + local_cursor.
    // The contract in the header allows partial fills — a source with
    // declared degree > actual appended count leaves a gap whose bytes
    // come from ftruncate's zero-fill. To preserve that, the per-section
    // coalescing buffers detect any gap between the previously-buffered
    // last word and the current edge_index, and flush the buffer
    // (repositioning its logical start) so the pwrite always lands at
    // the correct contiguous offset. In the common path (full fill,
    // src-monotone, which is what the integrated builder produces) this
    // branch never triggers and we stay in the 1 MiB coalescing regime.
    const uint64_t edge_index =
        row_ptr_[static_cast<std::size_t>(src_idx)]
        + write_cursor_[static_cast<std::size_t>(src_idx)];
    last_src_idx_ = src_idx;
    ++write_cursor_[static_cast<std::size_t>(src_idx)];

    // ---- COL_IDX ---------------------------------------------------
    {
        const uint64_t buffered_next =
            col_idx_flushed_words_ + col_idx_buf_.size();
        if (edge_index != buffered_next) {
            // Gap detected. Flush whatever is in the buffer at its
            // current offset, then reposition the logical "flushed"
            // cursor to the new edge_index so subsequent pwrites are
            // offset correctly. The gap words themselves are left
            // zero — ftruncate pre-filled the file.
            flush_col_idx_buffer_();
            col_idx_flushed_words_ = edge_index;
        }
        col_idx_buf_.push_back(dst_idx);
        if (col_idx_buf_.size() * element_size_() >= kCoalesceBytes) {
            flush_col_idx_buffer_();
        }
    }

    // ---- EDGE_IDS --------------------------------------------------
    if (include_edge_ids_) {
        uint64_t eid_store = edge_id.id;
        if (id_width_ == kTopologySnapshotIdWidthNarrow) {
            capture_tag_(edge_tag_, static_cast<uint8_t>(edge_id.id >> 56), "edge");
            eid_store = edge_id.id & kTopologySnapshotValueMask;
            if (eid_store > 0xFFFFFFFFULL) {
                throw std::runtime_error(
                    "TopologySnapshotWriter: edge ordinal "
                    + std::to_string(eid_store)
                    + " exceeds uint32 under narrow id_width");
            }
        }
        const uint64_t buffered_next =
            edge_ids_flushed_words_ + edge_ids_buf_.size();
        if (edge_index != buffered_next) {
            flush_edge_ids_buffer_();
            edge_ids_flushed_words_ = edge_index;
        }
        edge_ids_buf_.push_back(eid_store);
        if (edge_ids_buf_.size() * element_size_() >= kCoalesceBytes) {
            flush_edge_ids_buffer_();
        }
    } else {
        (void)edge_id;  // intentionally unused
    }
}

void TopologySnapshotWriter::append_subrange(
    uint64_t lo_src,
    uint64_t hi_src,
    const std::vector<uint64_t>& dst_buf,
    const std::vector<uint64_t>& edge_ids_buf)
{
    if (finalized_) {
        throw std::logic_error(
            "TopologySnapshotWriter: append_subrange after finalize");
    }
    if (lo_src > hi_src || hi_src > num_nodes_) {
        throw std::runtime_error(
            "TopologySnapshotWriter: append_subrange invalid range ["
            + std::to_string(lo_src) + ", " + std::to_string(hi_src) + ")"
            + " (num_nodes=" + std::to_string(num_nodes_) + ")");
    }
    if (lo_src == hi_src) {
        return;
    }

    const uint64_t base_edge_index =
        row_ptr_[static_cast<std::size_t>(lo_src)];
    const uint64_t end_edge_index =
        row_ptr_[static_cast<std::size_t>(hi_src)];
    const uint64_t expected_count = end_edge_index - base_edge_index;

    if (dst_buf.size() != expected_count) {
        throw std::runtime_error(
            "TopologySnapshotWriter: append_subrange dst_buf.size() ("
            + std::to_string(dst_buf.size())
            + ") != row_ptr[hi]-row_ptr[lo] ("
            + std::to_string(expected_count) + ")");
    }
    if (include_edge_ids_ && edge_ids_buf.size() != expected_count) {
        throw std::runtime_error(
            "TopologySnapshotWriter: append_subrange edge_ids_buf.size() ("
            + std::to_string(edge_ids_buf.size())
            + ") != expected (" + std::to_string(expected_count) + ")");
    }

    if (expected_count == 0) {
        return;
    }

    // Disjoint-region pwrite. Two workers calling this concurrently with
    // non-overlapping [lo_src, hi_src) intervals write to disjoint byte
    // ranges in COL_IDX and EDGE_IDS, so the kernel pwrite serialization
    // is per-write-region — no cross-thread data races on the file content.
    if (id_width_ == kTopologySnapshotIdWidthNarrow) {
        // Narrow: validate + tag-strip + widen-down to uint32 into a local
        // staging vector (local, not the member buffers, so concurrent
        // workers don't share state). capture_tag_ is CAS-safe across workers.
        std::vector<uint32_t> dst32(dst_buf.size());
        for (std::size_t i = 0; i < dst_buf.size(); ++i) {
            capture_tag_(dst_tag_, static_cast<uint8_t>(dst_buf[i] >> 56), "dst");
            const uint64_t v = dst_buf[i] & kTopologySnapshotValueMask;
            if (v > 0xFFFFFFFFULL) {
                throw std::runtime_error(
                    "TopologySnapshotWriter: dst ordinal " + std::to_string(v)
                    + " exceeds uint32 under narrow id_width");
            }
            dst32[i] = static_cast<uint32_t>(v);
        }
        pwrite_all(dst32.data(), dst32.size() * sizeof(uint32_t),
                   col_idx_offset_ + base_edge_index * element_size_());

        if (include_edge_ids_) {
            std::vector<uint32_t> eid32(edge_ids_buf.size());
            for (std::size_t i = 0; i < edge_ids_buf.size(); ++i) {
                capture_tag_(edge_tag_,
                             static_cast<uint8_t>(edge_ids_buf[i] >> 56), "edge");
                const uint64_t v = edge_ids_buf[i] & kTopologySnapshotValueMask;
                if (v > 0xFFFFFFFFULL) {
                    throw std::runtime_error(
                        "TopologySnapshotWriter: edge ordinal "
                        + std::to_string(v)
                        + " exceeds uint32 under narrow id_width");
                }
                eid32[i] = static_cast<uint32_t>(v);
            }
            pwrite_all(eid32.data(), eid32.size() * sizeof(uint32_t),
                       edge_ids_offset_ + base_edge_index * element_size_());
        }
        return;
    }

    pwrite_all(
        dst_buf.data(),
        dst_buf.size() * sizeof(uint64_t),
        col_idx_offset_ + base_edge_index * sizeof(uint64_t));

    if (include_edge_ids_) {
        pwrite_all(
            edge_ids_buf.data(),
            edge_ids_buf.size() * sizeof(uint64_t),
            edge_ids_offset_ + base_edge_index * sizeof(uint64_t));
    }
}

void TopologySnapshotWriter::flush_col_idx_buffer_() {
    if (col_idx_buf_.empty()) {
        return;
    }
    const std::size_t n = col_idx_buf_.size();
    const uint64_t off = col_idx_offset_ + col_idx_flushed_words_ * element_size_();
    if (id_width_ == kTopologySnapshotIdWidthNarrow) {
        // Words in col_idx_buf_ were already tag-stripped + range-checked in
        // append_edge; narrow them to uint32 for the on-disk layout.
        col_idx_buf32_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            col_idx_buf32_[i] = static_cast<uint32_t>(col_idx_buf_[i]);
        }
        pwrite_all(col_idx_buf32_.data(), n * sizeof(uint32_t), off);
    } else {
        pwrite_all(col_idx_buf_.data(), n * sizeof(uint64_t), off);
    }
    col_idx_flushed_words_ += n;
    col_idx_buf_.clear();
}

void TopologySnapshotWriter::flush_edge_ids_buffer_() {
    if (edge_ids_buf_.empty()) {
        return;
    }
    const std::size_t n = edge_ids_buf_.size();
    const uint64_t off = edge_ids_offset_ + edge_ids_flushed_words_ * element_size_();
    if (id_width_ == kTopologySnapshotIdWidthNarrow) {
        edge_ids_buf32_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            edge_ids_buf32_[i] = static_cast<uint32_t>(edge_ids_buf_[i]);
        }
        pwrite_all(edge_ids_buf32_.data(), n * sizeof(uint32_t), off);
    } else {
        pwrite_all(edge_ids_buf_.data(), n * sizeof(uint64_t), off);
    }
    edge_ids_flushed_words_ += n;
    edge_ids_buf_.clear();
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void TopologySnapshotWriter::finalize() {
    if (finalized_) {
        throw std::logic_error("TopologySnapshotWriter: finalize called twice");
    }
    if (out_fd_ < 0) {
        throw std::logic_error(
            "TopologySnapshotWriter: finalize on closed writer");
    }

    // Flush any residual COL_IDX / EDGE_IDS words still in the 1 MiB
    // coalescing buffers. Any un-flushed tail is <1 MiB, so this is a
    // single pwrite per section worst case.
    flush_col_idx_buffer_();
    flush_edge_ids_buffer_();

    // Hash the source(s). Symmetric mode chains the explicit source list into
    // one combined digest; directional mode hashes the single direction-derived
    // .leaf (legacy path).
    std::array<uint8_t, 32> source_hash{};
    if (symmetric_format_ || !source_leaf_paths_.empty()) {
        if (num_edges_ != 0) {
            for (const auto& p : source_leaf_paths_) {
                if (!std::filesystem::exists(p)) {
                    throw std::runtime_error(
                        "TopologySnapshotWriter: source .leaf missing: " + p.string());
                }
                if (std::filesystem::file_size(p) == 0) {
                    throw std::runtime_error(
                        "TopologySnapshotWriter: source .leaf is empty but graph "
                        "has " + std::to_string(num_edges_) + " edges: " + p.string());
                }
            }
        }
        bool any_present = false;
        for (const auto& p : source_leaf_paths_) {
            if (std::filesystem::exists(p)) { any_present = true; break; }
        }
        if (any_present) {
            source_hash = sha256_of_files_chained(source_leaf_paths_);
        }
    } else if (std::filesystem::exists(source_leaf_path_)) {
        // Guard against an empty source file paired with a non-empty graph:
        // hashing an empty file yields a well-defined digest, but that
        // silently accepts an obviously corrupt projection layout. Fail
        // fast so the caller can retry / rebuild the source.
        if (num_edges_ != 0
            && std::filesystem::file_size(source_leaf_path_) == 0) {
            throw std::runtime_error(
                "TopologySnapshotWriter: source .leaf is empty but graph has "
                + std::to_string(num_edges_) + " edges: "
                + source_leaf_path_.string());
        }
        source_hash = sha256_of_file(source_leaf_path_);
    } else {
        // Source .leaf absent is allowed only when the projection has no
        // edges at all (e.g. N=0). For non-empty graphs this is almost
        // certainly a configuration error — fail fast.
        if (num_edges_ != 0) {
            throw std::runtime_error(
                "TopologySnapshotWriter: source .leaf missing: "
                + source_leaf_path_.string());
        }
    }

    // Rewrite header at offset 0 with real values (§5.3 step 5). Symmetric mode
    // uses the "TOPOSYM1" magic/version so directional readers never mis-parse it.
    TopologySnapshotHeader header = symmetric_format_
        ? make_default_topology_snapshot_sym_header()
        : make_default_topology_snapshot_header();
    header.id_width = id_width_;
    if (include_edge_ids_) {
        header.flags |= TopologySnapshotFlags::kHasEdgeIds;
    }
    // Narrow uint32 layout: persist the per-section ObjectId type tag the reader
    // re-applies when reconstructing full 64-bit ObjectIds from the stored
    // tag-stripped uint32 ordinals (tag << 56 | ordinal). An empty section
    // never captured a tag — 0 is harmless (no values to reconstruct). Stays 0
    // for the wide uint64 layout where full ObjectIds are stored verbatim.
    if (id_width_ == kTopologySnapshotIdWidthNarrow) {
        const uint16_t dt = dst_tag_.load(std::memory_order_relaxed);
        header.dst_type_tag = (dt == kTagUnset) ? 0 : static_cast<uint8_t>(dt);
        if (include_edge_ids_) {
            const uint16_t et = edge_tag_.load(std::memory_order_relaxed);
            header.edge_type_tag = (et == kTagUnset) ? 0 : static_cast<uint8_t>(et);
        }
    }
    header.num_nodes = num_nodes_;
    header.num_edges = num_edges_;
    std::memcpy(header.source_sha256, source_hash.data(), 32);

    uint8_t hdr_buf[kTopologySnapshotHeaderSize];
    serialize_topology_snapshot_header(header, hdr_buf);
    pwrite_all(hdr_buf, kTopologySnapshotHeaderSize, 0);

    // fsync the file data + rename + fsync parent dir (§5.3 steps 6-8).
    if (::fsync(out_fd_) != 0) {
        int e = errno;
        throw std::runtime_error(
            "TopologySnapshotWriter: fsync failed on " + tmp_path_.string()
            + " (errno=" + std::to_string(e) + ")");
    }
    if (::close(out_fd_) != 0) {
        int e = errno;
        out_fd_ = -1;
        throw std::runtime_error(
            "TopologySnapshotWriter: close failed on " + tmp_path_.string()
            + " (errno=" + std::to_string(e) + ")");
    }
    out_fd_ = -1;

    // Atomic rename. std::filesystem::rename is POSIX ::rename on Linux,
    // which is atomic on the same filesystem. We rely on the .tmp living in
    // the same directory as the final (both in projection_dir_).
    std::error_code ec;
    std::filesystem::rename(tmp_path_, final_path_, ec);
    if (ec) {
        throw std::runtime_error(
            "TopologySnapshotWriter: rename "
            + tmp_path_.string() + " -> " + final_path_.string()
            + " failed: " + ec.message());
    }

    fsync_directory(projection_dir_);

    // Record final size for bytes_written(). ROW_PTR uint64; COL_IDX / EDGE_IDS
    // at element_size_() (4 or 8).
    const uint64_t expected =
        static_cast<uint64_t>(kTopologySnapshotHeaderSize)
        + sizeof(uint64_t) * (num_nodes_ + 1)
        + element_size_() * num_edges_ * (include_edge_ids_ ? 2 : 1);
    bytes_written_ = expected;

    finalized_ = true;
}

// ---------------------------------------------------------------------------
// capture_tag_ — first-writer-wins capture + consistency assert for the
// narrow uint32 layout's per-section ObjectId type tag
// ---------------------------------------------------------------------------

void TopologySnapshotWriter::capture_tag_(std::atomic<uint16_t>& slot,
                                          uint8_t tag, const char* what) {
    uint16_t expected = kTagUnset;
    // First narrow append for this section wins the CAS and stores the tag.
    if (slot.compare_exchange_strong(expected,
                                     static_cast<uint16_t>(tag),
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
        return;
    }
    // Already set (by us on a prior call or by another worker). The narrow
    // layout carries ONE type tag per section in the header, so a mismatch
    // means the section mixes node/edge types — not representable. Fail loud
    // rather than silently corrupt the reconstructed ObjectIds.
    if (static_cast<uint8_t>(expected) != tag) {
        throw std::runtime_error(
            std::string("TopologySnapshotWriter: inconsistent ") + what
            + " ObjectId type tag under narrow id_width (saw "
            + std::to_string(static_cast<unsigned>(tag)) + ", section already "
            + std::to_string(static_cast<unsigned>(expected))
            + "); a narrow section must share one type tag");
    }
}

// ---------------------------------------------------------------------------
// Low-level write helper — handle short writes (signal-driven EINTR etc.)
// ---------------------------------------------------------------------------

void TopologySnapshotWriter::pwrite_all(const void* data, std::size_t len,
                                        std::size_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    std::size_t remaining = len;
    std::size_t off = offset;
    while (remaining > 0) {
        ssize_t n = ::pwrite(out_fd_, p, remaining, static_cast<off_t>(off));
        if (n < 0) {
            if (errno == EINTR) continue;
            int e = errno;
            throw std::runtime_error(
                "TopologySnapshotWriter: pwrite failed on " + tmp_path_.string()
                + " (errno=" + std::to_string(e) + ")");
        }
        if (n == 0) {
            throw std::runtime_error(
                "TopologySnapshotWriter: pwrite returned 0 on "
                + tmp_path_.string());
        }
        p         += n;
        remaining -= static_cast<std::size_t>(n);
        off       += static_cast<std::size_t>(n);
    }
}

}  // namespace GQL::Projection
