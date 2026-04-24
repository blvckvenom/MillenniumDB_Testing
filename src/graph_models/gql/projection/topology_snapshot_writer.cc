#include "graph_models/gql/projection/topology_snapshot_writer.h"

#include <fcntl.h>        // open, O_WRONLY, O_CREAT, O_EXCL, O_TRUNC
#include <sys/stat.h>     // mode constants
#include <unistd.h>       // write, pwrite, fsync, close

#include <cassert>
#include <cerrno>
#include <cstdio>
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

    write_cursor_.assign(static_cast<std::size_t>(num_nodes_), 0);

    // Reserve 1 MiB of staging space for each per-section coalescing
    // buffer. Append_edge() pushes words here instead of issuing a
    // pwrite per call; the buffer flushes when full and once at
    // finalize(). Keeps peak RAM at ~2 MiB regardless of graph size.
    col_idx_buf_.reserve(kCoalesceBytes / sizeof(uint64_t));
    if (include_edge_ids_) {
        edge_ids_buf_.reserve(kCoalesceBytes / sizeof(uint64_t));
    }

    // Section offsets. Fixed once N and M are known.
    col_idx_offset_  = static_cast<uint64_t>(kTopologySnapshotHeaderSize)
                     + sizeof(uint64_t) * (num_nodes_ + 1);
    edge_ids_offset_ = col_idx_offset_
                     + sizeof(uint64_t) * num_edges_;
    const uint64_t file_size =
        col_idx_offset_
        + sizeof(uint64_t) * num_edges_ * (include_edge_ids_ ? 2 : 1);

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
    uint64_t dst_idx = dst.id;

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
        if (col_idx_buf_.size() * sizeof(uint64_t) >= kCoalesceBytes) {
            flush_col_idx_buffer_();
        }
    }

    // ---- EDGE_IDS --------------------------------------------------
    if (include_edge_ids_) {
        const uint64_t buffered_next =
            edge_ids_flushed_words_ + edge_ids_buf_.size();
        if (edge_index != buffered_next) {
            flush_edge_ids_buffer_();
            edge_ids_flushed_words_ = edge_index;
        }
        edge_ids_buf_.push_back(edge_id.id);
        if (edge_ids_buf_.size() * sizeof(uint64_t) >= kCoalesceBytes) {
            flush_edge_ids_buffer_();
        }
    } else {
        (void)edge_id;  // intentionally unused
    }
}

void TopologySnapshotWriter::flush_col_idx_buffer_() {
    if (col_idx_buf_.empty()) {
        return;
    }
    const std::size_t n = col_idx_buf_.size();
    pwrite_all(
        col_idx_buf_.data(),
        n * sizeof(uint64_t),
        col_idx_offset_ + col_idx_flushed_words_ * sizeof(uint64_t));
    col_idx_flushed_words_ += n;
    col_idx_buf_.clear();
}

void TopologySnapshotWriter::flush_edge_ids_buffer_() {
    if (edge_ids_buf_.empty()) {
        return;
    }
    const std::size_t n = edge_ids_buf_.size();
    pwrite_all(
        edge_ids_buf_.data(),
        n * sizeof(uint64_t),
        edge_ids_offset_ + edge_ids_flushed_words_ * sizeof(uint64_t));
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

    // Hash the source .leaf end-to-end (separate pass over the file).
    std::array<uint8_t, 32> source_hash{};
    if (std::filesystem::exists(source_leaf_path_)) {
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

    // Rewrite header at offset 0 with real values (§5.3 step 5).
    TopologySnapshotHeader header = make_default_topology_snapshot_header();
    if (include_edge_ids_) {
        header.flags |= TopologySnapshotFlags::kHasEdgeIds;
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

    // Record final size for bytes_written().
    const uint64_t expected =
        static_cast<uint64_t>(kTopologySnapshotHeaderSize)
        + sizeof(uint64_t) * (num_nodes_ + 1)
        + sizeof(uint64_t) * num_edges_ * (include_edge_ids_ ? 2 : 1);
    bytes_written_ = expected;

    finalized_ = true;
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
