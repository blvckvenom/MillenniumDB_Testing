#pragma once

// TopologySnapshotWriter — streams sorted edges into a CSR sidecar file.
//
// Writes `topology_fwd.csr` (FORWARD, hashes `from_to_edge.leaf`) or
// `topology_rev.csr` (REVERSE, hashes `to_from_edge.leaf`) next to the
// projection's existing B+Tree files. The file layout is the shared
// contract defined in `topology_snapshot.h`.
//
// Operating contract:
//   1. Caller precomputes a degree histogram of size `num_nodes` (a single
//      BPT<3> scan) and hands it to the constructor. `row_ptr` is computed
//      immediately inside the ctor as a prefix sum, so the writer becomes
//      single-pass from the BPT point of view: `append_edge()` is called
//      with edges in source-monotonic order.
//   2. `append_edge()` appends one (dst [, edge_id]) pair to the growing
//      COL_IDX (and, when `include_edge_ids == true`, EDGE_IDS) buffer.
//      Monotonicity of the source is enforced via a debug assert — this
//      matches the natural order of the source `.leaf` scan.
//   3. `finalize()` writes ROW_PTR + COL_IDX [+ EDGE_IDS], computes the
//      SHA-256 of the source `.leaf` in a separate streaming pass, rewrites
//      the header at offset 0 with the real values, then commits the file
//      atomically: fsync → rename `.tmp` → `.csr` → fsync parent dir.
//
// Error model: every I/O failure throws `std::runtime_error` with a
// filesystem-qualified message. The `.tmp` file is best-effort removed on
// destruction if `finalize()` did not run.
//
// Spec reference: docs/superpowers/specs/2026-04-25-topology-snapshot-design.md
//                 §3.7, §4.3, §5.1, §5.3

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "graph_models/object_id.h"
#include "graph_models/gql/projection/topology_snapshot.h"

namespace GQL::Projection {

class TopologySnapshotWriter {
public:
    /// Which direction's `.leaf` source is being mirrored. Controls both the
    /// output filename and the source file whose SHA-256 is stored in the
    /// header.
    enum class Direction {
        FORWARD,  // writes `topology_fwd.csr`, hashes `from_to_edge.leaf`
        REVERSE   // writes `topology_rev.csr`, hashes `to_from_edge.leaf`
    };

    /// @param projection_dir  Directory that holds `from_to_edge.leaf` /
    ///                        `to_from_edge.leaf`. The CSR sidecar is written
    ///                        alongside them.
    /// @param dir             FORWARD or REVERSE (see enum comment above).
    /// @param num_nodes       `N` as reported by the projection catalog. Used
    ///                        to size ROW_PTR (length = N+1).
    /// @param degrees         Per-node out-degree histogram of length `N`.
    ///                        Consumed to build the prefix-sum ROW_PTR stored
    ///                        as a private member; the parameter itself is
    ///                        not retained. Sum of degrees is `M`.
    /// @param include_edge_ids Emit the EDGE_IDS section and set the
    ///                        has_edge_ids flag bit. The writer still accepts
    ///                        an `edge_id` argument to `append_edge()` when
    ///                        this is false, but silently ignores it.
    ///
    /// The constructor opens `<projection_dir>/<basename>.csr.tmp` with
    /// O_EXCL — concurrent writers targeting the same file are rejected
    /// with `std::runtime_error`. It writes a 64-byte zero placeholder
    /// header and the full ROW_PTR section, leaving the file positioned at
    /// the start of COL_IDX.
    TopologySnapshotWriter(const std::filesystem::path& projection_dir,
                           Direction                    dir,
                           uint64_t                     num_nodes,
                           std::vector<uint64_t>        degrees,
                           bool                         include_edge_ids);

    /// Non-copyable, non-movable: owns a POSIX file descriptor and the
    /// `.tmp` on disk.
    TopologySnapshotWriter(const TopologySnapshotWriter&) = delete;
    TopologySnapshotWriter& operator=(const TopologySnapshotWriter&) = delete;
    TopologySnapshotWriter(TopologySnapshotWriter&&) = delete;
    TopologySnapshotWriter& operator=(TopologySnapshotWriter&&) = delete;

    /// On destruction, if `finalize()` never succeeded, remove the `.tmp`
    /// so a retried builder does not hit `O_EXCL`.
    ~TopologySnapshotWriter();

    /// Append one edge. Edges MUST arrive in `src`-monotonic order matching
    /// the degree histogram handed to the constructor; the `k`-th call with
    /// source `s` lands at index `row_ptr[s] + k` in COL_IDX.
    ///
    /// `edge_id` is ignored when `include_edge_ids == false`; callers can
    /// pass `ObjectId()` in that case.
    ///
    /// On (debug) violation of `src` monotonicity or when more edges are
    /// appended for a given source than its declared degree, the assertion
    /// fires. In release builds the write still goes through; correctness
    /// is the caller's responsibility — this matches the contract of the
    /// enclosing B+Tree scan driving the writer.
    void append_edge(ObjectId src, ObjectId dst, ObjectId edge_id);

    /// Parallel-append entry point.
    ///
    /// Append a contiguous block of edges that all belong to the half-open
    /// src range `[lo_src, hi_src)`. The provided vectors hold the dst
    /// (and optionally edge_id) words for those edges, ordered exactly as
    /// the legacy `append_edge` call sequence would have produced them
    /// when scanning the same src range — i.e. ascending src; for a fixed
    /// src, the original .leaf record order; total length equals
    /// `row_ptr[hi_src] - row_ptr[lo_src]`.
    ///
    /// pwrite is thread-safe and the writer's per-section offsets are
    /// computed from `row_ptr` (an immutable construct-time prefix sum),
    /// so calling this method concurrently from different workers — each
    /// with a disjoint src range — is safe: every worker's bytes land in
    /// a disjoint COL_IDX / EDGE_IDS slice. The writer's monotonicity
    /// cursors (`last_src_idx_`, `write_cursor_`) are NOT updated by this
    /// method, so it MUST NOT be interleaved with `append_edge()` for the
    /// same writer instance.
    ///
    /// `edge_ids_buf` is ignored when `include_edge_ids == false`. When
    /// included, its size must equal `dst_buf.size()`.
    void append_subrange(uint64_t lo_src,
                         uint64_t hi_src,
                         const std::vector<uint64_t>& dst_buf,
                         const std::vector<uint64_t>& edge_ids_buf);

    /// Read-only view of the row_ptr prefix sum. Available immediately
    /// after construction; used by parallel callers to compute their
    /// per-worker output buffer sizes ahead of the parallel_for.
    const std::vector<uint64_t>& row_ptr() const noexcept { return row_ptr_; }

    /// Commit the file. Safe to call at most once. After this returns, the
    /// `.tmp` has been renamed to the final name and the parent directory
    /// has been fsync'd.
    void finalize();

    /// Total bytes in the finalized file (header + sections). Valid after
    /// `finalize()`; returns 0 before.
    uint64_t bytes_written() const noexcept { return bytes_written_; }

    /// Absolute path of the final (post-rename) `.csr` file. Fixed at
    /// construction; callers can log / YIELD it directly.
    const std::filesystem::path& output_path() const noexcept { return final_path_; }

private:
    // Derived at construction.
    std::filesystem::path projection_dir_;
    Direction             direction_;
    uint64_t              num_nodes_;
    uint64_t              num_edges_;
    bool                  include_edge_ids_;

    std::filesystem::path source_leaf_path_;  // for SHA-256
    std::filesystem::path final_path_;        // `topology_{fwd,rev}.csr`
    std::filesystem::path tmp_path_;          // `.csr.tmp`

    // COL_IDX / EDGE_IDS element width in bytes. 8 (default, full tagged
    // ObjectId, byte-identical to the legacy layout) or 4 (the 8-bit ObjectId
    // type tag is stripped losslessly and reconstructed on read, ~halving
    // topology file size). Width 4 is selected at construction only when both
    // num_nodes and num_edges fit in uint32 AND the `MDB_GNN_TOPOLOGY_UINT32`
    // env opt-in is set. ROW_PTR is always uint64 regardless of this.
    // See `element_size_()`.
    uint8_t id_width_ = kTopologySnapshotIdWidth;

    // Sentinel for "type tag not yet captured" (valid tags are 0..255).
    static constexpr uint16_t kTagUnset = 0x100;
    // Per-section ObjectId type tag, captured once on the first narrow append
    // and asserted consistent thereafter (the narrow layout requires a single
    // tag per section). Atomic so the parallel `append_subrange` path is
    // race-free. Unused (stays kTagUnset) when id_width_ == 8.
    std::atomic<uint16_t> dst_tag_{kTagUnset};
    std::atomic<uint16_t> edge_tag_{kTagUnset};

    // ROW_PTR[0..N], prefix-sum of `degrees`. Kept in RAM for the cursor
    // tracking in `append_edge()`; sized (N+1) × 8 B.
    std::vector<uint64_t> row_ptr_;

    // Per-source write cursor: next COL_IDX index to write for source s is
    // `row_ptr_[s] + write_cursor_[s]`. Used only for the monotonicity
    // debug assert, not for file positioning (which is purely sequential).
    std::vector<uint64_t> write_cursor_;
    uint64_t              last_src_idx_ = 0;

    // Byte offsets of the COL_IDX and EDGE_IDS sections within the output
    // file. Both fixed at construction from (num_nodes_, num_edges_) so
    // append_edge() can pwrite directly to each section's cursor without
    // buffering.
    uint64_t col_idx_offset_  = 0;
    uint64_t edge_ids_offset_ = 0;

    int           out_fd_      = -1;  // POSIX fd owning `.tmp` with O_EXCL semantics
    bool          finalized_   = false;
    uint64_t      bytes_written_ = 0;

    // Per-section coalescing write buffers.
    // Without these, append_edge() would issue one pwrite per word — on
    // ogbn-arxiv (~1.2 M edges × 2 pwrites / edge × 2 directions) the
    // syscall cost alone was ~4.4 s, dominating the integrated path.
    //
    // The buffers flush when they reach kCoalesceBytes or at finalize().
    // COL_IDX / EDGE_IDS are always filled at a contiguous section-local
    // cursor (edge order is monotone in src AND monotone in row_ptr[src]
    // + cursor[src], so after all N sources are visited the buffer is a
    // dense prefix of the section). Cursor tracking is therefore a
    // single uint64_t per section — no sparse writes are possible under
    // the monotonicity contract.
    static constexpr std::size_t kCoalesceBytes = 1 << 20;  // 1 MiB
    std::vector<uint64_t> col_idx_buf_;   // pending COL_IDX words (already masked when narrow)
    std::vector<uint64_t> edge_ids_buf_;  // pending EDGE_IDS words (may be empty)
    uint64_t              col_idx_flushed_words_  = 0;
    uint64_t              edge_ids_flushed_words_ = 0;

    // Reused uint32 staging for the narrow (id_width_==4) flush: the
    // accumulation buffers stay uint64 (holding masked values < 2^32), and
    // each flush narrows into these before the pwrite. Empty when id_width_==8.
    std::vector<uint32_t> col_idx_buf32_;
    std::vector<uint32_t> edge_ids_buf32_;

    void flush_col_idx_buffer_();
    void flush_edge_ids_buffer_();

    // COL_IDX / EDGE_IDS on-disk element width in bytes (4 or 8). ROW_PTR is
    // always uint64 and never uses this.
    std::size_t element_size_() const noexcept {
        return static_cast<std::size_t>(id_width_);
    }

    // Capture (first call) or validate (subsequent) the per-section ObjectId
    // type tag for the narrow layout. Thread-safe via CAS on `slot`. Throws
    // `std::runtime_error` if a section mixes more than one type tag. `what`
    // names the section for the error message ("dst" / "edge").
    void capture_tag_(std::atomic<uint16_t>& slot, uint8_t tag, const char* what);

    // Write `len` bytes at file offset `offset`, handling short writes and
    // EINTR. All writes go via pwrite so the file-position cursor never
    // matters; append_edge() can freely interleave COL_IDX and EDGE_IDS.
    void pwrite_all(const void* data, std::size_t len, std::size_t offset);
};

}  // namespace GQL::Projection
