#pragma once

// TopologySnapshotWriter — streams sorted edges into a CSR sidecar file.
//
// Writes `topology_fwd.csr` (FORWARD, hashes `from_to_edge.leaf`) or
// `topology_rev.csr` (REVERSE, hashes `to_from_edge.leaf`) next to the
// projection's existing B+Tree files. The file layout is the shared
// contract defined in `topology_snapshot.h` (T4.3).
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

    // Per-section coalescing write buffers (T4.18 perf fix).
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
    std::vector<uint64_t> col_idx_buf_;   // pending COL_IDX words
    std::vector<uint64_t> edge_ids_buf_;  // pending EDGE_IDS words (may be empty)
    uint64_t              col_idx_flushed_words_  = 0;
    uint64_t              edge_ids_flushed_words_ = 0;

    void flush_col_idx_buffer_();
    void flush_edge_ids_buffer_();

    // Write `len` bytes at file offset `offset`, handling short writes and
    // EINTR. All writes go via pwrite so the file-position cursor never
    // matters; append_edge() can freely interleave COL_IDX and EDGE_IDS.
    void pwrite_all(const void* data, std::size_t len, std::size_t offset);
};

}  // namespace GQL::Projection
