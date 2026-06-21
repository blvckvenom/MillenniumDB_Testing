#pragma once

// TopologySnapshotReader — mmap-backed O(1) neighbor-slice accessor.
//
// Opens `topology_fwd.csr` (FORWARD) or `topology_rev.csr` (REVERSE) from a
// projection directory, validates the 64-byte header plus the structural
// invariants defined in §5.2 of the spec, and exposes O(1) slices into the
// COL_IDX (and optional EDGE_IDS) sections via `ConstU64Span` — a minimal
// `std::span<const uint64_t>`-shaped view. The spec writes `std::span`
// (§4.3) but the project targets C++17; `ConstU64Span` is the smallest
// possible substitute that keeps the reader's public surface unchanged
// when we later upgrade to C++20.
//
// Fallback-first architecture (§3.4): missing / malformed / stale sidecars
// are NOT errors. `open()` always succeeds (modulo out-of-memory during its
// own construction) and returns a reader whose `has_data() == false`. The
// caller — today `TopologyAccessor::Impl`, tomorrow anything that
// wants the fast path — is expected to branch on `has_data()` and fall back
// to the existing B+Tree path when the sidecar is absent. A single log line
// per failure mode is emitted on stderr for debuggability; there is no
// throw / no return-code on open.
//
// Staleness (SHA-256 of the source `.leaf`) is validated separately by
// `verify_source_sha256()`. `open()` runs the verification
// after structural validation and collapses a mismatch into the same
// has_data()==false fallback path used for the "sidecar absent" case.
//
// Spec reference: docs/superpowers/specs/2026-04-25-topology-snapshot-design.md
//                 §3.4 (fallback-first arch), §4.3 (C++ surface),
//                 §5.1 (on-disk layout), §5.2 (validation checklist).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "graph_models/gql/projection/topology_snapshot.h"

namespace GQL::Projection {

// Minimal `std::span<const uint64_t>`-shaped view over a contiguous range.
// Lives here (not a separate header) because it is only used by the reader
// and we don't want to leak a project-wide `Span` type through a public
// include. Interface matches the subset of `std::span` the reader actually
// needs: size, data, operator[], begin/end for range-based for.
class ConstU64Span {
public:
    constexpr ConstU64Span() noexcept = default;
    constexpr ConstU64Span(const uint64_t* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    constexpr const uint64_t* data()  const noexcept { return data_; }
    constexpr std::size_t     size()  const noexcept { return size_; }
    constexpr bool            empty() const noexcept { return size_ == 0; }

    constexpr const uint64_t& operator[](std::size_t i) const noexcept {
        return data_[i];
    }

    constexpr const uint64_t* begin() const noexcept { return data_; }
    constexpr const uint64_t* end()   const noexcept { return data_ + size_; }

private:
    const uint64_t* data_ = nullptr;
    std::size_t     size_ = 0;
};

class TopologySnapshotReader {
public:
    /// Mirrors `TopologySnapshotWriter::Direction` so callers can share a
    /// single enum across write + read sites. Defined as a separate type in
    /// this header so the reader does not pull in the writer's translation
    /// unit (which brings OpenSSL).
    enum class Direction {
        FORWARD,  // reads `topology_fwd.csr`
        REVERSE   // reads `topology_rev.csr`
    };

    /// Factory. Returns a reader regardless of whether the sidecar exists,
    /// is readable, or passes validation. Inspect `has_data()` to discover
    /// whether the fast path is available for this projection+direction.
    ///
    /// Diagnostic: on every failure mode other than "file absent", one line
    /// is written to `std::cerr` identifying the file and specific reason.
    /// Absent-file is silent — that's the normal "sidecar not built" case.
    static TopologySnapshotReader open(const std::filesystem::path& projection_dir,
                                       Direction                    dir);

    /// Move-only — the reader owns an mmap region + file descriptor.
    TopologySnapshotReader(TopologySnapshotReader&&) noexcept;
    TopologySnapshotReader& operator=(TopologySnapshotReader&&) noexcept;
    TopologySnapshotReader(const TopologySnapshotReader&)            = delete;
    TopologySnapshotReader& operator=(const TopologySnapshotReader&) = delete;
    ~TopologySnapshotReader();

    /// True iff the file opened, validated, and mmap'd successfully. When
    /// false, every other accessor returns a trivially-empty / zero value
    /// (except `neighbors()` / `edge_ids()` which throw — see their
    /// contracts). Callers are expected to pre-check this before slicing.
    bool     has_data()      const noexcept { return has_data_; }
    uint64_t num_nodes()     const noexcept { return has_data_ ? header_.num_nodes : 0; }
    uint64_t num_edges()     const noexcept { return has_data_ ? header_.num_edges : 0; }
    bool     has_edge_ids()  const noexcept {
        return has_data_
            && (header_.flags & TopologySnapshotFlags::kHasEdgeIds) != 0;
    }

    /// On-disk COL_IDX / EDGE_IDS element width in bytes: 8 (full tagged
    /// ObjectId, default) or 4 (compact uint32 sidecar variant, which strips
    /// the 8-bit ObjectId type tag and stores only the 56-bit payload ordinal,
    /// halving sidecar size; the type tag is reconstructed on read via
    /// `dst_type_tag()` / `edge_type_tag()`). Returns the default width when
    /// `has_data() == false`. Callers that need a single width-agnostic code
    /// path should use `copy_neighbors()` / `degree()` rather than branching
    /// on this.
    uint8_t  id_width() const noexcept {
        return has_data_ ? header_.id_width : kTopologySnapshotIdWidth;
    }

    /// ObjectId type tag re-applied to narrow (id_width==4) COL_IDX values to
    /// reconstruct the full tagged ObjectId. 0 when id_width==8 (values are
    /// already full ObjectIds on disk).
    uint8_t  dst_type_tag()  const noexcept { return header_.dst_type_tag; }
    /// ObjectId type tag re-applied to narrow (id_width==4) EDGE_IDS values.
    uint8_t  edge_type_tag() const noexcept { return header_.edge_type_tag; }

    /// O(1) neighbor count of `node_idx` — `ROW_PTR[node_idx+1] - ROW_PTR[node_idx]`.
    /// Width-agnostic (ROW_PTR is uint64 regardless of `id_width`), so this is
    /// the correct way to get a degree without materialising neighbors. Same
    /// out-of-range / no-data hardening as `neighbors()`.
    uint64_t degree(uint64_t node_idx) const;

    /// O(1) slice into the mmap'd COL_IDX section, as full tagged ObjectIds.
    /// Precondition: `has_data() == true`, `id_width() == 8`, and
    /// `node_idx < num_nodes()`.
    /// Bounds violation → throws `std::out_of_range` in every build config
    /// (same hardening discipline as `TopologySnapshotWriter::append_edge`'s
    /// always-on invariant checks — a silent out-of-range mmap read is a
    /// segfault-or-garbage-data waiting to happen at the call site).
    /// Calling while `has_data() == false` is also rejected with
    /// `std::out_of_range` rather than returning an empty span, so misuse
    /// surfaces at the caller rather than being papered over.
    /// When `id_width() == 4` the on-disk values are uint32, so a uint64 span
    /// would be a wrong reinterpret-cast: this throws `TopologySnapshotFormatError`
    /// directing the caller to `copy_neighbors()` (or the raw `col_idx32_row()`).
    ConstU64Span neighbors(uint64_t node_idx) const;

    /// O(1) slice into the mmap'd EDGE_IDS section. Returns an empty span
    /// when `has_edge_ids() == false` (but `has_data()` still constrains
    /// access — see `neighbors()` contract). Same out-of-range rules, and
    /// the same `id_width()==4` throw → use `copy_edge_ids()`.
    ConstU64Span edge_ids(uint64_t node_idx) const;

    /// Width-agnostic copy accessor: appends the neighbors of `node_idx` to
    /// `out` as full tagged ObjectIds, for BOTH id widths. For id_width==8
    /// this is a memcpy of the stored uint64s; for id_width==4 each stored
    /// uint32 is widened and OR'd with `dst_type_tag()` shifted into the top
    /// byte, reproducing the exact tagged ObjectId the uint64 layout would
    /// have stored (the losslessness contract — see spec §"Why lossless").
    /// `out` is appended to (not cleared); the caller may `reserve` via
    /// `degree()`. Same out-of-range / no-data hardening as `neighbors()`.
    void copy_neighbors(uint64_t node_idx, std::vector<uint64_t>& out) const;

    /// Width-agnostic copy accessor for EDGE_IDS — see `copy_neighbors()`.
    /// Appends nothing when `has_edge_ids() == false`.
    void copy_edge_ids(uint64_t node_idx, std::vector<uint64_t>& out) const;

    /// Raw pointer into the narrow (id_width==4) COL_IDX section at
    /// `node_idx`'s first neighbor. Returns nullptr unless `has_data()` and
    /// `id_width() == 4`. Length is `degree(node_idx)`. Values are the
    /// tag-stripped uint32 ordinals; the caller reconstructs full ObjectIds
    /// via `dst_type_tag()`. This zero-copy path exists for the four-level
    /// store's hot tier-3 dispatch, which must avoid a per-lookup allocation
    /// over the papers100M cold tail. Out-of-range `node_idx` throws.
    const uint32_t* col_idx32_row(uint64_t node_idx) const;
    /// Raw pointer into the narrow EDGE_IDS section — see `col_idx32_row()`.
    /// Returns nullptr unless `has_data()`, `id_width()==4`, and `has_edge_ids()`.
    const uint32_t* edge_ids32_row(uint64_t node_idx) const;

    /// Base pointer of the full ROW_PTR[num_nodes()+1] section (always uint64).
    /// nullptr when `has_data() == false`. This whole-section view exists so the
    /// dynamic GPU sampling path can pin the entire global CSR (ROW_PTR + the
    /// narrow COL_IDX) as one device-visible substrate without a per-node walk.
    const uint64_t* row_ptr_base() const noexcept { return row_ptr_; }
    /// Base pointer of the full narrow (id_width==4) COL_IDX[num_edges()] section.
    /// nullptr unless `has_data()` and `id_width() == 4` (the uint32 layout).
    /// Values are tag-stripped uint32 ordinals; reconstruct via `dst_type_tag()`.
    const uint32_t* col_idx32_base() const noexcept { return col_idx32_; }

    /// Re-advise the kernel about the access pattern over the mmap'd file.
    /// `open()` sets MADV_RANDOM (correct for the seed-driven runtime sampler).
    /// A forward, offset-monotone scan (the four-level populate) should flip to
    /// MADV_SEQUENTIAL for the scan and restore MADV_RANDOM after — that gives
    /// the kernel aggressive readahead AND frees pages behind the read pointer
    /// (lower peak cache than RANDOM). Best-effort: no-op when `has_data()` is
    /// false; madvise failure is silently ignored (it only affects perf).
    /// `sequential=true` → MADV_SEQUENTIAL; `false` → MADV_RANDOM.
    void advise_access(bool sequential) const noexcept;

    /// Asynchronously prefetch the COL_IDX bytes of rows [start_row, end_row)
    /// via `madvise(MADV_WILLNEED)`. Used by the ascending populate scan to keep
    /// a window of the sidecar in flight AHEAD of the cursor — raising the NVMe
    /// queue depth above the default readahead window that MADV_SEQUENTIAL alone
    /// reaches (the populate is QD1-I/O-bound on the 27.6 GB sidecar). Best-
    /// effort perf hint only: clamped to range, no-op when no data, never throws.
    void prefetch_rows(uint64_t start_row, uint64_t end_row) const noexcept;

    /// Access the parsed header. Zero-initialized when `has_data() == false`.
    /// Downstream components use `header().source_sha256` as the staleness
    /// gate against the projection's source `.leaf`.
    const TopologySnapshotHeader& header() const noexcept { return header_; }

    /// Staleness check vs the source `.leaf` file.
    /// Streams SHA-256 over `source_leaf_path` with a 64 KiB buffer
    /// (same chunking as the writer) and returns true iff the digest
    /// matches `header().source_sha256`. Returns false when the reader
    /// has no data, the source path is unreadable, or any OpenSSL step
    /// fails — conservative: unverifiable → untrusted. `open()` invokes
    /// this automatically and falls back on mismatch; callers only need
    /// to use it directly for after-the-fact rehashing (e.g. tests).
    bool verify_source_sha256(const std::filesystem::path& source_leaf_path) const;

private:
    // Constructed only via `open()`. Private to prevent direct instantiation
    // bypassing the validation pipeline.
    TopologySnapshotReader() = default;

    // Release any mmap / fd owned by `*this`. Idempotent; used by dtor and
    // move-assign.
    void release_resources_() noexcept;

    // True iff `open()` reached the end of the validation pipeline. All
    // other members are meaningful only when this is true.
    bool has_data_ = false;

    TopologySnapshotHeader header_ = {};

    // Raw mmap base + byte length. mmap'd with MAP_PRIVATE + MADV_RANDOM.
    // `file_size_` equals the size computed from the header (§5.2 step 4).
    void*       map_base_  = nullptr;
    std::size_t file_size_ = 0;

    // Pointers into `map_base_`. Valid iff `has_data_`.
    //   row_ptr_[0..N]  — offsets in element counts (not bytes) into col_idx.
    //                     ALWAYS uint64 regardless of id_width.
    //   col_idx_[0..M-1] / edge_ids_[0..M-1]   — set when id_width==8 (full
    //                     tagged ObjectIds); nullptr when id_width==4.
    //   col_idx32_[0..M-1] / edge_ids32_[0..M-1] — set when id_width==4
    //                     (tag-stripped uint32 ordinals); nullptr when ==8.
    // Exactly one of {col_idx_, col_idx32_} is non-null when has_data_.
    const uint64_t* row_ptr_    = nullptr;
    const uint64_t* col_idx_    = nullptr;
    const uint64_t* edge_ids_   = nullptr;
    const uint32_t* col_idx32_  = nullptr;
    const uint32_t* edge_ids32_ = nullptr;

    // Owned file descriptor for the mmap backing file. Kept open through
    // the reader's lifetime because some kernels require the fd for certain
    // madvise() operations; also simplifies teardown ordering.
    int fd_ = -1;
};

}  // namespace GQL::Projection
