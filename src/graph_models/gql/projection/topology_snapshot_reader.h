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
// caller — today `TopologyAccessor::Impl` (T4.7), tomorrow anything that
// wants the fast path — is expected to branch on `has_data()` and fall back
// to the existing B+Tree path when the sidecar is absent. A single log line
// per failure mode is emitted on stderr for debuggability; there is no
// throw / no return-code on open.
//
// Staleness (SHA-256 of the source `.leaf`) is validated separately by
// `verify_source_sha256()`. As of T4.10, `open()` runs the verification
// after structural validation and collapses a mismatch into the same
// has_data()==false fallback path used for the "sidecar absent" case.
//
// Spec reference: docs/superpowers/specs/2026-04-25-topology-snapshot-design.md
//                 §3.4 (fallback-first arch), §4.3 (C++ surface),
//                 §5.1 (on-disk layout), §5.2 (validation checklist).

#include <cstddef>
#include <cstdint>
#include <filesystem>

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

    /// O(1) slice into the mmap'd COL_IDX section.
    /// Precondition: `has_data() == true` and `node_idx < num_nodes()`.
    /// Bounds violation → throws `std::out_of_range` in every build config
    /// (same hardening discipline as `TopologySnapshotWriter::append_edge`'s
    /// always-on invariant checks — a silent out-of-range mmap read is a
    /// segfault-or-garbage-data waiting to happen at the call site).
    /// Calling while `has_data() == false` is also rejected with
    /// `std::out_of_range` rather than returning an empty span, so misuse
    /// surfaces at the caller rather than being papered over.
    ConstU64Span neighbors(uint64_t node_idx) const;

    /// O(1) slice into the mmap'd EDGE_IDS section. Returns an empty span
    /// when `has_edge_ids() == false` (but `has_data()` still constrains
    /// access — see `neighbors()` contract). Same out-of-range rules.
    ConstU64Span edge_ids(uint64_t node_idx) const;

    /// Access the parsed header. Zero-initialized when `has_data() == false`.
    /// Downstream components (T4.10) use `header().source_sha256` as the
    /// staleness gate against the projection's source `.leaf`.
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
    //   col_idx_[0..M-1] — neighbor ids.
    //   edge_ids_[0..M-1] — edge ids, only when `has_edge_ids()`.
    const uint64_t* row_ptr_  = nullptr;
    const uint64_t* col_idx_  = nullptr;
    const uint64_t* edge_ids_ = nullptr;

    // Owned file descriptor for the mmap backing file. Kept open through
    // the reader's lifetime because some kernels require the fd for certain
    // madvise() operations; also simplifies teardown ordering.
    int fd_ = -1;
};

}  // namespace GQL::Projection
