#pragma once

// TopologySnapshot — mmap-backed CSR sidecar for projection sampling.
//
// This header defines only the on-disk file format (the 64-byte fixed
// header) and the serialize/parse primitives that the sidecar writer and
// reader will share. No file I/O, no mmap, no SHA-256 state lives
// here — those layers build on top of this format contract.
//
// Spec reference: docs/superpowers/specs/2026-04-25-topology-snapshot-design.md
//                 §3.5, §4.3, §5.1, §5.2
//
// On-disk layout (little-endian, see §5.1):
//
//   Offset  Size  Field              Notes
//   ────────────────────────────────────────────────────────────────────
//   0       8     magic              "TOPOCSR1" (8 ASCII bytes, no NUL)
//   8       4     version            uint32 = 1
//   12      1     id_width           uint8 = 8 (full 64-bit ObjectId) or 4 (narrow uint32:
//                                    the 8-bit ObjectId type tag is stripped before writing
//                                    and re-OR'd from the header's dst_type_tag/edge_type_tag
//                                    on read; lossless when all values fit in 32 bits after
//                                    masking and every entry in a section shares one type tag)
//   13      1     flags              bit 0 = has_edge_ids; rest reserved=0
//   14      1     dst_type_tag       uint8 — ObjectId type tag re-applied to COL_IDX
//                                    values when id_width==4; 0 when id_width==8.
//   15      1     edge_type_tag      uint8 — ObjectId type tag re-applied to EDGE_IDS
//                                    values when id_width==4; 0 when id_width==8.
//   16      8     num_nodes          uint64 N
//   24      8     num_edges          uint64 M
//   32      32    source_sha256      SHA-256 of source .leaf file
//   64      —     (end of header)
//
// The body (ROW_PTR / COL_IDX / optional EDGE_IDS sections) follows the
// header and is handled by the writer/reader implementations — not here.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace GQL::Projection {

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

/// Magic bytes stored raw at file offset 0 (8 bytes, no NUL terminator).
inline constexpr std::array<uint8_t, 8> kTopologySnapshotMagic = {
    'T', 'O', 'P', 'O', 'C', 'S', 'R', '1'
};

/// Currently supported on-disk version.
inline constexpr uint32_t kTopologySnapshotVersion = 1;

/// Default ObjectId width in bytes (full tagged ObjectId.id). Writers emit
/// this unless the narrow uint32 variant is eligible (i.e., all node/edge ids
/// in the section fit in 32 bits after stripping the 8-bit type tag, allowing
/// the sidecar to store uint32 values instead of uint64, halving its size).
inline constexpr uint8_t kTopologySnapshotIdWidth = 8;

/// Narrow ObjectId width: COL_IDX / EDGE_IDS stored as uint32 of the
/// tag-stripped value (id & 0x00FFFFFFFFFFFFFF), with the constant per-section
/// type tag carried in the header (dst_type_tag / edge_type_tag) and re-OR'd
/// onto each value on read. Lossless iff every value < 2^32 after masking and
/// every entry in a section shares one type tag. Parse accepts {4, 8}.
inline constexpr uint8_t kTopologySnapshotIdWidthNarrow = 4;

/// Bit mask that strips the 8-bit ObjectId type tag, leaving the 56-bit value.
inline constexpr uint64_t kTopologySnapshotValueMask = 0x00FFFFFFFFFFFFFFULL;

/// True iff `w` is a supported on-disk id width.
inline constexpr bool topology_snapshot_id_width_valid(uint8_t w) {
    return w == kTopologySnapshotIdWidth || w == kTopologySnapshotIdWidthNarrow;
}

/// Flag bits in `TopologySnapshotHeader::flags`.
///
/// Unknown bits are *preserved* on round-trip but not acted upon. Readers
/// MUST ignore bits they do not recognize (forward-compatible).
namespace TopologySnapshotFlags {
inline constexpr uint8_t kHasEdgeIds = 0x01;  // bit 0 — EDGE_IDS section present
// bits 1..7 reserved, must round-trip untouched.
}  // namespace TopologySnapshotFlags

/// Fixed header size on disk.
inline constexpr std::size_t kTopologySnapshotHeaderSize = 64;

// ---------------------------------------------------------------------------
// Header struct — byte-exact on-disk representation
// ---------------------------------------------------------------------------
//
// Field ordering is intentional (§5.1) for natural alignment under the C++
// standard layout rules so that `sizeof == 64` holds without padding on
// every supported target. `static_assert` below locks this in at compile
// time.
//
// All multi-byte fields are little-endian on disk. On x86_64 / AArch64 (the
// only platforms MillenniumDB targets), the in-memory representation
// already matches little-endian, so `std::memcpy` between a byte buffer and
// this struct is a correct serialization / deserialization. Parse helpers
// below still validate the magic, version, and id_width explicitly rather
// than relying on layout alone.

struct TopologySnapshotHeader {
    uint8_t  magic[8];            // "TOPOCSR1"
    uint32_t version;             // kTopologySnapshotVersion
    uint8_t  id_width;            // 8 (full 64-bit ObjectId) or 4 (tag-stripped uint32)
    uint8_t  flags;               // TopologySnapshotFlags::*
    uint8_t  dst_type_tag;        // ObjectId type tag re-applied to COL_IDX when id_width==4; else 0
    uint8_t  edge_type_tag;       // ObjectId type tag re-applied to EDGE_IDS when id_width==4; else 0
    uint64_t num_nodes;           // N (number of source nodes)
    uint64_t num_edges;           // M (number of edges)
    uint8_t  source_sha256[32];   // SHA-256 of source .leaf file
};

static_assert(sizeof(TopologySnapshotHeader) == kTopologySnapshotHeaderSize,
              "TopologySnapshotHeader must be exactly 64 bytes — see spec §5.1");
static_assert(alignof(TopologySnapshotHeader) <= 8,
              "TopologySnapshotHeader alignment must be <= 8 to match disk layout");

// ---------------------------------------------------------------------------
// Serialize / parse helpers
// ---------------------------------------------------------------------------

/// Exception type raised by `parse_topology_snapshot_header` when a byte
/// buffer fails validation. Use this to drive the reader's "stale/corrupt
/// → fall back to B+Tree" policy (§3.4, §5.2).
class TopologySnapshotFormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Serializes `header` into the 64-byte little-endian on-disk form.
///
/// This does NOT validate `header` — callers (the writer) are expected to
/// fill in the final content (magic, version, id_width, etc.) correctly
/// before calling. The function is the inverse of `parse_topology_snapshot_header`.
inline void serialize_topology_snapshot_header(
    const TopologySnapshotHeader& header,
    uint8_t (&out)[kTopologySnapshotHeaderSize]) noexcept {
    std::memcpy(out, &header, kTopologySnapshotHeaderSize);
}

/// Parses the 64-byte header from `in`, validating:
///   - magic == "TOPOCSR1"
///   - version == kTopologySnapshotVersion
///   - id_width == kTopologySnapshotIdWidth
///
/// On any validation failure, throws `TopologySnapshotFormatError` with a
/// human-readable message. On success, returns a fully populated
/// `TopologySnapshotHeader`. Unknown flag bits are preserved; reserved
/// byte fields are copied through without interpretation.
inline TopologySnapshotHeader parse_topology_snapshot_header(
    const uint8_t (&in)[kTopologySnapshotHeaderSize]) {
    TopologySnapshotHeader header{};
    std::memcpy(&header, in, kTopologySnapshotHeaderSize);

    if (std::memcmp(header.magic,
                    kTopologySnapshotMagic.data(),
                    kTopologySnapshotMagic.size()) != 0) {
        throw TopologySnapshotFormatError(
            "TopologySnapshot: invalid magic (expected \"TOPOCSR1\")");
    }
    if (header.version != kTopologySnapshotVersion) {
        throw TopologySnapshotFormatError(
            "TopologySnapshot: unsupported version " +
            std::to_string(header.version) + " (expected " +
            std::to_string(kTopologySnapshotVersion) + ")");
    }
    if (!topology_snapshot_id_width_valid(header.id_width)) {
        throw TopologySnapshotFormatError(
            "TopologySnapshot: unsupported id_width " +
            std::to_string(header.id_width) + " (expected 4 or 8)");
    }
    return header;
}

/// Convenience wrapper: fills a fresh `TopologySnapshotHeader` with the
/// format constants (magic, version, id_width, zero-initialized reserved
/// fields). The caller is expected to then set `flags`, `num_nodes`,
/// `num_edges`, and `source_sha256` before serializing.
inline TopologySnapshotHeader make_default_topology_snapshot_header() noexcept {
    TopologySnapshotHeader header{};
    std::memcpy(header.magic,
                kTopologySnapshotMagic.data(),
                kTopologySnapshotMagic.size());
    header.version  = kTopologySnapshotVersion;
    header.id_width = kTopologySnapshotIdWidth;
    header.flags    = 0;
    // dst_type_tag, edge_type_tag, num_nodes, num_edges, source_sha256 are
    // zero-initialized via value-initialization above. The narrow (id_width==4)
    // writer sets the tag bytes explicitly.
    return header;
}

// ---------------------------------------------------------------------------
// Symmetric CSR variant (pre-merged undirected topology)
// ---------------------------------------------------------------------------
//
// `topology_sym.csr` is a pre-merged undirected CSR: byte-for-byte the SAME
// 64-byte header struct + ROW_PTR[N+1] + COL_IDX[M] body as a directional
// snapshot, with TWO differences: (1) a distinct magic ("TOPOSYM1") + version
// so a directional reader/parser never accepts it; (2) `source_sha256[32]`
// holds a COMBINED digest chaining BOTH source `.leaf` streams (from_to_edge
// then to_from_edge, in a fixed order) — the symmetric body depends on both
// directions, so a single-source digest cannot express staleness.

/// Magic bytes for the symmetric pre-merged CSR sidecar (8 bytes, no NUL).
inline constexpr std::array<uint8_t, 8> kTopologySnapshotSymMagic = {
    'T', 'O', 'P', 'O', 'S', 'Y', 'M', '1'
};

/// On-disk version of the symmetric variant.
inline constexpr uint32_t kTopologySnapshotSymVersion = 1;

/// Non-throwing magic peek: true iff `in` carries the symmetric magic. Lets a
/// caller (e.g. a unified opener) discriminate sym vs directional before
/// committing to a parser. Does NOT validate version or id_width.
inline bool topology_snapshot_header_is_symmetric(
    const uint8_t (&in)[kTopologySnapshotHeaderSize]) noexcept {
    return std::memcmp(in, kTopologySnapshotSymMagic.data(),
                       kTopologySnapshotSymMagic.size()) == 0;
}

/// Parses a 64-byte SYMMETRIC header, validating magic=="TOPOSYM1",
/// version==kTopologySnapshotSymVersion, and id_width in {4,8}. Throws
/// TopologySnapshotFormatError on any mismatch. The struct, body layout, and
/// every numeric field are identical to the directional header — only the
/// magic/version gate differs. `source_sha256` carries the combined digest.
inline TopologySnapshotHeader parse_topology_snapshot_sym_header(
    const uint8_t (&in)[kTopologySnapshotHeaderSize]) {
    TopologySnapshotHeader header{};
    std::memcpy(&header, in, kTopologySnapshotHeaderSize);

    if (std::memcmp(header.magic,
                    kTopologySnapshotSymMagic.data(),
                    kTopologySnapshotSymMagic.size()) != 0) {
        throw TopologySnapshotFormatError(
            "TopologySnapshot(sym): invalid magic (expected \"TOPOSYM1\")");
    }
    if (header.version != kTopologySnapshotSymVersion) {
        throw TopologySnapshotFormatError(
            "TopologySnapshot(sym): unsupported version " +
            std::to_string(header.version) + " (expected " +
            std::to_string(kTopologySnapshotSymVersion) + ")");
    }
    if (!topology_snapshot_id_width_valid(header.id_width)) {
        throw TopologySnapshotFormatError(
            "TopologySnapshot(sym): unsupported id_width " +
            std::to_string(header.id_width) + " (expected 4 or 8)");
    }
    return header;
}

/// Symmetric counterpart of make_default_topology_snapshot_header(): fills
/// magic/version/id_width and zeroes the rest. The caller sets flags,
/// num_nodes, num_edges, dst_type_tag, and the combined source_sha256.
inline TopologySnapshotHeader make_default_topology_snapshot_sym_header() noexcept {
    TopologySnapshotHeader header{};
    std::memcpy(header.magic,
                kTopologySnapshotSymMagic.data(),
                kTopologySnapshotSymMagic.size());
    header.version  = kTopologySnapshotSymVersion;
    header.id_width = kTopologySnapshotIdWidth;
    header.flags    = 0;
    return header;
}

}  // namespace GQL::Projection
