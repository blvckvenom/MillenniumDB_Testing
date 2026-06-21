// B+Tree leaf format identifiers and v2 header struct.
//
// This header introduces the LeafFormat enum and the on-disk layout of the
// v2 leaf header used by the delta + LEB128-varint leaf encoding.
// The original redundant-bitset encoding is retained as LeafFormat::BITSET
// (value 1); the delta + LEB128-varint encoding that exploits sort-order to
// compress records is LeafFormat::DELTA_VARINT (value 2). The varint codec,
// zigzag helpers, and the leaf reader/writer pair all dispatch on the format
// byte stored in this header.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace BPT {

enum class LeafFormat : uint8_t {
    BITSET       = 1,   // v1 — original redundant-bitset encoding (legacy; bytes shared with same-prefix neighbors)
    DELTA_VARINT = 2,   // v2 — delta + LEB128 varint encoding (exploits sort order; record 0 full, subsequent records zigzag-delta compressed)
    CSR_HYBRID   = 3,   // v3 — CSR-in-B+Tree hybrid for edge indexes: leaves store the CSR layout directly (edge indexes only)
};

// Exactly 16 bytes; on-disk little-endian layout for v2 (DELTA_VARINT) leaves.
// Layout:
//   0  : format_version  uint8 = 2
//   1  : record_width    uint8 = N (number of uint64 columns per record)
//   2  : flags           uint8  (reserved, must be 0)
//   3  : reserved        uint8 = 0
//   4  : value_count     uint32 LE (number of records stored in this leaf)
//   8  : next_leaf       uint32 LE (page id of the next leaf, 0 if last)
//   12 : reserved2       uint32 = 0
struct BPTLeafV2Header {
    uint8_t  format_version;
    uint8_t  record_width;
    uint8_t  flags;
    uint8_t  reserved;
    uint32_t value_count;
    uint32_t next_leaf;
    uint32_t reserved2;
};
static_assert(sizeof(BPTLeafV2Header) == 16,
              "BPTLeafV2Header must be exactly 16 bytes (v2 leaf disk format)");

// The on-disk layout is little-endian by contract. These helpers enforce that
// regardless of host endianness (future-proofing; x86_64 is LE already).

// Serialize header into a 16-byte buffer (little-endian).
void serialize_header(const BPTLeafV2Header& h, uint8_t out[16]) noexcept;

// Deserialize from 16 raw bytes. Does NOT validate semantic invariants
// (format_version==2, record_width==N, reserved==0) — caller owns that.
BPTLeafV2Header deserialize_header(const uint8_t in[16]) noexcept;

// Enum <-> string utilities for catalog / config parsing.
const char* leaf_format_to_string(LeafFormat f) noexcept;

// Case-SENSITIVE parser. Accepts "BITSET" and "DELTA_VARINT" only.
// Raises std::invalid_argument on unknown input.
LeafFormat parse_leaf_format(std::string_view s);

// Per-projection graph-storage selector.
//
// Controls whether a projection's edge indexes persist as classic per-index
// B+Tree leaves (`BTREE`) or as inline CSR-in-B+Tree leaves emitted by the
// CSR-hybrid edge-index pipeline. In the CSR-hybrid mode the edge-index
// B+Tree leaves ARE the CSR layout: each leaf stores an offset table, a src
// table, and a DELTA_VARINT-encoded dst stream, giving O(1) neighbor access
// without a separate sidecar file.  The value is threaded from graph_project's
// `graphStorage` config key (case-sensitive), through NativeProjectionBuilder,
// into ProjectionCatalog::graph_storage (catalog v1.6 byte), and back out
// through ProjectionStorage on open.
enum class GraphStorage : uint8_t {
    BTREE       = 1,   // default; classic per-index B+Tree leaves (leaf encoding follows LeafFormat: BITSET or DELTA_VARINT)
    CSR_HYBRID  = 2,   // edge indexes emit v3 CSR-in-B+Tree leaves (offset table + src table + DELTA_VARINT dst stream; edge indexes only)
};

// Debug-friendly enum <-> string helpers, symmetric with the leaf_format
// pair above. Useful for log lines that name the selected storage mode.
const char* graph_storage_to_string(GraphStorage s) noexcept;

// Case-SENSITIVE parser. Accepts "BTREE" and "CSR_HYBRID" only.
// Raises std::invalid_argument on unknown input with a message naming
// the key ("Unknown graphStorage value '<s>'.").
GraphStorage parse_graph_storage(std::string_view s);

}  // namespace BPT
