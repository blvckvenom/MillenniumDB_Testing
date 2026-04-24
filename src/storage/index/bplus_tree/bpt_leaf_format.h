// B+Tree leaf format identifiers and v2 header struct (Spec #5).
//
// This header introduces the LeafFormat enum and the on-disk layout of the
// v2 leaf header. The pre-Spec-#5 redundant-bitset encoding is retained as
// LeafFormat::BITSET (value 1); the new delta + LEB128 varint encoding is
// LeafFormat::DELTA_VARINT (value 2). Subsequent tasks (T5.4-T5.8) build on
// this foundation: T5.4 adds the varint codec, T5.5 the zigzag helpers, and
// T5.6+ refactors the existing leaf reader/writer pair to dispatch on the
// format byte recorded here.
//
// Spec reference: docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md
//                 (§4.2 enum / §5.2 disk layout)

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace BPT {

enum class LeafFormat : uint8_t {
    BITSET       = 1,   // v1 — existing redundant-bitset encoding (pre-Spec-#5)
    DELTA_VARINT = 2,   // v2 — delta + LEB128 varint encoding (Spec #5)
};

// Exactly 16 bytes, matches disk layout in design §5.2.
// Layout:
//   0  : format_version  uint8 = 2
//   1  : record_width    uint8 = N
//   2  : flags           uint8  (reserved; 0 in Spec #5)
//   3  : reserved        uint8 = 0
//   4  : value_count     uint32 LE
//   8  : next_leaf       uint32 LE
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
              "BPTLeafV2Header must be exactly 16 bytes (Spec #5 disk format)");

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

}  // namespace BPT
