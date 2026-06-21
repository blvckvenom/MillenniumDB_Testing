// B+Tree leaf v3 header format — CSR-in-B+Tree hybrid graph storage.
//
// This header introduces the 16-byte on-disk header struct used by the
// v3 leaf page format (LeafFormat::CSR_HYBRID). In this format, the edge-index
// B+Tree leaves directly encode a compressed-sparse-row adjacency layout,
// making them the CSR itself rather than an index over separate adjacency data.
// The layout is deliberately the same size as the v2 (delta + LEB128-varint
// leaf encoding) header so the upstream dispatcher can read 16 bytes once and
// route to the v1/v2/v3 decoder based on byte 0.
//
// A single struct covers BOTH page variants:
//   - Chain-head pages (flags & kIsContinuation == 0): carry src entries
//     with an in-page offset table.
//   - Continuation pages (flags & kIsContinuation == 1): carry the chunk
//     bodies for high-degree ("hub") source nodes whose adjacency list
//     spans multiple pages.
//
// On continuation pages the `value_count` field is repurposed as
// `chunk_count` (number of col_idx entries in this chunk) and the
// `min_src_id_low` field is repurposed as `chain_head_page_id` (the page
// id of the chain head that owns this continuation, for fault isolation
// and reader disambiguation).
//
// Scope of this file: type definitions only (header structs, flag constants,
// serialization declarations, and the decode-exception type). The reader,
// writer, and dispatch integration are in sibling translation units.

#pragma once

#include <cstdint>
#include <stdexcept>

#include "storage/index/bplus_tree/bpt_leaf_format.h"

namespace BPT {

// Exactly 16 bytes. Mirrors BPTLeafV2Header's footprint so the upstream
// leaf-decode dispatch site is uniform across v1/v2/v3.
//
// Layout (little-endian on disk, serialized via serialize_csr_header):
//
//   0  : format_version   uint8  == 3
//   1  : record_width     uint8  == 3 for edge indexes (defense-in-depth)
//   2  : flags            uint8  — bit 0 = is_continuation
//                                  bit 1 = has_edge_ids (chain head only;
//                                          continuation pages inherit from
//                                          their chain head)
//                                  bits 2..7 reserved, must be 0
//   3  : reserved         uint8  == 0
//   4  : value_count      uint32 LE
//                         — chain head: number of src entries on this page
//                         — continuation: number of col_idx entries in
//                           this chunk (aka chunk_count)
//   8  : next_leaf        uint32 LE — page id of next leaf (0 = last)
//   12 : min_src_id_low   uint32 LE
//                         — chain head: low 32 bits of the minimum src_id
//                           whose entry lives on this page (fsck
//                           cross-check against directory routing key)
//                         — continuation: page id of the chain head that
//                           owns this continuation (aka chain_head_page_id)
struct BPTLeafCSRHeader {
    uint8_t  format_version;
    uint8_t  record_width;
    uint8_t  flags;
    uint8_t  reserved;
    uint32_t value_count;
    uint32_t next_leaf;
    uint32_t min_src_id_low;
};
static_assert(sizeof(BPTLeafCSRHeader) == 16,
              "BPTLeafCSRHeader must be exactly 16 bytes (the CSR-hybrid v3 disk format)");

// Continuation-page view of the same 16-byte v3 header. Identical memory
// layout to BPTLeafCSRHeader (the on-disk format is single for both page
// variants), but field names reflect the continuation-page semantics so
// call-sites can self-document. The dispatch site sniffs flags bit 0 via
// is_csr_continuation() and then deserializes into the appropriate view.
//
// Layout (little-endian on disk, serialized via
// serialize_csr_continuation_header):
//
//   0  : format_version       uint8  == 3
//   1  : record_width         uint8  == 3
//   2  : flags                uint8  — bit 0 = 1 (kIsContinuation)
//   3  : reserved             uint8  == 0
//   4  : chunk_count          uint32 LE — number of col_idx entries in
//                                          this chunk
//   8  : next_leaf            uint32 LE — next continuation page id
//                                          (0 if last in chain)
//   12 : chain_head_page_id   uint32 LE — back-pointer to the chain-head
//                                          page that owns this continuation
struct BPTLeafCSRContinuationHeader {
    uint8_t  format_version;
    uint8_t  record_width;
    uint8_t  flags;
    uint8_t  reserved;
    uint32_t chunk_count;
    uint32_t next_leaf;
    uint32_t chain_head_page_id;
};
static_assert(sizeof(BPTLeafCSRContinuationHeader) == 16,
              "BPTLeafCSRContinuationHeader must be exactly 16 bytes (the CSR-hybrid v3 disk format)");

// Flag bit positions. Use namespace-scoped constants so the intent at call
// sites reads `header.flags & BPT::CSRHybridFlags::kIsContinuation` rather
// than a bare 0x01 literal.
namespace CSRHybridFlags {
    inline constexpr uint8_t kIsContinuation = 0x01;
    inline constexpr uint8_t kHasEdgeIds     = 0x02;

    // Set on the chain-head page of a hub node when the writer is emitting
    // parallel edge_ids alongside destination ids. The single
    // (value_count == 1) entry on such a page encodes a third varint after
    // (src_id, total_degree): the count `k_on_head` of dsts (and parallel
    // eids) physically stored on the chain-head. The chain-head's eid
    // stream has exactly k_on_head varints; continuation pages carry
    // chunk_count eid varints. Without this bit the chain-head's
    // physical_degrees_ walker could not distinguish dst varints from
    // the trailing eid varints.
    //
    // Mutually compatible with kHasEdgeIds: kIsHubChainHead WITHOUT
    // kHasEdgeIds is invalid (rejected at reader open).
    inline constexpr uint8_t kIsHubChainHead = 0x04;

    // Reserved mask — bits 3..7 must always be zero. Readers raise
    // BPTLeafCSRDecodeException if any bit in this mask is set.
    inline constexpr uint8_t kReservedMask   = 0xF8;
}  // namespace CSRHybridFlags

// Serialize the 16-byte header into a little-endian byte buffer. The
// format is endianness-independent on disk regardless of host byte order.
void serialize_csr_header(const BPTLeafCSRHeader& h, uint8_t out[16]) noexcept;

// Same, for the continuation-page view of the v3 header.
void serialize_csr_continuation_header(const BPTLeafCSRContinuationHeader& h,
                                       uint8_t out[16]) noexcept;

// Deserialize 16 little-endian bytes into a header struct. Does NOT
// validate semantic invariants (format_version == 3, reserved bits zero,
// etc.) — the caller (the CSR leaf reader) owns that validation.
BPTLeafCSRHeader deserialize_csr_header(const uint8_t in[16]) noexcept;

// Same, for the continuation-page view. The byte layout being identical,
// the choice of which deserializer to call is driven by the result of
// is_csr_continuation() applied to the same buffer.
BPTLeafCSRContinuationHeader deserialize_csr_continuation_header(
    const uint8_t in[16]) noexcept;

// Lightweight sniff: returns true iff byte 2 (flags) has bit 0 set,
// indicating a continuation page. Intended for use at dispatch sites that
// need to disambiguate chain-head vs continuation without paying for a
// full deserialize round trip. Does not validate any other invariant.
bool is_csr_continuation(const uint8_t in[16]) noexcept;

// Exception type raised by the CSR leaf reader on header validation failure.
// Declared here alongside the struct so both reader and writer tests can
// reference a single type.
class BPTLeafCSRDecodeException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace BPT
