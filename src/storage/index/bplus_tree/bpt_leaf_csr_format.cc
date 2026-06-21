// Implementation of the v3 (CSR_HYBRID) leaf header serialization helpers
// declared in bpt_leaf_csr_format.h.
//
// Serialization uses explicit per-byte shifts so the on-disk layout is
// little-endian regardless of the host byte order. We deliberately avoid
// memcpy of the whole struct because that would couple the disk format
// to the host's representation and to compiler-specific struct padding.
//
// Mirrors the pattern established by bpt_leaf_format.cc, which implements
// the delta + LEB128-varint B+Tree leaf compression format.
//
// Design reference: docs/superpowers/specs/2026-04-25-csr-hybrid-design.md

#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"

namespace BPT {

void serialize_csr_header(const BPTLeafCSRHeader& h, uint8_t out[16]) noexcept {
    out[0] = h.format_version;
    out[1] = h.record_width;
    out[2] = h.flags;
    out[3] = h.reserved;

    out[4] = static_cast<uint8_t>( h.value_count        & 0xFF);
    out[5] = static_cast<uint8_t>((h.value_count >>  8) & 0xFF);
    out[6] = static_cast<uint8_t>((h.value_count >> 16) & 0xFF);
    out[7] = static_cast<uint8_t>((h.value_count >> 24) & 0xFF);

    out[8]  = static_cast<uint8_t>( h.next_leaf        & 0xFF);
    out[9]  = static_cast<uint8_t>((h.next_leaf >>  8) & 0xFF);
    out[10] = static_cast<uint8_t>((h.next_leaf >> 16) & 0xFF);
    out[11] = static_cast<uint8_t>((h.next_leaf >> 24) & 0xFF);

    out[12] = static_cast<uint8_t>( h.min_src_id_low        & 0xFF);
    out[13] = static_cast<uint8_t>((h.min_src_id_low >>  8) & 0xFF);
    out[14] = static_cast<uint8_t>((h.min_src_id_low >> 16) & 0xFF);
    out[15] = static_cast<uint8_t>((h.min_src_id_low >> 24) & 0xFF);
}

BPTLeafCSRHeader deserialize_csr_header(const uint8_t in[16]) noexcept {
    BPTLeafCSRHeader h;
    h.format_version = in[0];
    h.record_width   = in[1];
    h.flags          = in[2];
    h.reserved       = in[3];

    h.value_count =  static_cast<uint32_t>(in[4])
                  | (static_cast<uint32_t>(in[5]) <<  8)
                  | (static_cast<uint32_t>(in[6]) << 16)
                  | (static_cast<uint32_t>(in[7]) << 24);

    h.next_leaf   =  static_cast<uint32_t>(in[8])
                  | (static_cast<uint32_t>(in[9])  <<  8)
                  | (static_cast<uint32_t>(in[10]) << 16)
                  | (static_cast<uint32_t>(in[11]) << 24);

    h.min_src_id_low =  static_cast<uint32_t>(in[12])
                     | (static_cast<uint32_t>(in[13]) <<  8)
                     | (static_cast<uint32_t>(in[14]) << 16)
                     | (static_cast<uint32_t>(in[15]) << 24);
    return h;
}

void serialize_csr_continuation_header(const BPTLeafCSRContinuationHeader& h,
                                       uint8_t out[16]) noexcept {
    out[0] = h.format_version;
    out[1] = h.record_width;
    out[2] = h.flags;
    out[3] = h.reserved;

    out[4] = static_cast<uint8_t>( h.chunk_count        & 0xFF);
    out[5] = static_cast<uint8_t>((h.chunk_count >>  8) & 0xFF);
    out[6] = static_cast<uint8_t>((h.chunk_count >> 16) & 0xFF);
    out[7] = static_cast<uint8_t>((h.chunk_count >> 24) & 0xFF);

    out[8]  = static_cast<uint8_t>( h.next_leaf        & 0xFF);
    out[9]  = static_cast<uint8_t>((h.next_leaf >>  8) & 0xFF);
    out[10] = static_cast<uint8_t>((h.next_leaf >> 16) & 0xFF);
    out[11] = static_cast<uint8_t>((h.next_leaf >> 24) & 0xFF);

    out[12] = static_cast<uint8_t>( h.chain_head_page_id        & 0xFF);
    out[13] = static_cast<uint8_t>((h.chain_head_page_id >>  8) & 0xFF);
    out[14] = static_cast<uint8_t>((h.chain_head_page_id >> 16) & 0xFF);
    out[15] = static_cast<uint8_t>((h.chain_head_page_id >> 24) & 0xFF);
}

BPTLeafCSRContinuationHeader deserialize_csr_continuation_header(
    const uint8_t in[16]) noexcept {
    BPTLeafCSRContinuationHeader h;
    h.format_version = in[0];
    h.record_width   = in[1];
    h.flags          = in[2];
    h.reserved       = in[3];

    h.chunk_count =  static_cast<uint32_t>(in[4])
                  | (static_cast<uint32_t>(in[5]) <<  8)
                  | (static_cast<uint32_t>(in[6]) << 16)
                  | (static_cast<uint32_t>(in[7]) << 24);

    h.next_leaf   =  static_cast<uint32_t>(in[8])
                  | (static_cast<uint32_t>(in[9])  <<  8)
                  | (static_cast<uint32_t>(in[10]) << 16)
                  | (static_cast<uint32_t>(in[11]) << 24);

    h.chain_head_page_id =  static_cast<uint32_t>(in[12])
                         | (static_cast<uint32_t>(in[13]) <<  8)
                         | (static_cast<uint32_t>(in[14]) << 16)
                         | (static_cast<uint32_t>(in[15]) << 24);
    return h;
}

bool is_csr_continuation(const uint8_t in[16]) noexcept {
    // Flags byte lives at offset 2; bit 0 distinguishes continuation from
    // chain-head pages. Intentionally does not touch any other byte so the
    // sniff is O(1) with no cache pressure beyond the first line.
    return (in[2] & CSRHybridFlags::kIsContinuation) != 0;
}

}  // namespace BPT
