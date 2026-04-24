// Implementation of the v2 leaf header serialization helpers and LeafFormat
// enum <-> string utilities declared in bpt_leaf_format.h.
//
// Serialization uses explicit per-byte shifts so the on-disk layout is
// little-endian regardless of the host byte order. We deliberately avoid
// memcpy of the whole struct because that would couple the disk format to
// the host's representation and to compiler-specific struct padding.
//
// Spec reference: docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md

#include "storage/index/bplus_tree/bpt_leaf_format.h"

#include <stdexcept>
#include <string>

namespace BPT {

void serialize_header(const BPTLeafV2Header& h, uint8_t out[16]) noexcept {
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

    out[12] = static_cast<uint8_t>( h.reserved2        & 0xFF);
    out[13] = static_cast<uint8_t>((h.reserved2 >>  8) & 0xFF);
    out[14] = static_cast<uint8_t>((h.reserved2 >> 16) & 0xFF);
    out[15] = static_cast<uint8_t>((h.reserved2 >> 24) & 0xFF);
}

BPTLeafV2Header deserialize_header(const uint8_t in[16]) noexcept {
    BPTLeafV2Header h;
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

    h.reserved2   =  static_cast<uint32_t>(in[12])
                  | (static_cast<uint32_t>(in[13]) <<  8)
                  | (static_cast<uint32_t>(in[14]) << 16)
                  | (static_cast<uint32_t>(in[15]) << 24);
    return h;
}

const char* leaf_format_to_string(LeafFormat f) noexcept {
    switch (f) {
        case LeafFormat::BITSET:       return "BITSET";
        case LeafFormat::DELTA_VARINT: return "DELTA_VARINT";
    }
    return "<invalid>";
}

LeafFormat parse_leaf_format(std::string_view s) {
    if (s == "BITSET") {
        return LeafFormat::BITSET;
    }
    if (s == "DELTA_VARINT") {
        return LeafFormat::DELTA_VARINT;
    }
    std::string msg = "Unknown leafFormat value '";
    msg.append(s.data(), s.size());
    msg += "'. Valid values: 'BITSET', 'DELTA_VARINT'.";
    throw std::invalid_argument(msg);
}

}  // namespace BPT
