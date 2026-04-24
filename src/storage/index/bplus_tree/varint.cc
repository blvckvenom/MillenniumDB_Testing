#include "varint.h"

namespace BPT {

size_t varint_encode(uint64_t value, uint8_t* out_buf, size_t max_bytes) noexcept {
    // Caller contract: max_bytes >= VARINT_MAX_BYTES (10). If violated, this
    // loop still bounds writes by `max_bytes - 1` so we never run past the
    // buffer; the result may be truncated but won't corrupt memory.
    size_t i = 0;
    while (value >= 0x80 && i + 1 < max_bytes) {
        out_buf[i++] = static_cast<uint8_t>((value & 0x7F) | 0x80);
        value >>= 7;
    }
    if (i < max_bytes) {
        out_buf[i++] = static_cast<uint8_t>(value);
    }
    return i;
}

namespace {

// Format an unsigned byte as "0xNN" (uppercase hex, 2-digit). Used only by
// exception messages; std::to_string would emit decimal, which is harder to
// correlate with on-disk leaf bytes when debugging a corruption.
std::string hex_byte(uint8_t b) {
    static const char digits[] = "0123456789ABCDEF";
    std::string s = "0x";
    s.push_back(digits[(b >> 4) & 0x0F]);
    s.push_back(digits[b & 0x0F]);
    return s;
}

}  // namespace

size_t varint_decode(const uint8_t* in, const uint8_t* end, uint64_t& out_value) {
    uint64_t result = 0;
    unsigned shift = 0;
    size_t consumed = 0;
    while (in + consumed < end && consumed < VARINT_MAX_BYTES) {
        uint8_t b = in[consumed];
        consumed++;
        if (consumed == VARINT_MAX_BYTES) {
            // 10th byte: can contribute at most 1 bit (since 9 * 7 = 63).
            // Continuation bit must be clear, and payload must be <= 0x01.
            if (b & 0x80) {
                throw BPTLeafV2DecodeException(
                    "varint continuation bit still set at 10th byte (offset "
                    + std::to_string(consumed - 1) + ")");
            }
            if (b > 0x01) {
                throw BPTLeafV2DecodeException(
                    "varint 10th-byte payload overflows uint64 (offset "
                    + std::to_string(consumed - 1) + ", payload "
                    + hex_byte(b) + ")");
            }
            // Overlong check at the 10th byte: a pure-zero terminator means
            // every earlier byte's payload was zero too, so the canonical
            // encoding would have been the single byte 0x00. Reject.
            if (b == 0x00) {
                throw BPTLeafV2DecodeException(
                    "varint overlong encoding (redundant zero-payload final byte at offset "
                    + std::to_string(consumed - 1) + ")");
            }
            result |= static_cast<uint64_t>(b) << shift;
            out_value = result;
            return consumed;
        }
        result |= static_cast<uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) {
            // Overlong check: if this is not the first byte AND payload is 0,
            // the canonical encoding would have terminated earlier. The
            // canonical encoding of 0 is the single byte 0x00; any multi-byte
            // sequence ending in a pure-zero byte is overlong.
            if (consumed > 1 && (b & 0x7F) == 0) {
                throw BPTLeafV2DecodeException(
                    "varint overlong encoding (redundant zero-payload final byte at offset "
                    + std::to_string(consumed - 1) + ")");
            }
            out_value = result;
            return consumed;
        }
        shift += 7;
    }
    // Fell through without hitting a terminator. Either the stream ended or
    // we exhausted 10 bytes with continuation bits still set (the 10th-byte
    // continuation case is already handled above, so this path means
    // truncation against `end`).
    throw BPTLeafV2DecodeException(
        "varint truncated: reached end of buffer without terminator (consumed "
        + std::to_string(consumed) + " bytes)");
}

size_t varint_size(uint64_t value) noexcept {
    size_t n = 1;
    while (value >= 0x80) {
        n++;
        value >>= 7;
    }
    return n;
}

}  // namespace BPT
