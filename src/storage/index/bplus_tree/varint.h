#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace BPT {

// Max bytes a LEB128 varint can consume for a uint64. ceil(64 / 7) = 10.
constexpr size_t VARINT_MAX_BYTES = 10;

// Raised on malformed or truncated varint input, and on overlong encodings.
// Carries the byte offset (relative to the decoder's start pointer) at which
// the failure was detected, so callers can attribute the corruption to a
// specific location within a leaf page.
class BPTLeafV2DecodeException : public std::runtime_error {
public:
    explicit BPTLeafV2DecodeException(const std::string& msg)
        : std::runtime_error(msg) {}
};

// LEB128 (unsigned, little-endian, continuation-bit) encoder.
//
// Writes the canonical encoding of `value` to `out_buf`, returns the number
// of bytes consumed (1..VARINT_MAX_BYTES). Never emits overlong encodings:
// for any `value`, the length is the smallest that fits.
//
// Precondition: `max_bytes >= VARINT_MAX_BYTES` (10). Caller owns the buffer
// and guarantees this. `noexcept` because the only error mode (undersized
// buffer) is the caller's contract violation.
//
// Layout: bytes emitted low-significance first. Each byte carries 7 payload
// bits in the low bits + 1 continuation bit in the high bit (1 = more bytes
// follow, 0 = this is the last byte).
size_t varint_encode(uint64_t value, uint8_t* out_buf, size_t max_bytes) noexcept;

// LEB128 decoder with bounds check.
//
// Decodes one varint starting at `in`, stopping when the continuation bit
// clears or when 10 bytes have been consumed. Writes the decoded value to
// `out_value` and returns the number of bytes consumed.
//
// Raises BPTLeafV2DecodeException if:
//   - the stream ends (in + consumed == end) before a terminator byte is seen;
//   - 10 bytes have been consumed and the 10th byte still has continuation bit set;
//   - the 10th byte (which can carry only the top 4 bits of a uint64) has
//     payload > 0x01 — that encoding would overflow uint64;
//   - the encoding is "overlong": the final byte is 0x00 AND the varint has
//     more than 1 byte (an encoding like `0x80 0x00` for value 0 instead of
//     the canonical `0x00`). Overlong is rejected because it breaks the
//     canonical-encoding invariant that secures the fuzz contract.
//
// Precondition: `in <= end`. If `in == end`, the function immediately raises.
size_t varint_decode(const uint8_t* in, const uint8_t* end, uint64_t& out_value);

// Upper bound on varint-encoded size for a range of uint64 values. Used by
// the v2 writer's overflow-check loop to pre-size records before committing
// to a page flush. `varint_size(v)` == `varint_encode(v, scratch, 10)` without
// actually writing — 1..10.
size_t varint_size(uint64_t value) noexcept;

}  // namespace BPT
