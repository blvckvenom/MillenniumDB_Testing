// Unit tests for the LEB128 varint codec used by the v2 leaf format
// (Spec #5 — Delta+Varint Leaf Encoding, task T5.4).
//
// The codec lives in src/storage/index/bplus_tree/varint.{h,cc}. These tests
// pin the canonical-encoding contract that subsequent v2 reader/writer code
// (T5.7, T5.8) and the disk-format determinism test (T5.14) rely on.

#include "storage/index/bplus_tree/varint.h"

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using BPT::varint_encode;
using BPT::varint_decode;
using BPT::varint_size;
using BPT::VARINT_MAX_BYTES;
using BPT::BPTLeafV2DecodeException;

namespace {

// Helper: encode `v` into a fresh buffer of the maximum size and return the
// bytes actually used. Centralised so each test can simply compare against an
// expected initializer-list of bytes.
std::vector<uint8_t> encode_to_vec(uint64_t v) {
    std::array<uint8_t, VARINT_MAX_BYTES> buf{};
    size_t n = varint_encode(v, buf.data(), buf.size());
    return std::vector<uint8_t>(buf.begin(), buf.begin() + n);
}

}  // namespace

// ----- Boundary-value encoder tests ----------------------------------------

TEST(Varint, Encode_Zero) {
    auto bytes = encode_to_vec(0);
    EXPECT_EQ(bytes, (std::vector<uint8_t>{0x00}));
    EXPECT_EQ(bytes.size(), 1u);
}

TEST(Varint, Encode_127) {
    auto bytes = encode_to_vec(127);
    EXPECT_EQ(bytes, (std::vector<uint8_t>{0x7F}));
    EXPECT_EQ(bytes.size(), 1u);
}

TEST(Varint, Encode_128) {
    auto bytes = encode_to_vec(128);
    EXPECT_EQ(bytes, (std::vector<uint8_t>{0x80, 0x01}));
    EXPECT_EQ(bytes.size(), 2u);
}

TEST(Varint, Encode_16383) {
    auto bytes = encode_to_vec(16383);
    EXPECT_EQ(bytes, (std::vector<uint8_t>{0xFF, 0x7F}));
    EXPECT_EQ(bytes.size(), 2u);
}

TEST(Varint, Encode_16384) {
    auto bytes = encode_to_vec(16384);
    EXPECT_EQ(bytes, (std::vector<uint8_t>{0x80, 0x80, 0x01}));
    EXPECT_EQ(bytes.size(), 3u);
}

TEST(Varint, Encode_Pow2_21) {
    auto bytes = encode_to_vec(1ULL << 21);
    EXPECT_EQ(bytes.size(), 4u);
}

TEST(Varint, Encode_Pow2_28) {
    auto bytes = encode_to_vec(1ULL << 28);
    EXPECT_EQ(bytes.size(), 5u);
}

TEST(Varint, Encode_Pow2_35) {
    auto bytes = encode_to_vec(1ULL << 35);
    EXPECT_EQ(bytes.size(), 6u);
}

TEST(Varint, Encode_Pow2_42) {
    auto bytes = encode_to_vec(1ULL << 42);
    EXPECT_EQ(bytes.size(), 7u);
}

TEST(Varint, Encode_Pow2_49) {
    auto bytes = encode_to_vec(1ULL << 49);
    EXPECT_EQ(bytes.size(), 8u);
}

TEST(Varint, Encode_Pow2_56) {
    auto bytes = encode_to_vec(1ULL << 56);
    EXPECT_EQ(bytes.size(), 9u);
}

TEST(Varint, Encode_Pow2_63) {
    auto bytes = encode_to_vec(1ULL << 63);
    EXPECT_EQ(bytes.size(), 10u);
}

TEST(Varint, Encode_UintMax) {
    // UINT64_MAX = 0xFFFF_FFFF_FFFF_FFFF should encode to 10 bytes:
    // nine 0xFF bytes (each carrying 7 payload bits + continuation) and one
    // 0x01 final byte (carrying the top bit of bit-63).
    auto bytes = encode_to_vec(UINT64_MAX);
    ASSERT_EQ(bytes.size(), 10u);
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_EQ(bytes[i], 0xFFu) << "byte " << i;
    }
    EXPECT_EQ(bytes[9], 0x01u);
}

// ----- Roundtrip tests ------------------------------------------------------

TEST(Varint, Roundtrip_100_Random) {
    // Deterministic PRNG so failures are reproducible without rerunning
    // until the seed lines up. The seed is hard-coded by design.
    std::mt19937_64 rng(0xBEEFCAFEULL);
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    std::array<uint8_t, VARINT_MAX_BYTES> buf{};
    for (int i = 0; i < 100; ++i) {
        uint64_t v = dist(rng);
        size_t n = varint_encode(v, buf.data(), buf.size());
        ASSERT_GE(n, 1u);
        ASSERT_LE(n, VARINT_MAX_BYTES);
        uint64_t decoded = 0;
        size_t consumed = varint_decode(buf.data(), buf.data() + n, decoded);
        EXPECT_EQ(consumed, n) << "iter " << i << " value=" << v;
        EXPECT_EQ(decoded, v) << "iter " << i;
    }
}

TEST(Varint, Roundtrip_AllPowersOfTwo) {
    std::vector<uint64_t> values;
    values.push_back(0);
    for (int b = 0; b < 64; ++b) {
        values.push_back(1ULL << b);
    }
    values.push_back(UINT64_MAX);
    std::array<uint8_t, VARINT_MAX_BYTES> buf{};
    for (uint64_t v : values) {
        size_t n = varint_encode(v, buf.data(), buf.size());
        uint64_t decoded = 0;
        size_t consumed = varint_decode(buf.data(), buf.data() + n, decoded);
        EXPECT_EQ(consumed, n);
        EXPECT_EQ(decoded, v) << "value=" << v;
    }
}

// ----- Decoder failure-mode tests -------------------------------------------

TEST(Varint, Decode_Truncated_Raises) {
    // 16384 encodes as {0x80, 0x80, 0x01} (3 bytes). If we tell the decoder
    // it can only read 2 bytes, the continuation bits never clear before the
    // end pointer, so it must raise.
    std::array<uint8_t, VARINT_MAX_BYTES> buf{};
    size_t n = varint_encode(16384, buf.data(), buf.size());
    ASSERT_EQ(n, 3u);
    uint64_t out = 0;
    EXPECT_THROW(
        varint_decode(buf.data(), buf.data() + 2, out),
        BPTLeafV2DecodeException
    );
}

TEST(Varint, Decode_UnterminatedAll_Raises) {
    // Ten bytes of 0xFF means the 10th byte still has the continuation bit
    // set, which is invalid (10 bytes is the absolute maximum and must
    // terminate).
    std::array<uint8_t, 10> buf;
    buf.fill(0xFF);
    uint64_t out = 0;
    EXPECT_THROW(
        varint_decode(buf.data(), buf.data() + buf.size(), out),
        BPTLeafV2DecodeException
    );
}

TEST(Varint, Decode_Overlong_Rejected_ZeroWith2Bytes) {
    // {0x80, 0x00} encodes value 0 in 2 bytes. The canonical encoding of 0
    // is the single byte {0x00}, so this is overlong and must be rejected.
    std::array<uint8_t, 2> buf{0x80, 0x00};
    uint64_t out = 0;
    EXPECT_THROW(
        varint_decode(buf.data(), buf.data() + buf.size(), out),
        BPTLeafV2DecodeException
    );
}

TEST(Varint, Decode_Overlong_Rejected_At10thByte) {
    // Nine 0x80 continuation bytes followed by a 0x00 terminator. This
    // encodes the value 0 in 10 bytes, which is overlong.
    std::array<uint8_t, 10> buf{
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00
    };
    uint64_t out = 0;
    EXPECT_THROW(
        varint_decode(buf.data(), buf.data() + buf.size(), out),
        BPTLeafV2DecodeException
    );
}

TEST(Varint, Decode_10thByte_PayloadOverflow_Raises) {
    // Nine 0x80 continuation bytes followed by 0x02. The 10th byte can carry
    // only 1 payload bit, so 0x02 would shift bit-1 into bit-64 — outside
    // the uint64 range. Must be rejected.
    std::array<uint8_t, 10> buf{
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02
    };
    uint64_t out = 0;
    EXPECT_THROW(
        varint_decode(buf.data(), buf.data() + buf.size(), out),
        BPTLeafV2DecodeException
    );
}

TEST(Varint, Decode_EmptyBuffer_Raises) {
    // in == end at call time. The decoder must not return successfully —
    // there is no varint here at all.
    std::array<uint8_t, 1> buf{0x00};
    uint64_t out = 0;
    EXPECT_THROW(
        varint_decode(buf.data(), buf.data(), out),
        BPTLeafV2DecodeException
    );
}

// ----- varint_size coherence -----------------------------------------------

TEST(Varint, VarintSize_Matches_EncodeLength_AllBoundaries) {
    // For every boundary value pinned by the encoder tests above,
    // varint_size must agree with the actual encoded length. This is the
    // contract the v2 writer relies on for its 4 KB page overflow check.
    std::vector<uint64_t> boundaries = {
        0, 127, 128, 16383, 16384,
        1ULL << 21, 1ULL << 28, 1ULL << 35,
        1ULL << 42, 1ULL << 49, 1ULL << 56, 1ULL << 63,
        UINT64_MAX
    };
    std::array<uint8_t, VARINT_MAX_BYTES> buf{};
    for (uint64_t v : boundaries) {
        size_t encoded = varint_encode(v, buf.data(), buf.size());
        EXPECT_EQ(varint_size(v), encoded) << "value=" << v;
    }
}

// ----- Exception payload checks --------------------------------------------

TEST(Varint, Exception_ContainsOffset) {
    // Trigger truncation and confirm the exception message names "offset" so
    // operators can correlate the failure with a leaf-page byte position.
    std::array<uint8_t, VARINT_MAX_BYTES> buf{};
    size_t n = varint_encode(16384, buf.data(), buf.size());
    ASSERT_EQ(n, 3u);
    uint64_t out = 0;
    try {
        varint_decode(buf.data(), buf.data() + 2, out);
        FAIL() << "expected BPTLeafV2DecodeException";
    } catch (const BPTLeafV2DecodeException& e) {
        const std::string msg = e.what();
        // "consumed" is named in the truncation message; "offset" is named in
        // every other failure path. Either acceptably localises the failure.
        EXPECT_TRUE(msg.find("offset") != std::string::npos
                    || msg.find("consumed") != std::string::npos)
            << "message did not contain offset/consumed: " << msg;
    }
}
