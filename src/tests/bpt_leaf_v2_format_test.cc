// Unit tests for the v2 B+Tree leaf header format (delta + LEB128-varint
// leaf encoding).
//
// Scope: enum identity, header size contract, serialize/deserialize round
// trip, byte-position invariants (format byte is always at offset 0), and
// case-sensitive enum parser including the documented error message format.
// Varint codec / writer / reader behavior is out of scope (covered by the
// varint unit tests and the v2 leaf reader/writer tests).
//
// Design reference: docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "storage/index/bplus_tree/bpt_leaf_format.h"

namespace {

TEST(BPTLeafV2Format, HeaderSizeIs16Bytes) {
    // Compile-time invariant is enforced by static_assert in the header;
    // this test pins the runtime sizeof to the same number to make the
    // contract visible in the test report.
    EXPECT_EQ(sizeof(BPT::BPTLeafV2Header), 16u);
}

TEST(BPTLeafV2Format, HeaderFields_DefaultZero) {
    // A zero-initialized header should have every byte of its underlying
    // storage equal to zero. We copy the struct out byte-by-byte because
    // reading struct fields directly does not exercise padding bytes.
    BPT::BPTLeafV2Header h{};
    uint8_t raw[sizeof(BPT::BPTLeafV2Header)];
    std::memcpy(raw, &h, sizeof(raw));
    for (size_t i = 0; i < sizeof(raw); ++i) {
        EXPECT_EQ(raw[i], 0u) << "byte " << i << " was not zero";
    }
}

TEST(BPTLeafV2Format, HeaderRoundtrip_SerializeDeserialize) {
    BPT::BPTLeafV2Header in;
    in.format_version = 2;
    in.record_width   = 3;
    in.flags          = 0x42;
    in.reserved       = 0;
    in.value_count    = 0x12345678u;
    in.next_leaf      = 0xDEADBEEFu;
    in.reserved2      = 0xCAFEBABEu;

    uint8_t buf[16];
    BPT::serialize_header(in, buf);
    BPT::BPTLeafV2Header out = BPT::deserialize_header(buf);

    EXPECT_EQ(out.format_version, in.format_version);
    EXPECT_EQ(out.record_width,   in.record_width);
    EXPECT_EQ(out.flags,          in.flags);
    EXPECT_EQ(out.reserved,       in.reserved);
    EXPECT_EQ(out.value_count,    in.value_count);
    EXPECT_EQ(out.next_leaf,      in.next_leaf);
    EXPECT_EQ(out.reserved2,      in.reserved2);
}

TEST(BPTLeafV2Format, FormatVersionByteIsFirst) {
    // The dispatch byte must always live at offset 0 so a reader can decide
    // which leaf decoder to invoke after a single byte read.
    BPT::BPTLeafV2Header h{};
    h.format_version = 2;
    uint8_t buf[16];
    BPT::serialize_header(h, buf);
    EXPECT_EQ(buf[0], 0x02);
}

TEST(BPTLeafV2Format, ParseString_Bitset_Returns1) {
    EXPECT_EQ(BPT::parse_leaf_format("BITSET"), BPT::LeafFormat::BITSET);
    EXPECT_EQ(static_cast<uint8_t>(BPT::parse_leaf_format("BITSET")), 1u);
}

TEST(BPTLeafV2Format, ParseString_DeltaVarint_Returns2) {
    EXPECT_EQ(BPT::parse_leaf_format("DELTA_VARINT"),
              BPT::LeafFormat::DELTA_VARINT);
    EXPECT_EQ(static_cast<uint8_t>(BPT::parse_leaf_format("DELTA_VARINT")),
              2u);
}

TEST(BPTLeafV2Format, ParseString_CaseSensitive) {
    // Lower-case, mixed-case and word-boundary variants must all be
    // rejected — the parser is intentionally case-sensitive (design §4.3).
    EXPECT_THROW(BPT::parse_leaf_format("bitset"),       std::invalid_argument);
    EXPECT_THROW(BPT::parse_leaf_format("Delta_Varint"), std::invalid_argument);
    EXPECT_THROW(BPT::parse_leaf_format("DeltaVarint"),  std::invalid_argument);
    EXPECT_THROW(BPT::parse_leaf_format("BitSet"),       std::invalid_argument);
}

TEST(BPTLeafV2Format, ParseString_Unknown_Raises) {
    // Empty input, plausible-looking aliases, and arbitrary garbage must
    // all throw, and the exception message must echo the offending input
    // verbatim so callers can build clear diagnostics.
    const char* bad_inputs[] = { "", "LZ4", "ZSTD", "garbage_!@#" };
    for (const char* input : bad_inputs) {
        try {
            BPT::parse_leaf_format(input);
            FAIL() << "expected std::invalid_argument for input \""
                   << input << "\"";
        } catch (const std::invalid_argument& ex) {
            const std::string msg = ex.what();
            EXPECT_NE(msg.find(input), std::string::npos)
                << "exception message did not contain offending input \""
                << input << "\": " << msg;
        }
    }
}

}  // namespace
