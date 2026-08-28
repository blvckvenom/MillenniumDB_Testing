// Unit tests for the v3 CSR_HYBRID B+Tree leaf header format.
//
// CSR_HYBRID is a graph storage mode in which the edge-index B+Tree leaves
// ARE the CSR layout: each leaf page carries a 16-byte header followed by
// a delta + LEB128-varint-encoded destination stream, so neighbor access
// for GNN sampling is O(1) per page rather than O(log N) per lookup.
//
// Scope: enum identity, header size contract, serialize/deserialize round
// trip for both the chain-head and continuation views, byte-position
// invariants (format byte at offset 0, flags byte at offset 2), and the
// is_csr_continuation flag-sniff helper. Reader/writer behavior and the
// page-level dispatch are covered by the reader and writer test suites.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "storage/index/bplus_tree/bpt_leaf_csr_format.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"

namespace {

TEST(BPTLeafCSRFormat, ChainHeaderSizeIs16Bytes) {
    // Compile-time invariant is enforced by static_assert in the header;
    // pin it at runtime as well so the contract surfaces in the test report.
    EXPECT_EQ(sizeof(BPT::BPTLeafCSRHeader), 16u);
}

TEST(BPTLeafCSRFormat, ContinuationHeaderSizeIs16Bytes) {
    // The two views of the v3 header MUST share an identical 16-byte
    // footprint so the page-open dispatch can read 16 bytes once and then
    // sniff the flags byte to choose between them.
    EXPECT_EQ(sizeof(BPT::BPTLeafCSRContinuationHeader), 16u);
}

TEST(BPTLeafCSRFormat, HeaderFields_DefaultZero) {
    // Zero-initialized headers must have every byte of their underlying
    // storage equal to zero. Walk the raw bytes — reading struct fields
    // does not exercise padding bytes.
    BPT::BPTLeafCSRHeader chain{};
    uint8_t chain_raw[sizeof(BPT::BPTLeafCSRHeader)];
    std::memcpy(chain_raw, &chain, sizeof(chain_raw));
    for (size_t i = 0; i < sizeof(chain_raw); ++i) {
        EXPECT_EQ(chain_raw[i], 0u)
            << "chain byte " << i << " was not zero";
    }

    BPT::BPTLeafCSRContinuationHeader cont{};
    uint8_t cont_raw[sizeof(BPT::BPTLeafCSRContinuationHeader)];
    std::memcpy(cont_raw, &cont, sizeof(cont_raw));
    for (size_t i = 0; i < sizeof(cont_raw); ++i) {
        EXPECT_EQ(cont_raw[i], 0u)
            << "continuation byte " << i << " was not zero";
    }
}

TEST(BPTLeafCSRFormat, ChainHeader_Roundtrip_AllFieldsPreserved) {
    BPT::BPTLeafCSRHeader in;
    in.format_version = 3;
    in.record_width   = 3;
    // flags has bit 1 (has_edge_ids) set, bit 0 (continuation) clear so
    // the resulting buffer is unambiguously a chain head.
    in.flags          = BPT::CSRHybridFlags::kHasEdgeIds;
    in.reserved       = 0;
    in.value_count    = 0x12345678u;
    in.next_leaf      = 0xDEADBEEFu;
    in.min_src_id_low = 0xCAFEBABEu;

    uint8_t buf[16];
    BPT::serialize_csr_header(in, buf);
    BPT::BPTLeafCSRHeader out = BPT::deserialize_csr_header(buf);

    EXPECT_EQ(out.format_version,  in.format_version);
    EXPECT_EQ(out.record_width,    in.record_width);
    EXPECT_EQ(out.flags,           in.flags);
    EXPECT_EQ(out.reserved,        in.reserved);
    EXPECT_EQ(out.value_count,     in.value_count);
    EXPECT_EQ(out.next_leaf,       in.next_leaf);
    EXPECT_EQ(out.min_src_id_low,  in.min_src_id_low);
}

TEST(BPTLeafCSRFormat, ContinuationHeader_Roundtrip_AllFieldsPreserved) {
    BPT::BPTLeafCSRContinuationHeader in;
    in.format_version       = 3;
    in.record_width         = 3;
    // flags MUST have bit 0 (continuation) set so the buffer round-trips
    // through the continuation view path and not the chain-head one.
    in.flags                = BPT::CSRHybridFlags::kIsContinuation;
    in.reserved             = 0;
    in.chunk_count          = 0xAABBCCDDu;
    in.next_leaf            = 0x11223344u;
    in.chain_head_page_id   = 0x55667788u;

    uint8_t buf[16];
    BPT::serialize_csr_continuation_header(in, buf);
    BPT::BPTLeafCSRContinuationHeader out =
        BPT::deserialize_csr_continuation_header(buf);

    EXPECT_EQ(out.format_version,      in.format_version);
    EXPECT_EQ(out.record_width,        in.record_width);
    EXPECT_EQ(out.flags,               in.flags);
    EXPECT_EQ(out.reserved,            in.reserved);
    EXPECT_EQ(out.chunk_count,         in.chunk_count);
    EXPECT_EQ(out.next_leaf,           in.next_leaf);
    EXPECT_EQ(out.chain_head_page_id,  in.chain_head_page_id);
}

TEST(BPTLeafCSRFormat, FormatVersionByteIsFirst_BothVariants) {
    // The dispatch byte must always live at offset 0 so a reader can
    // decide which leaf decoder to invoke after a single byte read,
    // regardless of which v3 variant the page carries.
    BPT::BPTLeafCSRHeader chain{};
    chain.format_version = 3;
    uint8_t chain_buf[16];
    BPT::serialize_csr_header(chain, chain_buf);
    EXPECT_EQ(chain_buf[0], 0x03);

    BPT::BPTLeafCSRContinuationHeader cont{};
    cont.format_version = 3;
    cont.flags          = BPT::CSRHybridFlags::kIsContinuation;
    uint8_t cont_buf[16];
    BPT::serialize_csr_continuation_header(cont, cont_buf);
    EXPECT_EQ(cont_buf[0], 0x03);
}

TEST(BPTLeafCSRFormat, IsCSRContinuation_Detects_Flag) {
    // Hand-craft a 16-byte buffer whose flags byte is the only thing that
    // changes; verify the sniff returns the expected bool. This pins the
    // dispatch contract used by the page-level CSR leaf dispatcher.
    uint8_t buf[16] = {};
    buf[0] = 3;  // format_version
    buf[1] = 3;  // record_width

    // Chain-head page: flags bit 0 clear.
    buf[2] = 0x00;
    EXPECT_FALSE(BPT::is_csr_continuation(buf));
    buf[2] = BPT::CSRHybridFlags::kHasEdgeIds;  // bit 1 only — still chain head
    EXPECT_FALSE(BPT::is_csr_continuation(buf));

    // Continuation page: flags bit 0 set, with and without other bits.
    buf[2] = BPT::CSRHybridFlags::kIsContinuation;
    EXPECT_TRUE(BPT::is_csr_continuation(buf));
    buf[2] = static_cast<uint8_t>(BPT::CSRHybridFlags::kIsContinuation |
                                  BPT::CSRHybridFlags::kHasEdgeIds);
    EXPECT_TRUE(BPT::is_csr_continuation(buf));
}

TEST(BPTLeafCSRFormat, ParseLeafFormat_CSR_HYBRID_Returns3) {
    // The CSR_HYBRID enum value was added alongside this format header —
    // verify the catalog parser accepts the new literal and yields the right
    // numeric value. The earlier BITSET and DELTA_VARINT leaf encodings
    // (delta + LEB128-varint compression) remain unchanged at their existing
    // numeric values.
    EXPECT_EQ(BPT::parse_leaf_format("CSR_HYBRID"),
              BPT::LeafFormat::CSR_HYBRID);
    EXPECT_EQ(static_cast<uint8_t>(BPT::parse_leaf_format("CSR_HYBRID")),
              3u);
}

TEST(BPTLeafCSRFormat, LeafFormatToString_CSR_HYBRID_Returns_Literal) {
    // Round-trippable name is required for catalog dump / debug output.
    const char* name =
        BPT::leaf_format_to_string(BPT::LeafFormat::CSR_HYBRID);
    EXPECT_STREQ(name, "CSR_HYBRID");
}

TEST(BPTLeafCSRFormat, ParseLeafFormat_CaseSensitive_RejectsLowercase) {
    // The parser is case-sensitive by design, consistent with the BITSET and
    // DELTA_VARINT literals; lowercased or mixed-case variants of CSR_HYBRID
    // must throw, mirroring the behaviour already pinned for the earlier
    // leaf-format names.
    EXPECT_THROW(BPT::parse_leaf_format("csr_hybrid"),
                 std::invalid_argument);
    EXPECT_THROW(BPT::parse_leaf_format("Csr_Hybrid"),
                 std::invalid_argument);
    EXPECT_THROW(BPT::parse_leaf_format("CSRHybrid"),
                 std::invalid_argument);
}

TEST(BPTLeafCSRFormat, ChainHeader_FlagsByteIsAtOffset2) {
    // The page-open dispatcher reads byte 2 to decide chain-head vs
    // continuation. Pin the offset so any future struct reordering is
    // caught in this format-only test rather than at integration time.
    BPT::BPTLeafCSRHeader h{};
    h.format_version = 3;
    h.record_width   = 3;
    h.flags          = 0xA5;  // distinctive non-zero, non-flag-meaningful pattern
    h.reserved       = 0;

    uint8_t buf[16];
    BPT::serialize_csr_header(h, buf);
    EXPECT_EQ(buf[2], 0xA5u);

    // Round-trip preserves the byte even though kReservedMask bits are set;
    // semantic validation belongs to the CSR leaf reader, not the format layer.
    BPT::BPTLeafCSRHeader out = BPT::deserialize_csr_header(buf);
    EXPECT_EQ(out.flags, 0xA5u);
}

TEST(BPTLeafCSRFormat, FlagConstants_HaveExpectedBitPositions) {
    // Pin the wire-level bit positions — the CSR leaf reader, writer, and
    // page-level dispatcher all depend on these values. A future reshuffle
    // must update both the constants and this test deliberately.
    EXPECT_EQ(BPT::CSRHybridFlags::kIsContinuation,  0x01u);
    EXPECT_EQ(BPT::CSRHybridFlags::kHasEdgeIds,      0x02u);
    EXPECT_EQ(BPT::CSRHybridFlags::kIsHubChainHead,  0x04u);
    EXPECT_EQ(BPT::CSRHybridFlags::kReservedMask,    0xF8u);
    // The four masks together must cover every bit exactly once: the three
    // defined flags plus the reserved mask span the full byte.
    constexpr uint8_t covered = BPT::CSRHybridFlags::kIsContinuation |
                                BPT::CSRHybridFlags::kHasEdgeIds |
                                BPT::CSRHybridFlags::kIsHubChainHead |
                                BPT::CSRHybridFlags::kReservedMask;
    EXPECT_EQ(covered, 0xFFu);
}

}  // namespace
