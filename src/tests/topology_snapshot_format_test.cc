// Unit tests for the TopologySnapshot on-disk file format (§5.1, §5.2).
//
// Scope: header round-trip, magic / version / id_width validation, flag
// bit preservation, and `sizeof == 64` compile-time contract. File I/O,
// mmap, and SHA-256 are covered by the topology snapshot writer and reader tests.
//
// Spec reference: docs/superpowers/specs/2026-04-25-topology-snapshot-design.md

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/topology_snapshot.h"

using GQL::Projection::kTopologySnapshotHeaderSize;
using GQL::Projection::kTopologySnapshotIdWidth;
using GQL::Projection::kTopologySnapshotIdWidthNarrow;
using GQL::Projection::kTopologySnapshotMagic;
using GQL::Projection::kTopologySnapshotVersion;
using GQL::Projection::make_default_topology_snapshot_header;
using GQL::Projection::parse_topology_snapshot_header;
using GQL::Projection::serialize_topology_snapshot_header;
using GQL::Projection::TopologySnapshotFormatError;
using GQL::Projection::TopologySnapshotHeader;
namespace Flags = GQL::Projection::TopologySnapshotFlags;

namespace {

// Builds a fully populated header with sentinel values in every field. Used
// by the round-trip test to ensure no field is silently dropped or swapped
// during serialize → parse.
TopologySnapshotHeader make_sentinel_header() {
    TopologySnapshotHeader h = make_default_topology_snapshot_header();
    h.flags        = Flags::kHasEdgeIds;
    h.dst_type_tag = 0x00;   // id_width==8 -> tag bytes stay zero
    h.edge_type_tag = 0x00;
    h.num_nodes    = 0x0123456789ABCDEFULL;
    h.num_edges    = 0xFEDCBA9876543210ULL;
    for (std::size_t i = 0; i < 32; ++i) {
        h.source_sha256[i] = static_cast<uint8_t>(0x80 + i);
    }
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------
// sizeof(TopologySnapshotHeader) must be exactly 64 bytes.
//
// The header struct is the shared contract between the topology snapshot writer and
// reader. Any change that alters its size silently breaks
// cross-build compatibility; catch it at compile time AND at test time.
// ---------------------------------------------------------------------------
TEST(TopologySnapshotFormat, SizeofIs64Bytes) {
    static_assert(sizeof(TopologySnapshotHeader) == 64,
                  "TopologySnapshotHeader must be 64 bytes");
    EXPECT_EQ(sizeof(TopologySnapshotHeader), kTopologySnapshotHeaderSize);
    EXPECT_EQ(kTopologySnapshotHeaderSize, 64u);
}

// ---------------------------------------------------------------------------
// Round-trip: every field written into the sentinel header survives a
// serialize → parse cycle byte-for-byte.
// ---------------------------------------------------------------------------
TEST(TopologySnapshotFormat, HeaderRoundTripPreservesFields) {
    const TopologySnapshotHeader src = make_sentinel_header();

    uint8_t buf[kTopologySnapshotHeaderSize] = {};
    serialize_topology_snapshot_header(src, buf);

    const TopologySnapshotHeader dst = parse_topology_snapshot_header(buf);

    EXPECT_EQ(0, std::memcmp(dst.magic,
                             kTopologySnapshotMagic.data(),
                             kTopologySnapshotMagic.size()));
    EXPECT_EQ(dst.version, kTopologySnapshotVersion);
    EXPECT_EQ(dst.id_width, kTopologySnapshotIdWidth);
    EXPECT_EQ(dst.flags, Flags::kHasEdgeIds);
    EXPECT_EQ(dst.dst_type_tag, 0);
    EXPECT_EQ(dst.edge_type_tag, 0);
    EXPECT_EQ(dst.num_nodes, src.num_nodes);
    EXPECT_EQ(dst.num_edges, src.num_edges);
    EXPECT_EQ(0, std::memcmp(dst.source_sha256, src.source_sha256, 32));
}

// ---------------------------------------------------------------------------
// Magic byte corruption is rejected by the parser.
//
// Policy: parse_topology_snapshot_header throws TopologySnapshotFormatError
// on any failure (the reader wraps this to set has_data() = false, per
// §3.4 fallback-first architecture).
// ---------------------------------------------------------------------------
TEST(TopologySnapshotFormat, BadMagicRejected) {
    const TopologySnapshotHeader src = make_sentinel_header();
    uint8_t buf[kTopologySnapshotHeaderSize] = {};
    serialize_topology_snapshot_header(src, buf);

    // Flip a single magic byte — e.g. 'T' (0x54) → 'X' (0x58).
    buf[0] = 'X';
    EXPECT_THROW(parse_topology_snapshot_header(buf),
                 TopologySnapshotFormatError);

    // And a completely zeroed magic.
    std::memset(buf, 0, 8);
    EXPECT_THROW(parse_topology_snapshot_header(buf),
                 TopologySnapshotFormatError);
}

// ---------------------------------------------------------------------------
// Version 0 and version 2 are rejected. Version is monotonic; v0 is the
// obvious "uninitialized zero block" mistake and v2 is a future format we
// haven't shipped yet.
// ---------------------------------------------------------------------------
TEST(TopologySnapshotFormat, BadVersionRejected) {
    TopologySnapshotHeader src = make_sentinel_header();
    uint8_t buf[kTopologySnapshotHeaderSize] = {};

    src.version = 0;
    serialize_topology_snapshot_header(src, buf);
    EXPECT_THROW(parse_topology_snapshot_header(buf),
                 TopologySnapshotFormatError);

    src.version = 2;
    serialize_topology_snapshot_header(src, buf);
    EXPECT_THROW(parse_topology_snapshot_header(buf),
                 TopologySnapshotFormatError);

    // Sanity: version = 1 does parse successfully. This guards against the
    // test accidentally testing "everything fails".
    src.version = kTopologySnapshotVersion;
    serialize_topology_snapshot_header(src, buf);
    EXPECT_NO_THROW(parse_topology_snapshot_header(buf));
}

// ---------------------------------------------------------------------------
// id_width: 8 (full 64-bit ObjectId) and 4 (tag-stripped uint32, which strips the
// 8-bit ObjectId type tag and reconstructs it on read, halving topology size) are
// valid; every other value is rejected. Catches off-by-one bugs in the validator.
// ---------------------------------------------------------------------------
TEST(TopologySnapshotFormat, BadIdWidthRejected) {
    TopologySnapshotHeader src = make_sentinel_header();
    uint8_t buf[kTopologySnapshotHeaderSize] = {};

    // id_width 4: tag-stripped uint32 encoding (strips the 8-bit ObjectId type tag,
    // reconstructs on read, halves topology sidecar size) is a VALID width.
    src.id_width = kTopologySnapshotIdWidthNarrow;  // 4
    serialize_topology_snapshot_header(src, buf);
    EXPECT_NO_THROW(parse_topology_snapshot_header(buf));

    // id_width 8 (full ObjectId) round-trips cleanly.
    src.id_width = kTopologySnapshotIdWidth;  // 8
    serialize_topology_snapshot_header(src, buf);
    EXPECT_NO_THROW(parse_topology_snapshot_header(buf));

    // Any other width is rejected.
    for (uint8_t bad : {uint8_t{0}, uint8_t{1}, uint8_t{2}, uint8_t{16}, uint8_t{255}}) {
        src.id_width = bad;
        serialize_topology_snapshot_header(src, buf);
        EXPECT_THROW(parse_topology_snapshot_header(buf),
                     TopologySnapshotFormatError) << "width=" << static_cast<int>(bad);
    }
}

// ---------------------------------------------------------------------------
// Flags round-trip: kHasEdgeIds must survive; currently-reserved upper
// bits are preserved (forward-compat) so a future MillenniumDB can set
// them and an older reader in the middle of a pipeline chain doesn't drop
// them.
// ---------------------------------------------------------------------------
TEST(TopologySnapshotFormat, FlagsBitfieldCorrect) {
    TopologySnapshotHeader src = make_sentinel_header();
    uint8_t buf[kTopologySnapshotHeaderSize] = {};

    // kHasEdgeIds = 0x01 round-trip.
    src.flags = Flags::kHasEdgeIds;
    serialize_topology_snapshot_header(src, buf);
    TopologySnapshotHeader dst = parse_topology_snapshot_header(buf);
    EXPECT_TRUE(dst.flags & Flags::kHasEdgeIds);
    EXPECT_EQ(dst.flags, static_cast<uint8_t>(Flags::kHasEdgeIds));

    // Flag cleared round-trip.
    src.flags = 0;
    serialize_topology_snapshot_header(src, buf);
    dst = parse_topology_snapshot_header(buf);
    EXPECT_FALSE(dst.flags & Flags::kHasEdgeIds);
    EXPECT_EQ(dst.flags, 0u);

    // Reserved upper bits are preserved unchanged (e.g. bit 7 set by a
    // future format revision must reach the reader intact).
    src.flags = 0x80 | Flags::kHasEdgeIds;
    serialize_topology_snapshot_header(src, buf);
    dst = parse_topology_snapshot_header(buf);
    EXPECT_EQ(dst.flags, static_cast<uint8_t>(0x80 | Flags::kHasEdgeIds));
}

// ---------------------------------------------------------------------------
// Little-endian byte layout of num_nodes / num_edges.
//
// MillenniumDB only targets little-endian platforms (x86_64, AArch64), so
// std::memcpy suffices for serialization — but this test pins the layout
// so a future big-endian port catches the assumption early.
// ---------------------------------------------------------------------------
TEST(TopologySnapshotFormat, MultiByteFieldsAreLittleEndian) {
    TopologySnapshotHeader src = make_default_topology_snapshot_header();
    src.num_nodes = 0x0123456789ABCDEFULL;
    src.num_edges = 0xFEDCBA9876543210ULL;

    uint8_t buf[kTopologySnapshotHeaderSize] = {};
    serialize_topology_snapshot_header(src, buf);

    // num_nodes at offset 16, little-endian: lowest byte first.
    EXPECT_EQ(buf[16], 0xEFu);
    EXPECT_EQ(buf[17], 0xCDu);
    EXPECT_EQ(buf[18], 0xABu);
    EXPECT_EQ(buf[19], 0x89u);
    EXPECT_EQ(buf[20], 0x67u);
    EXPECT_EQ(buf[21], 0x45u);
    EXPECT_EQ(buf[22], 0x23u);
    EXPECT_EQ(buf[23], 0x01u);

    // num_edges at offset 24, little-endian.
    EXPECT_EQ(buf[24], 0x10u);
    EXPECT_EQ(buf[25], 0x32u);
    EXPECT_EQ(buf[26], 0x54u);
    EXPECT_EQ(buf[27], 0x76u);
    EXPECT_EQ(buf[28], 0x98u);
    EXPECT_EQ(buf[29], 0xBAu);
    EXPECT_EQ(buf[30], 0xDCu);
    EXPECT_EQ(buf[31], 0xFEu);

    // version at offset 8 (uint32 LE).
    EXPECT_EQ(buf[8], 0x01u);
    EXPECT_EQ(buf[9], 0x00u);
    EXPECT_EQ(buf[10], 0x00u);
    EXPECT_EQ(buf[11], 0x00u);

    // id_width at offset 12; flags at offset 13.
    EXPECT_EQ(buf[12], kTopologySnapshotIdWidth);
}

// ---------------------------------------------------------------------------
// make_default_topology_snapshot_header() produces a parseable header.
//
// Anchors the zero-filled reserved fields and demonstrates the intended
// writer workflow: default-construct → set payload fields → serialize.
// ---------------------------------------------------------------------------
TEST(TopologySnapshotFormat, DefaultHeaderIsParseable) {
    const TopologySnapshotHeader src = make_default_topology_snapshot_header();

    uint8_t buf[kTopologySnapshotHeaderSize] = {};
    serialize_topology_snapshot_header(src, buf);

    TopologySnapshotHeader dst{};
    EXPECT_NO_THROW(dst = parse_topology_snapshot_header(buf));

    EXPECT_EQ(0, std::memcmp(dst.magic,
                             kTopologySnapshotMagic.data(),
                             kTopologySnapshotMagic.size()));
    EXPECT_EQ(dst.version, kTopologySnapshotVersion);
    EXPECT_EQ(dst.id_width, kTopologySnapshotIdWidth);
    EXPECT_EQ(dst.flags, 0u);
    EXPECT_EQ(dst.num_nodes, 0u);
    EXPECT_EQ(dst.num_edges, 0u);
    EXPECT_EQ(dst.dst_type_tag, 0);
    EXPECT_EQ(dst.edge_type_tag, 0);
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_EQ(dst.source_sha256[i], 0u);
    }
}
