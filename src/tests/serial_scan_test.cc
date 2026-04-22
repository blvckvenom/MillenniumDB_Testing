// src/tests/serial_scan_test.cc
#include <gtest/gtest.h>
#include <stdexcept>
#include "graph_models/gql/projection/edge_keep_bitmap.h"
#include "graph_models/gql/projection/native_projection_builder.h"

TEST(EdgeKeepBitmap, SetAndQuery) {
    GQL::EdgeKeepBitmap bm;
    bm.set_kept(5);
    bm.set_kept(1000);
    bm.finalize();
    EXPECT_TRUE(bm.is_kept(5));
    EXPECT_TRUE(bm.is_kept(1000));
    EXPECT_FALSE(bm.is_kept(0));
    EXPECT_FALSE(bm.is_kept(999));
    EXPECT_FALSE(bm.is_kept(10000));  // beyond size returns false
}

TEST(EdgeKeepBitmap, WriteAfterFinalizeThrows) {
    GQL::EdgeKeepBitmap bm;
    bm.set_kept(42);
    bm.finalize();
    EXPECT_THROW(bm.set_kept(100), std::logic_error);
}

TEST(EdgeKeepBitmap, AutoGrowsOnHighIndex) {
    GQL::EdgeKeepBitmap bm;
    bm.set_kept(1ULL << 20);  // 1 million
    EXPECT_TRUE(bm.is_kept(1ULL << 20));
    EXPECT_GE(bm.bytes_allocated(), ((1ULL << 20) + 1) / 8);
}

TEST(EdgeKeepBitmap, ReserveDoesNotMarkKept) {
    GQL::EdgeKeepBitmap bm;
    bm.reserve(1000);
    bm.finalize();
    for (std::uint64_t i = 0; i < 1000; ++i) {
        EXPECT_FALSE(bm.is_kept(i));
    }
}

TEST(EdgeKeepBitmap, SizeReflectsHighestIndex) {
    GQL::EdgeKeepBitmap bm;
    bm.set_kept(7);
    EXPECT_GE(bm.size(), 8u);  // 0..7
}

TEST(EdgeKeepBitmap, FinalizeIsIdempotent) {
    GQL::EdgeKeepBitmap bm;
    bm.set_kept(3);
    bm.finalize();
    bm.finalize();  // second call is a no-op
    EXPECT_TRUE(bm.is_kept(3));
    EXPECT_THROW(bm.set_kept(100), std::logic_error);
}

TEST(ProjectionIndex, BitmaskOperations) {
    using PI = GQL::ProjectionIndex;
    EXPECT_TRUE(GQL::has_flag(PI::ALL_NODE, PI::NODES));
    EXPECT_FALSE(GQL::has_flag(PI::ALL_NODE, PI::FROM_TO_EDGE));
    auto combined = PI::NODES | PI::NODE_LABEL;
    EXPECT_EQ(static_cast<uint32_t>(combined), 0x3u);
    EXPECT_EQ(PI::ALL & PI::ALL_NODE, PI::ALL_NODE);
}

TEST(ProjectionIndex, AllIsUnionOfNodeAndEdge) {
    using PI = GQL::ProjectionIndex;
    EXPECT_EQ(static_cast<uint32_t>(PI::ALL), 0x3FFFu);  // 14 bits set
    EXPECT_EQ(PI::ALL_NODE | PI::ALL_EDGE, PI::ALL);
}

TEST(ProjectionIndex, SingleBitDetection) {
    using PI = GQL::ProjectionIndex;
    EXPECT_TRUE(GQL::has_flag(PI::EDGE_DIRECTION, PI::EDGE_DIRECTION));
    EXPECT_FALSE(GQL::has_flag(PI::EDGE_DIRECTION, PI::EDGE_LABEL));
}

TEST(ScanMode, NullEnvReturnsClassic) {
    using SM = GQL::NativeProjectionBuilder::ScanMode;
    EXPECT_EQ(GQL::detail::init_scan_mode_for_test(nullptr), SM::CLASSIC);
}

TEST(ScanMode, TruthyValuesEnableSerial) {
    using SM = GQL::NativeProjectionBuilder::ScanMode;
    EXPECT_EQ(GQL::detail::init_scan_mode_for_test("1"), SM::SERIALIZED);
    EXPECT_EQ(GQL::detail::init_scan_mode_for_test("true"), SM::SERIALIZED);
    EXPECT_EQ(GQL::detail::init_scan_mode_for_test("yes"), SM::SERIALIZED);
}

TEST(ScanMode, UnknownValuesFallbackToClassic) {
    using SM = GQL::NativeProjectionBuilder::ScanMode;
    EXPECT_EQ(GQL::detail::init_scan_mode_for_test("0"), SM::CLASSIC);
    EXPECT_EQ(GQL::detail::init_scan_mode_for_test("garbage"), SM::CLASSIC);
    EXPECT_EQ(GQL::detail::init_scan_mode_for_test(""), SM::CLASSIC);
}
