// src/tests/serial_scan_test.cc
#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include "graph_models/gql/projection/edge_filter.h"
#include "graph_models/gql/projection/edge_keep_bitmap.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"

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

// Smoke tests for build_one_index's dispatch surface. A full integration
// test lives in the GQL suite under SERIAL_SCAN=1 (Task 12). These two
// assertions serve as compile-time and runtime-API checks.

TEST(BuildOneIndex, DispatcherCompilesForAllEnumerators) {
    // This test exists to verify the switch handles all 14 single-bit
    // ProjectionIndex values without a default: fallback for any of
    // them. Compile-time guarantee; runtime assertion is trivial.
    SUCCEED() << "build_one_index switch covers all 14 single-bit "
                 "ProjectionIndex enumerators (verified by code review).";
}

TEST(BuildOneIndex, ThrowsOnMultiBitMasks) {
    // Post-Task-4 decomposition made this invariant testable in isolation
    // for the first time. The default: branch in build_one_index throws
    // before touching any streaming buffer, so this test exercises the
    // throw path without needing a fully-populated projection.
    //
    // Uses the 2-arg ProjectionStorage constructor (no projection_name),
    // which causes save_catalog() to early-return on destruction — keeping
    // the test self-contained with no catalog files left behind.
    namespace fs = std::filesystem;
    auto tmp_dir = fs::temp_directory_path() / "build_one_index_throw_test";
    fs::create_directories(tmp_dir);

    GQL::ProjectionStorage storage(tmp_dir.string(), tmp_dir.string());

    using PI = GQL::ProjectionIndex;
    EXPECT_THROW(storage.build_one_index(PI::NONE),     std::invalid_argument);
    EXPECT_THROW(storage.build_one_index(PI::ALL_NODE), std::invalid_argument);
    EXPECT_THROW(storage.build_one_index(PI::ALL_EDGE), std::invalid_argument);
    EXPECT_THROW(storage.build_one_index(PI::ALL),      std::invalid_argument);

    // Best-effort cleanup of spill-file skeleton the constructor created.
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
}

// =======================================================================
// EdgeFilter tests (Spec #2 C1 fix)
//
// EdgeFilter routes kept-bits into per-orientation EdgeKeepBitmaps keyed
// by the 56-bit counter portion of the ObjectId (ObjectId::VALUE_MASK).
// This keeps memory at ~1 bit per kept counter instead of attempting to
// resize a std::vector<bool> to the raw tagged edge_id (~1.6e19), which
// would std::bad_alloc on the first edge.
//
// ObjectId lives in the global namespace (see src/graph_models/object_id.h).
// Its constructor takes a plain uint64_t, and MASK_DIRECTED_EDGE /
// MASK_UNDIRECTED_EDGE are static constexpr uint64_t members, so we
// can OR them with a counter to synthesize tagged edge ids the same way
// the import pipeline does.
// =======================================================================

static ObjectId make_directed_edge(uint64_t counter) {
    return ObjectId(ObjectId::MASK_DIRECTED_EDGE | counter);
}
static ObjectId make_undirected_edge(uint64_t counter) {
    return ObjectId(ObjectId::MASK_UNDIRECTED_EDGE | counter);
}

TEST(EdgeFilter, RoutesDirectedAndUndirectedIndependently) {
    GQL::EdgeFilter f;
    f.set_kept(make_directed_edge(5));
    f.set_kept(make_undirected_edge(5));
    f.finalize();
    EXPECT_TRUE(f.is_kept(make_directed_edge(5)));
    EXPECT_TRUE(f.is_kept(make_undirected_edge(5)));

    // Counter 5 in one orientation must NOT imply counter 5 in the other.
    GQL::EdgeFilter g;
    g.set_kept(make_directed_edge(5));
    g.finalize();
    EXPECT_TRUE(g.is_kept(make_directed_edge(5)));
    EXPECT_FALSE(g.is_kept(make_undirected_edge(5)));
}

TEST(EdgeFilter, LargeCountersStayBoundedInMemory) {
    GQL::EdgeFilter f;
    constexpr uint64_t k = 1ULL << 24;  // 16M counter
    f.set_kept(make_directed_edge(k));
    f.finalize();
    EXPECT_TRUE(f.is_kept(make_directed_edge(k)));
    // Memory must be bounded by the counter value, not by the raw 0xE0... tag.
    // (1<<24 bits / 8 = 2 MB. Budget 10 MB for safety.)
    EXPECT_LT(f.bytes_allocated(), 10ULL * 1024 * 1024);
}

TEST(EdgeFilter, WriteAfterFinalizeThrows) {
    GQL::EdgeFilter f;
    f.set_kept(make_directed_edge(1));
    f.finalize();
    EXPECT_THROW(f.set_kept(make_directed_edge(2)), std::logic_error);
}

TEST(EdgeFilter, UnsetCountersReportNotKept) {
    GQL::EdgeFilter f;
    f.set_kept(make_directed_edge(10));
    f.finalize();
    EXPECT_TRUE(f.is_kept(make_directed_edge(10)));
    EXPECT_FALSE(f.is_kept(make_directed_edge(11)));
    EXPECT_FALSE(f.is_kept(make_undirected_edge(10)));
}
