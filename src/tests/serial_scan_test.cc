// src/tests/serial_scan_test.cc
#include <gtest/gtest.h>
#include <stdexcept>
#include "graph_models/gql/projection/edge_keep_bitmap.h"

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
