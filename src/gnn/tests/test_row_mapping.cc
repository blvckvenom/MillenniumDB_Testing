#include "test_helpers.h"

#include <chrono>

#include "gnn/storage/row_mapping.h"
#include "graph_models/object_id.h"

// Fixture alias — keeps test names as RowMappingTest.* in output
using RowMappingTest = GnnStorageTest;

// ===========================================================================
// Roundtrip
// ===========================================================================

TEST_F(RowMappingTest, CreateAndOpenRoundtrip) {
    std::vector<ObjectId> ids;
    for (uint64_t i = 0; i < 100; ++i) {
        ids.push_back(ObjectId(i * 7 + 3));
    }

    auto path = test_path("mapping.rmap");
    auto rm_write = RowMapping::create(path, ids);
    EXPECT_EQ(rm_write.size(), 100u);

    auto rm_read = RowMapping::open(path);
    EXPECT_EQ(rm_read.size(), 100u);

    for (uint64_t i = 0; i < 100; ++i) {
        EXPECT_EQ(rm_read.get(i).id, ids[i].id);
    }
}

TEST_F(RowMappingTest, Find) {
    std::vector<ObjectId> ids = {ObjectId(42), ObjectId(99), ObjectId(7)};
    auto rm = RowMapping::create(test_path("find.rmap"), ids);

    auto idx = rm.find(ObjectId(99));
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(idx.value(), 1u);

    auto miss = rm.find(ObjectId(12345));
    EXPECT_FALSE(miss.has_value());
}

TEST_F(RowMappingTest, FindFirstOccurrence) {
    std::vector<ObjectId> ids = {ObjectId(5), ObjectId(10), ObjectId(5), ObjectId(10)};
    auto rm = RowMapping::create(test_path("dup_find.rmap"), ids);

    auto idx = rm.find(ObjectId(5));
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(idx.value(), 0u);

    auto idx2 = rm.find(ObjectId(10));
    ASSERT_TRUE(idx2.has_value());
    EXPECT_EQ(idx2.value(), 1u);
}

// ===========================================================================
// Empty
// ===========================================================================

TEST_F(RowMappingTest, Empty) {
    std::vector<ObjectId> ids;
    auto rm = RowMapping::create(test_path("empty.rmap"), ids);
    EXPECT_EQ(rm.size(), 0u);
    EXPECT_THROW(rm.get(0), std::out_of_range);
    EXPECT_FALSE(rm.find(ObjectId(1)).has_value());
}

// ===========================================================================
// Persistence
// ===========================================================================

TEST_F(RowMappingTest, PersistsAcrossOpenClose) {
    std::vector<ObjectId> ids = {ObjectId(100), ObjectId(200)};
    auto path = test_path("persist.rmap");

    { auto rm = RowMapping::create(path, ids); }

    auto rm = RowMapping::open(path);
    EXPECT_EQ(rm.size(), 2u);
    EXPECT_EQ(rm.get(0).id, 100u);
    EXPECT_EQ(rm.get(1).id, 200u);
}

// ===========================================================================
// Error Paths
// ===========================================================================

TEST_F(RowMappingTest, OpenNonExistentThrows) {
    EXPECT_THROW(RowMapping::open(test_path("nope.rmap")), std::runtime_error);
}

TEST_F(RowMappingTest, OutOfBoundsThrows) {
    std::vector<ObjectId> ids = {ObjectId(1)};
    auto rm = RowMapping::create(test_path("oob.rmap"), ids);
    EXPECT_THROW(rm.get(1), std::out_of_range);
    EXPECT_THROW(rm.get(999), std::out_of_range);
    EXPECT_NO_THROW(rm.get(0));
}

TEST_F(RowMappingTest, CorruptedHeaderThrows) {
    auto path = test_path("corrupt.rmap");
    std::vector<char> garbage(32, static_cast<char>(0xFF));
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(garbage.data(), garbage.size());
    }
    EXPECT_THROW(RowMapping::open(path), std::runtime_error);
}

TEST_F(RowMappingTest, TruncatedFileThrows) {
    auto path = test_path("truncated.rmap");

    std::vector<ObjectId> ids;
    for (uint64_t i = 0; i < 10; ++i) ids.push_back(ObjectId(i));
    { auto rm = RowMapping::create(path, ids); }

    fs::resize_file(path, RowMapping::HEADER_SIZE + 2 * sizeof(ObjectId));

    EXPECT_THROW(RowMapping::open(path), std::runtime_error);
}

TEST_F(RowMappingTest, FileTooSmallForHeaderThrows) {
    auto path = test_path("tiny.rmap");
    std::ofstream ofs(path, std::ios::binary);
    char byte = 0;
    ofs.write(&byte, 1);
    ofs.close();

    EXPECT_THROW(RowMapping::open(path), std::runtime_error);
}

// ===========================================================================
// Move Semantics
// ===========================================================================

TEST_F(RowMappingTest, MovedFromThrowsOnAccess) {
    std::vector<ObjectId> ids = {ObjectId(42)};
    auto rm1 = RowMapping::create(test_path("move.rmap"), ids);
    auto rm2 = std::move(rm1);

    EXPECT_EQ(rm2.size(), 1u);
    EXPECT_EQ(rm2.get(0).id, 42u);

    EXPECT_EQ(rm1.size(), 0u);
    EXPECT_THROW(rm1.get(0), std::runtime_error);
}

TEST_F(RowMappingTest, MoveAssignReleasesOldMapping) {
    std::vector<ObjectId> ids_a = {ObjectId(10), ObjectId(20)};
    std::vector<ObjectId> ids_b = {ObjectId(30), ObjectId(40), ObjectId(50)};

    auto rm = RowMapping::create(test_path("assign_a.rmap"), ids_a);
    EXPECT_EQ(rm.size(), 2u);
    EXPECT_EQ(rm.get(0).id, 10u);

    rm = RowMapping::create(test_path("assign_b.rmap"), ids_b);
    EXPECT_EQ(rm.size(), 3u);
    EXPECT_EQ(rm.get(0).id, 30u);
}

// ===========================================================================
// Forward Compatibility
// ===========================================================================

TEST_F(RowMappingTest, OpenWithExtraTrailingBytesSucceeds) {
    std::vector<ObjectId> ids = {ObjectId(1), ObjectId(2)};
    auto path = test_path("extra.rmap");
    { auto rm = RowMapping::create(path, ids); }

    {
        std::ofstream ofs(path, std::ios::binary | std::ios::app);
        char padding[64] = {};
        ofs.write(padding, sizeof(padding));
    }

    auto rm = RowMapping::open(path);
    EXPECT_EQ(rm.size(), 2u);
    EXPECT_EQ(rm.get(0).id, 1u);
    EXPECT_EQ(rm.get(1).id, 2u);
}

// ===========================================================================
// Scale
// ===========================================================================

TEST_F(RowMappingTest, LargeVerifyAll) {
    const uint64_t N = 10000;
    std::vector<ObjectId> ids;
    ids.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        ids.push_back(ObjectId(i * 13 + 7));
    }

    auto path = test_path("large_verify_all.rmap");
    { auto rm = RowMapping::create(path, ids); }

    auto rm = RowMapping::open(path);
    ASSERT_EQ(rm.size(), N);

    // Verify every 17th entry (~588 checks across the full range)
    for (uint64_t i = 0; i < N; i += 17) {
        EXPECT_EQ(rm.get(i).id, ids[i].id)
            << "Mismatch at index " << i;
    }

    // Verify find() for sampled entries
    for (uint64_t i = 0; i < N; i += 97) {
        auto idx = rm.find(ids[i]);
        ASSERT_TRUE(idx.has_value()) << "find() failed for index " << i;
        EXPECT_EQ(idx.value(), i);
    }
}

// ===========================================================================
// Sorted Index Lookup (O(log N) find)
// ===========================================================================

TEST_F(RowMappingTest, FindAllEntriesAfterOpen) {
    // Exhaustive: every single entry must be findable after open()
    const uint64_t N = 1000;
    std::vector<ObjectId> ids;
    ids.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        ids.push_back(ObjectId(i * 31 + 17)); // non-contiguous IDs
    }

    auto path = test_path("find_all_open.rmap");
    { auto rm = RowMapping::create(path, ids); }

    auto rm = RowMapping::open(path);
    for (uint64_t i = 0; i < N; ++i) {
        auto idx = rm.find(ids[i]);
        ASSERT_TRUE(idx.has_value())
            << "find() missed ObjectId(" << ids[i].id << ") at index " << i;
        EXPECT_EQ(idx.value(), i);
    }
}

TEST_F(RowMappingTest, FindAllEntriesAfterCreate) {
    // Same exhaustive check but on the created mapping (not re-opened)
    const uint64_t N = 1000;
    std::vector<ObjectId> ids;
    ids.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        ids.push_back(ObjectId(i * 31 + 17));
    }

    auto rm = RowMapping::create(test_path("find_all_create.rmap"), ids);
    for (uint64_t i = 0; i < N; ++i) {
        auto idx = rm.find(ids[i]);
        ASSERT_TRUE(idx.has_value())
            << "find() missed ObjectId(" << ids[i].id << ") at index " << i;
        EXPECT_EQ(idx.value(), i);
    }
}

TEST_F(RowMappingTest, FindAfterMoveConstructionPreservesLookup) {
    std::vector<ObjectId> ids = {ObjectId(100), ObjectId(200), ObjectId(300)};
    auto rm1 = RowMapping::create(test_path("find_move_ctor.rmap"), ids);

    auto rm2 = std::move(rm1);

    // Moved-to should have working find()
    auto idx = rm2.find(ObjectId(200));
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(idx.value(), 1u);

    // Moved-from find() should throw (not mapped)
    EXPECT_THROW(rm1.find(ObjectId(200)), std::runtime_error);
}

TEST_F(RowMappingTest, FindAfterMoveAssignmentPreservesLookup) {
    // Fix 4: Test find() after move-assignment (not just move-construction)
    std::vector<ObjectId> ids_a = {ObjectId(10), ObjectId(20)};
    std::vector<ObjectId> ids_b = {ObjectId(30), ObjectId(40), ObjectId(50)};

    auto rm = RowMapping::create(test_path("find_assign_a.rmap"), ids_a);
    auto idx_a = rm.find(ObjectId(20));
    ASSERT_TRUE(idx_a.has_value());
    EXPECT_EQ(idx_a.value(), 1u);

    // Move-assign a different mapping onto rm
    auto rm_b = RowMapping::create(test_path("find_assign_b.rmap"), ids_b);
    rm = std::move(rm_b);

    // find() should now work on the new mapping
    auto idx_b = rm.find(ObjectId(40));
    ASSERT_TRUE(idx_b.has_value());
    EXPECT_EQ(idx_b.value(), 1u);

    // Old IDs should not be found
    EXPECT_FALSE(rm.find(ObjectId(10)).has_value());
    EXPECT_FALSE(rm.find(ObjectId(20)).has_value());

    // Moved-from should throw
    EXPECT_THROW(rm_b.find(ObjectId(30)), std::runtime_error);
}

TEST_F(RowMappingTest, FindPerformanceLargeMapping) {
    // Fix 2: Separate correctness from timing to get accurate measurements.
    const uint64_t N = 100000;
    std::vector<ObjectId> ids;
    ids.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        ids.push_back(ObjectId(i * 7919 + 42)); // scattered IDs
    }

    auto path = test_path("find_perf.rmap");
    { auto rm = RowMapping::create(path, ids); }

    auto rm = RowMapping::open(path);

    // Pass 1: verify correctness (assertions, no timing)
    for (uint64_t i = 0; i < N; i += 2) {
        auto idx = rm.find(ids[i]);
        ASSERT_TRUE(idx.has_value());
        EXPECT_EQ(idx.value(), i);
    }

    // Pass 2: pure timing (no assertions — avoids string formatting overhead)
    auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < N; i += 2) { // 50K lookups
        volatile auto idx = rm.find(ids[i]);
        (void)idx;
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // With O(log N) this should take < 100ms. With O(N) it would take seconds.
    EXPECT_LT(ms, 2000) << "find() took " << ms << "ms for 50K lookups on 100K entries — likely O(N)";
}

TEST_F(RowMappingTest, FindBoundaryValues) {
    // Fix 3: ObjectId(0) = MASK_NULL, ObjectId(UINT64_MAX) = MASK_NOT_FOUND.
    // Both are sentinel values in MillenniumDB. The hash of 0 can be degenerate
    // on some implementations, and UINT64_MAX tests the top of the range.
    std::vector<ObjectId> ids = {
        ObjectId(0),
        ObjectId(42),
        ObjectId(UINT64_MAX)
    };
    auto rm = RowMapping::create(test_path("boundary.rmap"), ids);

    auto idx0 = rm.find(ObjectId(0));
    ASSERT_TRUE(idx0.has_value());
    EXPECT_EQ(idx0.value(), 0u);

    auto idx42 = rm.find(ObjectId(42));
    ASSERT_TRUE(idx42.has_value());
    EXPECT_EQ(idx42.value(), 1u);

    auto idxmax = rm.find(ObjectId(UINT64_MAX));
    ASSERT_TRUE(idxmax.has_value());
    EXPECT_EQ(idxmax.value(), 2u);

    // Nearby values should miss
    EXPECT_FALSE(rm.find(ObjectId(1)).has_value());
    EXPECT_FALSE(rm.find(ObjectId(UINT64_MAX - 1)).has_value());
}

// ===========================================================================
// Binary Search Lookup
// ===========================================================================

TEST(RowMappingBinarySearch, FindReturnsCorrectIndex) {
    auto tmp = fs::temp_directory_path() / "test_rm_bsearch";
    fs::create_directories(tmp);
    auto path = tmp / "test.rmap";

    std::vector<ObjectId> ids = {ObjectId(100), ObjectId(50), ObjectId(200), ObjectId(75)};
    auto rm = RowMapping::create(path, ids);

    EXPECT_EQ(rm.find(ObjectId(100)), 0u);
    EXPECT_EQ(rm.find(ObjectId(50)),  1u);
    EXPECT_EQ(rm.find(ObjectId(200)), 2u);
    EXPECT_EQ(rm.find(ObjectId(75)),  3u);
    EXPECT_EQ(rm.find(ObjectId(999)), std::nullopt);

    fs::remove_all(tmp);
}

TEST(RowMappingBinarySearch, LargeScaleLookup) {
    auto tmp = fs::temp_directory_path() / "test_rm_large";
    fs::create_directories(tmp);
    auto path = tmp / "test_large.rmap";

    constexpr uint64_t N = 10000;
    std::vector<ObjectId> ids(N);
    for (uint64_t i = 0; i < N; ++i) ids[i] = ObjectId(i * 7 + 3);

    auto rm = RowMapping::create(path, ids);
    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(rm.find(ObjectId(i * 7 + 3)), i);
    }
    EXPECT_EQ(rm.find(ObjectId(0)), std::nullopt);

    fs::remove_all(tmp);
}

// ===========================================================================
// Header Layout
// ===========================================================================

TEST_F(RowMappingTest, HeaderSizeIs16Bytes) {
    EXPECT_EQ(RowMapping::HEADER_SIZE, 16u);
}

// ===========================================================================
// Permutation-fingerprint staleness (regression for the 2026-06-01 L4
// feature-row corruption: a .idx built from one permutation was silently
// adopted after the .rmap was rebuilt with a DIFFERENT permutation at the
// same count, because the old guard validated only magic+version+count).
// ===========================================================================

TEST_F(RowMappingTest, StalePermutationIdxRejected) {
    // Reproduce the on-disk state that caused the bug: persist a .idx for
    // permutation A, then replace ONLY the .rmap data with permutation B
    // (same count) leaving A's .idx orphaned. open()+find() must serve B's
    // rows (the perm-fingerprint rejects the stale sidecar and rebuilds).
    const uint64_t N = 256;
    auto path = test_path("stale_perm.rmap");
    auto idx_path = fs::path(path.string() + ".idx");

    std::vector<ObjectId> a(N);
    for (uint64_t i = 0; i < N; ++i) a[i] = ObjectId(i * 2 + 1);
    {
        auto rm = RowMapping::create(path, a);
        auto r = rm.find(a[10]);          // first find() persists the .idx
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(r.value(), 10u);
    }
    ASSERT_TRUE(fs::exists(idx_path)) << "first find() should have persisted the .idx";

    // Permutation B: a full reverse — every entry moves (except a fixed point
    // only if N is odd; here N=256 is even so all entries move).
    std::vector<ObjectId> b(N);
    for (uint64_t i = 0; i < N; ++i) b[i] = ObjectId((N - 1 - i) * 2 + 1);

    // Overwrite the .rmap data region in place (header/count unchanged) so A's
    // .idx survives as an orphan — exactly the reorder-rebuild on-disk state.
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(f.is_open());
        f.seekp(static_cast<std::streamoff>(RowMapping::HEADER_SIZE));
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t v = b[i].id;
            f.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    }
    ASSERT_TRUE(fs::exists(idx_path)) << "the stale .idx must still be on disk for the test to be meaningful";

    auto rm = RowMapping::open(path);
    for (uint64_t i = 0; i < N; ++i) {
        auto r = rm.find(b[i]);
        ASSERT_TRUE(r.has_value()) << "find missed B[" << i << "]";
        EXPECT_EQ(r.value(), i)
            << "stale .idx was adopted: B[" << i << "] (oid " << b[i].id
            << ") resolved to the wrong row";
    }
}

TEST_F(RowMappingTest, CreateRemovesStaleSiblingIdx) {
    // create() must delete any pre-existing <path>.idx, because writing a new
    // permutation invalidates a sidecar built from the previous one.
    const uint64_t N = 64;
    auto path = test_path("create_removes_idx.rmap");
    auto idx_path = fs::path(path.string() + ".idx");

    std::vector<ObjectId> a(N);
    for (uint64_t i = 0; i < N; ++i) a[i] = ObjectId(i * 3 + 5);
    {
        auto rm = RowMapping::create(path, a);
        (void)rm.find(a[0]);              // persist the .idx
    }
    ASSERT_TRUE(fs::exists(idx_path));

    std::vector<ObjectId> b(N);
    for (uint64_t i = 0; i < N; ++i) b[i] = ObjectId((N - 1 - i) * 3 + 5);
    auto rm = RowMapping::create(path, b);

    EXPECT_FALSE(fs::exists(idx_path))
        << "create() must remove the stale sibling .idx before a new permutation is used";

    for (uint64_t i = 0; i < N; ++i) {
        auto r = rm.find(b[i]);
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(r.value(), i);
    }
}
