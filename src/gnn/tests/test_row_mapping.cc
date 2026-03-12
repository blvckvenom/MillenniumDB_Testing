#include "test_helpers.h"

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
// Header Layout
// ===========================================================================

TEST_F(RowMappingTest, HeaderSizeIs16Bytes) {
    EXPECT_EQ(RowMapping::HEADER_SIZE, 16u);
}
