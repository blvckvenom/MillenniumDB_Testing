#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "gnn/storage/packed_batch_store.h"

using PackedBatchTest = GnnStorageTest;

// ===========================================================================
// Header
// ===========================================================================

TEST(PackedBatchHeaderTest, MakeAndValidate) {
    auto h = PackedBatchHeader::make(100, 128, GnnDtype::FLOAT32);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.magic, PackedBatchHeader::MAGIC);
    EXPECT_EQ(h.version, PackedBatchHeader::VERSION);
    EXPECT_EQ(h.num_nodes, 100u);
    EXPECT_EQ(h.feature_dim, 128u);
    EXPECT_EQ(h.get_dtype(), GnnDtype::FLOAT32);
    EXPECT_EQ(h.data_bytes(), 100u * 128u * 4u);
}

TEST(PackedBatchHeaderTest, InvalidMagic) {
    auto h = PackedBatchHeader::make(10, 10, GnnDtype::FLOAT32);
    h.magic = 0xDEADBEEF;
    EXPECT_FALSE(h.is_valid());
}

TEST(PackedBatchHeaderTest, InvalidDtype) {
    auto h = PackedBatchHeader::make(10, 10, GnnDtype::FLOAT32);
    h.dtype = 255;
    EXPECT_FALSE(h.is_valid());
}

TEST(PackedBatchHeaderTest, ZeroNodesIsValid) {
    auto h = PackedBatchHeader::make(0, 128, GnnDtype::FLOAT32);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.data_bytes(), 0u);
}

TEST(PackedBatchHeaderTest, ZeroFeatureDimIsInvalid) {
    PackedBatchHeader h{};
    std::memset(&h, 0, sizeof(h));
    h.magic = PackedBatchHeader::MAGIC;
    h.version = PackedBatchHeader::VERSION;
    h.num_nodes = 10;
    h.feature_dim = 0;
    h.dtype = static_cast<uint8_t>(GnnDtype::FLOAT32);
    EXPECT_FALSE(h.is_valid());
}

TEST(PackedBatchHeaderTest, SizeIs32Bytes) {
    EXPECT_EQ(sizeof(PackedBatchHeader), 32u);
}

TEST(PackedBatchHeaderTest, DataBytesOverflow) {
    PackedBatchHeader h{};
    std::memset(&h, 0, sizeof(h));
    h.magic = PackedBatchHeader::MAGIC;
    h.version = PackedBatchHeader::VERSION;
    h.num_nodes = UINT64_MAX;
    h.feature_dim = UINT64_MAX;
    h.dtype = static_cast<uint8_t>(GnnDtype::FLOAT32);
    EXPECT_THROW(h.data_bytes(), std::overflow_error);
}

TEST(PackedBatchHeaderTest, MakeInvalidDtypeThrows) {
    EXPECT_THROW(
        PackedBatchHeader::make(10, 10, static_cast<GnnDtype>(255)),
        std::invalid_argument
    );
}
