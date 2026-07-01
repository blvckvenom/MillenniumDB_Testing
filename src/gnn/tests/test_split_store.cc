#include "test_helpers.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "gnn/training/split_store.h"

using SplitStoreTest = GnnStorageTest;

// ===========================================================================
// WriteAndReadRoundtrip
// ===========================================================================

TEST_F(SplitStoreTest, WriteAndReadRoundtrip) {
    auto path = test_path("splits.bin");

    std::vector<uint8_t> splits = {0, 1, 2, 255, 0};
    mdb::gnn::SplitStore::write(path, splits);

    auto ss = mdb::gnn::SplitStore::open(path);

    EXPECT_EQ(ss.num_nodes(), 5u);
    EXPECT_EQ(ss.get(0), mdb::gnn::SplitStore::TRAIN);
    EXPECT_EQ(ss.get(1), mdb::gnn::SplitStore::VAL);
    EXPECT_EQ(ss.get(2), mdb::gnn::SplitStore::TEST);
    EXPECT_EQ(ss.get(3), mdb::gnn::SplitStore::UNLABELED);
    EXPECT_EQ(ss.get(4), mdb::gnn::SplitStore::TRAIN);
}

// ===========================================================================
// GatherMaskTrain
// ===========================================================================

TEST_F(SplitStoreTest, GatherMaskTrain) {
    auto path = test_path("splits_mask_train.bin");

    // indices:  0=TRAIN, 1=VAL, 2=TEST, 3=UNLABELED, 4=TRAIN
    std::vector<uint8_t> splits = {0, 1, 2, 255, 0};
    mdb::gnn::SplitStore::write(path, splits);

    auto ss = mdb::gnn::SplitStore::open(path);

    std::vector<uint64_t> indices = {0, 1, 4, 3};
    auto mask = ss.gather_mask(indices, mdb::gnn::SplitStore::TRAIN);

    ASSERT_EQ(mask.dim(),   1);
    ASSERT_EQ(mask.size(0), 4);

    auto acc = mask.accessor<bool, 1>();
    EXPECT_TRUE (acc[0]);   // splits[0] = TRAIN  -> true
    EXPECT_FALSE(acc[1]);   // splits[1] = VAL    -> false
    EXPECT_TRUE (acc[2]);   // splits[4] = TRAIN  -> true
    EXPECT_FALSE(acc[3]);   // splits[3] = UNLABELED -> false
}

// ===========================================================================
// GatherMaskTest
// ===========================================================================

TEST_F(SplitStoreTest, GatherMaskTest) {
    auto path = test_path("splits_mask_test.bin");

    std::vector<uint8_t> splits = {0, 1, 2, 255, 0};
    mdb::gnn::SplitStore::write(path, splits);

    auto ss = mdb::gnn::SplitStore::open(path);

    std::vector<uint64_t> indices = {2, 0, 3, 1};
    auto mask = ss.gather_mask(indices, mdb::gnn::SplitStore::TEST);

    ASSERT_EQ(mask.dim(),   1);
    ASSERT_EQ(mask.size(0), 4);

    auto acc = mask.accessor<bool, 1>();
    EXPECT_TRUE (acc[0]);   // splits[2] = TEST      -> true
    EXPECT_FALSE(acc[1]);   // splits[0] = TRAIN     -> false
    EXPECT_FALSE(acc[2]);   // splits[3] = UNLABELED -> false
    EXPECT_FALSE(acc[3]);   // splits[1] = VAL       -> false
}

// ===========================================================================
// ParseSplitString
// ===========================================================================

TEST_F(SplitStoreTest, ParseSplitString) {
    EXPECT_EQ(mdb::gnn::SplitStore::parse_split_string("train"),      mdb::gnn::SplitStore::TRAIN);
    EXPECT_EQ(mdb::gnn::SplitStore::parse_split_string("val"),        mdb::gnn::SplitStore::VAL);
    // "valid" is the OGB split token; mapping it to UNLABELED silently wipes
    // every validation node (the ogbn-products bestValAccuracy=0 failure).
    EXPECT_EQ(mdb::gnn::SplitStore::parse_split_string("valid"),      mdb::gnn::SplitStore::VAL);
    EXPECT_EQ(mdb::gnn::SplitStore::parse_split_string("validation"), mdb::gnn::SplitStore::VAL);
    EXPECT_EQ(mdb::gnn::SplitStore::parse_split_string("test"),       mdb::gnn::SplitStore::TEST);
    EXPECT_EQ(mdb::gnn::SplitStore::parse_split_string("other"),      mdb::gnn::SplitStore::UNLABELED);
    EXPECT_EQ(mdb::gnn::SplitStore::parse_split_string(""),           mdb::gnn::SplitStore::UNLABELED);
    EXPECT_EQ(mdb::gnn::SplitStore::parse_split_string("TRAIN"),      mdb::gnn::SplitStore::UNLABELED);
}

// ===========================================================================
// InvalidMagicThrows
// ===========================================================================

TEST_F(SplitStoreTest, InvalidMagicThrows) {
    auto path = test_path("splits_bad_magic.bin");

    {
        std::ofstream f(path, std::ios::binary);
        const uint8_t bad_magic[8] = {'X','X','X','X','\0','\0','\0','\0'};
        f.write(reinterpret_cast<const char*>(bad_magic), 8);
        uint32_t version  = 1;
        uint32_t reserved = 0;
        uint64_t num_nodes = 0;
        f.write(reinterpret_cast<const char*>(&version),   4);
        f.write(reinterpret_cast<const char*>(&reserved),  4);
        f.write(reinterpret_cast<const char*>(&num_nodes), 8);
    }

    EXPECT_THROW(mdb::gnn::SplitStore::open(path), std::runtime_error);
}

// ===========================================================================
// GetOutOfBoundsThrows
// ===========================================================================

TEST_F(SplitStoreTest, GetOutOfBoundsThrows) {
    auto path = test_path("splits_oob.bin");

    std::vector<uint8_t> splits = {0, 1, 2, 255, 0};
    mdb::gnn::SplitStore::write(path, splits);

    auto ss = mdb::gnn::SplitStore::open(path);

    EXPECT_THROW(ss.get(999), std::out_of_range);
    EXPECT_THROW(ss.get(5),   std::out_of_range);
    // last valid index must NOT throw
    EXPECT_NO_THROW(ss.get(4));
}

// ===========================================================================
// MoveSemantics
// ===========================================================================

TEST_F(SplitStoreTest, MoveSemantics) {
    auto path = test_path("splits_move.bin");

    std::vector<uint8_t> splits = {0, 1, 255};
    mdb::gnn::SplitStore::write(path, splits);

    auto ss1 = mdb::gnn::SplitStore::open(path);
    EXPECT_EQ(ss1.num_nodes(), 3u);
    EXPECT_EQ(ss1.get(0), mdb::gnn::SplitStore::TRAIN);

    // Move construct
    auto ss2 = std::move(ss1);
    EXPECT_EQ(ss2.num_nodes(), 3u);
    EXPECT_EQ(ss2.get(1), mdb::gnn::SplitStore::VAL);

    // Source should be invalidated — num_nodes == 0
    EXPECT_EQ(ss1.num_nodes(), 0u);

    // Move assign (open a second store, then overwrite via move-assignment)
    auto ss3 = mdb::gnn::SplitStore::open(path);
    ss3 = std::move(ss2);
    EXPECT_EQ(ss3.num_nodes(), 3u);
    EXPECT_EQ(ss3.get(2), mdb::gnn::SplitStore::UNLABELED);
    EXPECT_EQ(ss2.num_nodes(), 0u);
}
