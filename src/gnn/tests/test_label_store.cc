#include "test_helpers.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "gnn/training/label_store.h"

using LabelStoreTest = GnnStorageTest;

// ===========================================================================
// WriteAndReadRoundtrip
// ===========================================================================

TEST_F(LabelStoreTest, WriteAndReadRoundtrip) {
    auto path = test_path("labels.bin");

    std::vector<int64_t> labels = {0, 1, 2, -1, 0};
    mdb::gnn::LabelStore::write(path, labels, 3);

    auto ls = mdb::gnn::LabelStore::open(path);

    EXPECT_EQ(ls.num_nodes(),   5u);
    EXPECT_EQ(ls.num_classes(), 3u);
    EXPECT_EQ(ls.get(0), 0);
    EXPECT_EQ(ls.get(1), 1);
    EXPECT_EQ(ls.get(2), 2);
    EXPECT_EQ(ls.get(3), -1);  // unlabeled
    EXPECT_EQ(ls.get(4), 0);
}

// ===========================================================================
// GatherReturnsTensor
// ===========================================================================

TEST_F(LabelStoreTest, GatherReturnsTensor) {
    auto path = test_path("labels_gather.bin");

    std::vector<int64_t> labels = {0, 1, 2, -1, 0};
    mdb::gnn::LabelStore::write(path, labels, 3);

    auto ls = mdb::gnn::LabelStore::open(path);

    std::vector<uint64_t> indices = {4, 3, 0, 2};
    auto tensor = ls.gather(indices);

    ASSERT_EQ(tensor.dim(),   1);
    ASSERT_EQ(tensor.size(0), 4);

    auto acc = tensor.accessor<int64_t, 1>();
    EXPECT_EQ(acc[0],  0);   // labels[4]
    EXPECT_EQ(acc[1], -1);   // labels[3] — unlabeled
    EXPECT_EQ(acc[2],  0);   // labels[0]
    EXPECT_EQ(acc[3],  2);   // labels[2]
}

// ===========================================================================
// EmptyGatherReturnsTensor
// ===========================================================================

TEST_F(LabelStoreTest, EmptyGatherReturnsTensor) {
    auto path = test_path("labels_empty_gather.bin");

    std::vector<int64_t> labels = {0, 1};
    mdb::gnn::LabelStore::write(path, labels, 2);

    auto ls = mdb::gnn::LabelStore::open(path);

    auto tensor = ls.gather({});
    EXPECT_EQ(tensor.size(0), 0);
}

// ===========================================================================
// InvalidMagicThrows
// ===========================================================================

TEST_F(LabelStoreTest, InvalidMagicThrows) {
    auto path = test_path("labels_bad_magic.bin");

    // Write garbage: wrong magic bytes followed by plausible header data
    {
        std::ofstream f(path, std::ios::binary);
        const uint8_t bad_magic[8] = {'X','X','X','X','\0','\0','\0','\0'};
        f.write(reinterpret_cast<const char*>(bad_magic), 8);
        uint32_t version  = 1;
        uint32_t reserved = 0;
        uint64_t num_nodes   = 0;
        uint64_t num_classes = 0;
        f.write(reinterpret_cast<const char*>(&version),    4);
        f.write(reinterpret_cast<const char*>(&reserved),   4);
        f.write(reinterpret_cast<const char*>(&num_nodes),  8);
        f.write(reinterpret_cast<const char*>(&num_classes),8);
    }

    EXPECT_THROW(mdb::gnn::LabelStore::open(path), std::runtime_error);
}

// ===========================================================================
// InvalidVersionThrows
// ===========================================================================

TEST_F(LabelStoreTest, InvalidVersionThrows) {
    auto path = test_path("labels_bad_version.bin");

    {
        std::ofstream f(path, std::ios::binary);
        const uint8_t good_magic[8] = {'G','N','N','L','\0','\0','\0','\0'};
        f.write(reinterpret_cast<const char*>(good_magic), 8);
        uint32_t bad_version = 99;
        uint32_t reserved    = 0;
        uint64_t num_nodes   = 0;
        uint64_t num_classes = 0;
        f.write(reinterpret_cast<const char*>(&bad_version),  4);
        f.write(reinterpret_cast<const char*>(&reserved),     4);
        f.write(reinterpret_cast<const char*>(&num_nodes),    8);
        f.write(reinterpret_cast<const char*>(&num_classes),  8);
    }

    EXPECT_THROW(mdb::gnn::LabelStore::open(path), std::runtime_error);
}

// ===========================================================================
// GetOutOfBoundsThrows
// ===========================================================================

TEST_F(LabelStoreTest, GetOutOfBoundsThrows) {
    auto path = test_path("labels_oob.bin");

    std::vector<int64_t> labels = {0, 1, 2, -1, 0};
    mdb::gnn::LabelStore::write(path, labels, 3);

    auto ls = mdb::gnn::LabelStore::open(path);

    EXPECT_THROW(ls.get(999), std::out_of_range);
    EXPECT_THROW(ls.get(5),   std::out_of_range);
    // last valid index must NOT throw
    EXPECT_NO_THROW(ls.get(4));
}

// ===========================================================================
// MoveSemantics
// ===========================================================================

TEST_F(LabelStoreTest, MoveSemantics) {
    auto path = test_path("labels_move.bin");

    std::vector<int64_t> labels = {7, 3, -1};
    mdb::gnn::LabelStore::write(path, labels, 8);

    auto ls1 = mdb::gnn::LabelStore::open(path);
    EXPECT_EQ(ls1.num_nodes(), 3u);
    EXPECT_EQ(ls1.get(0), 7);

    // Move construct
    auto ls2 = std::move(ls1);
    EXPECT_EQ(ls2.num_nodes(), 3u);
    EXPECT_EQ(ls2.get(1), 3);

    // Source should be invalidated — num_nodes == 0
    EXPECT_EQ(ls1.num_nodes(), 0u);

    // Move assign (open a second store, then overwrite via move-assignment)
    auto ls3 = mdb::gnn::LabelStore::open(path);
    ls3 = std::move(ls2);
    EXPECT_EQ(ls3.num_nodes(), 3u);
    EXPECT_EQ(ls3.get(2), -1);
    EXPECT_EQ(ls2.num_nodes(), 0u);
}

// ===========================================================================
// ZeroNodesFile
// ===========================================================================

TEST_F(LabelStoreTest, ZeroNodesFile) {
    auto path = test_path("labels_zero.bin");

    std::vector<int64_t> empty_labels;
    mdb::gnn::LabelStore::write(path, empty_labels, 0);

    auto ls = mdb::gnn::LabelStore::open(path);
    EXPECT_EQ(ls.num_nodes(),   0u);
    EXPECT_EQ(ls.num_classes(), 0u);
    EXPECT_THROW(ls.get(0), std::out_of_range);
}
