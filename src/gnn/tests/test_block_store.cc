#include <gtest/gtest.h>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>
#include "gnn/storage/block_format.h"
#include "gnn/storage/block_store.h"
#include "gnn/training/graph_block_builder.h"  // graph_block::build_active_indices / build_edge_indices
                                                // (pulls in gnn/sampling/graph_sample.h => GraphSample,
                                                //  SplitType, ObjectId)
using namespace mdb::gnn;

TEST(BlockFormat, MakeAndValidate) {
    auto h = BlockBatchHeader::make(/*num_layers=*/3, /*sample_fp=*/0xABCDull);
    EXPECT_TRUE(h.is_valid());
    EXPECT_EQ(h.magic, BlockBatchHeader::MAGIC);
    EXPECT_EQ(h.version, BlockBatchHeader::VERSION);
    EXPECT_EQ(h.num_layers, 3u);
    EXPECT_EQ(h.sample_fp, 0xABCDull);
}
TEST(BlockFormat, RejectsBadMagic) {
    BlockBatchHeader h{};  // zeroed
    EXPECT_FALSE(h.is_valid());
}
static_assert(sizeof(BlockBatchHeader) == 64, "block header 64 B");

TEST(BlockStore, WriteThenReadRoundTrip) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_rt";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {5, 3, 2};                   // M_0..M_2 (K+1=3 node layers => 2 conv layers)
    std::vector<torch::Tensor> edges = {                       // 2 conv layers
        torch::tensor({{0,1,2},{2,2,1}}, torch::kInt64),      // [2,3]
        torch::tensor({{0},{1}}, torch::kInt64),              // [2,1]
    };
    BlockWriter::write(tmp / "block_000000.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges);
    auto blk = BlockReader::open(tmp / "block_000000.blk", /*expected_sample_fp=*/0xFEEDull);
    ASSERT_TRUE(blk.has_value());
    EXPECT_EQ(blk->active_sizes, sizes);
    ASSERT_EQ(blk->edge_indices.size(), 2u);
    EXPECT_TRUE(torch::equal(blk->edge_indices[0], edges[0]));  // widened back to int64
    EXPECT_TRUE(torch::equal(blk->edge_indices[1], edges[1]));
}
TEST(BlockStore, StalenessRejectsWrongFp) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_stale";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {2, 1};                       // K=1 conv layer
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    BlockWriter::write(tmp / "b.blk", 0x1111ull, 0, sizes, edges);
    EXPECT_FALSE(BlockReader::open(tmp / "b.blk", /*expected=*/0x2222ull).has_value()); // stale → reject
    EXPECT_TRUE (BlockReader::open(tmp / "b.blk", /*expected=*/0x1111ull).has_value()); // fresh → accept
}
TEST(BlockStore, MissingFileReturnsNullopt) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_missing";
    std::filesystem::create_directories(tmp);
    EXPECT_FALSE(BlockReader::open(tmp / "does_not_exist.blk", 0).has_value());
}

// is_fresh: header-only freshness peek used for the idempotent bake-skip.
TEST(BlockStore, IsFreshTrueForMatchingFp) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_fresh_ok";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {2, 1};                       // K=1 conv layer
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    BlockWriter::write(tmp / "b.blk", /*sample_fp=*/0xABCDull, /*batch_id=*/0, sizes, edges);
    EXPECT_TRUE(BlockReader::is_fresh(tmp / "b.blk", /*expected=*/0xABCDull));
}
TEST(BlockStore, IsFreshFalseForWrongFp) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_fresh_wrong";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {2, 1};
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    BlockWriter::write(tmp / "b.blk", /*sample_fp=*/0x1111ull, /*batch_id=*/0, sizes, edges);
    EXPECT_FALSE(BlockReader::is_fresh(tmp / "b.blk", /*expected=*/0x2222ull));  // stale fp
}
TEST(BlockStore, IsFreshFalseForMissingFile) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_fresh_missing";
    std::filesystem::create_directories(tmp);
    EXPECT_FALSE(BlockReader::is_fresh(tmp / "does_not_exist.blk", 0));
}

// The bit-identical guarantee: a block written from the ONLINE
// graph_block::build_active_indices/build_edge_indices output and read back
// must yield active_sizes + per-layer edge_index tensors identical
// (torch::equal) to the online derivation. Proves write->read == online.
TEST(BlockStore, BakeMatchesOnlineDerivation) {
    std::vector<ObjectId> n;
    for (uint64_t i = 1; i <= 6; ++i) n.emplace_back(i);   // 6 distinct nonzero ids

    GraphSample s;
    s.batch_id = 7;
    s.split = SplitType::TRAIN;
    s.nodes_per_layer = { {n[0], n[1]}, {n[2], n[3]}, {n[4], n[5]} };  // K=2 conv layers
    s.edges_per_layer.resize(2);
    s.edges_per_layer[0].src_indices = {0, 1};  // layer1[0,1] = n2,n3
    s.edges_per_layer[0].dst_indices = {0, 1};  // layer0[0,1] = n0,n1
    s.edges_per_layer[0].edge_ids    = { ObjectId(0), ObjectId(0) };
    s.edges_per_layer[1].src_indices = {0, 1};  // layer2[0,1] = n4,n5
    s.edges_per_layer[1].dst_indices = {0, 1};  // layer1[0,1] = n2,n3
    s.edges_per_layer[1].edge_ids    = { ObjectId(0), ObjectId(0) };
    s.rebuild_unique_nodes();  // identity-prefix all_unique_nodes (layer order)
    ASSERT_EQ(s.all_unique_nodes.size(), 6u);  // non-degenerate active sets

    std::unordered_map<uint64_t, int64_t> oid_to_global;
    for (int64_t i = 0; i < static_cast<int64_t>(s.all_unique_nodes.size()); ++i)
        oid_to_global[s.all_unique_nodes[i].id] = i;

    auto active       = mdb::gnn::graph_block::build_active_indices(s, oid_to_global);
    auto online_edges = mdb::gnn::graph_block::build_edge_indices(s, active);
    ASSERT_EQ(online_edges.size(), 2u);  // K conv layers => K edge_index tensors

    auto tmp = std::filesystem::temp_directory_path() / "blk_equal";
    std::filesystem::create_directories(tmp);
    BlockWriter::write(tmp / "b.blk", /*sample_fp=*/0x55ull, s.batch_id,
                       active.sizes_per_layer, online_edges);
    auto blk = BlockReader::open(tmp / "b.blk", /*expected_sample_fp=*/0x55ull);
    ASSERT_TRUE(blk.has_value());

    EXPECT_EQ(blk->active_sizes, active.sizes_per_layer);
    ASSERT_EQ(blk->edge_indices.size(), online_edges.size());
    for (size_t k = 0; k < online_edges.size(); ++k)
        EXPECT_TRUE(torch::equal(blk->edge_indices[k], online_edges[k]));
}
