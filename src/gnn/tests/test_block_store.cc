#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

// ---------------------------------------------------------------------------
// SC-1: v2 self-contained format (store_fp + node count + seeds + split).
// ---------------------------------------------------------------------------

static_assert(sizeof(BlockBatchHeader) == 64, "v2 block header still 64 B");

TEST(BlockFormat, V2HeaderSelfContained) {
    auto h = BlockBatchHeader::make_self_contained(/*num_layers=*/3, /*sample_fp=*/0xABull,
                                                   /*batch_id=*/7, /*store_fp=*/0xCAFEull,
                                                   /*num_unique_nodes=*/100, /*num_seeds=*/4,
                                                   /*split=*/1);
    EXPECT_TRUE(h.is_valid());
    EXPECT_TRUE(h.is_self_contained());
    EXPECT_EQ(h.version, 2u);
    EXPECT_EQ(h.store_fp, 0xCAFEull);
    EXPECT_EQ(h.num_unique_nodes, 100u);
    EXPECT_EQ(h.num_seeds, 4u);
    EXPECT_EQ(h.split, 1u);
    EXPECT_EQ(h.num_layers, 3u);
    EXPECT_EQ(h.sample_fp, 0xABull);
    EXPECT_EQ(h.batch_id, 7u);

    // The plain make() is v2-format but NOT self-contained (store_fp==0).
    auto h0 = BlockBatchHeader::make(/*num_layers=*/3, /*sample_fp=*/0xABull, /*batch_id=*/7);
    EXPECT_TRUE(h0.is_valid());
    EXPECT_FALSE(h0.is_self_contained());
    EXPECT_EQ(h0.version, 2u);
    EXPECT_EQ(h0.store_fp, 0u);
}

TEST(BlockStore, SelfContainedRoundTrip) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_self_rt";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {5, 3, 2};                    // K=2 conv layers
    std::vector<torch::Tensor> edges = {
        torch::tensor({{0,1,2},{2,2,1}}, torch::kInt64),       // [2,3]
        torch::tensor({{0},{1}}, torch::kInt64),               // [2,1]
    };
    std::vector<uint64_t> seeds = {10, 11, 12};
    BlockWriter::write(tmp / "b.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges,
                       /*store_fp=*/0xBEEFull, /*num_unique_nodes=*/50, /*seed_ids=*/seeds,
                       /*split=*/2);
    auto blk = BlockReader::open(tmp / "b.blk", /*expected_sample_fp=*/0xFEEDull);
    ASSERT_TRUE(blk.has_value());
    EXPECT_EQ(blk->active_sizes, sizes);
    ASSERT_EQ(blk->edge_indices.size(), 2u);
    EXPECT_TRUE(torch::equal(blk->edge_indices[0], edges[0]));
    EXPECT_TRUE(torch::equal(blk->edge_indices[1], edges[1]));
    EXPECT_EQ(blk->store_fp, 0xBEEFull);
    EXPECT_EQ(blk->num_unique_nodes, 50u);
    EXPECT_EQ(blk->split, 2u);
    EXPECT_EQ(blk->seed_ids, seeds);
}

TEST(BlockStore, LegacyWriteHasNoSelfContainedFields) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_legacy_write";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {5, 3, 2};
    std::vector<torch::Tensor> edges = {
        torch::tensor({{0,1,2},{2,2,1}}, torch::kInt64),
        torch::tensor({{0},{1}}, torch::kInt64),
    };
    // OLD 5-arg form: no v2 args -> defaults (store_fp=0, no seeds).
    BlockWriter::write(tmp / "b.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges);
    auto blk = BlockReader::open(tmp / "b.blk", /*expected_sample_fp=*/0xFEEDull);
    ASSERT_TRUE(blk.has_value());
    // Self-contained fields absent...
    EXPECT_EQ(blk->store_fp, 0u);
    EXPECT_EQ(blk->num_unique_nodes, 0u);
    EXPECT_TRUE(blk->seed_ids.empty());
    // ...but the computation graph still round-trips intact.
    EXPECT_EQ(blk->active_sizes, sizes);
    ASSERT_EQ(blk->edge_indices.size(), 2u);
    EXPECT_TRUE(torch::equal(blk->edge_indices[0], edges[0]));
    EXPECT_TRUE(torch::equal(blk->edge_indices[1], edges[1]));
}

// ---------------------------------------------------------------------------
// SC-3: open_self_contained validates via the STORE fingerprint (not per-batch
// sample_fp) and is the train-time entry point for skipping batches.dat.
// ---------------------------------------------------------------------------

TEST(BlockStore, OpenSelfContainedMatchesStoreFp) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_sc_open_ok";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {5, 3, 2};                    // K=2 conv layers
    std::vector<torch::Tensor> edges = {
        torch::tensor({{0,1,2},{2,2,1}}, torch::kInt64),       // [2,3]
        torch::tensor({{0},{1}}, torch::kInt64),               // [2,1]
    };
    std::vector<uint64_t> seeds = {10, 11, 12};
    BlockWriter::write(tmp / "b.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges,
                       /*store_fp=*/0xBEEFull, /*num_unique_nodes=*/50, /*seed_ids=*/seeds,
                       /*split=*/2);
    // Correct store_fp -> has_value with all self-contained fields populated.
    auto blk = BlockReader::open_self_contained(tmp / "b.blk", /*expected_store_fp=*/0xBEEFull);
    ASSERT_TRUE(blk.has_value());
    EXPECT_EQ(blk->active_sizes, sizes);
    ASSERT_EQ(blk->edge_indices.size(), 2u);
    EXPECT_TRUE(torch::equal(blk->edge_indices[0], edges[0]));
    EXPECT_TRUE(torch::equal(blk->edge_indices[1], edges[1]));
    EXPECT_EQ(blk->store_fp, 0xBEEFull);
    EXPECT_EQ(blk->num_unique_nodes, 50u);
    EXPECT_EQ(blk->split, 2u);
    EXPECT_EQ(blk->seed_ids, seeds);
}

TEST(BlockStore, OpenSelfContainedRejectsWrongOrAbsentFp) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_sc_open_reject";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {2, 1};                       // K=1 conv layer
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    std::vector<uint64_t> seeds = {7, 8};

    // Self-contained block (store_fp != 0).
    BlockWriter::write(tmp / "sc.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges,
                       /*store_fp=*/0x1234ull, /*num_unique_nodes=*/9, /*seed_ids=*/seeds,
                       /*split=*/1);
    // Wrong store_fp -> nullopt.
    EXPECT_FALSE(BlockReader::open_self_contained(tmp / "sc.blk", /*expected=*/0x9999ull).has_value());
    // expected_store_fp == 0 (UNKNOWN provenance) -> nullopt even though it matches nothing.
    EXPECT_FALSE(BlockReader::open_self_contained(tmp / "sc.blk", /*expected=*/0ull).has_value());
    // Correct store_fp -> accepted.
    EXPECT_TRUE (BlockReader::open_self_contained(tmp / "sc.blk", /*expected=*/0x1234ull).has_value());

    // A non-self-contained block (legacy 5-arg write, store_fp==0) is rejected
    // regardless of the expected fp.
    BlockWriter::write(tmp / "legacy.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges);
    EXPECT_FALSE(BlockReader::open_self_contained(tmp / "legacy.blk", /*expected=*/0x1234ull).has_value());
    EXPECT_FALSE(BlockReader::open_self_contained(tmp / "legacy.blk", /*expected=*/0ull).has_value());

    // Missing file -> nullopt.
    EXPECT_FALSE(BlockReader::open_self_contained(tmp / "missing.blk", /*expected=*/0x1234ull).has_value());
}

TEST(BlockStore, ReadStoreFpHeaderOnly) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_read_storefp";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {2, 1};                       // K=1 conv layer
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    BlockWriter::write(tmp / "b.blk", /*sample_fp=*/0xABCDull, /*batch_id=*/0, sizes, edges,
                       /*store_fp=*/0x1234ull);
    EXPECT_EQ(BlockReader::read_store_fp(tmp / "b.blk"), 0x1234ull);
    EXPECT_EQ(BlockReader::read_store_fp(tmp / "does_not_exist.blk"), 0u);  // missing -> 0
}

// ---------------------------------------------------------------------------
// Freshness-guard hardening: the bake-skip predicate must reject legacy-format
// and stale-store_fp blocks so a re-bake rewrites them instead of silently
// keeping them (which would quietly disable the self-contained train path).
// ---------------------------------------------------------------------------

// Overwrite n bytes at offset `off` of an existing file (header fields are at
// fixed offsets: version u32 @4, num_layers u32 @16, num_seeds u64 @48; the
// first conv-layer count E[0] of a K=1 block sits at 64 + 2*8 = 80).
static void patch_bytes(const std::filesystem::path& p, std::streamoff off,
                        const void* data, size_t n) {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f.seekp(off);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    ASSERT_TRUE(f.good());
}

TEST(BlockStore, IsFreshRejectsLegacyVersionBlock) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_fresh_v1";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {2, 1};                       // K=1 conv layer
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    BlockWriter::write(tmp / "b.blk", /*sample_fp=*/0xABCDull, /*batch_id=*/0, sizes, edges);
    EXPECT_TRUE(BlockReader::is_fresh(tmp / "b.blk", /*expected=*/0xABCDull));
    // Downgrade the on-disk header to format v1: still readable (back-compat)...
    const uint32_t v1 = 1u;
    patch_bytes(tmp / "b.blk", /*off=version*/4, &v1, sizeof(v1));
    EXPECT_TRUE(BlockReader::open(tmp / "b.blk", /*expected=*/0xABCDull).has_value());
    // ...but NOT fresh: a re-bake must rewrite it at the current version.
    EXPECT_FALSE(BlockReader::is_fresh(tmp / "b.blk", /*expected=*/0xABCDull));
}

TEST(BlockStore, IsFreshChecksStoreFpForSelfContainedBake) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_fresh_storefp";
    std::filesystem::create_directories(tmp);
    std::vector<int64_t> sizes = {2, 1};                       // K=1 conv layer
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    std::vector<uint64_t> seeds = {7, 8};
    BlockWriter::write(tmp / "sc.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges,
                       /*store_fp=*/0x1234ull, /*num_unique_nodes=*/9, /*seed_ids=*/seeds,
                       /*split=*/1);
    // Store-agnostic caller (expected_store_fp omitted) keeps the old contract.
    EXPECT_TRUE (BlockReader::is_fresh(tmp / "sc.blk", /*expected=*/0xFEEDull));
    // A self-contained bake passes the store fingerprint: match -> fresh,
    // stale -> re-bake.
    EXPECT_TRUE (BlockReader::is_fresh(tmp / "sc.blk", 0xFEEDull, /*store_fp=*/0x1234ull));
    EXPECT_FALSE(BlockReader::is_fresh(tmp / "sc.blk", 0xFEEDull, /*store_fp=*/0x9999ull));
    // A non-self-contained block can never satisfy a self-contained bake.
    BlockWriter::write(tmp / "legacy.blk", /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges);
    EXPECT_FALSE(BlockReader::is_fresh(tmp / "legacy.blk", 0xFEEDull, /*store_fp=*/0x1234ull));
}

// ---------------------------------------------------------------------------
// Corrupt on-disk counts (bit rot / tampering past the magic+fp checks) must
// degrade to nullopt — the graceful fall-back-to-online contract — instead of
// throwing bad_alloc / c10::Error out of the reader.
// ---------------------------------------------------------------------------

static void write_k1_block(const std::filesystem::path& path) {
    std::vector<int64_t> sizes = {2, 1};                       // K=1 conv layer
    std::vector<torch::Tensor> edges = { torch::tensor({{0},{1}}, torch::kInt64) };
    std::vector<uint64_t> seeds = {5, 6};
    BlockWriter::write(path, /*sample_fp=*/0xFEEDull, /*batch_id=*/0, sizes, edges,
                       /*store_fp=*/0xBEEFull, /*num_unique_nodes=*/9, /*seed_ids=*/seeds,
                       /*split=*/1);
}

TEST(BlockStore, OpenRejectsNegativeEdgeCount) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_neg_edges";
    std::filesystem::create_directories(tmp);
    write_k1_block(tmp / "b.blk");
    const int64_t neg = -1;
    patch_bytes(tmp / "b.blk", /*off=E[0]*/80, &neg, sizeof(neg));
    EXPECT_FALSE(BlockReader::open(tmp / "b.blk", /*expected=*/0xFEEDull).has_value());
    EXPECT_FALSE(BlockReader::open_self_contained(tmp / "b.blk", /*expected=*/0xBEEFull).has_value());
}

TEST(BlockStore, OpenRejectsHugeSeedCount) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_huge_seeds";
    std::filesystem::create_directories(tmp);
    write_k1_block(tmp / "b.blk");
    const uint64_t huge = 0x4000000000000000ull;
    patch_bytes(tmp / "b.blk", /*off=num_seeds*/48, &huge, sizeof(huge));
    EXPECT_FALSE(BlockReader::open(tmp / "b.blk", /*expected=*/0xFEEDull).has_value());
    EXPECT_FALSE(BlockReader::open_self_contained(tmp / "b.blk", /*expected=*/0xBEEFull).has_value());
}

TEST(BlockStore, OpenRejectsOutOfBoundLayerCount) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_huge_layers";
    std::filesystem::create_directories(tmp);
    write_k1_block(tmp / "b.blk");
    const uint32_t layers = 65u;  // just past the reader's sanity bound
    patch_bytes(tmp / "b.blk", /*off=num_layers*/16, &layers, sizeof(layers));
    EXPECT_FALSE(BlockReader::open(tmp / "b.blk", /*expected=*/0xFEEDull).has_value());
}

TEST(BlockStore, OpenRejectsBodySizeMismatch) {
    auto tmp = std::filesystem::temp_directory_path() / "blk_size_mismatch";
    std::filesystem::create_directories(tmp);
    write_k1_block(tmp / "b.blk");
    // Inflate E[0] within the int32 bound: the implied body size no longer
    // matches the file size, so the reader must reject rather than decode
    // the seed tail as edge bytes.
    const int64_t inflated = 3;
    patch_bytes(tmp / "b.blk", /*off=E[0]*/80, &inflated, sizeof(inflated));
    EXPECT_FALSE(BlockReader::open(tmp / "b.blk", /*expected=*/0xFEEDull).has_value());
}
