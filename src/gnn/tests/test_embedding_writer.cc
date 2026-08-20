/**
 * @file test_embedding_writer.cc
 * @brief Unit tests for EmbeddingWriter.
 *
 * Phase C (write_to_projection) requires TensorManager + ProjectionStorage +
 * B+Tree infrastructure backed by an initialized System, which is covered by
 * the E2E test (gnn_e2e_test.sh).
 *
 * Coverage levels in this file — NOTE the distinction:
 *   - Tests 1-12 validate the Phase A/B ALGORITHM SPEC by re-implementing
 *     the collection / dedup / missing-node logic inline (the production
 *     methods are private). A regression inside EmbeddingWriter's own
 *     implementation of that logic is caught by the E2E gate, NOT by these.
 *   - Tests 13-16 drive the REAL production key-allocation statics
 *     (next_available_key_id / resolve_property_key_id).
 *   - Tests 17-18 construct a REAL EmbeddingWriter and drive write_all() on
 *     the orchestration paths executable without the full database stack.
 *   - Tests 19-20 cover the Coverage option's default and the arithmetic
 *     behind the Coverage::ALL affordability gate.
 *
 * Strategy:
 *   - Use the same GnnStorageTest fixture pattern as test_batch_assembler.cc
 *   - Build a small synthetic graph with known features and topology
 *   - Construct a GraphSAGEModel, SampleStorage, BatchAssembler, RowMapping
 *   - Invoke the model.get_embeddings() path (Phase A logic) directly
 *   - Verify shapes, deduplication, and missing-node detection
 */

// Include the System / storage headers FIRST, before any GNN header that
// transitively pulls in <linux/fs.h> (via liburing.h -> direct_io_reader.h),
// which #defines BLOCK_SIZE as a macro and would break the
// StringManager::BLOCK_SIZE member declaration. This mirrors the ordering
// constraint documented at the top of embedding_writer.cc.
#include "system/system.h"

#include "query/query_context.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"
#include "storage/page/page.h"
#include "system/buffer_manager.h"
#include "system/file_manager.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

#include <torch/torch.h>

#include "gnn/models/graphsage_model.h"
#include "gnn/output/embedding_writer.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_config.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/batch_assembler.h"
#include "gnn/training/label_store.h"
#include "gnn/training/mini_batch.h"
#include "gnn/training/split_store.h"
#include "gnn/tests/test_helpers.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/object_id.h"

namespace fs = std::filesystem;
using namespace mdb::gnn;

// =============================================================================
// Test Fixture
// =============================================================================
//
// Synthetic graph (N=8 nodes, D=4 features, 2 classes):
//
//   Class 0 nodes (OIDs 0..3): features are large positive values.
//   Class 1 nodes (OIDs 4..7): features are large negative values.
//
// This is the same pattern used in test_training_loop.cc, providing a
// linearly-separable dataset that a 1-layer GraphSAGE model can learn.
//
// Batch layout (3 train batches):
//   Batch 0: seeds = nodes 0, 1, 2, 3   (covers rows 0..3)
//   Batch 1: seeds = nodes 4, 5, 6, 7   (covers rows 4..7)
//   Batch 2: seeds = nodes 0, 2, 4, 6   (OVERLAP with batches 0 and 1)
//
// Batch 2 overlaps with 0 and 1 to test deduplication.
//
// Each batch is a 1-layer GraphSample with self-loop edges so that the
// model produces [num_seeds, hidden_dim] embeddings per batch.
//
// =============================================================================

class EmbeddingWriterTest : public GnnStorageTest {
protected:
    static constexpr uint64_t N           = 8;
    static constexpr uint64_t D           = 4;
    static constexpr uint64_t NUM_CLASSES = 2;

    std::vector<ObjectId> node_oids_;
    fs::path db_folder_;
    fs::path gnn_dir_;
    fs::path fmat_path_;
    fs::path rmap_path_;
    fs::path labels_path_;
    fs::path splits_path_;

    void SetUp() override {
        GnnStorageTest::SetUp();

        db_folder_ = test_dir_ / "emb_db";
        gnn_dir_   = db_folder_ / "gnn_data";
        fs::create_directories(gnn_dir_);

        fmat_path_   = gnn_dir_ / "features.fmat";
        rmap_path_   = gnn_dir_ / "nodes.rmap";
        labels_path_ = gnn_dir_ / "labels.bin";
        splits_path_ = gnn_dir_ / "splits.bin";

        // Build node OIDs
        node_oids_.resize(N);
        for (uint64_t i = 0; i < N; ++i) {
            node_oids_[i] = ObjectId(0xD400000000000000ULL | i);
        }

        // Features: class 0 (i<4) -> positive, class 1 (i>=4) -> negative
        std::vector<float> features(N * D);
        for (uint64_t r = 0; r < N; ++r) {
            float sign = (r < 4) ? +1.0f : -1.0f;
            for (uint64_t c = 0; c < D; ++c) {
                features[r * D + c] = sign * static_cast<float>((c + 1) * 100);
            }
        }
        FeatureMatrix::create(fmat_path_, N, D, GnnDtype::FLOAT32, features.data());
        RowMapping::create(rmap_path_, node_oids_);

        // Labels: 0..3 -> class 0, 4..7 -> class 1
        std::vector<int64_t> labels(N);
        for (uint64_t i = 0; i < N; ++i) {
            labels[i] = (i < 4) ? 0 : 1;
        }
        LabelStore::write(labels_path_, labels, NUM_CLASSES);

        // Splits: all TRAIN
        std::vector<uint8_t> splits(N, SplitStore::TRAIN);
        SplitStore::write(splits_path_, splits);
    }

    // -----------------------------------------------------------------------
    // Build a 1-layer GraphSample with self-loop edges (seeds are their own
    // neighbors).  Matches the pattern from test_training_loop.cc.
    // -----------------------------------------------------------------------
    GraphSample make_seed_sample(
        const std::vector<ObjectId>& seeds,
        uint64_t batch_id,
        SplitType split = SplitType::TRAIN
    ) const
    {
        GraphSample s;
        s.batch_id = batch_id;
        s.split    = split;

        const size_t M = seeds.size();

        // Layer 0: seed nodes (targets)
        // Layer 1: same seeds as "neighbors"
        s.nodes_per_layer.resize(2);
        s.nodes_per_layer[0] = seeds;
        s.nodes_per_layer[1] = seeds;

        // Self-loop edges: layer1[i] -> layer0[i]
        s.edges_per_layer.resize(1);
        LayerEdges& le = s.edges_per_layer[0];
        le.src_indices.resize(M);
        le.dst_indices.resize(M);
        le.edge_ids.resize(M, ObjectId(0));
        for (size_t i = 0; i < M; ++i) {
            le.src_indices[i] = static_cast<int32_t>(i);
            le.dst_indices[i] = static_cast<int32_t>(i);
        }

        // all_unique_nodes: seeds only (deduplicated since both layers are identical)
        s.all_unique_nodes = seeds;
        return s;
    }

    // -----------------------------------------------------------------------
    // Create SampleStorage with configurable batch layout.
    //
    // Default: 3 batches covering all 8 nodes with overlap in batch 2.
    // -----------------------------------------------------------------------
    struct StorageResult {
        SampleStorage storage;
        SampleCatalog catalog;
    };

    StorageResult create_storage_with_overlap(const std::string& name) {
        SamplingConfig config;
        config.projection_name = "test_proj";
        config.sample_name     = name;
        config.fanouts         = {4};   // 1-hop, fanout 4
        config.batch_size      = 4;
        config.train_ratio     = 1.0;
        config.val_ratio       = 0.0;
        config.test_ratio      = 0.0;

        auto storage = SampleStorage::create(db_folder_, config);

        // Batch 0: seeds = nodes 0, 1, 2, 3
        storage.write_sample(make_seed_sample(
            {node_oids_[0], node_oids_[1], node_oids_[2], node_oids_[3]},
            0, SplitType::TRAIN));

        // Batch 1: seeds = nodes 4, 5, 6, 7
        storage.write_sample(make_seed_sample(
            {node_oids_[4], node_oids_[5], node_oids_[6], node_oids_[7]},
            1, SplitType::TRAIN));

        // Batch 2: seeds = nodes 0, 2, 4, 6 (OVERLAP with batches 0, 1)
        storage.write_sample(make_seed_sample(
            {node_oids_[0], node_oids_[2], node_oids_[4], node_oids_[6]},
            2, SplitType::TRAIN));

        storage.finalize();

        SampleCatalog cat;
        cat.projection_name    = "test_proj";
        cat.sample_name        = name;
        cat.batch_size         = 4;
        cat.random_seed        = 0;
        cat.fanouts            = {4};
        cat.total_batches      = 3;
        cat.train_batches      = 3;
        cat.validation_batches = 0;
        cat.test_batches       = 0;
        cat.unique_nodes       = N;
        cat.total_edges        = 0;

        auto opened = SampleStorage::open(
            SampleStorage::get_storage_path(db_folder_, name));

        return StorageResult{std::move(opened), cat};
    }

    // Non-overlapping: exactly 2 batches that cover all 8 nodes once each.
    StorageResult create_storage_no_overlap(const std::string& name) {
        SamplingConfig config;
        config.projection_name = "test_proj";
        config.sample_name     = name;
        config.fanouts         = {4};
        config.batch_size      = 4;
        config.train_ratio     = 1.0;
        config.val_ratio       = 0.0;
        config.test_ratio      = 0.0;

        auto storage = SampleStorage::create(db_folder_, config);

        // Batch 0: nodes 0..3
        storage.write_sample(make_seed_sample(
            {node_oids_[0], node_oids_[1], node_oids_[2], node_oids_[3]},
            0, SplitType::TRAIN));

        // Batch 1: nodes 4..7
        storage.write_sample(make_seed_sample(
            {node_oids_[4], node_oids_[5], node_oids_[6], node_oids_[7]},
            1, SplitType::TRAIN));

        storage.finalize();

        SampleCatalog cat;
        cat.projection_name    = "test_proj";
        cat.sample_name        = name;
        cat.batch_size         = 4;
        cat.random_seed        = 0;
        cat.fanouts            = {4};
        cat.total_batches      = 2;
        cat.train_batches      = 2;
        cat.validation_batches = 0;
        cat.test_batches       = 0;
        cat.unique_nodes       = N;
        cat.total_edges        = 0;

        auto opened = SampleStorage::open(
            SampleStorage::get_storage_path(db_folder_, name));

        return StorageResult{std::move(opened), cat};
    }

    // Single empty batch (0 seeds).
    StorageResult create_storage_empty_batch(const std::string& name) {
        SamplingConfig config;
        config.projection_name = "test_proj";
        config.sample_name     = name;
        config.fanouts         = {4};
        config.batch_size      = 4;
        config.train_ratio     = 1.0;
        config.val_ratio       = 0.0;
        config.test_ratio      = 0.0;

        auto storage = SampleStorage::create(db_folder_, config);

        // Batch 0: empty (no seeds)
        GraphSample empty;
        empty.batch_id = 0;
        empty.split    = SplitType::TRAIN;
        empty.nodes_per_layer.resize(2);
        // both layers empty
        empty.edges_per_layer.resize(1);
        storage.write_sample(empty);

        storage.finalize();

        SampleCatalog cat;
        cat.projection_name    = "test_proj";
        cat.sample_name        = name;
        cat.batch_size         = 4;
        cat.random_seed        = 0;
        cat.fanouts            = {4};
        cat.total_batches      = 1;
        cat.train_batches      = 1;
        cat.validation_batches = 0;
        cat.test_batches       = 0;
        cat.unique_nodes       = 0;
        cat.total_edges        = 0;

        auto opened = SampleStorage::open(
            SampleStorage::get_storage_path(db_folder_, name));

        return StorageResult{std::move(opened), cat};
    }

    // Build a GraphSAGEModel with deterministic weights.
    GraphSAGEModel make_model() {
        GraphSAGEConfig cfg{
            .input_dim   = static_cast<int64_t>(D),
            .hidden_dim  = 16,
            .num_classes = static_cast<int64_t>(NUM_CLASSES),
            .num_layers  = 1,
            .dropout     = 0.0,   // deterministic forward pass
            .normalize   = false,
        };
        GraphSAGEModel model(cfg);
        // Fix random seed for reproducibility
        torch::manual_seed(42);
        return model;
    }
};

// =============================================================================
// Test 1: SeedEmbeddingShapes
//
// Verify that iterating all batches and running model.get_embeddings()
// produces tensors of the expected shape [num_seeds, hidden_dim].
// This exercises the core Phase A logic without the full EmbeddingWriter.
// =============================================================================

TEST_F(EmbeddingWriterTest, SeedEmbeddingShapes) {
    const std::string sname = "shapes_test";
    auto [storage, catalog] = create_storage_no_overlap(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);
    auto ls = LabelStore::open(labels_path_);

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    const int64_t hidden_dim = model.config().hidden_dim;

    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        MiniBatch mini = assembler.assemble(bid);
        GraphSample sample = storage.read_sample(bid);

        const auto& seed_oids = sample.nodes_per_layer[0];
        auto num_seeds = static_cast<int64_t>(seed_oids.size());

        ASSERT_GT(num_seeds, 0) << "Batch " << bid << " has no seeds";

        // get_embeddings returns [num_seeds, hidden_dim]
        auto emb = model.get_embeddings(
            mini.features, mini.edge_indices, mini.active_sizes_per_layer);

        ASSERT_EQ(emb.dim(), 2) << "Embedding tensor not 2D at batch " << bid;
        EXPECT_EQ(emb.size(0), num_seeds)
            << "Embedding row count mismatch at batch " << bid;
        EXPECT_EQ(emb.size(1), hidden_dim)
            << "Embedding column count mismatch at batch " << bid;

        // Ensure no NaN or Inf
        EXPECT_FALSE(emb.isnan().any().item<bool>())
            << "NaN in embeddings at batch " << bid;
        EXPECT_FALSE(emb.isinf().any().item<bool>())
            << "Inf in embeddings at batch " << bid;
    }
}

// =============================================================================
// Test 2: SeedToRowMapping
//
// Verify that seed ObjectIds from GraphSample.nodes_per_layer[0] can be
// correctly mapped back to RowMapping indices.  This is the critical
// oid -> row_index mapping step in collect_seed_embeddings.
// =============================================================================

TEST_F(EmbeddingWriterTest, SeedToRowMapping) {
    const std::string sname = "row_mapping_test";
    auto [storage, catalog] = create_storage_no_overlap(sname);

    auto rm = RowMapping::open(rmap_path_);

    // For each batch, every seed should be findable in the RowMapping
    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        GraphSample sample = storage.read_sample(bid);
        const auto& seed_oids = sample.nodes_per_layer[0];

        for (size_t i = 0; i < seed_oids.size(); ++i) {
            auto row_opt = rm.find(seed_oids[i]);
            ASSERT_TRUE(row_opt.has_value())
                << "Seed " << i << " in batch " << bid
                << " (oid=0x" << std::hex << seed_oids[i].id << std::dec
                << ") not found in RowMapping";

            // Row index should be in [0, N)
            EXPECT_LT(*row_opt, N)
                << "Row index out of range for seed " << i << " in batch " << bid;
        }
    }
}

// =============================================================================
// Test 3: DeduplicationAcrossBatches
//
// When a node appears as a seed in multiple batches, the deduplication
// logic (emb_map[idx] = ...) should keep exactly one embedding per node.
// Batch 2 in the overlap layout shares 4 nodes with batches 0 and 1.
//
// This test replicates the Phase A collection + dedup logic from write_all().
// =============================================================================

TEST_F(EmbeddingWriterTest, DeduplicationAcrossBatches) {
    const std::string sname = "dedup_test";
    auto [storage, catalog] = create_storage_with_overlap(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);
    auto ls = LabelStore::open(labels_path_);

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    // Phase A: collect seed embeddings (mirrors EmbeddingWriter::collect_seed_embeddings)
    std::vector<std::pair<uint64_t, torch::Tensor>> seed_embs;

    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        MiniBatch mini = assembler.assemble(bid);
        GraphSample sample = storage.read_sample(bid);

        const auto& seed_oids = sample.nodes_per_layer[0];
        auto num_seeds = static_cast<int64_t>(seed_oids.size());

        if (num_seeds == 0) continue;

        auto emb = model.get_embeddings(
            mini.features, mini.edge_indices, mini.active_sizes_per_layer);
        emb = emb.cpu().contiguous();

        for (int64_t i = 0; i < num_seeds; ++i) {
            auto row_opt = rm.find(seed_oids[static_cast<size_t>(i)]);
            if (row_opt) {
                seed_embs.emplace_back(*row_opt, emb[i].clone());
            }
        }
    }

    // Without dedup: we have 4 + 4 + 4 = 12 entries (batch 2 overlaps)
    EXPECT_EQ(seed_embs.size(), 12u) << "Raw collection should have 12 entries";

    // Deduplicate: last-write-wins (same as EmbeddingWriter::write_all)
    std::unordered_map<uint64_t, torch::Tensor> emb_map;
    emb_map.reserve(seed_embs.size());
    for (auto& [idx, emb] : seed_embs) {
        emb_map[idx] = std::move(emb);
    }

    // After dedup: exactly N=8 unique row indices
    EXPECT_EQ(emb_map.size(), N)
        << "After deduplication, exactly " << N << " unique embeddings expected";

    // Every row index in [0, N) should be present
    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_NE(emb_map.find(i), emb_map.end())
            << "Missing embedding for row index " << i;
    }
}

// =============================================================================
// Test 4: MissingNodeDetection
//
// When batches cover only a subset of nodes, the remaining nodes should
// be identified as "missing".  This tests the Phase B setup logic.
//
// Layout: 2 batches covering nodes 0..7, but RowMapping has 8 nodes.
// If we artificially use only 1 batch (nodes 0..3), nodes 4..7 are missing.
// =============================================================================

TEST_F(EmbeddingWriterTest, MissingNodeDetection) {
    const std::string sname = "missing_test";
    auto [storage, catalog] = create_storage_no_overlap(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);
    auto ls = LabelStore::open(labels_path_);

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    // Only process batch 0 (seeds = nodes 0..3)
    // This simulates a scenario where half the nodes were never seeds.
    std::unordered_map<uint64_t, torch::Tensor> emb_map;
    {
        MiniBatch mini = assembler.assemble(0);
        GraphSample sample = storage.read_sample(0);

        const auto& seed_oids = sample.nodes_per_layer[0];
        auto num_seeds = static_cast<int64_t>(seed_oids.size());

        auto emb = model.get_embeddings(
            mini.features, mini.edge_indices, mini.active_sizes_per_layer);
        emb = emb.cpu().contiguous();

        for (int64_t i = 0; i < num_seeds; ++i) {
            auto row_opt = rm.find(seed_oids[static_cast<size_t>(i)]);
            if (row_opt) {
                emb_map[*row_opt] = emb[i].clone();
            }
        }
    }

    // Should have 4 embeddings (nodes 0..3)
    ASSERT_EQ(emb_map.size(), 4u);

    // Identify missing nodes (same logic as EmbeddingWriter::write_all)
    std::vector<uint64_t> missing;
    for (uint64_t i = 0; i < rm.size(); ++i) {
        if (emb_map.find(i) == emb_map.end()) {
            missing.push_back(i);
        }
    }

    // Nodes 4..7 should be missing
    ASSERT_EQ(missing.size(), 4u);
    std::sort(missing.begin(), missing.end());
    EXPECT_EQ(missing[0], 4u);
    EXPECT_EQ(missing[1], 5u);
    EXPECT_EQ(missing[2], 6u);
    EXPECT_EQ(missing[3], 7u);
}

// =============================================================================
// Test 5: AllNodesCoveredNoMissing
//
// When all nodes appear as seeds, the missing list should be empty.
// =============================================================================

TEST_F(EmbeddingWriterTest, AllNodesCoveredNoMissing) {
    const std::string sname = "all_covered_test";
    auto [storage, catalog] = create_storage_no_overlap(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);
    auto ls = LabelStore::open(labels_path_);

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    // Process ALL batches
    std::unordered_map<uint64_t, torch::Tensor> emb_map;

    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        MiniBatch mini = assembler.assemble(bid);
        GraphSample sample = storage.read_sample(bid);

        const auto& seed_oids = sample.nodes_per_layer[0];
        auto num_seeds = static_cast<int64_t>(seed_oids.size());

        if (num_seeds == 0) continue;

        auto emb = model.get_embeddings(
            mini.features, mini.edge_indices, mini.active_sizes_per_layer);
        emb = emb.cpu().contiguous();

        for (int64_t i = 0; i < num_seeds; ++i) {
            auto row_opt = rm.find(seed_oids[static_cast<size_t>(i)]);
            if (row_opt) {
                emb_map[*row_opt] = emb[i].clone();
            }
        }
    }

    // All 8 nodes covered
    EXPECT_EQ(emb_map.size(), N);

    // No missing nodes
    std::vector<uint64_t> missing;
    for (uint64_t i = 0; i < rm.size(); ++i) {
        if (emb_map.find(i) == emb_map.end()) {
            missing.push_back(i);
        }
    }
    EXPECT_TRUE(missing.empty())
        << "Expected no missing nodes, found " << missing.size();
}

// =============================================================================
// Test 6: EmptyBatchProducesNoEmbeddings
//
// A batch with zero seeds should be safely skipped.
// =============================================================================

TEST_F(EmbeddingWriterTest, EmptyBatchProducesNoEmbeddings) {
    const std::string sname = "empty_batch_test";
    auto [storage, catalog] = create_storage_empty_batch(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    // The single batch has 0 seeds
    std::unordered_map<uint64_t, torch::Tensor> emb_map;

    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        GraphSample sample = storage.read_sample(bid);
        const auto& seed_oids = sample.nodes_per_layer[0];
        auto num_seeds = static_cast<int64_t>(seed_oids.size());

        if (num_seeds == 0) {
            // EmbeddingWriter skips empty batches — verify this path
            continue;
        }

        MiniBatch mini = assembler.assemble(bid);
        auto emb = model.get_embeddings(
            mini.features, mini.edge_indices, mini.active_sizes_per_layer);
        emb = emb.cpu().contiguous();

        for (int64_t i = 0; i < num_seeds; ++i) {
            auto row_opt = rm.find(seed_oids[static_cast<size_t>(i)]);
            if (row_opt) {
                emb_map[*row_opt] = emb[i].clone();
            }
        }
    }

    EXPECT_TRUE(emb_map.empty())
        << "Empty batch should produce no embeddings";

    // All nodes are "missing" since nothing was collected
    std::vector<uint64_t> missing;
    for (uint64_t i = 0; i < rm.size(); ++i) {
        if (emb_map.find(i) == emb_map.end()) {
            missing.push_back(i);
        }
    }
    EXPECT_EQ(missing.size(), N);
}

// =============================================================================
// Test 7: EmbeddingsDeterministic
//
// Running the same model on the same batch twice should produce identical
// embeddings (in eval mode, no dropout).
// =============================================================================

TEST_F(EmbeddingWriterTest, EmbeddingsDeterministic) {
    const std::string sname = "determinism_test";
    auto [storage, catalog] = create_storage_no_overlap(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    // Run batch 0 twice
    MiniBatch mini1 = assembler.assemble(0);
    GraphSample sample1 = storage.read_sample(0);
    auto num_seeds = static_cast<int64_t>(sample1.nodes_per_layer[0].size());

    auto emb1 = model.get_embeddings(mini1.features, mini1.edge_indices, mini1.active_sizes_per_layer);

    MiniBatch mini2 = assembler.assemble(0);
    auto emb2 = model.get_embeddings(mini2.features, mini2.edge_indices, mini2.active_sizes_per_layer);

    // Should be identical (eval mode, no dropout, same input)
    auto diff = (emb1 - emb2).abs().max().item<float>();
    EXPECT_FLOAT_EQ(diff, 0.0f)
        << "Embeddings should be deterministic in eval mode";
}

// =============================================================================
// Test 8: DeduplicatedEmbeddingIsFromLastBatch
//
// When a node is a seed in multiple batches, EmbeddingWriter keeps the
// last one (later batch_id overwrites earlier).  Verify this behavior.
// =============================================================================

TEST_F(EmbeddingWriterTest, DeduplicatedEmbeddingIsFromLastBatch) {
    const std::string sname = "last_wins_test";
    auto [storage, catalog] = create_storage_with_overlap(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    // Collect embeddings per batch, tracking which batch produced each
    std::unordered_map<uint64_t, torch::Tensor> per_batch_emb;  // row_idx -> emb from last batch
    std::unordered_map<uint64_t, torch::Tensor> batch2_emb;     // row_idx -> emb from batch 2

    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        MiniBatch mini = assembler.assemble(bid);
        GraphSample sample = storage.read_sample(bid);

        const auto& seed_oids = sample.nodes_per_layer[0];
        auto num_seeds = static_cast<int64_t>(seed_oids.size());
        if (num_seeds == 0) continue;

        auto emb = model.get_embeddings(
            mini.features, mini.edge_indices, mini.active_sizes_per_layer);
        emb = emb.cpu().contiguous();

        for (int64_t i = 0; i < num_seeds; ++i) {
            auto row_opt = rm.find(seed_oids[static_cast<size_t>(i)]);
            if (!row_opt) continue;

            per_batch_emb[*row_opt] = emb[i].clone();

            if (bid == 2) {
                batch2_emb[*row_opt] = emb[i].clone();
            }
        }
    }

    // Nodes 0, 2, 4, 6 appear in batch 2 (last).
    // Their final embedding in per_batch_emb should match batch2_emb.
    for (uint64_t row_idx : {0u, 2u, 4u, 6u}) {
        ASSERT_NE(per_batch_emb.find(row_idx), per_batch_emb.end());
        ASSERT_NE(batch2_emb.find(row_idx), batch2_emb.end());

        auto diff = (per_batch_emb[row_idx] - batch2_emb[row_idx]).abs().max().item<float>();
        EXPECT_FLOAT_EQ(diff, 0.0f)
            << "Row " << row_idx << ": deduped embedding should match batch 2";
    }
}

// =============================================================================
// Test 9: ModelGetEmbeddingsVsForward
//
// Verify that get_embeddings() produces [num_seeds, hidden_dim] while
// forward() produces [num_seeds, num_classes].  Both must be finite.
// =============================================================================

TEST_F(EmbeddingWriterTest, ModelGetEmbeddingsVsForward) {
    const std::string sname = "emb_vs_fwd_test";
    auto [storage, catalog] = create_storage_no_overlap(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    MiniBatch mini = assembler.assemble(0);
    GraphSample sample = storage.read_sample(0);
    auto num_seeds = static_cast<int64_t>(sample.nodes_per_layer[0].size());

    auto emb = model.get_embeddings(mini.features, mini.edge_indices, mini.active_sizes_per_layer);
    auto logits = model.forward(mini.features, mini.edge_indices, mini.active_sizes_per_layer);

    // Shapes must differ: embeddings are hidden_dim, logits are num_classes
    EXPECT_EQ(emb.size(0), num_seeds);
    EXPECT_EQ(emb.size(1), model.config().hidden_dim);

    EXPECT_EQ(logits.size(0), num_seeds);
    EXPECT_EQ(logits.size(1), model.config().num_classes);

    // Both must be finite
    EXPECT_FALSE(emb.isnan().any().item<bool>());
    EXPECT_FALSE(logits.isnan().any().item<bool>());
}

// =============================================================================
// Test 10: EmbeddingDimensionConsistency
//
// All embeddings across all batches must have the same hidden_dim.
// =============================================================================

TEST_F(EmbeddingWriterTest, EmbeddingDimensionConsistency) {
    const std::string sname = "dim_consistency_test";
    auto [storage, catalog] = create_storage_with_overlap(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);

    BatchAssembler assembler(fm, storage, nullptr, nullptr, rm);
    GraphSAGEModel model = make_model();
    model.eval();
    torch::NoGradGuard no_grad;

    int64_t expected_dim = model.config().hidden_dim;
    std::unordered_map<uint64_t, torch::Tensor> emb_map;

    for (uint64_t bid = 0; bid < catalog.total_batches; ++bid) {
        MiniBatch mini = assembler.assemble(bid);
        GraphSample sample = storage.read_sample(bid);

        const auto& seed_oids = sample.nodes_per_layer[0];
        auto num_seeds = static_cast<int64_t>(seed_oids.size());
        if (num_seeds == 0) continue;

        auto emb = model.get_embeddings(
            mini.features, mini.edge_indices, mini.active_sizes_per_layer);
        emb = emb.cpu().contiguous();

        EXPECT_EQ(emb.size(1), expected_dim)
            << "Hidden dim mismatch at batch " << bid;

        for (int64_t i = 0; i < num_seeds; ++i) {
            auto row_opt = rm.find(seed_oids[static_cast<size_t>(i)]);
            if (row_opt) {
                emb_map[*row_opt] = emb[i].clone();
            }
        }
    }

    // After dedup, verify all stored embeddings have the same dimension
    for (const auto& [idx, tensor] : emb_map) {
        EXPECT_EQ(tensor.size(0), expected_dim)
            << "Stored embedding at row " << idx << " has wrong dimension";
    }
}

// =============================================================================
// Test 11: TensorSerializationRoundTrip
//
// Verify that a float tensor can be serialized to raw bytes, round-tripped
// through memory, and reconstructed identically.  This is the serialization
// step that Phase C uses before calling TensorManager (tested without
// requiring the actual TensorManager infrastructure).
// =============================================================================

TEST_F(EmbeddingWriterTest, TensorSerializationRoundTrip) {
    // Create a known embedding tensor
    auto emb = torch::randn({64});
    auto emb_cpu = emb.contiguous().to(torch::kFloat32);

    // Serialize to raw bytes (same as embedding_writer.cc Phase C)
    const auto* bytes = reinterpret_cast<const char*>(emb_cpu.data_ptr<float>());
    size_t num_bytes = static_cast<size_t>(emb_cpu.numel()) * sizeof(float);

    ASSERT_EQ(num_bytes, 64 * sizeof(float));

    // "Round-trip": reconstruct tensor from raw bytes
    auto reconstructed = torch::from_blob(
        const_cast<char*>(bytes),
        {64},
        torch::kFloat32
    ).clone();  // clone to own the memory

    // Verify byte-identical reconstruction
    ASSERT_EQ(reconstructed.numel(), emb_cpu.numel());
    auto diff = (reconstructed - emb_cpu).abs().max().item<float>();
    EXPECT_FLOAT_EQ(diff, 0.0f)
        << "Tensor serialization round-trip should be exact";
}

// =============================================================================
// Test 12: TensorSerializationMultiDim
//
// Same round-trip but with a 2D tensor [4, 16] to verify that contiguous()
// + raw bytes serialization works for multi-dimensional embeddings.
// =============================================================================

TEST_F(EmbeddingWriterTest, TensorSerializationMultiDim) {
    auto emb = torch::randn({4, 16});
    auto emb_cpu = emb.contiguous().to(torch::kFloat32);

    const auto* bytes = reinterpret_cast<const char*>(emb_cpu.data_ptr<float>());
    size_t num_bytes = static_cast<size_t>(emb_cpu.numel()) * sizeof(float);
    ASSERT_EQ(num_bytes, 4 * 16 * sizeof(float));

    // Round-trip as a flat tensor
    auto flat = torch::from_blob(
        const_cast<char*>(bytes),
        {4 * 16},
        torch::kFloat32
    ).clone();

    // Reshape back and compare
    auto reconstructed = flat.reshape({4, 16});
    auto diff = (reconstructed - emb_cpu).abs().max().item<float>();
    EXPECT_FLOAT_EQ(diff, 0.0f);
}

// =============================================================================
// Test 13: NextAvailableKeyIdStartsAtSyntheticBase
//
// With no keys registered, allocation starts at the synthetic base shared
// with NativeProjectionBuilder's rename-key range.
// =============================================================================

TEST_F(EmbeddingWriterTest, NextAvailableKeyIdStartsAtSyntheticBase) {
    std::unordered_map<std::string, uint64_t> node_keys;
    std::unordered_map<std::string, uint64_t> edge_keys;

    EXPECT_EQ(EmbeddingWriter::next_available_key_id(node_keys, edge_keys),
              EmbeddingWriter::EMBEDDING_KEY_SYNTHETIC_BASE);
}

// =============================================================================
// Test 14: NextAvailableKeyIdGoesPastUsedIds
//
// Allocation must clear every id already in use in EITHER namespace —
// including builder-synthetic ids at/above the base (e.g. a renamed property
// at 2000) and edge keys (the builder allocates node and edge synthetic ids
// from one shared counter).
// =============================================================================

TEST_F(EmbeddingWriterTest, NextAvailableKeyIdGoesPastUsedIds) {
    std::unordered_map<std::string, uint64_t> node_keys{
        { "feat", 5 }, { "renamed", 2000 }
    };
    std::unordered_map<std::string, uint64_t> edge_keys{
        { "_count", 1000 }, { "weight", 2047 }
    };

    EXPECT_EQ(EmbeddingWriter::next_available_key_id(node_keys, edge_keys), 2048u);

    // Ids below the base never pull allocation under it.
    std::unordered_map<std::string, uint64_t> low_only{ { "feat", 3 } };
    std::unordered_map<std::string, uint64_t> empty;
    EXPECT_EQ(EmbeddingWriter::next_available_key_id(low_only, empty),
              EmbeddingWriter::EMBEDDING_KEY_SYNTHETIC_BASE);
}

// =============================================================================
// Test 15: ResolveDistinctIdsForDistinctProperties
//
// Writing two different property names must bind two DISTINCT key ids;
// a hardcoded id would alias them, making n.emb_b silently resolve to
// emb_a's records.  Resolving an already-registered name is idempotent.
// =============================================================================

TEST_F(EmbeddingWriterTest, ResolveDistinctIdsForDistinctProperties) {
    fs::path proj_dir = test_dir_ / "proj_alloc";
    fs::create_directories(proj_dir);
    GQL::ProjectionStorage storage(
        proj_dir.string(), test_dir_.string(), "proj_alloc");

    uint64_t id_a = EmbeddingWriter::resolve_property_key_id(storage, "emb_a");
    uint64_t id_b = EmbeddingWriter::resolve_property_key_id(storage, "emb_b");

    EXPECT_NE(id_a, id_b)
        << "two writeProperty names must not alias one key id";

    // Both bindings are registered and stable on re-resolution.
    EXPECT_EQ(EmbeddingWriter::resolve_property_key_id(storage, "emb_a"), id_a);
    EXPECT_EQ(EmbeddingWriter::resolve_property_key_id(storage, "emb_b"), id_b);
    EXPECT_EQ(storage.get_node_key_id("emb_a"), std::optional<uint64_t>(id_a));
    EXPECT_EQ(storage.get_node_key_id("emb_b"), std::optional<uint64_t>(id_b));
}

// =============================================================================
// Test 16: ResolveReusesPersistedCatalogKeys
//
// ProjectionStorage::open() does not restore the key maps from the on-disk
// catalog.  resolve_property_key_id must re-sync them so that (a) a property
// persisted by an earlier session keeps its id, (b) a NEW property does not
// get allocated an id that aliases it, and (c) the merged map carries every
// key forward so the next save_catalog() drops none.
// =============================================================================

TEST_F(EmbeddingWriterTest, ResolveReusesPersistedCatalogKeys) {
    fs::path proj_dir = test_dir_ / "proj_catalog";
    fs::create_directories(proj_dir);

    uint64_t id_a = 0;
    {
        GQL::ProjectionStorage storage(
            proj_dir.string(), test_dir_.string(), "proj_catalog");
        id_a = EmbeddingWriter::resolve_property_key_id(storage, "emb_a");
        storage.flush();  // persists catalog.dat with the key mapping
    }
    ASSERT_TRUE(fs::exists(proj_dir / "catalog.dat"));

    // Fresh storage on the same dir starts with empty in-memory key maps
    // (mirroring the open() path).
    GQL::ProjectionStorage storage2(
        proj_dir.string(), test_dir_.string(), "proj_catalog");
    ASSERT_TRUE(storage2.get_node_keys().empty());

    // A NEW property must be allocated past the persisted 'emb_a' id...
    uint64_t id_b = EmbeddingWriter::resolve_property_key_id(storage2, "emb_b");
    EXPECT_NE(id_b, id_a)
        << "second-session property must not alias the persisted key";

    // ...and the persisted property keeps its original id.
    EXPECT_EQ(EmbeddingWriter::resolve_property_key_id(storage2, "emb_a"), id_a);

    // Both keys are now live in the in-memory map, so a subsequent
    // save_catalog() persists the union instead of dropping 'emb_a'.
    EXPECT_EQ(storage2.get_node_keys().size(), 2u);
}

// =============================================================================
// Test 17: WriteAllRunsRealOrchestration
//
// Construct a REAL EmbeddingWriter (real model, assembler, storage, mapping,
// catalog, ProjectionStorage) and drive write_all() end-to-end on the only
// path executable without an initialized System (TensorManager + B+Tree
// buffer pool): a single empty batch with Phase B disabled (no fanouts).
//
// This exercises the instance entry points the per-phase tests above cannot
// reach: construction (which no longer opens a TopologyAccessor -- Phase B
// reaches the topology through the adjacency cache), the real Phase A batch
// loop (assemble + read_sample on the empty batch), missing-node detection,
// the Phase B skip condition, and the Phase C empty-map early return.
// =============================================================================

TEST_F(EmbeddingWriterTest, WriteAllRunsRealOrchestration) {
    const std::string sname = "real_writer_test";
    auto [storage, catalog] = create_storage_empty_batch(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);
    auto ls = LabelStore::open(labels_path_);

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);
    GraphSAGEModel model = make_model();

    fs::path proj_dir = test_dir_ / "proj_writer";
    fs::create_directories(proj_dir);
    GQL::ProjectionStorage proj_storage(
        proj_dir.string(), test_dir_.string(), "proj_writer");

    EmbeddingWriter::Config wconfig;
    wconfig.property_name       = "embedding";
    wconfig.batch_size          = 4;
    wconfig.fanouts             = {};  // Phase B disabled
    wconfig.feature_matrix_path = fmat_path_;

    EmbeddingWriter writer(model, assembler, storage, rm, catalog,
                           proj_storage, std::move(wconfig));

    EmbeddingWriter::Result res = writer.write_all();

    // The single batch has zero seeds and Phase B is disabled, so nothing
    // is collected, inferred, or written — but the full orchestration ran.
    EXPECT_EQ(res.nodes_written, 0u);
    EXPECT_EQ(res.nodes_inferred, 0u);
    EXPECT_GE(res.write_ms, 0.0);
}

// =============================================================================
// Test 18: SeedsCoverageSkipsNonSeedInference
//
// Coverage::SEEDS must skip the missing-node scan and Phase B even when the
// fanouts are non-empty and nodes are uncovered.  The empty-batch layout
// leaves all N nodes uncovered, so a Coverage::ALL run with these exact
// fanouts would enter Phase B; SEEDS must return with nothing inferred.
//
// This is the contract that makes the mode usable on papers100M: what is
// skipped is ~109.5 M on-the-fly k-hop inferences, not a formality.
// =============================================================================

TEST_F(EmbeddingWriterTest, SeedsCoverageSkipsNonSeedInference) {
    const std::string sname = "seeds_coverage_test";
    auto [storage, catalog] = create_storage_empty_batch(sname);

    auto fm = FeatureMatrix::open(fmat_path_);
    auto rm = RowMapping::open(rmap_path_);
    auto ls = LabelStore::open(labels_path_);

    BatchAssembler assembler(fm, storage, &ls, nullptr, rm);
    GraphSAGEModel model = make_model();

    fs::path proj_dir = test_dir_ / "proj_seeds_coverage";
    fs::create_directories(proj_dir);
    GQL::ProjectionStorage proj_storage(
        proj_dir.string(), test_dir_.string(), "proj_seeds_coverage");

    EmbeddingWriter::Config wconfig;
    wconfig.property_name       = "embedding";
    wconfig.batch_size          = 4;
    wconfig.fanouts             = {4};  // non-empty: Phase B is reachable
    wconfig.coverage            = EmbeddingWriter::Coverage::SEEDS;
    wconfig.feature_matrix_path = fmat_path_;

    EmbeddingWriter writer(model, assembler, storage, rm, catalog,
                           proj_storage, std::move(wconfig));

    EmbeddingWriter::Result res = writer.write_all();

    EXPECT_EQ(res.nodes_inferred, 0u)
        << "Coverage::SEEDS must not infer non-seed nodes";
    EXPECT_DOUBLE_EQ(res.inference_ms, 0.0)
        << "Phase B must not have run under Coverage::SEEDS";
    EXPECT_EQ(res.nodes_written, 0u)
        << "the only batch has no seeds, so there is nothing to write";
}

// =============================================================================
// Test 19: CoverageDefaultsToAll
//
// The new mode is opt-in: a Config left alone must still request the full
// coverage every pre-existing caller got.
// =============================================================================

TEST_F(EmbeddingWriterTest, CoverageDefaultsToAll) {
    EmbeddingWriter::Config cfg;
    EXPECT_TRUE(cfg.coverage == EmbeddingWriter::Coverage::ALL)
        << "default coverage must stay ALL -- the seeds-only mode is opt-in";
}

// =============================================================================
// Test 20: WriteBytesEstimateSeparatesSeedsFromFullCoverage
//
// write_all() refuses a Coverage::ALL run the host cannot hold, so the
// estimate behind that refusal has to be right at papers100M scale -- which no
// fixture can materialize.  Drive the estimators directly with that shape.
// =============================================================================

TEST_F(EmbeddingWriterTest, WriteBytesEstimateSeparatesSeedsFromFullCoverage) {
    constexpr uint64_t GB = 1024ull * 1024 * 1024;

    // ogbn-papers100M: 111,059,956 nodes and 1,615,685,872 edges, the latter
    // scanned in both directions by the UNDIRECTED adjacency cache.
    constexpr uint64_t PAPERS_NODES = 111059956ull;
    constexpr uint64_t PAPERS_ADJ   = 2ull * 1615685872ull;
    constexpr uint64_t HIDDEN       = 256;

    const uint64_t all_bytes = EmbeddingWriter::estimate_all_write_bytes(
        PAPERS_NODES, HIDDEN, PAPERS_ADJ);
    EXPECT_GT(all_bytes, 100 * GB)
        << "full coverage on papers100M must estimate far past a 30 GiB host";

    // The labelled train/val/test set is ~1.55 M nodes -- a couple of GB.
    constexpr uint64_t LABELLED = 1546782ull;
    const uint64_t seed_bytes =
        EmbeddingWriter::estimate_seed_write_bytes(LABELLED, HIDDEN);
    EXPECT_GT(seed_bytes, 1 * GB);
    EXPECT_LT(seed_bytes, 4 * GB);

    // Full coverage is the seed cost over every node, plus the scan structures
    // only that mode builds.
    EXPECT_GT(all_bytes,
              EmbeddingWriter::estimate_seed_write_bytes(PAPERS_NODES, HIDDEN));

    // An empty projection costs nothing, so it can never trip the gate.
    EXPECT_EQ(EmbeddingWriter::estimate_seed_write_bytes(0, HIDDEN), 0u);
    EXPECT_EQ(EmbeddingWriter::estimate_all_write_bytes(0, HIDDEN, 0), 0u);
}

// The directory-split separator regression that this suite once carried now
// lives in src/tests/bpt_dir_split_separator_test.cc. It is a defect in core
// storage, and this suite only builds under ENABLE_GNN, which is OFF by
// default, so the gate belonged somewhere a default build always compiles.
