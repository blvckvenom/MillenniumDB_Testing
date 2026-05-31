#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sampling_config.h"
#include "graph_models/object_id.h"

using namespace mdb::gnn;

// ===========================================================================
// Helper: build a valid 2-layer GraphSample
// ===========================================================================

static GraphSample make_2hop_sample(uint64_t batch_id = 0,
                                     SplitType split = SplitType::TRAIN) {
    GraphSample s;
    s.batch_id = batch_id;
    s.split = split;

    // Layer 0 (seeds): nodes 100, 101
    s.nodes_per_layer.push_back({ObjectId(100), ObjectId(101)});
    // Layer 1 (1-hop): nodes 200, 201, 202
    s.nodes_per_layer.push_back({ObjectId(200), ObjectId(201), ObjectId(202)});
    // Layer 2 (2-hop): nodes 300, 301
    s.nodes_per_layer.push_back({ObjectId(300), ObjectId(301)});

    // Edges layer 0: connects layer 1 → layer 0
    LayerEdges e0;
    e0.src_indices = {0, 1, 2};       // indices in layer 1
    e0.dst_indices = {0, 0, 1};       // indices in layer 0
    e0.edge_ids = {ObjectId(500), ObjectId(501), ObjectId(502)};
    s.edges_per_layer.push_back(e0);

    // Edges layer 1: connects layer 2 → layer 1
    LayerEdges e1;
    e1.src_indices = {0, 1};           // indices in layer 2
    e1.dst_indices = {0, 2};           // indices in layer 1
    e1.edge_ids = {ObjectId(503), ObjectId(504)};
    s.edges_per_layer.push_back(e1);

    s.rebuild_unique_nodes();
    return s;
}

// ===========================================================================
// LayerEdges
// ===========================================================================

TEST(LayerEdgesTest, EmptyIsValid) {
    LayerEdges e;
    EXPECT_TRUE(e.is_valid());
    EXPECT_EQ(e.size(), 0u);
}

TEST(LayerEdgesTest, ConsistentIsValid) {
    LayerEdges e;
    e.src_indices = {0, 1};
    e.dst_indices = {0, 1};
    e.edge_ids = {ObjectId(10), ObjectId(11)};
    EXPECT_TRUE(e.is_valid());
    EXPECT_EQ(e.size(), 2u);
}

TEST(LayerEdgesTest, InconsistentSizeIsInvalid) {
    LayerEdges e;
    e.src_indices = {0, 1};
    e.dst_indices = {0};  // mismatch
    e.edge_ids = {ObjectId(10), ObjectId(11)};
    EXPECT_FALSE(e.is_valid());
}

TEST(LayerEdgesTest, ReserveAndClear) {
    LayerEdges e;
    e.reserve(100);
    e.src_indices.push_back(0);
    e.dst_indices.push_back(0);
    e.edge_ids.push_back(ObjectId(1));
    EXPECT_EQ(e.size(), 1u);
    e.clear();
    EXPECT_EQ(e.size(), 0u);
}

// ===========================================================================
// GraphSample — Statistics
// ===========================================================================

TEST(GraphSampleTest, StatisticsOnValidSample) {
    auto s = make_2hop_sample();

    EXPECT_EQ(s.batch_size(), 2u);       // layer 0 has 2 seeds
    EXPECT_EQ(s.num_layers(), 2u);        // 3 node layers = 2 GNN layers
    EXPECT_EQ(s.total_nodes(), 7u);       // 2 + 3 + 2
    EXPECT_EQ(s.total_edges(), 5u);       // 3 + 2
    EXPECT_EQ(s.unique_node_count(), 7u); // all 7 are unique
}

TEST(GraphSampleTest, EmptySampleStatistics) {
    GraphSample s;
    s.batch_id = 0;
    s.split = SplitType::TRAIN;
    // No nodes, no edges
    EXPECT_EQ(s.batch_size(), 0u);
    EXPECT_EQ(s.num_layers(), 0u);
    EXPECT_EQ(s.total_nodes(), 0u);
    EXPECT_EQ(s.total_edges(), 0u);
    EXPECT_EQ(s.unique_node_count(), 0u);
}

TEST(GraphSampleTest, SingleLayerSample) {
    GraphSample s;
    s.batch_id = 0;
    s.split = SplitType::TRAIN;
    s.nodes_per_layer.push_back({ObjectId(1), ObjectId(2), ObjectId(3)});
    // No edges (0 layers = seeds only)
    EXPECT_EQ(s.batch_size(), 3u);
    EXPECT_EQ(s.num_layers(), 0u);
    EXPECT_EQ(s.total_edges(), 0u);
}

// ===========================================================================
// GraphSample — Validation
// ===========================================================================

TEST(GraphSampleTest, ValidSamplePassesValidation) {
    auto s = make_2hop_sample();
    EXPECT_NO_THROW(s.validate());
}

TEST(GraphSampleTest, EmptyNodeLayersThrows) {
    GraphSample s;
    s.batch_id = 0;
    s.split = SplitType::TRAIN;
    EXPECT_THROW(s.validate(), std::runtime_error);
}

TEST(GraphSampleTest, MismatchedEdgeLayerCountThrows) {
    GraphSample s;
    s.batch_id = 0;
    s.split = SplitType::TRAIN;
    s.nodes_per_layer.push_back({ObjectId(1)});
    s.nodes_per_layer.push_back({ObjectId(2)});
    // Should have 1 edge layer, but we add 0
    EXPECT_THROW(s.validate(), std::runtime_error);
}

TEST(GraphSampleTest, SrcIndexOutOfBoundsThrows) {
    auto s = make_2hop_sample();
    s.edges_per_layer[0].src_indices[0] = 99; // out of bounds for layer 1 (size 3)
    EXPECT_THROW(s.validate(), std::runtime_error);
}

TEST(GraphSampleTest, DstIndexOutOfBoundsThrows) {
    auto s = make_2hop_sample();
    s.edges_per_layer[0].dst_indices[0] = 99; // out of bounds for layer 0 (size 2)
    EXPECT_THROW(s.validate(), std::runtime_error);
}

TEST(GraphSampleTest, NegativeIndexThrows) {
    auto s = make_2hop_sample();
    s.edges_per_layer[0].src_indices[0] = -1;
    EXPECT_THROW(s.validate(), std::runtime_error);
}

TEST(GraphSampleTest, InconsistentEdgeSizesThrows) {
    auto s = make_2hop_sample();
    s.edges_per_layer[0].src_indices.push_back(0); // mismatched size
    EXPECT_THROW(s.validate(), std::runtime_error);
}

// ===========================================================================
// GraphSample — rebuild_unique_nodes
// ===========================================================================

TEST(GraphSampleTest, RebuildUniqueNodesDeduplicates) {
    GraphSample s;
    s.batch_id = 0;
    s.split = SplitType::TRAIN;
    // Node 100 appears in both layers
    s.nodes_per_layer.push_back({ObjectId(100), ObjectId(101)});
    s.nodes_per_layer.push_back({ObjectId(100), ObjectId(200)});
    s.edges_per_layer.push_back(LayerEdges{{0}, {0}, {ObjectId(500)}});

    s.rebuild_unique_nodes();

    EXPECT_EQ(s.unique_node_count(), 3u); // 100, 101, 200 (100 deduplicated)

    // Verify order: layer 0 first, then new nodes from layer 1
    EXPECT_EQ(s.all_unique_nodes[0].id, 100u);
    EXPECT_EQ(s.all_unique_nodes[1].id, 101u);
    EXPECT_EQ(s.all_unique_nodes[2].id, 200u);
}

TEST(GraphSampleTest, RebuildEmptySample) {
    GraphSample s;
    s.batch_id = 0;
    s.split = SplitType::TRAIN;
    s.nodes_per_layer.push_back({});
    s.rebuild_unique_nodes();
    EXPECT_EQ(s.unique_node_count(), 0u);
}

// ===========================================================================
// GraphSample — Serialization Roundtrip
// ===========================================================================

TEST(GraphSampleTest, SerializeDeserializeRoundtrip) {
    auto original = make_2hop_sample(42, SplitType::VALIDATION);

    std::stringstream ss;
    original.serialize(ss);

    ss.seekg(0);
    auto restored = GraphSample::deserialize(ss);

    EXPECT_EQ(restored.batch_id, 42u);
    EXPECT_EQ(restored.split, SplitType::VALIDATION);
    EXPECT_EQ(restored.nodes_per_layer.size(), original.nodes_per_layer.size());

    for (size_t layer = 0; layer < original.nodes_per_layer.size(); ++layer) {
        ASSERT_EQ(restored.nodes_per_layer[layer].size(),
                  original.nodes_per_layer[layer].size())
            << "Layer " << layer << " node count mismatch";
        for (size_t i = 0; i < original.nodes_per_layer[layer].size(); ++i) {
            EXPECT_EQ(restored.nodes_per_layer[layer][i].id,
                      original.nodes_per_layer[layer][i].id)
                << "Layer " << layer << " node " << i;
        }
    }

    for (size_t k = 0; k < original.edges_per_layer.size(); ++k) {
        ASSERT_EQ(restored.edges_per_layer[k].size(),
                  original.edges_per_layer[k].size())
            << "Edge layer " << k << " size mismatch";
        for (size_t i = 0; i < original.edges_per_layer[k].size(); ++i) {
            EXPECT_EQ(restored.edges_per_layer[k].src_indices[i],
                      original.edges_per_layer[k].src_indices[i]);
            EXPECT_EQ(restored.edges_per_layer[k].dst_indices[i],
                      original.edges_per_layer[k].dst_indices[i]);
            EXPECT_EQ(restored.edges_per_layer[k].edge_ids[i].id,
                      original.edges_per_layer[k].edge_ids[i].id);
        }
    }

    EXPECT_EQ(restored.unique_node_count(), original.unique_node_count());
}

TEST(GraphSampleTest, SerializeDeserializeAllSplitTypes) {
    for (auto split : {SplitType::TRAIN, SplitType::VALIDATION, SplitType::TEST}) {
        auto s = make_2hop_sample(0, split);
        std::stringstream ss;
        s.serialize(ss);
        ss.seekg(0);
        auto restored = GraphSample::deserialize(ss);
        EXPECT_EQ(restored.split, split);
    }
}

TEST(GraphSampleTest, DeserializeInvalidMagicThrows) {
    std::stringstream ss;
    uint32_t bad_magic = 0xDEADBEEF;
    ss.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
    ss.seekg(0);
    EXPECT_THROW(GraphSample::deserialize(ss), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Round 2A — v3 bulk format roundtrip and v2 backward compatibility
// ---------------------------------------------------------------------------

// Manually write a v2-formatted sample (element-by-element layout) and verify
// the new bulk-capable deserialize accepts it via the legacy path.
TEST(GraphSampleTest, DeserializeV2LegacyFormat) {
    auto reference = make_2hop_sample(7, SplitType::TEST);

    std::stringstream ss;
    // Header (v2)
    uint32_t magic = GraphSample::MAGIC;
    uint32_t v2 = GraphSample::VERSION_V2;
    ss.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ss.write(reinterpret_cast<const char*>(&v2), sizeof(v2));

    // batch_id + split
    uint64_t batch_id = reference.batch_id;
    uint8_t split = static_cast<uint8_t>(reference.split);
    ss.write(reinterpret_cast<const char*>(&batch_id), sizeof(batch_id));
    ss.write(reinterpret_cast<const char*>(&split), sizeof(split));

    auto write_oid_vec_v2 = [&](const std::vector<ObjectId>& vec) {
        uint64_t n = vec.size();
        ss.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& o : vec) {
            ss.write(reinterpret_cast<const char*>(&o.id), sizeof(o.id));
        }
    };
    auto write_i32_vec_v2 = [&](const std::vector<int32_t>& vec) {
        uint64_t n = vec.size();
        ss.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (int32_t v : vec) {
            ss.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    };

    uint64_t nl = reference.nodes_per_layer.size();
    ss.write(reinterpret_cast<const char*>(&nl), sizeof(nl));
    for (const auto& layer : reference.nodes_per_layer) {
        write_oid_vec_v2(layer);
    }

    uint64_t el = reference.edges_per_layer.size();
    ss.write(reinterpret_cast<const char*>(&el), sizeof(el));
    for (const auto& edges : reference.edges_per_layer) {
        write_i32_vec_v2(edges.src_indices);
        write_i32_vec_v2(edges.dst_indices);
        write_oid_vec_v2(edges.edge_ids);
    }
    write_oid_vec_v2(reference.all_unique_nodes);

    ss.seekg(0);
    auto restored = GraphSample::deserialize(ss);
    EXPECT_EQ(restored.batch_id, reference.batch_id);
    EXPECT_EQ(restored.split, reference.split);
    ASSERT_EQ(restored.nodes_per_layer.size(), reference.nodes_per_layer.size());
    for (size_t k = 0; k < reference.nodes_per_layer.size(); ++k) {
        ASSERT_EQ(restored.nodes_per_layer[k].size(),
                  reference.nodes_per_layer[k].size());
        for (size_t i = 0; i < reference.nodes_per_layer[k].size(); ++i) {
            EXPECT_EQ(restored.nodes_per_layer[k][i].id,
                      reference.nodes_per_layer[k][i].id);
        }
    }
    ASSERT_EQ(restored.edges_per_layer.size(), reference.edges_per_layer.size());
    for (size_t k = 0; k < reference.edges_per_layer.size(); ++k) {
        EXPECT_EQ(restored.edges_per_layer[k].src_indices,
                  reference.edges_per_layer[k].src_indices);
        EXPECT_EQ(restored.edges_per_layer[k].dst_indices,
                  reference.edges_per_layer[k].dst_indices);
        ASSERT_EQ(restored.edges_per_layer[k].edge_ids.size(),
                  reference.edges_per_layer[k].edge_ids.size());
        for (size_t i = 0; i < reference.edges_per_layer[k].edge_ids.size(); ++i) {
            EXPECT_EQ(restored.edges_per_layer[k].edge_ids[i].id,
                      reference.edges_per_layer[k].edge_ids[i].id);
        }
    }
    ASSERT_EQ(restored.all_unique_nodes.size(), reference.all_unique_nodes.size());
    for (size_t i = 0; i < reference.all_unique_nodes.size(); ++i) {
        EXPECT_EQ(restored.all_unique_nodes[i].id, reference.all_unique_nodes[i].id);
    }
}

// Verify the writer now emits VERSION_V3 by reading the version bytes directly
// from the stream.
TEST(GraphSampleTest, SerializeEmitsVersionV3) {
    auto s = make_2hop_sample(11, SplitType::TRAIN);
    std::stringstream ss;
    s.serialize(ss);
    ss.seekg(0);

    uint32_t magic = 0, version = 0;
    ss.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ss.read(reinterpret_cast<char*>(&version), sizeof(version));
    EXPECT_EQ(magic, GraphSample::MAGIC);
    EXPECT_EQ(version, GraphSample::VERSION_V3);
    EXPECT_EQ(version, GraphSample::VERSION);  // current write version
}

// Empty vectors are a frequent edge case — bulk read must not try to read
// `size * sizeof(T) = 0` bytes and must leave the stream usable for the next
// header. Use a 1-layer sample where edges_per_layer is empty.
TEST(GraphSampleTest, BulkRoundtripEmptyVectors) {
    GraphSample s;
    s.batch_id = 1;
    s.split = SplitType::TRAIN;
    s.nodes_per_layer.push_back({ObjectId(42)});  // single seed
    // No edge layers, no all_unique_nodes beyond the seed
    s.rebuild_unique_nodes();

    std::stringstream ss;
    s.serialize(ss);
    ss.seekg(0);
    auto restored = GraphSample::deserialize(ss);
    EXPECT_EQ(restored.batch_id, 1u);
    EXPECT_EQ(restored.nodes_per_layer.size(), 1u);
    EXPECT_EQ(restored.nodes_per_layer[0].size(), 1u);
    EXPECT_EQ(restored.nodes_per_layer[0][0].id, 42u);
    EXPECT_EQ(restored.edges_per_layer.size(), 0u);
    EXPECT_EQ(restored.all_unique_nodes.size(), 1u);
}

TEST(GraphSampleTest, ReadSplitWithoutFullDeserialize) {
    auto s = make_2hop_sample(0, SplitType::TEST);
    std::stringstream ss;
    s.serialize(ss);
    ss.seekg(0);
    auto split = GraphSample::read_split(ss);
    EXPECT_EQ(split, SplitType::TEST);
}

// ===========================================================================
// GraphSample — Clear
// ===========================================================================

TEST(GraphSampleTest, ClearResetsEverything) {
    auto s = make_2hop_sample(42, SplitType::TEST);
    s.clear();
    EXPECT_EQ(s.batch_id, 0u);
    EXPECT_EQ(s.split, SplitType::TRAIN);
    EXPECT_TRUE(s.nodes_per_layer.empty());
    EXPECT_TRUE(s.edges_per_layer.empty());
    EXPECT_TRUE(s.all_unique_nodes.empty());
}

// ===========================================================================
// SamplingConfig — Validation
// ===========================================================================

TEST(SamplingConfigTest, ValidConfigPasses) {
    SamplingConfig c;
    c.projection_name = "test";
    c.sample_name = "s1";
    c.fanouts = {15, 10};
    EXPECT_NO_THROW(c.validate());
}

TEST(SamplingConfigTest, EmptyProjectionNameThrows) {
    SamplingConfig c;
    c.sample_name = "s1";
    c.fanouts = {10};
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(SamplingConfigTest, EmptySampleNameThrows) {
    SamplingConfig c;
    c.projection_name = "test";
    c.fanouts = {10};
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(SamplingConfigTest, EmptyFanoutsThrows) {
    SamplingConfig c;
    c.projection_name = "test";
    c.sample_name = "s1";
    c.fanouts = {};
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(SamplingConfigTest, ZeroFanoutThrows) {
    SamplingConfig c;
    c.projection_name = "test";
    c.sample_name = "s1";
    c.fanouts = {10, 0, 5};
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(SamplingConfigTest, ZeroBatchSizeThrows) {
    SamplingConfig c;
    c.projection_name = "test";
    c.sample_name = "s1";
    c.fanouts = {10};
    c.batch_size = 0;
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(SamplingConfigTest, RatiosSumNot1Throws) {
    SamplingConfig c;
    c.projection_name = "test";
    c.sample_name = "s1";
    c.fanouts = {10};
    c.train_ratio = 0.5;
    c.val_ratio = 0.5;
    c.test_ratio = 0.5; // sum = 1.5
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(SamplingConfigTest, NegativeRatioThrows) {
    SamplingConfig c;
    c.projection_name = "test";
    c.sample_name = "s1";
    c.fanouts = {10};
    c.train_ratio = 1.1;
    c.val_ratio = -0.05;
    c.test_ratio = -0.05;
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(SamplingConfigTest, NumLayers) {
    SamplingConfig c;
    c.fanouts = {15, 10, 5};
    EXPECT_EQ(c.num_layers(), 3u);

    c.fanouts = {10};
    EXPECT_EQ(c.num_layers(), 1u);
}

TEST(SamplingConfigTest, EstimatedMaxNodesPerBatch) {
    SamplingConfig c;
    c.batch_size = 1024;
    c.fanouts = {15, 10};
    // Layer 0: 1024 seeds
    // Layer 1: 1024 * 15 = 15360
    // Layer 2: 15360 * 10 = 153600
    // Total: 1024 + 15360 + 153600 = 169984
    EXPECT_EQ(c.estimated_max_nodes_per_batch(), 169984u);
}

TEST(SamplingConfigTest, EstimatedMaxSingleLayer) {
    SamplingConfig c;
    c.batch_size = 512;
    c.fanouts = {25};
    // 512 + 512*25 = 512 + 12800 = 13312
    EXPECT_EQ(c.estimated_max_nodes_per_batch(), 13312u);
}
