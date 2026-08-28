#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_fingerprint.h"
#include "graph_models/object_id.h"

using namespace mdb::gnn;

// ===========================================================================
// Helper: build a valid 2-layer GraphSample (mirrors test_graph_sample.cc)
// ===========================================================================

static GraphSample make_2hop_sample(uint64_t batch_id = 0,
                                     SplitType split = SplitType::TRAIN) {
    GraphSample s;
    s.batch_id = batch_id;
    s.split = split;

    s.nodes_per_layer.push_back({ObjectId(100), ObjectId(101)});
    s.nodes_per_layer.push_back({ObjectId(200), ObjectId(201), ObjectId(202)});
    s.nodes_per_layer.push_back({ObjectId(300), ObjectId(301)});

    LayerEdges e0;
    e0.src_indices = {0, 1, 2};
    e0.dst_indices = {0, 0, 1};
    e0.edge_ids = {ObjectId(500), ObjectId(501), ObjectId(502)};
    s.edges_per_layer.push_back(e0);

    LayerEdges e1;
    e1.src_indices = {0, 1};
    e1.dst_indices = {0, 2};
    e1.edge_ids = {ObjectId(503), ObjectId(504)};
    s.edges_per_layer.push_back(e1);

    s.rebuild_unique_nodes();
    return s;
}

// ===========================================================================
// Determinism
// ===========================================================================

TEST(SampleFingerprint, Deterministic) {
    auto a = make_2hop_sample();
    auto b = make_2hop_sample();
    EXPECT_EQ(compute_batch_content_hash(a), compute_batch_content_hash(b));
}

// ===========================================================================
// Sensitivity — every content determinant must perturb the hash
// ===========================================================================

TEST(SampleFingerprint, ChangesWhenBatchIdChanges) {
    auto a = make_2hop_sample(0);
    auto b = make_2hop_sample(1);
    EXPECT_NE(compute_batch_content_hash(a), compute_batch_content_hash(b));
}

TEST(SampleFingerprint, ChangesWhenSplitChanges) {
    auto a = make_2hop_sample(7, SplitType::TRAIN);
    auto b = make_2hop_sample(7, SplitType::VALIDATION);
    EXPECT_NE(compute_batch_content_hash(a), compute_batch_content_hash(b));
}

TEST(SampleFingerprint, ChangesWhenNodeSetChanges) {
    auto a = make_2hop_sample();
    auto b = make_2hop_sample();
    // Swap one input-layer node for a different id (different node SET, same shape).
    b.nodes_per_layer[2][0] = ObjectId(999);
    b.rebuild_unique_nodes();
    EXPECT_NE(compute_batch_content_hash(a), compute_batch_content_hash(b));
}

TEST(SampleFingerprint, ChangesWhenPerLayerNodeCountChanges) {
    auto a = make_2hop_sample();
    auto b = make_2hop_sample();
    // Add a node to the 1-hop layer (changes the per-layer shape).
    b.nodes_per_layer[1].push_back(ObjectId(203));
    b.rebuild_unique_nodes();
    EXPECT_NE(compute_batch_content_hash(a), compute_batch_content_hash(b));
}

TEST(SampleFingerprint, ChangesWhenEdgeEndpointChanges) {
    auto a = make_2hop_sample();
    auto b = make_2hop_sample();
    // Reconnect one edge to a different destination node (same node set,
    // different connectivity — the residual orientation signal).
    b.edges_per_layer[0].dst_indices[2] = 0;  // was 1
    EXPECT_NE(compute_batch_content_hash(a), compute_batch_content_hash(b));
}

// ===========================================================================
// Layout independence — invariance to internal ordering
// ===========================================================================

TEST(SampleFingerprint, InvariantToUniqueNodePermutation) {
    auto a = make_2hop_sample();
    auto b = make_2hop_sample();
    // all_unique_nodes order should not matter (the hash sorts ids internally).
    std::reverse(b.all_unique_nodes.begin(), b.all_unique_nodes.end());
    EXPECT_EQ(compute_batch_content_hash(a), compute_batch_content_hash(b));
}

TEST(SampleFingerprint, InvariantToEdgeOrderWithinLayer) {
    auto a = make_2hop_sample();
    auto b = make_2hop_sample();
    // Reverse the edge order in layer 0 — same edge multiset, XOR fold is
    // commutative, so the hash must not change.
    auto& e = b.edges_per_layer[0];
    std::reverse(e.src_indices.begin(), e.src_indices.end());
    std::reverse(e.dst_indices.begin(), e.dst_indices.end());
    std::reverse(e.edge_ids.begin(), e.edge_ids.end());
    EXPECT_EQ(compute_batch_content_hash(a), compute_batch_content_hash(b));
}

TEST(SampleFingerprint, CrossBatchXorFoldIsOrderIndependent) {
    // The SampleStorage combiner XOR-folds per-batch hashes. Verify the fold
    // is invariant to the order in which batches are accumulated (numWorkers).
    auto b0 = make_2hop_sample(0);
    auto b1 = make_2hop_sample(1);
    auto b2 = make_2hop_sample(2);
    uint64_t h0 = compute_batch_content_hash(b0);
    uint64_t h1 = compute_batch_content_hash(b1);
    uint64_t h2 = compute_batch_content_hash(b2);
    uint64_t forward  = h0 ^ h1 ^ h2;
    uint64_t shuffled = h2 ^ h0 ^ h1;
    EXPECT_EQ(forward, shuffled);
    // Distinct batch_ids must not collide to 0 (would cancel under XOR).
    EXPECT_NE(h0, h1);
    EXPECT_NE(h1, h2);
}

// ===========================================================================
// mix_feature_store_fingerprint — feature identity + UNKNOWN sentinel
// ===========================================================================

TEST(SampleFingerprint, MixPropagatesUnknown) {
    // sample_content_fp == 0 (UNKNOWN/legacy) must propagate to 0 so the
    // feature store recomputes rather than reusing.
    EXPECT_EQ(mix_feature_store_fingerprint(0, "node_features", 128, 0), 0u);
}

TEST(SampleFingerprint, MixNeverReturnsZeroForKnownSample) {
    uint64_t fp = mix_feature_store_fingerprint(12345, "node_features", 128, 0);
    EXPECT_NE(fp, 0u);
}

TEST(SampleFingerprint, MixChangesWithFeatureIdentity) {
    uint64_t base = mix_feature_store_fingerprint(12345, "node_features", 128, 0);
    EXPECT_NE(base, mix_feature_store_fingerprint(12345, "other_features", 128, 0));
    EXPECT_NE(base, mix_feature_store_fingerprint(12345, "node_features", 256, 0));
    EXPECT_NE(base, mix_feature_store_fingerprint(12345, "node_features", 128, 1));
    EXPECT_NE(base, mix_feature_store_fingerprint(54321, "node_features", 128, 0));
}

TEST(SampleFingerprint, MixIsDeterministic) {
    EXPECT_EQ(mix_feature_store_fingerprint(98765, "f", 64, 2),
              mix_feature_store_fingerprint(98765, "f", 64, 2));
}
