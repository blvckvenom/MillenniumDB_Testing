// src/tests/parallel_edge_aggregation_test.cc
//
// Spec #17 — unit tests for the parallel edge-aggregation primitives:
//
//   * MDB_PROJECTION_PARALLEL_AGGREGATION env-var parser
//     (default ON; "0"/"false"/"off" — any case — flips to OFF; any
//     other value parses as ON, mirroring MDB_PROJECTION_PARALLEL_EDGE_SCAN).
//
//   * EdgeAggregator::merge_from semantics across the 5 strategies
//     (SINGLE / MIN / MAX / SUM / COUNT). The merge MUST produce a
//     value bit-identical to running the same edge stream through a
//     single non-partitioned EdgeAggregator — which is the bedrock
//     invariant that lets the parallel-aggregation path swap in
//     without changing user-visible projection output.
//
//   * ParallelEdgeDetector::merge_from across two synthetic worker
//     detectors:
//       (a) disjoint key sets — combined map is just the union;
//       (b) overlapping key set — the merged aggregator state for the
//           overlap key is what a single-detector run over the
//           concatenated stream would have produced (parallel ==
//           sequential output).
//
// The test does NOT spin up TBB or a live B+Tree — those paths are
// covered indirectly by the existing GQL integration suite under
// tests/gql/test_suites/projection_native, which now exercises the
// parallel path by default. Here we keep the unit-test scope tight
// to the merge math + env parsing.

#include <cstdint>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/object_id.h"
#include "query/exceptions.h"

namespace {

using GQL::Aggregation;
using GQL::EdgeAggregator;
using GQL::ParallelEdgeDetector;
using GQL::ParallelEdgeKey;

// Helper: build a typed ObjectId mimicking a directed-edge id (the
// 8-bit MASK_DIRECTED_EDGE prefix + a unique counter). The detector
// keys on the raw 64-bit id so the prefix doesn't actually matter
// for the merge math, but matching the production layout makes
// failures easier to read in gtest output.
ObjectId make_edge_id(std::uint64_t counter) {
    return ObjectId(ObjectId::MASK_DIRECTED_EDGE | counter);
}

// ----------------------------------------------------------------------
// Env-var parsing — MDB_PROJECTION_PARALLEL_AGGREGATION
// ----------------------------------------------------------------------

TEST(ParallelAggregationEnv, NullDefaultsToEnabled) {
    EXPECT_TRUE(GQL::detail::init_parallel_aggregation_for_test(nullptr));
}

TEST(ParallelAggregationEnv, FalseyValuesDisable) {
    EXPECT_FALSE(GQL::detail::init_parallel_aggregation_for_test("0"));
    EXPECT_FALSE(GQL::detail::init_parallel_aggregation_for_test("false"));
    EXPECT_FALSE(GQL::detail::init_parallel_aggregation_for_test("off"));
    // Case-insensitive for the canonical false values
    EXPECT_FALSE(GQL::detail::init_parallel_aggregation_for_test("FALSE"));
    EXPECT_FALSE(GQL::detail::init_parallel_aggregation_for_test("OFF"));
    EXPECT_FALSE(GQL::detail::init_parallel_aggregation_for_test("False"));
    EXPECT_FALSE(GQL::detail::init_parallel_aggregation_for_test("Off"));
}

TEST(ParallelAggregationEnv, TruthyAndUnknownValuesEnable) {
    EXPECT_TRUE(GQL::detail::init_parallel_aggregation_for_test("1"));
    EXPECT_TRUE(GQL::detail::init_parallel_aggregation_for_test("true"));
    EXPECT_TRUE(GQL::detail::init_parallel_aggregation_for_test("yes"));
    EXPECT_TRUE(GQL::detail::init_parallel_aggregation_for_test(""));
    EXPECT_TRUE(GQL::detail::init_parallel_aggregation_for_test("garbage"));
}

// ----------------------------------------------------------------------
// EdgeAggregator::merge_from — strategy-by-strategy
// ----------------------------------------------------------------------

TEST(EdgeAggregatorMerge, EmptyOtherIsNoOp) {
    EdgeAggregator a(Aggregation::COUNT);
    a.process_edge(make_edge_id(1), std::nullopt);
    a.process_edge(make_edge_id(2), std::nullopt);
    EXPECT_EQ(a.get_count(), 2u);

    EdgeAggregator empty(Aggregation::COUNT);
    a.merge_from(empty);
    EXPECT_EQ(a.get_count(), 2u);
    EXPECT_EQ(a.get_first_edge().id, make_edge_id(1).id);
}

TEST(EdgeAggregatorMerge, EmptyThisAdoptsOther) {
    EdgeAggregator a(Aggregation::SUM);
    EdgeAggregator b(Aggregation::SUM);
    b.process_edge(make_edge_id(7), 3.5);
    b.process_edge(make_edge_id(8), 2.5);
    a.merge_from(b);
    EXPECT_EQ(a.get_count(), 2u);
    EXPECT_DOUBLE_EQ(a.get_aggregated_value(), 6.0);
    EXPECT_EQ(a.get_first_edge().id, make_edge_id(7).id);
}

TEST(EdgeAggregatorMerge, CountAccumulates) {
    EdgeAggregator a(Aggregation::COUNT);
    EdgeAggregator b(Aggregation::COUNT);
    a.process_edge(make_edge_id(1), std::nullopt);
    a.process_edge(make_edge_id(2), std::nullopt);
    b.process_edge(make_edge_id(3), std::nullopt);
    b.process_edge(make_edge_id(4), std::nullopt);
    b.process_edge(make_edge_id(5), std::nullopt);
    a.merge_from(b);
    EXPECT_EQ(a.get_count(), 5u);
    EXPECT_DOUBLE_EQ(a.get_aggregated_value(), 5.0);
    // first_edge_id_ stays from the earlier (this) side of the merge
    EXPECT_EQ(a.get_first_edge().id, make_edge_id(1).id);
}

TEST(EdgeAggregatorMerge, SumAccumulates) {
    EdgeAggregator a(Aggregation::SUM);
    EdgeAggregator b(Aggregation::SUM);
    a.process_edge(make_edge_id(1), 10.0);
    a.process_edge(make_edge_id(2), 20.0);
    b.process_edge(make_edge_id(3), 30.0);
    b.process_edge(make_edge_id(4), 40.0);
    a.merge_from(b);
    EXPECT_EQ(a.get_count(), 4u);
    EXPECT_DOUBLE_EQ(a.get_aggregated_value(), 100.0);
    EXPECT_TRUE(a.has_value());
}

TEST(EdgeAggregatorMerge, MinTakesGlobalMinimum) {
    EdgeAggregator a(Aggregation::MIN);
    EdgeAggregator b(Aggregation::MIN);
    a.process_edge(make_edge_id(1), 5.0);
    a.process_edge(make_edge_id(2), 7.0);    // a's MIN is 5.0 @ edge 1
    b.process_edge(make_edge_id(3), 9.0);
    b.process_edge(make_edge_id(4), 2.0);    // b's MIN is 2.0 @ edge 4
    a.merge_from(b);
    EXPECT_EQ(a.get_count(), 4u);
    EXPECT_DOUBLE_EQ(a.get_aggregated_value(), 2.0);
    EXPECT_EQ(a.get_representative_edge().id, make_edge_id(4).id);
    // first_edge_id_ stays at 1 (earlier scan-order partition wins).
    EXPECT_EQ(a.get_first_edge().id, make_edge_id(1).id);
}

TEST(EdgeAggregatorMerge, MinKeepsThisRepresentativeWhenLowerOnLeft) {
    EdgeAggregator a(Aggregation::MIN);
    EdgeAggregator b(Aggregation::MIN);
    a.process_edge(make_edge_id(1), 1.0);    // a's MIN is 1.0 @ edge 1
    b.process_edge(make_edge_id(2), 99.0);
    a.merge_from(b);
    EXPECT_DOUBLE_EQ(a.get_aggregated_value(), 1.0);
    EXPECT_EQ(a.get_representative_edge().id, make_edge_id(1).id);
}

TEST(EdgeAggregatorMerge, MaxTakesGlobalMaximum) {
    EdgeAggregator a(Aggregation::MAX);
    EdgeAggregator b(Aggregation::MAX);
    a.process_edge(make_edge_id(1), 5.0);
    a.process_edge(make_edge_id(2), 7.0);
    b.process_edge(make_edge_id(3), 9.0);    // b's MAX is 9.0 @ edge 3
    b.process_edge(make_edge_id(4), 2.0);
    a.merge_from(b);
    EXPECT_EQ(a.get_count(), 4u);
    EXPECT_DOUBLE_EQ(a.get_aggregated_value(), 9.0);
    EXPECT_EQ(a.get_representative_edge().id, make_edge_id(3).id);
}

TEST(EdgeAggregatorMerge, SumWithNullValuesPreservesHasValue) {
    EdgeAggregator a(Aggregation::SUM);
    EdgeAggregator b(Aggregation::SUM);
    // a has a real value, b has only nulls.
    a.process_edge(make_edge_id(1), 5.0);
    b.process_edge(make_edge_id(2), std::nullopt);
    a.merge_from(b);
    EXPECT_TRUE(a.has_value());
    EXPECT_DOUBLE_EQ(a.get_aggregated_value(), 5.0);
}

// ----------------------------------------------------------------------
// ParallelEdgeDetector::merge_from
// ----------------------------------------------------------------------

TEST(ParallelEdgeDetectorMerge, DisjointKeysAreUnioned) {
    ParallelEdgeDetector p1(Aggregation::COUNT);
    ParallelEdgeDetector p2(Aggregation::COUNT);

    p1.process_edge(1, 2, 100, make_edge_id(1));
    p1.process_edge(3, 4, 100, make_edge_id(2));

    p2.process_edge(5, 6, 100, make_edge_id(3));
    p2.process_edge(7, 8, 100, make_edge_id(4));

    p1.merge_from(p2);
    EXPECT_EQ(p1.get_map_size(), 4u);

    auto values = p1.get_aggregated_property_values();
    EXPECT_EQ(values.size(), 4u);
    EXPECT_DOUBLE_EQ(values[make_edge_id(1).id], 1.0);
    EXPECT_DOUBLE_EQ(values[make_edge_id(2).id], 1.0);
    EXPECT_DOUBLE_EQ(values[make_edge_id(3).id], 1.0);
    EXPECT_DOUBLE_EQ(values[make_edge_id(4).id], 1.0);
}

TEST(ParallelEdgeDetectorMerge, OverlappingKeysCombineCount) {
    // Mimic two workers each seeing the same (from=1, to=2, type=100)
    // composite key. The merged COUNT for that group should be 5
    // (2 from p1 + 3 from p2) — bit-identical to running all 5 edges
    // through a single non-partitioned detector.
    ParallelEdgeDetector p1(Aggregation::COUNT);
    ParallelEdgeDetector p2(Aggregation::COUNT);

    p1.process_edge(1, 2, 100, make_edge_id(10));
    p1.process_edge(1, 2, 100, make_edge_id(11));

    p2.process_edge(1, 2, 100, make_edge_id(20));
    p2.process_edge(1, 2, 100, make_edge_id(21));
    p2.process_edge(1, 2, 100, make_edge_id(22));

    p1.merge_from(p2);

    auto values = p1.get_aggregated_property_values();
    ASSERT_EQ(values.size(), 1u);
    // The earlier-partition first occurrence (edge 10) is the
    // representative emitted from get_first_edge / first_edge_id_.
    EXPECT_TRUE(values.find(make_edge_id(10).id) != values.end());
    EXPECT_DOUBLE_EQ(values[make_edge_id(10).id], 5.0);
}

TEST(ParallelEdgeDetectorMerge, OverlappingKeysCombineSum) {
    ParallelEdgeDetector p1(Aggregation::SUM);
    ParallelEdgeDetector p2(Aggregation::SUM);

    p1.process_edge(10, 20, 1, make_edge_id(1), 1.0);
    p1.process_edge(10, 20, 1, make_edge_id(2), 2.0);
    p2.process_edge(10, 20, 1, make_edge_id(3), 4.0);
    p2.process_edge(10, 20, 1, make_edge_id(4), 8.0);

    p1.merge_from(p2);

    auto values = p1.get_aggregated_property_values();
    ASSERT_EQ(values.size(), 1u);
    EXPECT_DOUBLE_EQ(values[make_edge_id(1).id], 15.0);
}

TEST(ParallelEdgeDetectorMerge, OverlappingKeysCombineMin) {
    ParallelEdgeDetector p1(Aggregation::MIN);
    ParallelEdgeDetector p2(Aggregation::MIN);

    p1.process_edge(10, 20, 1, make_edge_id(1), 5.0);
    p1.process_edge(10, 20, 1, make_edge_id(2), 7.0);
    p2.process_edge(10, 20, 1, make_edge_id(3), 9.0);
    p2.process_edge(10, 20, 1, make_edge_id(4), 2.0);    // global MIN

    p1.merge_from(p2);

    auto values = p1.get_aggregated_property_values();
    ASSERT_EQ(values.size(), 1u);
    EXPECT_DOUBLE_EQ(values[make_edge_id(1).id], 2.0);

    // first_edge_id_ stays at edge 1 (earlier-partition
    // first-occurrence). representative_edge is edge 4 (where the
    // global min was observed).
    ParallelEdgeKey key{10, 20, 1};
    ASSERT_TRUE(p1.has_key(key));
    EXPECT_EQ(p1.get_aggregator(key).get_first_edge().id,
              make_edge_id(1).id);
    EXPECT_EQ(p1.get_aggregator(key).get_representative_edge().id,
              make_edge_id(4).id);
}

TEST(ParallelEdgeDetectorMerge, OverlappingKeysCombineMax) {
    ParallelEdgeDetector p1(Aggregation::MAX);
    ParallelEdgeDetector p2(Aggregation::MAX);

    p1.process_edge(10, 20, 1, make_edge_id(1), 5.0);
    p1.process_edge(10, 20, 1, make_edge_id(2), 7.0);
    p2.process_edge(10, 20, 1, make_edge_id(3), 99.0);   // global MAX
    p2.process_edge(10, 20, 1, make_edge_id(4), 2.0);

    p1.merge_from(p2);

    auto values = p1.get_aggregated_property_values();
    ASSERT_EQ(values.size(), 1u);
    EXPECT_DOUBLE_EQ(values[make_edge_id(1).id], 99.0);

    ParallelEdgeKey key{10, 20, 1};
    EXPECT_EQ(p1.get_aggregator(key).get_representative_edge().id,
              make_edge_id(3).id);
}

// ----------------------------------------------------------------------
// Parallel-vs-sequential parity: the merge math must match what a
// single non-partitioned detector would produce when fed the
// concatenated stream. This is the load-bearing invariant for the
// projection builder's parallel-aggregation path.
// ----------------------------------------------------------------------

TEST(ParallelVsSequential, CountOnInterleavedStream) {
    // 12 edges spread across 3 parallel groups, fed in two halves to
    // simulate two workers. Compare the merged detector to a single
    // detector that saw all 12.
    auto run_sequential = [](Aggregation strat) {
        ParallelEdgeDetector d(strat);
        // Group A (count=3)
        d.process_edge(1, 2, 100, make_edge_id(1));
        d.process_edge(1, 2, 100, make_edge_id(2));
        d.process_edge(1, 2, 100, make_edge_id(3));
        // Group B (count=4)
        d.process_edge(5, 6, 100, make_edge_id(4));
        d.process_edge(5, 6, 100, make_edge_id(5));
        d.process_edge(5, 6, 100, make_edge_id(6));
        d.process_edge(5, 6, 100, make_edge_id(7));
        // Group C (count=5)
        d.process_edge(9, 10, 100, make_edge_id(8));
        d.process_edge(9, 10, 100, make_edge_id(9));
        d.process_edge(9, 10, 100, make_edge_id(10));
        d.process_edge(9, 10, 100, make_edge_id(11));
        d.process_edge(9, 10, 100, make_edge_id(12));
        return d.get_aggregated_property_values();
    };

    auto run_parallel = [](Aggregation strat) {
        ParallelEdgeDetector p1(strat);
        ParallelEdgeDetector p2(strat);
        // Worker 1 sees first 6 edges in scan order.
        p1.process_edge(1, 2, 100, make_edge_id(1));
        p1.process_edge(1, 2, 100, make_edge_id(2));
        p1.process_edge(1, 2, 100, make_edge_id(3));
        p1.process_edge(5, 6, 100, make_edge_id(4));
        p1.process_edge(5, 6, 100, make_edge_id(5));
        p1.process_edge(5, 6, 100, make_edge_id(6));
        // Worker 2 sees the remaining 6.
        p2.process_edge(5, 6, 100, make_edge_id(7));
        p2.process_edge(9, 10, 100, make_edge_id(8));
        p2.process_edge(9, 10, 100, make_edge_id(9));
        p2.process_edge(9, 10, 100, make_edge_id(10));
        p2.process_edge(9, 10, 100, make_edge_id(11));
        p2.process_edge(9, 10, 100, make_edge_id(12));
        p1.merge_from(p2);
        return p1.get_aggregated_property_values();
    };

    auto seq = run_sequential(Aggregation::COUNT);
    auto par = run_parallel(Aggregation::COUNT);
    EXPECT_EQ(seq, par);
    EXPECT_DOUBLE_EQ(par[make_edge_id(1).id], 3.0);
    EXPECT_DOUBLE_EQ(par[make_edge_id(4).id], 4.0);
    EXPECT_DOUBLE_EQ(par[make_edge_id(8).id], 5.0);
}

TEST(ParallelVsSequential, SumOnInterleavedStream) {
    auto run_sequential = []() {
        ParallelEdgeDetector d(Aggregation::SUM);
        d.process_edge(1, 2, 100, make_edge_id(1), 10.0);
        d.process_edge(1, 2, 100, make_edge_id(2), 20.0);
        d.process_edge(5, 6, 100, make_edge_id(3), 100.0);
        d.process_edge(5, 6, 100, make_edge_id(4), 200.0);
        d.process_edge(5, 6, 100, make_edge_id(5), 300.0);
        d.process_edge(1, 2, 100, make_edge_id(6), 30.0);
        return d.get_aggregated_property_values();
    };

    auto run_parallel = []() {
        ParallelEdgeDetector p1(Aggregation::SUM);
        ParallelEdgeDetector p2(Aggregation::SUM);
        p1.process_edge(1, 2, 100, make_edge_id(1), 10.0);
        p1.process_edge(1, 2, 100, make_edge_id(2), 20.0);
        p1.process_edge(5, 6, 100, make_edge_id(3), 100.0);
        p2.process_edge(5, 6, 100, make_edge_id(4), 200.0);
        p2.process_edge(5, 6, 100, make_edge_id(5), 300.0);
        p2.process_edge(1, 2, 100, make_edge_id(6), 30.0);
        p1.merge_from(p2);
        return p1.get_aggregated_property_values();
    };

    auto seq = run_sequential();
    auto par = run_parallel();
    ASSERT_EQ(seq.size(), par.size());
    EXPECT_DOUBLE_EQ(par[make_edge_id(1).id], 60.0);   // 10 + 20 + 30
    EXPECT_DOUBLE_EQ(par[make_edge_id(3).id], 600.0);  // 100 + 200 + 300
    EXPECT_EQ(seq, par);
}

TEST(ParallelVsSequential, MinAcrossPartitions) {
    auto run_sequential = []() {
        ParallelEdgeDetector d(Aggregation::MIN);
        d.process_edge(1, 2, 100, make_edge_id(1), 50.0);
        d.process_edge(1, 2, 100, make_edge_id(2), 25.0);
        d.process_edge(1, 2, 100, make_edge_id(3), 10.0);  // global min
        d.process_edge(1, 2, 100, make_edge_id(4), 30.0);
        return d.get_aggregated_property_values();
    };

    auto run_parallel = []() {
        ParallelEdgeDetector p1(Aggregation::MIN);
        ParallelEdgeDetector p2(Aggregation::MIN);
        p1.process_edge(1, 2, 100, make_edge_id(1), 50.0);
        p1.process_edge(1, 2, 100, make_edge_id(2), 25.0);
        p2.process_edge(1, 2, 100, make_edge_id(3), 10.0);
        p2.process_edge(1, 2, 100, make_edge_id(4), 30.0);
        p1.merge_from(p2);
        return p1.get_aggregated_property_values();
    };

    auto seq = run_sequential();
    auto par = run_parallel();
    EXPECT_EQ(seq, par);
    EXPECT_DOUBLE_EQ(par[make_edge_id(1).id], 10.0);
}

TEST(ParallelVsSequential, MaxAcrossPartitions) {
    auto run_sequential = []() {
        ParallelEdgeDetector d(Aggregation::MAX);
        d.process_edge(1, 2, 100, make_edge_id(1), 50.0);
        d.process_edge(1, 2, 100, make_edge_id(2), 75.0);
        d.process_edge(1, 2, 100, make_edge_id(3), 90.0);  // global max
        d.process_edge(1, 2, 100, make_edge_id(4), 20.0);
        return d.get_aggregated_property_values();
    };

    auto run_parallel = []() {
        ParallelEdgeDetector p1(Aggregation::MAX);
        ParallelEdgeDetector p2(Aggregation::MAX);
        p1.process_edge(1, 2, 100, make_edge_id(1), 50.0);
        p1.process_edge(1, 2, 100, make_edge_id(2), 75.0);
        p2.process_edge(1, 2, 100, make_edge_id(3), 90.0);
        p2.process_edge(1, 2, 100, make_edge_id(4), 20.0);
        p1.merge_from(p2);
        return p1.get_aggregated_property_values();
    };

    auto seq = run_sequential();
    auto par = run_parallel();
    EXPECT_EQ(seq, par);
    EXPECT_DOUBLE_EQ(par[make_edge_id(1).id], 90.0);
}

// ----------------------------------------------------------------------
// SINGLE: detection of cross-partition duplicates after merge.
// ----------------------------------------------------------------------

TEST(ParallelEdgeDetectorMerge, SingleCrossPartitionDuplicateDetectable) {
    // Each partition individually only sees one edge per key — neither
    // would throw on its own (SINGLE throws at count_ == 2). After
    // merge, count_ becomes 2 — the caller in
    // run_parallel_edge_aggregation_ inspects this and re-throws the
    // same QueryException the inline path raises.
    ParallelEdgeDetector p1(Aggregation::SINGLE);
    ParallelEdgeDetector p2(Aggregation::SINGLE);
    p1.process_edge(1, 2, 100, make_edge_id(1));   // count=1, no throw
    p2.process_edge(1, 2, 100, make_edge_id(2));   // count=1, no throw
    p1.merge_from(p2);

    ParallelEdgeKey key{1, 2, 100};
    ASSERT_TRUE(p1.has_key(key));
    EXPECT_GE(p1.get_aggregator(key).get_count(), 2u);
}

}  // namespace
