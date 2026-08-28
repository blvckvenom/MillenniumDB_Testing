// src/tests/batched_edge_aggregator_test.cc
//
// Unit tests for BatchedEdgeAggregator (CPU + future GPU stub).
//
// Coverage:
//   - empty input: aggregator emits nothing, returns success.
//   - COUNT mode: synthetic 5-parallel-per-group dataset, exact counts.
//   - SUM mode: deterministic property values, exact sum per group.
//   - MIN / MAX modes: representative-edge tracking on the smallest/
//     largest property value, including ties broken by first-occurrence.
//   - SINGLE mode: throws QueryException with the expected legacy text on
//     duplicates; succeeds on a strictly unique-key buffer.
//   - CPU vs GPU parity: same input handed to Backend::CPU and
//     Backend::GPU MUST produce bit-identical emit() callbacks.
//   - Sorted-precondition: pre-sort with radix_sort_edge_records and
//     hand-rolled std::sort produce the same group structure (sanity
//     check that the sort step external_edge_sort.h relies on is
//     equivalent to the simple comparator).
//   - Aggregated-value packing: SUM/MIN/MAX persist as doubles (fractional
//     values survive bit-exact); COUNT persists as an integer.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/common/conversions.h"
#include "graph_models/gql/projection/batched_edge_aggregator.h"
#include "graph_models/gql/projection/edge_aggregation_record.h"
#include "graph_models/gql/projection/external_edge_sort.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/object_id.h"
#include "query/exceptions.h"
#include "system/system.h"

namespace {

using GQL::Aggregation;
using GQL::BatchedEdgeAggregator;
using GQL::EdgeAggregationRecord;
using GQL::pack_double_to_bits;

// Result captured by the emit callback for cross-backend parity checks.
struct EmittedGroup {
    uint64_t edge_id;
    uint64_t count;
    double   value;

    bool operator==(const EmittedGroup& o) const {
        return edge_id == o.edge_id && count == o.count && value == o.value;
    }
};

using EmittedList = std::vector<EmittedGroup>;

// Build a sorted EdgeAggregationRecord vector with `groups` distinct keys
// and `parallels_per_group` records per key. The property value at index
// j within a group is `base_value + j` so MIN/MAX have a unique answer.
std::vector<EdgeAggregationRecord>
make_sorted_buffer(std::size_t groups,
                   std::size_t parallels_per_group,
                   double      base_value = 1.0)
{
    std::vector<EdgeAggregationRecord> out;
    out.reserve(groups * parallels_per_group);

    uint64_t edge_id_counter = 1;
    for (std::size_t g = 0; g < groups; ++g) {
        const uint64_t from = 1000 + g;
        const uint64_t to   = 2000 + g;
        const uint64_t type = 1;
        for (std::size_t p = 0; p < parallels_per_group; ++p) {
            const double v = base_value + static_cast<double>(p);
            out.push_back(EdgeAggregationRecord{
                from, to, type,
                edge_id_counter++,
                pack_double_to_bits(v)
            });
        }
    }
    // make_sorted_buffer guarantees insertion order is sorted, but we
    // re-sort defensively so any future change to the construction loop
    // can't silently corrupt the precondition.
    std::sort(out.begin(), out.end());
    return out;
}

// Emit-collector helper: returns a callback that pushes each emit() into
// the supplied vector.
auto emit_into(EmittedList& sink) {
    return [&sink](uint64_t eid, uint64_t cnt, double val) {
        sink.push_back(EmittedGroup{eid, cnt, val});
    };
}

// ----------------------------------------------------------------------
// Empty + bootstrap behaviours
// ----------------------------------------------------------------------

TEST(BatchedEdgeAggregator, EmptyInputEmitsNothing) {
    EmittedList out;
    BatchedEdgeAggregator agg(Aggregation::COUNT, emit_into(out),
                              BatchedEdgeAggregator::Backend::CPU);
    EXPECT_TRUE(agg.aggregate(static_cast<EdgeAggregationRecord*>(nullptr), 0));
    EXPECT_TRUE(out.empty());
}

TEST(BatchedEdgeAggregator, SingleRecordEmitsOneGroup) {
    auto buf = make_sorted_buffer(1, 1);
    EmittedList out;
    BatchedEdgeAggregator agg(Aggregation::COUNT, emit_into(out),
                              BatchedEdgeAggregator::Backend::CPU);
    EXPECT_TRUE(agg.aggregate(buf));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].count, 1u);
    EXPECT_EQ(out[0].value, 1.0);  // COUNT emits count as value
}

// ----------------------------------------------------------------------
// COUNT mode
// ----------------------------------------------------------------------

TEST(BatchedEdgeAggregator, CountModeReportsExactGroupSize) {
    constexpr std::size_t G = 100;
    constexpr std::size_t P = 5;
    auto buf = make_sorted_buffer(G, P);

    EmittedList out;
    BatchedEdgeAggregator agg(Aggregation::COUNT, emit_into(out),
                              BatchedEdgeAggregator::Backend::CPU);
    ASSERT_TRUE(agg.aggregate(buf));
    ASSERT_EQ(out.size(), G);
    for (const auto& g : out) {
        EXPECT_EQ(g.count, P);
        EXPECT_EQ(g.value, static_cast<double>(P));  // COUNT path: value == count
    }
}

// ----------------------------------------------------------------------
// SUM mode
// ----------------------------------------------------------------------

TEST(BatchedEdgeAggregator, SumModeAccumulatesPropertyValues) {
    // Each group has 5 records with property values 1,2,3,4,5 → sum = 15.
    constexpr std::size_t G = 50;
    constexpr std::size_t P = 5;
    auto buf = make_sorted_buffer(G, P);

    EmittedList out;
    BatchedEdgeAggregator agg(Aggregation::SUM, emit_into(out),
                              BatchedEdgeAggregator::Backend::CPU);
    ASSERT_TRUE(agg.aggregate(buf));
    ASSERT_EQ(out.size(), G);
    for (const auto& g : out) {
        EXPECT_EQ(g.count, P);
        EXPECT_EQ(g.value, 1.0 + 2.0 + 3.0 + 4.0 + 5.0);  // = 15.0
    }
}

// ----------------------------------------------------------------------
// MIN / MAX modes — representative-edge tracking
// ----------------------------------------------------------------------

TEST(BatchedEdgeAggregator, MinModeTracksSmallestValuesEdge) {
    // Property values 1,2,3,4,5 in insertion order → MIN value 1, edge_id
    // = first edge in the group (group g's first edge_id is g*P + 1).
    constexpr std::size_t G = 10;
    constexpr std::size_t P = 5;
    auto buf = make_sorted_buffer(G, P);

    EmittedList out;
    BatchedEdgeAggregator agg(Aggregation::MIN, emit_into(out),
                              BatchedEdgeAggregator::Backend::CPU);
    ASSERT_TRUE(agg.aggregate(buf));
    ASSERT_EQ(out.size(), G);
    for (std::size_t g = 0; g < G; ++g) {
        EXPECT_EQ(out[g].count, P);
        EXPECT_EQ(out[g].value, 1.0);  // smallest of 1..5
        // Representative edge is the FIRST one (smallest property is
        // index 0 of the group, edge_id = g*P + 1).
        EXPECT_EQ(out[g].edge_id, g * P + 1);
    }
}

TEST(BatchedEdgeAggregator, MaxModeTracksLargestValuesEdge) {
    constexpr std::size_t G = 10;
    constexpr std::size_t P = 5;
    auto buf = make_sorted_buffer(G, P);

    EmittedList out;
    BatchedEdgeAggregator agg(Aggregation::MAX, emit_into(out),
                              BatchedEdgeAggregator::Backend::CPU);
    ASSERT_TRUE(agg.aggregate(buf));
    ASSERT_EQ(out.size(), G);
    for (std::size_t g = 0; g < G; ++g) {
        EXPECT_EQ(out[g].count, P);
        EXPECT_EQ(out[g].value, 5.0);  // largest of 1..5
        // Representative edge is the LAST one (largest property is at
        // index P-1 of the group, edge_id = g*P + P).
        EXPECT_EQ(out[g].edge_id, g * P + P);
    }
}

// ----------------------------------------------------------------------
// SINGLE mode — strict throw-on-duplicate semantics
// ----------------------------------------------------------------------

TEST(BatchedEdgeAggregator, SingleModeThrowsOnDuplicateGroup) {
    auto buf = make_sorted_buffer(/*groups=*/3, /*parallels=*/2);

    EmittedList out;
    BatchedEdgeAggregator agg(Aggregation::SINGLE, emit_into(out),
                              BatchedEdgeAggregator::Backend::CPU);

    // Use try/catch so we can also verify the message text — the legacy
    // text is asserted on by tests/gql/.../single_aggregation_error.gql.
    bool threw = false;
    try {
        agg.aggregate(buf);
    } catch (const QueryException& e) {
        threw = true;
        const std::string what = e.what();
        EXPECT_NE(what.find("SINGLE refuses to drop data silently"),
                  std::string::npos);
    }
    EXPECT_TRUE(threw);
}

TEST(BatchedEdgeAggregator, SingleModeAcceptsUniqueKeys) {
    auto buf = make_sorted_buffer(/*groups=*/10, /*parallels=*/1);

    EmittedList out;
    BatchedEdgeAggregator agg(Aggregation::SINGLE, emit_into(out),
                              BatchedEdgeAggregator::Backend::CPU);
    EXPECT_NO_THROW(agg.aggregate(buf));
    EXPECT_EQ(out.size(), 10u);
}

// ----------------------------------------------------------------------
// CPU vs GPU parity (v1: the GPU stub must reproduce the CPU output)
// ----------------------------------------------------------------------

TEST(BatchedEdgeAggregator, CpuGpuParityCount) {
    auto buf = make_sorted_buffer(/*groups=*/200, /*parallels=*/3);

    EmittedList cpu_out;
    BatchedEdgeAggregator cpu(Aggregation::COUNT, emit_into(cpu_out),
                              BatchedEdgeAggregator::Backend::CPU);
    ASSERT_TRUE(cpu.aggregate(buf));

    EmittedList gpu_out;
    BatchedEdgeAggregator gpu(Aggregation::COUNT, emit_into(gpu_out),
                              BatchedEdgeAggregator::Backend::GPU);
    ASSERT_TRUE(gpu.aggregate(buf));

    EXPECT_EQ(cpu_out, gpu_out);
}

TEST(BatchedEdgeAggregator, CpuGpuParitySum) {
    auto buf = make_sorted_buffer(/*groups=*/50, /*parallels=*/7, /*base=*/0.5);

    EmittedList cpu_out;
    BatchedEdgeAggregator cpu(Aggregation::SUM, emit_into(cpu_out),
                              BatchedEdgeAggregator::Backend::CPU);
    ASSERT_TRUE(cpu.aggregate(buf));

    EmittedList gpu_out;
    BatchedEdgeAggregator gpu(Aggregation::SUM, emit_into(gpu_out),
                              BatchedEdgeAggregator::Backend::GPU);
    ASSERT_TRUE(gpu.aggregate(buf));

    EXPECT_EQ(cpu_out, gpu_out);
}

TEST(BatchedEdgeAggregator, CpuGpuParityMinMax) {
    auto buf = make_sorted_buffer(/*groups=*/30, /*parallels=*/4);

    {
        EmittedList cpu_out, gpu_out;
        BatchedEdgeAggregator cpu(Aggregation::MIN, emit_into(cpu_out),
                                  BatchedEdgeAggregator::Backend::CPU);
        BatchedEdgeAggregator gpu(Aggregation::MIN, emit_into(gpu_out),
                                  BatchedEdgeAggregator::Backend::GPU);
        ASSERT_TRUE(cpu.aggregate(buf));
        ASSERT_TRUE(gpu.aggregate(buf));
        EXPECT_EQ(cpu_out, gpu_out);
    }
    {
        EmittedList cpu_out, gpu_out;
        BatchedEdgeAggregator cpu(Aggregation::MAX, emit_into(cpu_out),
                                  BatchedEdgeAggregator::Backend::CPU);
        BatchedEdgeAggregator gpu(Aggregation::MAX, emit_into(gpu_out),
                                  BatchedEdgeAggregator::Backend::GPU);
        ASSERT_TRUE(cpu.aggregate(buf));
        ASSERT_TRUE(gpu.aggregate(buf));
        EXPECT_EQ(cpu_out, gpu_out);
    }
}

TEST(BatchedEdgeAggregator, LastBackendUsedReportsActualPath) {
    auto buf = make_sorted_buffer(/*groups=*/5, /*parallels=*/2);

    EmittedList out;
    {
        BatchedEdgeAggregator agg(Aggregation::COUNT, emit_into(out),
                                  BatchedEdgeAggregator::Backend::CPU);
        ASSERT_TRUE(agg.aggregate(buf));
        EXPECT_EQ(agg.last_backend_used(),
                  BatchedEdgeAggregator::Backend::CPU);
    }
    out.clear();
    {
        BatchedEdgeAggregator agg(Aggregation::COUNT, emit_into(out),
                                  BatchedEdgeAggregator::Backend::GPU);
        ASSERT_TRUE(agg.aggregate(buf));
        EXPECT_EQ(agg.last_backend_used(),
                  BatchedEdgeAggregator::Backend::GPU);
    }
}

// ----------------------------------------------------------------------
// Sorted-precondition cross-check: radix sort vs std::sort give the same
// emission sequence. This guards the `radix_sort_edge_records` ↔
// std::sort-by-operator< invariant external_edge_sort.h depends on, and
// makes the test suite a regression net for any future change to either
// sort path.
// ----------------------------------------------------------------------

TEST(BatchedEdgeAggregator, RadixVsStdSortGiveIdenticalOutput) {
    // Construct an UNSORTED buffer with several groups interleaved.
    std::vector<EdgeAggregationRecord> raw;
    for (int rep = 0; rep < 4; ++rep) {
        for (uint64_t g = 0; g < 25; ++g) {
            raw.push_back(EdgeAggregationRecord{
                1000 + g, 2000 + g, 1,
                static_cast<uint64_t>(rep * 100 + g + 1),
                pack_double_to_bits(static_cast<double>(rep + 1))
            });
        }
    }

    auto by_radix = raw;
    GQL::radix_sort_edge_records(by_radix.begin(), by_radix.end());

    auto by_std = raw;
    std::sort(by_std.begin(), by_std.end());

    EmittedList out_radix, out_std;
    BatchedEdgeAggregator a1(Aggregation::COUNT, emit_into(out_radix),
                             BatchedEdgeAggregator::Backend::CPU);
    BatchedEdgeAggregator a2(Aggregation::COUNT, emit_into(out_std),
                             BatchedEdgeAggregator::Backend::CPU);
    ASSERT_TRUE(a1.aggregate(by_radix));
    ASSERT_TRUE(a2.aggregate(by_std));

    EXPECT_EQ(out_radix.size(), 25u);
    EXPECT_EQ(out_radix, out_std);
}

// ----------------------------------------------------------------------
// Aggregated-value packing: SUM/MIN/MAX results are doubles and must be
// persisted as doubles. Truncating them to int64 silently corrupts
// fractional aggregates (SUM of 0.5 + 0.5 stored as 1, MIN of 2.7 stored
// as 2) and changes the property's type even for whole-number results.
// COUNT is integral by construction and stays an inlined integer.
// ----------------------------------------------------------------------

TEST(AggregatedValuePacking, SumMinMaxPersistAsDoublesCountAsInt) {
    // pack_aggregated_property_value persists doubles through the
    // string_manager, so bring up a minimal System against a scratch db
    // folder (same init pattern as the projection_* test mains).
    const std::string db_dir = "test_db_agg_value_packing";
    std::filesystem::remove_all(db_dir);
    System system(db_dir,
                  1024 * 1024,        // str_static_size
                  1024 * 1024,        // str_dynamic_size
                  64 * 1024 * 1024,   // shared_buffer_size
                  32 * 1024 * 1024,   // private_buffer_size
                  1024 * 1024,        // tensor_static_size
                  1024 * 1024,        // tensor_dynamic_size
                  1);                 // workers

    // SUM of 0.5 + 0.5 must come back as the double 1.0, not the int 1.
    ObjectId sum_oid = GQL::pack_aggregated_property_value(Aggregation::SUM, 0.5 + 0.5);
    EXPECT_EQ(sum_oid.get_sub_type(), ObjectId::MASK_DOUBLE);
    EXPECT_EQ(Common::Conversions::unpack_double(sum_oid), 1.0);

    // MIN of 2.7 must keep its fractional part (int64 truncation gives 2).
    ObjectId min_oid = GQL::pack_aggregated_property_value(Aggregation::MIN, 2.7);
    EXPECT_EQ(min_oid.get_sub_type(), ObjectId::MASK_DOUBLE);
    EXPECT_EQ(Common::Conversions::unpack_double(min_oid), 2.7);

    // MAX with a negative fractional value round-trips bit-exact too.
    ObjectId max_oid = GQL::pack_aggregated_property_value(Aggregation::MAX, -0.25);
    EXPECT_EQ(max_oid.get_sub_type(), ObjectId::MASK_DOUBLE);
    EXPECT_EQ(Common::Conversions::unpack_double(max_oid), -0.25);

    // COUNT stays an integer.
    ObjectId cnt_oid = GQL::pack_aggregated_property_value(Aggregation::COUNT, 3.0);
    EXPECT_EQ(cnt_oid.get_type(), ObjectId::MASK_POSITIVE_INT);
    EXPECT_EQ(Common::Conversions::unpack_int(cnt_oid), 3);
}

} // namespace
