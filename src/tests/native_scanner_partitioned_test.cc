// native_scanner_partitioned_test.cc
//
// Unit tests for GQL::NativeScanner::scan_label_edge_endpoints_partitioned —
// the in-worker (no single-consumer funnel) partitioned scan seam used by the
// builder-level PARALLEL edge-scan mode (MDB_PROJECTION_PARALLEL_SCAN).
//
// Coverage:
//   1. Concatenating the per-partition triples in ASCENDING partition order
//      reproduces the EXACT sequence (content AND order) emitted by the
//      legacy single-consumer scan_label_edge_with_endpoints. This is the
//      byte-identity-critical invariant: the builder's ordered merge replays
//      the ParallelEdgeDetector window + add_edge in this same global order,
//      so the sorter sees byte-identical input.
//   2. The method reports the number of partitions it actually used, and
//      every triple lands in exactly one partition (no loss, no duplication).
//
// Mirrors the process-lifetime fixture style of
// native_scanner_parallel_abort_test.cc — the populated B+Trees (label_edge,
// edge_from_to) are bulk-built with the production projection writer
// (RadixPartitionSort -> BPTLeafWriter + BPTDirWriter), the same way
// graph_project materializes projection indexes.
//
// Scope note: this exercises the SCANNER seam (the in-worker partitioned
// scan). The BUILDER-level ordered merge + the windowed SINGLE-duplicate
// QueryException replay are covered by:
//   - scripts/test_projection_radix.sh (byte-identical .leaf/.dir for the
//     PARALLEL mode vs the classic baseline on cora — proves the merge
//     reproduces the classic global scan order), and
//   - native_scanner_parallel_abort_test.cc (a synthetic callback throw on
//     the parallel path propagates cleanly without std::terminate()).
// The SINGLE-duplicate throw is structurally identical to the classic path
// (same ParallelEdgeDetector::process_edge call replayed in the same global
// order), so it is not re-asserted here; it is only reachable through the
// full graph_project HTTP procedure on a dataset that actually contains
// parallel edges.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/native_scanner.h"
#include "graph_models/gql/projection/radix_partition_sort.h"
#include "graph_models/gql/projection/streaming_record_buffer.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"
#include "system/system.h"

namespace {

namespace fs = std::filesystem;

// Enough edges to span several partitions with multiple records each.
constexpr uint64_t kNumEdges = 5000;

constexpr uint64_t kTypeIdRaw = ObjectId::MASK_EDGE_LABEL | 11ULL;

// Process-lifetime fixture (System singletons can only be bound once per
// process). Uses a unique randomly-named db folder so the test never collides
// with a previous run.
class ScannerFixture {
public:
    static ScannerFixture& instance() {
        static ScannerFixture f;
        return f;
    }

    GQL::NativeScanner& scanner() { return *scanner_; }

    ObjectId type_id() const { return ObjectId(kTypeIdRaw); }

private:
    ScannerFixture() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        db_folder_ = "test_db_native_scanner_partitioned_" + std::to_string(rng());
        fs::remove_all(db_folder_);

        system_.reset(new System(
            db_folder_,
            1024 * 1024,        // str_static_size
            1024 * 1024,        // str_dynamic_size
            64 * 1024 * 1024,   // shared_buffer_size
            32 * 1024 * 1024,   // private_buffer_size
            1024 * 1024,        // tensor_static_size
            1024 * 1024,        // tensor_dynamic_size
            1                   // workers
        ));

        query_ctx_.reset(new QueryContext());
        QueryContext::set_query_ctx(query_ctx_.get());

        setenv("MDB_PROJECTION_RADIX_GPU", "0", 1);

        const std::string scratch = db_folder_ + "/sort_scratch";
        fs::create_directories(scratch);
        {
            GQL::StreamingRecordBuffer<2> le_buf(scratch + "/tmp_label_edge");
            GQL::StreamingRecordBuffer<3> eft_buf(scratch + "/tmp_edge_from_to");
            for (uint64_t i = 0; i < kNumEdges; ++i) {
                const uint64_t eid  = ObjectId::MASK_DIRECTED_EDGE | (i + 1);
                const uint64_t from = ObjectId::MASK_NODE | (2 * i + 1);
                const uint64_t to   = ObjectId::MASK_NODE | (2 * i + 2);

                le_buf.push_back(Record<2>{{ kTypeIdRaw, eid }});
                eft_buf.push_back(Record<3>{{ eid, from, to }});
            }

            GQL::RadixPartitionSort<2>::Config le_cfg;
            le_cfg.scratch_dir = scratch + "/le";
            fs::create_directories(le_cfg.scratch_dir);
            GQL::RadixPartitionSort<2> le_sort(le_cfg);
            le_sort.scan_and_partition(le_buf, kNumEdges);
            le_sort.sort_and_write(db_folder_ + "/label_edge");

            GQL::RadixPartitionSort<3>::Config eft_cfg;
            eft_cfg.scratch_dir = scratch + "/eft";
            fs::create_directories(eft_cfg.scratch_dir);
            GQL::RadixPartitionSort<3> eft_sort(eft_cfg);
            eft_sort.scan_and_partition(eft_buf, kNumEdges);
            eft_sort.sort_and_write(db_folder_ + "/edge_from_to");
        }

        label_node_   = std::make_unique<BPlusTree<2>>("label_node");
        label_edge_   = std::make_unique<BPlusTree<2>>("label_edge");
        from_to_edge_ = std::make_unique<BPlusTree<3>>("from_to_edge");
        edge_from_to_ = std::make_unique<BPlusTree<3>>("edge_from_to");
        n1_n2_edge_   = std::make_unique<BPlusTree<3>>("n1_n2_edge");
        edge_n1_n2_   = std::make_unique<BPlusTree<3>>("edge_n1_n2");

        scanner_ = std::make_unique<GQL::NativeScanner>(
            label_node_.get(), label_edge_.get(),
            from_to_edge_.get(), edge_from_to_.get(),
            n1_n2_edge_.get(), edge_n1_n2_.get());
    }

    std::string                         db_folder_;
    std::unique_ptr<System>             system_;
    std::unique_ptr<QueryContext>       query_ctx_;
    std::unique_ptr<BPlusTree<2>>       label_node_;
    std::unique_ptr<BPlusTree<2>>       label_edge_;
    std::unique_ptr<BPlusTree<3>>       from_to_edge_;
    std::unique_ptr<BPlusTree<3>>       edge_from_to_;
    std::unique_ptr<BPlusTree<3>>       n1_n2_edge_;
    std::unique_ptr<BPlusTree<3>>       edge_n1_n2_;
    std::unique_ptr<GQL::NativeScanner> scanner_;
};

using Triple = std::tuple<uint64_t, uint64_t, uint64_t>;

// Sequential reference via the legacy single-consumer scan, forced sequential
// so the reference is unambiguous.
std::vector<Triple> collect_sequential(GQL::NativeScanner& scanner, ObjectId type_id) {
    setenv("MDB_PROJECTION_PARALLEL_EDGE_SCAN", "0", 1);
    std::vector<Triple> out;
    scanner.scan_label_edge_with_endpoints(
        type_id,
        [&](ObjectId eid, ObjectId from, ObjectId to) {
            out.emplace_back(eid.id, from.id, to.id);
        });
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — concatenating per-partition output in ascending partition order
// reproduces the exact sequential sequence (content AND order). This is the
// invariant the builder's ordered merge relies on for byte-identical sorter
// input.
// ---------------------------------------------------------------------------
TEST(NativeScannerPartitioned, AscendingMergeMatchesSequential) {
    auto& fx = ScannerFixture::instance();

    const auto sequential = collect_sequential(fx.scanner(), fx.type_id());
    ASSERT_EQ(sequential.size(), kNumEdges);

    setenv("MDB_PROJECTION_EDGE_SCAN_PARTITIONS", "8", 1);
    std::vector<std::vector<Triple>> per_partition;
    std::size_t k = fx.scanner().scan_label_edge_endpoints_partitioned(
        fx.type_id(),
        /*num_partitions=*/8,
        [&](std::size_t part_idx, ObjectId eid, ObjectId from, ObjectId to) {
            // part_idx must be a valid partition index; grow lazily.
            if (part_idx >= per_partition.size()) {
                per_partition.resize(part_idx + 1);
            }
            per_partition[part_idx].emplace_back(eid.id, from.id, to.id);
        });

    ASSERT_GE(k, 1u);
    ASSERT_LE(per_partition.size(), k);

    std::vector<Triple> merged;
    merged.reserve(kNumEdges);
    for (std::size_t p = 0; p < per_partition.size(); ++p) {
        for (const auto& t : per_partition[p]) {
            merged.push_back(t);
        }
    }

    EXPECT_EQ(merged, sequential);
}

// ---------------------------------------------------------------------------
// Test 2 — every edge lands in exactly one partition (no loss, no
// duplication), and the reported partition count is honored.
// ---------------------------------------------------------------------------
TEST(NativeScannerPartitioned, EveryEdgeCoveredOnce) {
    auto& fx = ScannerFixture::instance();

    setenv("MDB_PROJECTION_EDGE_SCAN_PARTITIONS", "4", 1);
    std::vector<Triple> all;
    std::size_t k = fx.scanner().scan_label_edge_endpoints_partitioned(
        fx.type_id(),
        /*num_partitions=*/4,
        [&](std::size_t part_idx, ObjectId eid, ObjectId from, ObjectId to) {
            ASSERT_LT(part_idx, k);
            all.emplace_back(eid.id, from.id, to.id);
        });

    ASSERT_GE(k, 1u);
    EXPECT_EQ(all.size(), kNumEdges);

    // No duplicates: edge_id is unique per edge in the fixture.
    std::vector<uint64_t> eids;
    eids.reserve(all.size());
    for (const auto& t : all) eids.push_back(std::get<0>(t));
    std::sort(eids.begin(), eids.end());
    EXPECT_EQ(std::unique(eids.begin(), eids.end()), eids.end());
}
