// native_scanner_parallel_abort_test.cc
//
// Unit tests for the parallel branch of
// GQL::NativeScanner::scan_label_edge_with_endpoints (the TBB
// producer-consumer pipeline, default ON in TBB builds).
//
// Coverage:
//   1. An exception thrown by the user callback (the documented
//      duplicate-edge QueryException path in SINGLE aggregation mode)
//      propagates cleanly to the caller. The producer thread and any
//      workers blocked on queue backpressure are signalled and joined —
//      previously the unwind destroyed a joinable std::thread, calling
//      std::terminate() and aborting the whole server.
//   2. The scanner remains fully usable after such an aborted scan
//      (no stuck threads, no poisoned queues).
//   3. The parallel path emits the same (edge, from, to) sequence as the
//      sequential path — including the same callback ordering.
//
// Mirrors the process-lifetime fixture style of
// topology_accessor_adjacency_cache_test.cc (System singletons can only be
// bound once per process; unique randomly-named db folder per run). The
// populated B+Trees (label_edge, edge_from_to) are bulk-built with the
// production projection writer (RadixPartitionSort → BPTLeafWriter +
// BPTDirWriter, byte-identical to the CLASSIC sorter backend) — the same
// way graph_project materializes projection indexes. Runtime
// BPlusTree::insert is NOT used: production never populates projection
// indexes through it, and the legacy v1 leaf-split path corrupts the
// directory for record shapes whose page size lands exactly on Page::SIZE
// (both split-partition loops use `> Page::SIZE` while insert splits on
// `>= Page::SIZE`, yielding a double split with empty left/right leaves
// and a bogus all-zero separator key routing every search to an empty
// leaf).

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
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

// Enough edges that a worker fills its bounded queue (1024 slots) and
// blocks on backpressure while the consumer is throwing — the exact
// deadlock-then-terminate scenario the abort protocol must handle.
constexpr uint64_t kNumEdges = 5000;

constexpr uint64_t kTypeIdRaw = ObjectId::MASK_EDGE_LABEL | 7ULL;

// Process-lifetime fixture (System singletons can only be bound once per
// process). Uses a unique randomly-named db folder so the test never
// collides with a previous run.
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
        db_folder_ = "test_db_native_scanner_abort_" + std::to_string(rng());
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

        // Bulk-build the two populated indexes through the production
        // projection writer pipeline (RadixPartitionSort Phase 1-3 →
        // BPTLeafWriter/BPTDirWriter), then open them by name — the same
        // write-then-open sequence graph_project uses. Records carry the
        // exact production key layouts:
        //   label_edge   {label_id, edge_id}        (projection_storage.cc)
        //   edge_from_to {edge_id, from, to}        (projection_storage.cc)
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

        // Open the bulk-built indexes plus valid empty trees for the rest
        // (a brand-new BPlusTree over zeroed files is a valid empty tree).
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

void set_parallel_env(bool on) {
    setenv("MDB_PROJECTION_PARALLEL_EDGE_SCAN", on ? "1" : "0", 1);
    setenv("MDB_PROJECTION_EDGE_SCAN_PARTITIONS", "8", 1);
}

using Triple = std::tuple<uint64_t, uint64_t, uint64_t>;

std::vector<Triple> collect_all(GQL::NativeScanner& scanner, ObjectId type_id) {
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
// Test 1 — a callback exception on the parallel path must propagate to the
// caller (instead of unwinding past the joinable producer thread, which
// calls std::terminate() and aborts the process).
// ---------------------------------------------------------------------------
TEST(NativeScannerParallelAbort, CallbackExceptionPropagates) {
    auto& fx = ScannerFixture::instance();
    set_parallel_env(true);

    uint64_t seen = 0;
    EXPECT_THROW(
        fx.scanner().scan_label_edge_with_endpoints(
            fx.type_id(),
            [&](ObjectId, ObjectId, ObjectId) {
                if (++seen > 10) {
                    throw std::runtime_error(
                        "synthetic callback failure (stands in for the "
                        "duplicate-edge QueryException of SINGLE mode)");
                }
            }),
        std::runtime_error);

    // The callback ran at least up to the throw point, and the scan did
    // NOT silently run to completion after the failure.
    EXPECT_GT(seen, 10u);
    EXPECT_LT(seen, kNumEdges);
}

// ---------------------------------------------------------------------------
// Test 2 — after an aborted scan, the scanner is fully usable: all worker /
// producer threads were joined and a fresh scan sees every edge.
// ---------------------------------------------------------------------------
TEST(NativeScannerParallelAbort, ScannerUsableAfterCallbackThrow) {
    auto& fx = ScannerFixture::instance();
    set_parallel_env(true);

    EXPECT_THROW(
        fx.scanner().scan_label_edge_with_endpoints(
            fx.type_id(),
            [](ObjectId, ObjectId, ObjectId) {
                throw std::runtime_error("fail on first edge");
            }),
        std::runtime_error);

    auto triples = collect_all(fx.scanner(), fx.type_id());
    EXPECT_EQ(triples.size(), kNumEdges);
}

// ---------------------------------------------------------------------------
// Test 3 — the parallel path emits the same sequence (content AND callback
// order) as the sequential path.
// ---------------------------------------------------------------------------
TEST(NativeScannerParallelAbort, ParallelMatchesSequential) {
    auto& fx = ScannerFixture::instance();

    set_parallel_env(false);
    auto sequential = collect_all(fx.scanner(), fx.type_id());

    set_parallel_env(true);
    auto parallel = collect_all(fx.scanner(), fx.type_id());

    ASSERT_EQ(sequential.size(), kNumEdges);
    EXPECT_EQ(parallel, sequential);
}
