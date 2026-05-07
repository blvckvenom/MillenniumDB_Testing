// =============================================================================
// Spec C3 stage 1 (2026-05-07): AsyncBatchPrefetcher unit tests.
//
// Verifies the producer-consumer queue contract with a real BatchAssembler
// over the synthetic 8-node fixture from training_loop_test_fixture.h.
// Tests do NOT exercise the training loop — that integration is gated by
// stage 1.B in the C3 plan.
// =============================================================================

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/training/async_batch_prefetcher.h"
#include "gnn/training/batch_assembler.h"
#include "gnn/training/label_store.h"
#include "gnn/training/split_store.h"
#include "gnn/training/mini_batch.h"
#include "gnn/tests/training_loop_test_fixture.h"

namespace fs = std::filesystem;

using namespace mdb::gnn;
using namespace mdb::gnn::testing_util;

class AsyncBatchPrefetcherTest : public TrainingLoopTestFixture {};

// -----------------------------------------------------------------------------
// Test 1: Basic flow — prefetch a batch, retrieve it, content matches sync.
// -----------------------------------------------------------------------------
TEST_F(AsyncBatchPrefetcherTest, BasicFlow_PrefetchThenNext) {
    const std::string sname = "asynccp_basic";
    create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler asm_sync(fm, storage, &ls, &ss, rm);
    auto reference = asm_sync.assemble(/*batch_id=*/0);

    BatchAssembler asm_async(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(asm_async, /*queue_size=*/2);

    prefetcher.prefetch(0);
    auto async_batch = prefetcher.next();

    EXPECT_EQ(async_batch.batch_id, reference.batch_id);
    EXPECT_EQ(async_batch.num_seeds, reference.num_seeds);
    EXPECT_EQ(async_batch.num_nodes, reference.num_nodes);
    ASSERT_EQ(async_batch.features.sizes(), reference.features.sizes());
    EXPECT_TRUE(torch::equal(async_batch.features, reference.features))
        << "features tensor differs between sync and async assembly";
}

// -----------------------------------------------------------------------------
// Test 2: FIFO order — prefetched batches are returned in submission order.
// -----------------------------------------------------------------------------
TEST_F(AsyncBatchPrefetcherTest, FifoOrder_AcrossMultipleBatches) {
    const std::string sname = "asynccp_fifo";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/2);

    const uint64_t total = cat.total_batches;
    ASSERT_GE(total, 2u) << "Fixture must produce ≥ 2 batches for this test";

    // Submit each batch in order, immediately consume; each delivery must
    // match the order of submission.
    for (uint64_t b = 0; b < total; ++b) {
        prefetcher.prefetch(b);
        auto batch = prefetcher.next();
        EXPECT_EQ(batch.batch_id, b)
            << "FIFO violated at index " << b;
    }
}

// -----------------------------------------------------------------------------
// Test 3: Backpressure — prefetch blocks when queue is full.
// -----------------------------------------------------------------------------
TEST_F(AsyncBatchPrefetcherTest, Backpressure_PrefetchBlocksWhenFull) {
    const std::string sname = "asynccp_backpressure";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/2);

    ASSERT_GE(cat.total_batches, 2u);
    prefetcher.prefetch(0);
    prefetcher.prefetch(1);

    // Both fill in_flight up to the queue cap. A third prefetch must block.
    std::atomic<bool> third_returned{false};
    std::thread producer([&] {
        if (cat.total_batches >= 3) {
            prefetcher.prefetch(2);
            third_returned.store(true);
        } else {
            // Fixture only has 2 batches; we reuse 0 to exercise the bound.
            prefetcher.prefetch(0);
            third_returned.store(true);
        }
    });

    // Give the producer thread a generous chance to commit its prefetch.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(third_returned.load())
        << "third prefetch should be blocked on backpressure";

    // Drain one slot; the blocked producer should now complete.
    auto first = prefetcher.next();
    EXPECT_EQ(first.batch_id, 0u);
    producer.join();
    EXPECT_TRUE(third_returned.load()) << "producer should have unblocked";

    // Drain remaining items so destructor is clean.
    (void) prefetcher.next();
    (void) prefetcher.next();
}

// -----------------------------------------------------------------------------
// Test 4: shutdown is idempotent and prevents new prefetches.
// -----------------------------------------------------------------------------
TEST_F(AsyncBatchPrefetcherTest, Shutdown_Idempotent_RejectsNewPrefetches) {
    const std::string sname = "asynccp_shutdown";
    create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler);

    EXPECT_FALSE(prefetcher.is_shutdown());
    prefetcher.shutdown();
    EXPECT_TRUE(prefetcher.is_shutdown());
    prefetcher.shutdown();  // idempotent
    EXPECT_TRUE(prefetcher.is_shutdown());

    EXPECT_THROW(prefetcher.prefetch(0), std::runtime_error)
        << "prefetch after shutdown must throw";
}

// -----------------------------------------------------------------------------
// Test 5: Worker exception is propagated by next().
// -----------------------------------------------------------------------------
TEST_F(AsyncBatchPrefetcherTest, ErrorPropagation_InvalidBatchId) {
    const std::string sname = "asynccp_error";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/4);

    // batch_id beyond catalog total_batches: SampleStorage::read_sample
    // throws inside the worker; next() must rethrow.
    prefetcher.prefetch(cat.total_batches + 100);

    EXPECT_THROW({
        (void) prefetcher.next();
    }, std::exception)
        << "next() must rethrow the worker's exception for invalid batch_id";
}

// -----------------------------------------------------------------------------
// Test 6: Consume-after-shutdown — outstanding prefetches still deliver.
// -----------------------------------------------------------------------------
TEST_F(AsyncBatchPrefetcherTest, ConsumeAfterShutdown_DrainsOutstanding) {
    const std::string sname = "asynccp_drain";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/4);

    ASSERT_GE(cat.total_batches, 2u);
    prefetcher.prefetch(0);
    prefetcher.prefetch(1);
    prefetcher.shutdown();

    // After shutdown, the two queued batches are still deliverable.
    auto b0 = prefetcher.next();
    auto b1 = prefetcher.next();
    EXPECT_EQ(b0.batch_id, 0u);
    EXPECT_EQ(b1.batch_id, 1u);

    // Now genuinely drained — third next() must throw.
    EXPECT_THROW({
        (void) prefetcher.next();
    }, std::runtime_error);
}

// -----------------------------------------------------------------------------
// Test 7: queue_size=0 is rejected at construction.
// -----------------------------------------------------------------------------
TEST_F(AsyncBatchPrefetcherTest, ZeroQueueSize_Rejected) {
    const std::string sname = "asynccp_zero";
    create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    EXPECT_THROW({
        AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/0);
    }, std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Test 8: Destructor properly shuts down + joins worker even with batches in flight.
// -----------------------------------------------------------------------------
TEST_F(AsyncBatchPrefetcherTest, Destructor_CleanShutdown_WithBatchesInFlight) {
    const std::string sname = "asynccp_dtor";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    {
        AsyncBatchPrefetcher prefetcher(assembler);
        ASSERT_GE(cat.total_batches, 2u);
        prefetcher.prefetch(0);
        prefetcher.prefetch(1);
        // Leave scope WITHOUT consuming. Destructor must terminate cleanly.
    }
    // If we reached here without hanging or asan / leak triggers, the
    // destructor handled the in-flight batches gracefully.
    SUCCEED();
}
