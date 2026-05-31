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
// Spec C3 stage 3 module 4 tests: prefetcher with use_cuda_streams=true
// records a CUDAEvent into MiniBatch.ready_event.
// -----------------------------------------------------------------------------

#ifdef ENABLE_CUDA_ASSEMBLER
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

TEST_F(AsyncBatchPrefetcherTest, Stage3_StreamsDisabled_NoEventRecorded) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    const std::string sname = "asynccp_streams_off";
    create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/2,
                                    /*use_cuda_streams=*/false);

    prefetcher.prefetch(0);
    auto batch = prefetcher.next();

    // With streams disabled, the event must remain uncreated (legacy path).
    EXPECT_FALSE(batch.ready_event.isCreated())
        << "use_cuda_streams=false must not record an event";
}

TEST_F(AsyncBatchPrefetcherTest, Stage3_StreamsEnabled_EventRecordedAndQueryable) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    const std::string sname = "asynccp_streams_on";
    create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/2,
                                    /*use_cuda_streams=*/true);

    prefetcher.prefetch(0);
    auto batch = prefetcher.next();

    // The ready_event must be created (recorded) by the worker.
    EXPECT_TRUE(batch.ready_event.isCreated())
        << "use_cuda_streams=true must record an event after assembly";

    // Synchronously wait for the event — must return without error.
    if (batch.ready_event.isCreated()) {
        cudaEventSynchronize(batch.ready_event.event());
        EXPECT_TRUE(batch.ready_event.query())
            << "event must report completed after host synchronize";
    }
}

TEST_F(AsyncBatchPrefetcherTest, Stage3_StreamsEnabled_FeaturesContentMatchesSync) {
    // The most important correctness test: assembling with streams must
    // produce numerically identical features to assembling without streams,
    // assuming the consumer correctly blocks on the event.
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    const std::string sname = "asynccp_streams_match";
    create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    // Reference: synchronous assembly on default stream.
    BatchAssembler asm_sync(fm, storage, &ls, &ss, rm);
    auto reference = asm_sync.assemble(0);

    // With streams: prefetcher worker uses a pool stream.
    BatchAssembler asm_async(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(asm_async, /*queue_size=*/2,
                                    /*use_cuda_streams=*/true);
    prefetcher.prefetch(0);
    auto batch = prefetcher.next();

    // Consumer-side sync via event — get the consumer onto a fresh train
    // stream and block it on the producer event.
    auto train_stream = c10::cuda::getStreamFromPool();
    {
        c10::cuda::CUDAStreamGuard guard(train_stream);
        if (batch.ready_event.isCreated()) {
            batch.ready_event.block(train_stream);
        }
        // Use the tensor: read into CPU. Without proper sync this could be
        // garbage; with sync it must match the reference.
        auto features_host = batch.features.cpu();
        auto reference_host = reference.features.cpu();

        ASSERT_EQ(features_host.sizes(), reference_host.sizes());
        EXPECT_TRUE(torch::equal(features_host, reference_host))
            << "stream-assembled features must match sync-assembled features"
               " (proper event sync should guarantee correctness)";
    }
    train_stream.synchronize();
}
#endif // ENABLE_CUDA_ASSEMBLER

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

// -----------------------------------------------------------------------------
// Round 3B (2026-05-15): multi-worker AsyncBatchPrefetcher.
// -----------------------------------------------------------------------------

// Test 9: num_workers=0 is rejected at construction.
TEST_F(AsyncBatchPrefetcherTest, ZeroWorkers_Rejected) {
    const std::string sname = "asynccp_zero_workers";
    create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    EXPECT_THROW({
        AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/2,
                                        /*use_cuda_streams=*/false,
                                        /*num_workers=*/0);
    }, std::invalid_argument);
}

// Test 10: num_workers is clamped to queue_size.
TEST_F(AsyncBatchPrefetcherTest, NumWorkers_ClampedToQueueSize) {
    const std::string sname = "asynccp_clamp";
    create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);

    // Pass num_workers=10 with queue_size=2 → effective should be 2.
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/2,
                                    /*use_cuda_streams=*/false,
                                    /*num_workers=*/10);
    EXPECT_EQ(prefetcher.num_workers(), 2u)
        << "num_workers must be clamped to queue_size";
}

// Test 11: with num_workers=2 and FeatureMatrix mode, FIFO order is preserved.
TEST_F(AsyncBatchPrefetcherTest, MultiWorker_FifoOrder_Preserved) {
    const std::string sname = "asynccp_mw_fifo";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/4,
                                    /*use_cuda_streams=*/false,
                                    /*num_workers=*/2);

    const uint64_t total = cat.total_batches;
    ASSERT_GE(total, 2u);

    // Submit all upfront (with batch_id == submission_position) and consume.
    for (uint64_t b = 0; b < total; ++b) {
        prefetcher.prefetch(b);
    }
    for (uint64_t b = 0; b < total; ++b) {
        auto batch = prefetcher.next();
        EXPECT_EQ(batch.batch_id, b)
            << "FIFO violated at index " << b << " under num_workers=2";
    }
}

// Test 12: bit-identical results between num_workers=1 and num_workers=2.
// This is the correctness test the advisor flagged as mandatory.
//
// Submits with queue_size >= num_to_check so the producer can prime the
// full set up-front (no backpressure stall, no chance of consumer-side
// re-ordering hiding a non-deterministic assemble result).
TEST_F(AsyncBatchPrefetcherTest, MultiWorker_BitIdenticalToSingleWorker) {
    const std::string sname = "asynccp_mw_identical";
    auto cat = create_sample_storage(sname);
    ASSERT_GE(cat.total_batches, 2u);

    const uint64_t num_to_check = std::min<uint64_t>(cat.total_batches, 4u);
    const size_t   queue_size   = std::max<size_t>(num_to_check, 2u);

    // Reference run: num_workers=1 (single-threaded path).
    std::vector<torch::Tensor> ref_features;
    std::vector<torch::Tensor> ref_edges_layer0;
    std::vector<uint64_t>      ref_batch_ids;
    {
        auto fm      = FeatureMatrix::open(fmat_path_);
        auto rm      = RowMapping::open(rmap_path_);
        auto ls      = LabelStore::open(labels_path_);
        auto ss      = SplitStore::open(splits_path_);
        auto storage = SampleStorage::open(
            SampleStorage::get_storage_path(db_folder_, sname));

        BatchAssembler assembler(fm, storage, &ls, &ss, rm);
        AsyncBatchPrefetcher prefetcher(assembler, queue_size,
                                        /*use_cuda_streams=*/false,
                                        /*num_workers=*/1);

        for (uint64_t b = 0; b < num_to_check; ++b) prefetcher.prefetch(b);
        for (uint64_t b = 0; b < num_to_check; ++b) {
            auto batch = prefetcher.next();
            ref_batch_ids.push_back(batch.batch_id);
            ref_features.push_back(batch.features.clone());
            if (!batch.edge_indices.empty()) {
                ref_edges_layer0.push_back(batch.edge_indices[0].clone());
            } else {
                ref_edges_layer0.emplace_back();
            }
        }
    }

    // Multi-worker run: num_workers=2, same queue_size as reference.
    {
        auto fm      = FeatureMatrix::open(fmat_path_);
        auto rm      = RowMapping::open(rmap_path_);
        auto ls      = LabelStore::open(labels_path_);
        auto ss      = SplitStore::open(splits_path_);
        auto storage = SampleStorage::open(
            SampleStorage::get_storage_path(db_folder_, sname));

        BatchAssembler assembler(fm, storage, &ls, &ss, rm);
        AsyncBatchPrefetcher prefetcher(assembler, queue_size,
                                        /*use_cuda_streams=*/false,
                                        /*num_workers=*/2);

        for (uint64_t b = 0; b < num_to_check; ++b) prefetcher.prefetch(b);
        for (uint64_t b = 0; b < num_to_check; ++b) {
            auto batch = prefetcher.next();
            EXPECT_EQ(batch.batch_id, ref_batch_ids[b])
                << "batch_id mismatch at index " << b;
            ASSERT_EQ(batch.features.sizes(), ref_features[b].sizes())
                << "feature shape mismatch at batch " << b;
            EXPECT_TRUE(torch::equal(batch.features, ref_features[b]))
                << "features tensor differs at batch " << b
                << " between num_workers=1 and num_workers=2";
            if (!batch.edge_indices.empty() && ref_edges_layer0[b].defined()) {
                EXPECT_TRUE(torch::equal(batch.edge_indices[0], ref_edges_layer0[b]))
                    << "edge_indices[0] differs at batch " << b;
            }
        }
    }
}

// Test 13: multi-worker error propagation — bad batch_id from any worker
// surfaces via next() in submission order.
TEST_F(AsyncBatchPrefetcherTest, MultiWorker_ErrorPropagation) {
    const std::string sname = "asynccp_mw_error";
    auto cat = create_sample_storage(sname);

    auto fm      = FeatureMatrix::open(fmat_path_);
    auto rm      = RowMapping::open(rmap_path_);
    auto ls      = LabelStore::open(labels_path_);
    auto ss      = SplitStore::open(splits_path_);
    auto storage = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, sname));

    BatchAssembler assembler(fm, storage, &ls, &ss, rm);
    AsyncBatchPrefetcher prefetcher(assembler, /*queue_size=*/4,
                                    /*use_cuda_streams=*/false,
                                    /*num_workers=*/2);

    // Submit one valid batch then one out-of-range. The valid one should
    // deliver successfully, then the out-of-range should throw.
    ASSERT_GE(cat.total_batches, 1u);
    prefetcher.prefetch(0);
    prefetcher.prefetch(cat.total_batches + 100);

    auto good = prefetcher.next();
    EXPECT_EQ(good.batch_id, 0u);

    EXPECT_THROW({
        (void) prefetcher.next();
    }, std::exception);
}
