// Spec C3 stage 3 module 1: validate cross-stream sync primitives.
//
// These tests don't touch the training loop; they exercise the LibTorch
// CUDA stream + event API directly to ensure our understanding is correct
// before wiring it into the prefetcher (Module 4) and training loop
// (Module 5). Skip cleanly when CUDA is not available.

#include <gtest/gtest.h>

#ifdef ENABLE_CUDA_ASSEMBLER

#include <atomic>
#include <chrono>
#include <thread>

#include <torch/torch.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "gnn/core/stage3_streams.h"

using namespace mdb::gnn;

class Stage3StreamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
    }
};

// ---------------------------------------------------------------------------
// Test 1: getStreamFromPool returns distinct, usable streams.
// ---------------------------------------------------------------------------
TEST_F(Stage3StreamsTest, AcquirePoolStream_DifferentStreams) {
    auto a = acquire_pool_stream();
    auto b = acquire_pool_stream();
    // Pool is round-robin so consecutive calls return different streams.
    EXPECT_NE(a.stream(), b.stream())
        << "pool stream should not collide on consecutive acquire";

    // Both streams should be usable: synchronizing them must succeed.
    a.synchronize();
    b.synchronize();
}

// ---------------------------------------------------------------------------
// Test 2: CUDAEvent default-constructs lazily (no event until first record).
// ---------------------------------------------------------------------------
TEST_F(Stage3StreamsTest, CUDAEvent_LazyCreation) {
    at::cuda::CUDAEvent ev;
    EXPECT_FALSE(ev.isCreated())
        << "CUDAEvent should not allocate cudaEvent_t until first record";
    EXPECT_TRUE(ev.query())
        << "uncreated event reports query() = true (no work pending)";
}

// ---------------------------------------------------------------------------
// Test 3: record/query lifecycle on a stream.
// ---------------------------------------------------------------------------
TEST_F(Stage3StreamsTest, CUDAEvent_RecordAndQuery) {
    auto stream = acquire_pool_stream();

    // Launch a small kernel by doing a torch op on this stream — torch::ones
    // submits a CUDA kernel, so the event after it has work to wait for.
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        auto t = torch::ones({1024, 1024},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        // Force a non-trivial dependency to make sure the event is meaningful.
        t = t * 2.0f;
    }

    at::cuda::CUDAEvent ev;
    ev.record(stream);
    EXPECT_TRUE(ev.isCreated())
        << "after record() the event must be created";

    // Synchronize: after stream.synchronize(), the event must report ready.
    stream.synchronize();
    EXPECT_TRUE(ev.query())
        << "event should be ready after stream.synchronize()";
}

// ---------------------------------------------------------------------------
// Test 4: Cross-stream sync — block(stream B) makes B wait for A's event.
// This is THE primitive Stage 3 relies on.
// ---------------------------------------------------------------------------
TEST_F(Stage3StreamsTest, CrossStreamSync_BlockOrdering) {
    auto stream_a = acquire_pool_stream();
    auto stream_b = acquire_pool_stream();

    constexpr int64_t N = 4096;

    // Launch heavy work on stream A.
    torch::Tensor produced;
    {
        c10::cuda::CUDAStreamGuard guard(stream_a);
        produced = torch::ones({N, N},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        // Slow kernel — large matmul.
        produced = produced.matmul(produced) + 1.0f;
    }

    at::cuda::CUDAEvent ready;
    ready.record(stream_a);

    // Stream B blocks on A's event before reading the tensor.
    ready.block(stream_b);

    torch::Tensor consumed;
    {
        c10::cuda::CUDAStreamGuard guard(stream_b);
        // Simply read produced. Without the event sync, this could read
        // stale memory; with the sync, stream B waits until stream A's
        // matmul completes.
        consumed = produced.clone();
    }

    stream_b.synchronize();
    auto host = consumed.cpu();
    auto acc = host.accessor<float, 2>();

    // After matmul of N×N ones, each cell = N + 1 (the scalar "1" added).
    EXPECT_NEAR(acc[0][0], static_cast<float>(N + 1), 1e-3)
        << "cross-stream sync must observe completed work from stream A";
}

// ---------------------------------------------------------------------------
// Test 5: A stream can be used from a different host thread once acquired.
// This is critical: the prefetcher worker thread acquires the stream and
// records work on it; the main thread later blocks on the event.
// ---------------------------------------------------------------------------
TEST_F(Stage3StreamsTest, StreamUsableAcrossHostThreads) {
    auto worker_stream = acquire_pool_stream();
    auto main_stream   = acquire_pool_stream();

    constexpr int64_t N = 2048;
    torch::Tensor produced;
    at::cuda::CUDAEvent ready;

    std::atomic<bool> worker_done{false};
    std::thread worker([&] {
        c10::cuda::CUDAStreamGuard guard(worker_stream);
        produced = torch::full({N, N}, 7.0f,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        // Submit a kernel that takes nontrivial GPU time.
        produced = produced * produced;  // = 49 in every cell
        ready.record(worker_stream);
        worker_done.store(true);
    });
    worker.join();
    EXPECT_TRUE(worker_done.load());

    // Main thread reads on main_stream after blocking on ready.
    torch::Tensor consumed;
    {
        c10::cuda::CUDAStreamGuard guard(main_stream);
        ready.block(main_stream);
        consumed = produced.clone();
    }
    main_stream.synchronize();

    auto host = consumed.cpu();
    auto acc = host.accessor<float, 2>();
    EXPECT_NEAR(acc[N / 2][N / 2], 49.0f, 1e-3)
        << "main thread must observe worker's writes after event.block";
}

// ---------------------------------------------------------------------------
// Test 6: Move semantics on StreamSignal — must transfer cleanly between threads.
// ---------------------------------------------------------------------------
TEST_F(Stage3StreamsTest, StreamSignal_Movable) {
    StreamSignal s1{acquire_pool_stream()};
    auto raw_stream_id_before = s1.stream.stream();

    // Move-construct.
    StreamSignal s2{std::move(s1)};
    EXPECT_EQ(s2.stream.stream(), raw_stream_id_before)
        << "move-construct must preserve underlying stream";
}

// ---------------------------------------------------------------------------
// Test 7: Stress — many small events, no leaks (smoke test for cudaEventDestroy).
// ---------------------------------------------------------------------------
TEST_F(Stage3StreamsTest, EventStress_NoLeaks) {
    auto stream = acquire_pool_stream();

    for (int i = 0; i < 1000; ++i) {
        at::cuda::CUDAEvent ev;
        {
            c10::cuda::CUDAStreamGuard guard(stream);
            auto t = torch::zeros({16, 16},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            t.add_(1.0f);
        }
        ev.record(stream);
        // ev goes out of scope → ~CUDAEvent → cudaEventDestroy.
    }
    stream.synchronize();
    SUCCEED() << "1000 record/destroy cycles completed without OOM or leak";
}

#endif // ENABLE_CUDA_ASSEMBLER
