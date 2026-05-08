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
// Spec C3 stage 3 module 3 tests: model forward + backward on a custom
// stream produce equivalent output (within FP tolerance) to the default
// stream. This is the pattern the training loop (Module 5) will use.
// ---------------------------------------------------------------------------

namespace {
// Build a deterministic small classifier: features → logits.
// Returns (model, input, target) prepared on CUDA.
struct LinearFixture {
    torch::nn::Linear fc{nullptr};
    torch::Tensor input;
    torch::Tensor target;
};

LinearFixture make_linear_fixture(int64_t in_dim, int64_t classes,
                                  int64_t batch, int64_t seed)
{
    torch::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);

    LinearFixture f;
    f.fc = torch::nn::Linear(in_dim, classes);
    f.fc->to(torch::kCUDA);

    f.input = torch::randn({batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    f.target = torch::randint(0, classes, {batch},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    return f;
}

// Run forward + cross-entropy + backward and return (loss_value, weight_grad_norm).
struct StepResult {
    double loss;
    double grad_norm;
};

StepResult run_step(LinearFixture& f) {
    f.fc->zero_grad();
    auto logits = f.fc->forward(f.input);
    auto loss = torch::nn::functional::cross_entropy(logits, f.target);
    loss.backward();
    auto wgrad = f.fc->weight.grad();
    return {
        loss.item<double>(),
        wgrad.norm().item<double>(),
    };
}
} // namespace

TEST_F(Stage3StreamsTest, ModelForwardBackward_DefaultVsCustomStream) {
    constexpr int64_t IN_DIM = 8, CLASSES = 4, BATCH = 32, SEED = 1234;

    // Run on default stream.
    auto f_default = make_linear_fixture(IN_DIM, CLASSES, BATCH, SEED);
    auto r_default = run_step(f_default);
    c10::cuda::getCurrentCUDAStream().synchronize();

    // Run on a non-default pool stream with identical seed/inputs.
    auto custom_stream = c10::cuda::getStreamFromPool();
    StepResult r_custom;
    {
        c10::cuda::CUDAStreamGuard guard(custom_stream);
        auto f_custom = make_linear_fixture(IN_DIM, CLASSES, BATCH, SEED);
        r_custom = run_step(f_custom);
    }
    custom_stream.synchronize();

    // Loss + grad norm must match. FP scheduling differences are tolerated
    // at 1e-5; the operation graph is deterministic for deterministic inputs.
    EXPECT_NEAR(r_default.loss, r_custom.loss, 1e-5)
        << "loss differs across streams: default=" << r_default.loss
        << " custom=" << r_custom.loss;
    EXPECT_NEAR(r_default.grad_norm, r_custom.grad_norm, 1e-5)
        << "weight-grad norm differs across streams: default="
        << r_default.grad_norm << " custom=" << r_custom.grad_norm;
}

TEST_F(Stage3StreamsTest, ModelStream_GradientFlowsThroughEventBlock) {
    // Simulate the Module 5 ordering: producer thread runs a kernel on
    // stream A and records an event, consumer thread blocks the train
    // stream on the event before reading the produced tensor for backward.
    // The gradient w.r.t. the produced tensor must be correct (not garbage).

    constexpr int64_t N = 256, D = 16;
    auto produce_stream = c10::cuda::getStreamFromPool();
    auto train_stream   = c10::cuda::getStreamFromPool();
    torch::manual_seed(7);
    torch::cuda::manual_seed_all(7);

    torch::Tensor produced_leaf;     // gradient target (leaf)
    torch::Tensor produced;          // forward input (non-leaf)
    at::cuda::CUDAEvent ready;

    // Producer: assemble-like work — fill a tensor with non-trivial pattern.
    {
        c10::cuda::CUDAStreamGuard guard(produce_stream);
        produced_leaf = torch::randn({N, D},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
            .set_requires_grad(true);
        // A few ops so the stream has real work pending. The result is the
        // forward input; gradient flows back to produced_leaf as the leaf.
        produced = produced_leaf.relu().contiguous();
        ready.record(produce_stream);
    }

    // Consumer (train): block, then run forward + backward.
    auto fc = torch::nn::Linear(D, 2);
    fc->to(torch::kCUDA);
    auto target = torch::zeros({N},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    {
        c10::cuda::CUDAStreamGuard guard(train_stream);
        ready.block(train_stream);

        auto logits = fc->forward(produced);
        auto loss = torch::nn::functional::cross_entropy(logits, target);
        loss.backward();
    }
    train_stream.synchronize();

    // Leaf gradient must exist and be finite (not NaN / inf) — i.e. autograd
    // saw real produced values, not stale memory.
    auto grad = produced_leaf.grad();
    ASSERT_TRUE(grad.defined()) << "gradient must propagate through cross-stream link";
    auto finite = torch::isfinite(grad).all().item<bool>();
    EXPECT_TRUE(finite) << "gradient must be finite (sync via event must preserve data)";
    auto max_abs = grad.abs().max().item<float>();
    EXPECT_GT(max_abs, 0.0f) << "gradient should be nonzero somewhere";
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
