// Tests for ChunkPipeline<T>: a bounded SPSC queue used to overlap
// I/O with compute in create_reordered + L4 packed_slim worker loops.
//
// Each test exercises one invariant. Tests must remain green after Fix
// #21 ships; regressions here would silently break the L3/L4 throughput
// gain on papers100M-scale builds.

#include "gnn/common/pipeline_overlap.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using mdb::gnn::ChunkPipeline;

// 1. Round-trip a few items through a 2-slot pipeline.
TEST(ChunkPipeline, RoundTrip) {
    ChunkPipeline<int> p(2);
    std::thread prod([&] {
        for (int i = 0; i < 10; ++i) p.push(i);
        p.close();
    });
    std::vector<int> got;
    while (auto v = p.pop()) got.push_back(*v);
    prod.join();
    ASSERT_EQ(got.size(), 10u);
    for (int i = 0; i < 10; ++i) ASSERT_EQ(got[i], i);
}

// 2. Producer blocks when queue is full (backpressure).
TEST(ChunkPipeline, ProducerBackpressure) {
    ChunkPipeline<int> p(2);
    std::atomic<int> pushed{0};
    std::thread prod([&] {
        for (int i = 0; i < 5; ++i) {
            p.push(i);
            pushed.fetch_add(1);
        }
        p.close();
    });
    // After a brief sleep the producer should have filled 2 slots and
    // blocked on the third push.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_LE(pushed.load(), 2);
    // Drain: producer unblocks.
    while (p.pop()) {}
    prod.join();
    EXPECT_EQ(pushed.load(), 5);
}

// 3. Consumer returns nullopt on close() after all items drained.
TEST(ChunkPipeline, CloseDrainsThenReturnsNullopt) {
    ChunkPipeline<int> p(4);
    p.push(1); p.push(2); p.close();
    EXPECT_EQ(*p.pop(), 1);
    EXPECT_EQ(*p.pop(), 2);
    EXPECT_FALSE(p.pop().has_value());
}

// 4. Exception in producer propagates to consumer via set_error().
TEST(ChunkPipeline, ProducerErrorPropagates) {
    ChunkPipeline<int> p(2);
    std::thread prod([&] {
        p.push(1);
        try { throw std::runtime_error("boom"); }
        catch (...) { p.set_error(std::current_exception()); }
    });
    EXPECT_EQ(*p.pop(), 1);
    EXPECT_THROW(p.pop(), std::runtime_error);
    prod.join();
}

// 5. Single-slot queue (queue_size=1) still works end-to-end.
TEST(ChunkPipeline, SingleSlot) {
    ChunkPipeline<int> p(1);
    std::thread prod([&] {
        for (int i = 0; i < 5; ++i) p.push(i);
        p.close();
    });
    int sum = 0;
    while (auto v = p.pop()) sum += *v;
    prod.join();
    EXPECT_EQ(sum, 0 + 1 + 2 + 3 + 4);
}

// 6. Moves work (non-copyable payload).
TEST(ChunkPipeline, MoveOnlyPayload) {
    ChunkPipeline<std::unique_ptr<int>> p(2);
    std::thread prod([&] {
        p.push(std::make_unique<int>(42));
        p.close();
    });
    auto v = p.pop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(**v, 42);
    prod.join();
}

// 7. Pop after close with empty queue returns nullopt immediately.
TEST(ChunkPipeline, PopAfterEmptyClose) {
    ChunkPipeline<int> p(4);
    p.close();
    EXPECT_FALSE(p.pop().has_value());
}

// 8. Concurrent close + pop races safely (no UB, deterministic outcome).
TEST(ChunkPipeline, ConcurrentCloseAndPop) {
    ChunkPipeline<int> p(2);
    p.push(99);
    std::atomic<bool> got_value{false};
    std::thread consumer([&] {
        auto v = p.pop();
        if (v.has_value()) got_value.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    p.close();
    consumer.join();
    EXPECT_TRUE(got_value.load());  // Pre-close push must be visible.
}
