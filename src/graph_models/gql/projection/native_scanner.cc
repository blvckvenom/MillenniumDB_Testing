#include "native_scanner.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef HAS_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include "query/query_context.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"

namespace GQL {

namespace {

// Resolve a boolean env var. Returns the default if unset; "0"/"false"/"off"
// (any case) flip to false. Anything else is treated as true.
bool resolve_bool_env(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (env == nullptr) {
        return default_value;
    }
    std::string v(env);
    if (v == "0" || v == "false" || v == "off" ||
        v == "FALSE" || v == "OFF" || v == "False" || v == "Off")
    {
        return false;
    }
    return true;
}

// Resolve a partition-count env var. Default
// min(hardware_concurrency, 16). Clamped to [2, 64]. Values outside that
// range are silently clamped (the env var is an advisory tuning knob).
std::size_t resolve_partition_count_env(const char* name) {
    constexpr std::size_t kMin = 2;
    constexpr std::size_t kMaxCap = 64;
    constexpr std::size_t kDefaultCap = 16;

    std::size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 4;  // conservative fallback when sysconf is unhelpful
    }
    std::size_t k = std::min<std::size_t>(hw, kDefaultCap);

    if (const char* env = std::getenv(name)) {
        try {
            long parsed = std::stol(env);
            if (parsed > 0) {
                k = static_cast<std::size_t>(parsed);
            }
        } catch (...) {
            // Ignore malformed values, keep default.
        }
    }
    return std::clamp(k, kMin, kMaxCap);
}

bool resolve_parallel_node_scan_enabled() {
    return resolve_bool_env("MDB_PROJECTION_PARALLEL_NODE_SCAN", true);
}

std::size_t resolve_node_scan_partitions() {
    return resolve_partition_count_env("MDB_PROJECTION_NODE_SCAN_PARTITIONS");
}

bool resolve_parallel_edge_scan_enabled() {
    return resolve_bool_env("MDB_PROJECTION_PARALLEL_EDGE_SCAN", true);
}

std::size_t resolve_edge_scan_partitions() {
    return resolve_partition_count_env("MDB_PROJECTION_EDGE_SCAN_PARTITIONS");
}

// Build K disjoint sub-ranges that span [0, UINT64_MAX]. Sub-ranges are
// inclusive at both ends; we offset hi by -1 to make them disjoint, except
// the last partition which keeps UINT64_MAX so no record on the boundary
// is missed. Shared between scan_label_node and scan_label_edge_with_endpoints
// because both partition the secondary id space the same way.
std::vector<std::pair<uint64_t, uint64_t>>
build_uniform_subranges(std::size_t num_partitions) {
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.reserve(num_partitions);
    const uint64_t total_span = UINT64_MAX;  // inclusive size − 1
    const uint64_t step = total_span / num_partitions + 1;
    uint64_t lo = 0;
    for (std::size_t p = 0; p < num_partitions; ++p) {
        uint64_t hi;
        if (p + 1 == num_partitions) {
            hi = UINT64_MAX;
        } else {
            // Saturating add: avoid wraparound on the last interior partition.
            uint64_t next_lo = lo + step;
            hi = (next_lo == 0 || next_lo > UINT64_MAX) ? UINT64_MAX
                                                        : (next_lo - 1);
        }
        ranges.emplace_back(lo, hi);
        lo = (hi == UINT64_MAX) ? UINT64_MAX : (hi + 1);
    }
    return ranges;
}

} // namespace


NativeScanner::NativeScanner(
    BPlusTree<2>* label_node_idx,
    BPlusTree<2>* label_edge_idx,
    BPlusTree<3>* from_to_edge_idx,
    BPlusTree<3>* edge_from_to_idx,
    BPlusTree<3>* n1_n2_edge_idx,
    BPlusTree<3>* edge_n1_n2_idx
)
    : label_node_index(label_node_idx)
    , label_edge_index(label_edge_idx)
    , from_to_edge_index(from_to_edge_idx)
    , edge_from_to_index(edge_from_to_idx)
    , n1_n2_edge_index(n1_n2_edge_idx)
    , edge_n1_n2_index(edge_n1_n2_idx)
{
    if (!label_node_index || !label_edge_index || !from_to_edge_index || !n1_n2_edge_index) {
        throw std::runtime_error("NativeScanner: null index pointer provided");
    }
    // edge_from_to_index and edge_n1_n2_index are optional, can be nullptr
}

NativeScanner::~NativeScanner() {
    // Non-owning pointers, no cleanup needed
}

// Sequential helper: scan a single sub-range [lo, hi] of node ids for the
// given label, invoking `sink` for each node_id found in B+Tree key order.
// Pulled out so both the legacy fast path and each parallel worker share
// exactly the same iteration logic.
static uint64_t scan_label_node_subrange(
    BPlusTree<2>* label_node_index,
    uint64_t search_label_id,
    uint64_t lo_node,
    uint64_t hi_node,
    const std::function<void(ObjectId)>& sink)
{
    Record<2> min_record;
    min_record[0] = search_label_id;
    min_record[1] = lo_node;

    Record<2> max_record;
    max_record[0] = search_label_id;
    max_record[1] = hi_node;

    bool interruption_requested = false;
    auto iter = label_node_index->get_range(
        &interruption_requested, min_record, max_record);

    uint64_t count = 0;
    const Record<2>* record;
    while ((record = iter.next()) != nullptr) {
        ObjectId node_id((*record)[1]);
        sink(node_id);
        ++count;
    }
    return count;
}

uint64_t NativeScanner::scan_label_node(
    ObjectId label_id,
    std::function<void(ObjectId)> callback
) {
    // The label_node B+Tree stores full ObjectIds WITH type masks
    // We need to use the full label_id as-is
    const uint64_t search_label_id = label_id.id;

    // Sequential path — used when the parallel feature is disabled, when TBB
    // is not built in, or when only one partition is requested. Mirrors the
    // original single-threaded label_node scan behavior exactly.
    bool parallel_enabled = resolve_parallel_node_scan_enabled();
    std::size_t num_partitions = resolve_node_scan_partitions();

#ifndef HAS_TBB
    parallel_enabled = false;
#endif

    if (!parallel_enabled || num_partitions < 2) {
        return scan_label_node_subrange(
            label_node_index, search_label_id,
            /*lo_node=*/0, /*hi_node=*/UINT64_MAX,
            callback);
    }

#ifdef HAS_TBB
    // The QueryContext is held in a thread_local pointer
    // (QueryContext::_query_ctx) and is consulted on every BPT leaf decode
    // (bplus_tree_leaf.cc:301,345). TBB worker threads do not inherit the
    // main thread's thread_local state, so we must propagate it explicitly
    // before any worker calls get_range(). Capture the parent pointer here
    // and re-set on each worker's first iteration. Setting the same pointer
    // is idempotent and cheap; setting null on the parent thread is not
    // attempted (the variable carries the parent's context for the
    // duration of this call).
    QueryContext* parent_ctx = QueryContext::_query_ctx;

    // ---- Range partition strategy (Option A: uniform split of node-id range)
    //
    // The label_node B+Tree is sorted lexicographically on (label_id, node_id);
    // within a fixed label, records are ordered by the full 64-bit node_id.
    // GQL node ObjectIds carry a constant 8-bit type prefix in the high byte
    // (MASK_NODE = 0xD4'..) and a monotonically increasing 56-bit counter in
    // the low bytes, so a uniform split of [0, UINT64_MAX] produces sub-ranges
    // whose record counts are roughly proportional to the counter density —
    // good enough to balance worker load without paying for a histogram
    // prepass.
    auto ranges = build_uniform_subranges(num_partitions);

    // Phase 1: each worker writes into its own per-partition vector. No
    // shared mutable state across workers, so no locks. The B+Tree
    // BufferManager is thread-safe for concurrent reads (internal
    // vp_mutex / shared page latches), so concurrent get_range iterators
    // on the same tree are safe.
    std::vector<std::vector<ObjectId>> per_partition(num_partitions);
    // Reservation hint: assume roughly uniform distribution. 64 entries
    // is enough to cover small labels without over-allocating; the vector
    // grows naturally for larger ranges.
    for (auto& vec : per_partition) {
        vec.reserve(64);
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_partitions, 1),
        [&, parent_ctx](const tbb::blocked_range<std::size_t>& r) {
            // Inherit the parent thread's QueryContext into this worker.
            // BPT leaf decode requires it (bplus_tree_leaf.cc:301,345).
            // Set unconditionally: TBB pool threads persist across queries,
            // so a non-null slot may hold the context of a PREVIOUS query
            // (stale, possibly destroyed) and must be overwritten.
            QueryContext::set_query_ctx(parent_ctx);
            for (std::size_t p = r.begin(); p < r.end(); ++p) {
                auto& sink = per_partition[p];
                scan_label_node_subrange(
                    label_node_index, search_label_id,
                    ranges[p].first, ranges[p].second,
                    [&sink](ObjectId node_id) {
                        sink.push_back(node_id);
                    });
            }
        });

    // Phase 2 (serial merge): replay the collected node ids through the
    // user-supplied callback in ascending partition order. The legacy
    // sequential path visits records in B+Tree key order, and our
    // partitions are disjoint sub-ranges of that key order, so the merged
    // sequence is bit-identical to the legacy ordering. Single-threaded
    // replay also means the user callback (which reaches into shared
    // builder/storage state) does NOT have to be thread-safe — preserving
    // the legacy contract.
    uint64_t count = 0;
    for (auto& vec : per_partition) {
        for (ObjectId node_id : vec) {
            callback(node_id);
        }
        count += vec.size();
    }
    return count;
#else
    // Unreachable (parallel_enabled forced false above when !HAS_TBB).
    return 0;
#endif
}

uint64_t NativeScanner::scan_label_edge(
    ObjectId type_id,
    std::function<void(ObjectId)> callback
) {
    // The label_edge B+Tree stores full ObjectIds WITH type masks
    uint64_t search_type_id = type_id.id;

    // Define range: all records where first key = search_type_id
    Record<2> min_record;
    min_record[0] = search_type_id;
    min_record[1] = 0;

    Record<2> max_record;
    max_record[0] = search_type_id;
    max_record[1] = UINT64_MAX;

    // Create range iterator with interruption support
    bool interruption_requested = false;
    auto iter = label_edge_index->get_range(&interruption_requested, min_record, max_record);

    // Iterate over matching records
    uint64_t count = 0;
    const Record<2>* record;
    while ((record = iter.next()) != nullptr) {
        // Record format: {type_id, edge_id}
        // Extract edge_id from second field
        ObjectId edge_id((*record)[1]);
        callback(edge_id);
        count++;
    }

    return count;
}

// Resolve the (from, to) endpoints for a single edge using whichever
// secondary index is available. Returns true on success and writes the
// endpoints to `out_from` / `out_to`; returns false if the edge cannot
// be found in either index. Used both by the legacy sequential path and
// by the parallel workers — the only state it touches is read-only B+Tree
// reads, so it is safe to call concurrently from TBB workers.
static bool resolve_edge_endpoints(
    BPlusTree<3>* edge_from_to_index,
    BPlusTree<3>* from_to_edge_index,
    BPlusTree<3>* edge_n1_n2_index,
    BPlusTree<3>* n1_n2_edge_index,
    ObjectId edge_id,
    ObjectId& out_from,
    ObjectId& out_to)
{
    const uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
    const bool is_undirected = (edge_type == ObjectId::MASK_UNDIRECTED_EDGE);

    if (is_undirected) {
        // ===== UNDIRECTED EDGE (0xe4) =====
        if (edge_n1_n2_index) {
            // Fast path: O(log n) lookup in {edge_id, n1, n2}
            Record<3> min_rec;
            min_rec[0] = edge_id.id;
            min_rec[1] = 0;
            min_rec[2] = 0;

            Record<3> max_rec;
            max_rec[0] = edge_id.id;
            max_rec[1] = UINT64_MAX;
            max_rec[2] = UINT64_MAX;

            bool interrupt = false;
            auto endpoint_iter = edge_n1_n2_index->get_range(&interrupt, min_rec, max_rec);

            const Record<3>* endpoint_rec = endpoint_iter.next();
            if (endpoint_rec != nullptr) {
                out_from = ObjectId((*endpoint_rec)[1]);
                out_to   = ObjectId((*endpoint_rec)[2]);
                return true;
            }
            return false;
        }

        // Slow path: scan n1_n2_edge linearly (should not happen in practice).
        Record<3> min_rec;
        min_rec[0] = 0;
        min_rec[1] = 0;
        min_rec[2] = 0;

        Record<3> max_rec;
        max_rec[0] = UINT64_MAX;
        max_rec[1] = UINT64_MAX;
        max_rec[2] = UINT64_MAX;

        bool interrupt = false;
        auto endpoint_iter = n1_n2_edge_index->get_range(&interrupt, min_rec, max_rec);

        const Record<3>* endpoint_rec;
        while ((endpoint_rec = endpoint_iter.next()) != nullptr) {
            if ((*endpoint_rec)[2] == edge_id.id) {
                out_from = ObjectId((*endpoint_rec)[0]);
                out_to   = ObjectId((*endpoint_rec)[1]);
                return true;
            }
        }
        return false;
    }

    // ===== DIRECTED EDGE (0xe0) =====
    if (edge_from_to_index) {
        // Fast path: O(log n) lookup in {edge_id, from, to}
        Record<3> min_rec;
        min_rec[0] = edge_id.id;
        min_rec[1] = 0;
        min_rec[2] = 0;

        Record<3> max_rec;
        max_rec[0] = edge_id.id;
        max_rec[1] = UINT64_MAX;
        max_rec[2] = UINT64_MAX;

        bool interrupt = false;
        auto endpoint_iter = edge_from_to_index->get_range(&interrupt, min_rec, max_rec);

        const Record<3>* endpoint_rec = endpoint_iter.next();
        if (endpoint_rec != nullptr) {
            out_from = ObjectId((*endpoint_rec)[1]);
            out_to   = ObjectId((*endpoint_rec)[2]);
            return true;
        }
        return false;
    }

    // Slow path: scan from_to_edge linearly (should not happen in practice).
    Record<3> min_rec;
    min_rec[0] = 0;
    min_rec[1] = 0;
    min_rec[2] = 0;

    Record<3> max_rec;
    max_rec[0] = UINT64_MAX;
    max_rec[1] = UINT64_MAX;
    max_rec[2] = UINT64_MAX;

    bool interrupt = false;
    auto endpoint_iter = from_to_edge_index->get_range(&interrupt, min_rec, max_rec);

    const Record<3>* endpoint_rec;
    while ((endpoint_rec = endpoint_iter.next()) != nullptr) {
        if ((*endpoint_rec)[2] == edge_id.id) {
            out_from = ObjectId((*endpoint_rec)[0]);
            out_to   = ObjectId((*endpoint_rec)[1]);
            return true;
        }
    }
    return false;
}

// Triple type used by the parallel-branch streaming queues.
struct EdgeEndpointTriple {
    ObjectId edge_id;
    ObjectId from_node;
    ObjectId to_node;
};

// Callback variant of the sub-range scan: invokes `cb(edge_id, from, to)`
// for each successfully-resolved edge in [lo_edge, hi_edge]. `cb` returns
// true to continue and false to stop the scan early — the parallel workers
// use this to wind down promptly when the consumer or a peer worker has
// failed. Used by each parallel worker (callback pushes to a bounded
// queue) and could equally be used from the sequential path (in this file
// the sequential path inlines the same loop directly because its callback
// is the user's).
template <typename Callback>
static uint64_t scan_label_edge_with_endpoints_subrange_cb(
    BPlusTree<2>* label_edge_index,
    BPlusTree<3>* edge_from_to_index,
    BPlusTree<3>* from_to_edge_index,
    BPlusTree<3>* edge_n1_n2_index,
    BPlusTree<3>* n1_n2_edge_index,
    uint64_t search_type_id,
    uint64_t lo_edge,
    uint64_t hi_edge,
    Callback cb)
{
    Record<2> min_record;
    min_record[0] = search_type_id;
    min_record[1] = lo_edge;

    Record<2> max_record;
    max_record[0] = search_type_id;
    max_record[1] = hi_edge;

    bool interruption_requested = false;
    auto iter = label_edge_index->get_range(
        &interruption_requested, min_record, max_record);

    uint64_t count = 0;
    const Record<2>* record;
    while ((record = iter.next()) != nullptr) {
        ObjectId edge_id((*record)[1]);
        ObjectId from_node, to_node;
        if (!resolve_edge_endpoints(
                edge_from_to_index, from_to_edge_index,
                edge_n1_n2_index, n1_n2_edge_index,
                edge_id, from_node, to_node))
        {
            // Edge not found in endpoint index — skip, matching legacy semantics.
            continue;
        }
        if (!cb(edge_id, from_node, to_node)) {
            break;
        }
        ++count;
    }
    return count;
}

uint64_t NativeScanner::scan_label_edge_with_endpoints(
    ObjectId type_id,
    std::function<void(ObjectId, ObjectId, ObjectId)> callback
) {
    const uint64_t search_type_id = type_id.id;

    // Sequential path — used when the parallel feature is disabled, when TBB
    // is not built in, or when only one partition is requested. Mirrors the
    // original single-threaded label_edge_with_endpoints scan behavior: each
    // edge is emitted directly to the user callback as it is scanned.
    bool parallel_enabled = resolve_parallel_edge_scan_enabled();
    std::size_t num_partitions = resolve_edge_scan_partitions();

#ifndef HAS_TBB
    parallel_enabled = false;
#endif

    if (!parallel_enabled || num_partitions < 2) {
        // Inline callback path — never accumulates into a vector. Each edge is
        // emitted to the user callback as soon as its endpoints are resolved.
        // Required for large datasets (papers100M: 1.6B edges × 24 B/triple =
        // 36 GB if accumulated) where a collect-then-replay approach would
        // exceed available RAM.
        Record<2> min_record;
        min_record[0] = search_type_id;
        min_record[1] = 0;

        Record<2> max_record;
        max_record[0] = search_type_id;
        max_record[1] = UINT64_MAX;

        bool interruption_requested = false;
        auto iter = label_edge_index->get_range(
            &interruption_requested, min_record, max_record);

        uint64_t count = 0;
        const Record<2>* record;
        while ((record = iter.next()) != nullptr) {
            ObjectId edge_id((*record)[1]);
            ObjectId from_node, to_node;
            if (!resolve_edge_endpoints(
                    edge_from_to_index, from_to_edge_index,
                    edge_n1_n2_index, n1_n2_edge_index,
                    edge_id, from_node, to_node))
            {
                continue;
            }
            callback(edge_id, from_node, to_node);
            ++count;
        }
        return count;
    }

#ifdef HAS_TBB
    // The QueryContext is held in a thread_local pointer
    // (QueryContext::_query_ctx) and is consulted on every BPT leaf decode
    // (bplus_tree_leaf.cc:301,345). TBB worker threads do not inherit the
    // main thread's thread_local state, so we must propagate it explicitly
    // before any worker calls get_range(). Capture the parent pointer here
    // and re-set on each worker's first iteration. Setting the same pointer
    // is idempotent and cheap.
    QueryContext* parent_ctx = QueryContext::_query_ctx;

    // Range partition strategy: uniform split of the full 64-bit edge-id space
    // into num_partitions disjoint sub-ranges (same approach as the node scan).
    auto ranges = build_uniform_subranges(num_partitions);

    // ---- Streaming producer-consumer with bounded per-partition queues.
    //
    // A naive collect-then-replay design would accumulate all per-partition
    // triples in std::vector<std::vector<EdgeEndpointTriple>> before invoking
    // the user callback. For papers100M (1.6 B edges × 24 B/triple = 36 GB)
    // this exceeds 32 GB host RAM and OOMs.
    //
    // Streaming design: each partition has a bounded queue (kQueueMax slots).
    // Workers push triples; if the queue is full they block on cv_not_full.
    // The main thread consumes queues in ascending partition order so the
    // callback sees the bit-identical sequence that the legacy single-
    // threaded scan would emit. Consumer wakes the worker via cv_not_full
    // after each drain. Worker batching (kFlushBatch triples per acquire)
    // amortizes mutex cost — a 64-triple batch = 64× fewer locks than per-
    // edge synchronization.
    //
    // Memory bound: num_partitions × (kQueueMax + kFlushBatch) × 24 B.
    // With 16 partitions × 1024 + 64 = ~26 KB/partition × 16 = ~420 KB.
    constexpr std::size_t kQueueMax   = 1024;
    constexpr std::size_t kFlushBatch = 64;

    struct EdgeQueue {
        std::mutex              mu;
        std::condition_variable cv_not_empty;
        std::condition_variable cv_not_full;
        std::vector<EdgeEndpointTriple> buffer;
        bool finished = false;
    };

    // unique_ptr because std::mutex / std::condition_variable are not
    // movable — std::vector<EdgeQueue> would not compile.
    std::vector<std::unique_ptr<EdgeQueue>> queues(num_partitions);
    for (std::size_t i = 0; i < num_partitions; ++i) {
        queues[i] = std::make_unique<EdgeQueue>();
        queues[i]->buffer.reserve(kQueueMax);
    }

    // ---- Failure propagation.
    //
    // Any exception — from the user callback on the consumer side (e.g. the
    // documented duplicate-edge QueryException thrown by
    // ParallelEdgeDetector::process_edge in native_projection_builder.cc) or
    // from a worker's B+Tree reads — must unwind out of THIS frame, never
    // out of a worker/producer thread: an exception escaping a std::thread
    // body, or unwinding past a joinable std::thread, calls std::terminate()
    // and aborts the whole server. First exception wins under
    // exception_mutex; abort_requested makes every backpressure / drain wait
    // re-check and wind down; the consumer rethrows after the producer is
    // joined.
    std::atomic<bool> abort_requested{false};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;

    // Wake every thread blocked on a queue condition variable. The empty
    // critical section on each queue mutex pairs with the predicate
    // re-checks below: a waiter either observed abort_requested before
    // blocking or is fully blocked when the notify fires — no missed wakeup.
    auto request_abort = [&]() {
        abort_requested.store(true, std::memory_order_relaxed);
        for (auto& qp : queues) {
            { std::lock_guard<std::mutex> lk(qp->mu); }
            qp->cv_not_empty.notify_all();
            qp->cv_not_full.notify_all();
        }
    };

    auto record_first_exception = [&]() {
        {
            std::lock_guard<std::mutex> lk(exception_mutex);
            if (!first_exception) {
                first_exception = std::current_exception();
            }
        }
        request_abort();
    };

    // Producer thread: launches all workers via TBB parallel_for. Runs in
    // its own std::thread so the main thread is free to consume in parallel.
    std::thread producer([&, parent_ctx]() {
        try {
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, num_partitions, 1),
                [&, parent_ctx](const tbb::blocked_range<std::size_t>& r) {
                    // Inherit the parent thread's QueryContext into this
                    // worker. BPT leaf decode requires it
                    // (bplus_tree_leaf.cc:301,345). Set unconditionally: TBB
                    // pool threads persist across queries, so a non-null
                    // slot may hold the context of a PREVIOUS query (stale,
                    // possibly destroyed) and must be overwritten.
                    QueryContext::set_query_ctx(parent_ctx);
                    for (std::size_t p = r.begin(); p < r.end(); ++p) {
                        EdgeQueue& q = *queues[p];

                        if (!abort_requested.load(std::memory_order_relaxed)) {
                            try {
                                // Worker-local batch — accumulated then
                                // flushed under one lock acquire. Cuts mutex
                                // traffic by kFlushBatch×.
                                std::vector<EdgeEndpointTriple> local_batch;
                                local_batch.reserve(kFlushBatch);

                                auto flush_batch = [&]() {
                                    if (local_batch.empty()) return;
                                    {
                                        std::unique_lock<std::mutex> lk(q.mu);
                                        q.cv_not_full.wait(lk, [&] {
                                            return q.buffer.size() < kQueueMax ||
                                                   abort_requested.load(
                                                       std::memory_order_relaxed);
                                        });
                                        if (abort_requested.load(
                                                std::memory_order_relaxed)) {
                                            // Consumer is gone — drop the
                                            // batch and let the scan stop.
                                            local_batch.clear();
                                            return;
                                        }
                                        for (auto& t : local_batch) {
                                            q.buffer.push_back(t);
                                        }
                                    }
                                    q.cv_not_empty.notify_one();
                                    local_batch.clear();
                                };

                                scan_label_edge_with_endpoints_subrange_cb(
                                    label_edge_index,
                                    edge_from_to_index, from_to_edge_index,
                                    edge_n1_n2_index,   n1_n2_edge_index,
                                    search_type_id,
                                    ranges[p].first, ranges[p].second,
                                    [&](ObjectId eid, ObjectId fn, ObjectId tn) {
                                        if (abort_requested.load(
                                                std::memory_order_relaxed)) {
                                            return false;  // stop the scan
                                        }
                                        local_batch.push_back(
                                            EdgeEndpointTriple{eid, fn, tn});
                                        if (local_batch.size() >= kFlushBatch) {
                                            flush_batch();
                                        }
                                        return true;
                                    });

                                // Drain remainder.
                                flush_batch();
                            } catch (...) {
                                record_first_exception();
                            }
                        }

                        // Signal finished — even on abort — so the consumer
                        // can always advance past this partition.
                        {
                            std::lock_guard<std::mutex> lk(q.mu);
                            q.finished = true;
                        }
                        q.cv_not_empty.notify_all();
                    }
                });
        } catch (...) {
            // parallel_for itself failed (scheduler error) — workers swallow
            // their own exceptions above, so nothing user-level lands here.
            record_first_exception();
        }
        // Safety net: if parallel_for did not run every iteration, mark all
        // queues finished so the consumer can never block forever.
        for (auto& qp : queues) {
            {
                std::lock_guard<std::mutex> lk(qp->mu);
                qp->finished = true;
            }
            qp->cv_not_empty.notify_all();
        }
    });

    // Backstop: the producer must be joined on EVERY exit path — destroying
    // a joinable std::thread calls std::terminate(). request_abort() first
    // so workers blocked on backpressure (cv_not_full) wake up and wind
    // down instead of deadlocking the join. On the normal path the explicit
    // join below runs first and this dtor is a no-op.
    struct ProducerJoinGuard {
        std::thread& thread;
        decltype(request_abort)& abort;
        ~ProducerJoinGuard() {
            if (thread.joinable()) {
                abort();
                thread.join();
            }
        }
    };
    ProducerJoinGuard producer_guard{producer, request_abort};

    // Consumer (main thread): drain each partition queue in order. Each
    // drain swaps the queue buffer out (so workers can keep producing) and
    // then invokes the user callback outside the lock — preserves the
    // single-callback-thread guarantee that ParallelEdgeDetector relies on
    // (native_projection_builder.cc).
    uint64_t count = 0;
    std::vector<EdgeEndpointTriple> drained;
    drained.reserve(kQueueMax);
    try {
        for (std::size_t p = 0; p < num_partitions; ++p) {
            EdgeQueue& q = *queues[p];
            while (!abort_requested.load(std::memory_order_relaxed)) {
                {
                    std::unique_lock<std::mutex> lk(q.mu);
                    q.cv_not_empty.wait(lk, [&] {
                        return !q.buffer.empty() || q.finished ||
                               abort_requested.load(std::memory_order_relaxed);
                    });
                    if (abort_requested.load(std::memory_order_relaxed)) {
                        break;
                    }
                    if (q.buffer.empty() && q.finished) {
                        break;
                    }
                    drained.swap(q.buffer);
                    q.buffer.reserve(kQueueMax);
                }
                q.cv_not_full.notify_all();
                for (const auto& t : drained) {
                    callback(t.edge_id, t.from_node, t.to_node);
                    ++count;
                }
                drained.clear();
            }
            if (abort_requested.load(std::memory_order_relaxed)) {
                break;
            }
        }
    } catch (...) {
        // User callback threw (e.g. the duplicate-edge QueryException).
        // Record + abort, then fall through to the join + rethrow below so
        // the exception reaches the caller AFTER all threads are stopped.
        record_first_exception();
    }

    producer.join();

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    return count;
#else
    // Unreachable (parallel_enabled forced false above when !HAS_TBB).
    return 0;
#endif
}

std::size_t NativeScanner::scan_label_edge_endpoints_partitioned(
    ObjectId type_id,
    std::size_t num_partitions,
    const std::function<void(std::size_t, ObjectId, ObjectId, ObjectId)>& per_edge_cb
) {
    const uint64_t search_type_id = type_id.id;

    // Resolve the effective partition count. 0 = auto via the env knob,
    // otherwise honor the caller's request (clamped to the same [2, 64]
    // advisory band used everywhere else). A request of 1 (or TBB absent)
    // collapses to the sequential path.
    std::size_t k = (num_partitions == 0)
                        ? resolve_edge_scan_partitions()
                        : std::clamp<std::size_t>(num_partitions, 1, 64);

#ifndef HAS_TBB
    k = 1;
#endif

    // ---- Sequential fallback: single partition on the calling thread.
    if (k < 2) {
        auto ranges = build_uniform_subranges(1);
        scan_label_edge_with_endpoints_subrange_cb(
            label_edge_index,
            edge_from_to_index, from_to_edge_index,
            edge_n1_n2_index,   n1_n2_edge_index,
            search_type_id,
            ranges[0].first, ranges[0].second,
            [&](ObjectId eid, ObjectId fn, ObjectId tn) {
                per_edge_cb(0, eid, fn, tn);
                return true;
            });
        return 1;
    }

#ifdef HAS_TBB
    // TBB worker threads do not inherit the main thread's thread_local
    // QueryContext (consulted on every BPT leaf decode); propagate it.
    QueryContext* parent_ctx = QueryContext::_query_ctx;

    auto ranges = build_uniform_subranges(k);

    // First-wins exception capture. An exception escaping a TBB worker would
    // unwind out of the worker body and call std::terminate(); instead each
    // worker swallows its own exception, records it under the mutex, and
    // requests the remaining work to wind down. Rethrown on the main thread
    // after parallel_for returns.
    std::atomic<bool> abort_requested{false};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;

    auto record_first_exception = [&]() {
        std::lock_guard<std::mutex> lk(exception_mutex);
        if (!first_exception) {
            first_exception = std::current_exception();
        }
        abort_requested.store(true, std::memory_order_relaxed);
    };

    // Bound the worker pool to hardware_concurrency, mirroring the producer's
    // existing arena discipline elsewhere in the projection build.
    std::size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    tbb::task_arena arena(static_cast<int>(std::min<std::size_t>(hw, k)));

    arena.execute([&]() {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, k, 1),
            [&, parent_ctx](const tbb::blocked_range<std::size_t>& r) {
                // Inherit the parent's QueryContext (TBB pool threads persist
                // across queries; overwrite unconditionally).
                QueryContext::set_query_ctx(parent_ctx);
                for (std::size_t p = r.begin(); p < r.end(); ++p) {
                    if (abort_requested.load(std::memory_order_relaxed)) {
                        continue;
                    }
                    try {
                        scan_label_edge_with_endpoints_subrange_cb(
                            label_edge_index,
                            edge_from_to_index, from_to_edge_index,
                            edge_n1_n2_index,   n1_n2_edge_index,
                            search_type_id,
                            ranges[p].first, ranges[p].second,
                            [&](ObjectId eid, ObjectId fn, ObjectId tn) {
                                if (abort_requested.load(
                                        std::memory_order_relaxed)) {
                                    return false;  // wind down promptly
                                }
                                // Invoked IN the worker thread — the callback
                                // writes only to partition p's private slot,
                                // so no synchronization is needed.
                                per_edge_cb(p, eid, fn, tn);
                                return true;
                            });
                    } catch (...) {
                        record_first_exception();
                    }
                }
            });
    });

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    return k;
#else
    return 1;  // Unreachable (k forced to 1 above when !HAS_TBB).
#endif
}

std::pair<ObjectId, ObjectId> NativeScanner::get_edge_endpoints(ObjectId edge_id) {
    // Detect edge type by examining the ObjectId mask
    uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
    bool is_undirected = (edge_type == ObjectId::MASK_UNDIRECTED_EDGE);

    if (is_undirected) {
        // ===== UNDIRECTED EDGE (0xe4) =====
        // Fast path: Use edge_n1_n2 index if available (O(log n) lookup)
        // Index structure: {edge_id, n1, n2}
        if (edge_n1_n2_index) {

            Record<3> min_record;
            min_record[0] = edge_id.id;
            min_record[1] = 0;
            min_record[2] = 0;

            Record<3> max_record;
            max_record[0] = edge_id.id;
            max_record[1] = UINT64_MAX;
            max_record[2] = UINT64_MAX;

            bool interruption_requested = false;
            auto iter = edge_n1_n2_index->get_range(&interruption_requested, min_record, max_record);

            const Record<3>* record = iter.next();
            if (record != nullptr) {
                // Record format: {edge_id, n1, n2}
                ObjectId n1((*record)[1]);
                ObjectId n2((*record)[2]);
                return {n1, n2};
            }
            // Fall through to slow path if not found
        }

        // Slow path: Scan n1_n2_edge index (O(E) worst case)
        // Index structure: {n1, n2, edge_id}
        Record<3> min_record;
        min_record[0] = 0;
        min_record[1] = 0;
        min_record[2] = 0;

        Record<3> max_record;
        max_record[0] = UINT64_MAX;
        max_record[1] = UINT64_MAX;
        max_record[2] = UINT64_MAX;

        bool interruption_requested = false;
        auto iter = n1_n2_edge_index->get_range(&interruption_requested, min_record, max_record);

        // Linear scan to find edge
        const Record<3>* record;
        while ((record = iter.next()) != nullptr) {
            if ((*record)[2] == edge_id.id) {
                // Found it! Extract endpoints
                ObjectId n1((*record)[0]);
                ObjectId n2((*record)[1]);
                return {n1, n2};
            }
        }

        throw std::runtime_error(
            "NativeScanner::get_edge_endpoints: Undirected edge not found: " + std::to_string(edge_id.id)
        );

    } else {
        // ===== DIRECTED EDGE (0xe0) =====
        // Fast path: Use edge_from_to index if available (O(log n) lookup)
        // Index structure: {edge_id, from, to}
        if (edge_from_to_index) {

            Record<3> min_record;
            min_record[0] = edge_id.id;
            min_record[1] = 0;
            min_record[2] = 0;

            Record<3> max_record;
            max_record[0] = edge_id.id;
            max_record[1] = UINT64_MAX;
            max_record[2] = UINT64_MAX;

            bool interruption_requested = false;
            auto iter = edge_from_to_index->get_range(&interruption_requested, min_record, max_record);

            const Record<3>* record = iter.next();
            if (record != nullptr) {
                // Record format: {edge_id, from, to}
                ObjectId from_node((*record)[1]);
                ObjectId to_node((*record)[2]);
                return {from_node, to_node};
            }
            // Fall through to slow path if not found
        }

        // Slow path: Scan from_to_edge index (O(E) worst case)
        // Index structure: {from, to, edge_id}
        Record<3> min_record;
        min_record[0] = 0;
        min_record[1] = 0;
        min_record[2] = 0;

        Record<3> max_record;
        max_record[0] = UINT64_MAX;
        max_record[1] = UINT64_MAX;
        max_record[2] = UINT64_MAX;

        bool interruption_requested = false;
        auto iter = from_to_edge_index->get_range(&interruption_requested, min_record, max_record);

        // Linear scan to find edge
        const Record<3>* record;
        while ((record = iter.next()) != nullptr) {
            if ((*record)[2] == edge_id.id) {
                // Found it! Extract endpoints
                ObjectId from_node((*record)[0]);
                ObjectId to_node((*record)[1]);
                return {from_node, to_node};
            }
        }

        throw std::runtime_error(
            "NativeScanner::get_edge_endpoints: Directed edge not found: " + std::to_string(edge_id.id)
        );
    }
}

uint64_t NativeScanner::count_edges_by_type(ObjectId type_id) {
    // Quick count of edges with this type by scanning label_edge index
    uint64_t search_type_id = type_id.id;

    // Define range: all records where first key = search_type_id
    Record<2> min_record;
    min_record[0] = search_type_id;
    min_record[1] = 0;

    Record<2> max_record;
    max_record[0] = search_type_id;
    max_record[1] = UINT64_MAX;

    // Create range iterator with interruption support
    bool interruption_requested = false;
    auto iter = label_edge_index->get_range(&interruption_requested, min_record, max_record);

    // Count matching records
    uint64_t count = 0;
    while (iter.next() != nullptr) {
        count++;
    }

    return count;
}

} // namespace GQL
