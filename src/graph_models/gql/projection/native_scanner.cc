#include "native_scanner.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef HAS_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
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
    // pre-Spec-#15 behavior exactly.
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
            // BPT leaf decode requires it (bplus_tree_leaf.cc:301,345);
            // without this, worker threads dereference a null
            // thread_local pointer.
            if (QueryContext::_query_ctx == nullptr && parent_ctx != nullptr) {
                QueryContext::set_query_ctx(parent_ctx);
            }
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

// Sequential helper: scan a single sub-range [lo_edge, hi_edge] of edge ids
// for the given type, resolving each edge's endpoints inline and pushing
// successful (edge_id, from, to) triples into `sink`. Used by both the
// sequential path and each parallel worker — endpoint lookups are pure
// reads on the secondary B+Tree, so they are safe to issue concurrently.
struct EdgeEndpointTriple {
    ObjectId edge_id;
    ObjectId from_node;
    ObjectId to_node;
};

static uint64_t scan_label_edge_with_endpoints_subrange(
    BPlusTree<2>* label_edge_index,
    BPlusTree<3>* edge_from_to_index,
    BPlusTree<3>* from_to_edge_index,
    BPlusTree<3>* edge_n1_n2_index,
    BPlusTree<3>* n1_n2_edge_index,
    uint64_t search_type_id,
    uint64_t lo_edge,
    uint64_t hi_edge,
    std::vector<EdgeEndpointTriple>& sink)
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
        sink.push_back(EdgeEndpointTriple{edge_id, from_node, to_node});
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
    // pre-Spec-#16 behavior exactly: collect into a single stack-local
    // vector and replay through the user callback.
    bool parallel_enabled = resolve_parallel_edge_scan_enabled();
    std::size_t num_partitions = resolve_edge_scan_partitions();

#ifndef HAS_TBB
    parallel_enabled = false;
#endif

    if (!parallel_enabled || num_partitions < 2) {
        std::vector<EdgeEndpointTriple> sink;
        sink.reserve(64);
        scan_label_edge_with_endpoints_subrange(
            label_edge_index,
            edge_from_to_index, from_to_edge_index,
            edge_n1_n2_index,   n1_n2_edge_index,
            search_type_id,
            /*lo_edge=*/0, /*hi_edge=*/UINT64_MAX,
            sink);
        for (const auto& t : sink) {
            callback(t.edge_id, t.from_node, t.to_node);
        }
        return sink.size();
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

    // ---- Range partition strategy (Option A: uniform split of edge-id range)
    //
    // The label_edge B+Tree is sorted lexicographically on (label_id, edge_id);
    // within a fixed label, records are ordered by the full 64-bit edge_id.
    // GQL edge ObjectIds carry a constant 8-bit type prefix in the high byte
    // (MASK_DIRECTED_EDGE = 0xE0'.., MASK_UNDIRECTED_EDGE = 0xE4'..) and a
    // monotonically increasing 56-bit counter in the low bytes — exactly the
    // same shape the node-id case uses, so a uniform split of [0, UINT64_MAX]
    // produces sub-ranges with roughly balanced record counts without paying
    // for a histogram prepass.
    auto ranges = build_uniform_subranges(num_partitions);

    // Phase 1: each worker writes into its own per-partition vector. No
    // shared mutable state across workers, so no locks. The B+Tree
    // BufferManager is thread-safe for concurrent reads, so concurrent
    // get_range iterators on the label_edge tree AND on the secondary
    // endpoint trees (edge_from_to / edge_n1_n2 / fallback scans) are safe.
    // Per-edge endpoint lookups stay sequential within each worker — the
    // win is from N partitions running concurrently, not from batching the
    // lookups themselves (deferred to a follow-up if profiling justifies it).
    std::vector<std::vector<EdgeEndpointTriple>> per_partition(num_partitions);
    for (auto& vec : per_partition) {
        vec.reserve(64);
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_partitions, 1),
        [&, parent_ctx](const tbb::blocked_range<std::size_t>& r) {
            // Inherit the parent thread's QueryContext into this worker.
            // BPT leaf decode requires it (bplus_tree_leaf.cc:301,345);
            // without this, worker threads dereference a null
            // thread_local pointer.
            if (QueryContext::_query_ctx == nullptr && parent_ctx != nullptr) {
                QueryContext::set_query_ctx(parent_ctx);
            }
            for (std::size_t p = r.begin(); p < r.end(); ++p) {
                scan_label_edge_with_endpoints_subrange(
                    label_edge_index,
                    edge_from_to_index, from_to_edge_index,
                    edge_n1_n2_index,   n1_n2_edge_index,
                    search_type_id,
                    ranges[p].first, ranges[p].second,
                    per_partition[p]);
            }
        });

    // Phase 2 (serial merge): replay collected triples through the user
    // callback in ascending partition order. The legacy sequential path
    // visits records in B+Tree key order, and our partitions are disjoint
    // sub-ranges of that key order, so the merged sequence is bit-identical
    // to the legacy ordering. Single-threaded replay also means the user
    // callback (which mutates ParallelEdgeDetector aggregation state in
    // native_projection_builder.cc) does NOT have to be thread-safe.
    uint64_t count = 0;
    for (auto& vec : per_partition) {
        for (const auto& t : vec) {
            callback(t.edge_id, t.from_node, t.to_node);
        }
        count += vec.size();
    }
    return count;
#else
    // Unreachable (parallel_enabled forced false above when !HAS_TBB).
    return 0;
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
