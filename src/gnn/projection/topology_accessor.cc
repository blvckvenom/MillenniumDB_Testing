#include "topology_accessor.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "gnn/projection/four_level_topology_store.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "storage/index/bplus_tree/bplus_tree.h"

#include "gnn/core/cuda_context.h"

namespace mdb::gnn {

// ============================================================================
// TopologyAccessor Implementation
// ============================================================================

struct TopologyAccessor::Impl {
    GQL::ProjectionStorage& storage;
    torch::Device target_device;
    std::mt19937_64 rng;

    // Topology CSR sidecar readers for O(1) neighbor slice via mmap. Each
    // reader maps topology_{fwd,rev}.csr and holds its own mmap + fd state;
    // `has_data()` reports whether the fast path is usable for this direction.
    // Absent / stale / malformed sidecars leave the reader inert, in which
    // case every accessor silently falls back to the B+Tree path below.
    GQL::Projection::TopologySnapshotReader fwd_csr_;
    GQL::Projection::TopologySnapshotReader rev_csr_;

    // ----- In-memory adjacency cache -----
    //
    // Populated by a single full-scan of the from_to_edge / to_from_edge
    // B+Tree indexes into an unordered_map<src, vector<AdjEntry>>. Once built,
    // get_out_neighbors / get_in_neighbors consult these maps first instead of
    // issuing a B+Tree range query, turning O(log N) lookups into O(1) hash
    // lookups. When `cache_enabled_` is true and the corresponding direction map
    // is populated, `get_out_neighbors` / `get_in_neighbors` consult these maps
    // first. Maps are keyed by source-side raw uint64 node id (matches B+Tree
    // record convention) and hold a contiguous vector of `(neighbor_id,
    // edge_id)` pairs.
    //
    // Always-empty sentinel returned for absent keys keeps `get_neighbors_*`
    // const-correct without per-call allocation. Built atomically once per
    // direction; further `prebuild_adjacency_cache` calls are idempotent.
    bool cache_enabled_ = false;
    std::unordered_map<uint64_t, std::vector<TopologyAccessor::AdjEntry>> fwd_cache_;
    std::unordered_map<uint64_t, std::vector<TopologyAccessor::AdjEntry>> rev_cache_;
    bool fwd_cache_built_ = false;
    bool rev_cache_built_ = false;
    uint64_t fwd_cache_entries_ = 0;
    uint64_t rev_cache_entries_ = 0;

    // ----- Four-Level Topology Store -----
    //
    // Frequency-tiered neighbor store (L1 RAM hash / L2 compact uint32 CSR /
    // L3 mmap sidecar / L4 direct B+Tree). Owned by the accessor when
    // enable_four_level_store() has been called. Null otherwise — every
    // existing dispatch path (in-memory adjacency cache, topology CSR sidecar
    // mmap, B+Tree direct) is preserved byte-for-byte for backwards-compat.
    std::unique_ptr<FourLevelTopologyStore> four_level_store_;

    explicit Impl(GQL::ProjectionStorage& storage_)
        : storage(storage_),
          target_device(CudaContext::instance().torch_device()),
          rng(std::random_device{}()),
          fwd_csr_(GQL::Projection::TopologySnapshotReader::open(
              std::filesystem::path(storage_.get_projection_dir()),
              GQL::Projection::TopologySnapshotReader::Direction::FORWARD)),
          rev_csr_(GQL::Projection::TopologySnapshotReader::open(
              std::filesystem::path(storage_.get_projection_dir()),
              GQL::Projection::TopologySnapshotReader::Direction::REVERSE)) {
        std::cerr << "TopologyAccessor: fwd="
                  << (fwd_csr_.has_data() ? "csr" : "bpt")
                  << " rev="
                  << (rev_csr_.has_data() ? "csr" : "bpt") << "\n";
    }

    // -------------------------------------------------------------------------
    // Adjacency cache helpers — full B+Tree scan into unordered_map
    // -------------------------------------------------------------------------

    // Full-scan a single B+Tree edge index and merge every record into
    // `target` keyed on record[0] (the source-side endpoint). Returns the
    // number of directed entries appended.
    //
    // Mirrors the EmbeddingWriter Phase B scan (commit 6521cc21) but tied to
    // the accessor lifetime so multiple downstream consumers can share one
    // build. Reads the same `from_to_edge` / `to_from_edge` indexes the
    // BPT path uses, so cache contents are bit-identical to the live tree
    // contents at construction time.
    uint64_t scan_into_(
        BPlusTree<3>* index,
        std::unordered_map<uint64_t, std::vector<TopologyAccessor::AdjEntry>>& target)
    {
        uint64_t appended = 0;
        if (!index) {
            return appended;
        }
        Record<3> min_record = {0, 0, 0};
        Record<3> max_record = {UINT64_MAX, UINT64_MAX, UINT64_MAX};
        bool interruption_requested = false;
        auto it = index->get_range(&interruption_requested, min_record, max_record);
        const Record<3>* rec;
        while ((rec = it.next()) != nullptr) {
            const uint64_t a = std::get<0>(*rec);
            const uint64_t b = std::get<1>(*rec);
            const uint64_t e = std::get<2>(*rec);
            target[a].push_back(TopologyAccessor::AdjEntry{b, e});
            ++appended;
        }
        return appended;
    }

    // Build the forward (`from_to_edge`) cache once. Idempotent.
    void build_fwd_cache_() {
        if (fwd_cache_built_) {
            return;
        }
        const auto t_start = std::chrono::steady_clock::now();
        fwd_cache_.reserve(storage.get_node_count());
        auto* fwd_index = storage.get_from_to_edge_index();
        fwd_cache_entries_ = scan_into_(fwd_index, fwd_cache_);
        fwd_cache_built_ = true;
        const auto t_end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(
            t_end - t_start).count();
        std::cerr << "TopologyAccessor: fwd adjacency cache built — "
                  << fwd_cache_.size() << " keys, "
                  << fwd_cache_entries_ << " directed entries, "
                  << static_cast<int>(ms) << " ms" << std::endl;
    }

    // Build the reverse (`to_from_edge`) cache once. Idempotent.
    void build_rev_cache_() {
        if (rev_cache_built_) {
            return;
        }
        const auto t_start = std::chrono::steady_clock::now();
        rev_cache_.reserve(storage.get_node_count());
        auto* rev_index = storage.get_to_from_edge_index();
        rev_cache_entries_ = scan_into_(rev_index, rev_cache_);
        rev_cache_built_ = true;
        const auto t_end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(
            t_end - t_start).count();
        std::cerr << "TopologyAccessor: rev adjacency cache built — "
                  << rev_cache_.size() << " keys, "
                  << rev_cache_entries_ << " directed entries, "
                  << static_cast<int>(ms) << " ms" << std::endl;
    }

    // Returns the cached neighbours of `node_id` from `cache`, or an empty
    // Neighbors when the key is absent. Repackages the raw entries as
    // ObjectIds so callers get the same type they would from the BPT path.
    static Neighbors materialise_from_cache_(
        const std::unordered_map<uint64_t, std::vector<TopologyAccessor::AdjEntry>>& cache,
        uint64_t node_id)
    {
        Neighbors result;
        auto it = cache.find(node_id);
        if (it == cache.end()) {
            return result;
        }
        const auto& entries = it->second;
        result.node_ids.reserve(entries.size());
        result.edge_ids.reserve(entries.size());
        for (const auto& e : entries) {
            result.node_ids.push_back(ObjectId(e.node_id));
            result.edge_ids.push_back(ObjectId(e.edge_id));
        }
        return result;
    }

    // Buffer-reusing variant of materialise_from_cache_: fills the caller's
    // `out` (capacity retained) instead of allocating a fresh Neighbors per
    // node. Byte-identical content + order; only the allocation is avoided.
    static void materialise_from_cache_into_(
        const std::unordered_map<uint64_t, std::vector<TopologyAccessor::AdjEntry>>& cache,
        uint64_t node_id, Neighbors& out)
    {
        out.node_ids.clear();
        out.edge_ids.clear();
        auto it = cache.find(node_id);
        if (it == cache.end()) {
            return;
        }
        const auto& entries = it->second;
        out.node_ids.reserve(entries.size());
        out.edge_ids.reserve(entries.size());
        for (const auto& e : entries) {
            out.node_ids.push_back(ObjectId(e.node_id));
            out.edge_ids.push_back(ObjectId(e.edge_id));
        }
    }

    // CSR fast-path for outgoing neighbors. Returns std::nullopt when the
    // fast-path is not viable (reader inert, node_idx out of range) so the
    // caller falls through to the B+Tree path.
    std::optional<Neighbors> try_csr_out_neighbors(ObjectId node_id) const {
        // The CSR is indexed by dense row id (see native_projection_builder.cc
        // build_one_topology_snapshot_). B+Tree records store full ObjectIds
        // with the 8-bit type tag, so the caller's node_id is masked here to
        // match the ROW_PTR subscript convention.
        const uint64_t row_idx = node_id.get_value();
        if (!fwd_csr_.has_data() || row_idx >= fwd_csr_.num_nodes()) {
            return std::nullopt;
        }
        // Width-agnostic copy: handles both id_width==8 (full tagged ObjectIds,
        // memcpy) and id_width==4 (uint32 sidecar where the 8-bit ObjectId
        // type tag is stripped at write time and reconstructed by the reader,
        // halving topology size). The values returned are the exact same
        // tagged uint64s either way, so ObjectId() wrapping is identical
        // across widths.
        std::vector<uint64_t> dst_tmp;
        fwd_csr_.copy_neighbors(row_idx, dst_tmp);
        std::vector<uint64_t> eid_tmp;
        fwd_csr_.copy_edge_ids(row_idx, eid_tmp);
        Neighbors result;
        result.node_ids.reserve(dst_tmp.size());
        for (uint64_t v : dst_tmp) {
            result.node_ids.push_back(ObjectId(v));
        }
        if (eid_tmp.size() == dst_tmp.size()) {
            result.edge_ids.reserve(eid_tmp.size());
            for (uint64_t v : eid_tmp) {
                result.edge_ids.push_back(ObjectId(v));
            }
        }
        return result;
    }

    std::optional<Neighbors> try_csr_in_neighbors(ObjectId node_id) const {
        // Same masking convention as try_csr_out_neighbors — see its comment.
        const uint64_t row_idx = node_id.get_value();
        if (!rev_csr_.has_data() || row_idx >= rev_csr_.num_nodes()) {
            return std::nullopt;
        }
        // Width-agnostic copy — see try_csr_out_neighbors for explanation.
        std::vector<uint64_t> dst_tmp;
        rev_csr_.copy_neighbors(row_idx, dst_tmp);
        std::vector<uint64_t> eid_tmp;
        rev_csr_.copy_edge_ids(row_idx, eid_tmp);
        Neighbors result;
        result.node_ids.reserve(dst_tmp.size());
        for (uint64_t v : dst_tmp) {
            result.node_ids.push_back(ObjectId(v));
        }
        if (eid_tmp.size() == dst_tmp.size()) {
            result.edge_ids.reserve(eid_tmp.size());
            for (uint64_t v : eid_tmp) {
                result.edge_ids.push_back(ObjectId(v));
            }
        }
        return result;
    }

    /**
     * @brief Count neighbors of `node_id` in `index` without materialising
     *        them.
     *
     * Companion to `get_neighbors_from_index` for the degree fast-path.
     * Walks the same B+Tree range and applies the same defensive
     * same-key post-filter, but only increments a counter — no
     * `Neighbors` allocation, no per-tuple ObjectId construction. Critical
     * on papers100M scale where naively reusing the materialising path
     * (110M nodes × avg ~15 neighbors × 16 B/entry × 2 directions) would
     * thrash the allocator with ~25 GB of transient allocations during
     * `TopologyFrequencyProfiler::compute_from_degrees_`.
     *
     * The same-key filter is preserved verbatim from
     * `get_neighbors_from_index` to keep the count semantically identical
     * to that of the materialised path under both BTREE and CSR-hybrid
     * storage (where edge-index B+Tree leaves contain the CSR layout
     * directly). See `get_neighbors_from_index`'s comment block for the
     * cross-page range-iterator motivation.
     */
    int64_t count_neighbors_from_index(ObjectId node_id, BPlusTree<3>* index) {
        if (!index) {
            return 0;
        }
        Record<3> min_record = {node_id.id, 0, 0};
        Record<3> max_record = {node_id.id, UINT64_MAX, UINT64_MAX};
        bool interruption_requested = false;
        auto it = index->get_range(&interruption_requested, min_record, max_record);
        int64_t count = 0;
        const Record<3>* record;
        while ((record = it.next()) != nullptr) {
            if (std::get<0>(*record) != node_id.id) {
                continue;
            }
            ++count;
        }
        return count;
    }

    /**
     * @brief Get neighbors using from_to_edge index (outgoing).
     *
     * Filters emitted records by exact src-id match as a defensive post-pass.
     * The BPT range iterator (BptIter<3>::next) emits every record in
     * [min, max] by pairwise-comparing each field against max only. Under
     * plain BTREE storage, page ordering guarantees records with src < seed_id
     * never reach the iterator because search_leaf + search_index position
     * past min exactly. Under CSR-hybrid storage (where each edge-index
     * B+Tree leaf page IS a CSR row and v3 leaves embed the CSR layout
     * directly), cross-page transitions reset current_pos to 0 on the new
     * page; continuation pages carry the correct src automatically, but a
     * chain-head page reached immediately after the query's own tail may
     * surface tuples whose src is strictly less than max[0] yet unrelated
     * to the queried seed. Because those records satisfy record[0] < max[0]
     * in next()'s inequality check, they are returned. We discard them here
     * by comparing against the queried src_id. This post-filter is O(1) per
     * tuple and preserves correctness regardless of the underlying leaf format.
     */
    Neighbors get_neighbors_from_index(ObjectId node_id, BPlusTree<3>* index) {
        Neighbors result;

        if (!index) {
            return result;
        }

        // Search for (node_id, MIN, MIN) to (node_id, MAX, MAX)
        Record<3> min_record = {node_id.id, 0, 0};
        Record<3> max_record = {node_id.id, UINT64_MAX, UINT64_MAX};

        bool interruption_requested = false;
        auto it = index->get_range(&interruption_requested, min_record, max_record);

        const Record<3>* record;
        while ((record = it.next()) != nullptr) {
            // from_to_edge: (from, to, edge_id)
            // Defensive: drop records whose src does not exactly match.
            // See function-level comment above for the cross-page range-iterator motivation.
            if (std::get<0>(*record) != node_id.id) {
                continue;
            }
            result.node_ids.push_back(ObjectId(std::get<1>(*record)));
            result.edge_ids.push_back(ObjectId(std::get<2>(*record)));
        }

        return result;
    }

    /**
     * @brief Sample k items from a vector uniformly.
     */
    template<typename T>
    std::vector<T> uniform_sample(const std::vector<T>& items, size_t k) {
        if (items.size() <= k) {
            return items;
        }

        std::vector<T> result;
        result.reserve(k);

        // Fisher-Yates shuffle on indices
        std::vector<size_t> indices(items.size());
        std::iota(indices.begin(), indices.end(), 0);

        for (size_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<size_t> dist(i, indices.size() - 1);
            size_t j = dist(rng);
            std::swap(indices[i], indices[j]);
            result.push_back(items[indices[i]]);
        }

        return result;
    }
};

// ============================================================================
// TopologyAccessor Public Methods
// ============================================================================

TopologyAccessor::TopologyAccessor(GQL::ProjectionStorage& storage)
    : impl_(std::make_unique<Impl>(storage)) {}

TopologyAccessor::~TopologyAccessor() = default;

TopologyAccessor::TopologyAccessor(TopologyAccessor&&) noexcept = default;
TopologyAccessor& TopologyAccessor::operator=(TopologyAccessor&&) noexcept = default;

// ----- Single Node Neighbor Access -----

namespace {

// Convert a FourLevelTopologyStore::Neighbors into the legacy
// `mdb::gnn::Neighbors` shape returned by every existing API call.
Neighbors materialise_from_four_level_(
    const FourLevelTopologyStore::Neighbors& src)
{
    Neighbors out;
    out.node_ids.reserve(src.size());
    out.edge_ids.reserve(src.size());
    src.for_each_with_edge_id([&](uint64_t dst, uint64_t eid) {
        out.node_ids.push_back(ObjectId(dst));
        out.edge_ids.push_back(ObjectId(eid));
    });
    return out;
}

// Buffer-reusing variant: fill the caller's `out` (capacity retained across
// calls) instead of allocating a fresh Neighbors per node. Byte-identical
// content + order to materialise_from_four_level_; only the allocation is
// avoided. Measured: the per-node Neighbors allocation in the UNDIRECTED fetch
// is the dominant expand sub-cost.
void materialise_from_four_level_into_(
    const FourLevelTopologyStore::Neighbors& src, Neighbors& out)
{
    out.node_ids.clear();
    out.edge_ids.clear();
    out.node_ids.reserve(src.size());
    out.edge_ids.reserve(src.size());
    src.for_each_with_edge_id([&](uint64_t dst, uint64_t eid) {
        out.node_ids.push_back(ObjectId(dst));
        out.edge_ids.push_back(ObjectId(eid));
    });
}

}  // namespace

Neighbors TopologyAccessor::get_out_neighbors(ObjectId node_id) {
    // Highest-priority path: frequency-tiered Four-Level Topology Store
    // (L1 RAM hash / L2 compact uint32 CSR / L3 mmap sidecar / L4 B+Tree).
    // Dispatched first when enabled via enable_four_level_store().
    if (impl_->four_level_store_) {
        return materialise_from_four_level_(
            impl_->four_level_store_->get_out_neighbors(node_id));
    }
    // Fast path: in-memory adjacency cache (full B+Tree scan into
    // unordered_map<src, vector<AdjEntry>>, built once on demand). Supersedes
    // both the CSR mmap and the B+Tree range query when populated.
    if (impl_->cache_enabled_ && impl_->fwd_cache_built_) {
        return Impl::materialise_from_cache_(impl_->fwd_cache_, node_id.id);
    }
    // mmap path: mmap'd topology_fwd.csr (O(1) neighbor slice). Absent /
    // stale / out-of-range → fall through to the B+Tree path below.
    if (auto fast = impl_->try_csr_out_neighbors(node_id)) {
        return std::move(*fast);
    }
    return impl_->get_neighbors_from_index(
        node_id, impl_->storage.get_from_to_edge_index());
}

Neighbors TopologyAccessor::get_in_neighbors(ObjectId node_id) {
    // Highest-priority path: frequency-tiered Four-Level Topology Store.
    // Dispatched first when enabled via enable_four_level_store().
    if (impl_->four_level_store_) {
        return materialise_from_four_level_(
            impl_->four_level_store_->get_in_neighbors(node_id));
    }
    // Fast path: in-memory adjacency cache (full B+Tree scan into
    // unordered_map<src, vector<AdjEntry>>, built once on demand).
    if (impl_->cache_enabled_ && impl_->rev_cache_built_) {
        return Impl::materialise_from_cache_(impl_->rev_cache_, node_id.id);
    }
    // mmap path: mmap'd topology_rev.csr (O(1) neighbor slice).
    if (auto fast = impl_->try_csr_in_neighbors(node_id)) {
        return std::move(*fast);
    }

    // to_from_edge: (to, from, edge_id)
    Neighbors result;

    auto* index = impl_->storage.get_to_from_edge_index();
    if (!index) {
        return result;
    }

    Record<3> min_record = {node_id.id, 0, 0};
    Record<3> max_record = {node_id.id, UINT64_MAX, UINT64_MAX};

    bool interruption_requested = false;
    auto it = index->get_range(&interruption_requested, min_record, max_record);

    const Record<3>* record;
    while ((record = it.next()) != nullptr) {
        // to_from_edge: (to, from, edge_id)
        // Defensive: same-key post-filter as in get_neighbors_from_index().
        // See that function's comment for the cross-page range-iterator rationale.
        if (std::get<0>(*record) != node_id.id) {
            continue;
        }
        result.node_ids.push_back(ObjectId(std::get<1>(*record)));
        result.edge_ids.push_back(ObjectId(std::get<2>(*record)));
    }

    return result;
}

Neighbors TopologyAccessor::get_neighbors(ObjectId node_id) {
    return get_neighbors(node_id, EdgeOrientation::UNDIRECTED);
}

// Buffer-reusing variants: dispatch the hot tiers (four-level store / in-memory
// adjacency cache) directly into `out`, avoiding the per-node Neighbors
// allocation. Cold paths (B+Tree / mmap) fall back to the by-value path and
// move-assign — rare and small-scale, so the allocation there is not on the hot
// path. Content + order are byte-identical to get_out_neighbors / get_in_neighbors.
void TopologyAccessor::get_out_neighbors_into(ObjectId node_id, Neighbors& out) {
    if (impl_->four_level_store_) {
        materialise_from_four_level_into_(
            impl_->four_level_store_->get_out_neighbors(node_id), out);
        return;
    }
    if (impl_->cache_enabled_ && impl_->fwd_cache_built_) {
        Impl::materialise_from_cache_into_(impl_->fwd_cache_, node_id.id, out);
        return;
    }
    out = get_out_neighbors(node_id);  // cold path (mmap / B+Tree)
}

void TopologyAccessor::get_in_neighbors_into(ObjectId node_id, Neighbors& out) {
    if (impl_->four_level_store_) {
        materialise_from_four_level_into_(
            impl_->four_level_store_->get_in_neighbors(node_id), out);
        return;
    }
    if (impl_->cache_enabled_ && impl_->rev_cache_built_) {
        Impl::materialise_from_cache_into_(impl_->rev_cache_, node_id.id, out);
        return;
    }
    out = get_in_neighbors(node_id);  // cold path (mmap / B+Tree)
}

Neighbors TopologyAccessor::get_neighbors(ObjectId node_id, EdgeOrientation orientation) {
    // Thin wrapper over the buffer-reusing path — a single dedup implementation.
    Neighbors out;
    get_neighbors_into(node_id, orientation, out);
    return out;
}

void TopologyAccessor::get_neighbors_into(ObjectId node_id,
                                          EdgeOrientation orientation,
                                          Neighbors& out) {
    switch (orientation) {
        case EdgeOrientation::NATURAL:
            get_out_neighbors_into(node_id, out);
            return;

        case EdgeOrientation::REVERSE:
            get_in_neighbors_into(node_id, out);
            return;

        case EdgeOrientation::UNDIRECTED: {
            // Four-level store path: ONE pre-merged undirected fetch. The store
            // collapses out+in+dedup onto its symmetric tier when built, and
            // onto the same-rule merge otherwise — either way one call, one
            // result, byte-identical to the dual-fetch + merge below. This drops
            // the per-node out+in fetch + dedup that was the dominant expand
            // sub-cost.
            if (impl_->four_level_store_) {
                materialise_from_four_level_into_(
                    impl_->four_level_store_->get_neighbors(node_id), out);
                return;
            }

            // Per-worker reusable scratch for the two directions (each parallel
            // sampler thread gets its own via thread_local). Filled in place so
            // the UNDIRECTED fetch allocates no fresh Neighbors per node.
            thread_local Neighbors out_neighbors;
            thread_local Neighbors in_neighbors;
            get_out_neighbors_into(node_id, out_neighbors);
            get_in_neighbors_into(node_id, in_neighbors);

            // Deduplicate to merge results from both out- and in- index lookups.
            //
            // Historically we deduplicated by EDGE ID. That works for BTREE
            // storage where each edge has a unique id, but breaks under
            // CSR-hybrid storage (where edge-index B+Tree leaves ARE the CSR
            // layout) because the v3 leaf format omits edge_id (edge_id = 0
            // for every tuple). An all-zero dedup key collapses
            // ALL neighbors of the node to a single element, which in turn
            // makes Phase B k-hop sampling degenerate (layer-1 node count
            // always = 1) and drags chunk-level wall-clock time into the
            // O(N^2) regime for large projections (empirically observed on
            // arxiv: chunk N took ~6*N seconds before this fix).
            //
            // The robust fix is to detect whether edge_ids are meaningful
            // and fall back to NEIGHBOR NODE ID dedup otherwise. We sample
            // the first tuple from each side — if either side reports a
            // non-zero edge_id we trust the edge_id dedup key; otherwise
            // we use the neighbor node id, which is the natural dedup key
            // under simple graphs (no parallel edges), which covers every
            // citation / co-purchase / knowledge-graph workload we currently
            // target. Keep the result's parallel `edge_ids` array in sync
            // so downstream consumers (GraphSample edge_ids) still receive
            // zero-sentinels consistent with the storage's view of the edge
            // space rather than synthesized values.
            const bool has_edge_ids =
                (!out_neighbors.edge_ids.empty()
                     && out_neighbors.edge_ids.front().id != 0)
                || (!in_neighbors.edge_ids.empty()
                     && in_neighbors.edge_ids.front().id != 0);

            // Merge into the caller's `out` buffer (capacity retained across
            // calls). Cleared here; same dedup order as before.
            out.node_ids.clear();
            out.edge_ids.clear();
            // Per-thread dedup arena reused across UNDIRECTED lookups.
            // Avoids per-call hash-table reallocation; parallel sampling
            // workers each get their own arena via thread_local.
            thread_local std::unordered_set<uint64_t> seen_arena;
            seen_arena.clear();
            const size_t total = out_neighbors.node_ids.size()
                               + in_neighbors.node_ids.size();
            seen_arena.reserve(total);
            out.node_ids.reserve(total);
            out.edge_ids.reserve(total);

            for (size_t i = 0; i < out_neighbors.node_ids.size(); ++i) {
                const uint64_t key = has_edge_ids
                    ? out_neighbors.edge_ids[i].id
                    : out_neighbors.node_ids[i].id;
                if (seen_arena.insert(key).second) {
                    out.node_ids.push_back(out_neighbors.node_ids[i]);
                    out.edge_ids.push_back(out_neighbors.edge_ids[i]);
                }
            }

            for (size_t i = 0; i < in_neighbors.node_ids.size(); ++i) {
                const uint64_t key = has_edge_ids
                    ? in_neighbors.edge_ids[i].id
                    : in_neighbors.node_ids[i].id;
                if (seen_arena.insert(key).second) {
                    out.node_ids.push_back(in_neighbors.node_ids[i]);
                    out.edge_ids.push_back(in_neighbors.edge_ids[i]);
                }
            }

            return;
        }
    }
}

// ----- Batch Neighbor Access -----

std::unordered_map<uint64_t, Neighbors> TopologyAccessor::get_batch_out_neighbors(
    const std::vector<ObjectId>& node_ids
) {
    std::unordered_map<uint64_t, Neighbors> result;
    result.reserve(node_ids.size());

    for (const auto& node_id : node_ids) {
        result[node_id.id] = get_out_neighbors(node_id);
    }

    return result;
}

std::unordered_map<uint64_t, Neighbors> TopologyAccessor::get_batch_in_neighbors(
    const std::vector<ObjectId>& node_ids
) {
    std::unordered_map<uint64_t, Neighbors> result;
    result.reserve(node_ids.size());

    for (const auto& node_id : node_ids) {
        result[node_id.id] = get_in_neighbors(node_id);
    }

    return result;
}

std::unordered_map<uint64_t, Neighbors> TopologyAccessor::get_batch_neighbors(
    const std::vector<ObjectId>& node_ids,
    EdgeOrientation orientation
) {
    std::unordered_map<uint64_t, Neighbors> result;
    result.reserve(node_ids.size());

    for (const auto& node_id : node_ids) {
        result[node_id.id] = get_neighbors(node_id, orientation);
    }

    return result;
}

// ----- Edge Index Construction -----

EdgeIndex TopologyAccessor::build_edge_index(const std::vector<ObjectId>& node_ids) {
    // Build node set for filtering
    std::unordered_set<uint64_t> node_set;
    std::unordered_map<uint64_t, int64_t> id_to_idx;

    for (size_t i = 0; i < node_ids.size(); ++i) {
        node_set.insert(node_ids[i].id);
        id_to_idx[node_ids[i].id] = static_cast<int64_t>(i);
    }

    // Collect edges within the node set
    std::vector<int64_t> src_indices;
    std::vector<int64_t> dst_indices;

    for (const auto& node_id : node_ids) {
        Neighbors neighbors = get_out_neighbors(node_id);

        for (const auto& neighbor_id : neighbors.node_ids) {
            if (node_set.count(neighbor_id.id)) {
                src_indices.push_back(id_to_idx[node_id.id]);
                dst_indices.push_back(id_to_idx[neighbor_id.id]);
            }
        }
    }

    // Build tensor
    int64_t num_edges = static_cast<int64_t>(src_indices.size());
    int64_t num_nodes = static_cast<int64_t>(node_ids.size());

    torch::Tensor edge_index;
    if (num_edges > 0) {
        edge_index = torch::empty({2, num_edges}, torch::kInt64);
        auto accessor = edge_index.accessor<int64_t, 2>();

        for (int64_t i = 0; i < num_edges; ++i) {
            accessor[0][i] = src_indices[i];
            accessor[1][i] = dst_indices[i];
        }

        edge_index = edge_index.to(impl_->target_device);
    } else {
        edge_index = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(impl_->target_device));
    }

    return EdgeIndex{
        std::move(edge_index),
        num_nodes,
        num_nodes
    };
}

EdgeIndex TopologyAccessor::build_bipartite_edge_index(
    const std::vector<ObjectId>& src_nodes,
    const std::vector<ObjectId>& dst_nodes
) {
    // Build mappings
    std::unordered_map<uint64_t, int64_t> src_id_to_idx;
    std::unordered_map<uint64_t, int64_t> dst_id_to_idx;

    for (size_t i = 0; i < src_nodes.size(); ++i) {
        src_id_to_idx[src_nodes[i].id] = static_cast<int64_t>(i);
    }
    for (size_t i = 0; i < dst_nodes.size(); ++i) {
        dst_id_to_idx[dst_nodes[i].id] = static_cast<int64_t>(i);
    }

    // Collect edges from src to dst
    std::vector<int64_t> src_indices;
    std::vector<int64_t> dst_indices;

    for (const auto& src_node : src_nodes) {
        Neighbors neighbors = get_out_neighbors(src_node);

        for (const auto& neighbor_id : neighbors.node_ids) {
            auto it = dst_id_to_idx.find(neighbor_id.id);
            if (it != dst_id_to_idx.end()) {
                src_indices.push_back(src_id_to_idx[src_node.id]);
                dst_indices.push_back(it->second);
            }
        }
    }

    // Build tensor
    int64_t num_edges = static_cast<int64_t>(src_indices.size());

    torch::Tensor edge_index;
    if (num_edges > 0) {
        edge_index = torch::empty({2, num_edges}, torch::kInt64);
        auto accessor = edge_index.accessor<int64_t, 2>();

        for (int64_t i = 0; i < num_edges; ++i) {
            accessor[0][i] = src_indices[i];
            accessor[1][i] = dst_indices[i];
        }

        edge_index = edge_index.to(impl_->target_device);
    } else {
        edge_index = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(impl_->target_device));
    }

    return EdgeIndex{
        std::move(edge_index),
        static_cast<int64_t>(src_nodes.size()),
        static_cast<int64_t>(dst_nodes.size())
    };
}

// ----- Neighbor Sampling -----

SampledSubgraph TopologyAccessor::sample_neighbors(
    const std::vector<ObjectId>& seed_nodes,
    int64_t fanout,
    SamplingStrategy strategy,
    EdgeOrientation orientation
) {
    SampledSubgraph result;
    result.dst_nodes = seed_nodes;

    // Build dst mapping
    for (size_t i = 0; i < seed_nodes.size(); ++i) {
        result.dst_id_to_idx[seed_nodes[i].id] = static_cast<int64_t>(i);
    }

    // Single pass: collect neighbors, sample, and store selected IDs for edge building.
    // Uses one RNG source (impl_->rng) to guarantee src_set and edges are consistent.
    std::unordered_set<uint64_t> src_set;
    std::vector<std::vector<uint64_t>> per_dst_selected_ids;
    per_dst_selected_ids.reserve(seed_nodes.size());

    for (size_t dst_idx = 0; dst_idx < seed_nodes.size(); ++dst_idx) {
        Neighbors neighbors = get_neighbors(seed_nodes[dst_idx], orientation);

        // Sample if needed (Fisher-Yates partial shuffle)
        std::vector<size_t> selected_indices;
        if (fanout > 0 && strategy == SamplingStrategy::UNIFORM &&
            static_cast<int64_t>(neighbors.node_ids.size()) > fanout) {

            std::vector<size_t> indices(neighbors.node_ids.size());
            std::iota(indices.begin(), indices.end(), 0);

            for (int64_t i = 0; i < fanout; ++i) {
                std::uniform_int_distribution<size_t> dist(i, indices.size() - 1);
                size_t j = dist(impl_->rng);
                std::swap(indices[i], indices[j]);
            }
            selected_indices.assign(indices.begin(), indices.begin() + fanout);
        } else {
            selected_indices.resize(neighbors.node_ids.size());
            std::iota(selected_indices.begin(), selected_indices.end(), 0);
        }

        // Collect selected neighbor IDs for both src_set and deferred edge building
        std::vector<uint64_t> selected_ids;
        selected_ids.reserve(selected_indices.size());
        for (size_t idx : selected_indices) {
            uint64_t neighbor_id = neighbors.node_ids[idx].id;
            src_set.insert(neighbor_id);
            selected_ids.push_back(neighbor_id);
        }
        per_dst_selected_ids.push_back(std::move(selected_ids));
    }

    // Build src nodes list and mapping
    result.src_nodes.reserve(src_set.size());
    for (uint64_t id : src_set) {
        result.src_id_to_idx[id] = static_cast<int64_t>(result.src_nodes.size());
        result.src_nodes.push_back(ObjectId(id));
    }

    // Build edges from stored selections (consistent with src_set)
    std::vector<int64_t> src_indices;
    std::vector<int64_t> dst_indices;

    for (size_t dst_idx = 0; dst_idx < per_dst_selected_ids.size(); ++dst_idx) {
        for (uint64_t neighbor_id : per_dst_selected_ids[dst_idx]) {
            src_indices.push_back(result.src_id_to_idx.at(neighbor_id));
            dst_indices.push_back(static_cast<int64_t>(dst_idx));
        }
    }

    // Build edge tensor
    int64_t num_edges = static_cast<int64_t>(src_indices.size());

    torch::Tensor edge_index;
    if (num_edges > 0) {
        edge_index = torch::empty({2, num_edges}, torch::kInt64);
        auto accessor = edge_index.accessor<int64_t, 2>();

        for (int64_t i = 0; i < num_edges; ++i) {
            accessor[0][i] = src_indices[i];
            accessor[1][i] = dst_indices[i];
        }

        edge_index = edge_index.to(impl_->target_device);
    } else {
        edge_index = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(impl_->target_device));
    }

    result.edge_index = EdgeIndex{
        std::move(edge_index),
        static_cast<int64_t>(result.src_nodes.size()),
        static_cast<int64_t>(result.dst_nodes.size())
    };

    return result;
}

SampledSubgraph TopologyAccessor::sample_in_neighbors(
    const std::vector<ObjectId>& seed_nodes,
    int64_t fanout,
    SamplingStrategy strategy
) {
    // Legacy method: use REVERSE orientation (incoming edges)
    return sample_neighbors(seed_nodes, fanout, strategy, EdgeOrientation::REVERSE);
}

std::vector<SampledSubgraph> TopologyAccessor::sample_khop_neighbors(
    const std::vector<ObjectId>& seed_nodes,
    const std::vector<int64_t>& fanouts,
    SamplingStrategy strategy,
    EdgeOrientation orientation
) {
    std::vector<SampledSubgraph> layers;
    layers.reserve(fanouts.size());

    std::vector<ObjectId> current_seeds = seed_nodes;

    for (int64_t fanout : fanouts) {
        SampledSubgraph layer = sample_neighbors(current_seeds, fanout, strategy, orientation);
        current_seeds = layer.src_nodes;  // Next layer's seeds are this layer's sources
        layers.push_back(std::move(layer));
    }

    return layers;
}

// ----- Statistics -----

int64_t TopologyAccessor::get_out_degree(ObjectId node_id) {
    // Fastest path: in-memory adjacency cache (full B+Tree scan stored as
    // unordered_map). Hash lookup returns the entry vector size directly —
    // no copies.
    if (impl_->cache_enabled_ && impl_->fwd_cache_built_) {
        auto it = impl_->fwd_cache_.find(node_id.id);
        return it == impl_->fwd_cache_.end()
            ? 0
            : static_cast<int64_t>(it->second.size());
    }
    // Fast path: mmap'd topology_fwd.csr (O(1) ROW_PTR span). The reader
    // exposes the per-row span directly; we only need its size.
    const uint64_t row_idx = node_id.get_value();
    if (impl_->fwd_csr_.has_data() && row_idx < impl_->fwd_csr_.num_nodes()) {
        // degree() is width-agnostic (ROW_PTR is uint64 for both id widths)
        // and never throws on the narrow layout, unlike neighbors().
        return static_cast<int64_t>(impl_->fwd_csr_.degree(row_idx));
    }
    // B+Tree path: count tuples without materialising a Neighbors struct.
    return impl_->count_neighbors_from_index(
        node_id, impl_->storage.get_from_to_edge_index());
}

int64_t TopologyAccessor::get_in_degree(ObjectId node_id) {
    // Fastest path: in-memory adjacency cache (full B+Tree scan stored as
    // unordered_map).
    if (impl_->cache_enabled_ && impl_->rev_cache_built_) {
        auto it = impl_->rev_cache_.find(node_id.id);
        return it == impl_->rev_cache_.end()
            ? 0
            : static_cast<int64_t>(it->second.size());
    }
    // Fast path: mmap'd topology_rev.csr (O(1) ROW_PTR span).
    const uint64_t row_idx = node_id.get_value();
    if (impl_->rev_csr_.has_data() && row_idx < impl_->rev_csr_.num_nodes()) {
        // Width-agnostic degree — see get_out_degree.
        return static_cast<int64_t>(impl_->rev_csr_.degree(row_idx));
    }
    // B+Tree path: count without materialising.
    return impl_->count_neighbors_from_index(
        node_id, impl_->storage.get_to_from_edge_index());
}

uint64_t TopologyAccessor::get_edge_count() const {
    return impl_->storage.get_edge_count();
}

uint64_t TopologyAccessor::get_node_count() const {
    return impl_->storage.get_node_count();
}

// ----- Configuration -----

void TopologyAccessor::set_random_seed(uint64_t seed) {
    impl_->rng.seed(seed);
}

void TopologyAccessor::set_target_device(torch::Device device) {
    impl_->target_device = device;
}

// ----- In-memory adjacency cache (full B+Tree scan into unordered_map) -----

void TopologyAccessor::enable_adjacency_cache(bool enabled) {
    impl_->cache_enabled_ = enabled;
    if (!enabled) {
        // Drop existing entries so callers reverting to the BPT path are not
        // silently held to stale data, and so the RAM is reclaimed
        // immediately (the maps own their buckets).
        impl_->fwd_cache_.clear();
        impl_->rev_cache_.clear();
        impl_->fwd_cache_built_ = false;
        impl_->rev_cache_built_ = false;
        impl_->fwd_cache_entries_ = 0;
        impl_->rev_cache_entries_ = 0;
    }
}

bool TopologyAccessor::is_adjacency_cache_enabled() const {
    return impl_->cache_enabled_;
}

uint64_t TopologyAccessor::prebuild_adjacency_cache(EdgeOrientation orientation) {
    if (!impl_->cache_enabled_) {
        return 0;
    }
    const auto t_start = std::chrono::steady_clock::now();
    switch (orientation) {
        case EdgeOrientation::NATURAL:
            impl_->build_fwd_cache_();
            break;
        case EdgeOrientation::REVERSE:
            impl_->build_rev_cache_();
            break;
        case EdgeOrientation::UNDIRECTED:
            impl_->build_fwd_cache_();
            impl_->build_rev_cache_();
            break;
    }
    const auto t_end = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration<double, std::milli>(t_end - t_start).count());
}

bool TopologyAccessor::is_adjacency_cache_built(EdgeOrientation orientation) const {
    if (!impl_->cache_enabled_) {
        return false;
    }
    switch (orientation) {
        case EdgeOrientation::NATURAL:
            return impl_->fwd_cache_built_;
        case EdgeOrientation::REVERSE:
            return impl_->rev_cache_built_;
        case EdgeOrientation::UNDIRECTED:
            return impl_->fwd_cache_built_ && impl_->rev_cache_built_;
    }
    return false;
}

uint64_t TopologyAccessor::get_adjacency_cache_size_bytes() const {
    if (!impl_->cache_enabled_) {
        return 0;
    }
    // Each AdjEntry occupies 16 bytes (2 × uint64). Each map bucket pays an
    // estimated 32-byte hash overhead (next-pointer + hash + key + size).
    // The vector inside each value holds capacity ≥ size entries; use size
    // as a tight lower bound (vectors reserved exact in scan_into_ are
    // worst-case 1.5× via geometric growth). This is a rough resident
    // estimate intended for instrumentation, not allocation accounting.
    constexpr uint64_t kBucketOverhead = 32;
    constexpr uint64_t kEntryBytes     = sizeof(AdjEntry);
    uint64_t total = 0;
    total += impl_->fwd_cache_.size() * kBucketOverhead;
    total += impl_->fwd_cache_entries_ * kEntryBytes;
    total += impl_->rev_cache_.size() * kBucketOverhead;
    total += impl_->rev_cache_entries_ * kEntryBytes;
    return total;
}

uint64_t TopologyAccessor::get_adjacency_cache_fwd_entries() const {
    return impl_->fwd_cache_entries_;
}

uint64_t TopologyAccessor::get_adjacency_cache_rev_entries() const {
    return impl_->rev_cache_entries_;
}

// ----- Four-Level Topology Store (L1 RAM hash / L2 compact uint32 CSR /
//       L3 mmap sidecar / L4 direct B+Tree, frequency-tiered) -----

void TopologyAccessor::enable_four_level_store(
    const FourLevelTopologyStore::Config& config)
{
    if (impl_->four_level_store_) {
        throw std::logic_error(
            "TopologyAccessor::enable_four_level_store called twice — "
            "drop the accessor and recreate to swap configurations");
    }

    // Drop the single-tier in-memory adjacency cache if it was populated.
    // The four-level store contains its own L1 hash that subsumes it;
    // keeping both alive doubles RAM accounting.
    impl_->cache_enabled_ = false;
    impl_->fwd_cache_.clear();
    impl_->rev_cache_.clear();
    impl_->fwd_cache_built_ = false;
    impl_->rev_cache_built_ = false;
    impl_->fwd_cache_entries_ = 0;
    impl_->rev_cache_entries_ = 0;

    auto fwd_idx = impl_->storage.get_from_to_edge_index();
    auto rev_idx = impl_->storage.get_to_from_edge_index();
    auto proj_dir = std::filesystem::path(
        impl_->storage.get_projection_dir());

    impl_->four_level_store_ = std::make_unique<FourLevelTopologyStore>(
        fwd_idx,
        rev_idx,
        &impl_->storage,
        proj_dir,
        config);
    impl_->four_level_store_->build();
}

bool TopologyAccessor::is_four_level_store_enabled() const {
    return static_cast<bool>(impl_->four_level_store_);
}

bool TopologyAccessor::is_symmetric_topology_built() const {
    return impl_->four_level_store_
        && impl_->four_level_store_->is_symmetric_built();
}

const L2CompactCsr* TopologyAccessor::l2_fwd() const {
    return impl_->four_level_store_ ? impl_->four_level_store_->l2_fwd() : nullptr;
}

const L2CompactCsr* TopologyAccessor::l2_rev() const {
    return impl_->four_level_store_ ? impl_->four_level_store_->l2_rev() : nullptr;
}

const GQL::Projection::TopologySnapshotReader* TopologyAccessor::l3_fwd() const {
    return impl_->four_level_store_ ? impl_->four_level_store_->l3_fwd() : nullptr;
}

const GQL::Projection::TopologySnapshotReader* TopologyAccessor::l3_rev() const {
    return impl_->four_level_store_ ? impl_->four_level_store_->l3_rev() : nullptr;
}

void TopologyAccessor::enable_pinned_gpu_view(const SamplingBackendPlan& plan) {
    if (impl_->four_level_store_) {
        impl_->four_level_store_->enable_pinned_gpu_view(plan);
    }
}

const PinnedTopologyView* TopologyAccessor::pinned_view() const {
    return impl_->four_level_store_ ? impl_->four_level_store_->pinned_view()
                                    : nullptr;
}

// ============================================================================
// NodeIterator Implementation
// ============================================================================

struct NodeIterator::Impl {
    GQL::ProjectionStorage& storage;
    BPlusTree<1>* nodes_index;
    BptIter<1> iterator;
    bool interruption_requested;
    bool exhausted;
    uint64_t iterated;
    uint64_t total;

    explicit Impl(GQL::ProjectionStorage& storage_)
        : storage(storage_),
          nodes_index(storage_.get_nodes_index()),
          interruption_requested(false),
          exhausted(false),
          iterated(0),
          total(storage_.get_node_count()) {

        // Initialize iterator to scan all nodes
        if (nodes_index) {
            Record<1> min_record = {0};
            Record<1> max_record = {UINT64_MAX};
            iterator = nodes_index->get_range(&interruption_requested, min_record, max_record);
        } else {
            exhausted = true;
        }
    }

    void reset_iterator() {
        if (nodes_index) {
            Record<1> min_record = {0};
            Record<1> max_record = {UINT64_MAX};
            interruption_requested = false;
            iterator = nodes_index->get_range(&interruption_requested, min_record, max_record);
            exhausted = false;
            iterated = 0;
        }
    }
};

NodeIterator::NodeIterator(GQL::ProjectionStorage& storage)
    : impl_(std::make_unique<Impl>(storage)) {}

NodeIterator::~NodeIterator() = default;

NodeIterator::NodeIterator(NodeIterator&&) noexcept = default;
NodeIterator& NodeIterator::operator=(NodeIterator&&) noexcept = default;

std::optional<ObjectId> NodeIterator::next() {
    if (impl_->exhausted) {
        return std::nullopt;
    }

    const Record<1>* record = impl_->iterator.next();
    if (record == nullptr) {
        impl_->exhausted = true;
        return std::nullopt;
    }

    impl_->iterated++;
    return ObjectId(std::get<0>(*record));
}

std::optional<std::vector<ObjectId>> NodeIterator::next_batch(size_t batch_size) {
    if (impl_->exhausted) {
        return std::nullopt;
    }

    std::vector<ObjectId> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        const Record<1>* record = impl_->iterator.next();
        if (record == nullptr) {
            impl_->exhausted = true;
            break;
        }
        batch.push_back(ObjectId(std::get<0>(*record)));
        impl_->iterated++;
    }

    if (batch.empty()) {
        return std::nullopt;
    }

    return batch;
}

void NodeIterator::reset() {
    impl_->reset_iterator();
}

bool NodeIterator::has_next() const {
    return !impl_->exhausted;
}

uint64_t NodeIterator::total_count() const {
    return impl_->total;
}

uint64_t NodeIterator::iterated_count() const {
    return impl_->iterated;
}

} // namespace mdb::gnn
