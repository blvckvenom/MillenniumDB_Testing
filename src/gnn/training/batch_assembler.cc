#include "gnn/training/batch_assembler.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace mdb::gnn {

namespace {
// Nested (DGL-block) aggregation default, read once from the environment.
// MDB_GNN_NESTED_AGG=1/true/yes => every node within k hops re-aggregates its
// sampled neighbours at conv k (standard nested-neighbourhood message passing);
// unset/0 => legacy per-hop wiring (seeds aggregate only at the final conv).
bool nested_agg_env_default() {
    const char* e = std::getenv("MDB_GNN_NESTED_AGG");
    return e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "true") == 0 ||
                 std::strcmp(e, "yes") == 0);
}
} // namespace

// =============================================================================
// Constructors
// =============================================================================

BatchAssembler::BatchAssembler(
    FourLevelStore& feature_store,
    SampleStorage&  samples,
    LabelStore*     labels,
    SplitStore*     splits,
    const RowMapping& row_mapping
)
    : feature_store_(&feature_store)
    , feature_matrix_(nullptr)
    , samples_(samples)
    , labels_(labels)
    , splits_(splits)
    , row_mapping_(row_mapping)
{
    nested_aggregation_ = nested_agg_env_default();
}

BatchAssembler::BatchAssembler(
    const FeatureMatrix& feature_matrix,
    SampleStorage&       samples,
    LabelStore*          labels,
    SplitStore*          splits,
    const RowMapping&    row_mapping
)
    : feature_store_(nullptr)
    , feature_matrix_(&feature_matrix)
    , samples_(samples)
    , labels_(labels)
    , splits_(splits)
    , row_mapping_(row_mapping)
{
    nested_aggregation_ = nested_agg_env_default();
}

// =============================================================================
// Public: assemble(batch_id)
// =============================================================================

namespace {
// Estimated RAM footprint of a cached structural bundle (index tensors only).
size_t estimate_struct_bytes(const std::vector<torch::Tensor>& edges,
                             const std::vector<torch::Tensor>& active,
                             const torch::Tensor& labels,
                             const torch::Tensor& label_mask) {
    size_t b = 0;
    for (const auto& t : edges)  b += static_cast<size_t>(t.numel()) * t.element_size();
    for (const auto& t : active) b += static_cast<size_t>(t.numel()) * t.element_size();
    if (labels.defined())     b += static_cast<size_t>(labels.numel()) * labels.element_size();
    if (label_mask.defined()) b += static_cast<size_t>(label_mask.numel()) * label_mask.element_size();
    return b + 256;  // small per-entry overhead
}
} // namespace

MiniBatch BatchAssembler::assemble(uint64_t batch_id) {
    auto t0 = std::chrono::steady_clock::now();
    GraphSample sample = samples_.read_sample(batch_id);
    auto t1 = std::chrono::steady_clock::now();
    uint64_t sample_read_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    // Structural cache fast path: reuse the per-batch index/label build across
    // epochs (it is a pure function of the sample) and only re-run the feature
    // load. We copy the cached fields under the lock (torch tensors are
    // refcounted, so this shares storage — no data copy) and release the lock
    // before the feature load.
    bool hit = false;
    MiniBatch mini;
    if (struct_budget_ > 0) {
        std::lock_guard<std::mutex> lk(struct_mu_);
        auto it = struct_cache_.find(batch_id);
        if (it != struct_cache_.end()) {
            ++struct_hits_;
            struct_lru_.splice(struct_lru_.begin(), struct_lru_, it->second.lru_it);
            const CachedStruct& c = it->second.s;
            mini.batch_id                 = sample.batch_id;
            mini.split                    = sample.split;
            mini.edge_indices             = c.edge_indices;
            mini.active_indices_per_layer = c.active_indices_per_layer;
            mini.active_sizes_per_layer   = c.active_sizes_per_layer;
            mini.labels                   = c.labels;
            mini.label_mask               = c.label_mask;
            mini.num_seeds                = c.num_seeds;
            mini.num_nodes                = c.num_nodes;
            mini.num_labeled              = c.num_labeled;
            hit = true;
        } else {
            ++struct_misses_;
        }
    }

    if (hit) {
        auto tl0 = std::chrono::steady_clock::now();
        mini.features = load_features(sample);
        auto tl1 = std::chrono::steady_clock::now();
        mini.timing.sample_read_ns      = sample_read_ns;
        mini.timing.assembler_kernel_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tl1 - tl0).count());
        if (feature_store_) {
            mini.timing.used_addr_tables = feature_store_->last_used_addr_tables();
            mini.timing.addr_load_ns     = feature_store_->last_addr_load_us() * 1000ULL;
        }
        return mini;
    }

    // Miss: full structural build + feature load, then cache the structural part.
    mini = assemble_from_sample(sample);
    mini.timing.sample_read_ns = sample_read_ns;

    if (struct_budget_ > 0) {
        std::lock_guard<std::mutex> lk(struct_mu_);
        if (struct_cache_.find(batch_id) == struct_cache_.end()) {
            size_t sz = estimate_struct_bytes(mini.edge_indices,
                                              mini.active_indices_per_layer,
                                              mini.labels, mini.label_mask);
            while (struct_bytes_ + sz > struct_budget_ && !struct_lru_.empty()) {
                uint64_t victim = struct_lru_.back();
                struct_lru_.pop_back();
                auto vit = struct_cache_.find(victim);
                if (vit != struct_cache_.end()) {
                    struct_bytes_ -= vit->second.bytes;
                    struct_cache_.erase(vit);
                    ++struct_evictions_;
                }
            }
            if (sz <= struct_budget_) {
                CachedStruct c;
                c.edge_indices             = mini.edge_indices;
                c.active_indices_per_layer = mini.active_indices_per_layer;
                c.active_sizes_per_layer   = mini.active_sizes_per_layer;
                c.labels                   = mini.labels;
                c.label_mask               = mini.label_mask;
                c.num_seeds                = mini.num_seeds;
                c.num_nodes                = mini.num_nodes;
                c.num_labeled              = mini.num_labeled;
                struct_lru_.push_front(batch_id);
                struct_cache_.emplace(
                    batch_id, StructCacheEntry{std::move(c), struct_lru_.begin(), sz});
                struct_bytes_ += sz;
            }
        }
    }
    return mini;
}

void BatchAssembler::set_struct_cache_budget_bytes(size_t budget_bytes) {
    std::lock_guard<std::mutex> lk(struct_mu_);
    struct_budget_ = budget_bytes;
    if (budget_bytes == 0) {
        struct_cache_.clear();
        struct_lru_.clear();
        struct_bytes_ = 0;
    }
}

BatchAssembler::StructCacheStats BatchAssembler::struct_cache_stats() const {
    std::lock_guard<std::mutex> lk(struct_mu_);
    StructCacheStats s;
    s.hits      = struct_hits_;
    s.misses    = struct_misses_;
    s.evictions = struct_evictions_;
    s.bytes     = struct_bytes_;
    s.budget    = struct_budget_;
    s.entries   = struct_cache_.size();
    return s;
}

// =============================================================================
// Public: assemble_from_sample(sample)
// =============================================================================

MiniBatch BatchAssembler::assemble_from_sample(const GraphSample& sample) {
    MiniBatch mini;
    mini.batch_id = sample.batch_id;
    mini.split    = sample.split;

    // Step 1: Build global index map (ObjectId -> position in all_unique_nodes)
    std::unordered_map<uint64_t, int64_t> oid_to_global;
    oid_to_global.reserve(sample.all_unique_nodes.size());
    for (int64_t i = 0; i < static_cast<int64_t>(sample.all_unique_nodes.size()); ++i) {
        oid_to_global[sample.all_unique_nodes[i].id] = i;
    }

    // Step 2: Build per-layer active-set indices for the active-set-shrinking
    // model refactor — see plan ~/Desktop/2026-05-14-graphsage-active-set-shrinking-plan.md.
    // Must precede build_edge_indices so the edge tensors can be remapped from
    // global positions into local positions within each active set.
    //
    // Round 2C (2026-05-15): build_active_indices now also emits a per-layer
    // ObjectId.id -> local-A_k-index hash table, which build_edge_indices
    // uses directly to halve the per-edge hash count (one lookup per
    // endpoint instead of two).
    auto t_active_start = std::chrono::steady_clock::now();
    auto active = build_active_indices(sample, oid_to_global);
    auto t_active_end = std::chrono::steady_clock::now();
    mini.active_indices_per_layer = std::move(active.indices_per_layer);
    mini.active_sizes_per_layer   = std::move(active.sizes_per_layer);
    mini.timing.active_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_active_end - t_active_start).count());

    // Step 3: Load features
    // Round 2B (2026-05-15): pass the already-deserialized sample so the
    // FourLevelStore path skips re-reading the same ~55 MB sample file.
    auto t_load_start = std::chrono::steady_clock::now();
    mini.features = load_features(sample);
    auto t_load_end = std::chrono::steady_clock::now();
    mini.timing.assembler_kernel_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_load_end - t_load_start).count());

    // STEP 6 (2026-05-31): capture the v2 addr-table dispatch result for THIS
    // batch right after the load, while feature_store_->last_used_addr_tables()
    // still refers to it. Carried on the MiniBatch so TrainingLoop reports
    // correct v2 telemetry on the async-prefetcher path (the worker stamps
    // here; the consumer reads after next()).
    if (feature_store_) {
        mini.timing.used_addr_tables = feature_store_->last_used_addr_tables();
        mini.timing.addr_load_ns     = feature_store_->last_addr_load_us() * 1000ULL;
    }

    // Step 4: Build edge indices per layer (LOCAL to active sets).
    auto t_edge_start = std::chrono::steady_clock::now();
    mini.edge_indices = build_edge_indices(sample, active);
    auto t_edge_end = std::chrono::steady_clock::now();
    mini.timing.edge_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_edge_end - t_edge_start).count());

    // Step 4: Gather labels for seed nodes (layer 0)
    int64_t num_seeds = static_cast<int64_t>(
        sample.nodes_per_layer.empty() ? 0 : sample.nodes_per_layer[0].size()
    );
    mini.num_seeds = static_cast<uint64_t>(num_seeds);
    mini.num_nodes = sample.all_unique_nodes.size();

    if (labels_ && num_seeds > 0) {
        std::vector<uint64_t> seed_row_indices;
        seed_row_indices.reserve(static_cast<size_t>(num_seeds));
        for (const auto& oid : sample.nodes_per_layer[0]) {
            auto row = row_mapping_.find(oid);
            if (row) {
                seed_row_indices.push_back(*row);
            } else {
                // Unknown node — use sentinel that LabelStore will treat as -1
                seed_row_indices.push_back(std::numeric_limits<uint64_t>::max());
            }
        }
        mini.labels     = labels_->gather(seed_row_indices);
        mini.label_mask = (mini.labels != -1);
    } else {
        mini.labels     = torch::zeros({num_seeds}, torch::kInt64);
        mini.label_mask = torch::zeros({num_seeds}, torch::kBool);
    }

    // Round 1E (2026-05-15): CPU-side count of labeled seeds. label_mask is
    // still on CPU at this point (device transfer happens in TrainingLoop),
    // so reading it costs only an O(num_seeds) memory scan and avoids the
    // per-batch `label_mask.any().item<bool>()` GPU→CPU sync.
    mini.num_labeled = 0;
    if (mini.label_mask.numel() > 0) {
        auto mask_acc = mini.label_mask.accessor<bool, 1>();
        for (int64_t i = 0; i < mini.label_mask.size(0); ++i) {
            if (mask_acc[i]) mini.num_labeled++;
        }
    }

    return mini;
}

// =============================================================================
// Private: build_active_indices
// =============================================================================

BatchAssembler::ActiveIndicesResult
BatchAssembler::build_active_indices(
    const GraphSample& sample,
    const std::unordered_map<uint64_t, int64_t>& oid_to_global)
{
    // K+1 active sets for K-layer sample: A_0 .. A_K.
    const size_t  K = sample.edges_per_layer.size();
    const int64_t N = static_cast<int64_t>(sample.all_unique_nodes.size());
    ActiveIndicesResult out;
    out.indices_per_layer.reserve(K + 1);
    out.sizes_per_layer.reserve(K + 1);

    // --- Phase 1: accumulate the cumulative active set, layer by layer. ---
    // By the rebuild_unique_nodes() ordering invariant (nodes inserted into
    // all_unique_nodes in layer order, seeds first), adding layer k's global
    // positions in iteration order yields a monotonically increasing sequence,
    // and across all layers the full set is exactly [0,1,..,N-1]. We dedup with
    // an O(1) `seen` bitmap indexed by global position (positions are in
    // [0,N)), which is far cheaper than the prior unordered_set/unordered_map
    // churn — that allocation traffic was the dominant per-batch CPU cost at
    // N=8 (profile 2026-06-04: build_active_indices ~= 29% of epoch wall).
    std::vector<char>    seen(static_cast<size_t>(N), 0);
    std::vector<int64_t> active_positions;
    active_positions.reserve(static_cast<size_t>(N));
    std::vector<int64_t> prefix_sizes;  // |A_k| after processing layer k
    prefix_sizes.reserve(K + 1);

    // Per-layer-entry global positions, recorded for free from the same
    // oid_to_global lookups below. layer_gpos[k][i] is the global position of
    // nodes_per_layer[k][i] (every entry, incl. cross-layer dups). On the fast
    // path this lets build_edge_indices remap edges with zero hash lookups.
    const size_t num_node_layers = sample.nodes_per_layer.size();
    std::vector<std::vector<int64_t>> layer_gpos(num_node_layers);

    for (size_t k = 0; k <= K; ++k) {
        if (k < num_node_layers) {
            layer_gpos[k].reserve(sample.nodes_per_layer[k].size());
            for (const auto& oid : sample.nodes_per_layer[k]) {
                auto it = oid_to_global.find(oid.id);
                if (it == oid_to_global.end()) {
                    throw std::runtime_error(
                        "BatchAssembler::build_active_indices: node at layer " +
                        std::to_string(k) + " not in all_unique_nodes (oid=" +
                        std::to_string(oid.id) + ")"
                    );
                }
                const int64_t pos = it->second;
                layer_gpos[k].push_back(pos);
                if (pos >= 0 && pos < N && !seen[static_cast<size_t>(pos)]) {
                    seen[static_cast<size_t>(pos)] = 1;
                    active_positions.push_back(pos);
                }
            }
        }
        prefix_sizes.push_back(static_cast<int64_t>(active_positions.size()));
    }

    // --- Phase 2: identity check. The universal case is active_positions ==
    // [0,1,..,M-1]; then local index == global position for every layer, so the
    // per-layer gather tensors are aranges and edges remap through oid_to_global
    // with no per-layer maps at all. ---
    bool identity = true;
    for (size_t i = 0; i < active_positions.size(); ++i) {
        if (active_positions[i] != static_cast<int64_t>(i)) { identity = false; break; }
    }
    out.identity_prefix = identity;

    if (identity) {
        for (size_t k = 0; k <= K; ++k) {
            const int64_t Mk = prefix_sizes[k];
            out.sizes_per_layer.push_back(Mk);
            out.indices_per_layer.push_back(torch::arange(Mk, torch::kInt64));
        }
        // oid_to_local_per_layer intentionally left empty (fast-path signal);
        // carry the precomputed per-layer global positions for build_edge_indices.
        out.layer_global_pos = std::move(layer_gpos);
        return out;
    }

    // --- Defensive fallback (never observed: only if a future sampler change
    // breaks the rebuild_unique_nodes ordering). Reproduce the legacy result
    // BYTE-IDENTICALLY: each A_k is the sorted prefix active_positions[0,M_k),
    // with a per-layer ObjectId.id -> local-index map. ---
    out.oid_to_local_per_layer.reserve(K + 1);
    for (size_t k = 0; k <= K; ++k) {
        const size_t Mk = static_cast<size_t>(prefix_sizes[k]);
        std::vector<int64_t> sorted_positions(active_positions.begin(),
                                              active_positions.begin() + Mk);
        std::sort(sorted_positions.begin(), sorted_positions.end());

        auto t = torch::empty({static_cast<int64_t>(Mk)}, torch::kInt64);
        if (Mk > 0) {
            std::memcpy(t.data_ptr<int64_t>(), sorted_positions.data(),
                        Mk * sizeof(int64_t));
        }
        out.sizes_per_layer.push_back(static_cast<int64_t>(Mk));
        out.indices_per_layer.push_back(std::move(t));

        std::unordered_map<uint64_t, int64_t> oid_to_local;
        oid_to_local.reserve(Mk);
        for (int64_t local_idx = 0; local_idx < static_cast<int64_t>(Mk); ++local_idx) {
            const int64_t global_pos = sorted_positions[static_cast<size_t>(local_idx)];
            oid_to_local[sample.all_unique_nodes[static_cast<size_t>(global_pos)].id] =
                local_idx;
        }
        out.oid_to_local_per_layer.push_back(std::move(oid_to_local));
    }

    return out;
}

// =============================================================================
// Private: build_edge_indices
// =============================================================================

std::vector<torch::Tensor> BatchAssembler::build_edge_indices(
    const GraphSample& sample,
    const ActiveIndicesResult& active)
{
    std::vector<torch::Tensor> result;
    const size_t num_layers = sample.edges_per_layer.size();
    result.reserve(num_layers);

    // Fast path (2026-06-04): empty per-layer maps => active sets are identity
    // prefixes (local index == global position). Each endpoint is remapped by
    // pure array indexing into active.layer_global_pos[layer][entry_idx]
    // (precomputed in build_active_indices from the lookups it already did) —
    // ZERO per-edge hash lookups. Fallback: legacy per-layer A_{k+1}/A_k maps.
    const bool fast = active.oid_to_local_per_layer.empty();
    const auto& lgp = active.layer_global_pos;

    // Nested aggregation (2026-06-02): conv k operates on dst set A_k =
    // ∪_{j<=k} nodes_per_layer[j] (all nodes within k hops). For STANDARD
    // GraphSAGE / DGL-block message passing every such node must aggregate its
    // sampled neighbours at conv k, so edge_index[k] is the union of the
    // per-hop edge sets E_0..E_k. Each per-hop set E_j (= edges_per_layer[j])
    // carries dst in layer j ⊆ A_k and src in layer j+1 ⊆ A_{k+1}, so both
    // endpoints resolve in the cumulative A_k / A_{k+1} maps.
    //
    // LEGACY (nested_aggregation_ == false): edge_index[k] = E_k only. A seed
    // (layer 0) then has edges solely in edge_index[0] and aggregates its
    // neighbourhood at just the final conv — a strictly weaker variant whose
    // deviation compounds with depth (see set_nested_aggregation docs).
    const bool nested = nested_aggregation_;

    for (size_t k = 0; k < num_layers; ++k) {
        // Fallback per-layer maps (unused on the fast path).
        const std::unordered_map<uint64_t, int64_t>* src_map =
            fast ? nullptr : &active.oid_to_local_per_layer[k + 1];
        const std::unordered_map<uint64_t, int64_t>* dst_map =
            fast ? nullptr : &active.oid_to_local_per_layer[k];

        const size_t first_j = nested ? 0 : k;  // nested: E_0..E_k; legacy: E_k only

        int64_t E_total = 0;
        for (size_t j = first_j; j <= k; ++j) {
            E_total += static_cast<int64_t>(sample.edges_per_layer[j].size());
        }

        auto edge_index = torch::empty({2, E_total}, torch::kInt64);
        auto acc = edge_index.accessor<int64_t, 2>();

        int64_t out = 0;
        for (size_t j = first_j; j <= k; ++j) {
            const LayerEdges& edges = sample.edges_per_layer[j];
            const int64_t Ej = static_cast<int64_t>(edges.size());
            for (int64_t i = 0; i < Ej; ++i) {
                const size_t src_idx = static_cast<size_t>(edges.src_indices[i]);
                const size_t dst_idx = static_cast<size_t>(edges.dst_indices[i]);

                int64_t src_local, dst_local;
                if (fast) {
                    // Remap by precomputed global position (== local index).
                    // Debug-only bounds check on the disk-sourced edge endpoints
                    // (zero-cost in Release where NDEBUG disables assert). A
                    // stale/corrupt sample whose edge indices exceed the layer
                    // node counts would otherwise index lgp out of bounds (UB);
                    // the legacy fallback below throws on the same condition.
                    assert(j + 1 < lgp.size() && src_idx < lgp[j + 1].size() &&
                           j < lgp.size() && dst_idx < lgp[j].size() &&
                           "edge endpoint index out of active-set bounds");
                    src_local = lgp[j + 1][src_idx];  // src in A_{k+1}
                    dst_local = lgp[j][dst_idx];       // dst in A_k
                } else {
                    // E_j edges are layer-local to (layer j+1 src, layer j dst).
                    const ObjectId src_oid = sample.nodes_per_layer[j + 1][src_idx];
                    const ObjectId dst_oid = sample.nodes_per_layer[j][dst_idx];
                    auto src_it = src_map->find(src_oid.id);
                    auto dst_it = dst_map->find(dst_oid.id);
                    if (src_it == src_map->end()) {
                        throw std::runtime_error(
                            "BatchAssembler::build_edge_indices: src node not in A_" +
                            std::to_string(k + 1) + " active set (conv " +
                            std::to_string(k) + ", hop " + std::to_string(j) +
                            ", edge " + std::to_string(i) + ")"
                        );
                    }
                    if (dst_it == dst_map->end()) {
                        throw std::runtime_error(
                            "BatchAssembler::build_edge_indices: dst node not in A_" +
                            std::to_string(k) + " active set (conv " +
                            std::to_string(k) + ", hop " + std::to_string(j) +
                            ", edge " + std::to_string(i) + ")"
                        );
                    }
                    src_local = src_it->second;
                    dst_local = dst_it->second;
                }

                acc[0][out] = src_local;  // src local in A_{k+1}
                acc[1][out] = dst_local;  // dst local in A_k
                ++out;
            }
        }
        result.push_back(std::move(edge_index));
    }

    return result;
}

// =============================================================================
// Private: load_features
// =============================================================================

torch::Tensor BatchAssembler::load_features(const GraphSample& sample) {
    if (feature_store_) {
        // Full mode: FourLevelStore handles all four tiers.
        // Round 2B (2026-05-15): pass the GraphSample directly so the store
        // does not re-read it from disk inside load_batch_features.
        return feature_store_->load_batch_features(sample);
    }

    // Fallback mode: FeatureMatrix + RowMapping
    const auto& unique_nodes = sample.all_unique_nodes;
    std::vector<uint64_t> row_ids;
    row_ids.reserve(unique_nodes.size());
    for (const auto& oid : unique_nodes) {
        auto row = row_mapping_.find(oid);
        if (!row) {
            throw std::runtime_error(
                "BatchAssembler: node not in RowMapping: " + std::to_string(oid.id)
            );
        }
        row_ids.push_back(*row);
    }

    const size_t N          = row_ids.size();
    const size_t D          = feature_matrix_->num_cols();
    const size_t elem_bytes = feature_matrix_->row_bytes();  // D * sizeof(element)

    std::vector<char> buffer(N * elem_bytes);
    feature_matrix_->extract_rows(row_ids, buffer.data());

    // Determine LibTorch scalar type from stored GnnDtype
    torch::ScalarType scalar_type = torch::kFloat32;
    switch (feature_matrix_->dtype()) {
        case GnnDtype::FLOAT32: scalar_type = torch::kFloat32; break;
        case GnnDtype::FLOAT64: scalar_type = torch::kFloat64; break;
        case GnnDtype::INT32:   scalar_type = torch::kInt32;   break;
        case GnnDtype::INT64:   scalar_type = torch::kInt64;   break;
        case GnnDtype::UINT8:   scalar_type = torch::kUInt8;   break;
        case GnnDtype::BOOL:    scalar_type = torch::kBool;    break;
        default:
            throw std::runtime_error(
                "BatchAssembler: unsupported GnnDtype in FeatureMatrix fallback"
            );
    }

    // .clone() is required because buffer lives on the stack
    return torch::from_blob(
        buffer.data(),
        {static_cast<int64_t>(N), static_cast<int64_t>(D)},
        scalar_type
    ).clone();
}

} // namespace mdb::gnn
