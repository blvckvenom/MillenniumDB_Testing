#include "gnn/training/batch_assembler.h"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "gnn/sampling/sample_fingerprint.h"
#include "gnn/storage/block_store.h"

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
    init_blocks_();
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
    init_blocks_();
}

// =============================================================================
// Private: init_blocks_ — auto-detect baked blocks/ at construction
// =============================================================================

void BatchAssembler::init_blocks_() {
    std::error_code ec;
    blocks_dir_ = samples_.get_path() / "blocks";
    // Detected here; consumed only when also !nested_aggregation_ at use time.
    use_blocks_ = std::filesystem::exists(blocks_dir_, ec);
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

    // ------------------------------------------------------------------
    // Task 7: baked computation-graph block fast path.
    //
    // When a fresh per-batch block (baked offline in gnn_build_feature_store)
    // exists and matches this sample's FULL content hash, consume its
    // active_sizes + edge_indices and SKIP the online build_active_indices +
    // build_edge_indices work entirely. The block is keyed by the full
    // compute_batch_content_hash (which includes edges); we keep reading the
    // full edge-bearing sample, so the train side recomputes that exact hash
    // to verify the block per-batch, and any stale/missing block falls back
    // to the online build directly from the same sample (no re-read).
    //
    // Gated on !nested_aggregation_ (re-read at use time): blocks are baked
    // with the legacy per-hop wiring, so a late set_nested_aggregation(true)
    // must disable block consumption. NO skip_edges is requested (deferred):
    // the full sample is read so the hash + fallback both work directly.
    // ------------------------------------------------------------------
    bool used_block = false;
    if (use_blocks_ && !nested_aggregation_) {
        auto blk_path = block_filename(blocks_dir_, sample.batch_id);
        // Full content hash (sample has edges) — matches the bake's full hash.
        uint64_t fp = mdb::gnn::compute_batch_content_hash(sample);
        auto blk = BlockReader::open(blk_path, fp);
        if (blk) {
            mini.active_sizes_per_layer = blk->active_sizes;
            // Reconstruct the identity-prefix active gather indices: A_k is the
            // prefix [0, M_k) of all_unique_nodes (rebuild_unique_nodes inserts
            // nodes in layer order), so active_indices_per_layer[k] == arange(M_k).
            mini.active_indices_per_layer.clear();
            mini.active_indices_per_layer.reserve(blk->active_sizes.size());
            for (int64_t m : blk->active_sizes) {
                mini.active_indices_per_layer.push_back(torch::arange(m, torch::kInt64));
            }
            mini.edge_indices     = blk->edge_indices;
            mini.timing.active_ns = 0;   // baked offline
            mini.timing.edge_ns   = 0;   // baked offline
#ifndef NDEBUG
            // Identity-prefix bounds: for conv layer k, edge_index[k] row 0
            // (src, in A_{k+1}) must be < M_{k+1}; row 1 (dst, in A_k) < M_k.
            for (size_t k = 0;
                 k < mini.edge_indices.size()
                     && (k + 1) < mini.active_sizes_per_layer.size();
                 ++k) {
                const auto& e = mini.edge_indices[k];
                if (e.numel() > 0) {
                    assert(e.select(0, 0).max().item<int64_t>()
                           < mini.active_sizes_per_layer[k + 1]);
                    assert(e.select(0, 1).max().item<int64_t>()
                           < mini.active_sizes_per_layer[k]);
                }
            }
#endif
            used_block = true;
        } else if (!block_fallback_warned_.exchange(true)) {
            std::cerr << "[BatchAssembler] block missing/stale for batch "
                      << sample.batch_id << " (blocks_dir=" << blocks_dir_
                      << ") — falling back to online build for this and any "
                         "other stale batches.\n";
        }
    }

    // `active` carries the oid_to_local / layer_global_pos / identity_prefix
    // fields build_edge_indices needs; only populated on the online path.
    ActiveIndicesResult active;

    if (!used_block) {
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
        active = build_active_indices(sample, oid_to_global);
        auto t_active_end = std::chrono::steady_clock::now();
        // NOTE: only indices_per_layer + sizes_per_layer are moved into mini;
        // build_edge_indices uses active's OTHER fields (oid_to_local_per_layer,
        // layer_global_pos, identity_prefix), which are NOT moved — preserved.
        mini.active_indices_per_layer = std::move(active.indices_per_layer);
        mini.active_sizes_per_layer   = std::move(active.sizes_per_layer);
        mini.timing.active_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                t_active_end - t_active_start).count());
    }

    // Step 3: Load features — ALWAYS (both block and online paths need them).
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

    if (!used_block) {
        // Step 4: Build edge indices per layer (LOCAL to active sets).
        auto t_edge_start = std::chrono::steady_clock::now();
        mini.edge_indices = build_edge_indices(sample, active);
        auto t_edge_end = std::chrono::steady_clock::now();
        mini.timing.edge_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                t_edge_end - t_edge_start).count());
    }

    // Step 5: Gather labels for seed nodes (layer 0)
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
    return graph_block::build_active_indices(sample, oid_to_global);
}

// =============================================================================
// Private: build_edge_indices
// =============================================================================

std::vector<torch::Tensor> BatchAssembler::build_edge_indices(
    const GraphSample& sample,
    const ActiveIndicesResult& active)
{
    return graph_block::build_edge_indices(sample, active, nested_aggregation_);
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
