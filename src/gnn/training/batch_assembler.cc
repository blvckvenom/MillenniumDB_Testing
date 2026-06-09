#include "gnn/training/batch_assembler.h"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include "gnn/sampling/sample_fingerprint.h"
#include "gnn/storage/block_store.h"
#include "gnn/storage/direct_io_reader.h"

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
    // Read the two SC-5 env toggles once (same as before SC-5a) and delegate the
    // actual mode selection to apply_block_mode_, which is shared with the
    // per-call override setter set_block_mode_override(). With neither env var
    // set (the default), the booleans are both false and the selected flags are
    // byte-identical to the pre-SC-5a auto-detection.
    //   MDB_GNN_NO_BLOCKS=1        -> force fully-online (no blocks at all).
    //   MDB_GNN_NO_SELF_CONTAINED=1-> force Option-A per-batch blocks (still
    //                                 reads batches.dat in assemble_from_sample).
    const char* nb  = std::getenv("MDB_GNN_NO_BLOCKS");
    const char* nsc = std::getenv("MDB_GNN_NO_SELF_CONTAINED");
    const bool env_no_blocks        = (nb  && std::string(nb)  == "1");
    const bool env_no_self_contained = (nsc && std::string(nsc) == "1");
    const char* npf = std::getenv("MDB_GNN_NO_PACKED_FULL");
    const bool env_no_packed_full = (npf && std::string(npf) == "1");
    apply_block_mode_(env_no_blocks, env_no_self_contained, env_no_packed_full);
}

// =============================================================================
// Private: apply_block_mode_ — shared mode selection (env path + override path)
// =============================================================================

void BatchAssembler::apply_block_mode_(bool no_blocks, bool no_self_contained, bool no_packed_full) {
    std::error_code ec;
    blocks_dir_ = samples_.get_path() / "blocks";
    // Detected here; consumed only when also !nested_aggregation_ at use time.
    use_blocks_ = std::filesystem::exists(blocks_dir_, ec);

    // ------------------------------------------------------------------
    // SC-3: decide self-contained-block train mode (skip batches.dat).
    //
    // Eligible only when baked blocks exist, aggregation is the legacy per-hop
    // wiring the blocks were baked with, and features route through the
    // FourLevelStore (the v2 addr_table path — the ONLY gather safe with a
    // placeholder all_unique_nodes). We additionally verify, header-only:
    //   - the sample catalog carries a non-UNKNOWN content fingerprint,
    //   - an addr_tables/ directory is present (the v2 path needs it), and
    //   - batch 0's block is self-contained with store_fp == catalog fingerprint.
    // Any mismatch leaves self_contained_mode_ false -> assemble() takes the
    // existing real-sample path, byte-identical to today.
    //
    // The two booleans drive the SC-5 same-session A/B (from env vars in
    // init_blocks_, or from the per-call set_block_mode_override() in SC-5a):
    //   no_blocks==true        -> force fully-online (no blocks at all).
    //   no_self_contained==true-> force Option-A per-batch blocks (still
    //                             reads batches.dat in assemble_from_sample).
    // CAVEAT: both toggles only take effect INSIDE the `feature_store_ != nullptr`
    // eligibility block below, so they affect ONLY the FourLevelStore ctor path
    // (the measured config). On the FeatureMatrix-fallback ctor (feature_store_
    // == nullptr) the block is skipped entirely, so no_blocks does NOT disable
    // Option-A block consumption there — that path is not a measured
    // configuration and self-contained mode never applies to it anyway.
    // ------------------------------------------------------------------
    self_contained_mode_ = false;
    store_fp_            = 0;
    packed_full_mode_    = false;
    packed_full_.reset();
    const char* sc_mode_label = "online";
    if (use_blocks_) sc_mode_label = "Option-A blocks";

    if (use_blocks_ && !nested_aggregation_ && feature_store_ != nullptr) {
        if (no_blocks) {
            use_blocks_   = false;          // fully-online
            sc_mode_label = "online";
        } else if (!no_self_contained) {
            uint64_t catalog_fp = samples_.get_catalog().sample_content_fp;
            bool addr_present =
                std::filesystem::exists(samples_.get_path() / "addr_tables", ec);
            // Packed-full probe: features come from the consolidated pack (NO
            // addr_tables needed). Keyed by the MIXED feature-store fingerprint
            // (must match build_packed_full_'s cur_fp).
            std::optional<PackedFullReader> pf;
            if (!no_packed_full && catalog_fp != 0) {
                uint64_t pf_fp = mix_feature_store_fingerprint(
                    catalog_fp, "node_features",
                    feature_store_->feature_dim(),
                    static_cast<uint8_t>(feature_store_->dtype()));
                pf = PackedFullReader::open(samples_.get_path() / "packed_full", pf_fp);
            }
            if (catalog_fp != 0 && (addr_present || pf.has_value())) {
                // Header-only peek of batch 0's block store fingerprint.
                uint64_t b0 = BlockReader::read_store_fp(
                    block_filename(blocks_dir_, 0));
                if (b0 != 0 && b0 == catalog_fp) {
                    self_contained_mode_ = true;
                    store_fp_            = catalog_fp;
                    if (pf.has_value()) {
                        packed_full_      = std::move(pf);
                        packed_full_mode_ = true;
                        sc_mode_label     = "self-contained+packed-full";
                    } else {
                        sc_mode_label     = "self-contained";
                    }
                }
            }
        }
        // else no_self_contained: stays Option-A (use_blocks_ true,
        // self_contained_mode_ false) -> assemble_from_sample consumes per-batch
        // blocks but still reads batches.dat.
    }

    std::cerr << "[BatchAssembler] feature-load mode: " << sc_mode_label << "\n";
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
    // SC-3: self-contained fast path — read ONLY the baked block, never
    // batches.dat. Returns true on success; false requests the legacy
    // real-sample path below (always correct). See try_assemble_self_contained_.
    if (self_contained_mode_) {
        MiniBatch m;
        if (try_assemble_self_contained_(batch_id, m)) return m;
        // else: fall through to the unchanged real-sample path.
    }

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
        bool used_v2 = false;
        mini.features = load_features(sample, &used_v2);
        auto tl1 = std::chrono::steady_clock::now();
        mini.timing.sample_read_ns      = sample_read_ns;
        mini.timing.assembler_kernel_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tl1 - tl0).count());
        if (feature_store_) {
            mini.timing.used_addr_tables = used_v2;
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
                c.split                    = static_cast<uint32_t>(mini.split);
                struct_lru_.push_front(batch_id);
                struct_cache_.emplace(
                    batch_id, StructCacheEntry{std::move(c), struct_lru_.begin(), sz});
                struct_bytes_ += sz;
            }
        }
    }
    return mini;
}

// =============================================================================
// Private: try_assemble_self_contained_ (SC-3 fast path — never reads batches.dat)
// =============================================================================

bool BatchAssembler::load_self_contained_features_(uint64_t batch_id,
                                                   const GraphSample& ms,
                                                   MiniBatch& mini) {
    auto tl0 = std::chrono::steady_clock::now();
    if (packed_full_mode_) {
        const auto e = packed_full_->entry(batch_id);
        // Per-thread O_DIRECT reader: prefetch workers call assemble()
        // concurrently and DirectIoReader is not thread-safe across its io_uring
        // rings, so each worker thread owns one reader (amortizes io_uring setup
        // across batches; keyed by (path, store_fp) so a same-process rebuild of
        // the pack — which keeps the path but changes the size/fingerprint —
        // rebuilds the reader instead of serving a stale cached file_size_).
        static thread_local std::unique_ptr<DirectIoReader> tl_pf_reader;
        static thread_local std::string tl_pf_path;
        static thread_local uint64_t tl_pf_fp = 0;
        const std::string dat = packed_full_->dat_path().string();
        const uint64_t   fp  = packed_full_->header().store_fp;
        if (!tl_pf_reader || tl_pf_path != dat || tl_pf_fp != fp) {
            tl_pf_reader = std::make_unique<DirectIoReader>(packed_full_->dat_path());
            tl_pf_path   = dat;
            tl_pf_fp     = fp;
        }
        auto res = tl_pf_reader->read_range(e.offset, e.length);
        torch::ScalarType st = torch::kFloat32;
        switch (static_cast<GnnDtype>(packed_full_->dtype())) {
            case GnnDtype::FLOAT32: st = torch::kFloat32; break;
            case GnnDtype::FLOAT64: st = torch::kFloat64; break;
            case GnnDtype::INT32:   st = torch::kInt32;   break;
            case GnnDtype::INT64:   st = torch::kInt64;   break;
            case GnnDtype::UINT8:   st = torch::kUInt8;   break;
            case GnnDtype::BOOL:    st = torch::kBool;    break;
            default:
                throw std::runtime_error(
                    "BatchAssembler packed-full: unsupported GnnDtype");
        }
        mini.features = torch::from_blob(
            res.data.get(),
            {static_cast<int64_t>(e.num_nodes),
             static_cast<int64_t>(packed_full_->feature_dim())},
            torch::TensorOptions().dtype(st)
        ).clone().to(feature_store_->feature_device());
        auto tl1 = std::chrono::steady_clock::now();
        mini.timing.sample_read_ns      = 0;
        mini.timing.assembler_kernel_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tl1 - tl0).count());
        mini.timing.used_addr_tables    = false;
        mini.timing.addr_load_ns        = 0;
        return true;
    }
    // PER-CALL v2 dispatch outcome — the safety-net check below must NOT read
    // the store's shared last_used_addr_tables() flag: under N>1 prefetch
    // workers another worker's v2 success could overwrite it between our load
    // and the check, masking a legacy fallback over the PLACEHOLDER sample
    // (zero node contents -> silent wrong features).
    bool used_v2 = false;
    mini.features = load_features(ms, &used_v2);
    auto tl1 = std::chrono::steady_clock::now();
    if (!used_v2) {
        if (!self_contained_fallback_warned_.exchange(true)) {
            std::cerr << "[BatchAssembler] self-contained: v2 addr_table did not "
                         "serve batch " << batch_id
                      << " — falling back to real sample read for this and any "
                         "other such batches.\n";
        }
        return false;
    }
    mini.timing.sample_read_ns      = 0;
    mini.timing.assembler_kernel_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tl1 - tl0).count());
    mini.timing.used_addr_tables = true;
    mini.timing.addr_load_ns     = feature_store_->last_addr_load_us() * 1000ULL;
    return true;
}

bool BatchAssembler::try_assemble_self_contained_(uint64_t batch_id, MiniBatch& mini) {
    // ---- Fast path A: structural-cache hit (cross-epoch) ----------------
    // The structural bundle (edges/active/labels) is identical every epoch, so
    // on a hit we reuse it and only re-run the (cheap) v2 feature gather. We
    // still build a minimal placeholder sample for that gather.
    {
        bool hit = false;
        CachedStruct c;
        if (struct_budget_ > 0) {
            std::lock_guard<std::mutex> lk(struct_mu_);
            auto it = struct_cache_.find(batch_id);
            if (it != struct_cache_.end()) {
                ++struct_hits_;
                struct_lru_.splice(struct_lru_.begin(), struct_lru_, it->second.lru_it);
                c   = it->second.s;   // refcounted tensors — shares storage
                hit = true;
            }
            // NOTE: we DON'T ++struct_misses_ here; the miss is accounted once
            // below where the block is actually opened (parity with assemble()'s
            // single miss-count per batch).
        }

        if (hit) {
            mini.batch_id                 = batch_id;
            mini.split                    = static_cast<SplitType>(c.split);
            mini.edge_indices             = c.edge_indices;
            mini.active_indices_per_layer = c.active_indices_per_layer;
            mini.active_sizes_per_layer   = c.active_sizes_per_layer;
            mini.labels                   = c.labels;
            mini.label_mask               = c.label_mask;
            mini.num_seeds                = c.num_seeds;
            mini.num_nodes                = c.num_nodes;
            mini.num_labeled              = c.num_labeled;

            // Minimal placeholder sample for the v2 gather: only the SIZE of
            // all_unique_nodes and batch_id are consulted by the addr_table
            // path (verified: load_batch_features_v2_ never reads node contents).
            GraphSample ms;
            ms.batch_id = batch_id;
            ms.split    = mini.split;
            ms.all_unique_nodes.assign(static_cast<size_t>(c.num_nodes), ObjectId());

            if (!load_self_contained_features_(batch_id, ms, mini)) return false;
            return true;
        }
    }

    // ---- Fast path B: structural-cache miss — build from the block ------
    if (struct_budget_ > 0) {
        std::lock_guard<std::mutex> lk(struct_mu_);
        ++struct_misses_;
    }

    auto blk = BlockReader::open_self_contained(
        block_filename(blocks_dir_, batch_id), store_fp_);
    if (!blk) {
        if (!self_contained_fallback_warned_.exchange(true)) {
            std::cerr << "[BatchAssembler] self-contained: block missing/stale "
                         "(store_fp mismatch) for batch " << batch_id
                      << " (blocks_dir=" << blocks_dir_
                      << ") — falling back to real sample read for this and any "
                         "other such batches.\n";
        }
        return false;
    }

    mini.batch_id = batch_id;
    mini.split    = static_cast<SplitType>(blk->split);

    // Minimal sample: only the count + batch_id + split feed the v2 gather; the
    // seeds (nodes_per_layer[0]) feed the label gather below. Edges come from
    // the block, NOT from the sample (edges_per_layer left empty).
    GraphSample ms;
    ms.batch_id = batch_id;
    ms.split    = mini.split;
    ms.all_unique_nodes.assign(static_cast<size_t>(blk->num_unique_nodes), ObjectId());
    ms.nodes_per_layer.resize(1);
    ms.nodes_per_layer[0].reserve(blk->seed_ids.size());
    for (uint64_t id : blk->seed_ids) ms.nodes_per_layer[0].emplace_back(id);

    // Graph from the block — identity-prefix active gather indices (A_k is the
    // prefix [0, M_k) of all_unique_nodes), baked edges (already int64).
    mini.active_sizes_per_layer = blk->active_sizes;
    mini.active_indices_per_layer.clear();
    mini.active_indices_per_layer.reserve(blk->active_sizes.size());
    for (int64_t m : blk->active_sizes) {
        mini.active_indices_per_layer.push_back(torch::arange(m, torch::kInt64));
    }
    mini.edge_indices     = blk->edge_indices;
    mini.timing.active_ns = 0;   // baked offline
    mini.timing.edge_ns   = 0;   // baked offline
#ifndef NDEBUG
    // Identity-prefix bounds (same invariant as assemble_from_sample's block path).
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

    // Labels for seed nodes (layer 0) — identical logic to assemble_from_sample.
    int64_t num_seeds = static_cast<int64_t>(ms.nodes_per_layer[0].size());
    mini.num_seeds = static_cast<uint64_t>(num_seeds);
    mini.num_nodes = blk->num_unique_nodes;

    if (labels_ && num_seeds > 0) {
        std::vector<uint64_t> seed_row_indices;
        seed_row_indices.reserve(static_cast<size_t>(num_seeds));
        for (const auto& oid : ms.nodes_per_layer[0]) {
            auto row = row_mapping_.find(oid);
            if (row) {
                seed_row_indices.push_back(*row);
            } else {
                seed_row_indices.push_back(std::numeric_limits<uint64_t>::max());
            }
        }
        mini.labels     = labels_->gather(seed_row_indices);
        mini.label_mask = (mini.labels != -1);
    } else {
        mini.labels     = torch::zeros({num_seeds}, torch::kInt64);
        mini.label_mask = torch::zeros({num_seeds}, torch::kBool);
    }

    mini.num_labeled = 0;
    if (mini.label_mask.numel() > 0) {
        auto mask_acc = mini.label_mask.accessor<bool, 1>();
        for (int64_t i = 0; i < mini.label_mask.size(0); ++i) {
            if (mask_acc[i]) mini.num_labeled++;
        }
    }

    // Features via packed-full (one O_DIRECT read) or the v2 (addr_table) path
    // on the placeholder sample.
    if (!load_self_contained_features_(batch_id, ms, mini)) return false;

    // Cache the structural bundle (same budget/LRU dance as assemble()'s miss).
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
                c.split                    = static_cast<uint32_t>(mini.split);
                struct_lru_.push_front(batch_id);
                struct_cache_.emplace(
                    batch_id, StructCacheEntry{std::move(c), struct_lru_.begin(), sz});
                struct_bytes_ += sz;
            }
        }
    }

    return true;
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
    bool used_v2 = false;
    mini.features = load_features(sample, &used_v2);
    auto t_load_end = std::chrono::steady_clock::now();
    mini.timing.assembler_kernel_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_load_end - t_load_start).count());

    // STEP 6 (2026-05-31): capture the v2 addr-table dispatch result for THIS
    // batch, as reported per-call by load_features (the store's shared
    // last_used_addr_tables() flag can be overwritten by a concurrent prefetch
    // worker before we read it). Carried on the MiniBatch so TrainingLoop
    // reports correct v2 telemetry on the async-prefetcher path (the worker
    // stamps here; the consumer reads after next()).
    if (feature_store_) {
        mini.timing.used_addr_tables = used_v2;
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

torch::Tensor BatchAssembler::load_features(const GraphSample& sample,
                                            bool* used_addr_tables) {
    if (used_addr_tables) *used_addr_tables = false;
    if (feature_store_) {
        // Full mode: FourLevelStore handles all four tiers.
        // Round 2B (2026-05-15): pass the GraphSample directly so the store
        // does not re-read it from disk inside load_batch_features. The store
        // reports the per-call v2 dispatch outcome through the out-param.
        return feature_store_->load_batch_features(sample, used_addr_tables);
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
