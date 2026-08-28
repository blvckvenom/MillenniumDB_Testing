#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "graph_models/object_id.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/four_level_store.h"
#include "gnn/storage/packed_full_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/graph_block_builder.h"
#include "gnn/training/label_store.h"
#include "gnn/training/mini_batch.h"
#include "gnn/training/split_store.h"

namespace mdb::gnn {

/**
 * @brief Assembles MiniBatch objects from four data sources for the training loop.
 *
 * Supports two feature-loading modes:
 *   - Full mode: FourLevelStore (L1 GPU, L2 CPU, L3 mmap, L4 disk)
 *   - Fallback mode: FeatureMatrix directly via RowMapping
 *
 * Labels and splits are optional (nullptr = unsupervised / no-split).
 *
 * Usage:
 * @code
 *   BatchAssembler assembler(feature_matrix, samples, &labels, &splits, row_mapping);
 *   MiniBatch batch = assembler.assemble(batch_id);
 *   // or equivalently:
 *   GraphSample sample = samples.read_sample(batch_id);
 *   MiniBatch batch = assembler.assemble_from_sample(sample);
 * @endcode
 */
class BatchAssembler {
public:
    /**
     * @brief Full mode: FourLevelStore for features.
     *
     * @param feature_store FourLevelStore that owns all four tiers
     * @param samples       Pre-computed GraphSamples storage
     * @param labels        Optional label store (nullptr = unsupervised)
     * @param splits        Optional split store (nullptr = no split)
     * @param row_mapping   ObjectId <-> row_index mapping (used for label lookup)
     * @param feature_name  Name the feature store was built/opened with. Keys
     *                      the packed-full probe via
     *                      mix_feature_store_fingerprint — FourLevelStore::build
     *                      mixes the ACTUAL feature name into the pack
     *                      fingerprint, so a mismatched name here makes a valid
     *                      pack look stale and silently skips it. Callers
     *                      serving a non-default feature name MUST pass it.
     */
    BatchAssembler(
        FourLevelStore& feature_store,
        SampleStorage&  samples,
        LabelStore*     labels,
        SplitStore*     splits,
        const RowMapping& row_mapping,
        const std::string& feature_name = "node_features"
    );

    /**
     * @brief Fallback mode: FeatureMatrix directly (no FourLevelStore).
     *
     * @param feature_matrix Flat [N, D] feature matrix opened from disk
     * @param samples        Pre-computed GraphSamples storage
     * @param labels         Optional label store (nullptr = unsupervised)
     * @param splits         Optional split store (nullptr = no split)
     * @param row_mapping    ObjectId <-> row_index mapping
     */
    BatchAssembler(
        const FeatureMatrix& feature_matrix,
        SampleStorage&       samples,
        LabelStore*          labels,
        SplitStore*          splits,
        const RowMapping&    row_mapping
    );

    /**
     * @brief Assemble a MiniBatch from a stored sample identified by batch_id.
     *
     * Reads the GraphSample from storage, then delegates to assemble_from_sample().
     *
     * @param batch_id  Batch identifier to read
     * @return MiniBatch ready for the training loop
     */
    MiniBatch assemble(uint64_t batch_id);

    /**
     * @brief Assemble a MiniBatch directly from a GraphSample.
     *
     * Public method (rather than private) to facilitate testing without
     * requiring a fully-initialized SampleStorage on disk.
     *
     * @param sample Pre-populated GraphSample
     * @return MiniBatch ready for the training loop
     */
    MiniBatch assemble_from_sample(const GraphSample& sample);

    /**
     * @brief Whether this BatchAssembler routes feature lookups through a
     *        FourLevelStore (full mode) vs a flat FeatureMatrix fallback.
     *
     * Used by the multi-worker AsyncBatchPrefetcher to refuse concurrent
     * execution against FourLevelStore: that store currently has shared
     * mutable state (DirectIoReader io_uring rings + pinned host buffer
     * reused across calls) that races silently under N>1 workers.
     */
    bool uses_feature_store() const { return feature_store_ != nullptr; }

    /**
     * @brief Prepare the FourLevelStore for N concurrent prefetch workers by
     *        allocating per-worker DirectIoReader instances and pinned host
     *        buffers. No-op in FeatureMatrix-fallback mode. MUST be called
     *        once before an N>1 AsyncBatchPrefetcher is constructed. May throw
     *        std::runtime_error if per-worker O_DIRECT readers cannot be
     *        opened — TrainingLoop catches and falls back to a single worker.
     */
    void prepare_feature_store_workers(unsigned num_workers) {
        if (feature_store_) feature_store_->prepare_worker_io(num_workers);
    }

    /**
     * @brief Read-only access to the underlying FourLevelStore, if any.
     *
     * Returns nullptr in FeatureMatrix-fallback mode. Used by TrainingLoop's
     * per-batch profile instrumentation (`BatchTimingLog`) to pull per-tier
     * timings off the store after each batch assemble. Const-correct: the
     * timings are advertised via const accessors and the store is not
     * mutated through this pointer.
     */
    const FourLevelStore* feature_store() const { return feature_store_; }

    // =========================================================================
    // Structural per-batch cache (train hot path)
    // =========================================================================

    /**
     * @brief Cache the per-batch STRUCTURAL build (edge indices, active-set
     *        indices, labels/mask) in RAM, bounded by a byte budget.
     *
     * The cost model showed build_active_indices + build_edge_indices dominate
     * the assemble worker cost (the feature gather itself is cheap when nodes
     * are L1-resident). These structural tensors are a pure function of the
     * GraphSample content, so they are identical every epoch. With a budget
     * set, assemble(batch_id) reuses them across epochs and only re-runs the
     * (cheap) feature load. int64 index tensors are small and read-only, so
     * this is safe at scale; an LRU bounds it when batches don't all fit.
     *
     * @param budget_bytes Max bytes of cached structural tensors. 0 disables.
     */
    void set_struct_cache_budget_bytes(size_t budget_bytes);

    struct StructCacheStats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        size_t   bytes = 0;
        size_t   budget = 0;
        size_t   entries = 0;
    };
    StructCacheStats struct_cache_stats() const;

    // =========================================================================
    // Nested (DGL-style) aggregation toggle
    // =========================================================================

    /**
     * @brief Select between the legacy per-hop edge wiring and the nested
     *        (DGL-block / Hamilton Alg.2) wiring in build_edge_indices.
     *
     * LEGACY (false, the historical default): edge_index[k] connects ONLY the
     * k-th hop frontier (nodes_per_layer[k]) to its sampled neighbours. A seed
     * therefore aggregates its 1-hop neighbourhood at only the final conv, and
     * the function the model computes is a strictly weaker variant of GraphSAGE
     * whose deviation COMPOUNDS WITH DEPTH (fine at 2 layers, degrades at >=3).
     *
     * NESTED (true): edge_index[k] = union of the per-hop edge sets E_0..E_k,
     * so EVERY node within k hops (incl. the seeds) re-aggregates its sampled
     * neighbours at conv k — the standard nested-neighbourhood message passing.
     * The active sets A_k (cumulative unions) already satisfy the prefix
     * invariant the model needs; only the edges change. Reuses the existing
     * sample on disk (the per-hop edges E_j are all serialised) and the feature
     * store unchanged.
     *
     * Default is read once from env MDB_GNN_NESTED_AGG (1/true/yes => nested)
     * in the constructor; this setter overrides it (used by unit tests).
     * NOTE: a model trained with nested=ON must be inferred with nested=ON.
     */
    void set_nested_aggregation(bool on) { nested_aggregation_ = on; }
    bool nested_aggregation() const { return nested_aggregation_; }

    // =========================================================================
    // Baked computation-graph blocks (Task 7) — test seam
    // =========================================================================

    /**
     * @brief Test-only override of the baked-blocks directory.
     *
     * In production both constructors auto-detect `<sample_dir>/blocks` via
     * init_blocks_(). This setter lets a unit test point the assembler at a
     * temporary blocks/ dir it baked itself (the production path is never
     * exercised by tests because the on-disk SampleStorage they construct has
     * no blocks/ subdir). Sets use_blocks_ to whether @p dir exists.
     */
    void set_blocks_dir_for_test(const std::filesystem::path& dir) {
        std::error_code ec;
        blocks_dir_ = dir;
        use_blocks_ = std::filesystem::exists(dir, ec);
    }

    /**
     * @brief Per-gnn_train-call override of the block-consumption mode.
     *
     * Re-runs the SAME eligibility logic as the constructor's env/auto
     * detection (apply_block_mode_), but with the two booleans supplied by the
     * caller instead of the MDB_GNN_NO_BLOCKS / MDB_GNN_NO_SELF_CONTAINED env
     * vars. This lets a same-session A/B/C measurement pick online / Option-A /
     * self-contained without restarting the server (the env vars are read once
     * at server start, so they can't vary per call). Because it re-runs the
     * identical eligibility checks, it can never force self-contained when
     * ineligible (no blocks / stale store_fp / no addr_tables) — it falls to
     * Option-A or online, which is correct.
     *   no_blocks==true         -> online.
     *   no_self_contained==true -> Option-A (when blocks are otherwise eligible).
     *   both false              -> self-contained if eligible, else Option-A.
     */
    void set_block_mode_override(bool no_blocks, bool no_self_contained, bool no_packed_full = false) {
        apply_block_mode_(no_blocks, no_self_contained, no_packed_full);
    }

private:
    /**
     * @brief Auto-detect a baked blocks/ directory next to the samples and
     *        record whether it exists. Called once from each constructor.
     *
     * When `<sample_dir>/blocks` exists, assemble_from_sample consumes baked
     * per-batch blocks (active_sizes + edge_index) instead of rebuilding them
     * online — guarded additionally by !nested_aggregation_ at use time, and
     * with per-batch full-hash staleness verification + graceful fallback.
     * When absent (the default today, since blocks/ won't exist), use_blocks_
     * stays false and behavior is byte-identical to the online build.
     *
     * This now reads the env toggles into booleans (truthy values
     * 1/true/yes, like every other MDB_GNN_* boolean flag) and delegates to
     * apply_block_mode_ (the shared mode-selection body). With no env var
     * set, the result is byte-identical to the pre-env-toggle auto-detection.
     */
    void init_blocks_();

    /**
     * @brief Shared block-mode selection used by both init_blocks_()
     *        (with env-derived booleans) and set_block_mode_override() (with
     *        per-call booleans).
     *
     * Sets use_blocks_ / self_contained_mode_ / store_fp_ from the booleans
     * plus the existing eligibility checks (blocks/ dir presence,
     * !nested_aggregation_, feature_store_ != nullptr, catalog content fp != 0,
     * addr_tables/ present, batch-0 store_fp == catalog fp), and emits the
     * once-style feature-load mode log line. no_blocks gates use_blocks_ on
     * BOTH constructor paths (the FeatureMatrix-fallback assembler consumes
     * Option-A blocks too); the other two booleans only matter in full mode.
     */
    void apply_block_mode_(bool no_blocks, bool no_self_contained, bool no_packed_full);

    /// Load features for a self-contained batch: packed-full (one O_DIRECT read,
    /// when packed_full_mode_) else the v2 addr_table gather on @p ms. Sets
    /// mini.features + feature timing. Returns false ONLY when the v2 gather fell
    /// back to the legacy contents-reading path (caller must fall back to the
    /// real sample); packed-full always returns true (bit-identical by construction).
    bool load_self_contained_features_(uint64_t batch_id, const GraphSample& ms, MiniBatch& mini);

    /// Structural-cache lookup. On a hit, bumps the LRU and copies the cached
    /// bundle into @p mini (refcounted tensors — shares storage), restoring
    /// batch_id + split from the cached entry. On a miss, increments the miss
    /// counter only when @p count_miss is true (the self-contained path
    /// accounts its miss once, where the block is actually opened). Returns
    /// false without touching counters when the cache is disabled.
    bool try_struct_hit_(uint64_t batch_id, MiniBatch& mini, bool count_miss);

    /// Insert @p mini's structural bundle (everything except features) into
    /// the cache under @p batch_id, evicting LRU entries until the byte budget
    /// fits. No-op when the cache is disabled, the batch is already cached, or
    /// the bundle alone exceeds the budget.
    void cache_struct_(uint64_t batch_id, const MiniBatch& mini);

    /**
     * @brief Try to assemble batch @p batch_id WITHOUT reading batches.dat.
     *
     * Reads ONLY the baked self-contained block (active_sizes + edge_index +
     * num_unique_nodes + seeds + split) and builds a MINIMAL GraphSample whose
     * all_unique_nodes is a placeholder of the right SIZE — sufficient for the
     * v2 (addr_table) feature gather, which keys off the count + batch_id and
     * never reads node CONTENTS.
     *
     * Returns true iff the MiniBatch is fully + correctly populated (in which
     * case @p mini.timing.sample_read_ns == 0 — batches.dat was never touched).
     * Returns false to request the caller fall back to the legacy real-sample
     * path, which is always correct. THE LOAD-BEARING SAFETY NET: after
     * load_features on the placeholder, we check the PER-CALL v2 dispatch
     * outcome it reports (NOT the store's shared last_used_addr_tables() flag,
     * which a concurrent prefetch worker can overwrite between our load and
     * the check); if v2 did NOT serve (addr_table stale/absent -> silent
     * legacy v1 gather which DOES read node contents), the placeholder
     * features would be WRONG, so we discard and return false. A placeholder
     * MiniBatch is therefore NEVER returned with v1-gathered features.
     *
     * Only ever invoked when self_contained_mode_ is true.
     */
    bool try_assemble_self_contained_(uint64_t batch_id, MiniBatch& mini);

    /**
     * @brief Output bundle from build_active_indices.
     *
     * Relocated to mdb::gnn::graph_block (graph_block_builder.h) so the offline
     * bake (gnn_build_feature_store) and the train path share one definition.
     * Kept as a private type alias here so the rest of BatchAssembler still
     * names ActiveIndicesResult unchanged.
     */
    using ActiveIndicesResult = graph_block::ActiveIndicesResult;

    /**
     * @brief Build per-layer cumulative active-set gather indices.
     *
     * For K-layer sample (K = edges_per_layer.size()), produces K+1 Long
     * tensors. result.indices_per_layer[k] lists global positions in
     * sample.all_unique_nodes for every node in A_k = ∪_{j<=k} nodes_per_layer[j].
     *
     * Also produces result.oid_to_local_per_layer[k] mapping every
     * ObjectId.id present in A_k to its local position within A_k.
     *
     * Invariant: A_k is a prefix [0, |A_k|) of A_{k+1} because
     * rebuild_unique_nodes() inserts nodes in layer order (seeds first,
     * then layer 1, etc.). The model relies on this prefix property to
     * extract self-features via x.slice() without a gather.
     *
     * @param sample         Source sample with nodes_per_layer + all_unique_nodes.
     * @param oid_to_global  Map from ObjectId.id to global position in all_unique_nodes.
     * @return  ActiveIndicesResult bundle (indices + sizes + oid_to_local maps)
     */
    ActiveIndicesResult
    build_active_indices(
        const GraphSample& sample,
        const std::unordered_map<uint64_t, int64_t>& oid_to_global);

    /**
     * @brief Build per-layer edge index tensors with LOCAL active-set indices.
     *
     * For each layer k, produces a [2, E_k] Long tensor where:
     * - row 0 (src) is local indices within active_indices_per_layer[k+1]
     * - row 1 (dst) is local indices within active_indices_per_layer[k]
     *
     * This eliminates the need for an extra remap in the model — index_select
     * directly into x = features[active_indices_per_layer[k+1]] works.
     *
     * Takes the per-layer oid_to_local maps produced by build_active_indices
     * so each edge endpoint costs ONE hash lookup (ObjectId.id -> local idx)
     * instead of two (oid -> global -> local).
     *
     * Fast path: when active.oid_to_local_per_layer is EMPTY, the
     * active sets are identity prefixes (local index == global position), so each
     * endpoint is remapped by pure array indexing into active.layer_global_pos
     * (precomputed in build_active_indices) — ZERO per-edge hash lookups. When
     * non-empty, the legacy per-layer maps are used (defensive fallback for a
     * hypothetically non-identity order).
     */
    std::vector<torch::Tensor> build_edge_indices(
        const GraphSample& sample,
        const ActiveIndicesResult& active);

    /**
     * @brief Load features for the unique nodes of a sample.
     *
     * Routes through FourLevelStore (if in full mode) or FeatureMatrix fallback.
     *
     * Takes a `const GraphSample&` so the FourLevelStore path can reuse the
     * already-deserialized sample instead of re-reading it from disk inside
     * load_batch_features (eliminates the double-deserialize).
     *
     * @param sample  Source sample whose `all_unique_nodes` define the feature rows
     * @param used_addr_tables  Optional out-param: whether THIS call was served
     *                          by the FourLevelStore v2 (addr_table) path.
     *                          Always false in FeatureMatrix-fallback mode.
     *                          Per-call (safe under concurrent prefetch
     *                          workers), unlike the store's shared
     *                          last_used_addr_tables() telemetry flag.
     * @return [N, D] tensor matching the feature dtype
     */
    torch::Tensor load_features(const GraphSample& sample,
                                bool* used_addr_tables = nullptr);

    // Feature source — exactly one of the two is set.
    FourLevelStore*      feature_store_   = nullptr;  // full mode
    const FeatureMatrix* feature_matrix_  = nullptr;  // fallback mode

    SampleStorage&    samples_;
    LabelStore*       labels_;    // nullable
    SplitStore*       splits_;    // nullable
    const RowMapping& row_mapping_;

    // Feature-store name (full mode only): keys the packed-full pack probe in
    // apply_block_mode_. Unused in FeatureMatrix-fallback mode.
    std::string feature_name_ = "node_features";

    // Nested (DGL-block) edge wiring in build_edge_indices. Default read from
    // env MDB_GNN_NESTED_AGG in the constructor; see set_nested_aggregation().
    bool nested_aggregation_ = false;

    // --- Baked computation-graph blocks (Task 7) ---
    // use_blocks_ is set by init_blocks_() at construction: true iff a
    // `<sample_dir>/blocks` directory exists. When true (and not nested at use
    // time), assemble_from_sample consumes the baked block for the batch — full
    // active+edge build is skipped, with per-batch full-hash staleness check and
    // graceful fallback to the online build on any miss/stale block.
    bool use_blocks_ = false;
    std::filesystem::path blocks_dir_;   // <sample_dir>/blocks
    // Warn-once latch so the fallback log line is emitted at most once per
    // assembler (the first stale/missing batch), not once per batch per epoch.
    std::atomic<bool> block_fallback_warned_{false};

    // --- Self-contained-block train mode (skip batches.dat deserialization) ---
    // When true, assemble(batch_id) reads ONLY the baked self-contained block
    // (active_sizes + edge_index + num_unique_nodes + seeds + split) and SKIPS
    // deserializing batches.dat (the dominant per-epoch cost). Eligibility
    // (set once in init_blocks_): use_blocks_ && !nested_aggregation_ &&
    // feature_store_ != nullptr && catalog.sample_content_fp != 0 &&
    // addr_tables/ present && batch-0 block store_fp == catalog fingerprint, and
    // not forced off by MDB_GNN_NO_BLOCKS / MDB_GNN_NO_SELF_CONTAINED. When
    // false, assemble() takes the existing real-sample path verbatim (OFF path
    // byte-identical). store_fp_ is the validated store fingerprint reused for
    // every per-batch open_self_contained.
    bool     self_contained_mode_ = false;
    uint64_t store_fp_            = 0;

    // --- Packed-full feature store (consume per-batch contiguous [N_b,D] pack) ---
    // When true, a self-contained batch's features come from a single O_DIRECT
    // read of <sample_dir>/packed_full/{dat,idx} (keyed by the MIXED feature
    // fingerprint) instead of the v2 4-tier addr_table gather. Composes with the
    // self-contained block path (the block still drives graph + seeds + labels).
    bool packed_full_mode_ = false;
    std::optional<PackedFullReader> packed_full_;
    // Warn-once latch for the self-contained per-batch fallback (missing/stale
    // block, or v2 not served) so the cerr line is emitted at most once.
    std::atomic<bool> self_contained_fallback_warned_{false};

    // --- Structural per-batch cache (see set_struct_cache_budget_bytes) ---
    // Holds everything assemble_from_sample produces EXCEPT features (which are
    // re-loaded each epoch). Tensors are CPU + refcounted, so a cache hit shares
    // storage (no data copy); TrainingLoop's per-epoch .to(device) creates fresh
    // device tensors and never mutates the cached CPU originals.
    struct CachedStruct {
        std::vector<torch::Tensor> edge_indices;
        std::vector<torch::Tensor> active_indices_per_layer;
        std::vector<int64_t>       active_sizes_per_layer;
        torch::Tensor              labels;
        torch::Tensor              label_mask;
        torch::Tensor              seed_ids;
        torch::Tensor              seed_rows;
        uint64_t                   num_seeds = 0;
        uint64_t                   num_nodes = 0;
        uint64_t                   num_labeled = 0;
        // The batch's SplitType (as the SplitType ordinal). Cached so the
        // self-contained hit path can restore mini.split without reopening the
        // block header. (The real-sample hit path still restores split from
        // sample.split — it has the live sample; c.split exists for the
        // self-contained path, which has no live sample to read it from.)
        uint32_t                   split = 0;
    };
    struct StructCacheEntry {
        CachedStruct s;
        std::list<uint64_t>::iterator lru_it;
        size_t bytes;
    };
    mutable std::mutex struct_mu_;
    size_t struct_budget_ = 0;
    size_t struct_bytes_  = 0;
    std::unordered_map<uint64_t, StructCacheEntry> struct_cache_;
    std::list<uint64_t> struct_lru_;
    uint64_t struct_hits_ = 0;
    uint64_t struct_misses_ = 0;
    uint64_t struct_evictions_ = 0;
};

} // namespace mdb::gnn
