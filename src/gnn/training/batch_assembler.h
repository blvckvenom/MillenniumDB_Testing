#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "graph_models/object_id.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/four_level_store.h"
#include "gnn/storage/row_mapping.h"
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
     */
    BatchAssembler(
        FourLevelStore& feature_store,
        SampleStorage&  samples,
        LabelStore*     labels,
        SplitStore*     splits,
        const RowMapping& row_mapping
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
     * Used by Round 3B (multi-worker AsyncBatchPrefetcher) to refuse
     * concurrent execution against FourLevelStore: that store currently
     * has shared mutable state (DirectIoReader io_uring rings + pinned
     * host buffer reused across calls) that races silently under N>1
     * workers.
     */
    bool uses_feature_store() const { return feature_store_ != nullptr; }

    /**
     * @brief Read-only access to the underlying FourLevelStore, if any.
     *
     * Returns nullptr in FeatureMatrix-fallback mode. Used by TrainingLoop's
     * Phase 0 profile instrumentation (`BatchTimingLog`) to pull per-tier
     * timings off the store after each batch assemble. Const-correct: the
     * timings are advertised via const accessors and the store is not
     * mutated through this pointer.
     */
    const FourLevelStore* feature_store() const { return feature_store_; }

private:
    /**
     * @brief Output bundle from build_active_indices.
     *
     * Round 2C (2026-05-15): in addition to the per-layer global-position
     * tensors used by the model gather, also produce a per-layer
     * `ObjectId.id -> local-position-in-A_k` hash table. These are the
     * direct map build_edge_indices needs to remap edges from layer-local
     * indices straight into the active set, halving the number of hash
     * lookups per edge (was: oid->global, then global->local; now: oid->local).
     */
    struct ActiveIndicesResult {
        std::vector<torch::Tensor> indices_per_layer;          // K+1 Long tensors of global positions
        std::vector<int64_t>       sizes_per_layer;            // |A_k| for each k
        std::vector<std::unordered_map<uint64_t, int64_t>>
                                   oid_to_local_per_layer;     // K+1 maps: ObjectId.id -> local idx in A_k
    };

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
     * Round 2C (2026-05-15): takes the per-layer oid_to_local maps produced
     * by build_active_indices so each edge endpoint costs ONE hash lookup
     * (ObjectId.id -> local idx) instead of two (oid -> global -> local).
     */
    std::vector<torch::Tensor> build_edge_indices(
        const GraphSample& sample,
        const std::vector<std::unordered_map<uint64_t, int64_t>>& oid_to_local_per_layer);

    /**
     * @brief Load features for the unique nodes of a sample.
     *
     * Routes through FourLevelStore (if in full mode) or FeatureMatrix fallback.
     *
     * Round 2B (2026-05-15): takes a `const GraphSample&` so the FourLevelStore
     * path can reuse the already-deserialized sample instead of re-reading it
     * from disk inside load_batch_features (eliminates the double-deserialize).
     *
     * @param sample  Source sample whose `all_unique_nodes` define the feature rows
     * @return [N, D] tensor matching the feature dtype
     */
    torch::Tensor load_features(const GraphSample& sample);

    // Feature source — exactly one of the two is set.
    FourLevelStore*      feature_store_   = nullptr;  // full mode
    const FeatureMatrix* feature_matrix_  = nullptr;  // fallback mode

    SampleStorage&    samples_;
    LabelStore*       labels_;    // nullable
    SplitStore*       splits_;    // nullable
    const RowMapping& row_mapping_;
};

} // namespace mdb::gnn
