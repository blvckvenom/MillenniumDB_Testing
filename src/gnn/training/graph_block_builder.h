#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "gnn/sampling/graph_sample.h"

namespace mdb::gnn::graph_block {

/**
 * @brief Output bundle from build_active_indices.
 *
 * Edge-remap optimization (2026-05-15): in addition to the per-layer
 * global-position tensors used by the model gather, also produce a per-layer
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
                                                           // (populated ONLY on the defensive fallback;
                                                           //  empty on the fast identity-prefix path)
    // Fast-path only: layer_global_pos[k][i] = global position (== local
    // index under the identity prefix) of nodes_per_layer[k][i], for EVERY
    // entry incl. cross-layer duplicates. Filled for free in Phase 1 (reuses
    // the oid_to_global lookups build_active_indices already does), so
    // build_edge_indices can remap each edge endpoint by pure array indexing
    // with ZERO per-edge hash lookups. Empty on the defensive fallback.
    std::vector<std::vector<int64_t>> layer_global_pos;
    // Fast path flag. True (the universal case) when the cumulative active
    // positions came out as the identity sequence [0,1,..,N-1] — guaranteed
    // by rebuild_unique_nodes() inserting nodes in layer order, so local
    // index == global position for every layer. When true, indices_per_layer
    // is an arange and oid_to_local_per_layer is left empty (build_edge_indices
    // remaps via oid_to_global directly). When false (never observed in
    // practice, only if a future sampler change breaks the ordering), the
    // legacy per-layer sort + maps are produced for exact equivalence.
    bool identity_prefix = true;
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
 * Edge-endpoint single-lookup optimization (2026-05-15): takes the per-layer
 * oid_to_local maps produced by build_active_indices so each edge endpoint
 * costs ONE hash lookup (ObjectId.id -> local idx) instead of two
 * (oid -> global -> local).
 *
 * Fast path (2026-06-04): when active.oid_to_local_per_layer is EMPTY, the
 * active sets are identity prefixes (local index == global position), so each
 * endpoint is remapped by pure array indexing into active.layer_global_pos
 * (precomputed in build_active_indices) — ZERO per-edge hash lookups. When
 * non-empty, the legacy per-layer maps are used (defensive fallback for a
 * hypothetically non-identity order).
 *
 * @param nested_aggregation  Select nested (DGL-block) edge wiring (true) vs
 *                            legacy per-hop wiring (false, the historical
 *                            default). See BatchAssembler::set_nested_aggregation.
 */
std::vector<torch::Tensor> build_edge_indices(
    const GraphSample& sample,
    const ActiveIndicesResult& active,
    bool nested_aggregation = false);

} // namespace mdb::gnn::graph_block
