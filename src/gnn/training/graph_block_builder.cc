#include "gnn/training/graph_block_builder.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mdb::gnn::graph_block {

// =============================================================================
// build_active_indices
// =============================================================================

ActiveIndicesResult
build_active_indices(
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
                        "graph_block::build_active_indices: node at layer " +
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
// build_edge_indices
// =============================================================================

std::vector<torch::Tensor> build_edge_indices(
    const GraphSample& sample,
    const ActiveIndicesResult& active,
    bool nested_aggregation)
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

    // Always-on validation of the disk-sourced edge endpoints on the fast
    // path: every edges_per_layer[j] entry must index within the layer node
    // lists (src in layer j+1, dst in layer j). Hoisted out of the remap loop
    // below (one sequential pass per hop layer, zero per-edge cost in the hot
    // loop) so a stale/corrupt sample throws a descriptive error — matching
    // the legacy fallback's failure behavior — instead of indexing lgp out of
    // bounds.
    if (fast) {
        for (size_t j = 0; j < num_layers; ++j) {
            if (j + 1 >= lgp.size()) {
                throw std::runtime_error(
                    "graph_block::build_edge_indices: missing node layer " +
                    std::to_string(j + 1) + " for edge layer " + std::to_string(j)
                );
            }
            const LayerEdges& edges = sample.edges_per_layer[j];
            const size_t src_bound  = lgp[j + 1].size();
            const size_t dst_bound  = lgp[j].size();
            const size_t Ej         = edges.size();
            for (size_t i = 0; i < Ej; ++i) {
                if (static_cast<size_t>(edges.src_indices[i]) >= src_bound) {
                    throw std::runtime_error(
                        "graph_block::build_edge_indices: src index " +
                        std::to_string(edges.src_indices[i]) +
                        " out of layer " + std::to_string(j + 1) + " bounds (" +
                        std::to_string(src_bound) + ") at hop " +
                        std::to_string(j) + ", edge " + std::to_string(i)
                    );
                }
                if (static_cast<size_t>(edges.dst_indices[i]) >= dst_bound) {
                    throw std::runtime_error(
                        "graph_block::build_edge_indices: dst index " +
                        std::to_string(edges.dst_indices[i]) +
                        " out of layer " + std::to_string(j) + " bounds (" +
                        std::to_string(dst_bound) + ") at hop " +
                        std::to_string(j) + ", edge " + std::to_string(i)
                    );
                }
            }
        }
    }

    // Nested aggregation (2026-06-02): conv k operates on dst set A_k =
    // ∪_{j<=k} nodes_per_layer[j] (all nodes within k hops). For STANDARD
    // GraphSAGE / DGL-block message passing every such node must aggregate its
    // sampled neighbours at conv k, so edge_index[k] is the union of the
    // per-hop edge sets E_0..E_k. Each per-hop set E_j (= edges_per_layer[j])
    // carries dst in layer j ⊆ A_k and src in layer j+1 ⊆ A_{k+1}, so both
    // endpoints resolve in the cumulative A_k / A_{k+1} maps.
    //
    // LEGACY (nested_aggregation == false): edge_index[k] = E_k only. A seed
    // (layer 0) then has edges solely in edge_index[0] and aggregates its
    // neighbourhood at just the final conv — a strictly weaker variant whose
    // deviation compounds with depth (see set_nested_aggregation docs).
    const bool nested = nested_aggregation;

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
                    // Endpoints were validated against the layer bounds in the
                    // hoisted pass above, so this indexing cannot go OOB.
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
                            "graph_block::build_edge_indices: src node not in A_" +
                            std::to_string(k + 1) + " active set (conv " +
                            std::to_string(k) + ", hop " + std::to_string(j) +
                            ", edge " + std::to_string(i) + ")"
                        );
                    }
                    if (dst_it == dst_map->end()) {
                        throw std::runtime_error(
                            "graph_block::build_edge_indices: dst node not in A_" +
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

} // namespace mdb::gnn::graph_block
