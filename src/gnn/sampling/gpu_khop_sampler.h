#pragma once

// gpu_khop_sampler.h
//
// Host-side facade for GPU k-hop sampling (no CUDA types in the signatures, so
// this header is includable without nvcc). The kernel does the expensive
// parallel part: per frontier node it samples up to `fanout` neighbors without
// replacement from the adjacency CSR pinned in host RAM (read via UVA). The
// mechanism — reservoir sampling with a counter-based RNG — and its guarantees
// are documented at the top of gpu_khop_sampler.cu; the caller-visible contract
// is:
//
//   - Deterministic: a node's sample depends only on
//     (random_seed XOR batch_id, node, layer). It is invariant to thread/block
//     scheduling and to the number of workers, matching the per-batch reseeding
//     contract of the CPU sampler.
//   - Same distribution as the CPU sampler (uniform without replacement,
//     k = min(fanout, deg), and the k == deg case emits every neighbor without
//     consuming randomness), but NOT the same picks: the CPU path draws from a
//     serial mt19937_64 that cannot be replicated in parallel, so the two
//     backends are distribution-equal, never bit-equal. The RNG that produced a
//     sample is recorded in the sample metadata so consumers can tell them
//     apart.
//
// Hybrid GPU+host design: the GPU samples the neighbors of each layer (the
// expand); the host reuses the CPU sampler's assembly logic (local index maps +
// first-appearance dedup) to build the GraphSample, so the SHAPE of the output
// is identical across backends — only the drawn neighbors differ.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gnn/sampling/graph_sample.h"

namespace mdb::gnn {

class PinnedTopologyView;
struct SamplingBackendPlan;

/**
 * @brief Full k-hop sampling on GPU → host `GraphSample`.
 *
 * Takes `nodes_per_layer[0] = seeds` verbatim; expands layer by layer with the
 * reservoir kernel over the pinned CSR of `view`; assembles edges +
 * all_unique_nodes on the host with the same semantics as `BasicKHopSampler`.
 * `plan.directions` decides which graph directions the GPU serves (REVERSE by
 * default; BOTH for UNDIRECTED).
 *
 * Precondition: `view.is_registered()` (a GPU exists and the CSR is pinned).
 * The caller (the sampling engine) only invokes this when the backend plan
 * chose GPU and the view registered successfully.
 */
GraphSample sample_khop_gpu(const std::vector<ObjectId>&     seeds,
                            uint64_t                         batch_id,
                            SplitType                        split,
                            const std::vector<int>&          fanouts,
                            const PinnedTopologyView&        view,
                            const SamplingBackendPlan&       plan,
                            uint64_t                         random_seed);

/**
 * @brief TEST-ONLY entry point for the per-node sampling primitive.
 *
 * Exposes the reservoir+Philox kernel in isolation for statistical validation
 * without the full pipeline: uploads a synthetic host CSR to the device,
 * samples up to `fanout` neighbors of every node in `nodes`, and returns per
 * node the list of sampled neighbor ids (dense uint32). Deterministic per
 * `(batch_seed, node, layer)`.
 *
 * @param row_ptr   Host CSR row_ptr (length N+1).
 * @param col_idx   Host CSR col_idx (length M), dense tag-stripped ids.
 * @param nodes     Frontier node ids (dense) to sample.
 * @param fanout    Maximum neighbors per node.
 * @param batch_seed  `random_seed XOR batch_id`, used as the Philox key.
 * @param layer     Layer index (part of the Philox counter).
 * @return For each input node, its sampled neighbors (empty if degree 0).
 */
std::vector<std::vector<uint32_t>> gpu_sample_neighbors_for_test(
    const std::vector<uint64_t>& row_ptr,
    const std::vector<uint32_t>& col_idx,
    const std::vector<uint32_t>& nodes,
    int                          fanout,
    uint64_t                     batch_seed,
    int                          layer);

/**
 * @brief TILED variant of the test seam: same contract as
 *        gpu_sample_neighbors_for_test, but stages COL_IDX through node-aligned
 *        windows with soft cap `window_cap_edges` (one reusable pinned buffer).
 *
 * Verifies that the tiled path (whole ROW_PTR + windowed COL_IDX) produces a
 * BIT-IDENTICAL result to the whole-CSR path for any window cap, demonstrating
 * the windowing/partitioning/reassembly is correct without touching production
 * code or requiring a symmetric CSR file on disk.
 *
 * @param window_cap_edges Soft cap of edges per window (a node whose degree
 *        exceeds the cap forms its own window; the actual buffer is sized to
 *        max(cap, max degree)).
 */
std::vector<std::vector<uint32_t>> gpu_sample_neighbors_tiled_for_test(
    const std::vector<uint64_t>& row_ptr,
    const std::vector<uint32_t>& col_idx,
    const std::vector<uint32_t>& nodes,
    int                          fanout,
    uint64_t                     batch_seed,
    int                          layer,
    std::size_t                  window_cap_edges);

}  // namespace mdb::gnn
