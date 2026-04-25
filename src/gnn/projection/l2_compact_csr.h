#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gnn/projection/adj_entry.h"

namespace mdb::gnn {

/**
 * @brief Tier-2 compact-CSR adjacency cache for the Four-Level Topology
 *        Store (Spec #13 Phase 2 / T13.5).
 *
 * Holds the warm-tier adjacency in three flat arrays:
 *
 *   - `row_ptr_`  uint64 prefix-sum offsets, `(num_l2_nodes + 1)` entries.
 *   - `col_idx_`  uint32 destination row indexes (uint32 is safe because
 *                 papers100M < 2^32 nodes; assertion enforced at freeze).
 *   - `node_to_l2_idx_` unordered_map keying source ObjectId.id → L2 row.
 *
 * **edge_ids decision (Phase 2)** — `edge_ids` are intentionally
 * **omitted** from the L2 layout. Three reasons:
 *
 *   1. The Phase 1 byte-budget contract in
 *      `topology_frequency_profiler.h` allocates 8 B per edge for L2
 *      (`kL2PerEdgeBytes`). Adding a uint64 edge_id would force the
 *      contract to 12 B (4 B col_idx + 8 B edge_id) since ObjectId
 *      edge ids occupy 56 bits — the design doc §2.6 list of `uint32_t
 *      edge_ids` is wrong against `object_id.h` (8-bit type tag + 56-bit
 *      payload). Either the contract bumps to 12 (re-derives the
 *      profiler test) OR L2 holds no edge_ids. We chose the latter.
 *
 *   2. The sampler (`BasicKHopSampler`) only consults edge_ids for
 *      edge-property lookups, which are rare in GraphSAGE-style GNN
 *      workloads. The EmbeddingWriter Phase B path uses node-id dedup
 *      since commit `896b3897` (zero-edge-id sentinel handling).
 *
 *   3. Dropping edge_ids leaves the contract intact AND means L2 is
 *      effectively over-budgeted by 4 B/edge (we book 8, spend 4) —
 *      defensive safety margin at scale.
 *
 * Callers that *do* need edge_ids for an L2-tier node fall through to
 * L4 (B+Tree direct). Phase 3 may add an opt-in edge_ids array if a
 * workload demands it.
 *
 * Lifecycle: callers `add_node()` repeatedly during the build phase,
 * then call `freeze()` exactly once. After freeze the structure is
 * immutable and any further `add_node()` throws. `get()` works in
 * either state but is intended to be invoked only post-freeze for
 * deterministic results.
 */
class L2CompactCsr {
public:
    /**
     * @brief Lookup result: pointer + length view into `col_idx_`.
     *
     * The first member is `data` and the second is `size`, matching
     * the shape of `L1HashCache::Span` so callers can write a
     * uniform dispatch in `FourLevelTopologyStore`. Returns
     * `(nullptr, 0)` on miss.
     */
    using ColIdxSpan = std::pair<const uint32_t*, std::size_t>;

    /**
     * @brief Construct an empty CSR with capacity hints.
     *
     * @param num_l2_nodes_hint  Approximate L2 node count, used to
     *                           pre-reserve the row_ptr / map vectors.
     *                           A wrong hint costs only an extra
     *                           amortized realloc; not load-bearing.
     */
    explicit L2CompactCsr(std::size_t num_l2_nodes_hint = 0);

    L2CompactCsr(const L2CompactCsr&)            = delete;
    L2CompactCsr& operator=(const L2CompactCsr&) = delete;
    L2CompactCsr(L2CompactCsr&&) noexcept        = default;
    L2CompactCsr& operator=(L2CompactCsr&&) noexcept = default;

    /**
     * @brief Append a node's adjacency to the flat arrays.
     *
     * `add_node()` may be called in any order. Internally a per-call
     * unique L2 row index is assigned and recorded in
     * `node_to_l2_idx_`.
     *
     * Behavior on duplicate `src_node_id`: the second call throws
     * `std::invalid_argument`. The orchestrator is expected to
     * deduplicate up-front.
     *
     * @throws std::logic_error if the cache is already frozen.
     * @throws std::invalid_argument on duplicate `src_node_id`.
     */
    void add_node(uint64_t                       src_node_id,
                  const std::vector<AdjEntry>&   neighbors);

    /**
     * @brief Finalize the row-pointer prefix sum and validate
     *        invariants. After freeze the cache is immutable.
     *
     * @throws std::overflow_error if the total edge count would
     *         exceed UINT32_MAX (the per-edge col_idx is uint32, not
     *         the offset which is uint64; this guard exists so any
     *         future use of uint32 offsets in a packed-on-disk
     *         variant fails loudly here).
     */
    void freeze();

    /**
     * @brief O(1) lookup of a source node's destination span.
     *
     * @return `(nullptr, 0)` on miss. On hit, the pointer is into
     *         the cache-owned `col_idx_` vector and remains valid
     *         for the lifetime of the cache.
     */
    ColIdxSpan get(uint64_t src_node_id) const;

    bool        is_frozen()   const noexcept { return frozen_; }
    std::size_t node_count()  const noexcept { return node_to_l2_idx_.size(); }
    std::size_t edge_count()  const noexcept { return col_idx_.size(); }

    /**
     * @brief Approximate resident-byte cost using the Phase 1 contract.
     *
     * `kL2NodeFixedOverhead + kL2PerEdgeBytes * degree` per node.
     * Note: per the edge_ids decision above, the *actual* RSS will be
     * lower than this estimate (we book 8 B/edge but spend 4 B/edge).
     * The estimate is the upper bound the profiler used at sizing
     * time, so test parity against the Phase 1 contract checks holds.
     */
    std::size_t total_bytes() const;

private:
    bool                                        frozen_ = false;
    std::vector<uint64_t>                       row_ptr_;
    std::vector<uint32_t>                       col_idx_;
    std::unordered_map<uint64_t, uint32_t>      node_to_l2_idx_;

    // Per-row degrees, captured at add_node time. Used in two places:
    //   (1) to drive the freeze() prefix-sum step,
    //   (2) to compute total_bytes() in O(node_count) post-freeze.
    std::vector<uint32_t>                       degrees_;
};

}  // namespace mdb::gnn
