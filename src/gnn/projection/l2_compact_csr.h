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
 *        Store (frequency-tiered L1 RAM hash / L2 compact uint32 CSR /
 *        L3 mmap sidecar / L4 direct B+Tree), holding the warm-tier
 *        nodes assigned during the build phase's tier-2 population step.
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
 * immutable and any further `add_node()` throws. `get()` requires
 * the structure to be frozen and throws `std::logic_error` if called
 * pre-freeze (symmetric to add_node()'s post-freeze throw — the
 * project's "fail loud" discipline rejects silent-miss foot-guns).
 *
 * **L2 fixed-cost note (papers100M scale):** at N_L2 ≈ 16M warm nodes,
 * `row_ptr_` dominates the L2 fixed memory cost: 16M × 8 bytes = 128 MB
 * pinned in RAM, vs ~32 MB for `node_to_l2_idx_` hash overhead. The
 * Phase 1 byte-budget contract `kL2NodeFixedOverhead = 8` already
 * accounts for this; the note exists to set expectations for the
 * reader who might otherwise expect the hash map to dominate.
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
     *
     * @throws std::logic_error if called before `freeze()`. The
     *         pre-freeze prefix-sum table is empty, so a silent miss
     *         would mask orchestrator bugs — symmetric to
     *         `add_node()`'s post-freeze throw.
     */
    ColIdxSpan get(uint64_t src_node_id) const;

    bool        is_frozen()   const noexcept { return frozen_; }
    std::size_t node_count()  const noexcept { return node_to_l2_idx_.size(); }
    std::size_t edge_count()  const noexcept { return col_idx_.size(); }

    /**
     * @brief Raw flat-array accessors for pinning the L2 CSR as a GPU substrate.
     *
     * Expose the immutable post-freeze `row_ptr_` / `col_idx_` so the dynamic
     * GPU sampling path can register them with `cudaHostRegister` (no copy, no
     * layout change, no loss of immutability). `row_ptr_len()` is the prefix-sum
     * length (`node_count() + 1` after freeze, 0 before). The col_idx values are
     * tag-stripped uint32 ordinals — a consumer reconstructs full ObjectIds via
     * `dst_type_tag()`. Pointers are valid for the lifetime of this (move-only)
     * object; do NOT cache them across a move.
     */
    const uint64_t* row_ptr_data() const noexcept { return row_ptr_.data(); }
    std::size_t     row_ptr_len()  const noexcept { return row_ptr_.size(); }
    const uint32_t* col_idx_data() const noexcept { return col_idx_.data(); }
    std::size_t     col_idx_size() const noexcept { return col_idx_.size(); }

    /**
     * @brief Per-direction dst ObjectId type tag, PRE-SHIFTED into the top
     *        byte (tag << 56); 0 if no nodes were added.
     *
     * `col_idx_` stores tag-stripped uint32 ordinals (the 8-bit ObjectId
     * type tag is dropped at add_node time for density). Consumers MUST
     * OR this back to recover the exact tagged ObjectId:
     * `dst_type_tag() | static_cast<uint64_t>(col_idx_[i])` — the same
     * convention the L3-narrow tier uses with `l3_dst_tag`. Captured in
     * add_node from the stored dst's top byte (uniform per direction).
     */
    uint64_t    dst_type_tag() const noexcept { return dst_type_tag_; }

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
    // Pre-shifted (tag << 56) ObjectId type tag shared by all stored dst.
    // col_idx_ holds tag-stripped uint32 ordinals; this re-supplies the tag.
    uint64_t                                    dst_type_tag_ = 0;
    std::unordered_map<uint64_t, uint32_t>      node_to_l2_idx_;

    // Per-row degrees, captured at add_node time. Used in two places:
    //   (1) to drive the freeze() prefix-sum step,
    //   (2) to compute total_bytes() in O(node_count) post-freeze.
    std::vector<uint32_t>                       degrees_;
};

}  // namespace mdb::gnn
