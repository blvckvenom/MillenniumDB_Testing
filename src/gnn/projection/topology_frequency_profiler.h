#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "gnn/projection/topology_accessor.h"

namespace mdb::gnn {

// ===========================================================================
// Per-tier memory accounting constants for the Four-Level Topology Store
// ===========================================================================
//
// These four constants describe the bytes-per-node and bytes-per-edge cost
// of holding a node in each tier of the Four-Level Topology Store. They are
// declared in this public header (not buried in the .cc) so that Phase 2's
// `FourLevelTopologyStore::build_phase`, `L1HashCache`, and `L2CompactCsr`
// can `static_assert` / arithmetically share the same numbers used here by
// `compute_tier_assignment`. Any change to the on-disk / in-memory layout
// of those structures MUST be reflected here in lockstep.
//
//   L1 (RAM hash cache):
//     - 16 B per AdjEntry                 → kL1PerEdgeBytes
//     - 24 B vector header
//     - ~32 B per-bucket hash overhead    → 56 B total fixed → kL1NodeFixedOverhead
//
//   L2 (compact CSR):
//     -  8 B per edge (uint32 col_idx + uint32 edge_id) → kL2PerEdgeBytes
//     -  8 B per node (one uint64 row_ptr entry)        → kL2NodeFixedOverhead

constexpr std::size_t kL1NodeFixedOverhead = 56;
constexpr std::size_t kL1PerEdgeBytes      = 16;
constexpr std::size_t kL2NodeFixedOverhead = 8;
constexpr std::size_t kL2PerEdgeBytes      = 8;

/**
 * @brief Frequency profile of nodes in a projection, used to seed the
 *        Four-Level Topology Store tier assignment (L1 RAM hash / L2 compact
 *        uint32 CSR / L3 mmap sidecar / L4 direct B+Tree, frequency-tiered).
 *
 * The profiler exposes two source paths:
 *
 *   1. **Warm start** (`compute_from_node_counts`): reads a prior
 *      `node_counts.bin` produced by a previous `gnn_offline_sample` run.
 *      This file mirrors DiskGNN's "node access count" technique. Phase 1
 *      keeps this method as a stub that returns false (no warm-start file
 *      consumed) until `gnn_offline_sample` learns to persist the counts
 *      (tracked separately as a future phase).
 *
 *   2. **Cold start** (`compute_from_degrees`): falls back to the node's
 *      out / in / out+in degree (depending on `EdgeOrientation`) as a
 *      coarse popularity proxy. Always available, no prerequisite.
 *
 * The output is a `frequency()` vector indexed by node iteration order
 * (matching `NodeIterator` for the bound `TopologyAccessor`). Tier
 * assignment is delegated to `compute_tier_assignment()`, a free helper so
 * tests can drive it with synthetic frequencies without instantiating a
 * real projection.
 *
 * Lifetime: the profiler holds a non-owning reference to the
 * `TopologyAccessor`. Keep the accessor alive for the duration of any
 * `frequency()` consumer.
 */
class TopologyFrequencyProfiler {
public:
    /**
     * @brief Construct a profiler bound to a topology accessor.
     *
     * The accessor is taken by non-const reference to mirror the
     * lazy-cursor reality of `TopologyAccessor::get_*_degree` (and the
     * GNN-module convention used by `FeatureAccessor` over
     * `GQL::ProjectionStorage&`). The profiler is still semantically
     * read-only — it never mutates stored topology — but it threads the
     * accessor's internal cache state through unchanged.
     *
     * @param topo            Source of degree / iteration data. Must outlive
     *                        the profiler.
     * @param projection_dir  Directory used to look up an optional
     *                        `node_counts.bin` warm-start file.
     */
    TopologyFrequencyProfiler(TopologyAccessor& topo,
                              std::filesystem::path projection_dir);

    /**
     * @brief Compute a frequency vector for every node in the bound
     *        topology under the requested edge orientation.
     *
     * Tries the warm-start path first; on miss falls back to the degree
     * proxy. Idempotent: a second call overwrites the previous frequency
     * vector and `warm_start_used()` flag.
     */
    void compute(EdgeOrientation direction);

    /// Frequency vector indexed by node iteration order. Empty until
    /// `compute()` is called.
    const std::vector<uint64_t>& frequency() const { return frequency_; }

    /// True iff the most recent `compute()` consumed `node_counts.bin`.
    /// Phase 1 always returns false.
    bool warm_start_used() const { return warm_start_used_; }

private:
    /**
     * @brief Phase 1 STUB — read `<projection_dir>/node_counts.bin`.
     *
     * Returns false unconditionally in Phase 1 because
     * `gnn_offline_sample` does not yet persist the file. The method is
     * preserved in the API so Phase 2 can plug in the real reader without
     * touching call sites.
     */
    bool compute_from_node_counts_(EdgeOrientation direction);

    /// Cold-start path — populates `frequency_` from per-node degree.
    void compute_from_degrees_(EdgeOrientation direction);

    // Non-owning. Held as non-const because `TopologyAccessor::get_*_degree`
    // lazily fault in B+Tree cursors / adjacency caches.
    TopologyAccessor& topo_;
    std::filesystem::path projection_dir_;
    std::vector<uint64_t> frequency_;
    bool warm_start_used_ = false;
};

/**
 * @brief Tier assignment helper for the Four-Level Topology Store.
 *
 * Greedy partition: nodes are sorted by frequency descending; entries are
 * packed into tier 1 (L1 RAM hash) until `l1_budget_bytes` is exhausted,
 * then into tier 2 (L2 compact CSR) until `l2_budget_bytes` is exhausted,
 * then everything else goes to tier 3 (L3 mmap sidecar / L4 BPT direct).
 *
 * Per-node memory accounting (see constants declared at top of this header):
 *   - L1: `degree(i) * 16 + 56` bytes (16 B per AdjEntry +
 *         24 B vector header + ~32 B hash bucket overhead).
 *   - L2: `degree(i) * 8  +  8` bytes (uint32 col_idx + uint32 edge_id
 *         per edge + one uint64 row_ptr entry).
 *
 * @param frequency       Frequency vector (one entry per node, same indexing
 *                        as the source `TopologyAccessor`).
 * @param l1_budget_bytes Bytes available for tier 1.
 * @param l2_budget_bytes Bytes available for tier 2.
 * @param avg_degree      Mean degree across the graph; multiplied by
 *                        per-tier per-edge cost to size each node when the
 *                        caller chose not to pre-compute exact degrees.
 *                        Caller is expected to have computed this once.
 *
 * @return Vector with one byte per node, indexed identically to
 *         `frequency`. Values:
 *           - 1 → tier 1 (L1 hash)
 *           - 2 → tier 2 (L2 compact CSR)
 *           - 3 → tier 3 (L3 mmap / L4 BPT)
 *
 * Edge case: when the combined budget exceeds the graph's notional cost,
 * every node lands in tier 1.
 */
std::vector<uint8_t> compute_tier_assignment(
    const std::vector<uint64_t>& frequency,
    std::size_t l1_budget_bytes,
    std::size_t l2_budget_bytes,
    double avg_degree);

} // namespace mdb::gnn
