#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "gnn/projection/adj_entry.h"
#include "gnn/projection/l1_hash_cache.h"
#include "gnn/projection/l2_compact_csr.h"
#include "gnn/projection/topology_accessor.h"  // EdgeOrientation
#include "graph_models/object_id.h"

namespace GQL::Projection {
class TopologySnapshotReader;
}

namespace mdb::gnn {

/**
 * @brief Coordinator for the Four-Level Topology Store (Spec #13).
 *
 * Phase 2 (T13.6) skeleton. This class owns **only** the runtime
 * dispatch logic and a small amount of context plumbing. It does
 * NOT yet perform any of the heavy build orchestration (frequency
 * profiling, B+Tree scanning to populate L1/L2, mmap'ing the L3
 * sidecar). That is Phase 3 (T13.7).
 *
 * Phase 2 acceptance is: given pre-built L1 + L2 caches, an optional
 * pre-opened L3 sidecar, an optional L4 BPT fallback, a per-row tier
 * vector, and a `RowMapping`-equivalent callable that maps
 * `ObjectId → row_idx`, the dispatcher routes each
 * `get_out_neighbors` / `get_in_neighbors` call to the correct tier
 * and returns the raw neighbor span.
 *
 * Phase 3 will:
 *   - Add a `build()` method that consumes a `TopologyAccessor`
 *     plus a `TopologyFrequencyProfiler` to populate the caches.
 *   - Wire a real `RowMapping` instead of the testable
 *     `std::function<uint64_t(ObjectId)>` callable used here.
 *   - Plug the dispatcher into `TopologyAccessor::get_*_neighbors`.
 *
 * Threading: post-construction, this class is read-only and safe to
 * call from any number of sampler threads concurrently.
 */
class FourLevelTopologyStore {
public:
    /**
     * @brief Public neighbor span returned by every dispatch path.
     *
     * Holds the lookup result in two interchangeable shapes:
     *
     *   - L1 hits set `l1` and leave `l2_col_idx` / `l3_col_idx` null.
     *   - L2 hits set `l2_col_idx` (uint32 destination row indexes
     *     only — see `L2CompactCsr` header for the edge_ids decision)
     *     and `l2_size`.
     *   - L3 hits set `l3_col_idx` (uint64 from the mmap'd snapshot)
     *     and `l3_size`.
     *   - L4 hits materialise into `l4_owned` because the BPT path
     *     is the only tier that returns a freshly-allocated vector.
     *
     * Callers that only care about destination ids should call
     * `for_each_dst()` which abstracts over the four shapes.
     */
    struct Neighbors {
        // L1
        L1HashCache::Span         l1{};
        // L2: uint32 dst row indexes, length l2_size.
        const uint32_t*           l2_col_idx = nullptr;
        std::size_t               l2_size    = 0;
        // L3: uint64 dst node ids, length l3_size. Owned by mmap.
        const uint64_t*           l3_col_idx = nullptr;
        std::size_t               l3_size    = 0;
        // L4: BPT direct returns full Neighbors copies — the
        // dispatcher owns the allocation here.
        std::vector<AdjEntry>     l4_owned{};
        // Which tier produced the result.
        uint8_t                   tier = 0;

        bool empty() const noexcept;
        std::size_t size() const noexcept;
    };

    /**
     * @brief Lightweight L4 (B+Tree direct) callback.
     *
     * Phase 2 keeps the BPT integration testable in isolation by
     * accepting a callable instead of holding a `BPlusTree<3>*` and
     * its surrounding cursor / decoder code. The orchestrator wires
     * a `BPlusTree<3>::range_query` adapter here in Phase 3.
     *
     * Returns the source node's full adjacency. Empty vector for
     * isolated nodes; null callable means "no L4 available" — a
     * tier-4 dispatch with no callback throws.
     */
    using L4Lookup = std::function<std::vector<AdjEntry>(ObjectId)>;

    /**
     * @brief RowMapping-equivalent callable.
     *
     * Maps an `ObjectId` to its row index in the projection's node
     * iteration order. Phase 3 will wire this through the real
     * `RowMapping` class; Phase 2 keeps the abstraction lightweight
     * so unit tests can drive synthetic graphs without spinning up a
     * full projection.
     *
     * Must return a value `>= tier_lookup.size()` for nodes outside
     * the projection (or any sentinel; the dispatcher treats out-of-
     * range as "fall through to L4" / "throw if no L4").
     */
    using RowLookup = std::function<uint64_t(ObjectId)>;

    /**
     * @brief Phase 2 minimal config. Phase 3 will widen this.
     */
    struct Config {
        EdgeOrientation orientation = EdgeOrientation::UNDIRECTED;
    };

    /**
     * @brief Construct a dispatcher from pre-built tier sources.
     *
     * @param l1_fwd          Forward L1 cache. Always required (may be
     *                        empty).
     * @param l1_rev          Reverse L1 cache. Required when
     *                        `orientation` is REVERSE or UNDIRECTED.
     * @param l2_fwd          Forward L2 CSR. Always required.
     * @param l2_rev          Reverse L2 CSR. Required for non-NATURAL.
     * @param l3_fwd          Optional forward L3 mmap reader. May be
     *                        nullptr; tier-3 lookups then fall to L4.
     * @param l3_rev          Optional reverse L3 mmap reader.
     * @param l4_fwd          Optional forward L4 BPT callback.
     * @param l4_rev          Optional reverse L4 BPT callback.
     * @param tier_lookup     Per-row tier vector from
     *                        `compute_tier_assignment()`. Held by
     *                        const reference; must outlive *this.
     * @param row_lookup      Callable mapping ObjectId to row index.
     * @param config          Orientation + future-extension knobs.
     */
    FourLevelTopologyStore(
        const L1HashCache&                          l1_fwd,
        const L1HashCache&                          l1_rev,
        const L2CompactCsr&                         l2_fwd,
        const L2CompactCsr&                         l2_rev,
        const GQL::Projection::TopologySnapshotReader* l3_fwd,
        const GQL::Projection::TopologySnapshotReader* l3_rev,
        L4Lookup                                    l4_fwd,
        L4Lookup                                    l4_rev,
        const std::vector<uint8_t>&                 tier_lookup,
        RowLookup                                   row_lookup,
        Config                                      config);

    FourLevelTopologyStore(const FourLevelTopologyStore&)            = delete;
    FourLevelTopologyStore& operator=(const FourLevelTopologyStore&) = delete;

    /**
     * @brief Get outgoing neighbors (NATURAL).
     *
     * @throws std::out_of_range when `row_lookup(v)` is past
     *         `tier_lookup` AND no L4 callback is wired (defensive
     *         choice: silent empty would mask projection / config
     *         bugs).
     */
    Neighbors get_out_neighbors(ObjectId v) const;

    /**
     * @brief Get incoming neighbors (REVERSE).
     */
    Neighbors get_in_neighbors(ObjectId v) const;

    const Config& config() const noexcept { return config_; }

private:
    // Helper that performs the tier switch for one direction. The two
    // public `get_*_neighbors` methods just bind the per-direction
    // tier sources before delegating here.
    Neighbors dispatch_(
        ObjectId                                       v,
        const L1HashCache&                             l1,
        const L2CompactCsr&                            l2,
        const GQL::Projection::TopologySnapshotReader* l3,
        const L4Lookup&                                l4) const;

    const L1HashCache&                              l1_fwd_;
    const L1HashCache&                              l1_rev_;
    const L2CompactCsr&                             l2_fwd_;
    const L2CompactCsr&                             l2_rev_;
    const GQL::Projection::TopologySnapshotReader*  l3_fwd_ = nullptr;
    const GQL::Projection::TopologySnapshotReader*  l3_rev_ = nullptr;
    L4Lookup                                        l4_fwd_;
    L4Lookup                                        l4_rev_;
    const std::vector<uint8_t>&                     tier_lookup_;
    RowLookup                                       row_lookup_;
    Config                                          config_;
};

}  // namespace mdb::gnn
