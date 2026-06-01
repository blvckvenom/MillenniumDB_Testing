#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "gnn/projection/adj_entry.h"
#include "gnn/projection/edge_orientation.h"
#include "gnn/projection/l1_hash_cache.h"
#include "gnn/projection/l2_compact_csr.h"
#include "graph_models/object_id.h"

// Forward declarations to keep this header lightweight (no torch / mmap /
// projection storage transitives leaking through).
template <std::size_t N> class BPlusTree;

namespace GQL {
class ProjectionStorage;
namespace Projection {
class TopologySnapshotReader;
}
}  // namespace GQL

namespace mdb::gnn {

class TopologyAccessor;
class TopologyFrequencyProfiler;

/**
 * @brief Coordinator for the Four-Level Topology Store (Spec #13).
 *
 * Phase 3 (T13.7) replaces the Phase 2 dispatcher-only skeleton with a
 * full coordinator that:
 *
 *   1. Owns the four tier sources internally (forward + reverse L1
 *      hash caches and L2 compact CSRs, plus optional non-owning L3
 *      sidecar reader pointers and an L4 B+Tree pointer pair for
 *      fallback).
 *   2. Builds the tiers from a `ProjectionStorage` (or directly from
 *      its B+Tree pointers) using a `TopologyFrequencyProfiler` to
 *      decide tier assignment.
 *   3. Dispatches `get_out_neighbors` / `get_in_neighbors` /
 *      `get_neighbors(UNDIRECTED)` to the correct tier per node.
 *
 * Threading: post-`build()` the store is read-only and safe for
 * concurrent sampler threads. Calling `build()` twice is rejected; the
 * caller must drop and recreate the store to rebuild.
 */
class FourLevelTopologyStore {
public:
    /**
     * @brief Public neighbor span returned by every dispatch path.
     *
     * Holds the lookup result in four interchangeable shapes (one per
     * tier). Callers should consume the result through
     * `for_each_dst()` / `for_each_with_edge_id()` rather than
     * switching on `tier` themselves — the helpers absorb the
     * tier-specific boilerplate so downstream consumers stay tier-
     * agnostic.
     */
    struct Neighbors {
        // L1
        L1HashCache::Span         l1{};
        // L2: uint32 dst row indexes, length l2_size.
        const uint32_t*           l2_col_idx = nullptr;
        std::size_t               l2_size    = 0;
        // L3: dst node ids, length l3_size. Owned by mmap (zero-copy).
        // Two on-disk widths (Spec #6):
        //   id_width==8 → l3_col_idx / l3_edge_ids point at uint64 sections
        //                 (full tagged ObjectIds; l3_*_32 stay nullptr).
        //   id_width==4 → l3_col_idx32 / l3_edge_ids32 point at uint32
        //                 sections (tag-stripped ordinals); l3_dst_tag /
        //                 l3_eid_tag carry the per-section ObjectId type tag
        //                 PRE-SHIFTED into the top byte (tag << 56), OR'd onto
        //                 each widened value by the for_each_* helpers to
        //                 reconstruct the exact tagged ObjectId. This keeps
        //                 the hot tier-3 dispatch zero-copy + zero-alloc for
        //                 BOTH widths over the papers100M cold tail.
        const uint64_t*           l3_col_idx   = nullptr;
        const uint64_t*           l3_edge_ids  = nullptr;  // may be nullptr
        const uint32_t*           l3_col_idx32  = nullptr;
        const uint32_t*           l3_edge_ids32 = nullptr;  // may be nullptr
        uint64_t                  l3_dst_tag    = 0;        // (dst tag << 56)
        uint64_t                  l3_eid_tag    = 0;        // (edge tag << 56)
        std::size_t               l3_size    = 0;
        // L4: BPT direct returns full Neighbors copies — the
        // dispatcher owns the allocation here.
        std::vector<AdjEntry>     l4_owned{};
        // Which tier produced the result.
        uint8_t                   tier = 0;

        bool empty() const noexcept;
        std::size_t size() const noexcept;

        /**
         * @brief Iterate destination ids only (edge-id-agnostic path).
         *
         * Consumers that don't care about edge ids (the GraphSAGE
         * sampler in its hot path, EmbeddingWriter Phase B
         * traversal) can use this helper to walk the result without
         * branching on `tier`. The callback is invoked once per
         * destination in storage order.
         *
         * Callback signature: `void(uint64_t dst_node_id)`.
         */
        template <typename Fn>
        void for_each_dst(Fn&& callback) const {
            switch (tier) {
                case 1:
                    for (std::size_t i = 0; i < l1.size; ++i) {
                        callback(l1.data[i].node_id);
                    }
                    break;
                case 2:
                    for (std::size_t i = 0; i < l2_size; ++i) {
                        callback(static_cast<uint64_t>(l2_col_idx[i]));
                    }
                    break;
                case 3:
                    if (l3_col_idx32 != nullptr) {
                        // Narrow (uint32) layout: widen + re-apply the tag.
                        for (std::size_t i = 0; i < l3_size; ++i) {
                            callback(l3_dst_tag
                                     | static_cast<uint64_t>(l3_col_idx32[i]));
                        }
                    } else {
                        for (std::size_t i = 0; i < l3_size; ++i) {
                            callback(l3_col_idx[i]);
                        }
                    }
                    break;
                case 4:
                    for (const auto& e : l4_owned) {
                        callback(e.node_id);
                    }
                    break;
                default:
                    break;
            }
        }

        /**
         * @brief Iterate `(dst_node_id, edge_id)` pairs.
         *
         * For tiers without edge ids in their layout (currently L2 and
         * L3-without-edge-ids) the callback receives `0` as the edge
         * id. Callers that need true edge-id resolution for L2/L3
         * nodes must fall through to L4 explicitly; the four-level
         * store does not promote on a per-call basis.
         *
         * Callback signature:
         *   `void(uint64_t dst_node_id, uint64_t edge_id_or_zero)`.
         */
        template <typename Fn>
        void for_each_with_edge_id(Fn&& callback) const {
            switch (tier) {
                case 1:
                    for (std::size_t i = 0; i < l1.size; ++i) {
                        callback(l1.data[i].node_id, l1.data[i].edge_id);
                    }
                    break;
                case 2:
                    for (std::size_t i = 0; i < l2_size; ++i) {
                        callback(static_cast<uint64_t>(l2_col_idx[i]),
                                 uint64_t{0});
                    }
                    break;
                case 3:
                    if (l3_col_idx32 != nullptr) {
                        // Narrow (uint32) layout: widen + re-apply both tags.
                        for (std::size_t i = 0; i < l3_size; ++i) {
                            const uint64_t eid =
                                (l3_edge_ids32 != nullptr)
                                    ? (l3_eid_tag
                                       | static_cast<uint64_t>(l3_edge_ids32[i]))
                                    : 0ULL;
                            callback(l3_dst_tag
                                     | static_cast<uint64_t>(l3_col_idx32[i]),
                                     eid);
                        }
                    } else {
                        for (std::size_t i = 0; i < l3_size; ++i) {
                            const uint64_t eid =
                                (l3_edge_ids != nullptr) ? l3_edge_ids[i] : 0ULL;
                            callback(l3_col_idx[i], eid);
                        }
                    }
                    break;
                case 4:
                    for (const auto& e : l4_owned) {
                        callback(e.node_id, e.edge_id);
                    }
                    break;
                default:
                    break;
            }
        }
    };

    /**
     * @brief Build configuration. All fields have safe defaults; the
     *        only mandatory consumer-side decision is `orientation`.
     */
    struct Config {
        /// L1 (RAM hot hash) budget in bytes. 0 = auto-detect (25% of
        /// 70% of MemAvailable from `src/misc/available_ram.h`).
        std::size_t l1_budget_mb = 0;

        /// L2 (RAM warm compact-CSR) budget in bytes. 0 = auto-detect
        /// (75% of 70% of MemAvailable).
        std::size_t l2_budget_mb = 0;

        /// Open the Spec #4-B mmap sidecar files for the L3 cold tier
        /// when present. Composes with `indexSet` — the sidecar is
        /// silently absent when the projection wasn't built with
        /// `buildTopologySnapshot:true`.
        bool use_l3_mmap_sidecar = false;

        /// Edge orientation that drives which directions to build:
        ///   - NATURAL    -> forward L1+L2 only.
        ///   - REVERSE    -> reverse L1+L2 only.
        ///   - UNDIRECTED -> both forward + reverse.
        EdgeOrientation orientation = EdgeOrientation::UNDIRECTED;
    };

    // ------------------------------------------------------------------
    //  Phase 2 dispatcher constructor (kept for unit-test compatibility).
    // ------------------------------------------------------------------
    //
    // Pre-built tier sources are passed in by reference / pointer; the
    // resulting store does NOT own L1/L2 (they live in caller scope) and
    // `build()` is a no-op. Used exclusively by
    // `four_level_topology_store_test.cc` to drive the dispatcher with
    // synthetic L1/L2 fixtures.
    //
    // For real usage, prefer the Phase 3 constructor below + `build()`.

    using L4Lookup = std::function<std::vector<AdjEntry>(ObjectId)>;
    using RowLookup = std::function<uint64_t(ObjectId)>;

    FourLevelTopologyStore(
        const L1HashCache&                              l1_fwd,
        const L1HashCache&                              l1_rev,
        const L2CompactCsr&                             l2_fwd,
        const L2CompactCsr&                             l2_rev,
        const GQL::Projection::TopologySnapshotReader*  l3_fwd,
        const GQL::Projection::TopologySnapshotReader*  l3_rev,
        L4Lookup                                        l4_fwd,
        L4Lookup                                        l4_rev,
        const std::vector<uint8_t>&                     tier_lookup,
        RowLookup                                       row_lookup,
        Config                                          config);

    // ------------------------------------------------------------------
    //  Phase 3 build constructor.
    // ------------------------------------------------------------------
    //
    // Constructs a store wired to live B+Trees + an optional storage
    // pointer (used to source `topology_*.csr` sidecars when the L3
    // tier is enabled and the projection has them on disk). After
    // construction the store is *not* yet built — call `build()` to
    // populate the tiers. `is_built()` reports whether the store is
    // ready for queries.
    //
    // Lifetimes: `fwd_bpt` / `rev_bpt` / `storage` (when non-null)
    // must outlive the store.
    //
    /// @note `fwd_bpt` or `rev_bpt` may individually be null when the
    ///       projection is restricted to one direction
    ///       (NATURAL → fwd only, REVERSE → rev only). When both are
    ///       null, `build()` produces an empty store and the dispatcher
    ///       returns empty `Neighbors` for all queries. Synthetic-test
    ///       paths that drive the dispatcher with pre-built tiers use
    ///       the Phase 2 reference-taking constructor instead.
    FourLevelTopologyStore(BPlusTree<3>*               fwd_bpt,
                           BPlusTree<3>*               rev_bpt,
                           GQL::ProjectionStorage*     storage,
                           std::filesystem::path       projection_dir,
                           Config                      config);

    FourLevelTopologyStore(const FourLevelTopologyStore&)            = delete;
    FourLevelTopologyStore& operator=(const FourLevelTopologyStore&) = delete;

    ~FourLevelTopologyStore();

    /**
     * @brief Orchestrate the build phase (Phase 3 constructor only).
     *
     * Sequence (per design §2.3):
     *   1. Auto-detect L1/L2 budgets from /proc/meminfo when
     *      Config::l1_budget_mb / l2_budget_mb are 0.
     *   2. Run the frequency profiler (degree proxy when no
     *      `node_counts.bin` exists yet).
     *   3. Compute tier_assignment[] greedily by descending frequency.
     *   4. Walk the live B+Trees, populating L1 (reserve+move) and L2
     *      (add_node + freeze) per tier_assignment[].
     *   5. When `use_l3_mmap_sidecar`, open the Spec #4-B sidecar
     *      readers (silently no-op when the files are absent).
     *   6. Mark the store built.
     *
     * Throws std::logic_error when called on the dispatcher
     * constructor or when called twice. Throws std::runtime_error on
     * any underlying I/O / B+Tree failure (no silent partial builds).
     */
    void build();

    /**
     * @brief Whether the store is ready for queries.
     *
     * Always true for the dispatcher constructor (caller-managed
     * tiers). True for the Phase 3 constructor only after a
     * successful `build()`.
     */
    bool is_built() const noexcept;

    /**
     * @brief Get outgoing neighbors (NATURAL).
     *
     * @throws std::logic_error when the store was constructed via the
     *         Phase 3 ctor and `build()` has not been called.
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

    /**
     * @brief Get neighbors per Config::orientation.
     *
     * Convenience wrapper. NATURAL -> get_out_neighbors,
     * REVERSE -> get_in_neighbors, UNDIRECTED -> merges fwd + rev
     * (the merge step allocates).
     */
    Neighbors get_neighbors(ObjectId v) const;

    const Config& config() const noexcept { return config_; }

    // -----------------------------
    //  Diagnostics (Phase 3 ctor)
    // -----------------------------
    std::size_t l1_node_count() const noexcept;
    std::size_t l2_node_count() const noexcept;
    std::size_t l3_node_count() const noexcept;
    std::size_t l4_node_count() const noexcept;

    /// Sum of L1 + L2 resident bytes (using the `kL1*` / `kL2*` Phase
    /// 1 contracts) across both directions. L3 / L4 are mmap / disk
    /// resident and do not count toward in-RAM accounting.
    std::size_t total_ram_used() const noexcept;

    /// Phase 4 / T13.11: MinHash permutation over L3-tier nodes (empty
    /// on cold start). Exposed for testing and for future Phase 5+
    /// consumers that will apply the permutation to the on-disk L3
    /// sidecar.
    const std::vector<uint64_t>& l3_reorder_permutation() const noexcept {
        return l3_reorder_permutation_;
    }

private:
    // Helper that performs the tier switch for one direction. Both
    // public per-direction methods bind their tier sources before
    // delegating here.
    Neighbors dispatch_(
        ObjectId                                        v,
        const L1HashCache&                              l1,
        const L2CompactCsr&                             l2,
        const GQL::Projection::TopologySnapshotReader*  l3,
        const L4Lookup&                                 l4) const;

    // -------------------------------------------------
    //  Build helpers (only used by the Phase 3 ctor)
    // -------------------------------------------------
    void auto_detect_budgets_(std::size_t& l1_bytes,
                              std::size_t& l2_bytes) const;
    /**
     * @brief Stream-distribute BPT records into L1 / L2 by tier.
     *
     * Walks `index` once in `(src, dst, edge_id)` lexicographic order
     * (the BPT iterator's natural order — verified at
     * `bplus_tree.cc::BptIter<N>::next()`). Buffers the neighbors of
     * the *current* src in a single `staging_buffer` and flushes that
     * buffer to L1 (tier 1), L2 (tier 2), or drops it (tier 3 / 4)
     * the moment the src key advances. Peak transient memory is
     * therefore O(max_node_degree × sizeof(AdjEntry)) — bounded by
     * ~2 MB on papers100M scale rather than ~50 GB if all per-node
     * vectors were materialized first.
     *
     * @param frequency Per-row directed-edge degree, used purely as a
     *                  `reserve(degree_hint)` so the staging buffer's
     *                  growth doubling doesn't allocate beyond the
     *                  actual degree (and so L1HashCache::total_bytes
     *                  reports a tight estimate). May be empty in
     *                  test paths; the streaming distribution stays
     *                  correct without it (only the reserve hint is
     *                  lost).
     */
    void populate_direction_(
        BPlusTree<3>*                                                 index,
        const std::vector<uint8_t>&                                    tiers,
        const std::vector<uint64_t>&                                   frequency,
        const GQL::Projection::TopologySnapshotReader*                 sidecar,
        std::unique_ptr<L1HashCache>&                                  l1_out,
        std::unique_ptr<L2CompactCsr>&                                 l2_out) const;
    /**
     * @brief Fast L1+L2 build path that reads the Spec #4-B sidecar
     *        directly instead of walking the B+Tree (Phase 4 / T13.10).
     *
     * Walks `row_idx` in [0, sidecar.num_nodes()), looks up the tier,
     * and dispatches the slice into L1 (reserve+move) or L2 (add_node).
     * Tier-3 / tier-4 nodes are skipped. The sidecar already exposes
     * O(1) per-node mmap reads, so the inner loop is ~1-5 us/node vs
     * the BPT path's ~30-100 us/node — translates to roughly 5x faster
     * build on arxiv/products and ~5-10 minutes saved on papers100M.
     *
     * Uses the same `node_to_l2_idx_` ordering invariant as the BPT
     * path: both traverse row_idx ascending and call `L2CompactCsr::
     * add_node` in the same order, so the post-freeze map is bit-
     * identical between the two paths.
     */
    void populate_direction_via_sidecar_(
        const GQL::Projection::TopologySnapshotReader& sidecar,
        const std::vector<uint8_t>&                    tiers,
        const std::vector<uint64_t>&                   frequency,
        std::unique_ptr<L1HashCache>&                  l1_out,
        std::unique_ptr<L2CompactCsr>&                 l2_out) const;
    void open_l3_sidecars_();
    /**
     * @brief Compute a permutation that clusters L3-tier nodes by
     *        sample-set similarity (Phase 4 / T13.11).
     *
     * Uses `MinHashReorderer::Strategy::SEGMENTED` (DiskGNN Algorithm
     * 1, validated by Spec #5). Only fires when the frequency profiler
     * consumed `<projection_dir>/node_counts.bin` (warm start). On
     * cold start the permutation is left empty and a one-line cerr
     * message is emitted documenting the skip reason.
     *
     * @note The permutation is STORED but not APPLIED. Rewriting the
     *       on-disk `topology_*.csr` sidecar requires a full rebuild
     *       and is deferred to a future Phase 5+ task. The infra is
     *       laid here so when `gnn_offline_sample` learns to persist
     *       `node_counts.bin` (Phase 5), the permutation becomes
     *       available to downstream consumers.
     */
    void compute_l3_minhash_reorder_(bool warm_start_used);

    // Tier sources — Phase 3 ctor owns them; Phase 2 dispatcher ctor
    // leaves them null and uses the caller-provided `*_ref_` members.
    std::unique_ptr<L1HashCache>                       owned_l1_fwd_;
    std::unique_ptr<L1HashCache>                       owned_l1_rev_;
    std::unique_ptr<L2CompactCsr>                      owned_l2_fwd_;
    std::unique_ptr<L2CompactCsr>                      owned_l2_rev_;
    std::unique_ptr<GQL::Projection::TopologySnapshotReader>
                                                       owned_l3_fwd_;
    std::unique_ptr<GQL::Projection::TopologySnapshotReader>
                                                       owned_l3_rev_;
    std::vector<uint8_t>                               owned_tier_assignment_;

    // Phase 4 / T13.11: MinHash permutation over L3-tier nodes. Empty on
    // cold start (no `node_counts.bin` yet) — populated only when warm
    // start is reached. Currently stored but not applied; future work
    // (Phase 5+) consumes it to rewrite the L3 sidecar layout.
    std::vector<uint64_t>                              l3_reorder_permutation_;

    // References to whichever tier sources are active (owned-or-borrowed).
    const L1HashCache*                                 l1_fwd_ = nullptr;
    const L1HashCache*                                 l1_rev_ = nullptr;
    const L2CompactCsr*                                l2_fwd_ = nullptr;
    const L2CompactCsr*                                l2_rev_ = nullptr;
    const GQL::Projection::TopologySnapshotReader*     l3_fwd_ = nullptr;
    const GQL::Projection::TopologySnapshotReader*     l3_rev_ = nullptr;
    L4Lookup                                           l4_fwd_;
    L4Lookup                                           l4_rev_;

    // tier_lookup_ref_ points either into owned_tier_assignment_ (Phase
    // 3 ctor) or into a caller-provided const vector (Phase 2 ctor).
    const std::vector<uint8_t>*                        tier_lookup_ref_ = nullptr;
    RowLookup                                          row_lookup_;

    // Phase 3 ctor only:
    BPlusTree<3>*                                      fwd_bpt_       = nullptr;
    BPlusTree<3>*                                      rev_bpt_       = nullptr;
    GQL::ProjectionStorage*                            storage_       = nullptr;
    std::filesystem::path                              projection_dir_;
    bool                                               phase3_ctor_   = false;
    bool                                               built_         = false;

    Config                                             config_;
};

}  // namespace mdb::gnn
