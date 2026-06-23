#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "gnn/projection/adj_entry.h"
#include "gnn/projection/pinned_topology_view.h"  // HostCsrArrays (complete type)
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
class PinnedTopologyView;
struct SamplingBackendPlan;

namespace detail {
// Merge a node's directional out+in lists into the canonical UNDIRECTED order
// (out(u) first, then in(u) survivors). The dedup key is the edge_id when
// has_edge_ids (distinct edge_ids -> nothing removed, so parallel/mutual edges
// are PRESERVED), else the neighbor node id. Replicates
// TopologyAccessor::get_neighbors_into(UNDIRECTED). `out` is cleared and filled.
void symmetric_merge_row(const std::vector<uint64_t>& dst_fwd,
                         const std::vector<uint64_t>& eid_fwd,
                         const std::vector<uint64_t>& dst_rev,
                         const std::vector<uint64_t>& eid_rev,
                         bool has_edge_ids,
                         std::vector<AdjEntry>& out);
}  // namespace detail

// Merge two narrow uint32 directional CSRs into one undirected CSR (the in-RAM
// substrate the GPU-UVA single slice pins).
//   _concat:     out(u) ++ in(u) with NO dedup (the has_edge_ids / distinct-edge
//                case — parallel/mutual edges preserved).
//   _node_dedup: out(u) ++ (in(u) not already present) (the edge_id==0 case).
// Per-row peak O(degree); out_row_ptr is sized from the realized per-row counts.
void merge_symmetric_csr_concat(
    const std::vector<uint64_t>& fwd_row_ptr, const std::vector<uint32_t>& fwd_col_idx,
    const std::vector<uint64_t>& rev_row_ptr, const std::vector<uint32_t>& rev_col_idx,
    std::vector<uint64_t>& out_row_ptr, std::vector<uint32_t>& out_col_idx);
void merge_symmetric_csr_node_dedup(
    const std::vector<uint64_t>& fwd_row_ptr, const std::vector<uint32_t>& fwd_col_idx,
    const std::vector<uint64_t>& rev_row_ptr, const std::vector<uint32_t>& rev_col_idx,
    std::vector<uint64_t>& out_row_ptr, std::vector<uint32_t>& out_col_idx);

/**
 * @brief Coordinator for the frequency-tiered Four-Level Topology Store.
 *
 * Implements a four-tier neighbor lookup hierarchy (L1 RAM hash / L2
 * compact uint32 CSR / L3 mmap sidecar / L4 direct B+Tree) where tier
 * assignment is driven by per-node access frequency: high-frequency hub
 * nodes go to fast RAM tiers, cold-tail nodes are served from disk.
 *
 * The full coordinator (constructed with the Phase 3 B+Tree constructor):
 * replaces the earlier dispatcher-only skeleton (Phase 2 constructor, kept
 * for unit tests) with a full build+dispatch cycle that:
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
        // L2: per-direction dst ObjectId type tag, PRE-SHIFTED (tag << 56).
        //   l2_col_idx stores tag-stripped uint32 ordinals (see
        //   l2_compact_csr.cc); for_each_* OR this back to reconstruct the
        //   exact tagged ObjectId — identical to the l3_dst_tag convention.
        //   0 for an empty L2 (harmless: empty spans emit nothing).
        uint64_t                  l2_dst_tag = 0;
        // L3: dst node ids, length l3_size. Owned by mmap (zero-copy).
        // Two on-disk widths (the topology CSR sidecar may be stored as
        // uint64 or uint32 depending on how the projection was built):
        //   id_width==8 → l3_col_idx / l3_edge_ids point at uint64 sections
        //                 (full tagged ObjectIds; l3_*_32 stay nullptr).
        //   id_width==4 → l3_col_idx32 / l3_edge_ids32 point at uint32
        //                 sections (tag-stripped ordinals; the 8-bit ObjectId
        //                 type tag is stripped when building the sidecar and
        //                 must be reconstructed at read time). l3_dst_tag /
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
                    // L2 col_idx holds tag-stripped uint32 ordinals; OR the
                    // per-direction dst tag back to recover the tagged ObjectId.
                    for (std::size_t i = 0; i < l2_size; ++i) {
                        callback(l2_dst_tag
                                 | static_cast<uint64_t>(l2_col_idx[i]));
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
                    // L2 omits edge ids by design (callback gets 0); re-apply
                    // the dst tag to the tag-stripped uint32 ordinal.
                    for (std::size_t i = 0; i < l2_size; ++i) {
                        callback(l2_dst_tag
                                 | static_cast<uint64_t>(l2_col_idx[i]),
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

        /// Open the topology CSR sidecar files (`topology_fwd.csr` /
        /// `topology_rev.csr`) as the L3 cold-tier source when they are
        /// present on disk. These mmap-backed CSR files provide O(1)
        /// per-node neighbor slices (no B+Tree directory walk). The sidecar
        /// is silently absent when the projection wasn't built with
        /// `buildTopologySnapshot:true`, in which case L3 is skipped and
        /// cold-tail nodes fall through to the L4 B+Tree.
        bool use_l3_mmap_sidecar = false;

        /// Edge orientation that drives which directions to build:
        ///   - NATURAL    -> forward L1+L2 only.
        ///   - REVERSE    -> reverse L1+L2 only.
        ///   - UNDIRECTED -> both forward + reverse.
        EdgeOrientation orientation = EdgeOrientation::UNDIRECTED;

        /// Drop edge_ids from the symmetric (pre-merged undirected) tier: the
        /// merge keys dedup on node-id (so the emitted edge_ids are zeroed) and
        /// the receptive field collapses parallel/mutual edges. Default false
        /// (keep the edge_id-keyed merge that preserves duplicates byte-for-byte
        /// with the accessor). When true AND the graph has parallel edges the
        /// build REFUSES the drop (sym_refused_edge_id_drop()), leaving the
        /// symmetric tier unbuilt so the runtime out+in+merge fallback engages.
        bool drop_edge_ids = false;

        /// Build the symmetric (pre-merged undirected) tier for UNDIRECTED
        /// orientation. Default true. When false, get_neighbors(UNDIRECTED) uses
        /// the runtime out+in+merge fallback (same dedup rule, so byte-identical)
        /// — the bit-reproducible reference for the symmetric ON/OFF gate.
        bool build_symmetric_tier = true;
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
     *   5. When `use_l3_mmap_sidecar`, open the topology CSR sidecar
     *      readers (`topology_fwd.csr` / `topology_rev.csr`) for mmap-backed
     *      O(1) neighbor slices at the L3 cold tier (silently no-op when the
     *      files are absent from the projection directory).
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
     * @brief Whether the pre-merged symmetric (undirected) tier is populated.
     *
     * When true, get_neighbors(UNDIRECTED) collapses to a single dispatch over
     * the symmetric tier (l1_sym_ / l2_sym_ / l3_sym_ / l4_sym_) instead of the
     * runtime out+in+merge. Built only for UNDIRECTED orientation; false
     * otherwise. The symmetric tier reuses the SAME per-node tier assignment
     * (tier_lookup_ref_ / row_lookup_) as the directional tiers — it is
     * direction-agnostic.
     */
    bool is_symmetric_built() const noexcept { return sym_built_; }

    /// True iff a requested edge_id-drop (Config::drop_edge_ids) was REFUSED
    /// because the graph has parallel edges, which node-id dedup would collapse.
    /// When this is true the symmetric tier is left unbuilt and undirected
    /// fetches use the runtime out+in+merge fallback (preserving edge_ids).
    bool sym_refused_edge_id_drop() const noexcept {
        return sym_refused_edge_id_drop_;
    }

    /**
     * @brief Read-only access to the tier-2 compact CSR for each direction.
     *
     * Used only to SIZE the dynamic GPU/CPU sampling backend decision
     * (`plan_sampling_backend` reads node_count()/edge_count()). Returns the
     * active L2 CSR pointer for that direction (post-build, frozen, immutable)
     * or nullptr when that direction has no L2 tier. Callers must NOT cache the
     * raw row_ptr_/col_idx_ data pointers across the store's lifetime.
     */
    const L2CompactCsr* l2_fwd() const noexcept { return l2_fwd_; }
    const L2CompactCsr* l2_rev() const noexcept { return l2_rev_; }

    /**
     * @brief The active L3 global topology sidecar reader per direction.
     *
     * This is the substrate `enable_pinned_gpu_view` actually pins, so the GPU
     * sampling-backend decision must SIZE on these (num_nodes/num_edges over the
     * whole graph), not the warm-tier L2. Returns nullptr when no L3 sidecar is
     * active for that direction.
     */
    const GQL::Projection::TopologySnapshotReader* l3_fwd() const noexcept {
        return l3_fwd_;
    }
    const GQL::Projection::TopologySnapshotReader* l3_rev() const noexcept {
        return l3_rev_;
    }

    /**
     * @brief Pin the global topology CSR as a device-visible GPU substrate.
     *
     * Builds a `PinnedTopologyView` over the active L3 sidecar arrays (the
     * global ROW_PTR + narrow COL_IDX, correctness-complete across all nodes)
     * for whichever directions `plan.directions` selects, registering them with
     * `cudaHostRegister` so a k-hop kernel can walk them over PCIe.
     *
     * No-op when `plan.backend == CPU_OUT_OF_CORE`, when no capable GPU is
     * present at runtime, when the build has no CUDA, or when the L3 sidecar is
     * absent / not the narrow uint32 layout (the only pinnable global substrate
     * today). Idempotent: re-enabling first releases the prior view. The store
     * stays read-only and the CPU sampling path is unchanged, so output remains
     * byte-identical when the view is unused.
     */
    void enable_pinned_gpu_view(const SamplingBackendPlan& plan);

    /// The registered GPU view, or nullptr when none was enabled. Owned by the
    /// store; valid until the next `enable_pinned_gpu_view()` or destruction.
    const PinnedTopologyView* pinned_view() const noexcept {
        return pinned_view_.get();
    }

    /**
     * @brief Build (once, idempotent) an in-RAM merged undirected CSR from the
     *        active L3 fwd+rev narrow readers and cache it.
     *
     * Replicates the accessor's undirected emission (out ++ in, dedup keyed on
     * node-id when edge_ids are absent/zero, concat when edge_ids are distinct)
     * so the merged slice is the byte-identical receptive field the runtime
     * out+in+merge produces. Returns a stable pointer to the cached HostCsrArrays,
     * or nullptr when neither L3 narrow (id_width==4) reader is present (the GPU
     * path then stays off and the CPU merge runs unchanged). NOT a second build():
     * pure additive RAM allocation, safe to call after build(). Idempotent.
     */
    const HostCsrArrays* materialize_symmetric_arrays();

    /// Resident bytes of the materialized symmetric arrays ((N+1)*8 + M*4),
    /// 0 when not built. For RAM diagnostics + telemetry.
    std::size_t symmetric_ram_bytes() const noexcept;

    /**
     * @brief Release the directional fwd/rev tiers once the symmetric slice is
     *        materialized + pinned, reclaiming their RAM.
     *
     * Once materialize_symmetric_arrays() has built the merged undirected slice
     * (sym_arrays_, the GPU-pinned full-coverage CSR), the directional tiers are
     * dead weight on the symmetric GPU path: the GPU kernel walks ONLY the
     * pinned slice (never l3_fwd_/l3_rev_), and the CPU UNDIRECTED dispatch uses
     * the symmetric tier (l*_sym_) whose L4 fallback closes over the BPTs, not
     * the directional L3 readers. This frees the owned directional L1/L2 heap
     * caches and munmaps the two L3 sidecars (the only way to drop their
     * faulted-in page-cache — there is no madvise(DONTNEED) path), then nulls
     * the borrowed aliases so any stray directional dispatch faults loudly.
     *
     * Intended caller: the offline sampling engine, AFTER enable_pinned_gpu_view
     * registered a symmetric pin and AFTER the one-time backend-sizing block
     * (which reads l3_fwd()/l3_rev()/total_ram_used()) has run. No-op (returns 0)
     * when the slice was never materialized — i.e. the CPU path, which still
     * reads the directional tiers — or when already released (idempotent).
     *
     * @return Best-effort estimate of the bytes released (owned L1/L2 heap +
     *         the munmapped L3 sidecar footprints); 0 on no-op.
     * @post   get_out_neighbors()/get_in_neighbors() throw std::logic_error
     *         (NATURAL/REVERSE directional fetches are no longer serviceable);
     *         get_neighbors(UNDIRECTED) over the symmetric tier is unaffected.
     */
    std::size_t release_directional_after_symmetric_pin();

    /**
     * @brief Free the directional fwd/rev tiers BEFORE the symmetric slice is
     *        pinned. Valid only when a baked topology_sym.csr is confirmed
     *        present.
     *
     * Same effect as release_directional_after_symmetric_pin(), but intended to
     * run *before* enable_pinned_gpu_view() on the baked path: there
     * materialize_symmetric_arrays() copies the merged slice from the
     * topology_sym.csr mmap and never reads l3_fwd_/l3_rev_, so the tiers are
     * already dead and freeing them up front keeps the ~13 GB heap slice from
     * coexisting with the ~15 GB tiers during the copy+pin (transient host peak
     * ~28 -> ~13 GB on papers100M). Does NOT require the slice to be
     * materialized (no sym_arrays_built_ gate); idempotent. MUST NOT be used on
     * the in-RAM fallback merge path, which still consumes the tiers.
     *
     * @return Best-effort estimate of the bytes released; 0 on no-op.
     */
    std::size_t release_directional_for_baked_symmetric();

    /// True once release_directional_after_symmetric_pin() has run.
    bool directional_released() const noexcept { return directional_released_; }

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
    /// Node counts held by the symmetric (pre-merged undirected) L1/L2 tiers;
    /// 0 when the symmetric tier is not built.
    std::size_t l1_sym_node_count() const noexcept;
    std::size_t l2_sym_node_count() const noexcept;

    /// Sum of L1 + L2 resident bytes (using the `kL1*` / `kL2*` Phase
    /// 1 contracts) across both directions. L3 / L4 are mmap / disk
    /// resident and do not count toward in-RAM accounting.
    std::size_t total_ram_used() const noexcept;

    /// MinHash reorder permutation over L3-tier nodes (empty on cold
    /// start; populated only when `node_counts.bin` was available at
    /// build time). Exposed for testing and for future consumers that
    /// will apply the permutation to the on-disk L3 sidecar to improve
    /// disk-read locality.
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

    // Shared implementation behind release_directional_after_symmetric_pin() and
    // release_directional_for_baked_symmetric(): frees the owned directional
    // L1/L2 heap caches, munmaps the two L3 directional sidecars, nulls the
    // borrowed aliases, and sets directional_released_. Assumes the caller has
    // already checked the appropriate precondition.
    std::size_t release_directional_tiers_();

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
     * @brief Fast L1+L2 build path that reads the topology CSR sidecar
     *        (`topology_fwd.csr` / `topology_rev.csr`) directly instead
     *        of walking the B+Tree.
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
    // Populate the symmetric (pre-merged undirected) L1/L2 tier by merging the
    // two directional L3 sidecars row-by-row via detail::symmetric_merge_row.
    // Reuses the same per-node tier assignment; tier-3/4 rows are skipped (the
    // L4 symmetric merge covers them at runtime). Used when no on-disk
    // topology_sym.csr is present (the merge-fallback path).
    void populate_direction_symmetric_(
        const GQL::Projection::TopologySnapshotReader& l3_fwd,
        const GQL::Projection::TopologySnapshotReader& l3_rev,
        const std::vector<uint8_t>&                    tiers,
        const std::vector<uint64_t>&                   frequency,
        std::unique_ptr<L1HashCache>&                  l1_out,
        std::unique_ptr<L2CompactCsr>&                 l2_out) const;
    void open_l3_sidecars_();
    /**
     * @brief Compute a MinHash permutation that clusters L3-tier nodes by
     *        sample-set similarity so that nearby seeds in the random-walk
     *        order share L3 mmap pages, improving disk read locality.
     *
     * Uses `MinHashReorderer::Strategy::SEGMENTED` (the GLOBAL composite-hash
     * ordering `(segment_id<<32) | minhash` from DiskGNN Algorithm 1). Only
     * fires when the frequency profiler has consumed
     * `<projection_dir>/node_counts.bin` (warm start — requires a prior
     * completed sample run to have written that file). On a cold start (no
     * `node_counts.bin` present) the permutation is left empty and a one-line
     * diagnostic is written to stderr documenting the skip reason.
     *
     * @note The permutation is STORED but not APPLIED to disk. Rewriting the
     *       on-disk `topology_*.csr` sidecar to the reordered layout requires
     *       a full file rebuild pass and is not yet implemented; the
     *       permutation vector is exposed here so that a future pass can
     *       consume it directly without re-running the MinHash computation.
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
    // Symmetric (pre-merged undirected) tier — owned when the UNDIRECTED build
    // populates it (from topology_sym.csr or by merging the two directional
    // sidecars). Reuses owned_tier_assignment_ (per-node, direction-agnostic).
    std::unique_ptr<L1HashCache>                       owned_l1_sym_;
    std::unique_ptr<L2CompactCsr>                      owned_l2_sym_;
    std::unique_ptr<GQL::Projection::TopologySnapshotReader>
                                                       owned_l3_sym_;
    std::vector<uint8_t>                               owned_tier_assignment_;

    // MinHash permutation over L3-tier nodes for disk-read locality.
    // Empty on cold start (no `node_counts.bin` yet) — populated only when
    // a prior sample run has written that access-count file (warm start).
    // Currently stored but not applied to the on-disk sidecar; a future
    // reorder pass will consume it to rewrite the L3 sidecar layout.
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
    // Active symmetric tier references (owned-or-borrowed). Null until an
    // UNDIRECTED build populates them; gated by sym_built_.
    const L1HashCache*                                 l1_sym_ = nullptr;
    const L2CompactCsr*                                l2_sym_ = nullptr;
    const GQL::Projection::TopologySnapshotReader*     l3_sym_ = nullptr;
    L4Lookup                                           l4_sym_;

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
    // Set true once the symmetric tier is populated (UNDIRECTED build only).
    bool                                               sym_built_     = false;
    // Set true when Config::drop_edge_ids was requested but refused due to a
    // parallel-edge multigraph (sym tier left unbuilt -> runtime merge fallback).
    bool                                               sym_refused_edge_id_drop_ = false;
    // Set true once release_directional_after_symmetric_pin() freed the
    // directional fwd/rev tiers. After this the directional l1/l2/l3 pointers
    // are null and get_out_neighbors/get_in_neighbors throw.
    bool                                               directional_released_ = false;

    Config                                             config_;

    // Optional device-visible view over the global topology CSR for GPU-UVA
    // sampling. Declared LAST so it is destroyed FIRST — its cudaHostUnregister
    // must run before the L3 sidecar readers / L2 caches it pins are freed.
    // nullptr unless enable_pinned_gpu_view() registered one.
    std::unique_ptr<PinnedTopologyView>                pinned_view_;

    // In-RAM merged undirected CSR for the GPU-UVA single slice (Part D). Owns
    // its backing vectors so the pinned pages stay valid; sym_arrays_ points into
    // them. Built lazily by materialize_symmetric_arrays(). NOTE: distinct from
    // the Part C sym_built_ tier-dispatch flag — this is the flat uint32 CSR for
    // pinning, gated by its own sym_arrays_built_.
    std::vector<uint64_t>                              sym_row_ptr_;
    std::vector<uint32_t>                              sym_col_idx_;
    HostCsrArrays                                      sym_arrays_{};
    bool                                               sym_arrays_built_ = false;
    // True when sym_arrays_ points at the l3_sym_ mmap (a baked topology_sym.csr
    // opened zero-copy) rather than the owned sym_row_ptr_/sym_col_idx_ heap
    // vectors. When true, owned_l3_sym_ backs the pin and must outlive it (and
    // the heap vectors are empty). When false, the slice was merged in RAM.
    bool                                               sym_slice_is_mmap_ = false;
};

}  // namespace mdb::gnn
