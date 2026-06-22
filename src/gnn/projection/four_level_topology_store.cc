#include "gnn/projection/four_level_topology_store.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gnn/projection/pinned_topology_view.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/projection/topology_frequency_profiler.h"
#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sampling_backend_plan.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "graph_models/gql/projection/topology_symmetric_merge.h"  // detect_parallel_edges
#include "misc/available_ram.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace mdb::gnn {

namespace {

/// Walk a B+Tree once, grouping records by their src key (record[0]).
/// Returns a map node_id -> adjacency list. Used by both the L4 bridge
/// closure and the in-build per-node materialisation pass.
void scan_index_into_(
    BPlusTree<3>*                                              index,
    std::unordered_map<uint64_t, std::vector<AdjEntry>>&       out)
{
    if (index == nullptr) return;
    Record<3> min_record = {0, 0, 0};
    Record<3> max_record = {UINT64_MAX, UINT64_MAX, UINT64_MAX};
    bool interruption = false;
    auto it = index->get_range(&interruption, min_record, max_record);
    const Record<3>* rec;
    while ((rec = it.next()) != nullptr) {
        const uint64_t a = std::get<0>(*rec);
        const uint64_t b = std::get<1>(*rec);
        const uint64_t e = std::get<2>(*rec);
        out[a].push_back(AdjEntry{ b, e });
    }
}

// Read the cumulative "read_bytes" counter from /proc/self/io — the bytes
// this process actually fetched from the block device (NOT page-cache hits).
// Used to measure the physical/logical read amplification of the populate
// phase under MADV_RANDOM (i.e., how many bytes the kernel had to read from
// disk vs. how many logical bytes the L1/L2 tier covers). Returns 0 on any
// failure (non-Linux, permission, parse) so the diagnostic degrades silently.
uint64_t read_proc_io_read_bytes_() {
    std::ifstream f("/proc/self/io");
    if (!f) return 0;
    std::string key;
    uint64_t val = 0;
    while (f >> key) {
        if (key == "read_bytes:") {
            f >> val;
            return val;
        }
        f >> val;  // skip this field's value
    }
    return 0;
}

// Milliseconds elapsed between two steady_clock samples.
double elapsed_ms_(std::chrono::steady_clock::time_point a,
                   std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Magic of `<projection_dir>/node_counts.bin`. Must stay in sync with the
// writers (offline_sampling_engine.cc / node_counts_io.cc) and the canonical
// reader (TopologyFrequencyProfiler::compute_from_node_counts_).
constexpr uint8_t kNodeCountsMagic[8] = {'N','O','D','E','C','N','T','0'};

}  // namespace

// ---------------------------------------------------------------------------
//  Neighbors helpers
// ---------------------------------------------------------------------------

bool FourLevelTopologyStore::Neighbors::empty() const noexcept {
    return size() == 0;
}

std::size_t FourLevelTopologyStore::Neighbors::size() const noexcept {
    switch (tier) {
        case 1: return l1.size;
        case 2: return l2_size;
        case 3: return l3_size;
        case 4: return l4_owned.size();
        default: return 0;
    }
}

// ---------------------------------------------------------------------------
//  Phase 2 dispatcher constructor
// ---------------------------------------------------------------------------

FourLevelTopologyStore::FourLevelTopologyStore(
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
    Config                                          config)
    : l1_fwd_(&l1_fwd),
      l1_rev_(&l1_rev),
      l2_fwd_(&l2_fwd),
      l2_rev_(&l2_rev),
      l3_fwd_(l3_fwd),
      l3_rev_(l3_rev),
      l4_fwd_(std::move(l4_fwd)),
      l4_rev_(std::move(l4_rev)),
      tier_lookup_ref_(&tier_lookup),
      row_lookup_(std::move(row_lookup)),
      phase3_ctor_(false),
      built_(true),
      config_(config)
{}

// ---------------------------------------------------------------------------
//  Phase 3 build constructor
// ---------------------------------------------------------------------------

FourLevelTopologyStore::FourLevelTopologyStore(
    BPlusTree<3>*               fwd_bpt,
    BPlusTree<3>*               rev_bpt,
    GQL::ProjectionStorage*     storage,
    std::filesystem::path       projection_dir,
    Config                      config)
    : fwd_bpt_(fwd_bpt),
      rev_bpt_(rev_bpt),
      storage_(storage),
      projection_dir_(std::move(projection_dir)),
      phase3_ctor_(true),
      built_(false),
      config_(config)
{}

FourLevelTopologyStore::~FourLevelTopologyStore() = default;

bool FourLevelTopologyStore::is_built() const noexcept {
    return built_;
}

void FourLevelTopologyStore::enable_pinned_gpu_view(
    const SamplingBackendPlan& plan)
{
    // Always drop any prior view first (idempotent; unregisters before re-pin).
    pinned_view_.reset();

    if (plan.backend == SamplingBackend::CPU_OUT_OF_CORE) {
        return;  // CPU path: nothing to pin.
    }

    // Symmetric single-slice path: serve ONE pre-merged undirected CSR as fwd,
    // rev null. plan.directions is FORWARD_ONLY here (the engine forces it); BOTH
    // would double-count an already-merged list and is never used.
    if (plan.use_symmetric) {
        const HostCsrArrays* sym = materialize_symmetric_arrays();
        if (sym == nullptr || sym->row_ptr == nullptr) {
            return;  // nothing pinnable -> stay on CPU path
        }
        auto view = std::make_unique<PinnedTopologyView>();
        view->build_and_register(*sym, HostCsrArrays{});
        if (view->is_registered()) {
            pinned_view_ = std::move(view);
        }
        return;
    }

    // Substrate A: the global L3 sidecar (full ROW_PTR + narrow uint32 COL_IDX)
    // covers every node by dense row, so it is the correctness-complete pinnable
    // CSR. Only the narrow (id_width==4) layout is pinnable as uint32.
    auto make_arrays =
        [](const GQL::Projection::TopologySnapshotReader* r) -> HostCsrArrays {
        HostCsrArrays a;
        if (r != nullptr && r->has_data() && r->id_width() == 4) {
            a.row_ptr      = r->row_ptr_base();
            a.col_idx      = r->col_idx32_base();
            a.n_rows       = r->num_nodes();
            a.n_edges      = r->num_edges();
            a.dst_type_tag = static_cast<uint64_t>(r->dst_type_tag()) << 56;
        }
        return a;  // null arrays => the view skips this direction
    };

    const bool want_fwd = plan.directions == GpuDirections::FORWARD_ONLY ||
                          plan.directions == GpuDirections::BOTH;
    const bool want_rev = plan.directions == GpuDirections::REVERSE_ONLY ||
                          plan.directions == GpuDirections::BOTH;

    HostCsrArrays fwd = want_fwd ? make_arrays(l3_fwd_) : HostCsrArrays{};
    HostCsrArrays rev = want_rev ? make_arrays(l3_rev_) : HostCsrArrays{};

    // No pinnable global substrate present (no narrow L3 sidecar for the wanted
    // directions) => stay on the CPU path.
    if (fwd.row_ptr == nullptr && rev.row_ptr == nullptr) {
        return;
    }

    auto view = std::make_unique<PinnedTopologyView>();
    view->build_and_register(fwd, rev);
    // Keep the view only if registration actually happened (a GPU was present).
    // Without a GPU at runtime build_and_register is a no-op and the CPU path is
    // left entirely unaffected.
    if (view->is_registered()) {
        pinned_view_ = std::move(view);
    }
}

// ---------------------------------------------------------------------------
// In-RAM symmetric CSR merge (Part D — GPU-UVA single slice).
// ---------------------------------------------------------------------------

void merge_symmetric_csr_concat(
    const std::vector<uint64_t>& f_rp, const std::vector<uint32_t>& f_ci,
    const std::vector<uint64_t>& r_rp, const std::vector<uint32_t>& r_ci,
    std::vector<uint64_t>& o_rp, std::vector<uint32_t>& o_ci) {
    const std::size_t n = f_rp.size() ? f_rp.size() - 1 : 0;
    o_rp.assign(n + 1, 0);
    o_ci.clear();
    o_ci.reserve(f_ci.size() + r_ci.size());
    for (std::size_t u = 0; u < n; ++u) {
        for (uint64_t i = f_rp[u]; i < f_rp[u + 1]; ++i) o_ci.push_back(f_ci[i]);
        for (uint64_t i = r_rp[u]; i < r_rp[u + 1]; ++i) o_ci.push_back(r_ci[i]);
        o_rp[u + 1] = o_ci.size();
    }
}

void merge_symmetric_csr_node_dedup(
    const std::vector<uint64_t>& f_rp, const std::vector<uint32_t>& f_ci,
    const std::vector<uint64_t>& r_rp, const std::vector<uint32_t>& r_ci,
    std::vector<uint64_t>& o_rp, std::vector<uint32_t>& o_ci) {
    const std::size_t n = f_rp.size() ? f_rp.size() - 1 : 0;
    o_rp.assign(n + 1, 0);
    o_ci.clear();
    o_ci.reserve(f_ci.size() + r_ci.size());
    std::unordered_set<uint32_t> seen;  // reused per row; small (O(degree))
    for (std::size_t u = 0; u < n; ++u) {
        seen.clear();
        for (uint64_t i = f_rp[u]; i < f_rp[u + 1]; ++i) {
            uint32_t v = f_ci[i];
            if (seen.insert(v).second) o_ci.push_back(v);   // out first, dedup
        }
        for (uint64_t i = r_rp[u]; i < r_rp[u + 1]; ++i) {
            uint32_t v = r_ci[i];
            if (seen.insert(v).second) o_ci.push_back(v);   // in not already present
        }
        o_rp[u + 1] = o_ci.size();
    }
}

const HostCsrArrays* FourLevelTopologyStore::materialize_symmetric_arrays() {
    if (sym_arrays_built_) return &sym_arrays_;

    auto narrow = [](const GQL::Projection::TopologySnapshotReader* r) {
        return r != nullptr && r->has_data() && r->id_width() == 4;
    };
    const bool have_f = narrow(l3_fwd_);
    const bool have_r = narrow(l3_rev_);
    if (!have_f && !have_r) return nullptr;  // nothing pinnable to merge

    const uint64_t n = have_f ? l3_fwd_->num_nodes() : l3_rev_->num_nodes();
    auto lift = [n](const GQL::Projection::TopologySnapshotReader* r, bool present,
                    std::vector<uint64_t>& rp, std::vector<uint32_t>& ci) {
        rp.assign(static_cast<std::size_t>(n) + 1, 0);
        ci.clear();
        if (!present) return;
        for (uint64_t u = 0; u < n; ++u) {
            const uint64_t d = r->degree(u);
            const uint32_t* row = r->col_idx32_row(u);
            for (uint64_t i = 0; i < d; ++i) ci.push_back(row[i]);
            rp[u + 1] = ci.size();
        }
    };
    std::vector<uint64_t> f_rp, r_rp;
    std::vector<uint32_t> f_ci, r_ci;
    lift(l3_fwd_, have_f, f_rp, f_ci);
    lift(l3_rev_, have_r, r_rp, r_ci);

    // §4 contract: distinct real edge_ids => concat (no dedup, matching the
    // edge_id-keyed merge that removes nothing); edge_id==0 => node-id dedup.
    const bool distinct_eids =
        (have_f && l3_fwd_->has_edge_ids()) || (have_r && l3_rev_->has_edge_ids());
    if (distinct_eids) {
        merge_symmetric_csr_concat(f_rp, f_ci, r_rp, r_ci,
                                   sym_row_ptr_, sym_col_idx_);
    } else {
        merge_symmetric_csr_node_dedup(f_rp, f_ci, r_rp, r_ci,
                                       sym_row_ptr_, sym_col_idx_);
    }

    sym_arrays_.row_ptr      = sym_row_ptr_.data();
    sym_arrays_.col_idx      = sym_col_idx_.data();
    sym_arrays_.n_rows       = n;
    sym_arrays_.n_edges      = sym_col_idx_.size();
    sym_arrays_.dst_type_tag =
        static_cast<uint64_t>((have_f ? l3_fwd_ : l3_rev_)->dst_type_tag()) << 56;
    sym_arrays_built_ = true;
    return &sym_arrays_;
}

std::size_t FourLevelTopologyStore::symmetric_ram_bytes() const noexcept {
    if (!sym_arrays_built_) return 0;
    return sym_row_ptr_.size() * sizeof(uint64_t)
         + sym_col_idx_.size() * sizeof(uint32_t);
}

// ---------------------------------------------------------------------------
//  build()
// ---------------------------------------------------------------------------

void FourLevelTopologyStore::auto_detect_budgets_(
    std::size_t& l1_bytes,
    std::size_t& l2_bytes) const
{
    // Total cache budget = 70% of MemAvailable. Of that, 25% goes to
    // L1 and 75% goes to L2 — the default RAM split for the two in-memory
    // tiers.
    const uint64_t mem_available = get_mem_available();
    const uint64_t total_cache = (mem_available * 7ULL) / 10ULL;
    if (config_.l1_budget_mb > 0) {
        l1_bytes = config_.l1_budget_mb * 1024ULL * 1024ULL;
    } else {
        l1_bytes = static_cast<std::size_t>((total_cache * 25ULL) / 100ULL);
    }
    if (config_.l2_budget_mb > 0) {
        l2_bytes = config_.l2_budget_mb * 1024ULL * 1024ULL;
    } else {
        l2_bytes = static_cast<std::size_t>((total_cache * 75ULL) / 100ULL);
    }
}

void FourLevelTopologyStore::populate_direction_(
    BPlusTree<3>*                                          index,
    const std::vector<uint8_t>&                            tiers,
    const std::vector<uint64_t>&                           frequency,
    const GQL::Projection::TopologySnapshotReader*         sidecar,
    std::unique_ptr<L1HashCache>&                          l1_out,
    std::unique_ptr<L2CompactCsr>&                         l2_out) const
{
    // Prefer the mmap-backed CSR sidecar fast path (topology_{fwd,rev}.csr)
    // when one was opened for this direction. The sidecar exposes O(1)
    // per-node neighbor slices via mmap, so populating L1/L2 from it is
    // ~10-20× faster than walking the BPT directory + leaf chain. Streaming
    // determinism is preserved because both paths traverse row_idx ascending
    // and call `L2CompactCsr::add_node` in the same order — `node_to_l2_idx_`
    // is bit-identical between the two paths.
    if (sidecar != nullptr && sidecar->has_data()) {
        populate_direction_via_sidecar_(*sidecar, tiers, frequency,
                                        l1_out, l2_out);
        return;
    }

    l1_out = std::make_unique<L1HashCache>(tiers);
    l2_out = std::make_unique<L2CompactCsr>(/*hint=*/0);

    if (index == nullptr) {
        l2_out->freeze();
        return;
    }

    // Streaming distribution into the Four-Level Topology Store tiers
    // (designed to bound peak RSS on large graphs): walk the BPT once in
    // (src, dst, edge_id) lex order, buffer the current src's neighbors
    // in a single staging vector, and flush to L1 / L2 / drop the moment
    // src advances.
    //
    // Why streaming: the previous "materialize vector<vector<AdjEntry>>(N)
    // first" approach allocated an outer header (24 B × N) plus per-node
    // inner vectors for ALL N nodes — including tier-3 / tier-4 nodes that
    // get immediately dropped. On papers100M (110 M nodes × ~30 directed
    // edges/node) the transient peak was ~50 GB, exceeding the 30 GB
    // commodity-RAM target the Four-Level Topology Store was designed to hit.
    //
    // Streaming bounds the transient at:
    //   O(max_node_degree × sizeof(AdjEntry)) = O(deg_max × 16 B).
    // On papers100M with deg_max ~ 10⁵ that is < 2 MB regardless of N.
    //
    // Order requirement: BPT iteration is lexicographic over the (src,
    // dst, edge_id) record, so all records of one src appear consecutively
    // and `staging_buffer` only ever holds one src's neighbors at a time.
    // Verified at `src/storage/index/bplus_tree/bplus_tree.cc::next()`
    // (line 378): the iterator drains the current leaf in stored order,
    // then walks the leaf chain — both paths preserve sort order.
    //
    // Tier-3 / tier-4 nodes are dropped without ever entering a
    // long-lived allocation. The L3 mmap or L4 BPT path serves them at
    // lookup time.

    Record<3> min_record = {0, 0, 0};
    Record<3> max_record = {UINT64_MAX, UINT64_MAX, UINT64_MAX};
    bool interruption = false;
    auto it = index->get_range(&interruption, min_record, max_record);

    constexpr uint64_t kSentinelNoSrc = UINT64_MAX;
    uint64_t                last_src = kSentinelNoSrc;
    std::vector<AdjEntry>   staging_buffer;

    // BPT records carry the full ObjectId (8-bit type tag in the high byte
    // + 56-bit value payload). The tier vector and L1/L2 caches are dense-
    // row-indexed (matching the mmap CSR sidecar's ROW_PTR layout and
    // mirroring the masking convention that
    // populate_direction_via_sidecar_ uses), so we strip the type tag
    // here at the boundary. Without this both:
    //   (a) tiers[last_src] over-runs the tier vector (last_src ~ 0xD4..),
    //       and the previous version dropped the entry as "out-of-range"
    //       even though the projection had a valid tier for the row, and
    //   (b) the L1/L2 keys would be the tagged variant — disagreeing
    //       with the sidecar path so the same BPT-walk projection could
    //       not be queried by row_lookup_'s masked dispatch contract.
    auto flush_current_src = [&]() {
        if (last_src == kSentinelNoSrc) {
            staging_buffer.clear();
            return;
        }
        const uint64_t row_idx = last_src & ObjectId::VALUE_MASK;
        if (row_idx >= tiers.size()) {
            // Out-of-range row (the BPT carries an id past the
            // RowMapping's tier vector). Drop. The dispatcher's
            // out-of-range warning will fire if such an id is queried.
            staging_buffer.clear();
            return;
        }
        const uint8_t tier = tiers[row_idx];
        if (tier == 1) {
            // L1 wants `capacity == size` so total_bytes() is accurate.
            // staging_buffer was sized via reserve(frequency[row_idx])
            // at the start of the run, so capacity already matches the
            // expected degree; a final shrink covers nodes whose
            // observed degree fell below the hint (or whose hint was
            // missing).
            std::vector<AdjEntry> tight(std::move(staging_buffer));
            tight.shrink_to_fit();
            l1_out->insert(/*src_node_id=*/row_idx,
                           std::move(tight),
                           /*row_idx=*/static_cast<std::size_t>(row_idx));
            // After move-from, staging_buffer is in a valid but unspecified
            // state — re-initialise to a known-empty vector so the next
            // src's reserve_for_src call starts from a clean slate.
            staging_buffer = std::vector<AdjEntry>();
        } else if (tier == 2) {
            l2_out->add_node(/*src_node_id=*/row_idx, staging_buffer);
            staging_buffer.clear();
        } else {
            // tier == 3 / 4 -> drop without allocating beyond staging.
            staging_buffer.clear();
        }
    };

    auto reserve_for_src = [&](uint64_t src_id) {
        // src_id arrives tagged (raw BPT key); use the masked variant for
        // the frequency lookup since the profiler indexes by dense row.
        const uint64_t row_idx = src_id & ObjectId::VALUE_MASK;
        if (row_idx < frequency.size()) {
            const uint64_t hint = frequency[row_idx];
            if (hint > 0) staging_buffer.reserve(static_cast<std::size_t>(hint));
        }
    };

    const Record<3>* rec;
    while ((rec = it.next()) != nullptr) {
        const uint64_t a = std::get<0>(*rec);
        const uint64_t b = std::get<1>(*rec);
        const uint64_t e = std::get<2>(*rec);

        if (a != last_src) {
            flush_current_src();
            last_src = a;
            reserve_for_src(a);
        }
        staging_buffer.push_back(AdjEntry{ b, e });
    }
    flush_current_src();

    l2_out->freeze();
}

void FourLevelTopologyStore::populate_direction_via_sidecar_(
    const GQL::Projection::TopologySnapshotReader& sidecar,
    const std::vector<uint8_t>&                    tiers,
    const std::vector<uint64_t>&                   frequency,
    std::unique_ptr<L1HashCache>&                  l1_out,
    std::unique_ptr<L2CompactCsr>&                 l2_out) const
{
    l1_out = std::make_unique<L1HashCache>(tiers);
    l2_out = std::make_unique<L2CompactCsr>(/*hint=*/0);

    // The sidecar is opened with MADV_RANDOM (disables readahead) for the
    // runtime sampler, but this populate scan reads rows in ascending order
    // (row_idx 0..N-1 → ascending file offset). Without sequential advice
    // the scan faults pages one-at-a-time at QD1 (measured: ~98% of the
    // 27.6 GB uint32 topology sidecar faulted @ 55 MB/s, dominating build()
    // at 489s/62%). Issuing MADV_SEQUENTIAL for the duration of the scan
    // gives the kernel aggressive readahead AND frees pages behind the
    // cursor (lower peak cache pressure), then restores MADV_RANDOM for the
    // seed-driven runtime sampler. Pure perf hint (cannot change the bytes
    // read), RAII-restored on any exit. Opt-out:
    // MDB_GNN_NO_POPULATE_SEQUENTIAL=1 for A/B.
    struct SeqAdviseGuard {
        const GQL::Projection::TopologySnapshotReader* r;
        bool active = false;
        explicit SeqAdviseGuard(const GQL::Projection::TopologySnapshotReader& rr)
            : r(&rr) {
            const char* off = std::getenv("MDB_GNN_NO_POPULATE_SEQUENTIAL");
            if (!(off && (off[0] == '1' || off[0] == 't' || off[0] == 'T'))) {
                r->advise_access(/*sequential=*/true);
                active = true;
            }
        }
        ~SeqAdviseGuard() { if (active) r->advise_access(/*sequential=*/false); }
    } seq_guard_(sidecar);

    const uint64_t num_nodes = sidecar.num_nodes();
    const bool     has_eids  = sidecar.has_edge_ids();

    // MADV_WILLNEED ahead-prefetch (default ON; opt-out MDB_GNN_NO_POPULATE_PREFETCH=1).
    // MADV_SEQUENTIAL above gives readahead but is capped by the kernel's default
    // window; explicitly prefetching a large COL_IDX window AHEAD of the cursor
    // keeps more of the 27.6 GB sidecar in flight, raising the NVMe queue depth.
    // Pure perf hint; correctness-neutral.
    const bool do_prefetch = []() {
        const char* off = std::getenv("MDB_GNN_NO_POPULATE_PREFETCH");
        return !(off && (off[0] == '1' || off[0] == 't' || off[0] == 'T'));
    }();
    constexpr uint64_t kPrefetchStride = 1ull << 20;  // ~1M rows / window
    uint64_t next_prefetch_row = 0;
    if (do_prefetch) {
        sidecar.prefetch_rows(0, std::min<uint64_t>(2 * kPrefetchStride, num_nodes));
        next_prefetch_row = 2 * kPrefetchStride;
    }

    // Walk row_idx in [0, num_nodes) ascending — same order the BPT path
    // observes via lex-sorted (src, dst, edge_id) iteration. `tier_lookup_`
    // is also indexed by row_idx (== ObjectId.id under the current
    // identity-RowMapping assumption documented in build()). Tier-3 / 4
    // nodes are skipped: their primary home is the L3 sidecar itself
    // (or the L4 BPT direct path); promoting them to L1/L2 would defeat
    // the tier-budget contract.
    std::vector<AdjEntry> staging;
    // Reused per-node scratch for the width-agnostic copy accessors. For
    // id_width==8 these receive memcpy'd full ObjectIds; for id_width==4 the
    // reader widens the uint32 ordinals + re-applies the type tag, so the
    // values stored into AdjEntry are byte-identical across widths.
    std::vector<uint64_t> dst_scratch;
    std::vector<uint64_t> eid_scratch;
    for (uint64_t row_idx = 0; row_idx < num_nodes; ++row_idx) {
        if (do_prefetch && row_idx >= next_prefetch_row && next_prefetch_row < num_nodes) {
            sidecar.prefetch_rows(next_prefetch_row,
                                  std::min<uint64_t>(next_prefetch_row + kPrefetchStride, num_nodes));
            next_prefetch_row += kPrefetchStride;
        }
        if (row_idx >= tiers.size()) break;  // tier vector exhausted
        const uint8_t tier = tiers[row_idx];
        if (tier != 1 && tier != 2) continue;  // skip L3 / L4

        dst_scratch.clear();
        sidecar.copy_neighbors(row_idx, dst_scratch);
        eid_scratch.clear();
        if (has_eids) {
            sidecar.copy_edge_ids(row_idx, eid_scratch);
        }
        const bool eids_ok = (eid_scratch.size() == dst_scratch.size());

        // Reuse the staging buffer across nodes so per-iteration alloc
        // overhead doesn't dominate the 1-5 us/node target. capacity is
        // retained, only `size` is reset.
        staging.clear();
        if (row_idx < frequency.size() && frequency[row_idx] > 0) {
            staging.reserve(static_cast<std::size_t>(frequency[row_idx]));
        } else {
            staging.reserve(dst_scratch.size());
        }
        for (std::size_t i = 0; i < dst_scratch.size(); ++i) {
            const uint64_t eid = eids_ok ? eid_scratch[i] : 0ULL;
            staging.push_back(AdjEntry{ dst_scratch[i], eid });
        }

        if (tier == 1) {
            std::vector<AdjEntry> tight(std::move(staging));
            tight.shrink_to_fit();
            l1_out->insert(/*src_node_id=*/row_idx,
                           std::move(tight),
                           /*row_idx=*/static_cast<std::size_t>(row_idx));
            // staging was moved-from; restore to a known-empty vector so
            // the next iteration's reserve starts from a clean slate.
            staging = std::vector<AdjEntry>();
        } else {
            // tier == 2
            l2_out->add_node(/*src_node_id=*/row_idx, staging);
        }
    }

    l2_out->freeze();
}

namespace detail {
void symmetric_merge_row(const std::vector<uint64_t>& dst_fwd,
                         const std::vector<uint64_t>& eid_fwd,
                         const std::vector<uint64_t>& dst_rev,
                         const std::vector<uint64_t>& eid_rev,
                         bool has_edge_ids,
                         std::vector<AdjEntry>& out) {
    out.clear();
    out.reserve(dst_fwd.size() + dst_rev.size());
    const bool ef_ok = (eid_fwd.size() == dst_fwd.size());
    const bool er_ok = (eid_rev.size() == dst_rev.size());
    std::unordered_set<uint64_t> seen;
    seen.reserve(dst_fwd.size() + dst_rev.size());
    for (std::size_t i = 0; i < dst_fwd.size(); ++i) {
        const uint64_t eid = ef_ok ? eid_fwd[i] : 0ULL;
        const uint64_t key = has_edge_ids ? eid : dst_fwd[i];
        if (seen.insert(key).second) out.push_back(AdjEntry{ dst_fwd[i], eid });
    }
    for (std::size_t i = 0; i < dst_rev.size(); ++i) {
        const uint64_t eid = er_ok ? eid_rev[i] : 0ULL;
        const uint64_t key = has_edge_ids ? eid : dst_rev[i];
        if (seen.insert(key).second) out.push_back(AdjEntry{ dst_rev[i], eid });
    }
}
}  // namespace detail

void FourLevelTopologyStore::populate_direction_symmetric_(
    const GQL::Projection::TopologySnapshotReader& l3_fwd,
    const GQL::Projection::TopologySnapshotReader& l3_rev,
    const std::vector<uint8_t>&                    tiers,
    const std::vector<uint64_t>&                   /*frequency*/,
    std::unique_ptr<L1HashCache>&                  l1_out,
    std::unique_ptr<L2CompactCsr>&                 l2_out) const
{
    l1_out = std::make_unique<L1HashCache>(tiers);
    l2_out = std::make_unique<L2CompactCsr>(/*hint=*/0);

    // Sequential advice over BOTH sidecars for the duration of the ascending
    // row scan (same rationale as populate_direction_via_sidecar_), restored on
    // any exit.
    struct SeqAdviseGuard {
        const GQL::Projection::TopologySnapshotReader* a;
        const GQL::Projection::TopologySnapshotReader* b;
        bool active = false;
        SeqAdviseGuard(const GQL::Projection::TopologySnapshotReader& aa,
                       const GQL::Projection::TopologySnapshotReader& bb)
            : a(&aa), b(&bb) {
            const char* off = std::getenv("MDB_GNN_NO_POPULATE_SEQUENTIAL");
            if (!(off && (off[0] == '1' || off[0] == 't' || off[0] == 'T'))) {
                a->advise_access(/*sequential=*/true);
                b->advise_access(/*sequential=*/true);
                active = true;
            }
        }
        ~SeqAdviseGuard() {
            if (active) {
                a->advise_access(/*sequential=*/false);
                b->advise_access(/*sequential=*/false);
            }
        }
    } seq_guard_(l3_fwd, l3_rev);

    const uint64_t num_nodes = std::max(l3_fwd.num_nodes(), l3_rev.num_nodes());
    // Matches the accessor's per-node rule: has_edge_ids when EITHER direction
    // carries real edge_ids (the directional sidecars are built together, so in
    // practice both have or both lack them).
    const bool has_eids = l3_fwd.has_edge_ids() || l3_rev.has_edge_ids();

    std::vector<uint64_t> df, ef, dr, er;
    std::vector<AdjEntry> merged;
    for (uint64_t row = 0; row < num_nodes; ++row) {
        if (row >= tiers.size()) break;
        const uint8_t tier = tiers[row];
        if (tier != 1 && tier != 2) continue;  // L3/L4 handled at runtime

        df.clear(); ef.clear(); dr.clear(); er.clear();
        if (row < l3_fwd.num_nodes()) {
            l3_fwd.copy_neighbors(row, df);
            if (l3_fwd.has_edge_ids()) l3_fwd.copy_edge_ids(row, ef);
        }
        if (row < l3_rev.num_nodes()) {
            l3_rev.copy_neighbors(row, dr);
            if (l3_rev.has_edge_ids()) l3_rev.copy_edge_ids(row, er);
        }
        // Edge_id-drop gate: when dropping, the dedup key switches to node-id
        // (parallel/mutual edges collapse) and the emitted edge_ids are zeroed.
        // The build() guard refuses this on a parallel-edge multigraph, so by
        // the time we get here either the graph has no parallel edges or the
        // caller accepts the collapse.
        const bool effective_has_eids = has_eids && !config_.drop_edge_ids;
        detail::symmetric_merge_row(df, ef, dr, er, effective_has_eids, merged);
        if (config_.drop_edge_ids) {
            for (auto& e : merged) e.edge_id = 0;
        }

        if (tier == 1) {
            std::vector<AdjEntry> tight(merged);
            tight.shrink_to_fit();
            l1_out->insert(/*src_node_id=*/row, std::move(tight),
                           /*row_idx=*/static_cast<std::size_t>(row));
        } else {
            l2_out->add_node(/*src_node_id=*/row, merged);
        }
    }

    l2_out->freeze();
}

void FourLevelTopologyStore::open_l3_sidecars_() {
    if (!config_.use_l3_mmap_sidecar) return;
    if (projection_dir_.empty())     return;

    if (config_.orientation == EdgeOrientation::NATURAL ||
        config_.orientation == EdgeOrientation::UNDIRECTED)
    {
        owned_l3_fwd_ = std::make_unique<
            GQL::Projection::TopologySnapshotReader>(
                GQL::Projection::TopologySnapshotReader::open(
                    projection_dir_,
                    GQL::Projection::TopologySnapshotReader::Direction::FORWARD));
        l3_fwd_ = owned_l3_fwd_.get();
    }
    if (config_.orientation == EdgeOrientation::REVERSE ||
        config_.orientation == EdgeOrientation::UNDIRECTED)
    {
        owned_l3_rev_ = std::make_unique<
            GQL::Projection::TopologySnapshotReader>(
                GQL::Projection::TopologySnapshotReader::open(
                    projection_dir_,
                    GQL::Projection::TopologySnapshotReader::Direction::REVERSE));
        l3_rev_ = owned_l3_rev_.get();
    }
    // Optional on-disk pre-merged undirected sidecar (topology_sym.csr). When
    // present it is the fast path for the symmetric tier; when absent the
    // symmetric tier is built by merging l3_fwd_ + l3_rev_ at build() time.
    if (config_.orientation == EdgeOrientation::UNDIRECTED) {
        owned_l3_sym_ = std::make_unique<
            GQL::Projection::TopologySnapshotReader>(
                GQL::Projection::TopologySnapshotReader::open_symmetric(
                    projection_dir_));
        l3_sym_ = (owned_l3_sym_ && owned_l3_sym_->has_data())
                      ? owned_l3_sym_.get() : nullptr;
        if (l3_sym_ == nullptr) owned_l3_sym_.reset();
    }
}

void FourLevelTopologyStore::compute_l3_minhash_reorder_(bool warm_start_used) {
    // Compute a frequency-aware MinHash reorder permutation for the L3
    // tier, using the SEGMENTED variant of the MinHashReorderer
    // (matching DiskGNN Algorithm 1).
    //
    // The reorder ideally uses per-batch access sets (the `BatchProvider`
    // callback drives MinHash through every batch's accessed-node set).
    // This implementation uses **Option B**: frequency-band clustering
    // from the per-node access counts already persisted in
    // `<projection_dir>/node_counts.bin`. Nodes are bucketed into N
    // synthetic frequency bins (log-spaced) and each bin is fed to
    // MinHash as a "batch". The resulting permutation clusters L3-tier
    // nodes by access-frequency similarity instead of true co-access
    // (DiskGNN's Option A), which captures ~50% of the spatial-locality
    // value with zero new on-disk artifact. Option A
    // (`batch_access_sets.bin`) can be wired in later as a pure
    // `BatchProvider` swap.
    //
    // SCOPE LIMIT: the permutation is STORED in
    // `l3_reorder_permutation_` but NOT applied to the on-disk
    // `topology_*.csr` sidecar — that rewrite requires a full sidecar
    // rebuild and is deferred to a future pass that rewrites the sidecar
    // in-place using the computed permutation.
    l3_reorder_permutation_.clear();

    if (!warm_start_used) {
        std::cerr << "[FourLevelTopologyStore] Cold start (no "
                  << "node_counts.bin) — skipping L3 MinHash reorder. "
                  << "Run gnn_offline_sample once with "
                  << "useFourLevelTopologyStore:true to enable warm "
                  << "start on the next build.\n";
        return;
    }

    if (tier_lookup_ref_ == nullptr || tier_lookup_ref_->empty()) {
        return;
    }

    // Re-acquire the per-node frequency vector. We hold no profiler
    // reference (build() instantiates a stack-local profiler), so the
    // simplest path is to re-run the warm-start reader; it is O(N)
    // bytes and runs once per build. When the file is absent or
    // malformed we degrade gracefully to an empty permutation (the
    // `warm_start_used` flag guarantees the file existed at the moment
    // the profiler ran, so an absent file here would only happen under
    // a concurrent unlink — vanishingly rare and harmless).
    std::vector<uint64_t> frequency;
    {
        if (!projection_dir_.empty()) {
            std::filesystem::path path = projection_dir_ / "node_counts.bin";
            std::ifstream f(path, std::ios::binary);
            if (f) {
                uint8_t magic[8] = {};
                uint64_t num_nodes = 0;
                uint64_t direction_bitmask = 0;
                if (f.read(reinterpret_cast<char*>(magic), 8) &&
                    std::memcmp(magic, kNodeCountsMagic, 8) == 0 &&
                    f.read(reinterpret_cast<char*>(&num_nodes),         sizeof(num_nodes)) &&
                    f.read(reinterpret_cast<char*>(&direction_bitmask), sizeof(direction_bitmask)) &&
                    num_nodes == tier_lookup_ref_->size())
                {
                    frequency.assign(static_cast<std::size_t>(num_nodes), 0);
                    if (num_nodes > 0) {
                        if (!f.read(reinterpret_cast<char*>(frequency.data()),
                                    static_cast<std::streamsize>(num_nodes * sizeof(uint64_t))))
                        {
                            frequency.clear();
                        }
                    }
                }
            }
        }
    }
    if (frequency.empty()) {
        return;
    }

    // Collect L3-tier node IDs (row indices, identity RowMapping).
    std::vector<uint64_t> l3_node_ids;
    l3_node_ids.reserve(tier_lookup_ref_->size() / 4);
    for (std::size_t i = 0; i < tier_lookup_ref_->size(); ++i) {
        if ((*tier_lookup_ref_)[i] == 3) {
            l3_node_ids.push_back(static_cast<uint64_t>(i));
        }
    }
    if (l3_node_ids.empty()) {
        // Warm-start file was consumed but the build budget was generous
        // enough to land every node in L1/L2. The reorder is a no-op,
        // but the activation itself is the user-visible signal we want
        // to surface so bench harnesses + diagnostics see it.
        std::cerr << "[FourLevelTopologyStore] Warm start activated — "
                  << "node_counts.bin consumed, but every node fits "
                  << "in L1/L2 with current budgets. L3 set is empty; "
                  << "MinHash reorder skipped (no-op).\n";
        return;
    }

    // Build N log-spaced frequency bins over the L3 subset. Each bin
    // is presented to MinHash as one synthetic "batch" — nodes that
    // share a bin therefore share a MinHash signature with high
    // probability and cluster together in the resulting permutation.
    //
    // Bin count chosen at 16 — empirically matches DiskGNN's default
    // segment_size:100 batches scaled down to the synthetic case
    // where every "batch" is much larger than a real one.
    constexpr uint64_t kNumBins = 16;
    uint64_t freq_max = 0;
    for (uint64_t row : l3_node_ids) {
        if (row < frequency.size() && frequency[row] > freq_max) {
            freq_max = frequency[row];
        }
    }

    std::vector<std::vector<uint64_t>> bins(kNumBins);
    for (uint64_t row : l3_node_ids) {
        const uint64_t f = (row < frequency.size()) ? frequency[row] : 0;
        // Log-spaced bin index: bin = floor(log2(f+1) / log2(freq_max+2) * N).
        uint64_t bin_idx = 0;
        if (freq_max > 0) {
            const double t = std::log2(static_cast<double>(f) + 1.0)
                           / std::log2(static_cast<double>(freq_max) + 2.0);
            const double scaled = t * static_cast<double>(kNumBins);
            int64_t b = static_cast<int64_t>(scaled);
            if (b < 0) b = 0;
            if (b >= static_cast<int64_t>(kNumBins)) b = kNumBins - 1;
            bin_idx = static_cast<uint64_t>(b);
        }
        bins[bin_idx].push_back(row);
    }

    MinHashReorderer::Config minhash_cfg;
    minhash_cfg.strategy   = MinHashReorderer::Strategy::SEGMENTED;
    MinHashReorderer reorderer(minhash_cfg);

    auto provider = [&bins](uint64_t batch_id) -> std::vector<uint64_t> {
        if (batch_id >= bins.size()) return {};
        return bins[batch_id];
    };
    reorderer.build_access_graph(/*num_batches=*/kNumBins, provider);

    // total_rows must equal the underlying topology's row count, not
    // just the L3 subset, so the reorderer's "unaccessed nodes
    // appended" logic produces a complete permutation indexed by
    // row_idx ∈ [0, num_nodes).
    l3_reorder_permutation_ =
        reorderer.compute_permutation(tier_lookup_ref_->size());

    std::cerr << "[FourLevelTopologyStore] Warm start activated — "
              << "computed MinHash permutation over " << l3_node_ids.size()
              << " L3-tier nodes via " << kNumBins
              << " frequency bins (frequency-band clustering; stored only "
              << "— applying it to the on-disk sidecar is deferred to a "
              << "future pass).\n";
}

void FourLevelTopologyStore::build() {
    if (!phase3_ctor_) {
        throw std::logic_error(
            "FourLevelTopologyStore::build() called on a dispatcher-mode "
            "instance — only the BPlusTree-based ctor supports build()");
    }
    if (built_) {
        throw std::logic_error(
            "FourLevelTopologyStore::build() called twice — recreate the "
            "store to rebuild");
    }

    // Build-phase per-span instrumentation: split the otherwise-opaque
    // build() wall-clock into open/SHA, profile, MinHash, and populate
    // spans, plus the physical bytes faulted during populate
    // (/proc/self/io read_bytes). Emitted once to stderr next to the
    // "built — ..." summary so the server log carries the breakdown
    // without any cross-layer yield plumbing. Pure observation — never
    // branches the build, cannot perturb sampling output.
    using clk_ = std::chrono::steady_clock;
    double sha_ms = 0.0, profile_ms = 0.0, minhash_ms = 0.0, populate_ms = 0.0;
    uint64_t io_read_before = 0, io_read_after = 0;

    // Step 1: compute budgets.
    std::size_t l1_bytes = 0;
    std::size_t l2_bytes = 0;
    auto_detect_budgets_(l1_bytes, l2_bytes);

    // Step 1.5: open the mmap-backed CSR sidecar files (topology_fwd.csr /
    // topology_rev.csr) early so that populate_direction_ can use the
    // O(1)-slice fast path instead of walking the BPT directory + leaf
    // chain. The opens are no-ops when the sidecar files are absent
    // (e.g., projection built without buildTopologySnapshot:true) or
    // stale — in which case populate_direction_ falls through to the
    // BPT path automatically. open() also runs the source-.leaf SHA-256
    // staleness gate, so this span captures the SHA cost.
    {
        auto t0 = clk_::now();
        open_l3_sidecars_();
        sha_ms = elapsed_ms_(t0, clk_::now());
    }

    // Step 2: build a TopologyAccessor over the storage so the profiler
    // can query degrees through a stable API. When `storage_` is null
    // (test contexts that drive build() with raw BPTs) we fall back to
    // a degree pass directly over the BPT iterators here, but the
    // simpler path for production is via TopologyAccessor.
    //
    // We need a non-owning TopologyAccessor view with the same lifetime
    // as the build phase. The helper is constructed inline rather than
    // owned by *this so we don't pay its memory cost forever.
    if (storage_ == nullptr) {
        // Synthetic-test path: derive degrees directly from BPT scans.
        // This is rarely used in production but keeps the store testable
        // without spinning up a full ProjectionStorage fixture.
        std::unordered_map<uint64_t, std::vector<AdjEntry>> fwd_map, rev_map;
        scan_index_into_(fwd_bpt_, fwd_map);
        scan_index_into_(rev_bpt_, rev_map);

        // Derive node count from the maximum src/dst seen in either map
        // plus 1. This is conservative: it only undercounts when there
        // are isolated nodes with id > all observed endpoints, which the
        // synthetic-test path is responsible for avoiding.
        uint64_t max_id = 0;
        for (const auto& kv : fwd_map) {
            max_id = std::max(max_id, kv.first);
            for (const auto& e : kv.second) max_id = std::max(max_id, e.node_id);
        }
        for (const auto& kv : rev_map) {
            max_id = std::max(max_id, kv.first);
            for (const auto& e : kv.second) max_id = std::max(max_id, e.node_id);
        }
        const std::size_t n = static_cast<std::size_t>(max_id) + 1;

        // Build a frequency vector (degree proxy).
        std::vector<uint64_t> frequency(n, 0);
        for (const auto& kv : fwd_map) {
            if (kv.first < n) frequency[kv.first] += kv.second.size();
        }
        for (const auto& kv : rev_map) {
            if (kv.first < n) frequency[kv.first] += kv.second.size();
        }
        const double avg_degree =
            (n > 0)
                ? static_cast<double>(std::accumulate(frequency.begin(),
                                                      frequency.end(),
                                                      uint64_t{0}))
                      / static_cast<double>(n)
                : 0.0;

        owned_tier_assignment_ = compute_tier_assignment(
            frequency, l1_bytes, l2_bytes, avg_degree);
        tier_lookup_ref_ = &owned_tier_assignment_;

        // Synthetic-test path always cold-starts (no profiler ⇒ no
        // node_counts.bin). Skip MinHash reorder as a no-op.
        compute_l3_minhash_reorder_(/*warm_start_used=*/false);

        populate_direction_(fwd_bpt_, owned_tier_assignment_, frequency,
                            l3_fwd_, owned_l1_fwd_, owned_l2_fwd_);
        populate_direction_(rev_bpt_, owned_tier_assignment_, frequency,
                            l3_rev_, owned_l1_rev_, owned_l2_rev_);
    } else {
        // Production path: use TopologyAccessor + TopologyFrequencyProfiler.
        TopologyAccessor accessor(*storage_);
        TopologyFrequencyProfiler profiler(accessor, projection_dir_);
        {
            auto t0 = clk_::now();
            profiler.compute(config_.orientation);
            profile_ms = elapsed_ms_(t0, clk_::now());
        }

        const auto& freq = profiler.frequency();
        const std::size_t n = freq.size();

        // The `avg_degree` passed to `compute_tier_assignment` is used
        // ONLY to estimate bytes-per-node when packing the L1/L2
        // budgets. It must reflect actual graph degree, NOT access
        // frequency: when the `frequency` vector is sparse (e.g. after
        // a cold-start random-walk profile pass where many entries are
        // 0 or 1), `total_freq/N` collapses to near zero and the
        // heuristic mistakenly believes every node costs only its fixed
        // overhead — packing the entire graph into L1/L2 and leaving
        // L3 empty. Real per-node cost is then 10-100× higher, causing
        // populate_direction_ to overrun the intended RAM envelope and
        // dominate the build phase.
        //
        // Prefer the mmap CSR sidecar's `num_edges` / `num_nodes` ratio
        // when available (constant-time read from the mmap'd header).
        // Fall back to the access-count proxy only when no sidecar is
        // open — in that case the cold-start path is already running
        // with limited info and the heuristic can misestimate (callers
        // can override via explicit budgets).
        double avg_degree = 0.0;
        if (l3_rev_ != nullptr && l3_rev_->has_data() && l3_rev_->num_nodes() > 0) {
            avg_degree = static_cast<double>(l3_rev_->num_edges()) /
                         static_cast<double>(l3_rev_->num_nodes());
        } else if (l3_fwd_ != nullptr && l3_fwd_->has_data() && l3_fwd_->num_nodes() > 0) {
            avg_degree = static_cast<double>(l3_fwd_->num_edges()) /
                         static_cast<double>(l3_fwd_->num_nodes());
        } else if (n > 0) {
            uint64_t total_freq = 0;
            for (uint64_t f : freq) total_freq += f;
            avg_degree = static_cast<double>(total_freq) / static_cast<double>(n);
        }

        owned_tier_assignment_ = compute_tier_assignment(
            freq, l1_bytes, l2_bytes, avg_degree);
        tier_lookup_ref_ = &owned_tier_assignment_;

        // Compute the MinHash reorder permutation for the L3 tier when
        // warm-start data (node_counts.bin) is available. On a cold
        // start, compute_l3_minhash_reorder_ emits a single info log
        // and leaves the permutation empty (no-op until a prior
        // gnn_offline_sample run has written node_counts.bin).
        {
            auto t0 = clk_::now();
            compute_l3_minhash_reorder_(profiler.warm_start_used());
            minhash_ms = elapsed_ms_(t0, clk_::now());
        }

        // Populate span — the dominant, I/O-bound-at-QD1 cost on
        // papers100M. Bracket both directions + sample /proc/self/io
        // read_bytes around them to expose the physical read amplification.
        io_read_before = read_proc_io_read_bytes_();
        auto t_pop = clk_::now();

        // Opt-in parallel populate (MDB_GNN_POPULATE_PARALLEL=1, default OFF):
        // for UNDIRECTED orientation both directions must be populated, and they
        // are INDEPENDENT — disjoint L1/L2 outputs (owned_l1_fwd_/owned_l2_fwd_
        // vs *_rev_), read-only shared tier_assignment + freq, separate L3
        // sidecars. The sidecar populate path is mmap-only (no BPT walk, hence
        // no QueryContext thread_local requirement). Running fwd on a worker
        // thread and rev on this thread overlaps the QD1-I/O-bound populate
        // (the dominant build cost on papers100M). Output is bit-identical to
        // the sequential path (each direction writes only its own caches);
        // validate via sampleContentFp OFF==ON. Default OFF => sequential.
        const bool s1a_parallel_populate =
            (config_.orientation == EdgeOrientation::UNDIRECTED) &&
            []() {
                const char* e = std::getenv("MDB_GNN_POPULATE_PARALLEL");
                return e != nullptr && std::string(e) == "1";
            }();

        if (s1a_parallel_populate) {
            std::exception_ptr fwd_ex;
            std::thread tf([&]() {
                try {
                    populate_direction_(fwd_bpt_, owned_tier_assignment_, freq,
                                        l3_fwd_, owned_l1_fwd_, owned_l2_fwd_);
                } catch (...) {
                    fwd_ex = std::current_exception();
                }
            });
            populate_direction_(rev_bpt_, owned_tier_assignment_, freq,
                                l3_rev_, owned_l1_rev_, owned_l2_rev_);
            tf.join();
            if (fwd_ex) std::rethrow_exception(fwd_ex);
        } else {

        // Build forward direction when needed.
        if (config_.orientation == EdgeOrientation::NATURAL ||
            config_.orientation == EdgeOrientation::UNDIRECTED)
        {
            populate_direction_(fwd_bpt_, owned_tier_assignment_, freq,
                                l3_fwd_, owned_l1_fwd_, owned_l2_fwd_);
        } else {
            // Reverse-only: still allocate empty L1/L2 forward so the
            // dispatcher's pointer accesses don't deref null.
            owned_l1_fwd_ = std::make_unique<L1HashCache>(owned_tier_assignment_);
            owned_l2_fwd_ = std::make_unique<L2CompactCsr>(0);
            owned_l2_fwd_->freeze();
        }

        // Build reverse direction when needed.
        if (config_.orientation == EdgeOrientation::REVERSE ||
            config_.orientation == EdgeOrientation::UNDIRECTED)
        {
            populate_direction_(rev_bpt_, owned_tier_assignment_, freq,
                                l3_rev_, owned_l1_rev_, owned_l2_rev_);
        } else {
            owned_l1_rev_ = std::make_unique<L1HashCache>(owned_tier_assignment_);
            owned_l2_rev_ = std::make_unique<L2CompactCsr>(0);
            owned_l2_rev_->freeze();
        }

        }  // end sequential populate (MDB_GNN_POPULATE_PARALLEL=0)

        populate_ms = elapsed_ms_(t_pop, clk_::now());
        io_read_after = read_proc_io_read_bytes_();

        // Symmetric (pre-merged undirected) tier. Reuses the SAME per-node tier
        // assignment + freq (direction-agnostic). Prefer the on-disk
        // topology_sym.csr; else merge the two directional sidecars row-by-row.
        // Skipped (sym_built_ stays false) when neither source is available, so
        // get_neighbors(UNDIRECTED) keeps the runtime out+in+merge fallback.
        if (config_.orientation == EdgeOrientation::UNDIRECTED) {
            // Edge_id-drop guard: dropping forces node-id dedup, which would
            // collapse parallel edges and change the receptive field. Refuse on
            // a parallel-edge multigraph and leave the sym tier unbuilt so the
            // runtime out+in+merge fallback (which preserves edge_ids) engages.
            bool refuse_drop = false;
            if (config_.drop_edge_ids && fwd_bpt_ != nullptr) {
                refuse_drop = GQL::Projection::detect_parallel_edges(
                    fwd_bpt_, owned_tier_assignment_.size());
            }
            if (refuse_drop) {
                sym_refused_edge_id_drop_ = true;
            } else if (l3_sym_ != nullptr && l3_sym_->has_data()) {
                populate_direction_via_sidecar_(*l3_sym_, owned_tier_assignment_,
                                                freq, owned_l1_sym_, owned_l2_sym_);
                sym_built_ = true;
            } else if (l3_fwd_ != nullptr && l3_fwd_->has_data()
                       && l3_rev_ != nullptr && l3_rev_->has_data()) {
                populate_direction_symmetric_(*l3_fwd_, *l3_rev_,
                                              owned_tier_assignment_, freq,
                                              owned_l1_sym_, owned_l2_sym_);
                sym_built_ = true;
            }
            if (sym_built_) {
                l1_sym_ = owned_l1_sym_.get();
                l2_sym_ = owned_l2_sym_.get();
            }
        }
    }

    // Wire the active references to the owned tier sources.
    l1_fwd_ = owned_l1_fwd_.get();
    l1_rev_ = owned_l1_rev_.get();
    l2_fwd_ = owned_l2_fwd_.get();
    l2_rev_ = owned_l2_rev_.get();

    // l3_fwd_/l3_rev_ are already wired by Step 1.5's open_l3_sidecars_().

    // Wire L4 fallback closures to the live BPTs. Required so L3-tier
    // nodes whose sidecar is absent / out-of-range fall through to a
    // working BPT direct path (per design §2.4).
    if (fwd_bpt_ != nullptr) {
        BPlusTree<3>* idx = fwd_bpt_;
        l4_fwd_ = [idx](ObjectId v) -> std::vector<AdjEntry> {
            std::vector<AdjEntry> out;
            Record<3> mn = {v.id, 0, 0};
            Record<3> mx = {v.id, UINT64_MAX, UINT64_MAX};
            bool interruption = false;
            auto it = idx->get_range(&interruption, mn, mx);
            const Record<3>* rec;
            while ((rec = it.next()) != nullptr) {
                if (std::get<0>(*rec) != v.id) continue;
                out.push_back(AdjEntry{ std::get<1>(*rec), std::get<2>(*rec) });
            }
            return out;
        };
    }
    if (rev_bpt_ != nullptr) {
        BPlusTree<3>* idx = rev_bpt_;
        l4_rev_ = [idx](ObjectId v) -> std::vector<AdjEntry> {
            std::vector<AdjEntry> out;
            Record<3> mn = {v.id, 0, 0};
            Record<3> mx = {v.id, UINT64_MAX, UINT64_MAX};
            bool interruption = false;
            auto it = idx->get_range(&interruption, mn, mx);
            const Record<3>* rec;
            while ((rec = it.next()) != nullptr) {
                if (std::get<0>(*rec) != v.id) continue;
                out.push_back(AdjEntry{ std::get<1>(*rec), std::get<2>(*rec) });
            }
            return out;
        };
    }

    // L4 symmetric fallback: merge the two directional L4 closures so a tier-3/4
    // sym node (no L1/L2 entry) still resolves to the canonical UNDIRECTED list.
    if (sym_built_ && fwd_bpt_ != nullptr && rev_bpt_ != nullptr) {
        L4Lookup lf = l4_fwd_;
        L4Lookup lr = l4_rev_;
        l4_sym_ = [lf, lr](ObjectId v) -> std::vector<AdjEntry> {
            std::vector<AdjEntry> a = lf ? lf(v) : std::vector<AdjEntry>{};
            std::vector<AdjEntry> b = lr ? lr(v) : std::vector<AdjEntry>{};
            std::vector<uint64_t> df, ef, dr, er;
            df.reserve(a.size()); ef.reserve(a.size());
            dr.reserve(b.size()); er.reserve(b.size());
            for (auto& e : a) { df.push_back(e.node_id); ef.push_back(e.edge_id); }
            for (auto& e : b) { dr.push_back(e.node_id); er.push_back(e.edge_id); }
            const bool has_eids = (!a.empty() && a.front().edge_id != 0)
                               || (!b.empty() && b.front().edge_id != 0);
            std::vector<AdjEntry> out;
            detail::symmetric_merge_row(df, ef, dr, er, has_eids, out);
            return out;
        };
    }

    // Default row_lookup_ for the build-ctor path: strip the 8-bit ObjectId
    // type tag to recover the dense row index. Production projections
    // (cora_gnn, ogbn-*, papers100M) enumerate row indices 0..N-1 and the
    // row_idx == (ObjectId.id & VALUE_MASK) assumption holds — matching
    // the TopologyFrequencyProfiler's degree pass that ran at row index
    // `i == ObjectId(i).get_value()` and the mmap CSR sidecar's dense
    // ROW_PTR layout (writer at native_projection_builder.cc:2552 uses
    // `(*rec)[0] & ObjectId::VALUE_MASK` for the same reason). The
    // tag-bearing variant (`v.id`) drove every lookup past
    // `tier_lookup_ref_->size()` and silently fell through to the L4
    // BPT path, defeating the entire cache hierarchy and emitting the
    // "ObjectId.id=… exceeds tier_lookup_ size=…" out-of-range warning.
    // When a projection ever moves to a non-identity RowMapping this
    // closure is the single point of update.
    row_lookup_ = [](ObjectId v) -> uint64_t { return v.get_value(); };

    built_ = true;

    std::cerr << "FourLevelTopologyStore: built — "
              << "l1_fwd=" << owned_l1_fwd_->node_count()
              << " l2_fwd=" << owned_l2_fwd_->node_count()
              << " l1_rev=" << owned_l1_rev_->node_count()
              << " l2_rev=" << owned_l2_rev_->node_count()
              << " l3_fwd="
              << (l3_fwd_ && l3_fwd_->has_data() ? l3_fwd_->num_nodes() : 0)
              << " l3_rev="
              << (l3_rev_ && l3_rev_->has_data() ? l3_rev_->num_nodes() : 0)
              << " ram_used=" << total_ram_used()
              << " bytes\n";

    // Emit per-span build timing. populateBytesFaulted is the physical
    // block-device read during populate (/proc/self/io read_bytes delta);
    // compare to the logical tier-1/2 sidecar bytes to measure the
    // MADV_RANDOM read amplification of the populate phase.
    const double pop_mb_per_s =
        (populate_ms > 0.0)
            ? (static_cast<double>(io_read_after - io_read_before) / 1e6)
              / (populate_ms / 1000.0)
            : 0.0;
    std::cerr << "FourLevelTopologyStore: build phase split — "
              << "openShaMs=" << static_cast<long long>(sha_ms)
              << " profileMs=" << static_cast<long long>(profile_ms)
              << " minhashMs=" << static_cast<long long>(minhash_ms)
              << " populateMs=" << static_cast<long long>(populate_ms)
              << " populateBytesFaulted="
              << (io_read_after >= io_read_before ? io_read_after - io_read_before : 0)
              << " populateReadMBps=" << static_cast<long long>(pop_mb_per_s)
              << "\n";
}

// ---------------------------------------------------------------------------
//  Lookup paths
// ---------------------------------------------------------------------------

FourLevelTopologyStore::Neighbors
FourLevelTopologyStore::get_out_neighbors(ObjectId v) const
{
    if (!built_) {
        throw std::logic_error(
            "FourLevelTopologyStore::get_out_neighbors before build()");
    }
    return dispatch_(v, *l1_fwd_, *l2_fwd_, l3_fwd_, l4_fwd_);
}

FourLevelTopologyStore::Neighbors
FourLevelTopologyStore::get_in_neighbors(ObjectId v) const
{
    if (!built_) {
        throw std::logic_error(
            "FourLevelTopologyStore::get_in_neighbors before build()");
    }
    return dispatch_(v, *l1_rev_, *l2_rev_, l3_rev_, l4_rev_);
}

FourLevelTopologyStore::Neighbors
FourLevelTopologyStore::get_neighbors(ObjectId v) const
{
    switch (config_.orientation) {
        case EdgeOrientation::NATURAL:
            return get_out_neighbors(v);
        case EdgeOrientation::REVERSE:
            return get_in_neighbors(v);
        case EdgeOrientation::UNDIRECTED:
            break;
    }

    // Symmetric single-dispatch: when the pre-merged undirected tier is built,
    // one dispatch over it replaces the per-node out+in fetch + merge. It reuses
    // the SAME tier_lookup_ref_ + row_lookup_ (per-node, direction-agnostic).
    if (sym_built_) {
        return dispatch_(v, *l1_sym_, *l2_sym_, l3_sym_, l4_sym_);
    }

    // Fallback (sym tier not built): out + in merged with the SAME rule as the
    // bake and the accessor (detail::symmetric_merge_row) — key = edge_id when
    // edge_ids are real (mutual/parallel edges preserved), else node-id. This
    // makes get_neighbors(UNDIRECTED) byte-identical to
    // TopologyAccessor::get_neighbors_into(UNDIRECTED) regardless of the sym
    // tier, so routing the accessor through this one call never changes the
    // sampled receptive field.
    Neighbors out;
    out.tier = 4;  // materialise into l4_owned for the merge result.

    auto fwd = get_out_neighbors(v);
    auto rev = get_in_neighbors(v);

    std::vector<uint64_t> df, ef, dr, er;
    df.reserve(fwd.size()); ef.reserve(fwd.size());
    dr.reserve(rev.size()); er.reserve(rev.size());
    fwd.for_each_with_edge_id([&](uint64_t d, uint64_t e) {
        df.push_back(d); ef.push_back(e);
    });
    rev.for_each_with_edge_id([&](uint64_t d, uint64_t e) {
        dr.push_back(d); er.push_back(e);
    });
    const bool has_eids = (!ef.empty() && ef.front() != 0)
                       || (!er.empty() && er.front() != 0);
    detail::symmetric_merge_row(df, ef, dr, er, has_eids, out.l4_owned);

    return out;
}

FourLevelTopologyStore::Neighbors
FourLevelTopologyStore::dispatch_(
    ObjectId                                            v,
    const L1HashCache&                                  l1,
    const L2CompactCsr&                                 l2,
    const GQL::Projection::TopologySnapshotReader*     l3,
    const L4Lookup&                                     l4) const
{
    Neighbors out;

    if (tier_lookup_ref_ == nullptr) {
        // Defensive: should never happen — both ctors wire it.
        throw std::logic_error(
            "FourLevelTopologyStore: tier_lookup is unset");
    }

    const uint64_t row_idx = row_lookup_(v);
    const bool row_in_range = (row_idx < tier_lookup_ref_->size());

    if (!row_in_range) {
        // The Four-Level Topology Store currently assumes an identity
        // RowMapping (row_idx == ObjectId.id with type tag stripped). A
        // miss here means a future projection moved to a non-identity
        // RowMapping without updating the row_lookup_ closure (or the
        // caller queried a node id outside the projection altogether).
        // We fall through to L4 (correct, just slow) and emit a single
        // warning per process the first time it fires — repeated
        // warnings would drown sampler logs at scale.
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::cerr << "[FourLevelTopologyStore] WARNING: ObjectId.id="
                      << v.id << " exceeds tier_lookup_ size="
                      << tier_lookup_ref_->size()
                      << " - falling through to L4. The projection's "
                      << "RowMapping may not be identity (the tiered store "
                      << "currently assumes ObjectId.id == row_idx). "
                      << "Subsequent warnings suppressed.\n";
        }
        if (l4) {
            out.l4_owned = l4(v);
            out.tier     = 4;
            return out;
        }
        throw std::out_of_range(
            "FourLevelTopologyStore: row_lookup returned out-of-range "
            "row_idx and no L4 fallback is configured");
    }

    const uint8_t tier =
        (*tier_lookup_ref_)[static_cast<std::size_t>(row_idx)];

    switch (tier) {
        case 1: {
            // L1 is keyed by dense row idx (no type tag) — see
            // populate_direction_via_sidecar_ + populate_direction_ which
            // mask the tag at insert time. Use the same masked id here so
            // the lookup hits the entry that build() materialised.
            out.l1   = l1.get(row_idx);
            out.tier = 1;
            return out;
        }
        case 2: {
            // Same masking story as case 1 — L2 is keyed by dense row idx.
            auto span = l2.get(row_idx);
            out.l2_col_idx = span.first;
            out.l2_size    = span.second;
            // col_idx_ stores tag-stripped uint32 ordinals; carry the
            // per-direction dst type tag so for_each_* reconstructs the exact
            // tagged ObjectId. Without it, L2 neighbours leak as tag-0 ids
            // that miss the tagged feature RowMapping (BatchMaterializer
            // "no corresponding feature row") — silent sample corruption.
            out.l2_dst_tag = l2.dst_type_tag();
            out.tier       = 2;
            return out;
        }
        case 3: {
            if (l3 != nullptr && l3->has_data()
                && row_idx < l3->num_nodes())
            {
                // Zero-copy into the mmap for BOTH id widths. degree() is
                // width-agnostic; the section pointers differ by width.
                out.l3_size = static_cast<std::size_t>(l3->degree(row_idx));
                if (l3->id_width()
                        == GQL::Projection::kTopologySnapshotIdWidthNarrow) {
                    // Narrow (uint32): raw pointers + pre-shifted tags; the
                    // for_each_* helpers widen + re-OR the tag inline.
                    out.l3_col_idx32 = l3->col_idx32_row(row_idx);
                    out.l3_dst_tag =
                        static_cast<uint64_t>(l3->dst_type_tag()) << 56;
                    if (l3->has_edge_ids()) {
                        out.l3_edge_ids32 = l3->edge_ids32_row(row_idx);
                        out.l3_eid_tag =
                            static_cast<uint64_t>(l3->edge_type_tag()) << 56;
                    }
                } else {
                    auto dst_span = l3->neighbors(row_idx);
                    out.l3_col_idx = dst_span.data();
                    if (l3->has_edge_ids()) {
                        auto eid_span = l3->edge_ids(row_idx);
                        if (eid_span.size() == dst_span.size()) {
                            out.l3_edge_ids = eid_span.data();
                        }
                    }
                }
                out.tier = 3;
                return out;
            }
            if (l4) {
                out.l4_owned = l4(v);
                out.tier     = 4;
                return out;
            }
            throw std::runtime_error(
                "FourLevelTopologyStore: tier-3 dispatch reached but no "
                "L3 sidecar and no L4 fallback are configured for this "
                "direction");
        }
        default: {
            if (l4) {
                out.l4_owned = l4(v);
                out.tier     = 4;
                return out;
            }
            throw std::runtime_error(
                "FourLevelTopologyStore: tier-4 dispatch reached but no "
                "L4 BPT fallback is configured for this direction");
        }
    }
}

// ---------------------------------------------------------------------------
//  Diagnostics
// ---------------------------------------------------------------------------

std::size_t FourLevelTopologyStore::l1_node_count() const noexcept {
    std::size_t n = 0;
    if (l1_fwd_ != nullptr) n += l1_fwd_->node_count();
    if (l1_rev_ != nullptr) n += l1_rev_->node_count();
    return n;
}

std::size_t FourLevelTopologyStore::l2_node_count() const noexcept {
    std::size_t n = 0;
    if (l2_fwd_ != nullptr) n += l2_fwd_->node_count();
    if (l2_rev_ != nullptr) n += l2_rev_->node_count();
    return n;
}

std::size_t FourLevelTopologyStore::l3_node_count() const noexcept {
    std::size_t n = 0;
    if (l3_fwd_ != nullptr && l3_fwd_->has_data()) n += l3_fwd_->num_nodes();
    if (l3_rev_ != nullptr && l3_rev_->has_data()) n += l3_rev_->num_nodes();
    return n;
}

std::size_t FourLevelTopologyStore::l1_sym_node_count() const noexcept {
    return l1_sym_ != nullptr ? l1_sym_->node_count() : 0;
}

std::size_t FourLevelTopologyStore::l2_sym_node_count() const noexcept {
    return l2_sym_ != nullptr ? l2_sym_->node_count() : 0;
}

std::size_t FourLevelTopologyStore::l4_node_count() const noexcept {
    if (tier_lookup_ref_ == nullptr) return 0;
    std::size_t n = 0;
    for (uint8_t t : *tier_lookup_ref_) {
        // Tier 3 with no L3 sidecar means the lookup falls into L4 at
        // runtime. We count those as L4 to surface the practical
        // dispatch reality to diagnostics consumers. Tier 4 entries
        // (rare; not produced by compute_tier_assignment but possible
        // via the dispatcher ctor's caller-supplied vector) likewise
        // count as L4.
        if (t == 4) ++n;
        else if (t == 3 && (l3_fwd_ == nullptr || !l3_fwd_->has_data())) ++n;
    }
    return n;
}

std::size_t FourLevelTopologyStore::total_ram_used() const noexcept {
    std::size_t bytes = 0;
    if (l1_fwd_ != nullptr) bytes += l1_fwd_->total_bytes();
    if (l1_rev_ != nullptr) bytes += l1_rev_->total_bytes();
    if (l2_fwd_ != nullptr) bytes += l2_fwd_->total_bytes();
    if (l2_rev_ != nullptr) bytes += l2_rev_->total_bytes();
    if (l1_sym_ != nullptr) bytes += l1_sym_->total_bytes();
    if (l2_sym_ != nullptr) bytes += l2_sym_->total_bytes();
    return bytes;
}

}  // namespace mdb::gnn
