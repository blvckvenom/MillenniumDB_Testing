#include "gnn/projection/four_level_topology_store.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gnn/projection/topology_accessor.h"
#include "gnn/projection/topology_frequency_profiler.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
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

// ---------------------------------------------------------------------------
//  build()
// ---------------------------------------------------------------------------

void FourLevelTopologyStore::auto_detect_budgets_(
    std::size_t& l1_bytes,
    std::size_t& l2_bytes) const
{
    // Total cache budget = 70% of MemAvailable. Of that, 25% goes to
    // L1 and 75% goes to L2 — as documented in design §2.2 / D2.
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

void FourLevelTopologyStore::scan_bpt_into_per_node_(
    BPlusTree<3>*                                              index,
    std::vector<std::vector<AdjEntry>>&                         per_node) const
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
        if (a >= per_node.size()) {
            per_node.resize(a + 1);
        }
        per_node[a].push_back(AdjEntry{ b, e });
    }
}

void FourLevelTopologyStore::populate_direction_(
    BPlusTree<3>*                              index,
    const std::vector<uint8_t>&                tiers,
    std::unique_ptr<L1HashCache>&              l1_out,
    std::unique_ptr<L2CompactCsr>&             l2_out) const
{
    l1_out = std::make_unique<L1HashCache>(tiers);
    l2_out = std::make_unique<L2CompactCsr>(/*hint=*/0);

    if (index == nullptr) {
        l2_out->freeze();
        return;
    }

    // Materialise per-node adjacency once. For papers100M scale this
    // dominates RAM during the build; the temporary vector is freed
    // immediately after distributing entries to L1 / L2 (tier-3/4
    // nodes are dropped — those tiers have no in-RAM home).
    std::vector<std::vector<AdjEntry>> per_node;
    per_node.resize(tiers.size());
    scan_bpt_into_per_node_(index, per_node);

    // First pass: distribute to L1 / L2 by tier.
    //
    // L2 must see add_node calls in the same order tiers are scanned
    // so the L2 row index is deterministic across runs (matters for
    // reproducible byte-identical sample output between sessions).
    for (std::size_t row_idx = 0; row_idx < per_node.size(); ++row_idx) {
        if (row_idx >= tiers.size()) break;
        const uint8_t tier = tiers[row_idx];
        auto& adj = per_node[row_idx];

        if (tier == 1) {
            // Phase 2 carry-forward (c): reserve(degree) before insert
            // so L1HashCache::total_bytes() accurately reflects RSS
            // (capacity == size at insertion time).
            std::vector<AdjEntry> tight;
            tight.reserve(adj.size());
            for (auto& e : adj) tight.push_back(e);
            l1_out->insert(/*src_node_id=*/row_idx,
                           std::move(tight),
                           /*row_idx=*/row_idx);
        } else if (tier == 2) {
            l2_out->add_node(/*src_node_id=*/row_idx, adj);
        }
        // tier == 3 / 4 -> drop. The L3 mmap or L4 BPT path serves
        // those nodes at lookup time.

        // Free the per-node vector eagerly to reduce peak RSS.
        std::vector<AdjEntry>().swap(adj);
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

    // Step 1: compute budgets.
    std::size_t l1_bytes = 0;
    std::size_t l2_bytes = 0;
    auto_detect_budgets_(l1_bytes, l2_bytes);

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

        populate_direction_(fwd_bpt_, owned_tier_assignment_,
                            owned_l1_fwd_, owned_l2_fwd_);
        populate_direction_(rev_bpt_, owned_tier_assignment_,
                            owned_l1_rev_, owned_l2_rev_);
    } else {
        // Production path: use TopologyAccessor + TopologyFrequencyProfiler.
        TopologyAccessor accessor(*storage_);
        TopologyFrequencyProfiler profiler(accessor, projection_dir_);
        profiler.compute(config_.orientation);

        const auto& freq = profiler.frequency();
        const std::size_t n = freq.size();
        uint64_t total_freq = 0;
        for (uint64_t f : freq) total_freq += f;
        const double avg_degree =
            (n > 0)
                ? static_cast<double>(total_freq) / static_cast<double>(n)
                : 0.0;

        owned_tier_assignment_ = compute_tier_assignment(
            freq, l1_bytes, l2_bytes, avg_degree);
        tier_lookup_ref_ = &owned_tier_assignment_;

        // Build forward direction when needed.
        if (config_.orientation == EdgeOrientation::NATURAL ||
            config_.orientation == EdgeOrientation::UNDIRECTED)
        {
            populate_direction_(fwd_bpt_, owned_tier_assignment_,
                                owned_l1_fwd_, owned_l2_fwd_);
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
            populate_direction_(rev_bpt_, owned_tier_assignment_,
                                owned_l1_rev_, owned_l2_rev_);
        } else {
            owned_l1_rev_ = std::make_unique<L1HashCache>(owned_tier_assignment_);
            owned_l2_rev_ = std::make_unique<L2CompactCsr>(0);
            owned_l2_rev_->freeze();
        }
    }

    // Wire the active references to the owned tier sources.
    l1_fwd_ = owned_l1_fwd_.get();
    l1_rev_ = owned_l1_rev_.get();
    l2_fwd_ = owned_l2_fwd_.get();
    l2_rev_ = owned_l2_rev_.get();

    // Open Spec #4-B sidecar readers when requested.
    open_l3_sidecars_();

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

    // Default row_lookup_ for the Phase 3 ctor: identity over ObjectId.id.
    // Production projections enumerate row indices 0..N-1 and the
    // row_idx == ObjectId.id assumption holds (matching the
    // TopologyFrequencyProfiler's degree pass at row index `i ==
    // ObjectId(i)`). When a projection ever moves to a non-identity
    // RowMapping, this closure is the single point of update.
    row_lookup_ = [](ObjectId v) -> uint64_t { return v.id; };

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

    // UNDIRECTED merge: collect dst ids from both directions, dedupe.
    Neighbors out;
    out.tier = 4;  // materialise into l4_owned for the merge result.

    auto fwd = get_out_neighbors(v);
    auto rev = get_in_neighbors(v);

    std::unordered_set<uint64_t> seen;
    seen.reserve(fwd.size() + rev.size());

    auto append = [&](uint64_t dst, uint64_t eid) {
        if (seen.insert(dst).second) {
            out.l4_owned.push_back(AdjEntry{ dst, eid });
        }
    };
    fwd.for_each_with_edge_id(append);
    rev.for_each_with_edge_id(append);

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
            out.l1   = l1.get(v.id);
            out.tier = 1;
            return out;
        }
        case 2: {
            auto span = l2.get(v.id);
            out.l2_col_idx = span.first;
            out.l2_size    = span.second;
            out.tier       = 2;
            return out;
        }
        case 3: {
            if (l3 != nullptr && l3->has_data()
                && row_idx < l3->num_nodes())
            {
                auto dst_span = l3->neighbors(row_idx);
                out.l3_col_idx = dst_span.data();
                out.l3_size    = dst_span.size();
                if (l3->has_edge_ids()) {
                    auto eid_span = l3->edge_ids(row_idx);
                    if (eid_span.size() == dst_span.size()) {
                        out.l3_edge_ids = eid_span.data();
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
    return bytes;
}

}  // namespace mdb::gnn
