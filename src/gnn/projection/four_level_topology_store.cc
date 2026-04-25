#include "gnn/projection/four_level_topology_store.h"

#include <stdexcept>
#include <utility>

#include "graph_models/gql/projection/topology_snapshot_reader.h"

namespace mdb::gnn {

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

FourLevelTopologyStore::FourLevelTopologyStore(
    const L1HashCache&                                  l1_fwd,
    const L1HashCache&                                  l1_rev,
    const L2CompactCsr&                                 l2_fwd,
    const L2CompactCsr&                                 l2_rev,
    const GQL::Projection::TopologySnapshotReader*     l3_fwd,
    const GQL::Projection::TopologySnapshotReader*     l3_rev,
    L4Lookup                                            l4_fwd,
    L4Lookup                                            l4_rev,
    const std::vector<uint8_t>&                         tier_lookup,
    RowLookup                                           row_lookup,
    Config                                              config)
    : l1_fwd_(l1_fwd),
      l1_rev_(l1_rev),
      l2_fwd_(l2_fwd),
      l2_rev_(l2_rev),
      l3_fwd_(l3_fwd),
      l3_rev_(l3_rev),
      l4_fwd_(std::move(l4_fwd)),
      l4_rev_(std::move(l4_rev)),
      tier_lookup_(tier_lookup),
      row_lookup_(std::move(row_lookup)),
      config_(config)
{}

FourLevelTopologyStore::Neighbors
FourLevelTopologyStore::get_out_neighbors(ObjectId v) const
{
    return dispatch_(v, l1_fwd_, l2_fwd_, l3_fwd_, l4_fwd_);
}

FourLevelTopologyStore::Neighbors
FourLevelTopologyStore::get_in_neighbors(ObjectId v) const
{
    return dispatch_(v, l1_rev_, l2_rev_, l3_rev_, l4_rev_);
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

    // Map ObjectId -> row index. The callable is allowed to return
    // any value >= tier_lookup_.size() to signal "unknown row";
    // the dispatcher then treats the node as tier 4 (BPT direct) or
    // throws when no L4 is wired.
    const uint64_t row_idx = row_lookup_(v);
    const bool row_in_range = (row_idx < tier_lookup_.size());

    // Out-of-range row → tier 4 dispatch (or throw).
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

    const uint8_t tier = tier_lookup_[static_cast<std::size_t>(row_idx)];

    switch (tier) {
        case 1: {
            // L1 hot hash cache.
            out.l1   = l1.get(v.id);
            out.tier = 1;
            return out;
        }
        case 2: {
            // L2 compact CSR. col_idx_ holds uint32 dst row indexes.
            auto span = l2.get(v.id);
            out.l2_col_idx = span.first;
            out.l2_size    = span.second;
            out.tier       = 2;
            return out;
        }
        case 3: {
            // L3 mmap sidecar (Spec #4-B). Falls through to L4 when
            // the sidecar is absent (nullptr) or has_data() == false
            // (writer never produced the file or staleness rejected
            // it at open()).
            if (l3 != nullptr && l3->has_data()
                && row_idx < l3->num_nodes())
            {
                auto span      = l3->neighbors(row_idx);
                out.l3_col_idx = span.data();
                out.l3_size    = span.size();
                out.tier       = 3;
                return out;
            }
            // Fall through to L4.
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
            // Tier 4 (or any unrecognised tier) → BPT direct.
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

}  // namespace mdb::gnn
