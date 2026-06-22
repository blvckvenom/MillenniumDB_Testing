// Unit tests for the FourLevelTopologyStore symmetric (pre-merged undirected)
// tier: the members + accessor (this file), the per-row merge populate
// (extended in later tasks), the single-dispatch collapse, and the edge_id-drop
// gate. The symmetric tier reuses the SAME per-node tier assignment as the
// directional tiers (direction-agnostic), so a node's tier (L1/L2/L3/L4) is the
// same whether read out, in, or undirected.

#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/adj_entry.h"
#include "gnn/projection/four_level_topology_store.h"
#include "gnn/projection/l1_hash_cache.h"
#include "gnn/projection/l2_compact_csr.h"
#include "graph_models/object_id.h"

using mdb::gnn::AdjEntry;
using mdb::gnn::FourLevelTopologyStore;
using mdb::gnn::L1HashCache;
using mdb::gnn::L2CompactCsr;

// The symmetric tier is opt-in: a dispatcher-constructed store (no build())
// never has it, so is_symmetric_built() defaults false.
TEST(FourLevelTopologySym, SymTierDefaultsOff) {
    const std::vector<uint8_t> tiers = {1, 1};
    L1HashCache l1f(tiers), l1r(tiers);
    L2CompactCsr l2f, l2r;
    l2f.freeze();
    l2r.freeze();
    FourLevelTopologyStore::Config cfg;  // orientation UNDIRECTED default
    FourLevelTopologyStore store(
        l1f, l1r, l2f, l2r,
        /*l3_fwd=*/nullptr, /*l3_rev=*/nullptr,
        /*l4_fwd=*/{}, /*l4_rev=*/{},
        tiers, [](ObjectId v) { return v.id; }, cfg);
    EXPECT_FALSE(store.is_symmetric_built());
}
