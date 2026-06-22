// symmetric_topology_view_test.cc
//
// Unit tests for the symmetric-CSR single-slice GPU-UVA wiring: the
// useSymmetricTopology AUTO/ON/OFF resolution, the in-RAM merged undirected
// arrays, the FORWARD_ONLY single-slice pin (never BOTH), and the lazy
// materialization. CI-friendly: passes with OR without a GPU (the pin is a
// documented no-op without a runtime GPU, exactly like PinnedTopologyView).
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "gnn/projection/edge_orientation.h"
#include "gnn/projection/four_level_topology_store.h"
#include "gnn/projection/pinned_topology_view.h"
#include "gnn/sampling/sampling_backend_plan.h"
#include "gnn/sampling/sampling_config.h"

namespace mdb::gnn {
namespace {

TEST(SymmetricResolution, AutoOnlyForUndirected) {
    SamplingConfig c;
    c.use_symmetric_topology = SamplingConfig::SymmetricTopologyMode::AUTO;
    EXPECT_TRUE(c.symmetric_resolved_on(EdgeOrientation::UNDIRECTED));
    EXPECT_FALSE(c.symmetric_resolved_on(EdgeOrientation::NATURAL));
    EXPECT_FALSE(c.symmetric_resolved_on(EdgeOrientation::REVERSE));
}

TEST(SymmetricResolution, OnForcesEvenNonUndirected) {
    SamplingConfig c;
    c.use_symmetric_topology = SamplingConfig::SymmetricTopologyMode::ON;
    EXPECT_TRUE(c.symmetric_resolved_on(EdgeOrientation::UNDIRECTED));
    EXPECT_TRUE(c.symmetric_resolved_on(EdgeOrientation::NATURAL));
}

TEST(SymmetricResolution, OffAlwaysFalse) {
    SamplingConfig c;
    c.use_symmetric_topology = SamplingConfig::SymmetricTopologyMode::OFF;
    EXPECT_FALSE(c.symmetric_resolved_on(EdgeOrientation::UNDIRECTED));
}

TEST(SymmetricPlan, UseSymmetricDefaultsFalse) {
    SamplingBackendPlan p;
    EXPECT_FALSE(p.use_symmetric);  // pure planner never sets it; the engine does
}

// --- merge math, decoupled from the live reader ---

TEST(SymmetricMerge, OutPlusIn_NodeIdDedup) {
    // node 0: out={1,2} in={2}  -> undirected {1,2}   (2 dedup'd)
    // node 1: out={}   in={0}   -> undirected {0}
    // node 2: out={0}  in={0,1} -> undirected {0,1}   (0 dedup'd)
    std::vector<uint64_t> fwd_rp{0, 2, 2, 3}, rev_rp{0, 1, 2, 4};
    std::vector<uint32_t> fwd_ci{1, 2, 0};
    std::vector<uint32_t> rev_ci{2, 0, 0, 1};
    std::vector<uint64_t> sym_rp;
    std::vector<uint32_t> sym_ci;
    merge_symmetric_csr_node_dedup(fwd_rp, fwd_ci, rev_rp, rev_ci, sym_rp, sym_ci);
    EXPECT_EQ(sym_rp, (std::vector<uint64_t>{0, 2, 3, 5}));
    // row order: out first, then in-not-already-present
    EXPECT_EQ(sym_ci, (std::vector<uint32_t>{1, 2, 0, 0, 1}));
}

TEST(SymmetricMerge, EdgeIdDistinct_NoDedup) {
    // distinct edge_ids -> out ++ in with NO dedup.
    std::vector<uint64_t> fwd_rp{0, 1, 1}, rev_rp{0, 0, 1};
    std::vector<uint32_t> fwd_ci{1};  // 0 -> 1
    std::vector<uint32_t> rev_ci{0};  // node 1 in = {0}
    std::vector<uint64_t> sym_rp;
    std::vector<uint32_t> sym_ci;
    merge_symmetric_csr_concat(fwd_rp, fwd_ci, rev_rp, rev_ci, sym_rp, sym_ci);
    EXPECT_EQ(sym_rp, (std::vector<uint64_t>{0, 1, 2}));
    EXPECT_EQ(sym_ci, (std::vector<uint32_t>{1, 0}));
}

// The merged undirected slice is served as fwd ONLY (rev null), never BOTH.
// CI-friendly: without a runtime GPU the pin is a documented no-op.
TEST(SymmetricPin, SingleSliceForwardOnly_RevNull) {
    std::vector<uint64_t> rp = {0, 2, 3, 5};
    std::vector<uint32_t> ci = {1, 2, 0, 0, 1};
    HostCsrArrays sym;
    sym.row_ptr = rp.data();
    sym.col_idx = ci.data();
    sym.n_rows = 3;
    sym.n_edges = 5;
    sym.dst_type_tag = static_cast<uint64_t>(0xD4) << 56;

    PinnedTopologyView view;
    view.build_and_register(sym, HostCsrArrays{});  // rev null == symmetric pin
    if (!view.is_registered()) {                    // no GPU: documented no-op
        EXPECT_EQ(view.fwd(), nullptr);
        EXPECT_EQ(view.rev(), nullptr);
        return;
    }
    ASSERT_NE(view.fwd(), nullptr);
    EXPECT_EQ(view.rev(), nullptr);                 // NEVER BOTH for symmetric
    EXPECT_EQ(view.fwd()->n_rows, 3u);
    EXPECT_EQ(view.fwd()->n_edges, 5u);
}

}  // namespace
}  // namespace mdb::gnn
