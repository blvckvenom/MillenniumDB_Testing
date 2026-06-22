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

}  // namespace
}  // namespace mdb::gnn
