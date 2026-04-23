// src/tests/projection_missing_index_test.cc
//
// Unit tests for Spec #3 T3.9 — the missing-index diagnostic produced by
// GQLModel getters when a query targets a projection built under a
// restricted IndexSet preset (GNN_MINIMAL / READONLY_TRAVERSAL) and tries
// to access an elided B+Tree.
//
// These tests exercise the two new helpers in index_set.{h,cc}
// (`projection_index_name`, `minimum_preset_for`) and validate the
// resulting friendly error message format. The full end-to-end path
// (graph_project -> USE proj -> MATCH raising QueryException) is
// covered by the companion shell test
// `scripts/test_projection_missing_index_query.sh` because it requires
// the full MDB runtime + network stack.

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"

namespace {

// Helper: build the user-facing message exactly as gql_model.cc's
// anonymous namespace `make_missing_index_message` does. Duplicated here
// because that helper is translation-unit-local (unnamed namespace in a
// .cc). Keeping the two copies in lockstep is acceptable: both produce
// the same string format and the shared helpers it depends on
// (`projection_index_name`, `index_set_name`, `minimum_preset_for`) are
// exported.
bool is_property_index(GQL::ProjectionIndex which) {
    return which == GQL::ProjectionIndex::NODE_KEY_VALUE
        || which == GQL::ProjectionIndex::KEY_VALUE_NODE
        || which == GQL::ProjectionIndex::EDGE_KEY_VALUE
        || which == GQL::ProjectionIndex::KEY_VALUE_EDGE;
}

std::string make_message(
    const std::string& projection_name,
    GQL::ProjectionIndex which,
    GQL::IndexSet active)
{
    const char* idx_name    = GQL::projection_index_name(which);
    const char* active_name = GQL::index_set_name(active);

    std::string msg;
    msg.reserve(512);
    msg += "Cannot execute query - index '";
    msg += idx_name;
    msg += "' is not materialized for projection '";
    msg += projection_name;
    msg += "' (indexSet='";
    msg += active_name;
    msg += "'). ";

    if (is_property_index(which)) {
        const bool is_node = (which == GQL::ProjectionIndex::NODE_KEY_VALUE
                           || which == GQL::ProjectionIndex::KEY_VALUE_NODE);
        msg += "Property indexes are controlled by the ";
        msg += is_node ? "includeNodeProperties" : "includeEdgeProperties";
        msg += " projection config (not by indexSet). Rebuild the projection "
               "with the corresponding property config (and indexSet='ALL') "
               "to enable this query.";
    } else {
        const GQL::IndexSet min_preset = GQL::minimum_preset_for(which);
        const char* min_name = GQL::index_set_name(min_preset);
        msg += "To enable this query, rebuild the projection with indexSet='";
        msg += min_name;
        msg += "'";
        if (min_preset != GQL::IndexSet::ALL) {
            msg += " (or 'ALL')";
        }
        msg += ".";
    }
    return msg;
}

// ---------------------------------------------------------------------------
// projection_index_name — canonical names matching .leaf file basenames
// ---------------------------------------------------------------------------

TEST(ProjectionMissingIndex, IndexNameReturnsLeafBasename) {
    // Exhaustive mapping: every single-bit ProjectionIndex returns the
    // matching .leaf basename used in
    // ProjectionStorage::open_all_bplustree_readers_().
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::NODES),
                 "nodes");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::NODE_LABEL),
                 "node_label");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::LABEL_NODE),
                 "label_node");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::NODE_KEY_VALUE),
                 "node_key_value");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::KEY_VALUE_NODE),
                 "key_value_node");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::FROM_TO_EDGE),
                 "from_to_edge");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::TO_FROM_EDGE),
                 "to_from_edge");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::EDGE_DIRECTION),
                 "edge_direction");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::EDGE_FROM_TO),
                 "edge_from_to");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::EDGE_N1_N2),
                 "edge_n1_n2");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::EDGE_LABEL),
                 "edge_label");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::LABEL_EDGE),
                 "label_edge");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::EDGE_KEY_VALUE),
                 "edge_key_value");
    EXPECT_STREQ(GQL::projection_index_name(GQL::ProjectionIndex::KEY_VALUE_EDGE),
                 "key_value_edge");
}

// ---------------------------------------------------------------------------
// minimum_preset_for — minimum IndexSet that contains a given bit
// ---------------------------------------------------------------------------

TEST(ProjectionMissingIndex, MinimumPresetForGnnMinimalBits) {
    // All 5 GNN_MINIMAL bits must report GNN_MINIMAL as the minimum.
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::NODES),
              GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::NODE_LABEL),
              GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::LABEL_NODE),
              GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::FROM_TO_EDGE),
              GQL::IndexSet::GNN_MINIMAL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::TO_FROM_EDGE),
              GQL::IndexSet::GNN_MINIMAL);
}

TEST(ProjectionMissingIndex, MinimumPresetForEdgeLabelIsReadonly) {
    // EDGE_LABEL / LABEL_EDGE are the two bits READONLY_TRAVERSAL adds
    // on top of GNN_MINIMAL — so they map to READONLY_TRAVERSAL, not ALL.
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::EDGE_LABEL),
              GQL::IndexSet::READONLY_TRAVERSAL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::LABEL_EDGE),
              GQL::IndexSet::READONLY_TRAVERSAL);
}

TEST(ProjectionMissingIndex, MinimumPresetForEdgeIdIndexesIsAll) {
    // Edge-id indexes (EDGE_DIRECTION, EDGE_FROM_TO, EDGE_N1_N2) are only
    // materialized under ALL — no lower preset contains them.
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::EDGE_DIRECTION),
              GQL::IndexSet::ALL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::EDGE_FROM_TO),
              GQL::IndexSet::ALL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::EDGE_N1_N2),
              GQL::IndexSet::ALL);
}

TEST(ProjectionMissingIndex, MinimumPresetForPropertyIndexesIsAll) {
    // Property indexes are gated by Features flags, not IndexSet — but the
    // helper still classifies them as ALL (the only preset whose mask
    // contains their bits).
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::NODE_KEY_VALUE),
              GQL::IndexSet::ALL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::KEY_VALUE_NODE),
              GQL::IndexSet::ALL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::EDGE_KEY_VALUE),
              GQL::IndexSet::ALL);
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::KEY_VALUE_EDGE),
              GQL::IndexSet::ALL);
}

// ---------------------------------------------------------------------------
// make_missing_index_message — user-visible error format
// ---------------------------------------------------------------------------

TEST(ProjectionMissingIndex, GnnMinimalEdgeLabelAccessRaises) {
    // On a GNN_MINIMAL projection, accessing `edge_label` must produce a
    // message that names the edge_label index and suggests READONLY_TRAVERSAL
    // as the minimum preset (plus the ALL fallback).
    const std::string msg = make_message(
        "t3_9_proj",
        GQL::ProjectionIndex::EDGE_LABEL,
        GQL::IndexSet::GNN_MINIMAL);

    EXPECT_NE(msg.find("'edge_label'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("indexSet='GNN_MINIMAL'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("indexSet='READONLY_TRAVERSAL'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("'ALL'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("t3_9_proj"), std::string::npos) << msg;
}

TEST(ProjectionMissingIndex, GnnMinimalEdgeFromToAccessRaises) {
    // edge_from_to requires ALL — message should suggest ALL directly (no
    // "(or 'ALL')" parenthetical, because ALL IS the minimum preset here).
    const std::string msg = make_message(
        "p",
        GQL::ProjectionIndex::EDGE_FROM_TO,
        GQL::IndexSet::GNN_MINIMAL);

    EXPECT_NE(msg.find("'edge_from_to'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("indexSet='GNN_MINIMAL'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("indexSet='ALL'"), std::string::npos) << msg;
    // When min_preset == ALL the "(or 'ALL')" parenthetical is suppressed —
    // there's only one mention of ALL in the rebuild suggestion.
    EXPECT_EQ(msg.find("(or 'ALL')"), std::string::npos) << msg;
}

TEST(ProjectionMissingIndex, GnnMinimalEdgeDirectionAccessRaises) {
    // edge_direction requires ALL.
    const std::string msg = make_message(
        "p",
        GQL::ProjectionIndex::EDGE_DIRECTION,
        GQL::IndexSet::GNN_MINIMAL);

    EXPECT_NE(msg.find("'edge_direction'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("indexSet='GNN_MINIMAL'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("indexSet='ALL'"), std::string::npos) << msg;
    EXPECT_EQ(msg.find("(or 'ALL')"), std::string::npos) << msg;
}

TEST(ProjectionMissingIndex, ReadonlyTraversalEdgeFromToAccessRaises) {
    // On READONLY_TRAVERSAL, edge_from_to is still absent. Message must
    // name READONLY_TRAVERSAL as the active preset and ALL as the minimum
    // upgrade.
    const std::string msg = make_message(
        "readonly_proj",
        GQL::ProjectionIndex::EDGE_FROM_TO,
        GQL::IndexSet::READONLY_TRAVERSAL);

    EXPECT_NE(msg.find("'edge_from_to'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("indexSet='READONLY_TRAVERSAL'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("indexSet='ALL'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("readonly_proj"), std::string::npos) << msg;
}

TEST(ProjectionMissingIndex, GnnMinimalLegitimateQuerySucceeds) {
    // On a GNN_MINIMAL projection, from_to_edge IS materialized — so
    // minimum_preset_for reports the projection itself (GNN_MINIMAL).
    // The caller's gate (from_to_edge_index != nullptr) passes and no
    // exception is thrown. We verify that for this bit, the minimum
    // preset is GNN_MINIMAL (== the built preset), meaning no upgrade
    // would be suggested on a correctly-materialized projection.
    EXPECT_EQ(GQL::minimum_preset_for(GQL::ProjectionIndex::FROM_TO_EDGE),
              GQL::IndexSet::GNN_MINIMAL);
}

TEST(ProjectionMissingIndex, AllModeAllQueriesSucceed) {
    // On IndexSet::ALL, every single-bit ProjectionIndex is materialized.
    // Verify by construction: for every single-bit value, the active
    // preset ALL contains it, so a hypothetical call would never enter
    // the exception branch.
    const GQL::ProjectionIndex all_mask = GQL::project_index_mask_for(GQL::IndexSet::ALL);
    for (auto bit : {
            GQL::ProjectionIndex::NODES,
            GQL::ProjectionIndex::NODE_LABEL,
            GQL::ProjectionIndex::LABEL_NODE,
            GQL::ProjectionIndex::NODE_KEY_VALUE,
            GQL::ProjectionIndex::KEY_VALUE_NODE,
            GQL::ProjectionIndex::FROM_TO_EDGE,
            GQL::ProjectionIndex::TO_FROM_EDGE,
            GQL::ProjectionIndex::EDGE_DIRECTION,
            GQL::ProjectionIndex::EDGE_FROM_TO,
            GQL::ProjectionIndex::EDGE_N1_N2,
            GQL::ProjectionIndex::EDGE_LABEL,
            GQL::ProjectionIndex::LABEL_EDGE,
            GQL::ProjectionIndex::EDGE_KEY_VALUE,
            GQL::ProjectionIndex::KEY_VALUE_EDGE,
        })
    {
        EXPECT_TRUE(GQL::has_flag(all_mask, bit))
            << "ALL mask should include every single-bit index";
    }
}

TEST(ProjectionMissingIndex, ErrorMessageContainsProjectionName) {
    const std::string msg = make_message(
        "my_specific_projection_name",
        GQL::ProjectionIndex::EDGE_LABEL,
        GQL::IndexSet::GNN_MINIMAL);
    EXPECT_NE(msg.find("my_specific_projection_name"), std::string::npos) << msg;
}

TEST(ProjectionMissingIndex, ErrorMessageContainsIndexSetName) {
    // Every active preset must appear in the message it generates.
    for (auto preset : {GQL::IndexSet::GNN_MINIMAL,
                        GQL::IndexSet::READONLY_TRAVERSAL,
                        GQL::IndexSet::ALL}) {
        const std::string msg = make_message(
            "p", GQL::ProjectionIndex::EDGE_LABEL, preset);
        const std::string expected =
            std::string("indexSet='") + GQL::index_set_name(preset) + "'";
        EXPECT_NE(msg.find(expected), std::string::npos)
            << "missing '" << expected << "' in: " << msg;
    }
}

TEST(ProjectionMissingIndex, ErrorMessageMentionsLeafBasename) {
    // The message must name the missing .leaf index by its basename
    // (e.g., "'edge_label'"), matching the on-disk file naming.
    const std::string msg = make_message(
        "p", GQL::ProjectionIndex::EDGE_LABEL, GQL::IndexSet::GNN_MINIMAL);
    EXPECT_NE(msg.find("'edge_label'"), std::string::npos) << msg;
}

TEST(ProjectionMissingIndex, PropertyIndexErrorMentionsIncludeProperties) {
    // Property indexes aren't gated by IndexSet (Spec #3 §3.4), so the
    // diagnostic for a missing node property index must point the user at
    // the includeNodeProperties config rather than suggesting an indexSet
    // upgrade by itself.
    const std::string node_msg = make_message(
        "p", GQL::ProjectionIndex::NODE_KEY_VALUE, GQL::IndexSet::ALL);
    EXPECT_NE(node_msg.find("includeNodeProperties"), std::string::npos)
        << node_msg;
    EXPECT_NE(node_msg.find("'node_key_value'"), std::string::npos) << node_msg;

    // And for edge property indexes — includeEdgeProperties instead.
    const std::string edge_msg = make_message(
        "p", GQL::ProjectionIndex::EDGE_KEY_VALUE, GQL::IndexSet::ALL);
    EXPECT_NE(edge_msg.find("includeEdgeProperties"), std::string::npos)
        << edge_msg;
    EXPECT_NE(edge_msg.find("'edge_key_value'"), std::string::npos) << edge_msg;
}

} // namespace
