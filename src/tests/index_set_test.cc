// src/tests/index_set_test.cc
//
// Unit tests for the IndexSet enum and preset mask helper introduced by
// Spec #3 T3.3. Validates the preset-to-ProjectionIndex bitmask mapping,
// string parsing (including rejection of invalid / lowercase input), and
// canonical name conversion.

#include <cstdint>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"

namespace {

// Helper: extract underlying uint32_t from a ProjectionIndex bitmask.
constexpr uint32_t raw(GQL::ProjectionIndex m) {
    return static_cast<uint32_t>(m);
}

TEST(IndexSet, AllPresetIncludesEveryIndex) {
    const auto mask = GQL::project_index_mask_for(GQL::IndexSet::ALL);
    EXPECT_EQ(raw(mask), raw(GQL::ProjectionIndex::ALL));
    EXPECT_EQ(raw(mask), 0x3FFFu);
}

TEST(IndexSet, GnnMinimalContainsExpectedFlags) {
    const auto mask = GQL::project_index_mask_for(GQL::IndexSet::GNN_MINIMAL);

    // Must contain the 5 GNN k-hop sampling indexes.
    EXPECT_TRUE(GQL::has_flag(mask, GQL::ProjectionIndex::NODES));
    EXPECT_TRUE(GQL::has_flag(mask, GQL::ProjectionIndex::NODE_LABEL));
    EXPECT_TRUE(GQL::has_flag(mask, GQL::ProjectionIndex::LABEL_NODE));
    EXPECT_TRUE(GQL::has_flag(mask, GQL::ProjectionIndex::FROM_TO_EDGE));
    EXPECT_TRUE(GQL::has_flag(mask, GQL::ProjectionIndex::TO_FROM_EDGE));

    // Must NOT contain edge-label, property, direction, or edge-id indexes.
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_LABEL));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::LABEL_EDGE));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_DIRECTION));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_FROM_TO));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_N1_N2));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::NODE_KEY_VALUE));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::KEY_VALUE_NODE));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_KEY_VALUE));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::KEY_VALUE_EDGE));
}

TEST(IndexSet, ReadonlyTraversalExtendsGnnMinimal) {
    const auto gnn = GQL::project_index_mask_for(GQL::IndexSet::GNN_MINIMAL);
    const auto ro  = GQL::project_index_mask_for(GQL::IndexSet::READONLY_TRAVERSAL);
    // Every bit set in GNN_MINIMAL must also be set in READONLY_TRAVERSAL.
    EXPECT_EQ(raw(gnn) & raw(ro), raw(gnn));
}

TEST(IndexSet, ReadonlyDoesNotIncludeEdgeFromTo) {
    const auto mask = GQL::project_index_mask_for(GQL::IndexSet::READONLY_TRAVERSAL);
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_FROM_TO));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_N1_N2));
    EXPECT_FALSE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_DIRECTION));
}

TEST(IndexSet, GnnMinimalIsStrictSubsetOfReadonly) {
    const auto gnn = GQL::project_index_mask_for(GQL::IndexSet::GNN_MINIMAL);
    const auto ro  = GQL::project_index_mask_for(GQL::IndexSet::READONLY_TRAVERSAL);
    // Strict subset: difference is non-empty.
    EXPECT_NE(raw(gnn), raw(ro));
    const uint32_t extra = raw(ro) & ~raw(gnn);
    EXPECT_NE(extra, 0u);

    // The extra bits in READONLY_TRAVERSAL are exactly the edge-label indexes.
    const uint32_t expected_extra =
        raw(GQL::ProjectionIndex::EDGE_LABEL) | raw(GQL::ProjectionIndex::LABEL_EDGE);
    EXPECT_EQ(extra, expected_extra);
}

TEST(IndexSet, ReadonlyContainsEdgeLabelIndexes) {
    const auto mask = GQL::project_index_mask_for(GQL::IndexSet::READONLY_TRAVERSAL);
    EXPECT_TRUE(GQL::has_flag(mask, GQL::ProjectionIndex::EDGE_LABEL));
    EXPECT_TRUE(GQL::has_flag(mask, GQL::ProjectionIndex::LABEL_EDGE));
}

TEST(IndexSet, ParseAllReturnsAll) {
    EXPECT_EQ(GQL::parse_index_set("ALL"), GQL::IndexSet::ALL);
}

TEST(IndexSet, ParseGnnMinimalReturnsGnnMinimal) {
    EXPECT_EQ(GQL::parse_index_set("GNN_MINIMAL"), GQL::IndexSet::GNN_MINIMAL);
}

TEST(IndexSet, ParseReadonlyTraversalReturnsReadonly) {
    EXPECT_EQ(GQL::parse_index_set("READONLY_TRAVERSAL"),
              GQL::IndexSet::READONLY_TRAVERSAL);
}

TEST(IndexSet, ParseUnknownThrows) {
    try {
        (void) GQL::parse_index_set("foo");
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("ALL"), std::string::npos);
        EXPECT_NE(msg.find("GNN_MINIMAL"), std::string::npos);
        EXPECT_NE(msg.find("READONLY_TRAVERSAL"), std::string::npos);
    } catch (...) {
        FAIL() << "Expected std::invalid_argument, got different exception";
    }
}

TEST(IndexSet, ParseEmptyStringThrows) {
    EXPECT_THROW((void) GQL::parse_index_set(""), std::invalid_argument);
}

TEST(IndexSet, ParseLowercaseRejected) {
    // parse_index_set is case-sensitive.
    EXPECT_THROW((void) GQL::parse_index_set("gnn_minimal"),
                 std::invalid_argument);
    EXPECT_THROW((void) GQL::parse_index_set("all"),
                 std::invalid_argument);
    EXPECT_THROW((void) GQL::parse_index_set("readonly_traversal"),
                 std::invalid_argument);
}

TEST(IndexSet, IndexSetNameReturnsCanonical) {
    EXPECT_STREQ(GQL::index_set_name(GQL::IndexSet::ALL), "ALL");
    EXPECT_STREQ(GQL::index_set_name(GQL::IndexSet::GNN_MINIMAL), "GNN_MINIMAL");
    EXPECT_STREQ(GQL::index_set_name(GQL::IndexSet::READONLY_TRAVERSAL),
                 "READONLY_TRAVERSAL");
}

TEST(IndexSet, RoundTripNameAndParse) {
    for (auto preset : {GQL::IndexSet::ALL,
                        GQL::IndexSet::GNN_MINIMAL,
                        GQL::IndexSet::READONLY_TRAVERSAL}) {
        const char* name = GQL::index_set_name(preset);
        ASSERT_NE(name, nullptr);
        EXPECT_EQ(GQL::parse_index_set(name), preset);
    }
}

} // namespace
