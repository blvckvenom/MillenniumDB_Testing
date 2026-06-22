// Unit tests for merge_symmetric_row — the per-node out+in merge that the
// symmetric topology bake uses. It must replicate the canonical UNDIRECTED
// dedup the accessor performs (topology_accessor.cc:584-623): the dedup key is
// the edge_id when edge_ids are real+distinct (nothing removed), otherwise the
// neighbor node id; emission order is out(u) first, then in(u) survivors.
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/topology_symmetric_merge.h"

using GQL::Projection::merge_symmetric_row;

// Rule A: edge_ids are real+distinct (BTREE) -> out ++ in, NO dedup.
TEST(SymmetricMerge, RealEdgeIdsConcatNoDedup) {
    std::vector<uint64_t> out_dst{1, 2}, out_eid{100, 101};
    std::vector<uint64_t> in_dst{3, 1}, in_eid{102, 103};  // dst 1 repeats, eid distinct
    std::vector<uint64_t> md, me;
    uint64_t deg = merge_symmetric_row(out_dst, out_eid, in_dst, in_eid,
                                       /*has_edge_ids=*/true, md, me);
    EXPECT_EQ(deg, 4u);
    EXPECT_EQ(md, (std::vector<uint64_t>{1, 2, 3, 1}));  // node 1 kept twice
    EXPECT_EQ(me, (std::vector<uint64_t>{100, 101, 102, 103}));
}

// Rule B: edge_id==0 (CSR-hybrid) -> node-id dedup; in-side dup of an out node dropped.
TEST(SymmetricMerge, ZeroEdgeIdsNodeDedup) {
    std::vector<uint64_t> out_dst{1, 2}, out_eid{0, 0};
    std::vector<uint64_t> in_dst{3, 1}, in_eid{0, 0};  // dst 1 already present from out side
    std::vector<uint64_t> md, me;
    uint64_t deg = merge_symmetric_row(out_dst, out_eid, in_dst, in_eid,
                                       /*has_edge_ids=*/false, md, me);
    EXPECT_EQ(deg, 3u);
    EXPECT_EQ(md, (std::vector<uint64_t>{1, 2, 3}));  // node 1 deduped
    EXPECT_EQ(me, (std::vector<uint64_t>{0, 0, 0}));  // edge_ids stay zero
}

// Order contract: out(u) first, then in(u) survivors — byte-identical to accessor order.
TEST(SymmetricMerge, OrderOutThenIn) {
    std::vector<uint64_t> out_dst{5}, out_eid{0};
    std::vector<uint64_t> in_dst{4, 5}, in_eid{0, 0};
    std::vector<uint64_t> md, me;
    merge_symmetric_row(out_dst, out_eid, in_dst, in_eid, false, md, me);
    EXPECT_EQ(md, (std::vector<uint64_t>{5, 4}));  // 5 from out, 4 from in, 5-dup dropped
}

TEST(SymmetricMerge, EmptyBothSides) {
    std::vector<uint64_t> md, me;
    EXPECT_EQ(merge_symmetric_row({}, {}, {}, {}, false, md, me), 0u);
    EXPECT_TRUE(md.empty());
    EXPECT_TRUE(me.empty());
}
