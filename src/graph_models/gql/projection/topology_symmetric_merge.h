#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "graph_models/object_id.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"

namespace GQL::Projection {

// Replicates TopologyAccessor::get_neighbors_into(UNDIRECTED) node-list dedup
// (topology_accessor.cc:584-623): when has_edge_ids the dedup key is edge_id
// (distinct -> nothing removed); else the key is the neighbor node id. Emission
// order is out(u) first, then in(u) survivors. merged_dst/merged_eid are CLEARED
// and filled; returns the merged degree (= merged_dst.size()).
//
// Inputs are raw `ObjectId.id` values (NOT tag-stripped), matching the accessor
// which keys on the full `.id`. The bake must feed the same raw ids so the
// receptive field is byte-identical to today's runtime out+in+merge.
inline uint64_t merge_symmetric_row(const std::vector<uint64_t>& out_dst,
                                    const std::vector<uint64_t>& out_eid,
                                    const std::vector<uint64_t>& in_dst,
                                    const std::vector<uint64_t>& in_eid,
                                    bool has_edge_ids,
                                    std::vector<uint64_t>& merged_dst,
                                    std::vector<uint64_t>& merged_eid) {
    merged_dst.clear();
    merged_eid.clear();
    const std::size_t total = out_dst.size() + in_dst.size();
    merged_dst.reserve(total);
    merged_eid.reserve(total);
    std::unordered_set<uint64_t> seen;
    seen.reserve(total);
    for (std::size_t i = 0; i < out_dst.size(); ++i) {
        const uint64_t key = has_edge_ids ? out_eid[i] : out_dst[i];
        if (seen.insert(key).second) {
            merged_dst.push_back(out_dst[i]);
            merged_eid.push_back(out_eid[i]);
        }
    }
    for (std::size_t i = 0; i < in_dst.size(); ++i) {
        const uint64_t key = has_edge_ids ? in_eid[i] : in_dst[i];
        if (seen.insert(key).second) {
            merged_dst.push_back(in_dst[i]);
            merged_eid.push_back(in_eid[i]);
        }
    }
    return merged_dst.size();
}

// True iff the from_to_edge BPT holds >=2 distinct edges on the same (src,dst)
// (a meaningful parallel edge). The key layout (src,dst,edge_id) ascending makes
// such edges adjacent: equal [0]&[1], differing [2]. Tags are stripped before
// compare. A null BPT returns false (nothing to refuse). Used by the symmetric
// edge_id-drop bake to ABSTAIN on a true multigraph rather than silently
// node-id-dedup parallel edges away.
inline bool detect_parallel_edges(BPlusTree<3>* fwd_bpt, uint64_t /*num_nodes*/) {
    if (fwd_bpt == nullptr) {
        return false;
    }
    bool interrupt = false;
    Record<3> min_rec = {0, 0, 0};
    Record<3> max_rec = {UINT64_MAX, UINT64_MAX, UINT64_MAX};
    auto iter = fwd_bpt->get_range(&interrupt, min_rec, max_rec);
    const Record<3>* rec = nullptr;
    bool have_prev = false;
    uint64_t p_src = 0, p_dst = 0, p_eid = 0;
    while ((rec = iter.next()) != nullptr) {
        const uint64_t src = (*rec)[0] & ObjectId::VALUE_MASK;
        const uint64_t dst = (*rec)[1] & ObjectId::VALUE_MASK;
        const uint64_t eid = (*rec)[2] & ObjectId::VALUE_MASK;
        if (have_prev && src == p_src && dst == p_dst && eid != p_eid) {
            return true;  // two distinct edges on the same (src,dst)
        }
        p_src = src;
        p_dst = dst;
        p_eid = eid;
        have_prev = true;
    }
    return false;
}

}  // namespace GQL::Projection
