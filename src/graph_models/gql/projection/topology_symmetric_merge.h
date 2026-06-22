#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

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

}  // namespace GQL::Projection
