#include "gnn/projection/l1_hash_cache.h"

#include <cstddef>

#include "gnn/projection/topology_frequency_profiler.h"

namespace mdb::gnn {

L1HashCache::L1HashCache(const std::vector<uint8_t>& tier_assignment)
    : tier_assignment_(tier_assignment) {}

void L1HashCache::insert(uint64_t              src_node_id,
                         std::vector<AdjEntry> neighbors,
                         std::size_t           row_idx)
{
    // Silently drop inserts whose row index is out of range, or whose
    // tier_assignment entry is not 1. The orchestrator is permitted to
    // call insert() for every scanned node — only L1-tagged ones land
    // here.
    if (row_idx >= tier_assignment_.size())     return;
    if (tier_assignment_[row_idx] != uint8_t{1}) return;

    entries_[src_node_id] = std::move(neighbors);
}

L1HashCache::Span L1HashCache::get(uint64_t src_node_id) const {
    auto it = entries_.find(src_node_id);
    if (it == entries_.end()) {
        return Span{};
    }
    return Span{ it->second.data(), it->second.size() };
}

bool L1HashCache::contains(uint64_t src_node_id) const {
    return entries_.find(src_node_id) != entries_.end();
}

std::size_t L1HashCache::total_bytes() const {
    // Per the topology-frequency profiler contract (topology_frequency_profiler.h):
    //   bytes(node) = kL1NodeFixedOverhead + kL1PerEdgeBytes * degree
    //
    // The constant captures the hash-bucket + vector-header overhead
    // (kL1NodeFixedOverhead = 56) plus 16 bytes per AdjEntry
    // (kL1PerEdgeBytes = 16). The profiler relies on this same formula
    // to compute how many high-frequency nodes fit in the L1 RAM budget
    // before spilling the remainder to the L2 compact CSR or lower tiers.
    std::size_t bytes = 0;
    for (const auto& kv : entries_) {
        bytes += kL1NodeFixedOverhead;
        bytes += kL1PerEdgeBytes * kv.second.size();
    }
    return bytes;
}

std::size_t L1HashCache::total_edges() const {
    std::size_t edges = 0;
    for (const auto& kv : entries_) {
        edges += kv.second.size();
    }
    return edges;
}

}  // namespace mdb::gnn
