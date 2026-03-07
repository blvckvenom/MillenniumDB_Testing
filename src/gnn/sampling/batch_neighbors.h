#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "graph_models/object_id.h"

namespace mdb::gnn {

/**
 * @brief Result of batch neighbor sampling.
 *
 * Maps each seed node to its sampled neighbors and edge IDs.
 * Used by LeapfrogGnnSampler, SortedBatchSampler, SeekBasedGnnSampler,
 * and BasicKhopSampler.
 */
struct BatchNeighbors {
    /// node_id -> [(neighbor_id, edge_id), ...]
    std::unordered_map<uint64_t, std::vector<std::pair<ObjectId, ObjectId>>> neighbors;

    /// Total neighbors collected across all seeds
    size_t total_neighbors() const {
        size_t count = 0;
        for (const auto& [_, vec] : neighbors) {
            count += vec.size();
        }
        return count;
    }

    /// Get neighbors for a specific node (by ObjectId)
    const std::vector<std::pair<ObjectId, ObjectId>>& get(ObjectId node_id) const {
        static const std::vector<std::pair<ObjectId, ObjectId>> empty;
        auto it = neighbors.find(node_id.id);
        return it != neighbors.end() ? it->second : empty;
    }

    /// Check if a node has any neighbors (by raw uint64_t)
    bool has_neighbors(uint64_t node_id) const {
        auto it = neighbors.find(node_id);
        return it != neighbors.end() && !it->second.empty();
    }

    /// Check if a node has any neighbors (by ObjectId)
    bool has_neighbors(ObjectId node_id) const {
        return has_neighbors(node_id.id);
    }
};

} // namespace mdb::gnn
