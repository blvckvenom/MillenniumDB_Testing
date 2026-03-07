#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "graph_models/object_id.h"

namespace mdb::gnn {

/**
 * @brief Feature matrix with node ID mapping.
 *
 * Holds a tensor of node features along with mappings between
 * ObjectIds and row indices for efficient lookup.
 */
struct FeatureMatrix {
    torch::Tensor features;                          ///< [num_nodes, feature_dim]
    std::vector<ObjectId> node_ids;                  ///< ObjectId for each row
    std::unordered_map<uint64_t, int64_t> id_to_row; ///< Reverse mapping (ObjectId.id -> row)

    int64_t num_nodes() const {
        return features.defined() ? features.size(0) : 0;
    }

    int64_t feature_dim() const {
        return features.dim() > 1 ? features.size(1) : 0;
    }

    /**
     * @brief Get row index for a node.
     * @return Row index, or -1 if not found
     */
    int64_t get_row(ObjectId node_id) const {
        auto it = id_to_row.find(node_id.id);
        return (it != id_to_row.end()) ? it->second : -1;
    }

    int64_t get_row(uint64_t node_id) const {
        auto it = id_to_row.find(node_id);
        return (it != id_to_row.end()) ? it->second : -1;
    }

    /**
     * @brief Get features for a specific node.
     * @throws std::out_of_range if node not found in matrix
     */
    torch::Tensor get_node_features(ObjectId node_id) const {
        int64_t row = get_row(node_id);
        if (row < 0) {
            throw std::out_of_range("Node not found in FeatureMatrix");
        }
        return features[row];
    }
};

} // namespace mdb::gnn
