#include "gnn/sampling/sorted_batch_sampler.h"

#include <algorithm>
#include <numeric>

#include "graph_models/gql/projection/projection_storage.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace mdb::gnn {

// =============================================================================
// Implementation Details
// =============================================================================

struct SortedBatchSampler::Impl {
    GQL::ProjectionStorage& storage;
    EdgeOrientation orientation;
    uint64_t fanout = 0;  // 0 = unlimited
    uint64_t reservoir_threshold = 10000;
    uint64_t random_seed = 42;
    std::mt19937_64 rng;

    // Statistics
    uint64_t pages_accessed = 0;
    uint64_t edges_scanned = 0;

    Impl(GQL::ProjectionStorage& storage_, EdgeOrientation orientation_)
        : storage(storage_)
        , orientation(orientation_)
        , rng(random_seed)
    {
    }

    /**
     * @brief Get the appropriate B+tree index based on orientation.
     *
     * - NATURAL/UNDIRECTED: from_to_edge index (traverse outgoing)
     * - REVERSE: to_from_edge index (traverse incoming)
     */
    BPlusTree<3>* get_index() {
        switch (orientation) {
            case EdgeOrientation::NATURAL:
            case EdgeOrientation::UNDIRECTED:
                return storage.get_from_to_edge_index();
            case EdgeOrientation::REVERSE:
                return storage.get_to_from_edge_index();
        }
        return storage.get_from_to_edge_index();
    }

    /**
     * @brief Sample k elements from a vector using partial Fisher-Yates.
     */
    template<typename T>
    std::vector<T> sample_k(std::vector<T>& vec, size_t k) {
        if (k >= vec.size()) {
            return vec;
        }

        // Partial Fisher-Yates shuffle
        for (size_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<size_t> dist(i, vec.size() - 1);
            size_t j = dist(rng);
            std::swap(vec[i], vec[j]);
        }

        return std::vector<T>(vec.begin(), vec.begin() + k);
    }

    /**
     * @brief Reservoir sampling during streaming iteration.
     *
     * Maintains a reservoir of size k while iterating through stream.
     * Uses Algorithm R (Vitter, 1985).
     */
    void reservoir_add(
        std::vector<std::pair<ObjectId, ObjectId>>& reservoir,
        const std::pair<ObjectId, ObjectId>& item,
        uint64_t seen_count,
        uint64_t k
    ) {
        if (reservoir.size() < k) {
            reservoir.push_back(item);
        } else {
            std::uniform_int_distribution<uint64_t> dist(0, seen_count - 1);
            uint64_t j = dist(rng);
            if (j < k) {
                reservoir[j] = item;
            }
        }
    }

    /**
     * @brief Core sorted batch sampling algorithm.
     *
     * Single B+tree traversal for the entire batch.
     */
    BatchNeighbors do_sample_batch(
        const std::vector<ObjectId>& nodes,
        bool apply_sampling
    ) {
        BatchNeighbors result;
        pages_accessed = 0;
        edges_scanned = 0;

        if (nodes.empty()) {
            return result;
        }

        BPlusTree<3>* index = get_index();
        if (!index) {
            return result;
        }

        // Sort nodes for sequential access
        std::vector<ObjectId> sorted_nodes = nodes;
        std::sort(sorted_nodes.begin(), sorted_nodes.end(),
                  [](const ObjectId& a, const ObjectId& b) { return a.id < b.id; });

        // Remove duplicates
        sorted_nodes.erase(
            std::unique(sorted_nodes.begin(), sorted_nodes.end(),
                       [](const ObjectId& a, const ObjectId& b) { return a.id == b.id; }),
            sorted_nodes.end()
        );

        // Create node set for O(1) lookup
        std::unordered_set<uint64_t> node_set;
        for (const auto& node : sorted_nodes) {
            node_set.insert(node.id);
        }

        // Range: [first_node, last_node] with all possible edges
        Record<3> min_record = {sorted_nodes.front().id, 0, 0};
        Record<3> max_record = {sorted_nodes.back().id, UINT64_MAX, UINT64_MAX};

        bool interruption_requested = false;
        auto iter = index->get_range(&interruption_requested, min_record, max_record);

        // Track current node being processed
        size_t node_idx = 0;
        uint64_t current_node_id = sorted_nodes[0].id;
        std::vector<std::pair<ObjectId, ObjectId>> current_neighbors;
        uint64_t current_seen_count = 0;

        const Record<3>* record;
        while ((record = iter.next()) != nullptr) {
            edges_scanned++;

            uint64_t edge_src = (*record)[0];  // First key is the "from" node
            uint64_t edge_dst = (*record)[1];  // Second key is the "to" node
            uint64_t edge_id = (*record)[2];

            // Skip edges from nodes not in our batch
            if (node_set.find(edge_src) == node_set.end()) {
                continue;
            }

            // If we've moved to a new source node, finalize the previous one
            while (node_idx < sorted_nodes.size() && sorted_nodes[node_idx].id < edge_src) {
                // Finalize previous node
                if (!current_neighbors.empty()) {
                    if (apply_sampling && fanout > 0 && current_neighbors.size() > fanout) {
                        result.neighbors[current_node_id] = sample_k(current_neighbors, fanout);
                    } else {
                        result.neighbors[current_node_id] = std::move(current_neighbors);
                    }
                }
                current_neighbors.clear();
                current_seen_count = 0;
                node_idx++;
                if (node_idx < sorted_nodes.size()) {
                    current_node_id = sorted_nodes[node_idx].id;
                }
            }

            // Process this edge
            if (node_idx < sorted_nodes.size() && sorted_nodes[node_idx].id == edge_src) {
                current_node_id = edge_src;
                current_seen_count++;

                std::pair<ObjectId, ObjectId> neighbor_edge = {
                    ObjectId(edge_dst),
                    ObjectId(edge_id)
                };

                if (apply_sampling && fanout > 0 && current_seen_count > reservoir_threshold) {
                    // Use reservoir sampling for high-degree nodes
                    reservoir_add(current_neighbors, neighbor_edge, current_seen_count, fanout);
                } else {
                    current_neighbors.push_back(neighbor_edge);
                }
            }
        }

        // Finalize remaining nodes
        while (node_idx < sorted_nodes.size()) {
            if (!current_neighbors.empty()) {
                if (apply_sampling && fanout > 0 && current_neighbors.size() > fanout) {
                    result.neighbors[current_node_id] = sample_k(current_neighbors, fanout);
                } else {
                    result.neighbors[current_node_id] = std::move(current_neighbors);
                }
            }
            current_neighbors.clear();
            current_seen_count = 0;
            node_idx++;
            if (node_idx < sorted_nodes.size()) {
                current_node_id = sorted_nodes[node_idx].id;
            }
        }

        // Handle UNDIRECTED: also traverse reverse index
        if (orientation == EdgeOrientation::UNDIRECTED) {
            add_reverse_edges(sorted_nodes, node_set, result, apply_sampling);
        }

        return result;
    }

    /**
     * @brief Add reverse edges for UNDIRECTED orientation.
     */
    void add_reverse_edges(
        const std::vector<ObjectId>& sorted_nodes,
        const std::unordered_set<uint64_t>& node_set,
        BatchNeighbors& result,
        bool apply_sampling
    ) {
        BPlusTree<3>* reverse_index = storage.get_to_from_edge_index();
        if (!reverse_index) return;

        Record<3> min_record = {sorted_nodes.front().id, 0, 0};
        Record<3> max_record = {sorted_nodes.back().id, UINT64_MAX, UINT64_MAX};

        bool interruption_requested = false;
        auto iter = reverse_index->get_range(&interruption_requested, min_record, max_record);

        // Track seen edges to avoid duplicates
        std::unordered_set<uint64_t> seen_edges;
        for (const auto& [node_id, neighbors] : result.neighbors) {
            for (const auto& [_, edge_id] : neighbors) {
                seen_edges.insert(edge_id.id);
            }
        }

        const Record<3>* record;
        while ((record = iter.next()) != nullptr) {
            edges_scanned++;

            uint64_t edge_dst = (*record)[0];  // In to_from_edge, first key is "to"
            uint64_t edge_src = (*record)[1];
            uint64_t edge_id = (*record)[2];

            // Skip if not in our batch or already seen
            if (node_set.find(edge_dst) == node_set.end()) continue;
            if (seen_edges.find(edge_id) != seen_edges.end()) continue;

            seen_edges.insert(edge_id);

            auto& neighbors = result.neighbors[edge_dst];
            neighbors.emplace_back(ObjectId(edge_src), ObjectId(edge_id));
        }

        // Apply sampling to combined results if needed
        if (apply_sampling && fanout > 0) {
            for (auto& [node_id, neighbors] : result.neighbors) {
                if (neighbors.size() > fanout) {
                    neighbors = sample_k(neighbors, fanout);
                }
            }
        }
    }
};

// =============================================================================
// SortedBatchSampler Public Interface
// =============================================================================

SortedBatchSampler::SortedBatchSampler(
    GQL::ProjectionStorage& storage,
    EdgeOrientation orientation
)
    : impl_(std::make_unique<Impl>(storage, orientation))
{
}

SortedBatchSampler::~SortedBatchSampler() = default;

SortedBatchSampler::SortedBatchSampler(SortedBatchSampler&&) noexcept = default;
SortedBatchSampler& SortedBatchSampler::operator=(SortedBatchSampler&&) noexcept = default;

void SortedBatchSampler::set_fanout(uint64_t fanout) {
    impl_->fanout = fanout;
}

uint64_t SortedBatchSampler::get_fanout() const {
    return impl_->fanout;
}

void SortedBatchSampler::set_random_seed(uint64_t seed) {
    impl_->random_seed = seed;
    impl_->rng.seed(seed);
}

void SortedBatchSampler::set_orientation(EdgeOrientation orientation) {
    impl_->orientation = orientation;
}

EdgeOrientation SortedBatchSampler::get_orientation() const {
    return impl_->orientation;
}

void SortedBatchSampler::set_reservoir_threshold(uint64_t threshold) {
    impl_->reservoir_threshold = threshold;
}

BatchNeighbors SortedBatchSampler::sample_batch(const std::vector<ObjectId>& nodes) {
    return impl_->do_sample_batch(nodes, true);
}

BatchNeighbors SortedBatchSampler::collect_all_batch(const std::vector<ObjectId>& nodes) {
    return impl_->do_sample_batch(nodes, false);
}

uint64_t SortedBatchSampler::last_pages_accessed() const {
    return impl_->pages_accessed;
}

uint64_t SortedBatchSampler::last_edges_scanned() const {
    return impl_->edges_scanned;
}

} // namespace mdb::gnn
