#include "leapfrog_gnn_sampler.h"

#include <algorithm>
#include <random>
#include <unordered_set>

#include "gnn/sampling/parallel_cursor_merger.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"

namespace mdb::gnn {

/**
 * @brief Reservoir sampling state for a single node.
 *
 * Implements Algorithm R (Vitter, 1985) for O(k) memory sampling.
 */
struct ReservoirState {
    std::vector<std::pair<ObjectId, ObjectId>> samples;  ///< Sampled (neighbor, edge) pairs
    uint64_t count = 0;  ///< Total neighbors seen

    void reset() {
        samples.clear();
        count = 0;
    }
};

struct LeapfrogGnnSampler::Impl {
    GQL::ProjectionStorage& storage;
    uint64_t fanout = 0;       ///< 0 = unlimited
    uint64_t random_seed = 42;
    std::mt19937_64 rng;

    // Statistics
    uint64_t edges_scanned = 0;
    uint64_t neighbors_collected = 0;

    // Interruption flag (required by B+Tree iterators)
    bool interruption_requested = false;

    explicit Impl(GQL::ProjectionStorage& storage_)
        : storage(storage_), rng(random_seed) {}

    /**
     * @brief Sample neighbors from index using coordinated sweep.
     *
     * @param sorted_nodes Sorted batch of seed node IDs
     * @param index B+Tree index (from_to_edge or to_from_edge)
     * @param effective_fanout Maximum neighbors per node
     * @return BatchNeighbors result
     */
    BatchNeighbors coordinated_sweep(
        const std::vector<uint64_t>& sorted_nodes,
        BPlusTree<3>* index,
        uint64_t effective_fanout
    ) {
        BatchNeighbors result;
        edges_scanned = 0;
        neighbors_collected = 0;

        if (sorted_nodes.empty() || !index) {
            return result;
        }

        // Initialize reservoir states for each node
        std::unordered_map<uint64_t, ReservoirState> reservoirs;
        for (uint64_t node_id : sorted_nodes) {
            reservoirs[node_id] = ReservoirState{};
            if (effective_fanout > 0) {
                reservoirs[node_id].samples.reserve(effective_fanout);
            }
        }

        // Create range covering all potential edges from batch nodes
        // Range: [min_node, 0, 0] to [max_node, MAX, MAX]
        uint64_t min_node = sorted_nodes.front();
        uint64_t max_node = sorted_nodes.back();

        Record<3> min_key = { min_node, 0, 0 };
        Record<3> max_key = { max_node, UINT64_MAX, UINT64_MAX };

        // Get range iterator
        auto iter = index->get_range(&interruption_requested, min_key, max_key);

        // Track current position in sorted_nodes
        size_t current_idx = 0;
        uint64_t current_node = sorted_nodes[current_idx];

        // Process edges in sorted order
        while (const Record<3>* record = iter.next()) {
            if (interruption_requested) {
                break;
            }

            edges_scanned++;

            uint64_t from_id = (*record)[0];
            uint64_t to_id = (*record)[1];
            uint64_t edge_id = (*record)[2];

            // Skip edges not from any batch node
            while (current_idx < sorted_nodes.size() && from_id > current_node) {
                current_idx++;
                if (current_idx < sorted_nodes.size()) {
                    current_node = sorted_nodes[current_idx];
                }
            }

            if (current_idx >= sorted_nodes.size()) {
                break;  // Processed all batch nodes
            }

            // Skip edges from nodes not in batch (gap between sorted_nodes)
            if (from_id < current_node) {
                continue;
            }

            // Edge is from current_node - process it
            auto& reservoir = reservoirs[current_node];
            reservoir.count++;

            // Apply reservoir sampling
            if (effective_fanout == 0 || reservoir.samples.size() < effective_fanout) {
                // Haven't reached fanout limit - just add
                reservoir.samples.emplace_back(ObjectId(to_id), ObjectId(edge_id));
                neighbors_collected++;
            } else {
                // Reservoir sampling: replace with probability fanout/count
                std::uniform_int_distribution<uint64_t> dist(0, reservoir.count - 1);
                uint64_t j = dist(rng);
                if (j < effective_fanout) {
                    reservoir.samples[j] = { ObjectId(to_id), ObjectId(edge_id) };
                }
            }
        }

        // Build result from reservoirs
        for (uint64_t node_id : sorted_nodes) {
            result.neighbors[node_id] = std::move(reservoirs[node_id].samples);
        }

        return result;
    }

    /**
     * @brief Sample with both from_to and to_from indexes for undirected.
     */
    BatchNeighbors sample_undirected(
        const std::vector<uint64_t>& sorted_nodes,
        uint64_t effective_fanout
    ) {
        // Sample from both directions
        auto result_out = coordinated_sweep(
            sorted_nodes,
            storage.get_from_to_edge_index(),
            effective_fanout
        );

        auto result_in = coordinated_sweep(
            sorted_nodes,
            storage.get_to_from_edge_index(),
            effective_fanout
        );

        // Merge results, deduplicating by (neighbor, edge): UNDIRECTED
        // projections store every edge in both indexes, so the two sweeps
        // return the same edge twice. Dedup matches ParallelCursorMerger
        // (SEEK) and TopologyAccessor (PER_NODE) semantics so the strategy
        // choice does not change sample content.
        BatchNeighbors merged;
        for (uint64_t node_id : sorted_nodes) {
            auto& out_neighbors = result_out.neighbors[node_id];
            auto& in_neighbors = result_in.neighbors[node_id];

            std::unordered_set<MergedEdge, MergedEdgeHash> seen;
            seen.reserve(out_neighbors.size() + in_neighbors.size());

            std::vector<std::pair<ObjectId, ObjectId>> combined;
            combined.reserve(out_neighbors.size() + in_neighbors.size());
            for (const auto* direction : { &out_neighbors, &in_neighbors }) {
                for (const auto& [neighbor, edge] : *direction) {
                    if (seen.insert(MergedEdge { neighbor.id, edge.id, direction == &out_neighbors })
                            .second)
                    {
                        combined.emplace_back(neighbor, edge);
                    }
                }
            }

            // If fanout is limited, we need to sub-sample the combined result
            if (effective_fanout > 0 && combined.size() > effective_fanout) {
                std::shuffle(combined.begin(), combined.end(), rng);
                combined.resize(effective_fanout);
            }

            merged.neighbors[node_id] = std::move(combined);
        }

        return merged;
    }
};

LeapfrogGnnSampler::LeapfrogGnnSampler(GQL::ProjectionStorage& storage)
    : impl_(std::make_unique<Impl>(storage)) {}

LeapfrogGnnSampler::~LeapfrogGnnSampler() = default;

LeapfrogGnnSampler::LeapfrogGnnSampler(LeapfrogGnnSampler&&) noexcept = default;
LeapfrogGnnSampler& LeapfrogGnnSampler::operator=(LeapfrogGnnSampler&&) noexcept = default;

void LeapfrogGnnSampler::set_fanout(uint64_t fanout) {
    impl_->fanout = fanout;
}

uint64_t LeapfrogGnnSampler::get_fanout() const {
    return impl_->fanout;
}

void LeapfrogGnnSampler::set_random_seed(uint64_t seed) {
    impl_->random_seed = seed;
    impl_->rng.seed(seed);
}

uint64_t LeapfrogGnnSampler::get_random_seed() const {
    return impl_->random_seed;
}

BatchNeighbors LeapfrogGnnSampler::sample_batch(
    const std::vector<ObjectId>& nodes,
    EdgeOrientation orientation
) {
    return sample_batch(nodes, impl_->fanout, orientation);
}

BatchNeighbors LeapfrogGnnSampler::sample_batch(
    const std::vector<ObjectId>& nodes,
    uint64_t fanout,
    EdgeOrientation orientation
) {
    if (nodes.empty()) {
        return BatchNeighbors{};
    }

    // Sort node IDs for coordinated sweep
    std::vector<uint64_t> sorted_ids;
    sorted_ids.reserve(nodes.size());
    for (const auto& node : nodes) {
        sorted_ids.push_back(node.id);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end());

    // Remove duplicates
    sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());

    // Select index based on orientation
    switch (orientation) {
        case EdgeOrientation::NATURAL:
            return impl_->coordinated_sweep(
                sorted_ids,
                impl_->storage.get_from_to_edge_index(),
                fanout
            );

        case EdgeOrientation::REVERSE:
            return impl_->coordinated_sweep(
                sorted_ids,
                impl_->storage.get_to_from_edge_index(),
                fanout
            );

        case EdgeOrientation::UNDIRECTED:
            return impl_->sample_undirected(sorted_ids, fanout);
    }

    return BatchNeighbors{};
}

uint64_t LeapfrogGnnSampler::last_edges_scanned() const {
    return impl_->edges_scanned;
}

uint64_t LeapfrogGnnSampler::last_neighbors_collected() const {
    return impl_->neighbors_collected;
}

} // namespace mdb::gnn
