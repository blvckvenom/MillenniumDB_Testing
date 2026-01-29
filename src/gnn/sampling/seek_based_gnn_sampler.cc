#include "gnn/sampling/seek_based_gnn_sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "graph_models/gql/projection/projection_storage.h"
#include "gnn/sampling/parallel_cursor_merger.h"
#include "gnn/sampling/seekable_edge_iter.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace mdb::gnn {

/**
 * @brief Reservoir sampling state for a single node.
 *
 * Implements Algorithm R (Vitter, 1985) for O(k) memory sampling.
 * Same structure as in LeapfrogGnnSampler for consistency.
 */
struct SeekReservoirState {
    std::vector<std::pair<ObjectId, ObjectId>> samples;  ///< Sampled (neighbor, edge) pairs
    uint64_t count = 0;  ///< Total neighbors seen

    void reset() {
        samples.clear();
        count = 0;
    }
};

struct SeekBasedGnnSampler::Impl {
    GQL::ProjectionStorage& storage;
    uint64_t random_seed = 42;
    std::mt19937_64 rng;

    // Statistics
    uint64_t seeks_performed = 0;
    uint64_t edges_iterated = 0;
    uint64_t neighbors_collected = 0;

    // Interruption flag (required by B+Tree iterators)
    bool interruption_requested = false;

    explicit Impl(GQL::ProjectionStorage& storage_)
        : storage(storage_), rng(random_seed) {}

    /**
     * @brief Sample neighbors from index using seek-based iteration.
     *
     * Core algorithm:
     * 1. For each node in sorted order, seek directly to that node's edges
     * 2. Collect edges with reservoir sampling until source changes
     * 3. Seek to next node (O(log E) jump)
     *
     * @param sorted_nodes Sorted batch of unique seed node IDs
     * @param index B+Tree index (from_to_edge or to_from_edge)
     * @param fanout Maximum neighbors per node (0 = unlimited)
     * @return BatchNeighbors result
     */
    BatchNeighbors seek_sample_directed(
        const std::vector<uint64_t>& sorted_nodes,
        BPlusTree<3>* index,
        uint64_t fanout
    ) {
        BatchNeighbors result;
        seeks_performed = 0;
        edges_iterated = 0;
        neighbors_collected = 0;

        if (sorted_nodes.empty() || !index) {
            return result;
        }

        // Create seekable iterator
        SeekableEdgeIter iter(*index, &interruption_requested);

        // Process each node with direct seeks
        for (size_t i = 0; i < sorted_nodes.size(); ++i) {
            if (interruption_requested) {
                break;
            }

            uint64_t target_node = sorted_nodes[i];

            // Seek directly to first edge from target_node
            if (!iter.seek_from(target_node)) {
                // Index exhausted - no more edges from any node
                // Remaining nodes have no outgoing edges
                break;
            }

            // Check if we actually found edges from target_node
            EdgeRecord edge = iter.current();
            if (edge.from_id != target_node) {
                // No edges from this node - continue to next
                // The iterator is positioned at the next node's edges
                // which is fine for the forward-only invariant
                continue;
            }

            // Collect all edges from target_node with reservoir sampling
            SeekReservoirState reservoir;
            if (fanout > 0) {
                reservoir.samples.reserve(fanout);
            }

            do {
                edges_iterated++;
                edge = iter.current();

                if (edge.from_id != target_node) {
                    // Moved to different source - stop collecting
                    break;
                }

                reservoir.count++;

                // Apply reservoir sampling (Algorithm R)
                if (fanout == 0 || reservoir.samples.size() < fanout) {
                    // Haven't reached fanout limit - just add
                    reservoir.samples.emplace_back(
                        ObjectId(edge.to_id),
                        ObjectId(edge.edge_id)
                    );
                    neighbors_collected++;
                } else {
                    // Reservoir sampling: replace with probability fanout/count
                    std::uniform_int_distribution<uint64_t> dist(0, reservoir.count - 1);
                    uint64_t j = dist(rng);
                    if (j < fanout) {
                        reservoir.samples[j] = {
                            ObjectId(edge.to_id),
                            ObjectId(edge.edge_id)
                        };
                    }
                }
            } while (iter.next_from_current());

            // Store result
            result.neighbors[target_node] = std::move(reservoir.samples);
        }

        // Update statistics from iterator
        seeks_performed = iter.seeks_performed();

        return result;
    }

    /**
     * @brief Sample with both indexes for undirected orientation.
     *
     * Uses ParallelCursorMerger for coordinated dual-index seeks with better
     * cache locality compared to two separate full passes.
     *
     * Algorithm:
     * 1. For each node, seek in both indexes simultaneously
     * 2. Collect and deduplicate edges from both directions
     * 3. Apply reservoir sampling to combined result
     *
     * This interleaved approach keeps both index pages hot in the buffer pool.
     */
    BatchNeighbors seek_sample_undirected(
        const std::vector<uint64_t>& sorted_nodes,
        uint64_t fanout
    ) {
        BatchNeighbors result;
        seeks_performed = 0;
        edges_iterated = 0;
        neighbors_collected = 0;

        BPlusTree<3>* from_to_index = storage.get_from_to_edge_index();
        BPlusTree<3>* to_from_index = storage.get_to_from_edge_index();

        if (sorted_nodes.empty() || !from_to_index || !to_from_index) {
            return result;
        }

        // Create parallel cursor merger for coordinated dual-index access
        ParallelCursorMerger merger(*from_to_index, *to_from_index, &interruption_requested);

        // Process each node with parallel seeks
        for (uint64_t target_node : sorted_nodes) {
            if (interruption_requested) {
                break;
            }

            // Get all edges (deduplicated) from both directions
            std::vector<MergedEdge> all_edges = merger.get_all_edges(target_node);

            if (all_edges.empty()) {
                continue;
            }

            // Apply reservoir sampling if needed
            std::vector<std::pair<ObjectId, ObjectId>> sampled;

            if (fanout == 0 || all_edges.size() <= fanout) {
                // Take all edges
                sampled.reserve(all_edges.size());
                for (const auto& edge : all_edges) {
                    sampled.emplace_back(ObjectId(edge.neighbor_id), ObjectId(edge.edge_id));
                }
            } else {
                // Reservoir sampling for combined edges
                // Use Fisher-Yates partial shuffle for efficiency
                sampled.reserve(fanout);

                // Shuffle indices for random selection
                std::vector<size_t> indices(all_edges.size());
                std::iota(indices.begin(), indices.end(), 0);

                for (size_t i = 0; i < fanout; ++i) {
                    std::uniform_int_distribution<size_t> dist(i, all_edges.size() - 1);
                    size_t j = dist(rng);
                    std::swap(indices[i], indices[j]);
                }

                for (size_t i = 0; i < fanout; ++i) {
                    const auto& edge = all_edges[indices[i]];
                    sampled.emplace_back(ObjectId(edge.neighbor_id), ObjectId(edge.edge_id));
                }
            }

            neighbors_collected += sampled.size();
            result.neighbors[target_node] = std::move(sampled);
        }

        // Update statistics from merger
        auto [seek_stats, edge_stats] = merger.get_statistics();
        seeks_performed = seek_stats;
        edges_iterated = edge_stats;

        return result;
    }
};

// =============================================================================
// Public Interface
// =============================================================================

SeekBasedGnnSampler::SeekBasedGnnSampler(GQL::ProjectionStorage& storage)
    : impl_(std::make_unique<Impl>(storage))
{}

SeekBasedGnnSampler::~SeekBasedGnnSampler() = default;

SeekBasedGnnSampler::SeekBasedGnnSampler(SeekBasedGnnSampler&&) noexcept = default;
SeekBasedGnnSampler& SeekBasedGnnSampler::operator=(SeekBasedGnnSampler&&) noexcept = default;

void SeekBasedGnnSampler::set_random_seed(uint64_t seed) {
    impl_->random_seed = seed;
    impl_->rng.seed(seed);
}

uint64_t SeekBasedGnnSampler::get_random_seed() const {
    return impl_->random_seed;
}

BatchNeighbors SeekBasedGnnSampler::sample_batch(
    const std::vector<ObjectId>& nodes,
    uint64_t fanout,
    EdgeOrientation orientation
) {
    if (nodes.empty()) {
        return BatchNeighbors{};
    }

    // Sort node IDs for forward-only seek pattern
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
            return impl_->seek_sample_directed(
                sorted_ids,
                impl_->storage.get_from_to_edge_index(),
                fanout
            );

        case EdgeOrientation::REVERSE:
            return impl_->seek_sample_directed(
                sorted_ids,
                impl_->storage.get_to_from_edge_index(),
                fanout
            );

        case EdgeOrientation::UNDIRECTED:
            return impl_->seek_sample_undirected(sorted_ids, fanout);
    }

    return BatchNeighbors{};
}

uint64_t SeekBasedGnnSampler::last_seeks_performed() const {
    return impl_->seeks_performed;
}

uint64_t SeekBasedGnnSampler::last_edges_iterated() const {
    return impl_->edges_iterated;
}

uint64_t SeekBasedGnnSampler::last_neighbors_collected() const {
    return impl_->neighbors_collected;
}

double SeekBasedGnnSampler::estimate_seek_cost(
    size_t batch_size,
    uint64_t total_edges,
    double overhead_factor
) {
    if (total_edges == 0) {
        return 0.0;
    }

    // Cost model: B × log2(E) × overhead_factor
    // - B = batch_size (number of seeks)
    // - log2(E) = B+Tree depth approximation
    // - overhead_factor accounts for:
    //   - Page I/O variance
    //   - Buffer pool misses
    //   - CPU overhead per seek
    double log_factor = std::log2(static_cast<double>(total_edges) + 1.0);
    return static_cast<double>(batch_size) * log_factor * overhead_factor;
}

} // namespace mdb::gnn
