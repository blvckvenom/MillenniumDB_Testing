#include "gnn/sampling/basic_khop_sampler.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "gnn/projection/topology_accessor.h"
#include "gnn/sampling/leapfrog_gnn_sampler.h"
#include "gnn/sampling/seek_based_gnn_sampler.h"
#include "graph_models/gql/projection/projection_storage.h"

namespace mdb::gnn {

// =============================================================================
// Implementation Details
// =============================================================================

struct BasicKHopSampler::Impl {
    GQL::ProjectionStorage& storage;
    SamplingConfig config;
    std::unique_ptr<TopologyAccessor> topology;
    std::unique_ptr<LeapfrogGnnSampler> leapfrog_sampler;    ///< Sweep-based sampler
    std::unique_ptr<SeekBasedGnnSampler> seek_sampler;       ///< Seek-based sampler
    std::mt19937_64 rng;
    bool use_leapfrog = true;  ///< Enable batch optimization by default

    Impl(GQL::ProjectionStorage& storage_, const SamplingConfig& config_)
        : storage(storage_)
        , config(config_)
        , topology(std::make_unique<TopologyAccessor>(storage_))
        , leapfrog_sampler(std::make_unique<LeapfrogGnnSampler>(storage_))
        , seek_sampler(std::make_unique<SeekBasedGnnSampler>(storage_))
        , rng(config_.random_seed)
    {
        config.validate();
        // Sync random seed with all samplers
        leapfrog_sampler->set_random_seed(config_.random_seed);
        seek_sampler->set_random_seed(config_.random_seed);
    }

    /**
     * @brief Choose optimal batch strategy based on batch characteristics.
     *
     * Decision logic:
     * 1. If batch_strategy is not AUTO, use that strategy
     * 2. If batch is very small (<10 nodes), use PER_NODE
     * 3. Compare estimated costs of SWEEP vs SEEK
     *
     * Cost model:
     * - sweep_cost ≈ edges_in_range (from min_node to max_node)
     * - seek_cost ≈ batch_size × log2(total_edges) × overhead_factor
     *
     * @param sorted_nodes Sorted batch of node IDs
     * @param total_edges Total edges in projection
     * @return Selected strategy
     */
    BatchStrategy choose_strategy(
        const std::vector<ObjectId>& nodes,
        uint64_t total_edges
    ) {
        // If explicitly configured, use that strategy
        if (config.batch_strategy != BatchStrategy::AUTO) {
            return config.batch_strategy;
        }

        size_t batch_size = nodes.size();

        // Very small batches: per-node is simpler and has less overhead
        if (batch_size < 10) {
            return BatchStrategy::PER_NODE;
        }

        // Below leapfrog threshold: use per-node (existing behavior)
        if (batch_size < LEAPFROG_BATCH_THRESHOLD) {
            return BatchStrategy::PER_NODE;
        }

        // For larger batches, compare estimated costs
        if (total_edges == 0) {
            return BatchStrategy::PER_NODE;
        }

        // Estimate seek cost: B × log2(E) × overhead_factor
        double seek_cost = SeekBasedGnnSampler::estimate_seek_cost(
            batch_size,
            total_edges,
            config.seek_overhead_factor
        );

        // Estimate sweep cost: edges in the ID range
        // Heuristic: For batches >= threshold, assume sweep touches ~batch_size × avg_degree edges
        // This is a rough approximation; actual range depends on node ID distribution
        double avg_degree = static_cast<double>(total_edges) / std::max(1.0, static_cast<double>(batch_size));
        double sweep_cost = static_cast<double>(batch_size) * avg_degree;

        // Compare costs: use seek if it's cheaper
        if (seek_cost < sweep_cost) {
            return BatchStrategy::SEEK;
        }

        return BatchStrategy::SWEEP;
    }

    /**
     * @brief Sample up to `fanout` neighbors uniformly at random.
     *
     * Uses Fisher-Yates partial shuffle for efficiency when fanout < degree.
     */
    std::vector<std::pair<ObjectId, ObjectId>> sample_neighbors_uniform(
        ObjectId node_id,
        uint64_t fanout
    ) {
        Neighbors all_neighbors = topology->get_neighbors(node_id, config.orientation);

        std::vector<std::pair<ObjectId, ObjectId>> result;

        if (all_neighbors.node_ids.empty()) {
            return result;
        }

        size_t n = all_neighbors.node_ids.size();
        size_t k = std::min(static_cast<size_t>(fanout), n);

        result.reserve(k);

        if (k == n) {
            // Take all neighbors
            for (size_t i = 0; i < n; ++i) {
                result.emplace_back(all_neighbors.node_ids[i], all_neighbors.edge_ids[i]);
            }
        } else {
            // Fisher-Yates partial shuffle: sample k elements from n
            // We shuffle indices to preserve the node_id/edge_id pairing
            std::vector<size_t> indices(n);
            std::iota(indices.begin(), indices.end(), 0);

            for (size_t i = 0; i < k; ++i) {
                std::uniform_int_distribution<size_t> dist(i, n - 1);
                size_t j = dist(rng);
                std::swap(indices[i], indices[j]);
            }

            for (size_t i = 0; i < k; ++i) {
                size_t idx = indices[i];
                result.emplace_back(all_neighbors.node_ids[idx], all_neighbors.edge_ids[idx]);
            }
        }

        return result;
    }

    /**
     * @brief Build the computational graph from sampled layers.
     *
     * Creates edges_per_layer with local indices mapping.
     */
    void build_edges(
        GraphSample& sample,
        const std::vector<std::unordered_map<uint64_t, std::vector<std::pair<ObjectId, ObjectId>>>>& sampled_edges
    ) {
        size_t num_layers = sample.nodes_per_layer.size();
        sample.edges_per_layer.resize(num_layers - 1);

        // Build node_id -> local_index mapping for each layer
        std::vector<std::unordered_map<uint64_t, int32_t>> layer_mappings(num_layers);

        for (size_t layer = 0; layer < num_layers; ++layer) {
            const auto& nodes = sample.nodes_per_layer[layer];
            for (size_t i = 0; i < nodes.size(); ++i) {
                layer_mappings[layer][nodes[i].id] = static_cast<int32_t>(i);
            }
        }

        // Build edges for each layer connection
        // edges_per_layer[k] connects layer k+1 (src) to layer k (dst)
        for (size_t k = 0; k < num_layers - 1; ++k) {
            auto& edges = sample.edges_per_layer[k];

            // sampled_edges[k] maps dst_node -> [(neighbor_node, edge_id), ...]
            // dst is in layer k, src (neighbors) are in layer k+1
            for (const auto& [dst_id, neighbor_edges] : sampled_edges[k]) {
                auto dst_it = layer_mappings[k].find(dst_id);
                if (dst_it == layer_mappings[k].end()) continue;
                int32_t dst_idx = dst_it->second;

                for (const auto& [src_node, edge_id] : neighbor_edges) {
                    auto src_it = layer_mappings[k + 1].find(src_node.id);
                    if (src_it == layer_mappings[k + 1].end()) continue;
                    int32_t src_idx = src_it->second;

                    edges.src_indices.push_back(src_idx);
                    edges.dst_indices.push_back(dst_idx);
                    edges.edge_ids.push_back(edge_id);
                }
            }
        }
    }

    /**
     * @brief Main sampling algorithm.
     */
    GraphSample do_sample(
        const std::vector<ObjectId>& seeds,
        uint64_t batch_id,
        SplitType split
    ) {
        GraphSample sample;
        sample.batch_id = batch_id;
        sample.split = split;

        if (seeds.empty() || config.fanouts.empty()) {
            return sample;
        }

        size_t K = config.fanouts.size();  // Number of layers

        // nodes_per_layer[0] = seeds, nodes_per_layer[k] = k-hop neighbors
        sample.nodes_per_layer.resize(K + 1);
        sample.nodes_per_layer[0] = seeds;

        // Track which edges were sampled at each layer
        // sampled_edges[k] = {dst_node_id -> [(src_node, edge_id), ...]}
        std::vector<std::unordered_map<uint64_t, std::vector<std::pair<ObjectId, ObjectId>>>> sampled_edges(K);

        // Get total edge count for strategy selection
        uint64_t total_edges = topology->get_edge_count();

        // Sample layer by layer
        for (size_t k = 0; k < K; ++k) {
            uint64_t layer_fanout = config.fanouts[k];
            std::unordered_set<uint64_t> next_layer_set;
            const auto& current_layer = sample.nodes_per_layer[k];

            // Choose optimal strategy for this layer
            BatchStrategy strategy = use_leapfrog
                ? choose_strategy(current_layer, total_edges)
                : BatchStrategy::PER_NODE;

            BatchNeighbors batch_result;

            switch (strategy) {
                case BatchStrategy::SWEEP:
                    // Coordinated B+Tree sweep (LeapfrogGnnSampler)
                    batch_result = leapfrog_sampler->sample_batch(
                        current_layer,
                        layer_fanout,
                        config.orientation
                    );
                    break;

                case BatchStrategy::SEEK:
                    // O(log E) seeks per node (SeekBasedGnnSampler)
                    batch_result = seek_sampler->sample_batch(
                        current_layer,
                        layer_fanout,
                        config.orientation
                    );
                    break;

                case BatchStrategy::AUTO:
                case BatchStrategy::PER_NODE:
                default:
                    // Per-node sampling (original algorithm)
                    for (const ObjectId& node_id : current_layer) {
                        auto neighbors = sample_neighbors_uniform(node_id, layer_fanout);
                        if (!neighbors.empty()) {
                            batch_result.neighbors[node_id.id] = std::move(neighbors);
                        }
                    }
                    break;
            }

            // Convert batch results to sampled_edges format
            for (const ObjectId& node_id : current_layer) {
                auto it = batch_result.neighbors.find(node_id.id);
                if (it != batch_result.neighbors.end() && !it->second.empty()) {
                    sampled_edges[k][node_id.id] = it->second;

                    for (const auto& [neighbor_node, edge_id] : it->second) {
                        next_layer_set.insert(neighbor_node.id);
                    }
                }
            }

            // Convert set to vector for next layer
            sample.nodes_per_layer[k + 1].reserve(next_layer_set.size());
            for (uint64_t node_id : next_layer_set) {
                sample.nodes_per_layer[k + 1].emplace_back(node_id);
            }
        }

        // Build edge indices
        build_edges(sample, sampled_edges);

        // Build all_unique_nodes
        sample.rebuild_unique_nodes();

        return sample;
    }
};

// =============================================================================
// BasicKHopSampler Public Interface
// =============================================================================

BasicKHopSampler::BasicKHopSampler(GQL::ProjectionStorage& storage, const SamplingConfig& config)
    : impl_(std::make_unique<Impl>(storage, config))
{
}

BasicKHopSampler::~BasicKHopSampler() = default;

BasicKHopSampler::BasicKHopSampler(BasicKHopSampler&&) noexcept = default;
BasicKHopSampler& BasicKHopSampler::operator=(BasicKHopSampler&&) noexcept = default;

GraphSample BasicKHopSampler::sample(
    const std::vector<ObjectId>& seeds,
    uint64_t batch_id,
    SplitType split
) {
    return impl_->do_sample(seeds, batch_id, split);
}

GraphSample BasicKHopSampler::sample(const std::vector<ObjectId>& seeds) {
    return impl_->do_sample(seeds, 0, SplitType::TRAIN);
}

size_t BasicKHopSampler::num_layers() const {
    return impl_->config.fanouts.size();
}

uint64_t BasicKHopSampler::fanout(size_t layer) const {
    if (layer >= impl_->config.fanouts.size()) {
        return 0;
    }
    return impl_->config.fanouts[layer];
}

void BasicKHopSampler::set_random_seed(uint64_t seed) {
    impl_->rng.seed(seed);
    impl_->leapfrog_sampler->set_random_seed(seed);
}

uint64_t BasicKHopSampler::get_random_seed() const {
    return impl_->config.random_seed;
}

void BasicKHopSampler::set_use_leapfrog(bool enable) {
    impl_->use_leapfrog = enable;
}

bool BasicKHopSampler::get_use_leapfrog() const {
    return impl_->use_leapfrog;
}

} // namespace mdb::gnn
