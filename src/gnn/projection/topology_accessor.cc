#include "topology_accessor.h"

#include <algorithm>
#include <unordered_set>

#include "graph_models/gql/projection/projection_storage.h"
#include "storage/index/bplus_tree/bplus_tree.h"

#include "gnn/core/cuda_context.h"

namespace mdb::gnn {

// ============================================================================
// TopologyAccessor Implementation
// ============================================================================

struct TopologyAccessor::Impl {
    GQL::ProjectionStorage& storage;
    torch::Device target_device;
    std::mt19937_64 rng;

    explicit Impl(GQL::ProjectionStorage& storage_)
        : storage(storage_),
          target_device(CudaContext::instance().torch_device()),
          rng(std::random_device{}()) {}

    /**
     * @brief Get neighbors using from_to_edge index (outgoing).
     */
    Neighbors get_neighbors_from_index(ObjectId node_id, BPlusTree<3>* index) {
        Neighbors result;

        if (!index) {
            return result;
        }

        // Search for (node_id, MIN, MIN) to (node_id, MAX, MAX)
        Record<3> min_record = {node_id.id, 0, 0};
        Record<3> max_record = {node_id.id, UINT64_MAX, UINT64_MAX};

        bool interruption_requested = false;
        auto it = index->get_range(&interruption_requested, min_record, max_record);

        const Record<3>* record;
        while ((record = it.next()) != nullptr) {
            // from_to_edge: (from, to, edge_id)
            result.node_ids.push_back(ObjectId(std::get<1>(*record)));
            result.edge_ids.push_back(ObjectId(std::get<2>(*record)));
        }

        return result;
    }

    /**
     * @brief Sample k items from a vector uniformly.
     */
    template<typename T>
    std::vector<T> uniform_sample(const std::vector<T>& items, size_t k) {
        if (items.size() <= k) {
            return items;
        }

        std::vector<T> result;
        result.reserve(k);

        // Fisher-Yates shuffle on indices
        std::vector<size_t> indices(items.size());
        std::iota(indices.begin(), indices.end(), 0);

        for (size_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<size_t> dist(i, indices.size() - 1);
            size_t j = dist(rng);
            std::swap(indices[i], indices[j]);
            result.push_back(items[indices[i]]);
        }

        return result;
    }
};

// ============================================================================
// TopologyAccessor Public Methods
// ============================================================================

TopologyAccessor::TopologyAccessor(GQL::ProjectionStorage& storage)
    : impl_(std::make_unique<Impl>(storage)) {}

TopologyAccessor::~TopologyAccessor() = default;

TopologyAccessor::TopologyAccessor(TopologyAccessor&&) noexcept = default;
TopologyAccessor& TopologyAccessor::operator=(TopologyAccessor&&) noexcept = default;

// ----- Single Node Neighbor Access -----

Neighbors TopologyAccessor::get_out_neighbors(ObjectId node_id) {
    return impl_->get_neighbors_from_index(node_id, impl_->storage.get_from_to_edge_index());
}

Neighbors TopologyAccessor::get_in_neighbors(ObjectId node_id) {
    // to_from_edge: (to, from, edge_id)
    Neighbors result;

    auto* index = impl_->storage.get_to_from_edge_index();
    if (!index) {
        return result;
    }

    Record<3> min_record = {node_id.id, 0, 0};
    Record<3> max_record = {node_id.id, UINT64_MAX, UINT64_MAX};

    bool interruption_requested = false;
    auto it = index->get_range(&interruption_requested, min_record, max_record);

    const Record<3>* record;
    while ((record = it.next()) != nullptr) {
        // to_from_edge: (to, from, edge_id)
        result.node_ids.push_back(ObjectId(std::get<1>(*record)));
        result.edge_ids.push_back(ObjectId(std::get<2>(*record)));
    }

    return result;
}

Neighbors TopologyAccessor::get_neighbors(ObjectId node_id) {
    return get_neighbors(node_id, EdgeOrientation::UNDIRECTED);
}

Neighbors TopologyAccessor::get_neighbors(ObjectId node_id, EdgeOrientation orientation) {
    switch (orientation) {
        case EdgeOrientation::NATURAL:
            return get_out_neighbors(node_id);

        case EdgeOrientation::REVERSE:
            return get_in_neighbors(node_id);

        case EdgeOrientation::UNDIRECTED: {
            Neighbors out_neighbors = get_out_neighbors(node_id);
            Neighbors in_neighbors = get_in_neighbors(node_id);

            // Deduplicate by EDGE ID (not node ID) to handle undirected edges correctly.
            // With canonical storage (memory optimization), undirected edges are stored once
            // with (min_id, max_id, edge_id). Both indexes still provide bidirectional access:
            // - from_to_edge: (min_id, max_id, edge_id) -> found via out_neighbors for min_id
            // - to_from_edge: (max_id, min_id, edge_id) -> found via in_neighbors for min_id
            // Deduplication remains necessary to merge results from both index lookups.
            std::unordered_set<uint64_t> seen_edges;
            Neighbors result;

            for (size_t i = 0; i < out_neighbors.node_ids.size(); ++i) {
                if (seen_edges.insert(out_neighbors.edge_ids[i].id).second) {
                    result.node_ids.push_back(out_neighbors.node_ids[i]);
                    result.edge_ids.push_back(out_neighbors.edge_ids[i]);
                }
            }

            for (size_t i = 0; i < in_neighbors.node_ids.size(); ++i) {
                if (seen_edges.insert(in_neighbors.edge_ids[i].id).second) {
                    result.node_ids.push_back(in_neighbors.node_ids[i]);
                    result.edge_ids.push_back(in_neighbors.edge_ids[i]);
                }
            }

            return result;
        }
    }

    // Should never reach here, but satisfy compiler
    return Neighbors{};
}

// ----- Batch Neighbor Access -----

std::unordered_map<uint64_t, Neighbors> TopologyAccessor::get_batch_out_neighbors(
    const std::vector<ObjectId>& node_ids
) {
    std::unordered_map<uint64_t, Neighbors> result;
    result.reserve(node_ids.size());

    for (const auto& node_id : node_ids) {
        result[node_id.id] = get_out_neighbors(node_id);
    }

    return result;
}

std::unordered_map<uint64_t, Neighbors> TopologyAccessor::get_batch_in_neighbors(
    const std::vector<ObjectId>& node_ids
) {
    std::unordered_map<uint64_t, Neighbors> result;
    result.reserve(node_ids.size());

    for (const auto& node_id : node_ids) {
        result[node_id.id] = get_in_neighbors(node_id);
    }

    return result;
}

std::unordered_map<uint64_t, Neighbors> TopologyAccessor::get_batch_neighbors(
    const std::vector<ObjectId>& node_ids,
    EdgeOrientation orientation
) {
    std::unordered_map<uint64_t, Neighbors> result;
    result.reserve(node_ids.size());

    for (const auto& node_id : node_ids) {
        result[node_id.id] = get_neighbors(node_id, orientation);
    }

    return result;
}

// ----- Edge Index Construction -----

EdgeIndex TopologyAccessor::build_edge_index(const std::vector<ObjectId>& node_ids) {
    // Build node set for filtering
    std::unordered_set<uint64_t> node_set;
    std::unordered_map<uint64_t, int64_t> id_to_idx;

    for (size_t i = 0; i < node_ids.size(); ++i) {
        node_set.insert(node_ids[i].id);
        id_to_idx[node_ids[i].id] = static_cast<int64_t>(i);
    }

    // Collect edges within the node set
    std::vector<int64_t> src_indices;
    std::vector<int64_t> dst_indices;

    for (const auto& node_id : node_ids) {
        Neighbors neighbors = get_out_neighbors(node_id);

        for (const auto& neighbor_id : neighbors.node_ids) {
            if (node_set.count(neighbor_id.id)) {
                src_indices.push_back(id_to_idx[node_id.id]);
                dst_indices.push_back(id_to_idx[neighbor_id.id]);
            }
        }
    }

    // Build tensor
    int64_t num_edges = static_cast<int64_t>(src_indices.size());
    int64_t num_nodes = static_cast<int64_t>(node_ids.size());

    torch::Tensor edge_index;
    if (num_edges > 0) {
        edge_index = torch::empty({2, num_edges}, torch::kInt64);
        auto accessor = edge_index.accessor<int64_t, 2>();

        for (int64_t i = 0; i < num_edges; ++i) {
            accessor[0][i] = src_indices[i];
            accessor[1][i] = dst_indices[i];
        }

        edge_index = edge_index.to(impl_->target_device);
    } else {
        edge_index = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(impl_->target_device));
    }

    return EdgeIndex{
        std::move(edge_index),
        num_nodes,
        num_nodes
    };
}

EdgeIndex TopologyAccessor::build_bipartite_edge_index(
    const std::vector<ObjectId>& src_nodes,
    const std::vector<ObjectId>& dst_nodes
) {
    // Build mappings
    std::unordered_map<uint64_t, int64_t> src_id_to_idx;
    std::unordered_map<uint64_t, int64_t> dst_id_to_idx;

    for (size_t i = 0; i < src_nodes.size(); ++i) {
        src_id_to_idx[src_nodes[i].id] = static_cast<int64_t>(i);
    }
    for (size_t i = 0; i < dst_nodes.size(); ++i) {
        dst_id_to_idx[dst_nodes[i].id] = static_cast<int64_t>(i);
    }

    // Collect edges from src to dst
    std::vector<int64_t> src_indices;
    std::vector<int64_t> dst_indices;

    for (const auto& src_node : src_nodes) {
        Neighbors neighbors = get_out_neighbors(src_node);

        for (const auto& neighbor_id : neighbors.node_ids) {
            auto it = dst_id_to_idx.find(neighbor_id.id);
            if (it != dst_id_to_idx.end()) {
                src_indices.push_back(src_id_to_idx[src_node.id]);
                dst_indices.push_back(it->second);
            }
        }
    }

    // Build tensor
    int64_t num_edges = static_cast<int64_t>(src_indices.size());

    torch::Tensor edge_index;
    if (num_edges > 0) {
        edge_index = torch::empty({2, num_edges}, torch::kInt64);
        auto accessor = edge_index.accessor<int64_t, 2>();

        for (int64_t i = 0; i < num_edges; ++i) {
            accessor[0][i] = src_indices[i];
            accessor[1][i] = dst_indices[i];
        }

        edge_index = edge_index.to(impl_->target_device);
    } else {
        edge_index = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(impl_->target_device));
    }

    return EdgeIndex{
        std::move(edge_index),
        static_cast<int64_t>(src_nodes.size()),
        static_cast<int64_t>(dst_nodes.size())
    };
}

// ----- Neighbor Sampling -----

SampledSubgraph TopologyAccessor::sample_neighbors(
    const std::vector<ObjectId>& seed_nodes,
    int64_t fanout,
    SamplingStrategy strategy,
    EdgeOrientation orientation
) {
    SampledSubgraph result;
    result.dst_nodes = seed_nodes;

    // Build dst mapping
    for (size_t i = 0; i < seed_nodes.size(); ++i) {
        result.dst_id_to_idx[seed_nodes[i].id] = static_cast<int64_t>(i);
    }

    // Single pass: collect neighbors, sample, and store selected IDs for edge building.
    // Uses one RNG source (impl_->rng) to guarantee src_set and edges are consistent.
    std::unordered_set<uint64_t> src_set;
    std::vector<std::vector<uint64_t>> per_dst_selected_ids;
    per_dst_selected_ids.reserve(seed_nodes.size());

    for (size_t dst_idx = 0; dst_idx < seed_nodes.size(); ++dst_idx) {
        Neighbors neighbors = get_neighbors(seed_nodes[dst_idx], orientation);

        // Sample if needed (Fisher-Yates partial shuffle)
        std::vector<size_t> selected_indices;
        if (fanout > 0 && strategy == SamplingStrategy::UNIFORM &&
            static_cast<int64_t>(neighbors.node_ids.size()) > fanout) {

            std::vector<size_t> indices(neighbors.node_ids.size());
            std::iota(indices.begin(), indices.end(), 0);

            for (int64_t i = 0; i < fanout; ++i) {
                std::uniform_int_distribution<size_t> dist(i, indices.size() - 1);
                size_t j = dist(impl_->rng);
                std::swap(indices[i], indices[j]);
            }
            selected_indices.assign(indices.begin(), indices.begin() + fanout);
        } else {
            selected_indices.resize(neighbors.node_ids.size());
            std::iota(selected_indices.begin(), selected_indices.end(), 0);
        }

        // Collect selected neighbor IDs for both src_set and deferred edge building
        std::vector<uint64_t> selected_ids;
        selected_ids.reserve(selected_indices.size());
        for (size_t idx : selected_indices) {
            uint64_t neighbor_id = neighbors.node_ids[idx].id;
            src_set.insert(neighbor_id);
            selected_ids.push_back(neighbor_id);
        }
        per_dst_selected_ids.push_back(std::move(selected_ids));
    }

    // Build src nodes list and mapping
    result.src_nodes.reserve(src_set.size());
    for (uint64_t id : src_set) {
        result.src_id_to_idx[id] = static_cast<int64_t>(result.src_nodes.size());
        result.src_nodes.push_back(ObjectId(id));
    }

    // Build edges from stored selections (consistent with src_set)
    std::vector<int64_t> src_indices;
    std::vector<int64_t> dst_indices;

    for (size_t dst_idx = 0; dst_idx < per_dst_selected_ids.size(); ++dst_idx) {
        for (uint64_t neighbor_id : per_dst_selected_ids[dst_idx]) {
            src_indices.push_back(result.src_id_to_idx.at(neighbor_id));
            dst_indices.push_back(static_cast<int64_t>(dst_idx));
        }
    }

    // Build edge tensor
    int64_t num_edges = static_cast<int64_t>(src_indices.size());

    torch::Tensor edge_index;
    if (num_edges > 0) {
        edge_index = torch::empty({2, num_edges}, torch::kInt64);
        auto accessor = edge_index.accessor<int64_t, 2>();

        for (int64_t i = 0; i < num_edges; ++i) {
            accessor[0][i] = src_indices[i];
            accessor[1][i] = dst_indices[i];
        }

        edge_index = edge_index.to(impl_->target_device);
    } else {
        edge_index = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(impl_->target_device));
    }

    result.edge_index = EdgeIndex{
        std::move(edge_index),
        static_cast<int64_t>(result.src_nodes.size()),
        static_cast<int64_t>(result.dst_nodes.size())
    };

    return result;
}

SampledSubgraph TopologyAccessor::sample_in_neighbors(
    const std::vector<ObjectId>& seed_nodes,
    int64_t fanout,
    SamplingStrategy strategy
) {
    // Legacy method: use REVERSE orientation (incoming edges)
    return sample_neighbors(seed_nodes, fanout, strategy, EdgeOrientation::REVERSE);
}

std::vector<SampledSubgraph> TopologyAccessor::sample_khop_neighbors(
    const std::vector<ObjectId>& seed_nodes,
    const std::vector<int64_t>& fanouts,
    SamplingStrategy strategy,
    EdgeOrientation orientation
) {
    std::vector<SampledSubgraph> layers;
    layers.reserve(fanouts.size());

    std::vector<ObjectId> current_seeds = seed_nodes;

    for (int64_t fanout : fanouts) {
        SampledSubgraph layer = sample_neighbors(current_seeds, fanout, strategy, orientation);
        current_seeds = layer.src_nodes;  // Next layer's seeds are this layer's sources
        layers.push_back(std::move(layer));
    }

    return layers;
}

// ----- Statistics -----

int64_t TopologyAccessor::get_out_degree(ObjectId node_id) {
    Neighbors neighbors = get_out_neighbors(node_id);
    return static_cast<int64_t>(neighbors.node_ids.size());
}

int64_t TopologyAccessor::get_in_degree(ObjectId node_id) {
    Neighbors neighbors = get_in_neighbors(node_id);
    return static_cast<int64_t>(neighbors.node_ids.size());
}

uint64_t TopologyAccessor::get_edge_count() const {
    return impl_->storage.get_edge_count();
}

uint64_t TopologyAccessor::get_node_count() const {
    return impl_->storage.get_node_count();
}

// ----- Configuration -----

void TopologyAccessor::set_random_seed(uint64_t seed) {
    impl_->rng.seed(seed);
}

void TopologyAccessor::set_target_device(torch::Device device) {
    impl_->target_device = device;
}

// ============================================================================
// NodeIterator Implementation
// ============================================================================

struct NodeIterator::Impl {
    GQL::ProjectionStorage& storage;
    BPlusTree<1>* nodes_index;
    BptIter<1> iterator;
    bool interruption_requested;
    bool exhausted;
    uint64_t iterated;
    uint64_t total;

    explicit Impl(GQL::ProjectionStorage& storage_)
        : storage(storage_),
          nodes_index(storage_.get_nodes_index()),
          interruption_requested(false),
          exhausted(false),
          iterated(0),
          total(storage_.get_node_count()) {

        // Initialize iterator to scan all nodes
        if (nodes_index) {
            Record<1> min_record = {0};
            Record<1> max_record = {UINT64_MAX};
            iterator = nodes_index->get_range(&interruption_requested, min_record, max_record);
        } else {
            exhausted = true;
        }
    }

    void reset_iterator() {
        if (nodes_index) {
            Record<1> min_record = {0};
            Record<1> max_record = {UINT64_MAX};
            interruption_requested = false;
            iterator = nodes_index->get_range(&interruption_requested, min_record, max_record);
            exhausted = false;
            iterated = 0;
        }
    }
};

NodeIterator::NodeIterator(GQL::ProjectionStorage& storage)
    : impl_(std::make_unique<Impl>(storage)) {}

NodeIterator::~NodeIterator() = default;

NodeIterator::NodeIterator(NodeIterator&&) noexcept = default;
NodeIterator& NodeIterator::operator=(NodeIterator&&) noexcept = default;

std::optional<ObjectId> NodeIterator::next() {
    if (impl_->exhausted) {
        return std::nullopt;
    }

    const Record<1>* record = impl_->iterator.next();
    if (record == nullptr) {
        impl_->exhausted = true;
        return std::nullopt;
    }

    impl_->iterated++;
    return ObjectId(std::get<0>(*record));
}

std::optional<std::vector<ObjectId>> NodeIterator::next_batch(size_t batch_size) {
    if (impl_->exhausted) {
        return std::nullopt;
    }

    std::vector<ObjectId> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        const Record<1>* record = impl_->iterator.next();
        if (record == nullptr) {
            impl_->exhausted = true;
            break;
        }
        batch.push_back(ObjectId(std::get<0>(*record)));
        impl_->iterated++;
    }

    if (batch.empty()) {
        return std::nullopt;
    }

    return batch;
}

void NodeIterator::reset() {
    impl_->reset_iterator();
}

bool NodeIterator::has_next() const {
    return !impl_->exhausted;
}

uint64_t NodeIterator::total_count() const {
    return impl_->total;
}

uint64_t NodeIterator::iterated_count() const {
    return impl_->iterated;
}

} // namespace mdb::gnn
