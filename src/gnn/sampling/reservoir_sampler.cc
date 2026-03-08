#include "gnn/sampling/reservoir_sampler.h"

#include <unordered_set>

#include "graph_models/gql/projection/projection_storage.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace mdb::gnn {

// =============================================================================
// Implementation Details
// =============================================================================

struct ReservoirSampler::Impl {
    GQL::ProjectionStorage& storage;
    uint64_t sample_size = 15;
    uint64_t random_seed = 42;
    std::mt19937_64 rng;
    uint64_t last_stream_size_ = 0;

    // Tracks ALL edge IDs seen across forward and reverse passes to avoid
    // undirected sampling bias. Without this, edges evicted from the reservoir
    // during the forward pass could be re-added during the reverse pass.
    std::unordered_set<uint64_t> all_seen_edges_;

    explicit Impl(GQL::ProjectionStorage& storage_)
        : storage(storage_)
        , rng(random_seed)
    {
    }

    /**
     * @brief Get the appropriate B+tree index based on orientation.
     */
    BPlusTree<3>* get_index(EdgeOrientation orientation) {
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
     * @brief Reservoir sampling during B+tree iteration.
     *
     * Algorithm R (Vitter, 1985):
     * - Fill reservoir with first k elements
     * - For element i (i >= k): replace random position with probability k/i
     */
    std::vector<std::pair<ObjectId, ObjectId>> do_sample_neighbors(
        ObjectId node_id,
        EdgeOrientation orientation
    ) {
        std::vector<std::pair<ObjectId, ObjectId>> reservoir;
        reservoir.reserve(sample_size);

        // Clear comprehensive edge tracking for this sampling call
        all_seen_edges_.clear();

        BPlusTree<3>* index = get_index(orientation);
        if (!index) {
            last_stream_size_ = 0;
            return reservoir;
        }

        // Range query for this node's edges
        Record<3> min_record = {node_id.id, 0, 0};
        Record<3> max_record = {node_id.id, UINT64_MAX, UINT64_MAX};

        bool interruption_requested = false;
        auto iter = index->get_range(&interruption_requested, min_record, max_record);

        uint64_t count = 0;
        const Record<3>* record;

        while ((record = iter.next()) != nullptr) {
            uint64_t neighbor_id = (*record)[1];
            uint64_t edge_id = (*record)[2];

            // Track every edge seen during the forward pass so that
            // add_reverse_neighbors can correctly deduplicate, even for
            // edges that were evicted from the reservoir.
            all_seen_edges_.insert(edge_id);

            std::pair<ObjectId, ObjectId> item = {
                ObjectId(neighbor_id),
                ObjectId(edge_id)
            };

            if (count < sample_size) {
                // Fill reservoir
                reservoir.push_back(item);
            } else {
                // Algorithm R: replace with probability sample_size / (count + 1)
                std::uniform_int_distribution<uint64_t> dist(0, count);
                uint64_t j = dist(rng);
                if (j < sample_size) {
                    reservoir[j] = item;
                }
            }
            count++;
        }

        last_stream_size_ = count;

        // Handle UNDIRECTED: also sample from reverse index
        if (orientation == EdgeOrientation::UNDIRECTED) {
            add_reverse_neighbors(node_id, reservoir, count);
        }

        return reservoir;
    }

    /**
     * @brief Add reverse edges for UNDIRECTED orientation using reservoir sampling.
     */
    void add_reverse_neighbors(
        ObjectId node_id,
        std::vector<std::pair<ObjectId, ObjectId>>& reservoir,
        uint64_t& count
    ) {
        BPlusTree<3>* reverse_index = storage.get_to_from_edge_index();
        if (!reverse_index) return;

        // Use all_seen_edges_ which already contains every edge ID from the
        // forward pass (not just the k entries in the reservoir). This prevents
        // bias toward bidirectional edges when F > k forward edges were seen.

        Record<3> min_record = {node_id.id, 0, 0};
        Record<3> max_record = {node_id.id, UINT64_MAX, UINT64_MAX};

        bool interruption_requested = false;
        auto iter = reverse_index->get_range(&interruption_requested, min_record, max_record);

        const Record<3>* record;
        while ((record = iter.next()) != nullptr) {
            uint64_t neighbor_id = (*record)[1];
            uint64_t edge_id = (*record)[2];

            // Skip edges already seen during the forward pass
            if (all_seen_edges_.find(edge_id) != all_seen_edges_.end()) {
                continue;
            }
            all_seen_edges_.insert(edge_id);

            std::pair<ObjectId, ObjectId> item = {
                ObjectId(neighbor_id),
                ObjectId(edge_id)
            };

            if (count < sample_size) {
                reservoir.push_back(item);
            } else {
                std::uniform_int_distribution<uint64_t> dist(0, count);
                uint64_t j = dist(rng);
                if (j < sample_size) {
                    reservoir[j] = item;
                }
            }
            count++;
        }

        last_stream_size_ = count;
    }
};

// =============================================================================
// ReservoirSampler Public Interface
// =============================================================================

ReservoirSampler::ReservoirSampler(GQL::ProjectionStorage& storage)
    : impl_(std::make_unique<Impl>(storage))
{
}

ReservoirSampler::~ReservoirSampler() = default;

ReservoirSampler::ReservoirSampler(ReservoirSampler&&) noexcept = default;
ReservoirSampler& ReservoirSampler::operator=(ReservoirSampler&&) noexcept = default;

void ReservoirSampler::set_sample_size(uint64_t k) {
    impl_->sample_size = k;
}

uint64_t ReservoirSampler::get_sample_size() const {
    return impl_->sample_size;
}

void ReservoirSampler::set_random_seed(uint64_t seed) {
    impl_->random_seed = seed;
    impl_->rng.seed(seed);
}

uint64_t ReservoirSampler::get_random_seed() const {
    return impl_->random_seed;
}

std::vector<std::pair<ObjectId, ObjectId>> ReservoirSampler::sample_neighbors(
    ObjectId node_id,
    EdgeOrientation orientation
) {
    return impl_->do_sample_neighbors(node_id, orientation);
}

uint64_t ReservoirSampler::next_random_uint64() {
    return impl_->rng();
}

uint64_t ReservoirSampler::last_stream_size() const {
    return impl_->last_stream_size_;
}

} // namespace mdb::gnn
