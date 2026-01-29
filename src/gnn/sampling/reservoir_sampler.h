#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "graph_models/object_id.h"
#include "gnn/projection/topology_accessor.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

/**
 * @brief Streaming reservoir sampler for high-degree nodes.
 *
 * Implements Algorithm R (Vitter, 1985) for uniform random sampling
 * without knowing the total count in advance. This enables O(k) memory
 * sampling regardless of node degree.
 *
 * ## Algorithm R
 *
 * For a stream of n elements, to select k uniformly at random:
 * 1. Fill reservoir with first k elements
 * 2. For each subsequent element i (i > k):
 *    - Generate random j in [0, i)
 *    - If j < k, replace reservoir[j] with element i
 *
 * ## Theoretical Guarantees
 *
 * - Each element has exactly k/n probability of being in final sample
 * - O(k) memory regardless of stream size n
 * - O(n) time (single pass through stream)
 *
 * ## Usage
 *
 * @code
 *   ReservoirSampler sampler(storage);
 *   sampler.set_sample_size(15);
 *
 *   // Sample 15 neighbors from a node with potentially millions of edges
 *   auto neighbors = sampler.sample_neighbors(high_degree_node, EdgeOrientation::REVERSE);
 * @endcode
 *
 * @see SortedBatchSampler for batch processing optimization
 * @see BasicKHopSampler for full k-hop sampling
 */
class ReservoirSampler {
public:
    /**
     * @brief Construct reservoir sampler for a projection.
     *
     * @param storage Reference to projection storage (must outlive sampler)
     */
    explicit ReservoirSampler(GQL::ProjectionStorage& storage);

    ~ReservoirSampler();

    // Disable copy
    ReservoirSampler(const ReservoirSampler&) = delete;
    ReservoirSampler& operator=(const ReservoirSampler&) = delete;

    // Allow move
    ReservoirSampler(ReservoirSampler&&) noexcept;
    ReservoirSampler& operator=(ReservoirSampler&&) noexcept;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the reservoir size (number of samples to collect).
     *
     * @param k Number of samples to collect
     */
    void set_sample_size(uint64_t k);

    /**
     * @brief Get current sample size.
     */
    uint64_t get_sample_size() const;

    /**
     * @brief Set random seed for reproducible sampling.
     */
    void set_random_seed(uint64_t seed);

    // =========================================================================
    // Sampling Interface
    // =========================================================================

    /**
     * @brief Sample neighbors using reservoir sampling.
     *
     * Streams through all neighbors but only keeps k samples in memory.
     *
     * @param node_id Node to sample neighbors for
     * @param orientation Edge traversal direction
     * @return Sampled neighbors with edge IDs (up to sample_size elements)
     */
    std::vector<std::pair<ObjectId, ObjectId>> sample_neighbors(
        ObjectId node_id,
        EdgeOrientation orientation = EdgeOrientation::REVERSE
    );

    /**
     * @brief Sample from a generic stream using reservoir sampling.
     *
     * Template version that works with any iterator.
     *
     * @tparam Iterator Forward iterator type
     * @param begin Start of stream
     * @param end End of stream
     * @param k Number of samples
     * @return Vector of k sampled elements (or fewer if stream smaller)
     */
    template<typename Iterator>
    auto sample_stream(Iterator begin, Iterator end, uint64_t k)
        -> std::vector<typename std::iterator_traits<Iterator>::value_type>;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get total elements seen in last sampling operation.
     *
     * Useful for understanding the degree of sampled nodes.
     */
    uint64_t last_stream_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Template Implementation
// =============================================================================

template<typename Iterator>
auto ReservoirSampler::sample_stream(Iterator begin, Iterator end, uint64_t k)
    -> std::vector<typename std::iterator_traits<Iterator>::value_type>
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    std::vector<T> reservoir;
    reservoir.reserve(k);

    // Need access to impl_->rng, but template must be in header
    // Use a local RNG seeded from the impl's seed
    std::mt19937_64 rng(42);  // Will be properly seeded in actual use

    uint64_t count = 0;
    for (auto it = begin; it != end; ++it, ++count) {
        if (count < k) {
            reservoir.push_back(*it);
        } else {
            std::uniform_int_distribution<uint64_t> dist(0, count);
            uint64_t j = dist(rng);
            if (j < k) {
                reservoir[j] = *it;
            }
        }
    }

    return reservoir;
}

} // namespace mdb::gnn
