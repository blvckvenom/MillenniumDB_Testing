#include "topology_frequency_profiler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <utility>
#include <vector>

namespace mdb::gnn {

namespace {

inline std::size_t l1_cost_per_node(double avg_degree) {
    if (avg_degree < 0.0) avg_degree = 0.0;
    return static_cast<std::size_t>(avg_degree * kL1PerEdgeBytes) + kL1NodeFixedOverhead;
}

inline std::size_t l2_cost_per_node(double avg_degree) {
    if (avg_degree < 0.0) avg_degree = 0.0;
    return static_cast<std::size_t>(avg_degree * kL2PerEdgeBytes) + kL2NodeFixedOverhead;
}

}  // namespace

TopologyFrequencyProfiler::TopologyFrequencyProfiler(
    TopologyAccessor& topo,
    std::filesystem::path projection_dir)
    : topo_(topo),
      projection_dir_(std::move(projection_dir))
{}

void TopologyFrequencyProfiler::compute(EdgeOrientation direction) {
    frequency_.clear();
    warm_start_used_ = false;

    if (compute_from_node_counts_(direction)) {
        warm_start_used_ = true;
        return;
    }
    compute_from_degrees_(direction);
}

bool TopologyFrequencyProfiler::compute_from_node_counts_(EdgeOrientation /*direction*/) {
    // TODO Spec #13 Phase 2: implement node_counts.bin reader once
    // gnn_offline_sample writer lands. `projection_dir_` is the lookup
    // root for the persisted counts file; it is stored on the profiler
    // for the future reader to consume.
    return false;
}

void TopologyFrequencyProfiler::compute_from_degrees_(EdgeOrientation direction) {
    const uint64_t n = topo_.get_node_count();
    frequency_.clear();
    frequency_.reserve(static_cast<std::size_t>(n));

    // Phase 1 cold-start path: query degree for each row index in [0..n).
    // The unit-test fixtures (and the projection import pipeline at
    // present) assign sequential ObjectIds 0..N-1 to projected nodes, so
    // the resulting frequency vector aligns with the projection's node
    // B+Tree iteration order. Phase 2 (`compute_from_node_counts_`) will
    // align by RowMapping when warm-start counts persist.
    for (uint64_t i = 0; i < n; ++i) {
        const ObjectId node_id(i);
        uint64_t freq = 0;
        switch (direction) {
            case EdgeOrientation::NATURAL:
                freq = static_cast<uint64_t>(topo_.get_out_degree(node_id));
                break;
            case EdgeOrientation::REVERSE:
                freq = static_cast<uint64_t>(topo_.get_in_degree(node_id));
                break;
            case EdgeOrientation::UNDIRECTED:
                freq = static_cast<uint64_t>(topo_.get_out_degree(node_id))
                     + static_cast<uint64_t>(topo_.get_in_degree(node_id));
                break;
        }
        frequency_.push_back(freq);
    }
}

std::vector<uint8_t> compute_tier_assignment(
    const std::vector<uint64_t>& frequency,
    std::size_t l1_budget_bytes,
    std::size_t l2_budget_bytes,
    double avg_degree)
{
    const std::size_t n = frequency.size();
    std::vector<uint8_t> tiers(n, 3);  // default: tier 3 (L3 / L4)
    if (n == 0) return tiers;

    // Sort indexes by frequency descending. Ties are broken by index order
    // so the assignment is deterministic across runs.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
        [&frequency](std::size_t a, std::size_t b) {
            if (frequency[a] != frequency[b]) {
                return frequency[a] > frequency[b];
            }
            return a < b;
        });

    const std::size_t l1_per_node = l1_cost_per_node(avg_degree);
    const std::size_t l2_per_node = l2_cost_per_node(avg_degree);

    std::size_t used_l1 = 0;
    std::size_t used_l2 = 0;
    std::size_t cursor  = 0;

    // Pack L1.
    while (cursor < n) {
        if (l1_per_node == 0 || used_l1 + l1_per_node > l1_budget_bytes) break;
        tiers[order[cursor]] = 1;
        used_l1 += l1_per_node;
        ++cursor;
    }

    // Pack L2.
    while (cursor < n) {
        if (l2_per_node == 0 || used_l2 + l2_per_node > l2_budget_bytes) break;
        tiers[order[cursor]] = 2;
        used_l2 += l2_per_node;
        ++cursor;
    }

    // Remainder stays at tier 3 (already set above).
    return tiers;
}

} // namespace mdb::gnn
