#include "gnn/projection/topology_walk_profiler.h"

#include <algorithm>
#include <chrono>
#include <random>

namespace mdb::gnn {

TopologyWalkProfiler::Result TopologyWalkProfiler::profile(
    const GQL::Projection::TopologySnapshotReader& reader,
    std::size_t                                    num_walks,
    std::size_t                                    walk_length,
    uint64_t                                       seed)
{
    Result result;

    if (!reader.has_data()) {
        // No sidecar to profile — return empty counts; caller will fall
        // back to the legacy cold path (degree proxy).
        return result;
    }

    const uint64_t n = reader.num_nodes();
    result.counts.assign(static_cast<std::size_t>(n), 0);
    if (n == 0) return result;

    if (num_walks   == 0) num_walks   = kDefaultNumWalks;
    if (walk_length == 0) walk_length = kDefaultWalkLength;

    auto t0 = std::chrono::steady_clock::now();

    std::mt19937_64 rng(seed);

    // Seed selection: uniform over [0, n). For walks that hit an
    // isolated node (no neighbors), we abandon the remainder of the
    // walk and start a fresh one from a new uniform seed. This biases
    // counts slightly toward non-isolated regions, which is exactly
    // what tier assignment wants (isolated nodes don't need L1/L2
    // residency anyway).
    std::uniform_int_distribution<uint64_t> seed_dist(0, n - 1);

    for (std::size_t w = 0; w < num_walks; ++w) {
        uint64_t curr = seed_dist(rng);
        result.counts[static_cast<std::size_t>(curr)]++;
        result.lookups_done++;

        for (std::size_t step = 1; step < walk_length; ++step) {
            auto neighbors = reader.neighbors(curr);
            if (neighbors.empty()) {
                // Dead end. Restart from a fresh seed (does not count
                // against `lookups_done` since we never moved).
                result.restarts++;
                break;
            }
            std::uniform_int_distribution<std::size_t> nb_dist(
                0, neighbors.size() - 1);
            const uint64_t next = neighbors[nb_dist(rng)];

            // Guard against malformed sidecars that contain
            // out-of-range neighbor ids (defensive — the writer
            // validates this, but we never trust mmap data blindly).
            if (next >= n) {
                result.restarts++;
                break;
            }
            curr = next;
            result.counts[static_cast<std::size_t>(curr)]++;
            result.lookups_done++;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_seconds =
        std::chrono::duration<double>(t1 - t0).count();

    return result;
}

}  // namespace mdb::gnn
