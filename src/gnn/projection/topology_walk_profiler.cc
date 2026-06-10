#include "gnn/projection/topology_walk_profiler.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

#include "graph_models/object_id.h"

namespace mdb::gnn {

namespace {

// Degree alias-method table for O(1) weighted sampling.
//
// Why this matters: on real-world graphs (Friendster / Papers100M /
// MAG240M) the vast majority of nodes are leaves under the REVERSE
// orientation (papers nobody cited, users with no inbound edges, ...).
// Naive uniform seed selection lands ~99% of walks on isolated nodes,
// every walk dead-ends at step 0, and the resulting counts (`1` for
// each seed, `0` for everything else) carry zero information about the
// actual access frequency a real k-hop sampler would observe.
//
// The fix: sample seeds proportional to degree. A node with degree=100
// is 100× more likely to be picked than a degree=1 node. That matches
// the rough shape of real GNN access patterns (high-degree hubs are
// touched far more often than tail nodes during multi-hop expansion),
// and crucially avoids the "all 100k walks died on step 0" pathology.
//
// Vose's alias method gives O(1) sampling after O(N) setup. We build
// the table once at the start of `profile()` from `row_ptr` differences
// (which we compute from `reader.neighbors(i).size()` rather than
// touching the mmap'd row_ptr directly — the public reader API
// guarantees that lookup is O(1) and constant-page-fault).
class AliasTable {
public:
    explicit AliasTable(const std::vector<uint64_t>& weights) {
        const std::size_t n = weights.size();
        prob_.assign(n, 0.0);
        alias_.assign(n, 0);

        double total = 0.0;
        for (uint64_t w : weights) total += static_cast<double>(w);
        if (total <= 0.0) return;  // pathological: no neighbors anywhere

        std::vector<double> scaled(n);
        for (std::size_t i = 0; i < n; ++i) {
            scaled[i] = static_cast<double>(weights[i]) * static_cast<double>(n) / total;
        }

        std::vector<std::size_t> small, large;
        small.reserve(n);
        large.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (scaled[i] < 1.0) small.push_back(i);
            else                 large.push_back(i);
        }

        while (!small.empty() && !large.empty()) {
            const std::size_t s = small.back(); small.pop_back();
            const std::size_t l = large.back(); large.pop_back();
            prob_[s]  = scaled[s];
            alias_[s] = l;
            scaled[l] = scaled[l] + scaled[s] - 1.0;
            if (scaled[l] < 1.0) small.push_back(l);
            else                 large.push_back(l);
        }
        while (!large.empty()) {
            const std::size_t l = large.back(); large.pop_back();
            prob_[l] = 1.0;
        }
        while (!small.empty()) {
            const std::size_t s = small.back(); small.pop_back();
            prob_[s] = 1.0;
        }
        valid_ = true;
    }

    bool valid() const { return valid_; }

    std::size_t draw(std::mt19937_64& rng) const {
        std::uniform_int_distribution<std::size_t> i_dist(0, prob_.size() - 1);
        std::uniform_real_distribution<double>     u_dist(0.0, 1.0);
        const std::size_t i = i_dist(rng);
        return (u_dist(rng) < prob_[i]) ? i : alias_[i];
    }

private:
    std::vector<double>      prob_;
    std::vector<std::size_t> alias_;
    bool                     valid_ = false;
};

}  // namespace

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

    // Pre-filter eligible seeds: only nodes with degree > 0 can start a
    // useful walk. On papers-style graphs ~99% of nodes are leaves under
    // REVERSE (papers nobody cites), so without this filter the
    // alias-method draw can land on a leftover node whose alias slot
    // points to another isolated node, producing 100% restarts (observed
    // empirically as the "100k walks → 100k restarts" pathology of
    // commit 42c6970b's first attempt). Building the eligibility list
    // costs one O(N) pass over row_ptr (sequential mmap reads); after
    // it, every alias draw is guaranteed to land on a node with at
    // least one neighbor.
    //
    // Memory cost on papers100M: 2 × N × 8 B = 1.77 GB temporary. We
    // could compress to (node_id, degree) tuples with a non-zero-only
    // filter making this much smaller, but the eligibility list itself
    // is the simpler primitive and the alias table is freed before
    // the rest of the sample build starts. Free at end of profile().
    std::vector<uint64_t> eligible_nodes;
    std::vector<uint64_t> eligible_weights;
    eligible_nodes.reserve(static_cast<std::size_t>(n));
    eligible_weights.reserve(static_cast<std::size_t>(n));
    for (uint64_t i = 0; i < n; ++i) {
        // degree() is width-agnostic and never throws on the narrow (uint32)
        // layout, unlike neighbors() — see TopologySnapshotReader (Spec #6).
        const uint64_t deg = reader.degree(i);
        if (deg > 0) {
            eligible_nodes.push_back(i);
            eligible_weights.push_back(deg);
        }
    }
    AliasTable alias(eligible_weights);

    std::mt19937_64 rng(seed);

    // Fallback: if no node in this direction has any neighbor at all,
    // we cannot produce useful counts. Return what we have (all zeros)
    // — caller treats this as cold-start.
    if (eligible_nodes.empty() || !alias.valid()) {
        auto t1 = std::chrono::steady_clock::now();
        result.elapsed_seconds =
            std::chrono::duration<double>(t1 - t0).count();
        return result;
    }

    // Reused scratch for the width-agnostic copy accessor. copy_neighbors()
    // works for both id widths (memcpy for uint64; widen + re-tag for the
    // narrow uint32 layout) and returns the exact tagged ObjectIds that the
    // uint64 layout would expose — so the tag-strip below is unchanged.
    std::vector<uint64_t> walk_scratch;
    for (std::size_t w = 0; w < num_walks; ++w) {
        // alias.draw() returns an index INTO eligible_nodes — map back
        // to the global node id.
        const std::size_t e_idx = alias.draw(rng);
        uint64_t curr = eligible_nodes[e_idx];
        result.counts[static_cast<std::size_t>(curr)]++;
        result.lookups_done++;

        for (std::size_t step = 1; step < walk_length; ++step) {
            walk_scratch.clear();
            reader.copy_neighbors(curr, walk_scratch);
            if (walk_scratch.empty()) {
                // Dead end. Restart from a fresh seed (does not count
                // against `lookups_done` since we never moved).
                result.restarts++;
                break;
            }
            std::uniform_int_distribution<std::size_t> nb_dist(
                0, walk_scratch.size() - 1);
            const std::size_t picked = nb_dist(rng);
            // The sidecar stores `dst` as ObjectId values (top 8 bits
            // are the type tag set by `Node`/`Edge` encoding). Strip
            // the tag to get the dense row_idx in [0, n) that
            // row_ptr is keyed by. Verified empirically on papers100M
            // 2026-05-11 where un-stripped values surfaced as
            // 15276209936111918080 (= 0xD4 << 56 | 0x2A2EFC0), causing
            // 100% of walks to falsely restart on the out-of-range
            // defensive check.
            const uint64_t next = walk_scratch[picked] & ObjectId::VALUE_MASK;

            // Guard against malformed sidecars whose row_idx exceeds
            // the declared num_nodes. Genuine corruption only — the
            // ObjectId tag has already been masked off above.
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
