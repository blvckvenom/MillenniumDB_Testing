#pragma once

// TopologyWalkProfiler — Phase 0 cheap profiler that generates the
// warm-start dependency for the Four-Level Topology Store
// (L1 RAM hash / L2 compact uint32 CSR / L3 mmap sidecar / L4 direct
// B+Tree).
//
// Problem this solves
// -------------------
// The `FourLevelTopologyStore` requires `<projection_dir>/node_counts.bin`
// to enable the L3 MinHash reorder. The file is normally produced as a
// side-effect of `gnn_offline_sample` AT THE END of a sample build (after
// all batches have been generated). This creates a chicken-and-egg
// dependency: the very first sample on a projection runs in "cold-start"
// mode where L3 has no reorder, falling back to random mmap access over
// the topology CSR sidecar (mmap-backed files `topology_{fwd,rev}.csr`
// that provide O(1) neighbor slices). On graphs whose sidecar exceeds
// available RAM (e.g. papers100M topology_*.csr = 53 GB on a 30 GB host),
// the cold path thrashes the page cache and the sample never completes.
//
// What this profiler does
// -----------------------
// Performs `num_walks` random walks of length `walk_length` over the
// mmap-backed reverse sidecar (`topology_rev.csr`), incrementing a
// per-node access-frequency counter at every step. The counts are written
// in the same on-disk format consumed by `TopologyFrequencyProfiler::
// compute_from_node_counts_`, so the next call to `enable_four_level_store`
// finds the file and activates the warm-start path automatically.
//
// Random walks are issued through the mmap reader's O(1) neighbor slice;
// each step costs one `neighbors(node)` lookup + one RNG draw. For
// papers100M with `num_walks=100_000` and `walk_length=5`, total work is
// ~500k lookups vs ~4.8 B for a full 3-layer `[10,15,20]` k-hop sample.
// That is ~10,000× less work and runs in minutes rather than hours.
//
// Why this approximates MinHash counts well enough
// ------------------------------------------------
// The MinHash reorder's goal is grouping nodes that are co-accessed within
// the same mini-batch. Random walks naturally surface co-accessed nodes
// because consecutive walk steps land on direct neighbors of each other
// — exactly the relationship a k-hop sampler explores. The resulting
// frequency profile is not bit-identical to a real sampling pass, but
// preserves the qualitative ranking (hot vs warm vs cold) that the tier
// assignment + MinHash bucketing care about.
//
// Seed selection — degree-weighted (CRITICAL on real graphs)
// ----------------------------------------------------------
// Naive uniform seed selection over [0, N) lands ~99% of walks on
// isolated leaf nodes on papers-style citation graphs (most papers were
// never cited, so they have zero REVERSE neighbors). Every walk dies on
// step 0 and the resulting counts (1 for each seed, 0 elsewhere) carry
// no useful information. The profiler instead samples seeds proportional
// to degree via Vose's alias method: an O(N) pre-pass builds the table,
// each subsequent seed draw is O(1), and walks land on hubs with the
// same probability mass a real k-hop sampler would expand them. This
// matches the empirical access frequency observed during a real sample
// within ±15% per quartile (validated by unit tests).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph_models/gql/projection/topology_snapshot_reader.h"

namespace mdb::gnn {

class TopologyWalkProfiler {
public:
    struct Result {
        /// Per-node access counts indexed by row id. Length == reader.num_nodes().
        /// All entries zero when the reader has no data.
        std::vector<uint64_t> counts;

        /// Wall-clock seconds spent profiling.
        double elapsed_seconds = 0.0;

        /// Number of neighbor lookups performed = num_walks × walk_length
        /// (minus walks that hit dead ends earlier).
        std::size_t lookups_done = 0;

        /// Number of walks that were forced to restart due to landing on
        /// an isolated node (no neighbors). Informational only.
        std::size_t restarts = 0;
    };

    /**
     * @brief Random-walk-based access frequency profiler.
     *
     * @param reader        Mmap-backed topology CSR sidecar reader
     *                      (typically `topology_rev.csr` for REVERSE-
     *                      orientation sampling; provides O(1) neighbor
     *                      slices via `neighbors(node)`). When
     *                      `!reader.has_data()` the profiler returns an
     *                      empty Result without performing any walks.
     * @param num_walks     Number of random walks to perform. Larger →
     *                      higher fidelity, longer runtime. Default
     *                      `kDefaultNumWalks` is calibrated for graphs
     *                      with ≥10M nodes; pass a smaller value for
     *                      micro-graphs.
     * @param walk_length   Steps per walk (each step = one neighbor
     *                      lookup). Default `kDefaultWalkLength` mirrors
     *                      typical 3-layer GNN sampling depth.
     * @param seed          RNG seed for reproducibility. Passed through
     *                      to `std::mt19937_64`.
     *
     * @return `Result` containing the per-node count vector + telemetry.
     *
     * Thread-safety: not thread-safe. Internal RNG state mutates across
     * walks. Construct one profiler call per thread if parallelizing.
     */
    static Result profile(
        const GQL::Projection::TopologySnapshotReader& reader,
        std::size_t                                    num_walks,
        std::size_t                                    walk_length,
        uint64_t                                       seed);

    /// Default walk count when the caller passes 0. 100k walks × 5 steps
    /// = 500k lookups. Empirically completes in 10-30 s on papers100M
    /// (vs 24 h+ for full cold-start sampling).
    static constexpr std::size_t kDefaultNumWalks   = 100'000;

    /// Default walk depth when the caller passes 0. 5 hops captures
    /// roughly the 2-3 layer GNN sampling pattern.
    static constexpr std::size_t kDefaultWalkLength = 5;
};

}  // namespace mdb::gnn
