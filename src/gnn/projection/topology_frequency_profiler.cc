#include "topology_frequency_profiler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace mdb::gnn {

namespace {

// 8-byte magic for `node_counts.bin`. Version 0; little-endian only (the
// project targets x86-64 and the format mirrors the host byte order at
// write time, mirroring the GnnMeta convention in src/gnn/projection/
// gnn_meta.h which also writes uint64s native and assumes little-endian).
constexpr uint8_t  kNodeCountsMagic[8] = {'N','O','D','E','C','N','T','0'};
constexpr std::size_t kNodeCountsHeaderBytes = 8 /*magic*/ + 8 /*num_nodes*/
                                             + 8 /*direction_bitmask*/;

}  // namespace

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

bool TopologyFrequencyProfiler::compute_from_node_counts_(EdgeOrientation direction) {
    // Warm-start reader for the Four-Level Topology Store (L1 RAM hash /
    // L2 compact uint32 CSR / L3 mmap sidecar / L4 direct B+Tree): reads
    // `<projection_dir>/node_counts.bin` if present. Format:
    //
    //   [8B magic "NODECNT0"]
    //   [uint64_t num_nodes]
    //   [uint64_t direction_bitmask]   (1=NATURAL, 2=REVERSE, 3=UNDIRECTED)
    //   [num_nodes × uint64_t counts]
    //
    // All multibyte fields are written in host byte order (little-endian
    // on x86-64; the project target). The reader is fail-safe: any I/O
    // or validation error returns false so the caller falls back to the
    // degree-proxy cold path. Logging is suppressed for the cold case
    // (file absent) since that is the expected first-run path; warnings
    // are emitted for malformed / stale files only.
    if (projection_dir_.empty()) return false;
    std::filesystem::path path = projection_dir_ / "node_counts.bin";
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return false;  // expected on first run — cold start.
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[TopologyFrequencyProfiler] WARNING: cannot open "
                  << path.string() << " for warm-start read; falling "
                  << "back to degree proxy.\n";
        return false;
    }

    uint8_t magic[8] = {};
    if (!f.read(reinterpret_cast<char*>(magic), 8) ||
        std::memcmp(magic, kNodeCountsMagic, 8) != 0)
    {
        std::cerr << "[TopologyFrequencyProfiler] WARNING: "
                  << path.string() << " has invalid magic; ignoring "
                  << "(treating as cold start).\n";
        return false;
    }

    uint64_t num_nodes = 0;
    uint64_t direction_bitmask = 0;
    if (!f.read(reinterpret_cast<char*>(&num_nodes),         sizeof(num_nodes)) ||
        !f.read(reinterpret_cast<char*>(&direction_bitmask), sizeof(direction_bitmask)))
    {
        std::cerr << "[TopologyFrequencyProfiler] WARNING: truncated "
                  << "header in " << path.string()
                  << "; treating as cold start.\n";
        return false;
    }

    const uint64_t expected_n = topo_.get_node_count();
    if (num_nodes != expected_n) {
        std::cerr << "[TopologyFrequencyProfiler] WARNING: "
                  << path.string() << " num_nodes=" << num_nodes
                  << " mismatches projection num_nodes="
                  << expected_n << " (stale file from a previous "
                  << "projection?); treating as cold start.\n";
        return false;
    }

    // direction_bitmask is informational. Counts gathered under any
    // direction (typically UNDIRECTED, the bitmask=3 case) work as a
    // popularity proxy for any subsequent profiler direction request:
    // a node frequently visited by sampling is a good L1/L2 candidate
    // regardless of which direction the future build will populate. Log
    // a one-line note when the bitmask doesn't include the requested
    // direction, but proceed.
    uint64_t requested = 0;
    switch (direction) {
        case EdgeOrientation::NATURAL:    requested = 1; break;
        case EdgeOrientation::REVERSE:    requested = 2; break;
        case EdgeOrientation::UNDIRECTED: requested = 3; break;
    }
    if ((direction_bitmask & requested) != requested) {
        std::cerr << "[TopologyFrequencyProfiler] note: "
                  << path.string() << " direction_bitmask="
                  << direction_bitmask << " does not include requested "
                  << "direction=" << requested << "; using counts "
                  << "anyway (popularity is direction-agnostic).\n";
    }

    frequency_.assign(static_cast<std::size_t>(num_nodes), 0);
    if (num_nodes > 0) {
        if (!f.read(reinterpret_cast<char*>(frequency_.data()),
                    static_cast<std::streamsize>(num_nodes * sizeof(uint64_t))))
        {
            std::cerr << "[TopologyFrequencyProfiler] WARNING: truncated "
                      << "counts payload in " << path.string()
                      << "; treating as cold start.\n";
            frequency_.clear();
            return false;
        }
    }

    return true;
}

void TopologyFrequencyProfiler::compute_from_degrees_(EdgeOrientation direction) {
    const uint64_t n = topo_.get_node_count();

    // Cold-start path: query degree for each dense row ordinal in [0..n).
    //
    // The projection's edge B+Trees key records by the FULL ObjectId — the
    // 8-bit type tag plus the 56-bit value. Production projected nodes carry
    // `ObjectId::MASK_NODE` (0xD4'00...) and enumerate row ordinals 0..N-1
    // in the value bits (`row_idx == ObjectId.id & VALUE_MASK`, the same
    // invariant `FourLevelTopologyStore::row_lookup_` and the mmap-backed
    // topology CSR sidecar ROW_PTR layout (topology_{fwd,rev}.csr) rely on). Ranging with the bare ordinal
    // `ObjectId(i)` matches no tagged key, so every node would profile as
    // degree 0 — collapsing `avg_degree` to 0 and letting the L1/L2 budget
    // account only the fixed per-node overhead against a 56 + 16*deg
    // reality. Query with the tagged id first; if the whole pass comes back
    // empty while the projection does contain edges, retry with the raw
    // ordinal so synthetic builds whose nodes were stored untagged keep
    // profiling correctly. The resulting vector stays indexed by row
    // ordinal in both cases.
    constexpr uint64_t id_tags[] = { ObjectId::MASK_NODE, 0 };
    for (const uint64_t tag : id_tags) {
        frequency_.assign(static_cast<std::size_t>(n), 0);
        bool any_degree = false;
        for (uint64_t i = 0; i < n; ++i) {
            const ObjectId node_id(tag | i);
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
            frequency_[static_cast<std::size_t>(i)] = freq;
            any_degree = any_degree || (freq != 0);
        }
        if (any_degree || topo_.get_edge_count() == 0) {
            return;
        }
    }

    std::cerr << "[TopologyFrequencyProfiler] WARNING: degree profile is "
              << "all-zero despite edge_count=" << topo_.get_edge_count()
              << "; tier assignment will run blind (per-node cost may be "
              << "underestimated).\n";
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
