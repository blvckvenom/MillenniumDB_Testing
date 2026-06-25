#include "gnn/projection/symmetric_snapshot_builder.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "gnn/projection/edge_orientation.h"
#include "gnn/projection/topology_accessor.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "graph_models/gql/projection/topology_snapshot_writer.h"
#include "graph_models/gql/projection/topology_symmetric_merge.h"
#include "graph_models/object_id.h"

namespace fs = std::filesystem;

namespace mdb::gnn {

namespace {

// FAST PATH — parallel concat bake straight from the two directional narrow CSR
// sidecars. Eligible when both topology_{fwd,rev}.csr are present, narrow
// (id_width==4) and carry edge_ids: then the undirected merge is the edge_id-
// keyed dedup over DISTINCT ids, which removes nothing, i.e. a pure concat
// (out(u) ++ in(u)) — byte-identical to the accessor concat path. Concat needs
// no per-node hash set, so it parallelizes trivially across node ranges and
// reads raw col_idx32 rows (no per-node Neighbors object), exploiting every
// core + the NVMe instead of the single-threaded accessor 2-pass. Reading the
// mmap'd sidecars touches no B+Tree, so no QueryContext propagation is needed.
uint64_t build_symmetric_concat_parallel_(
    const fs::path&                                proj_dir,
    uint64_t                                       N,
    GQL::Projection::TopologySnapshotReader&       rf,
    GQL::Projection::TopologySnapshotReader&       rr,
    bool                                           dedup_by_node) {
    using GQL::Projection::TopologySnapshotWriter;

    // Per-row undirected merge out(u) ++ in(u). On a SIMPLE graph
    // (dedup_by_node) a u<->v mutual pair is ONE undirected neighbor, so dedup
    // by node id; on a multigraph keep every entry (pure concat) so real
    // parallel edges survive. The dedup set is per-row (O(degree)), so the bake
    // stays out-of-core and parallel across disjoint node ranges.
    auto merge_row = [dedup_by_node](const uint32_t* f, uint64_t fd,
                                     const uint32_t* r, uint64_t rd,
                                     std::vector<uint64_t>& dst,
                                     std::unordered_set<uint32_t>& seen) {
        if (!dedup_by_node) {
            for (uint64_t i = 0; i < fd; ++i) dst.push_back(f[i]);
            for (uint64_t i = 0; i < rd; ++i) dst.push_back(r[i]);
            return;
        }
        seen.clear();
        for (uint64_t i = 0; i < fd; ++i) if (seen.insert(f[i]).second) dst.push_back(f[i]);
        for (uint64_t i = 0; i < rd; ++i) if (seen.insert(r[i]).second) dst.push_back(r[i]);
    };

    // PASS 1 — row_ptr degrees. Concat degree is an O(1) ROW_PTR subtraction;
    // the dedup degree is the realized per-row merge length, computed in
    // parallel across disjoint ranges (each writes disjoint degrees[] slots).
    std::vector<uint64_t> degrees(static_cast<std::size_t>(N));
    if (!dedup_by_node) {
        for (uint64_t u = 0; u < N; ++u) degrees[u] = rf.degree(u) + rr.degree(u);
    } else {
        std::atomic<bool> failed1{false};
        std::string err1;
        std::mutex err1_mu;
        auto counter = [&](uint64_t lo, uint64_t hi) {
            try {
                std::vector<uint64_t> dst;
                std::unordered_set<uint32_t> seen;
                for (uint64_t u = lo; u < hi; ++u) {
                    dst.clear();
                    merge_row(rf.col_idx32_row(u), rf.degree(u),
                              rr.col_idx32_row(u), rr.degree(u), dst, seen);
                    degrees[u] = dst.size();
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err1_mu);
                if (err1.empty()) err1 = e.what();
                failed1.store(true);
            }
        };
        unsigned W1 = std::max(1u, std::thread::hardware_concurrency());
        if (N < static_cast<uint64_t>(W1) * 4096ULL)
            W1 = std::max(1u, static_cast<unsigned>(std::max<uint64_t>(1, N / 4096ULL)));
        std::vector<std::thread> pool1;
        pool1.reserve(W1);
        const uint64_t per1 = (N + W1 - 1) / W1;
        for (unsigned w = 0; w < W1; ++w) {
            const uint64_t lo = std::min<uint64_t>(N, static_cast<uint64_t>(w) * per1);
            const uint64_t hi = std::min<uint64_t>(N, lo + per1);
            if (lo >= hi) break;
            pool1.emplace_back(counter, lo, hi);
        }
        for (auto& t : pool1) t.join();
        if (failed1.load())
            throw std::runtime_error(
                "build_symmetric_snapshot(dedup degree pass): " + err1);
    }

    TopologySnapshotWriter writer(
        proj_dir,
        std::string("topology_sym.csr"),
        std::vector<fs::path>{proj_dir / "from_to_edge.leaf",
                              proj_dir / "to_from_edge.leaf"},
        N, std::move(degrees), /*include_edge_ids=*/false,
        /*symmetric_format=*/true);

    // Worker pool sized to the machine — dynamic by available cores. Each worker
    // owns a disjoint, contiguous src range and pwrites its COL_IDX slice via the
    // parallel-safe append_subrange (no shared mutable writer state on that path).
    unsigned hw = std::thread::hardware_concurrency();
    unsigned W = std::max(1u, hw);
    // Don't oversubscribe tiny graphs.
    if (N < static_cast<uint64_t>(W) * 4096ULL) {
        W = static_cast<unsigned>(std::max<uint64_t>(1, N / 4096ULL));
        W = std::max(1u, W);
    }

    std::atomic<bool> failed{false};
    std::string err;
    std::mutex err_mu;
    auto worker = [&](uint64_t lo, uint64_t hi) {
        try {
            // Out-of-core bake: flush per row-chunk so resident RAM is
            // O(chunk edges) instead of O(range edges). On huge undirected
            // graphs the previous whole-range dst materialized the entire
            // undirected COL_IDX in heap (~M_undir * 8 B, ~26 GB on papers100M)
            // across W workers and OOMed the auto-bake before sampling even
            // started. append_subrange pwrites each sub-range at row_ptr[cs],
            // so the chunked output is byte-identical to the whole-range write.
            static const std::vector<uint64_t> kNoEids;
            constexpr uint64_t kChunkRows = 1ULL << 20;  // 1M rows per flush
            std::vector<uint64_t> dst;
            std::unordered_set<uint32_t> seen;
            for (uint64_t cs = lo; cs < hi; cs += kChunkRows) {
                const uint64_t ce = std::min<uint64_t>(hi, cs + kChunkRows);
                std::size_t cap = 0;
                for (uint64_t u = cs; u < ce; ++u) cap += rf.degree(u) + rr.degree(u);
                dst.clear();
                dst.reserve(cap);
                for (uint64_t u = cs; u < ce; ++u) {
                    merge_row(rf.col_idx32_row(u), rf.degree(u),
                              rr.col_idx32_row(u), rr.degree(u), dst, seen);
                }
                writer.append_subrange(cs, ce, dst, kNoEids);
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(err_mu);
            if (err.empty()) err = e.what();
            failed.store(true);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(W);
    const uint64_t per = (N + W - 1) / W;
    for (unsigned w = 0; w < W; ++w) {
        const uint64_t lo = std::min<uint64_t>(N, static_cast<uint64_t>(w) * per);
        const uint64_t hi = std::min<uint64_t>(N, lo + per);
        if (lo >= hi) break;
        pool.emplace_back(worker, lo, hi);
    }
    for (auto& t : pool) t.join();
    if (failed.load()) {
        throw std::runtime_error(
            "build_symmetric_snapshot(parallel concat): " + err);
    }

    writer.finalize();
    return writer.bytes_written();
}

}  // namespace

uint64_t build_symmetric_snapshot(GQL::ProjectionStorage& storage,
                                  bool verify, uint64_t verify_sample,
                                  bool* refused) {
    using GQL::Projection::TopologySnapshotWriter;
    using GQL::Projection::TopologySnapshotReader;
    *refused = false;

    if (storage.get_from_to_edge_index() == nullptr
        || storage.get_to_from_edge_index() == nullptr) {
        throw std::runtime_error(
            "build_symmetric_snapshot: both FROM_TO_EDGE and TO_FROM_EDGE "
            "indexes must be open to build topology_sym.csr");
    }

    const uint64_t N = storage.get_node_count();
    const fs::path proj_dir = storage.get_projection_dir();

    // FAST PATH: both directional narrow sidecars present + carrying edge_ids =>
    // the merge is a pure concat (distinct edge_ids remove nothing) => parallel.
    {
        auto rf = TopologySnapshotReader::open(
            proj_dir, TopologySnapshotReader::Direction::FORWARD);
        auto rr = TopologySnapshotReader::open(
            proj_dir, TopologySnapshotReader::Direction::REVERSE);
        const bool fast =
            rf.has_data() && rr.has_data()
            && rf.id_width() == 4 && rr.id_width() == 4
            && rf.num_nodes() == N && rr.num_nodes() == N
            && (rf.has_edge_ids() || rr.has_edge_ids());
        if (fast) {
            // PRESERVE duplicate (parallel) neighbors ONLY on a genuine
            // multigraph; a simple graph node-id dedups its mutual edges.
            const bool is_multigraph = GQL::Projection::detect_parallel_edges(
                storage.get_from_to_edge_index(), N);
            return build_symmetric_concat_parallel_(
                proj_dir, N, rf, rr, /*dedup_by_node=*/!is_multigraph);
        }
    }

    // FALLBACK: accessor 2-pass — handles node-id dedup (no real edge_ids) and
    // the no-sidecar / B+Tree case. Single-threaded, but always correct.
    TopologyAccessor acc(storage);
    Neighbors out_n, in_n;
    std::vector<uint64_t> out_dst, out_eid, in_dst, in_eid, m_dst, m_eid;

    auto fetch = [&](uint64_t u) {
        acc.get_out_neighbors_into(ObjectId{u}, out_n);
        acc.get_in_neighbors_into(ObjectId{u}, in_n);
        out_dst.clear(); out_eid.clear(); in_dst.clear(); in_eid.clear();
        for (auto& x : out_n.node_ids) out_dst.push_back(x.id);
        for (auto& x : out_n.edge_ids) out_eid.push_back(x.id);
        for (auto& x : in_n.node_ids)  in_dst.push_back(x.id);
        for (auto& x : in_n.edge_ids)  in_eid.push_back(x.id);
    };
    // PRESERVE parallel edges only on a true multigraph; a simple graph
    // (even with real edge_ids) node-id dedups its mutual edges. Computed once
    // over the from_to B+Tree.
    const bool is_multigraph = GQL::Projection::detect_parallel_edges(
        storage.get_from_to_edge_index(), N);

    std::vector<uint64_t> degrees(N, 0);
    for (uint64_t u = 0; u < N; ++u) {
        fetch(u);
        degrees[u] = GQL::Projection::merge_symmetric_row(
            out_dst, out_eid, in_dst, in_eid, is_multigraph, m_dst, m_eid);
    }

    TopologySnapshotWriter writer(
        proj_dir,
        std::string("topology_sym.csr"),
        std::vector<fs::path>{proj_dir / "from_to_edge.leaf",
                              proj_dir / "to_from_edge.leaf"},
        N, std::move(degrees), /*include_edge_ids=*/false,
        /*symmetric_format=*/true);

    std::mt19937_64 vrng(0xC0FFEE);
    for (uint64_t u = 0; u < N; ++u) {
        fetch(u);
        GQL::Projection::merge_symmetric_row(
            out_dst, out_eid, in_dst, in_eid, is_multigraph, m_dst, m_eid);
        for (uint64_t d : m_dst) {
            writer.append_edge(ObjectId{u},
                               ObjectId{d & ObjectId::VALUE_MASK}, ObjectId{});
        }
        const bool check = verify
            && (N <= 10'000'000ULL || (vrng() % N) < verify_sample);
        if (check) {
            Neighbors live;
            acc.get_neighbors_into(ObjectId{u},
                                   EdgeOrientation::UNDIRECTED, live);
            if (live.node_ids.size() != m_dst.size()) {
                throw std::runtime_error(
                    "build_symmetric_snapshot: self-verify degree mismatch at "
                    "node " + std::to_string(u));
            }
            for (std::size_t i = 0; i < m_dst.size(); ++i) {
                if ((live.node_ids[i].id & ObjectId::VALUE_MASK)
                    != (m_dst[i] & ObjectId::VALUE_MASK)) {
                    throw std::runtime_error(
                        "build_symmetric_snapshot: self-verify node mismatch at "
                        "node " + std::to_string(u));
                }
            }
        }
    }
    writer.finalize();
    return writer.bytes_written();
}

}  // namespace mdb::gnn
