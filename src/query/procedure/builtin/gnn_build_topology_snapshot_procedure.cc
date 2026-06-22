#include "query/procedure/builtin/gnn_build_topology_snapshot_procedure.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "gnn/projection/edge_orientation.h"
#include "gnn/projection/topology_accessor.h"
#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot.h"
#include "graph_models/gql/projection/topology_snapshot_writer.h"
#include "graph_models/gql/projection/topology_symmetric_merge.h"
#include "graph_models/object_id.h"
#include "query/exceptions.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"
#include "query/procedure/procedure_context.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

namespace {

// Emit a single-direction CSR sidecar for a projection whose corresponding
// edge B+Tree is already open. Shared by the GQL execute() path (whose
// arguments are parsed + validated there) and the test hook exposed via
// `run_for_test`. Mirrors `NativeProjectionBuilder::build_one_topology_snapshot_`
// byte-for-byte: two BPT<3> scans (degree histogram + src-monotonic
// streaming append) feeding a single `TopologySnapshotWriter::finalize()`.
//
// Returns the number of bytes written to the finalized `.csr`. Throws on
// any writer / BPT error so the caller can log + continue with the other
// direction (the GQL execute() path surfaces per-direction failures as
// stderr warnings and keeps total-time accounting accurate).
uint64_t build_one_snapshot_post_hoc(
    GQL::ProjectionStorage& storage,
    Projection::TopologySnapshotWriter::Direction dir)
{
    using Projection::TopologySnapshotWriter;

    BPlusTree<3>* edge_bpt = (dir == TopologySnapshotWriter::Direction::FORWARD)
        ? storage.get_from_to_edge_index()
        : storage.get_to_from_edge_index();

    if (edge_bpt == nullptr) {
        throw std::runtime_error(
            "gnn_build_topology_snapshot: requested direction's edge B+Tree "
            "is not open on the projection — IndexSet may have dropped it");
    }

    const uint64_t num_nodes = storage.get_node_count();
    const fs::path proj_dir  = storage.get_projection_dir();

    // Warn once when overwriting. The writer itself still serializes through
    // `.tmp → rename`, so the prior file stays intact until the new one
    // finalizes — the "overwrite" is atomic at the filesystem level.
    const char* basename =
        (dir == TopologySnapshotWriter::Direction::FORWARD)
            ? "topology_fwd.csr"
            : "topology_rev.csr";
    if (fs::exists(proj_dir / basename)) {
        std::cerr << "gnn_build_topology_snapshot: overwriting existing "
                  << basename << " at " << (proj_dir / basename).string()
                  << std::endl;
    }

    // Pass 1: per-source degree histogram. The BPT key layout for both
    // from_to_edge (src, dst, edge_id) and to_from_edge (dst, src, edge_id)
    // places the CSR-key node at index 0.
    std::vector<uint64_t> degrees(num_nodes, 0);
    bool interrupt = false;
    Record<3> min_rec = {0, 0, 0};
    Record<3> max_rec = {UINT64_MAX, UINT64_MAX, UINT64_MAX};

    {
        auto iter = edge_bpt->get_range(&interrupt, min_rec, max_rec);
        const Record<3>* rec = nullptr;
        while ((rec = iter.next()) != nullptr) {
            // B+Tree records store full ObjectIds with the 8-bit type tag.
            // The CSR ROW_PTR is indexed by dense row id, so strip the tag
            // via get_value() (matches topology_accessor.cc read convention).
            uint64_t src_idx = ObjectId{(*rec)[0]}.get_value();
            if (src_idx < num_nodes) {
                ++degrees[src_idx];
            }
        }
    }

    // Pass 2: stream edges in src-monotonic order into the writer.
    TopologySnapshotWriter writer(
        proj_dir,
        dir,
        num_nodes,
        std::move(degrees),
        /*include_edge_ids=*/true);

    {
        auto iter = edge_bpt->get_range(&interrupt, min_rec, max_rec);
        const Record<3>* rec = nullptr;
        while ((rec = iter.next()) != nullptr) {
            // Pass the dense row id (tag stripped) as the CSR row key; dst and
            // edge_id stay tag-bearing — COL_IDX/EDGE_IDS store raw ObjectId.id
            // and the sampler masks away the 8-bit ObjectId type tag on read
            // (consistent with the topology CSR sidecar read convention).
            writer.append_edge(
                ObjectId{ObjectId{(*rec)[0]}.get_value()},
                ObjectId{(*rec)[1]},
                ObjectId{(*rec)[2]});
        }
    }

    writer.finalize();
    return writer.bytes_written();
}

// Build the pre-merged undirected CSR sidecar (topology_sym.csr) post-hoc by
// merging each node's out+in neighbor lists via the canonical UNDIRECTED dedup
// (Projection::merge_symmetric_row). edge_ids are dropped (the symmetric sample
// CSR never carries them); the dst node receptive field stays byte-identical to
// the runtime out+in+merge. Returns bytes written. Throws std::runtime_error on
// a self-verify mismatch (verify=true) so a corrupt bake never finalizes.
//
// This bake NEVER refuses: it keys the dedup on the SOURCE `has_edge_ids` flag,
// exactly like the accessor, so a graph with parallel / mutual edges (e.g. cora
// UNDIRECTED, where 151 mutual citations become duplicated (src,dst) records)
// keeps every duplicate neighbor — the node sequence is reproduced byte-for-byte
// and only the edge_id VALUES are zeroed (model-irrelevant for node
// classification). The parallel-edge guard (detect_parallel_edges) belongs to
// the four-level edge_id-DROP path, where the dedup key is forced to node-id and
// duplicates WOULD collapse, changing the receptive field. `*refused` is always
// set false here and kept only for yield/API stability.
//
// `verify_sample`: when N is large only ~verify_sample randomly-chosen rows are
// cross-checked against the live accessor; all rows are checked when N is small.
uint64_t build_symmetric_snapshot_post_hoc(GQL::ProjectionStorage& storage,
                                           bool verify, uint64_t verify_sample,
                                           bool* refused) {
    using Projection::TopologySnapshotWriter;
    *refused = false;

    if (storage.get_from_to_edge_index() == nullptr
        || storage.get_to_from_edge_index() == nullptr) {
        throw std::runtime_error(
            "gnn_build_topology_snapshot(symmetric): both FROM_TO_EDGE and "
            "TO_FROM_EDGE indexes must be open to build topology_sym.csr");
    }

    const uint64_t N = storage.get_node_count();
    const fs::path proj_dir = storage.get_projection_dir();

    mdb::gnn::TopologyAccessor acc(storage);
    mdb::gnn::Neighbors out_n, in_n;
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
    // edge_ids are "real" (matching the accessor's per-node rule) when the
    // first out OR in edge_id is non-zero.
    auto has_eids = [&]() {
        return (!out_eid.empty() && out_eid.front() != 0)
            || (!in_eid.empty()  && in_eid.front()  != 0);
    };

    // Pass 1: merged-degree histogram (post-dedup, so ROW_PTR is sized right).
    std::vector<uint64_t> degrees(N, 0);
    for (uint64_t u = 0; u < N; ++u) {
        fetch(u);
        degrees[u] = Projection::merge_symmetric_row(
            out_dst, out_eid, in_dst, in_eid, has_eids(), m_dst, m_eid);
    }

    // Pass 2: emit the merged node list per row; edge_id dropped (always 0).
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
        Projection::merge_symmetric_row(
            out_dst, out_eid, in_dst, in_eid, has_eids(), m_dst, m_eid);
        for (uint64_t d : m_dst) {
            writer.append_edge(ObjectId{u},
                               ObjectId{d & ObjectId::VALUE_MASK}, ObjectId{});
        }
        // Self-verify against the live accessor UNDIRECTED list: all rows when
        // small, a random sample when huge.
        const bool check = verify
            && (N <= 10'000'000ULL || (vrng() % N) < verify_sample);
        if (check) {
            mdb::gnn::Neighbors live;
            acc.get_neighbors_into(ObjectId{u},
                                   mdb::gnn::EdgeOrientation::UNDIRECTED, live);
            if (live.node_ids.size() != m_dst.size()) {
                throw std::runtime_error(
                    "gnn_build_topology_snapshot(symmetric): self-verify degree "
                    "mismatch at node " + std::to_string(u));
            }
            for (std::size_t i = 0; i < m_dst.size(); ++i) {
                if ((live.node_ids[i].id & ObjectId::VALUE_MASK)
                    != (m_dst[i] & ObjectId::VALUE_MASK)) {
                    throw std::runtime_error(
                        "gnn_build_topology_snapshot(symmetric): self-verify node "
                        "mismatch at node " + std::to_string(u));
                }
            }
        }
    }
    writer.finalize();
    return writer.bytes_written();
}

} // namespace

std::tuple<uint64_t, uint64_t, int64_t>
GnnBuildTopologySnapshotProcedure::run_for_test(
    GQL::ProjectionStorage& storage,
    bool                    build_forward,
    bool                    build_reverse)
{
    using Projection::TopologySnapshotWriter;

    const auto t_start = std::chrono::steady_clock::now();

    uint64_t fwd_bytes = 0;
    uint64_t rev_bytes = 0;

    if (build_forward) {
        fwd_bytes = build_one_snapshot_post_hoc(
            storage, TopologySnapshotWriter::Direction::FORWARD);
    }
    if (build_reverse) {
        rev_bytes = build_one_snapshot_post_hoc(
            storage, TopologySnapshotWriter::Direction::REVERSE);
    }

    const auto t_end = std::chrono::steady_clock::now();
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    return {fwd_bytes, rev_bytes, duration_ms};
}

std::tuple<uint64_t, int64_t, bool>
GnnBuildTopologySnapshotProcedure::run_symmetric_for_test(
    GQL::ProjectionStorage& storage)
{
    const auto t0 = std::chrono::steady_clock::now();
    bool refused = false;
    uint64_t bytes = build_symmetric_snapshot_post_hoc(
        storage, /*verify=*/true, /*verify_sample=*/UINT64_MAX, &refused);
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    return {bytes, ms, refused};
}

std::tuple<uint64_t, std::string, bool>
GnnBuildTopologySnapshotProcedure::run_mode_for_test(
    GQL::ProjectionStorage& storage, const std::string& mode)
{
    uint64_t    sym_bytes = 0;
    std::string sym_status = "skipped";
    bool        refused = false;
    if (mode == "symmetric" || mode == "both") {
        if (storage.get_from_to_edge_index() == nullptr
            || storage.get_to_from_edge_index() == nullptr) {
            sym_status =
                "skipped: symmetric requires both FROM_TO_EDGE and TO_FROM_EDGE";
        } else if (storage.get_graph_storage() == BPT::GraphStorage::CSR_HYBRID) {
            sym_status = "skipped: CSR_HYBRID";
        } else {
            try {
                sym_bytes = build_symmetric_snapshot_post_hoc(
                    storage, /*verify=*/true,
                    /*verify_sample=*/storage.get_node_count() <= 10'000'000ULL
                                      ? UINT64_MAX : 100'000ULL,
                    &refused);
                sym_status = refused ? "refused: parallel-edge multigraph"
                                     : "built";
            } catch (const std::exception& e) {
                sym_status = std::string("failed: ") + e.what();
            }
        }
    }
    return {sym_bytes, sym_status, refused};
}

void GnnBuildTopologySnapshotProcedure::execute(ProcedureContext& ctx) {
    using Projection::TopologySnapshotWriter;

    // -------------------------------------------------------------------
    // Step 1: parse arguments
    // -------------------------------------------------------------------
    if (ctx.arguments.size() < 1 || ctx.arguments.size() > 2) {
        throw QueryException(
            "gnn_build_topology_snapshot requires 1 or 2 arguments.\n\n"
            "Usage: CALL gnn_build_topology_snapshot(projection [, mode])\n"
            "  mode: 'directional' (default), 'symmetric', or 'both'\n"
            "Example: CALL gnn_build_topology_snapshot('my_proj', 'both')");
    }

    std::string projection_name;
    try {
        projection_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw QueryException(
            std::string("gnn_build_topology_snapshot: projection argument must "
                        "be a STRING (") + e.what() + ")");
    }
    if (projection_name.empty()) {
        throw QueryException(
            "gnn_build_topology_snapshot: projection name cannot be empty");
    }
    validate_safe_name(projection_name, "projection");

    // Optional build mode: 'directional' (default), 'symmetric', or 'both'.
    std::string mode = "directional";
    if (ctx.arguments.size() == 2) {
        try {
            mode = ctx.get_string_argument(1);
        } catch (const std::exception& e) {
            throw QueryException(
                std::string("gnn_build_topology_snapshot: mode argument must be "
                            "a STRING (") + e.what() + ")");
        }
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (mode != "directional" && mode != "symmetric" && mode != "both") {
            throw QueryException(
                "gnn_build_topology_snapshot: mode must be 'directional', "
                "'symmetric', or 'both' (got '" + mode + "')");
        }
    }
    const bool want_directional = (mode == "directional" || mode == "both");
    const bool want_symmetric   = (mode == "symmetric"   || mode == "both");

    // -------------------------------------------------------------------
    // Step 2: resolve projection directory + open storage read-only
    // -------------------------------------------------------------------
    auto& manager = GQL::ProjectionManager::get_instance();
    if (!manager.projection_exists(projection_name)) {
        throw QueryException(
            "gnn_build_topology_snapshot: projection '" + projection_name
            + "' not found. Create it first with CALL graph_project(...)");
    }

    const std::string proj_dir   = manager.get_projection_dir(projection_name);
    const std::string db_folder  = manager.get_db_folder();

    GQL::ProjectionStorage storage(proj_dir, db_folder);
    storage.open();

    // -------------------------------------------------------------------
    // Step 3: gate on the active IndexSet mask
    // -------------------------------------------------------------------
    const IndexSet active_preset     = storage.get_index_set();
    const ProjectionIndex active_mask =
        project_index_mask_for(active_preset);
    const bool fwd_eligible =
        has_flag(active_mask, ProjectionIndex::FROM_TO_EDGE)
        && storage.get_from_to_edge_index() != nullptr;
    const bool rev_eligible =
        has_flag(active_mask, ProjectionIndex::TO_FROM_EDGE)
        && storage.get_to_from_edge_index() != nullptr;

    if (!fwd_eligible && !rev_eligible) {
        throw QueryException(
            "Cannot build topology snapshot: projection '" + projection_name
            + "' lacks both FROM_TO_EDGE and TO_FROM_EDGE indexes. Rebuild "
              "with indexSet='ALL', 'GNN_MINIMAL', or 'READONLY_TRAVERSAL'.");
    }

    // -------------------------------------------------------------------
    // Step 4: build per-direction sidecars (non-fatal per direction)
    // -------------------------------------------------------------------
    const auto t_start = std::chrono::steady_clock::now();

    uint64_t fwd_bytes = 0;
    uint64_t rev_bytes = 0;
    // Per-direction outcome surfaced to the client: bytes alone cannot
    // distinguish "skipped by IndexSet" from "failed" (both report 0).
    std::string fwd_status = "skipped";
    std::string rev_status = "skipped";

    if (want_directional && fwd_eligible) {
        try {
            fwd_bytes = build_one_snapshot_post_hoc(
                storage, TopologySnapshotWriter::Direction::FORWARD);
            fwd_status = "built";
        } catch (const std::exception& e) {
            fwd_status = std::string("failed: ") + e.what();
            std::cerr << "gnn_build_topology_snapshot: failed to build "
                         "topology_fwd.csr for projection '"
                      << projection_name << "': " << e.what() << std::endl;
        }
    }
    if (want_directional && rev_eligible) {
        try {
            rev_bytes = build_one_snapshot_post_hoc(
                storage, TopologySnapshotWriter::Direction::REVERSE);
            rev_status = "built";
        } catch (const std::exception& e) {
            rev_status = std::string("failed: ") + e.what();
            std::cerr << "gnn_build_topology_snapshot: failed to build "
                         "topology_rev.csr for projection '"
                      << projection_name << "': " << e.what() << std::endl;
        }
    }

    // Symmetric (pre-merged undirected) sidecar — built only when requested and
    // both directions are eligible. CSR_HYBRID projections already provide O(1)
    // neighbor access via their v3 leaves, so the symmetric sidecar is skipped.
    uint64_t    sym_bytes = 0;
    std::string sym_status = "skipped";
    bool        parallel_edge_refused = false;
    if (want_symmetric) {
        if (!(fwd_eligible && rev_eligible)) {
            sym_status =
                "skipped: symmetric requires both FROM_TO_EDGE and TO_FROM_EDGE";
        } else if (storage.get_graph_storage()
                   == BPT::GraphStorage::CSR_HYBRID) {
            sym_status = "skipped: CSR_HYBRID";
        } else {
            try {
                sym_bytes = build_symmetric_snapshot_post_hoc(
                    storage, /*verify=*/true,
                    /*verify_sample=*/storage.get_node_count() <= 10'000'000ULL
                                      ? UINT64_MAX : 100'000ULL,
                    &parallel_edge_refused);
                sym_status = parallel_edge_refused
                    ? "refused: parallel-edge multigraph"
                    : "built";
            } catch (const std::exception& e) {
                sym_status = std::string("failed: ") + e.what();
                std::cerr << "gnn_build_topology_snapshot: symmetric build "
                             "failed for projection '"
                          << projection_name << "': " << e.what() << std::endl;
            }
        }
    }

    // When every requested build failed/skipped nothing was materialized — raise
    // instead of yielding a row a remote client would read as success.
    const bool any_built = (want_directional && fwd_eligible && fwd_status == "built")
                        || (want_directional && rev_eligible && rev_status == "built")
                        || (want_symmetric && sym_status == "built");
    if (!any_built) {
        throw QueryException(
            "gnn_build_topology_snapshot: no sidecar could be built for "
            "projection '" + projection_name + "'. fwd: " + fwd_status
            + "; rev: " + rev_status + "; sym: " + sym_status);
    }

    const auto t_end = std::chrono::steady_clock::now();
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    // -------------------------------------------------------------------
    // Step 5: yield results
    // -------------------------------------------------------------------
    ctx.yield("projectionName",      ctx.create_string(projection_name));
    ctx.yield("fwdBytes",            ctx.create_int(static_cast<int64_t>(fwd_bytes)));
    ctx.yield("revBytes",            ctx.create_int(static_cast<int64_t>(rev_bytes)));
    ctx.yield("fwdStatus",           ctx.create_string(fwd_status));
    ctx.yield("revStatus",           ctx.create_string(rev_status));
    ctx.yield("symBytes",            ctx.create_int(static_cast<int64_t>(sym_bytes)));
    ctx.yield("symStatus",           ctx.create_string(sym_status));
    ctx.yield("parallelEdgeRefused", ctx.create_bool(parallel_edge_refused));
    ctx.yield("durationMillis",      ctx.create_int(duration_ms));
    ctx.yield_row();
}

} // namespace GQL::Procedures
