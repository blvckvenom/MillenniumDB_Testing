#include "query/procedure/builtin/gnn_build_topology_snapshot_procedure.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/projection/topology_snapshot.h"
#include "graph_models/gql/projection/topology_snapshot_writer.h"
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
            // and the sampler masks on read (Spec #4-B convention, f71b3bf0).
            writer.append_edge(
                ObjectId{ObjectId{(*rec)[0]}.get_value()},
                ObjectId{(*rec)[1]},
                ObjectId{(*rec)[2]});
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

void GnnBuildTopologySnapshotProcedure::execute(ProcedureContext& ctx) {
    using Projection::TopologySnapshotWriter;

    // -------------------------------------------------------------------
    // Step 1: parse arguments
    // -------------------------------------------------------------------
    if (ctx.arguments.size() != 1) {
        throw QueryException(
            "gnn_build_topology_snapshot requires exactly 1 argument.\n\n"
            "Usage: CALL gnn_build_topology_snapshot(projection)\n"
            "Example: CALL gnn_build_topology_snapshot('my_proj')");
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

    if (fwd_eligible) {
        try {
            fwd_bytes = build_one_snapshot_post_hoc(
                storage, TopologySnapshotWriter::Direction::FORWARD);
        } catch (const std::exception& e) {
            std::cerr << "gnn_build_topology_snapshot: failed to build "
                         "topology_fwd.csr for projection '"
                      << projection_name << "': " << e.what() << std::endl;
        }
    }
    if (rev_eligible) {
        try {
            rev_bytes = build_one_snapshot_post_hoc(
                storage, TopologySnapshotWriter::Direction::REVERSE);
        } catch (const std::exception& e) {
            std::cerr << "gnn_build_topology_snapshot: failed to build "
                         "topology_rev.csr for projection '"
                      << projection_name << "': " << e.what() << std::endl;
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    // -------------------------------------------------------------------
    // Step 5: yield results
    // -------------------------------------------------------------------
    ctx.yield("projectionName", ctx.create_string(projection_name));
    ctx.yield("fwdBytes",       ctx.create_int(static_cast<int64_t>(fwd_bytes)));
    ctx.yield("revBytes",       ctx.create_int(static_cast<int64_t>(rev_bytes)));
    ctx.yield("durationMillis", ctx.create_int(duration_ms));
    ctx.yield_row();
}

} // namespace GQL::Procedures
