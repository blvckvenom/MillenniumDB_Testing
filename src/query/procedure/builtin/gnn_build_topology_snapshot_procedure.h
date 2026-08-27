#pragma once

// gnn_build_topology_snapshot procedure.
//
// Post-hoc generator for `topology_fwd.csr` / `topology_rev.csr` — the
// mmap-backed CSR sidecar files that enable O(1) neighbor slicing during
// GNN sampling — on a projection that was built without
// `buildTopologySnapshot: true`.
// Structurally mirrors `gnn_build_feature_store` — open an existing
// projection, perform a bounded post-processing step, emit YIELD rows — but
// touches only the projection directory (no FourLevelStore, no samples).
//
// Signature:
//
//     CALL gnn_build_topology_snapshot(projection: STRING)
//     YIELD projectionName  STRING,
//           fwdBytes        INT,
//           revBytes        INT,
//           fwdStatus       STRING,
//           revStatus       STRING,
//           durationMillis  INT
//
// Semantics:
//   * Raises `QueryException` when the projection does not exist.
//   * Raises `QueryException` when the projection's IndexSet carries
//     neither `FROM_TO_EDGE` nor `TO_FROM_EDGE` — the caller is told to
//     rebuild with `indexSet='ALL'`, `'GNN_MINIMAL'`, or
//     `'READONLY_TRAVERSAL'`.
//   * Raises `QueryException` when every eligible direction fails to
//     build; a single-direction failure is reported via the per-direction
//     status yields ('built', 'skipped', or 'failed: <error>') so a remote
//     client can distinguish IndexSet-skip from a build failure.
//   * Idempotent: when a sidecar file already exists it is overwritten with
//     a one-line stderr warning. This matches the topology-snapshot writer's
//     atomic `.tmp → rename` contract — the old file is replaced only if the
//     new one finalizes cleanly.
//   * Per-direction emission is gated by the active IndexSet mask
//     (same check as the builder's `build_topology_snapshots_()` in
//     `native_projection_builder.cc`), so a `READONLY_TRAVERSAL` projection
//     produces both files while a hypothetical FWD-only preset produces
//     just `topology_fwd.csr`.

#include <cstdint>
#include <filesystem>
#include <tuple>

#include "query/procedure/procedure.h"

namespace GQL {
class ProjectionStorage;
}

namespace GQL::Procedures {

class GnnBuildTopologySnapshotProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_build_topology_snapshot"; }
    std::string qualified_name() const override { return "gnn_build_topology_snapshot"; }
    std::string description() const override {
        return "Generate topology_fwd.csr / topology_rev.csr sidecar files for "
               "an existing projection that was built without "
               "buildTopologySnapshot: true. Idempotent — existing sidecars are "
               "overwritten.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("projection", ParamType::STRING, true,
                      "Name of the projection to augment with a CSR sidecar"),
            Parameter("mode", ParamType::STRING, false,
                      "Build mode: 'directional' (default), 'symmetric', or 'both'"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"projectionName", YieldType::STRING, "Projection name"},
            YieldField{"fwdBytes",       YieldType::INT,
                       "Bytes written to topology_fwd.csr (0 when not emitted)"},
            YieldField{"revBytes",       YieldType::INT,
                       "Bytes written to topology_rev.csr (0 when not emitted)"},
            YieldField{"fwdStatus",      YieldType::STRING,
                       "Forward direction outcome: 'built', 'skipped' "
                       "(IndexSet-ineligible), or 'failed: <error>'"},
            YieldField{"revStatus",      YieldType::STRING,
                       "Reverse direction outcome: 'built', 'skipped' "
                       "(IndexSet-ineligible), or 'failed: <error>'"},
            YieldField{"symBytes",       YieldType::INT,
                       "Bytes written to topology_sym.csr (0 when not emitted)"},
            YieldField{"symStatus",      YieldType::STRING,
                       "Symmetric outcome: 'built', 'skipped[: reason]', "
                       "'refused: parallel-edge multigraph', or 'failed: <error>'"},
            YieldField{"parallelEdgeRefused", YieldType::BOOL,
                       "True when the symmetric bake abstained on a parallel-edge "
                       "multigraph"},
            YieldField{"durationMillis", YieldType::INT,
                       "Total wall-clock time in milliseconds"},
        };
    }

    void execute(ProcedureContext& ctx) override;

    // Test-only hook: runs the same BPT-scan + writer body the GQL
    // execute() path drives, but against a supplied ProjectionStorage
    // instead of constructing a ProcedureContext. Returns
    // (fwd_bytes, rev_bytes, duration_millis); raises `std::runtime_error`
    // (not `QueryException`) on per-direction failure so unit tests can
    // see the real error.
    //
    // `build_forward` / `build_reverse` are gated by the caller so tests
    // can exercise partial emission without tampering with the storage's
    // IndexSet state.
    static std::tuple<uint64_t, uint64_t, int64_t>
    run_for_test(GQL::ProjectionStorage& storage,
                 bool build_forward,
                 bool build_reverse);

    // Test-only hook for the post-hoc symmetric (pre-merged undirected) bake.
    // Builds topology_sym.csr with build-time self-verification enabled and
    // returns (sym_bytes, duration_millis, parallel_edge_refused). bytes==0 with
    // refused==true means a parallel-edge multigraph was detected and the bake
    // abstained (no file written).
    static std::tuple<uint64_t, int64_t, bool>
    run_symmetric_for_test(GQL::ProjectionStorage& storage);

    // Test-only hook mirroring execute()'s symmetric block for a given mode
    // ('directional' | 'symmetric' | 'both'). Returns (sym_bytes, sym_status,
    // parallel_edge_refused); it never builds the directional sidecars (those
    // are covered by run_for_test). For 'directional' it is a no-op returning
    // (0, "skipped", false).
    static std::tuple<uint64_t, std::string, bool>
    run_mode_for_test(GQL::ProjectionStorage& storage, const std::string& mode);
};

} // namespace GQL::Procedures
