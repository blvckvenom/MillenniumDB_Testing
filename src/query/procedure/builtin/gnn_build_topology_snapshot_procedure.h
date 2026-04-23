#pragma once

// gnn_build_topology_snapshot procedure (Spec #4-B T4.9).
//
// Post-hoc generator for `topology_fwd.csr` / `topology_rev.csr` sidecar
// files on a projection that was built without `buildTopologySnapshot: true`.
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
//           durationMillis  INT
//
// Semantics:
//   * Raises `QueryException` when the projection does not exist.
//   * Raises `QueryException` when the projection's IndexSet carries
//     neither `FROM_TO_EDGE` nor `TO_FROM_EDGE` — the caller is told to
//     rebuild with `indexSet='ALL'`, `'GNN_MINIMAL'`, or
//     `'READONLY_TRAVERSAL'`.
//   * Idempotent: when a sidecar file already exists it is overwritten with
//     a one-line stderr warning. This matches the T4.4 writer's atomic
//     `.tmp → rename` contract — the old file is replaced only if the new
//     one finalizes cleanly.
//   * Per-direction emission is gated by the active IndexSet mask
//     (same check as the builder's `build_topology_snapshots_()` in
//     `native_projection_builder.cc`), so a `READONLY_TRAVERSAL` projection
//     produces both files while a hypothetical FWD-only preset produces
//     just `topology_fwd.csr`.
//
// Spec reference: docs/superpowers/specs/2026-04-25-topology-snapshot-design.md §4.2
// Plan: docs/superpowers/plans/2026-04-25-topology-snapshot-plan.md §T4.9

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
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"projectionName", YieldType::STRING, "Projection name"},
            YieldField{"fwdBytes",       YieldType::INT,
                       "Bytes written to topology_fwd.csr (0 when not emitted)"},
            YieldField{"revBytes",       YieldType::INT,
                       "Bytes written to topology_rev.csr (0 when not emitted)"},
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
};

} // namespace GQL::Procedures
