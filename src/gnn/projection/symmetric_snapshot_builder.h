#pragma once

// symmetric_snapshot_builder.h — the single owner of the symmetric (pre-merged
// undirected) topology snapshot.
//
// Builds / refreshes `<projection_dir>/topology_sym.csr`: for every node u, the
// merged undirected neighbor list out(u) ∪ in(u), with edge_ids dropped, written
// in the narrow uint32 layout when ids fit (magic TOPOSYM1). The merge rule is
// the canonical `Projection::merge_symmetric_row`: edge_id-keyed when real
// edge_ids exist (distinct => concat, nothing removed) else node-id dedup;
// emission is out(u) first, then in(u) survivors.
//
// This function is the SINGLE home of the symmetric merge. Two callers share it:
//   - the `gnn_build_topology_snapshot` procedure (explicit, user-triggered bake)
//   - the offline sampling engine (auto-bake when topology_sym.csr is missing,
//     so the GPU-UVA symmetric path opens the baked slice instead of merging the
//     two directional sidecars in RAM on every sample).
//
// Neighbor data is read through a `TopologyAccessor`, which uses the directional
// CSR sidecars (topology_fwd/rev.csr) when present (fast O(1) slices) and falls
// back to the B+Trees otherwise — so the bake is as cheap as a sidecar scan when
// the projection has them. Lives in the gnn layer (not the projection layer)
// because it depends on TopologyAccessor.

#include <cstdint>

namespace GQL { class ProjectionStorage; }

namespace mdb::gnn {

/**
 * @brief Build (or overwrite) `<projection_dir>/topology_sym.csr` from the
 *        projection's directional topology.
 *
 * @param storage        Open projection storage; both FROM_TO_EDGE and
 *                        TO_FROM_EDGE indexes must be open (throws otherwise).
 * @param verify         When true, self-verify merged rows against the live
 *                        accessor's UNDIRECTED list (all rows for small graphs,
 *                        a random sample of ~verify_sample rows for huge ones).
 * @param verify_sample  Expected number of rows to spot-check when N is huge.
 * @param refused        Always set to false (the bake reproduces the accessor's
 *                        per-node receptive field byte-for-byte using the source
 *                        has_edge_ids flag, so it never abstains). Kept for
 *                        signature symmetry with the directional drop path.
 * @return Bytes written to topology_sym.csr.
 * @throws std::runtime_error on a missing edge index, a writer/BPT error, or a
 *         self-verify mismatch.
 */
uint64_t build_symmetric_snapshot(GQL::ProjectionStorage& storage,
                                  bool verify, uint64_t verify_sample,
                                  bool* refused);

}  // namespace mdb::gnn
