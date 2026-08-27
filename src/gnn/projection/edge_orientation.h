#pragma once

// edge_orientation.h
//
// Lightweight header that hosts the `mdb::gnn::EdgeOrientation` enum
// without pulling in the full `topology_accessor.h` (which transitively
// includes torch + projection_storage + a sizeable portion of the GNN
// public surface).
//
// Hoisted out of `topology_accessor.h` when the Four-Level Topology Store
// (frequency-tiered L1 RAM hash / L2 compact uint32 CSR / L3 mmap sidecar /
// L4 direct B+Tree) was introduced, so that consumers that only need the
// orientation enum — `sampling_config.h`, `four_level_topology_store.h`, and
// `RowMapping` — do not pay the full transitive include cost of pulling in
// torch, projection_storage, and the rest of the GNN public surface.
//
// `topology_accessor.h` keeps its existing definition as a re-include
// of this header; downstream files that already include
// `topology_accessor.h` see the enum unchanged.
//
// This header is intentionally header-only and dependency-free (no
// project includes, no STL beyond what the enum itself requires) so it
// can be included from any layer of the GNN module.

namespace mdb::gnn {

/**
 * @brief Edge orientation for neighbor traversal.
 *
 * Mirrors GQL::Procedures::Orientation but kept separate for module isolation.
 * Controls how edges are traversed during neighbor lookup and sampling.
 *
 * Terminology note: ISO/IEC 39075:2024 (GQL) defines directionality as a
 * property of the stored edge itself — a directed edge distinguishes a source
 * and a destination endpoint (§3.4.12 "directed edge"), an undirected edge
 * "does not distinguish between its endpoints" (§3.4.13 "undirected edge") —
 * and every edge carries "the indication of whether the edge is a directed
 * edge or an undirected edge" (§4.3.5 "Graphs"). This enum is a different,
 * implementation-side notion the standard does not define: a TRAVERSAL
 * orientation applied to the projection's stored directed edges. UNDIRECTED
 * here means "visit both directions of a stored directed edge", not that the
 * edge itself is undirected.
 */
enum class EdgeOrientation {
    NATURAL,     ///< Follow edge direction as stored (from -> to)
    REVERSE,     ///< Reverse edge direction (to -> from)
    UNDIRECTED   ///< Traverse both directions (bidirectional access)
};

}  // namespace mdb::gnn
