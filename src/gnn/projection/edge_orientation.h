#pragma once

// edge_orientation.h
//
// Lightweight header that hosts the `mdb::gnn::EdgeOrientation` enum
// without pulling in the full `topology_accessor.h` (which transitively
// includes torch + projection_storage + a sizeable portion of the GNN
// public surface).
//
// Hoisted out of `topology_accessor.h` in Spec #13 Phase 3 (T13.7
// carry-forward (b)) so that consumers that only need the orientation
// enum — `sampling_config.h`, `four_level_topology_store.h`, and the
// future `RowMapping` shared header — do not pay the full transitive
// include cost.
//
// `topology_accessor.h` keeps its existing definition as a re-include
// of this header; downstream files that already include
// `topology_accessor.h` see the enum unchanged.
//
// Spec reference:
//   docs/superpowers/specs/2026-04-25-four-level-topology-store-design.md
//   §3.2 (C++ class) — Config::orientation field type.
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
 * @see ISO/IEC 39075:2024 §4.3.5 (Undirected Edge Handling)
 */
enum class EdgeOrientation {
    NATURAL,     ///< Follow edge direction as stored (from -> to)
    REVERSE,     ///< Reverse edge direction (to -> from)
    UNDIRECTED   ///< Traverse both directions (bidirectional access)
};

}  // namespace mdb::gnn
