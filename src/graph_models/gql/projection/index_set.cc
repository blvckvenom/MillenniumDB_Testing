#include "graph_models/gql/projection/index_set.h"

#include <cassert>
#include <string>

#include "query/exceptions.h"

namespace GQL {

ProjectionIndex project_index_mask_for(IndexSet preset) noexcept {
    switch (preset) {
    case IndexSet::ALL:
        return ProjectionIndex::ALL;

    case IndexSet::GNN_MINIMAL:
        // Bits required by the GNN k-hop sampling pipeline:
        //   NODES         — enumerate nodes
        //   NODE_LABEL    — label lookup per node
        //   LABEL_NODE    — node enumeration per label
        //   FROM_TO_EDGE  — forward-neighbor traversal
        //   TO_FROM_EDGE  — reverse-neighbor traversal (UNDIRECTED sampling)
        return ProjectionIndex::NODES
             | ProjectionIndex::NODE_LABEL
             | ProjectionIndex::LABEL_NODE
             | ProjectionIndex::FROM_TO_EDGE
             | ProjectionIndex::TO_FROM_EDGE;

    case IndexSet::READONLY_TRAVERSAL:
        // GNN_MINIMAL plus edge-label indexes for label-filtered traversal.
        return ProjectionIndex::NODES
             | ProjectionIndex::NODE_LABEL
             | ProjectionIndex::LABEL_NODE
             | ProjectionIndex::FROM_TO_EDGE
             | ProjectionIndex::TO_FROM_EDGE
             | ProjectionIndex::EDGE_LABEL
             | ProjectionIndex::LABEL_EDGE;
    }
    // Unreachable for valid enum values; for out-of-range casts (e.g., corrupt
    // catalog reading static_cast<IndexSet>(99)), return NONE (visibly wrong)
    // rather than ALL (which would silently mask the bug and materialize all
    // indexes, defeating the purpose of the indexSet selection feature that
    // allows projections to build only the B+Tree subset they need).
    assert(false && "project_index_mask_for: unhandled IndexSet value");
    return ProjectionIndex::NONE;
}

IndexSet parse_index_set(const std::string& s) {
    if (s == "ALL") {
        return IndexSet::ALL;
    }
    if (s == "GNN_MINIMAL") {
        return IndexSet::GNN_MINIMAL;
    }
    if (s == "READONLY_TRAVERSAL") {
        return IndexSet::READONLY_TRAVERSAL;
    }
    throw QueryException(
        "Invalid indexSet value: \"" + s + "\". "
        "Valid values are: ALL, GNN_MINIMAL, READONLY_TRAVERSAL "
        "(case-sensitive).");
}

const char* index_set_name(IndexSet preset) noexcept {
    switch (preset) {
    case IndexSet::ALL:                return "ALL";
    case IndexSet::GNN_MINIMAL:        return "GNN_MINIMAL";
    case IndexSet::READONLY_TRAVERSAL: return "READONLY_TRAVERSAL";
    }
    // Out-of-range cast: "UNKNOWN" is already visibly distinct from any valid
    // preset name (unlike returning ALL from project_index_mask_for, which
    // would mask a bug), but we still assert in debug builds for early detection.
    assert(false && "index_set_name: unhandled IndexSet value");
    return "UNKNOWN";
}

const char* projection_index_name(ProjectionIndex which) noexcept {
    // Names intentionally mirror the .leaf file naming from
    // ProjectionStorage::open_all_bplustree_readers_() so error
    // messages refer to the exact on-disk artifact the caller would
    // expect to find in `<proj_dir>/<name>.leaf`.
    switch (which) {
    case ProjectionIndex::NODES:          return "nodes";
    case ProjectionIndex::NODE_LABEL:     return "node_label";
    case ProjectionIndex::LABEL_NODE:     return "label_node";
    case ProjectionIndex::NODE_KEY_VALUE: return "node_key_value";
    case ProjectionIndex::KEY_VALUE_NODE: return "key_value_node";
    case ProjectionIndex::FROM_TO_EDGE:   return "from_to_edge";
    case ProjectionIndex::TO_FROM_EDGE:   return "to_from_edge";
    case ProjectionIndex::EDGE_DIRECTION: return "edge_direction";
    case ProjectionIndex::EDGE_FROM_TO:   return "edge_from_to";
    case ProjectionIndex::EDGE_N1_N2:     return "edge_n1_n2";
    case ProjectionIndex::EDGE_LABEL:     return "edge_label";
    case ProjectionIndex::LABEL_EDGE:     return "label_edge";
    case ProjectionIndex::EDGE_KEY_VALUE: return "edge_key_value";
    case ProjectionIndex::KEY_VALUE_EDGE: return "key_value_edge";
    // Composite masks have no .leaf file → not mappable to a single name.
    case ProjectionIndex::NONE:
    case ProjectionIndex::ALL_NODE:
    case ProjectionIndex::ALL_EDGE:
    case ProjectionIndex::ALL:
        break;
    }
    assert(false && "projection_index_name: must be a single-bit ProjectionIndex");
    return "UNKNOWN";
}

IndexSet minimum_preset_for(ProjectionIndex which) noexcept {
    // Walk the presets from most-restrictive to least-restrictive and
    // return the first one whose bitmask contains `which`. Ordering
    // matches IndexSet enum numerics: 1 (GNN_MINIMAL) < 2 (READONLY_TRAVERSAL)
    // < 0 (ALL), but ALL's ordinal 0 is historical (default preset); the
    // hierarchy we actually want for "minimum" is
    //   GNN_MINIMAL (5 bits) ⊂ READONLY_TRAVERSAL (7 bits) ⊂ ALL (14 bits).
    const ProjectionIndex gnn = project_index_mask_for(IndexSet::GNN_MINIMAL);
    if (has_flag(gnn, which)) {
        return IndexSet::GNN_MINIMAL;
    }
    const ProjectionIndex ro = project_index_mask_for(IndexSet::READONLY_TRAVERSAL);
    if (has_flag(ro, which)) {
        return IndexSet::READONLY_TRAVERSAL;
    }
    const ProjectionIndex all = project_index_mask_for(IndexSet::ALL);
    if (has_flag(all, which)) {
        return IndexSet::ALL;
    }
    // Not a single-bit value contained in any preset — out-of-range cast or
    // composite mask. ALL is a safe conservative fallback (still correct, and
    // never hides the bug in a way that would silently materialize *more*
    // indexes than any preset provides). Assert in debug.
    assert(false && "minimum_preset_for: value not contained in any IndexSet preset");
    return IndexSet::ALL;
}

} // namespace GQL
