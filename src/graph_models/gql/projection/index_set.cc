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
    // indexes, defeating the purpose of Spec #3).
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

} // namespace GQL
