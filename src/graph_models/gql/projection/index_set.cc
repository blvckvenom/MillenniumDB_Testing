#include "graph_models/gql/projection/index_set.h"

#include <stdexcept>
#include <string>

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
    // Unreachable for well-formed enum values; default to ALL to preserve
    // the safest (current) behavior if a future value slips through.
    return ProjectionIndex::ALL;
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
    throw std::invalid_argument(
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
    return "UNKNOWN";
}

} // namespace GQL
