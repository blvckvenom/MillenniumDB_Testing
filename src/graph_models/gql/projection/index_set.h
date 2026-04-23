#pragma once

#include <cstdint>
#include <string>

#include "graph_models/gql/projection/native_projection_builder.h"

namespace GQL {

/**
 * @brief IndexSet — user-facing preset names for index selection in graph_project.
 *
 * Each preset corresponds to a specific subset of ProjectionIndex bits
 * appropriate for a workload class. Foundation for Spec #3 — user-selectable
 * index materialization for GNN workloads.
 *
 * Preset membership:
 *   ALL                = ProjectionIndex::ALL (all 14 bits)
 *   GNN_MINIMAL        = NODES | NODE_LABEL | LABEL_NODE | FROM_TO_EDGE | TO_FROM_EDGE
 *                        (+ property indexes at build time if configured)
 *   READONLY_TRAVERSAL = GNN_MINIMAL | EDGE_LABEL | LABEL_EDGE
 *
 * Property indexes (NODE_KEY_VALUE, KEY_VALUE_NODE, EDGE_KEY_VALUE, KEY_VALUE_EDGE)
 * are NOT controlled by IndexSet; they're already conditional on property
 * configuration in the relProjection / nodeProjection maps.
 */
enum class IndexSet : uint8_t {
    ALL                = 0,  // All 14 indexes (default, current behavior)
    GNN_MINIMAL        = 1,  // Only indexes needed for GNN k-hop sampling
    READONLY_TRAVERSAL = 2,  // GNN_MINIMAL + label indexes (no edge-id lookups)
};

/**
 * @brief Returns the ProjectionIndex bitmask corresponding to a preset.
 *
 * Property indexes are NOT controlled by IndexSet; they're already conditional
 * on property configuration in the relProjection / nodeProjection maps.
 */
ProjectionIndex project_index_mask_for(IndexSet preset) noexcept;

/**
 * @brief Parses a user-supplied string into an IndexSet value.
 *
 * Matches "ALL", "GNN_MINIMAL", "READONLY_TRAVERSAL" exactly (case-sensitive).
 *
 * @throws std::invalid_argument if the string is unrecognized, with a message
 *         listing valid values.
 */
IndexSet parse_index_set(const std::string& s);

/**
 * @brief Returns the canonical string name for an IndexSet value.
 */
const char* index_set_name(IndexSet preset) noexcept;

} // namespace GQL
