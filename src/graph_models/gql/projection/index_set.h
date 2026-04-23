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

// Compile-time drift guard: if ProjectionIndex::ALL changes (e.g., a new bit
// added), this fails to compile and forces re-audit of the IndexSet presets
// below. Mirrors the assertion already in native_projection_builder.h.
static_assert(
    static_cast<uint32_t>(ProjectionIndex::ALL) == 0x3FFFu,
    "ProjectionIndex::ALL changed — re-audit IndexSet preset definitions");

/**
 * @brief Returns the ProjectionIndex bitmask corresponding to a preset.
 *
 * Property indexes are NOT controlled by IndexSet; they're already conditional
 * on property configuration in the relProjection / nodeProjection maps.
 *
 * For valid enum values, returns the preset's bitmask. For out-of-range casts
 * (e.g., corrupt catalog reading static_cast<IndexSet>(99)), asserts in debug
 * builds and returns ProjectionIndex::NONE in release — visibly wrong rather
 * than silently defaulting to ALL.
 *
 * Note: not marked constexpr because the assert(false) path is not a constant
 * expression pre-C++23; visibility of the bug in debug builds was prioritized
 * over constexpr-ability.
 */
ProjectionIndex project_index_mask_for(IndexSet preset) noexcept;

/**
 * @brief Parses a user-supplied string into an IndexSet value.
 *
 * Matches "ALL", "GNN_MINIMAL", "READONLY_TRAVERSAL" exactly (case-sensitive).
 *
 * @throws QueryException if the string is unrecognized, with a message
 *         listing valid values. QueryException is used (rather than
 *         std::invalid_argument) to match the project convention for
 *         config-validation errors surfaced to GQL clients.
 */
IndexSet parse_index_set(const std::string& s);

/**
 * @brief Returns the canonical string name for an IndexSet value.
 *
 * For out-of-range casts, asserts in debug builds and returns "UNKNOWN" in
 * release (still visibly distinct from any valid preset name).
 *
 * Note: not marked constexpr, see project_index_mask_for for rationale.
 */
const char* index_set_name(IndexSet preset) noexcept;

} // namespace GQL
