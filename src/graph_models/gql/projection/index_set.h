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

/**
 * @brief Returns the canonical string name for a single-bit ProjectionIndex
 *        value, matching the on-disk .leaf file naming (e.g. "edge_label"
 *        for ProjectionIndex::EDGE_LABEL, "from_to_edge" for FROM_TO_EDGE).
 *
 * Used by the query-layer error diagnostic (Spec #3 T3.9) to name the
 * missing index in QueryException messages. Only accepts single-bit
 * values — composite masks (NONE, ALL_NODE, ALL_EDGE, ALL) return
 * "UNKNOWN" because they cannot be mapped to a unique .leaf file.
 *
 * For out-of-range casts, asserts in debug builds and returns "UNKNOWN"
 * in release.
 */
const char* projection_index_name(ProjectionIndex which) noexcept;

/**
 * @brief Returns the LOWEST (most restrictive) IndexSet preset whose
 *        bitmask contains the given single-bit ProjectionIndex.
 *
 * Used by T3.9's query-layer diagnostic to suggest the minimum rebuild
 * required to unblock a query. For example:
 *   EDGE_LABEL, LABEL_EDGE              -> READONLY_TRAVERSAL
 *   NODES, FROM_TO_EDGE, TO_FROM_EDGE,
 *   NODE_LABEL, LABEL_NODE              -> GNN_MINIMAL
 *   EDGE_DIRECTION, EDGE_FROM_TO,
 *   EDGE_N1_N2                          -> ALL (no lower preset contains)
 *   NODE_KEY_VALUE, KEY_VALUE_NODE,
 *   EDGE_KEY_VALUE, KEY_VALUE_EDGE      -> ALL (property indexes are
 *                                               gated by Features flags,
 *                                               not IndexSet — callers
 *                                               still need ALL + the
 *                                               appropriate includeProperties
 *                                               config; see Spec #3 §3.4)
 *
 * For composite masks or out-of-range casts, asserts in debug builds and
 * returns IndexSet::ALL in release (safe fallback: ALL always contains
 * every single-bit index so is a correct — if conservative — suggestion).
 */
IndexSet minimum_preset_for(ProjectionIndex which) noexcept;

} // namespace GQL
