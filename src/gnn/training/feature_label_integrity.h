#pragma once

#include <cstddef>
#include <cstdint>

namespace mdb::gnn {

/**
 * @brief Result of a feature<->label integrity check.
 *
 * `accuracy` is the nearest-class-centroid accuracy of a sample of labeled
 * nodes, computed from their features alone (no training). For an aligned
 * store this is well above `chance` (= 1/num_classes); for a store whose
 * feature rows are misaligned to node identity it collapses to ~chance.
 */
struct FeatureLabelIntegrityResult {
    double accuracy   = 0.0;  ///< nearest-centroid accuracy on the scored nodes
    double chance     = 0.0;  ///< 1 / num_classes
    double threshold  = 0.0;  ///< pass bar = threshold_mult * chance
    size_t num_scored = 0;    ///< number of labeled nodes actually scored
    bool   ran        = false;///< false if skipped (too few samples / <2 classes / D==0)
    bool   passed     = true; ///< accuracy >= threshold (true when skipped)
};

/**
 * @brief Cheap, training-free feature<->label alignment check.
 *
 * Catches the failure mode where a node's stored features do NOT correspond to
 * its label (e.g. a feature matrix whose rows were permuted/misordered relative
 * to node identity) — which lets a model "learn" yet caps accuracy near the
 * aligned fraction, INVARIANT to all graph/model/sampling choices. The 2026-06-02
 * papers100M bug (node_features.fmat[r] held the wrong node's features) is exactly
 * this class; a linear/centroid probe on the raw features dropped from ~0.44 to
 * ~chance. This is the source-independent guard for it.
 *
 * Method: fit per-class feature centroids (mean) on the sample, L2-normalize them,
 * predict each node's class as the nearest centroid (cosine), report accuracy.
 * Aligned => accuracy >> chance; misaligned => accuracy ~ chance.
 */
class FeatureLabelIntegrity {
public:
    /**
     * @param features      [M, D] float32 row-major features of M sampled nodes.
     * @param M             number of sampled nodes.
     * @param D             feature dimension.
     * @param labels        [M] int64 labels; entries <0 or >=num_classes are ignored.
     * @param num_classes   total class count.
     * @param threshold_mult pass bar multiplier on chance (default 2.0 — conservative;
     *                       a misaligned store sits at ~1x chance, an aligned one many x).
     * @param min_samples   skip the check (ran=false, passed=true) below this many scorable nodes.
     */
    static FeatureLabelIntegrityResult check(
        const float*   features,
        size_t         M,
        size_t         D,
        const int64_t* labels,
        int64_t        num_classes,
        double         threshold_mult = 2.0,
        size_t         min_samples    = 200);
};

} // namespace mdb::gnn
