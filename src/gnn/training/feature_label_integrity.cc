#include "gnn/training/feature_label_integrity.h"

#include <cmath>
#include <vector>

namespace mdb::gnn {

FeatureLabelIntegrityResult FeatureLabelIntegrity::check(
    const float*   X,
    size_t         M,
    size_t         D,
    const int64_t* y,
    int64_t        num_classes,
    double         threshold_mult,
    size_t         min_samples)
{
    FeatureLabelIntegrityResult r;
    r.chance     = (num_classes > 0) ? 1.0 / static_cast<double>(num_classes) : 0.0;
    r.threshold  = threshold_mult * r.chance;

    if (num_classes < 2 || D == 0 || M < min_samples) {
        r.ran = false;
        r.passed = true;  // not enough signal to judge — do not block
        return r;
    }

    const size_t K = static_cast<size_t>(num_classes);

    // Per-class centroid (mean feature vector) accumulated in double for stability.
    std::vector<double>   cent(K * D, 0.0);
    std::vector<uint64_t> cnt(K, 0);
    for (size_t i = 0; i < M; ++i) {
        const int64_t c = y[i];
        if (c < 0 || c >= num_classes) continue;
        const float* x = X + i * D;
        double* cd = cent.data() + static_cast<size_t>(c) * D;
        for (size_t d = 0; d < D; ++d) cd[d] += static_cast<double>(x[d]);
        cnt[static_cast<size_t>(c)]++;
    }

    // Mean, then L2-normalize each populated centroid (so argmax dot == cosine NN).
    for (size_t c = 0; c < K; ++c) {
        if (cnt[c] == 0) continue;
        double* cd = cent.data() + c * D;
        double nrm = 0.0;
        for (size_t d = 0; d < D; ++d) { cd[d] /= static_cast<double>(cnt[c]); nrm += cd[d] * cd[d]; }
        nrm = std::sqrt(nrm);
        if (nrm > 1e-12) for (size_t d = 0; d < D; ++d) cd[d] /= nrm;
    }

    // Nearest-centroid prediction (cosine). ||x|| is constant across centroids for
    // a fixed node, so argmax over unit-centroid dot products is the cosine NN.
    size_t correct = 0, scored = 0;
    for (size_t i = 0; i < M; ++i) {
        const int64_t c = y[i];
        if (c < 0 || c >= num_classes) continue;
        const float* x = X + i * D;
        int64_t best = -1;
        double  best_sim = -1e300;
        for (size_t cc = 0; cc < K; ++cc) {
            if (cnt[cc] == 0) continue;
            const double* cd = cent.data() + cc * D;
            double dot = 0.0;
            for (size_t d = 0; d < D; ++d) dot += static_cast<double>(x[d]) * cd[d];
            if (dot > best_sim) { best_sim = dot; best = static_cast<int64_t>(cc); }
        }
        if (best == c) correct++;
        scored++;
    }

    r.num_scored = scored;
    r.accuracy   = scored ? static_cast<double>(correct) / static_cast<double>(scored) : 0.0;
    r.ran        = (scored >= min_samples);
    r.passed     = (!r.ran) || (r.accuracy >= r.threshold);
    return r;
}

} // namespace mdb::gnn
