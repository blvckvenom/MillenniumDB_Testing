#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "gnn/training/feature_label_integrity.h"

using namespace mdb::gnn;

namespace {

// Build M class-separable feature vectors: each class has a fixed random base
// vector; a node of class c = base[c] + small noise. labels[i] = class of node i.
struct Synthetic {
    std::vector<float>   X;       // [M, D]
    std::vector<int64_t> y;       // [M]
    size_t M, D;
    int64_t K;
};

Synthetic make_separable(size_t M, size_t D, int64_t K, double noise, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> base_d(0.0, 1.0);
    std::normal_distribution<double> noise_d(0.0, noise);
    std::vector<std::vector<float>> base(K, std::vector<float>(D));
    for (int64_t c = 0; c < K; ++c)
        for (size_t d = 0; d < D; ++d) base[c][d] = static_cast<float>(base_d(rng));

    Synthetic s; s.M = M; s.D = D; s.K = K;
    s.X.resize(M * D); s.y.resize(M);
    for (size_t i = 0; i < M; ++i) {
        int64_t c = static_cast<int64_t>(i % static_cast<size_t>(K));
        s.y[i] = c;
        for (size_t d = 0; d < D; ++d)
            s.X[i * D + d] = base[c][d] + static_cast<float>(noise_d(rng));
    }
    return s;
}

} // namespace

TEST(FeatureLabelIntegrity, AlignedSeparableFeaturesPass) {
    auto s = make_separable(/*M=*/2000, /*D=*/16, /*K=*/8, /*noise=*/0.15, /*seed=*/42);
    auto r = FeatureLabelIntegrity::check(s.X.data(), s.M, s.D, s.y.data(), s.K);
    EXPECT_TRUE(r.ran);
    EXPECT_TRUE(r.passed);
    EXPECT_GT(r.accuracy, r.threshold);
    EXPECT_GT(r.accuracy, 0.5);          // separable -> centroid recovers class easily
    EXPECT_NEAR(r.chance, 1.0 / 8.0, 1e-9);
}

TEST(FeatureLabelIntegrity, MisalignedShuffledLabelsFail) {
    auto s = make_separable(2000, 16, 8, 0.15, 42);
    // Shuffle labels => features no longer predict labels (the misalignment bug).
    std::mt19937_64 rng(7);
    std::shuffle(s.y.begin(), s.y.end(), rng);
    auto r = FeatureLabelIntegrity::check(s.X.data(), s.M, s.D, s.y.data(), s.K);
    EXPECT_TRUE(r.ran);
    EXPECT_FALSE(r.passed);              // accuracy ~ chance < 2x chance threshold
    EXPECT_LT(r.accuracy, r.threshold);
    EXPECT_LT(r.accuracy, 0.30);
}

TEST(FeatureLabelIntegrity, SkipsBelowMinSamples) {
    auto s = make_separable(50, 16, 8, 0.15, 42);   // < min_samples (200)
    auto r = FeatureLabelIntegrity::check(s.X.data(), s.M, s.D, s.y.data(), s.K);
    EXPECT_FALSE(r.ran);
    EXPECT_TRUE(r.passed);               // skipped -> never blocks
}

TEST(FeatureLabelIntegrity, DegenerateInputsSkip) {
    std::vector<float> X(10 * 4, 0.0f);
    std::vector<int64_t> y(10, 0);
    // <2 classes
    auto r1 = FeatureLabelIntegrity::check(X.data(), 10, 4, y.data(), /*K=*/1);
    EXPECT_FALSE(r1.ran); EXPECT_TRUE(r1.passed);
    // D == 0
    auto r2 = FeatureLabelIntegrity::check(X.data(), 10, 0, y.data(), /*K=*/8);
    EXPECT_FALSE(r2.ran); EXPECT_TRUE(r2.passed);
}
