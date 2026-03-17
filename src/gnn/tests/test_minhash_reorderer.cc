#include "test_helpers.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "gnn/sampling/minhash_reorderer.h"

using namespace mdb::gnn;

// ===========================================================================
// Config validation
// ===========================================================================

TEST(MinHashConfigTest, ZeroHashesThrows) {
    EXPECT_THROW(
        MinHashReorderer({.num_hashes = 0}),
        std::invalid_argument
    );
}

TEST(MinHashConfigTest, ZeroHashesPerPassThrows) {
    EXPECT_THROW(
        MinHashReorderer({.strategy = MinHashReorderer::Strategy::MULTIPASS_BOUNDED,
                          .num_hashes = 2, .hashes_per_pass = 0}),
        std::invalid_argument
    );
}

TEST(MinHashConfigTest, ValidConfigAccepted) {
    EXPECT_NO_THROW(MinHashReorderer({.num_hashes = 2}));
    EXPECT_NO_THROW(MinHashReorderer({.num_hashes = 128, .segment_size = 50}));
}

// ===========================================================================
// compute_inverse
// ===========================================================================

TEST(MinHashInverseTest, SmallPermutation) {
    std::vector<uint64_t> perm = {2, 0, 3, 1};
    auto inv = MinHashReorderer::compute_inverse(perm);
    EXPECT_EQ(inv[0], 1u);
    EXPECT_EQ(inv[1], 3u);
    EXPECT_EQ(inv[2], 0u);
    EXPECT_EQ(inv[3], 2u);
}

TEST(MinHashInverseTest, Roundtrip) {
    std::vector<uint64_t> perm = {4, 2, 0, 3, 1};
    auto inv = MinHashReorderer::compute_inverse(perm);
    for (size_t i = 0; i < perm.size(); ++i) {
        EXPECT_EQ(inv[perm[i]], i) << "roundtrip failed at i=" << i;
    }
}

TEST(MinHashInverseTest, Identity) {
    std::vector<uint64_t> perm = {0, 1, 2, 3, 4};
    auto inv = MinHashReorderer::compute_inverse(perm);
    for (size_t i = 0; i < perm.size(); ++i) {
        EXPECT_EQ(inv[i], i);
    }
}

// ===========================================================================
// Strategy A: Segmented (DiskGNN Algorithm 1)
// ===========================================================================

TEST(MinHashSegmentedTest, BuildNotCalledThrows) {
    MinHashReorderer r({.num_hashes = 2});
    EXPECT_THROW(r.compute_permutation(10), std::runtime_error);
}

TEST(MinHashSegmentedTest, BuildCalledTwiceThrows) {
    MinHashReorderer r({.num_hashes = 2});
    auto provider = [](uint64_t) { return std::vector<uint64_t>{0, 1}; };
    r.build_access_graph(1, provider);
    EXPECT_THROW(r.build_access_graph(1, provider), std::runtime_error);
}

TEST(MinHashSegmentedTest, TotalRowsTooSmallThrows) {
    MinHashReorderer r({.num_hashes = 2});
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{0, 5, 10}; });
    EXPECT_THROW(r.compute_permutation(5), std::invalid_argument);
}

TEST(MinHashSegmentedTest, ZeroBatchesIdentity) {
    MinHashReorderer r({.num_hashes = 2});
    r.build_access_graph(0, [](uint64_t) { return std::vector<uint64_t>{}; });
    auto perm = r.compute_permutation(5);
    EXPECT_EQ(perm.size(), 5u);
    for (uint64_t i = 0; i < 5; ++i) {
        EXPECT_EQ(perm[i], i);
    }
}

TEST(MinHashSegmentedTest, SingleBatchAllAccessed) {
    MinHashReorderer r({.num_hashes = 2});
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{0, 1, 2, 3, 4}; });
    auto perm = r.compute_permutation(5);
    EXPECT_EQ(perm.size(), 5u);

    std::unordered_set<uint64_t> seen(perm.begin(), perm.end());
    EXPECT_EQ(seen.size(), 5u);
    for (auto v : perm) EXPECT_LT(v, 5u);
}

TEST(MinHashSegmentedTest, UnaccessedNodesAtEnd) {
    MinHashReorderer r({.num_hashes = 2});
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{0, 2, 4}; });
    auto perm = r.compute_permutation(6);

    EXPECT_EQ(perm.size(), 6u);

    std::unordered_set<uint64_t> first3(perm.begin(), perm.begin() + 3);
    EXPECT_TRUE(first3.count(0));
    EXPECT_TRUE(first3.count(2));
    EXPECT_TRUE(first3.count(4));

    std::unordered_set<uint64_t> last3(perm.begin() + 3, perm.end());
    EXPECT_TRUE(last3.count(1));
    EXPECT_TRUE(last3.count(3));
    EXPECT_TRUE(last3.count(5));
}

TEST(MinHashSegmentedTest, PermutationIsValidBijection) {
    const uint64_t N = 100;
    MinHashReorderer r({.num_hashes = 2});

    r.build_access_graph(10, [N](uint64_t batch_id) {
        std::vector<uint64_t> rows;
        for (uint64_t i = 0; i < 20; ++i) {
            rows.push_back((batch_id * 7 + i * 13) % N);
        }
        return rows;
    });

    auto perm = r.compute_permutation(N);
    EXPECT_EQ(perm.size(), N);

    std::vector<uint64_t> sorted_perm = perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_EQ(sorted_perm[i], i) << "Missing row " << i;
    }
}

TEST(MinHashSegmentedTest, DeterministicWithSameSeed) {
    auto make = []() {
        MinHashReorderer r({.num_hashes = 2, .random_seed = 42});
        r.build_access_graph(5, [](uint64_t b) {
            return std::vector<uint64_t>{b * 3, b * 3 + 1, b * 3 + 2};
        });
        return r.compute_permutation(20);
    };

    EXPECT_EQ(make(), make());
}

TEST(MinHashSegmentedTest, DifferentSeedDifferentResult) {
    auto make = [](uint64_t seed) {
        MinHashReorderer r({.num_hashes = 2, .random_seed = seed});
        r.build_access_graph(10, [](uint64_t b) {
            return std::vector<uint64_t>{b * 2, b * 2 + 1};
        });
        return r.compute_permutation(30);
    };

    EXPECT_NE(make(42), make(123));
}

TEST(MinHashSegmentedTest, SimilarNodesGrouped) {
    MinHashReorderer r({.num_hashes = 4, .segment_size = 0});

    r.build_access_graph(10, [](uint64_t batch_id) -> std::vector<uint64_t> {
        if (batch_id < 5) return {0, 1, 2, 6, 7, 8}; // group A
        else              return {3, 4, 5};             // group B
    });

    auto perm = r.compute_permutation(9);
    auto inv = MinHashReorderer::compute_inverse(perm);

    std::vector<uint64_t> group_a_pos = {inv[0], inv[1], inv[2], inv[6], inv[7], inv[8]};
    std::vector<uint64_t> group_b_pos = {inv[3], inv[4], inv[5]};

    std::sort(group_a_pos.begin(), group_a_pos.end());
    std::sort(group_b_pos.begin(), group_b_pos.end());

    uint64_t a_span = group_a_pos.back() - group_a_pos.front();
    uint64_t b_span = group_b_pos.back() - group_b_pos.front();
    EXPECT_EQ(a_span, 5u) << "Group A should be contiguous";
    EXPECT_EQ(b_span, 2u) << "Group B should be contiguous";
}

TEST(MinHashSegmentedTest, StatsAfterBuild) {
    MinHashReorderer r({.num_hashes = 2});
    r.build_access_graph(3, [](uint64_t b) {
        return std::vector<uint64_t>{b, b + 10};
    });
    auto stats = r.get_stats();
    EXPECT_EQ(stats.total_batches, 3u);
    EXPECT_EQ(stats.accessed_nodes, 6u);
}

TEST(MinHashSegmentedTest, LargeScale) {
    const uint64_t N = 10000;
    MinHashReorderer r({.num_hashes = 2, .segment_size = 50});

    r.build_access_graph(100, [N](uint64_t batch_id) {
        std::vector<uint64_t> rows;
        for (uint64_t i = 0; i < 200; ++i) {
            rows.push_back((batch_id * 97 + i * 31) % N);
        }
        return rows;
    });

    auto perm = r.compute_permutation(N);
    EXPECT_EQ(perm.size(), N);

    std::vector<uint64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (uint64_t i = 0; i < N; ++i) {
        EXPECT_EQ(sorted[i], i);
    }
}

// ===========================================================================
// Strategy B: Multi-pass Bounded
// ===========================================================================

TEST(MinHashMultipassTest, ValidBijection) {
    const uint64_t N = 200;
    MinHashReorderer r({.strategy = MinHashReorderer::Strategy::MULTIPASS_BOUNDED,
                         .num_hashes = 8, .hashes_per_pass = 4});

    r.build_access_graph(20, [N](uint64_t b) {
        std::vector<uint64_t> rows;
        for (uint64_t i = 0; i < 30; ++i) rows.push_back((b * 11 + i * 7) % N);
        return rows;
    });

    auto perm = r.compute_permutation(N);
    EXPECT_EQ(perm.size(), N);

    std::vector<uint64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (uint64_t i = 0; i < N; ++i) EXPECT_EQ(sorted[i], i);
}

TEST(MinHashMultipassTest, DeterministicWithSameSeed) {
    auto make = []() {
        MinHashReorderer r({.strategy = MinHashReorderer::Strategy::MULTIPASS_BOUNDED,
                             .num_hashes = 4, .hashes_per_pass = 2, .random_seed = 99});
        r.build_access_graph(5, [](uint64_t b) {
            return std::vector<uint64_t>{b, b + 10};
        });
        return r.compute_permutation(20);
    };
    EXPECT_EQ(make(), make());
}

TEST(MinHashMultipassTest, UnaccessedAtEnd) {
    MinHashReorderer r({.strategy = MinHashReorderer::Strategy::MULTIPASS_BOUNDED,
                         .num_hashes = 2, .hashes_per_pass = 1});
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{0, 2, 4}; });
    auto perm = r.compute_permutation(6);

    std::unordered_set<uint64_t> last3(perm.begin() + 3, perm.end());
    EXPECT_TRUE(last3.count(1));
    EXPECT_TRUE(last3.count(3));
    EXPECT_TRUE(last3.count(5));
}

TEST(MinHashMultipassTest, BothStrategiesProduceValidBijections) {
    auto make_provider = [](uint64_t batch_id) -> std::vector<uint64_t> {
        std::vector<uint64_t> rows;
        for (uint64_t i = 0; i < 10; ++i) {
            rows.push_back((batch_id * 7 + i * 3) % 50);
        }
        return rows;
    };

    MinHashReorderer rA({.strategy = MinHashReorderer::Strategy::SEGMENTED,
                          .num_hashes = 4, .segment_size = 0, .random_seed = 42});
    rA.build_access_graph(20, make_provider);
    auto permA = rA.compute_permutation(50);

    MinHashReorderer rB({.strategy = MinHashReorderer::Strategy::MULTIPASS_BOUNDED,
                          .num_hashes = 4, .hashes_per_pass = 2, .random_seed = 42});
    rB.build_access_graph(20, make_provider);
    auto permB = rB.compute_permutation(50);

    for (auto& perm : {permA, permB}) {
        auto sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        for (uint64_t i = 0; i < 50; ++i) {
            EXPECT_EQ(sorted[i], i);
        }
    }
}

// ===========================================================================
// End-to-end with FeatureMatrix
// ===========================================================================

using MinHashE2ETest = GnnStorageTest;

TEST_F(MinHashE2ETest, EndToEndReorderAndVerify) {
    const uint64_t N = 20, D = 3;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i);

    auto fmat_path = test_path("e2e.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    MinHashReorderer reorderer({.num_hashes = 4, .segment_size = 0});
    reorderer.build_access_graph(5, [](uint64_t b) -> std::vector<uint64_t> {
        return {b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3};
    });
    auto perm = reorderer.compute_permutation(N);
    auto inv = MinHashReorderer::compute_inverse(perm);

    auto reordered_path = test_path("e2e_reordered.fmat");
    auto reordered = FeatureMatrix::create_reordered(fm, perm, reordered_path);

    EXPECT_EQ(reordered.num_rows(), N);
    EXPECT_EQ(reordered.num_cols(), D);

    for (uint64_t old_row = 0; old_row < N; ++old_row) {
        uint64_t new_pos = inv[old_row];
        const float* original = fm.row_as<float>(old_row);
        const float* reordered_row = reordered.row_as<float>(new_pos);
        for (uint64_t c = 0; c < D; ++c) {
            EXPECT_FLOAT_EQ(original[c], reordered_row[c])
                << "old_row=" << old_row << " new_pos=" << new_pos << " col=" << c;
        }
    }
}
