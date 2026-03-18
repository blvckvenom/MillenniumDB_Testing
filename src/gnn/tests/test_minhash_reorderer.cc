#include "test_helpers.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "gnn/sampling/minhash_reorderer.h"

using namespace mdb::gnn;

// Helper to build a Config without C++20 designated initializers
static MinHashReorderer::Config make_config(
    MinHashReorderer::Strategy strategy = MinHashReorderer::Strategy::SEGMENTED,
    uint32_t num_hashes = 2,
    uint32_t hashes_per_pass = 8,
    uint32_t segment_size = 100,
    uint64_t random_seed = 42)
{
    MinHashReorderer::Config c;
    c.strategy = strategy;
    c.num_hashes = num_hashes;
    c.hashes_per_pass = hashes_per_pass;
    c.segment_size = segment_size;
    c.random_seed = random_seed;
    return c;
}

// ===========================================================================
// Config validation
// ===========================================================================

TEST(MinHashConfigTest, ZeroHashesThrows) {
    EXPECT_THROW(MinHashReorderer(make_config(MinHashReorderer::Strategy::SEGMENTED, 0)),
                 std::invalid_argument);
}

TEST(MinHashConfigTest, ZeroHashesPerPassThrowsForMultipass) {
    EXPECT_THROW(
        MinHashReorderer(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 2, 0)),
        std::invalid_argument);
}

TEST(MinHashConfigTest, ZeroHashesPerPassOkForSegmented) {
    // hashes_per_pass is irrelevant for SEGMENTED — should not throw
    EXPECT_NO_THROW(MinHashReorderer(make_config(MinHashReorderer::Strategy::SEGMENTED, 2, 0)));
}

TEST(MinHashConfigTest, ValidConfigAccepted) {
    EXPECT_NO_THROW(MinHashReorderer(make_config()));
    EXPECT_NO_THROW(MinHashReorderer(make_config(MinHashReorderer::Strategy::SEGMENTED, 128, 8, 50)));
}

TEST(MinHashConfigTest, StatsBeforeBuildThrows) {
    MinHashReorderer r(make_config());
    EXPECT_THROW(r.get_stats(), std::runtime_error);
}

// ===========================================================================
// Deterministic expected values (golden test)
// ===========================================================================

// Fix C4: Hardcoded expected values — catches any change to hash function,
// prime, sort order, or composite key layout.
// Generated with: seed=42, prime=4294967291, num_hashes=2
// To update after intentional algorithm changes: run the test, capture actual,
// replace the expected vector below.
TEST(MinHashGoldenTest, SegmentedKnownPermutation) {
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::SEGMENTED, 2, 8, 0, 42));

    r.build_access_graph(3, [](uint64_t b) -> std::vector<uint64_t> {
        if (b == 0) return {0, 1, 2};
        if (b == 1) return {1, 2, 3};
        return {3, 4, 0};
    });

    auto perm = r.compute_permutation(5);

    // HARDCODED expected: captured from verified run (2026-03-17)
    std::vector<uint64_t> expected = {0, 1, 2, 3, 4};
    EXPECT_EQ(perm, expected)
        << "Segmented permutation changed — hash function, prime, or sort order modified?";
}

TEST(MinHashGoldenTest, MultipassKnownPermutation) {
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 2, 1, 0, 42));

    r.build_access_graph(3, [](uint64_t b) -> std::vector<uint64_t> {
        if (b == 0) return {0, 1, 2};
        if (b == 1) return {1, 2, 3};
        return {3, 4, 0};
    });

    auto perm = r.compute_permutation(5);

    // HARDCODED expected: captured from verified run (2026-03-17)
    std::vector<uint64_t> expected = {0, 3, 4, 1, 2};
    EXPECT_EQ(perm, expected)
        << "Multipass permutation changed — hash function, fingerprint mixing, or sort modified?";
}

// Fix C5: Verify the hash function actually differentiates nodes.
// Each node has a UNIQUE batch membership set → each gets a different hash value
// → the permutation must NOT be identity.
TEST(MinHashGoldenTest, HashProducesNonTrivialReordering) {
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::SEGMENTED, 4, 8, 0, 42));

    // 10 batches, each accessing a DIFFERENT subset of 8 nodes.
    // Node 0: batches {0,1}, Node 1: batches {2,3}, Node 2: batches {4,5}, etc.
    // This ensures each node has a unique batch set → unique hash → non-trivial ordering.
    r.build_access_graph(10, [](uint64_t b) -> std::vector<uint64_t> {
        // Each batch accesses one node (its index / 2) plus a shared node 7
        return {b / 2, 7};
    });

    auto perm = r.compute_permutation(8);

    // With unique batch memberships per node, the permutation MUST differ from identity
    std::vector<uint64_t> identity = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_NE(perm, identity)
        << "Permutation is identity — hash function may not be differentiating nodes";

    // Verify it's still a valid bijection
    auto sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (uint64_t i = 0; i < 8; ++i) {
        EXPECT_EQ(sorted[i], i);
    }
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

TEST(MinHashInverseTest, Empty) {
    auto inv = MinHashReorderer::compute_inverse({});
    EXPECT_TRUE(inv.empty());
}

// ===========================================================================
// Strategy A: Segmented (DiskGNN Algorithm 1)
// ===========================================================================

TEST(MinHashSegmentedTest, BuildNotCalledThrows) {
    MinHashReorderer r(make_config());
    EXPECT_THROW(r.compute_permutation(10), std::runtime_error);
}

TEST(MinHashSegmentedTest, BuildCalledTwiceThrows) {
    MinHashReorderer r(make_config());
    auto provider = [](uint64_t) { return std::vector<uint64_t>{0, 1}; };
    r.build_access_graph(1, provider);
    EXPECT_THROW(r.build_access_graph(1, provider), std::runtime_error);
}

TEST(MinHashSegmentedTest, TotalRowsTooSmallThrows) {
    MinHashReorderer r(make_config());
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{0, 5, 10}; });
    EXPECT_THROW(r.compute_permutation(5), std::invalid_argument);
}

TEST(MinHashSegmentedTest, ZeroBatchesIdentity) {
    MinHashReorderer r(make_config());
    r.build_access_graph(0, [](uint64_t) { return std::vector<uint64_t>{}; });
    auto perm = r.compute_permutation(5);
    EXPECT_EQ(perm.size(), 5u);
    for (uint64_t i = 0; i < 5; ++i) {
        EXPECT_EQ(perm[i], i);
    }
}

TEST(MinHashSegmentedTest, SingleBatchAllAccessed) {
    MinHashReorderer r(make_config());
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{0, 1, 2, 3, 4}; });
    auto perm = r.compute_permutation(5);
    EXPECT_EQ(perm.size(), 5u);

    std::unordered_set<uint64_t> seen(perm.begin(), perm.end());
    EXPECT_EQ(seen.size(), 5u);
    for (auto v : perm) EXPECT_LT(v, 5u);
}

TEST(MinHashSegmentedTest, UnaccessedNodesAtEnd) {
    MinHashReorderer r(make_config());
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
    MinHashReorderer r(make_config());

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
        MinHashReorderer r(make_config(MinHashReorderer::Strategy::SEGMENTED, 2, 8, 100, 42));
        r.build_access_graph(5, [](uint64_t b) {
            return std::vector<uint64_t>{b * 3, b * 3 + 1, b * 3 + 2};
        });
        return r.compute_permutation(20);
    };

    EXPECT_EQ(make(), make());
}

TEST(MinHashSegmentedTest, DifferentSeedDifferentResult) {
    auto make = [](uint64_t seed) {
        MinHashReorderer r(make_config(MinHashReorderer::Strategy::SEGMENTED, 2, 8, 100, seed));
        r.build_access_graph(10, [](uint64_t b) {
            return std::vector<uint64_t>{b * 2, b * 2 + 1};
        });
        return r.compute_permutation(30);
    };

    EXPECT_NE(make(42), make(123));
}

// Fix #10: Relaxed assertion — allows minor interleaving due to hash collisions
TEST(MinHashSegmentedTest, SimilarNodesGrouped) {
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::SEGMENTED, 4, 8, 0, 42));

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

    // Groups should be mostly contiguous (allow +1 for potential hash collision interleaving)
    EXPECT_LE(a_span, 6u) << "Group A nodes should be mostly contiguous";
    EXPECT_LE(b_span, 3u) << "Group B nodes should be mostly contiguous";
}

TEST(MinHashSegmentedTest, StatsAfterBuild) {
    MinHashReorderer r(make_config());
    r.build_access_graph(3, [](uint64_t b) {
        return std::vector<uint64_t>{b, b + 10};
    });
    auto stats = r.get_stats();
    EXPECT_EQ(stats.total_batches, 3u);
    EXPECT_EQ(stats.accessed_nodes, 6u); // {0,10,1,11,2,12}
    EXPECT_EQ(stats.total_accesses, 6u); // 3 batches × 2 nodes each
    EXPECT_DOUBLE_EQ(stats.avg_batches_per_node, 1.0); // 6 accesses / 6 nodes
}

// Fix #14: Duplicate row IDs in a batch are harmless
TEST(MinHashSegmentedTest, DuplicateRowIdsHarmless) {
    MinHashReorderer r(make_config());
    // Batch has duplicate row IDs
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{0, 1, 0, 1, 0}; });
    auto perm = r.compute_permutation(3);
    EXPECT_EQ(perm.size(), 3u);

    // Must be valid bijection
    std::vector<uint64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted[0], 0u);
    EXPECT_EQ(sorted[1], 1u);
    EXPECT_EQ(sorted[2], 2u);
}

TEST(MinHashSegmentedTest, LargeScale) {
    const uint64_t N = 10000;
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::SEGMENTED, 2, 8, 50));

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
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 8, 4));

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
        MinHashReorderer r(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 4, 2, 100, 99));
        r.build_access_graph(5, [](uint64_t b) {
            return std::vector<uint64_t>{b, b + 10};
        });
        return r.compute_permutation(20);
    };
    EXPECT_EQ(make(), make());
}

TEST(MinHashMultipassTest, DifferentSeedDifferentResult) {
    auto make = [](uint64_t seed) {
        MinHashReorderer r(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 4, 2, 100, seed));
        r.build_access_graph(10, [](uint64_t b) {
            return std::vector<uint64_t>{b * 2, b * 2 + 1};
        });
        return r.compute_permutation(30);
    };
    EXPECT_NE(make(42), make(123));
}

TEST(MinHashMultipassTest, UnaccessedAtEnd) {
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 2, 1));
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{0, 2, 4}; });
    auto perm = r.compute_permutation(6);

    std::unordered_set<uint64_t> last3(perm.begin() + 3, perm.end());
    EXPECT_TRUE(last3.count(1));
    EXPECT_TRUE(last3.count(3));
    EXPECT_TRUE(last3.count(5));
}

TEST(MinHashMultipassTest, StatsAfterBuild) {
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 2, 1));
    r.build_access_graph(3, [](uint64_t b) {
        return std::vector<uint64_t>{b, b + 10};
    });
    auto stats = r.get_stats();
    EXPECT_EQ(stats.total_batches, 3u);
    EXPECT_EQ(stats.accessed_nodes, 6u);
    EXPECT_EQ(stats.total_accesses, 6u);
}

// Fix #11: Strategy B LargeScale — exercises temp file I/O at scale
TEST(MinHashMultipassTest, LargeScale) {
    const uint64_t N = 5000;
    MinHashReorderer r(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 4, 2));

    r.build_access_graph(50, [N](uint64_t batch_id) {
        std::vector<uint64_t> rows;
        for (uint64_t i = 0; i < 100; ++i) {
            rows.push_back((batch_id * 97 + i * 31) % N);
        }
        return rows;
    });

    auto perm = r.compute_permutation(N);
    EXPECT_EQ(perm.size(), N);

    std::vector<uint64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (uint64_t i = 0; i < N; ++i) EXPECT_EQ(sorted[i], i);
}

TEST(MinHashMultipassTest, BothStrategiesProduceValidBijections) {
    auto make_provider = [](uint64_t batch_id) -> std::vector<uint64_t> {
        std::vector<uint64_t> rows;
        for (uint64_t i = 0; i < 10; ++i) {
            rows.push_back((batch_id * 7 + i * 3) % 50);
        }
        return rows;
    };

    MinHashReorderer rA(make_config(MinHashReorderer::Strategy::SEGMENTED, 4, 8, 0, 42));
    rA.build_access_graph(20, make_provider);
    auto permA = rA.compute_permutation(50);

    MinHashReorderer rB(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 4, 2, 100, 42));
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

// Strategy B: segment_size should be ignored
TEST(MinHashMultipassTest, SegmentSizeIgnored) {
    auto make = [](uint32_t seg_size) {
        MinHashReorderer r(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 4, 2, seg_size, 42));
        r.build_access_graph(10, [](uint64_t b) {
            return std::vector<uint64_t>{b * 2, b * 2 + 1};
        });
        return r.compute_permutation(30);
    };

    EXPECT_EQ(make(0), make(50));
    EXPECT_EQ(make(50), make(200));
}

// ===========================================================================
// End-to-end with FeatureMatrix
// ===========================================================================

using MinHashE2ETest = GnnStorageTest;

// Edge case: single node accessed, rest unaccessed
TEST(MinHashSegmentedTest, SingleNodeAccessed) {
    MinHashReorderer r(make_config());
    r.build_access_graph(1, [](uint64_t) { return std::vector<uint64_t>{5}; });
    auto perm = r.compute_permutation(10);
    EXPECT_EQ(perm.size(), 10u);
    EXPECT_EQ(perm[0], 5u); // only accessed node goes first

    std::vector<uint64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (uint64_t i = 0; i < 10; ++i) EXPECT_EQ(sorted[i], i);
}

TEST_F(MinHashE2ETest, EndToEndReorderAndVerify) {
    const uint64_t N = 20, D = 3;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i);

    auto fmat_path = test_path("e2e.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    MinHashReorderer reorderer(make_config(MinHashReorderer::Strategy::SEGMENTED, 4, 8, 0));
    reorderer.build_access_graph(5, [](uint64_t b) -> std::vector<uint64_t> {
        return {b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3};
    });
    auto perm = reorderer.compute_permutation(N);
    auto inv = MinHashReorderer::compute_inverse(perm);

    auto reordered_path = test_path("e2e_reordered.fmat");
    auto reordered = FeatureMatrix::create_reordered(fm, perm, reordered_path);

    EXPECT_EQ(reordered.num_rows(), N);
    EXPECT_EQ(reordered.num_cols(), D);

    // Verify: every original row's features are preserved at the new position
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

TEST_F(MinHashE2ETest, EndToEndReorderStrategyB) {
    const uint64_t N = 20, D = 3;
    std::vector<float> features(N * D);
    for (uint64_t i = 0; i < N * D; ++i) features[i] = static_cast<float>(i);

    auto fmat_path = test_path("e2e_b.fmat");
    auto fm = FeatureMatrix::create(fmat_path, N, D, GnnDtype::FLOAT32, features.data());

    MinHashReorderer reorderer(make_config(MinHashReorderer::Strategy::MULTIPASS_BOUNDED, 4, 2, 0));
    reorderer.build_access_graph(5, [](uint64_t b) -> std::vector<uint64_t> {
        return {b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3};
    });
    auto perm = reorderer.compute_permutation(N);
    auto inv = MinHashReorderer::compute_inverse(perm);

    auto reordered_path = test_path("e2e_b_reordered.fmat");
    auto reordered = FeatureMatrix::create_reordered(fm, perm, reordered_path);

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
