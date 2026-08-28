#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_config.h"
#include "gnn/tests/test_helpers.h"
#include "graph_models/object_id.h"

namespace fs = std::filesystem;
using namespace mdb::gnn;

// Exercises the deserialized-sample LRU cache in SampleStorage (train hot path).
class SampleStorageCacheTest : public GnnStorageTest {
protected:
    fs::path db_folder_;
    static constexpr uint64_t NUM_BATCHES = 6;

    void SetUp() override {
        GnnStorageTest::SetUp();
        db_folder_ = test_dir_ / "db";
        fs::create_directories(db_folder_);

        SamplingConfig config;
        config.projection_name = "p";
        config.sample_name = "s";
        config.fanouts = {2};
        config.batch_size = 4;
        config.train_ratio = 1.0;
        config.val_ratio = 0.0;
        config.test_ratio = 0.0;

        auto storage = SampleStorage::create(db_folder_, config);
        for (uint64_t bid = 0; bid < NUM_BATCHES; ++bid) {
            GraphSample s;
            s.batch_id = bid;
            s.split = SplitType::TRAIN;
            s.nodes_per_layer.resize(2);
            // Distinct node sets per batch so cached==disk is meaningful.
            for (uint64_t k = 0; k < 4; ++k) {
                ObjectId seed(0xD400000000000000ULL | (bid * 100 + k));
                s.nodes_per_layer[0].push_back(seed);
                s.all_unique_nodes.push_back(seed);
            }
            for (uint64_t k = 0; k < 6; ++k) {
                ObjectId nb(0xD400000000000000ULL | (bid * 100 + 50 + k));
                s.nodes_per_layer[1].push_back(nb);
                s.all_unique_nodes.push_back(nb);
            }
            storage.write_sample(s);
        }
        storage.finalize();
    }

    SampleStorage open() {
        return SampleStorage::open(
            SampleStorage::get_storage_path(db_folder_, "s"));
    }
};

TEST_F(SampleStorageCacheTest, DisabledByDefault) {
    auto st = open();
    st.read_sample(0);
    st.read_sample(0);
    auto s = st.sample_cache_stats();
    EXPECT_EQ(s.budget, 0u);
    EXPECT_EQ(s.hits, 0u);
    EXPECT_EQ(s.misses, 0u);
    EXPECT_EQ(s.entries, 0u);
}

TEST_F(SampleStorageCacheTest, HitAfterMiss) {
    auto st = open();
    st.set_sample_cache_budget_bytes(64 * 1024 * 1024);  // ample
    st.read_sample(0);                  // miss -> insert
    st.read_sample(0);                  // hit
    st.read_sample(0);                  // hit
    auto s = st.sample_cache_stats();
    EXPECT_EQ(s.misses, 1u);
    EXPECT_EQ(s.hits, 2u);
    EXPECT_EQ(s.entries, 1u);
    EXPECT_GT(s.bytes, 0u);
}

TEST_F(SampleStorageCacheTest, CachedEqualsDisk) {
    auto st = open();
    GraphSample fresh = st.read_sample(3);          // disk
    st.set_sample_cache_budget_bytes(64 * 1024 * 1024);
    GraphSample miss = st.read_sample(3);           // disk + insert
    GraphSample hit  = st.read_sample(3);           // cache
    EXPECT_EQ(hit.batch_id, fresh.batch_id);
    ASSERT_EQ(hit.nodes_per_layer.size(), fresh.nodes_per_layer.size());
    ASSERT_EQ(hit.all_unique_nodes.size(), fresh.all_unique_nodes.size());
    for (size_t i = 0; i < hit.all_unique_nodes.size(); ++i) {
        EXPECT_EQ(hit.all_unique_nodes[i].id, fresh.all_unique_nodes[i].id);
    }
    EXPECT_EQ(miss.all_unique_nodes.size(), fresh.all_unique_nodes.size());
}

TEST_F(SampleStorageCacheTest, EvictsUnderTightBudget) {
    auto st = open();
    // Size the budget to roughly one sample so reading distinct batches evicts.
    st.set_sample_cache_budget_bytes(16 * 1024 * 1024);
    size_t one;
    {
        st.read_sample(0);
        one = st.sample_cache_stats().bytes;  // bytes of a single sample
    }
    ASSERT_GT(one, 0u);
    st.set_sample_cache_budget_bytes(one + one / 2);  // holds 1, not 2

    auto st2 = open();
    st2.set_sample_cache_budget_bytes(one + one / 2);
    for (uint64_t bid = 0; bid < NUM_BATCHES; ++bid) st2.read_sample(bid);
    auto s = st2.sample_cache_stats();
    EXPECT_LE(s.entries, 1u);          // budget holds at most one
    EXPECT_GT(s.evictions, 0u);        // and evicted the rest
    EXPECT_LE(s.bytes, one + one / 2); // never exceeds budget
}

TEST_F(SampleStorageCacheTest, DisableClearsCache) {
    auto st = open();
    st.set_sample_cache_budget_bytes(64 * 1024 * 1024);
    st.read_sample(0);
    st.read_sample(1);
    ASSERT_EQ(st.sample_cache_stats().entries, 2u);
    st.set_sample_cache_budget_bytes(0);
    auto s = st.sample_cache_stats();
    EXPECT_EQ(s.entries, 0u);
    EXPECT_EQ(s.bytes, 0u);
    EXPECT_EQ(s.budget, 0u);
}

// Commit/abort protocol of the write phase: only an explicit finalize()
// persists a valid catalog; destruction (or abort()) without finalize must
// discard the partial sample so re-runs don't hit "already exists" and
// readers cannot consume a truncated sample.
class SampleStorageCommitTest : public GnnStorageTest {
protected:
    fs::path db_folder_;

    void SetUp() override {
        GnnStorageTest::SetUp();
        db_folder_ = test_dir_ / "db_commit";
        fs::create_directories(db_folder_);
    }

    SamplingConfig make_config(const std::string& sample_name) {
        SamplingConfig config;
        config.projection_name = "p";
        config.sample_name = sample_name;
        config.fanouts = {2};
        config.batch_size = 4;
        config.train_ratio = 1.0;
        config.val_ratio = 0.0;
        config.test_ratio = 0.0;
        return config;
    }

    GraphSample make_sample(uint64_t bid) {
        GraphSample s;
        s.batch_id = bid;
        s.split = SplitType::TRAIN;
        s.nodes_per_layer.resize(2);
        ObjectId seed(0xD400000000000000ULL | (bid * 100));
        ObjectId nb(0xD400000000000000ULL | (bid * 100 + 50));
        s.nodes_per_layer[0].push_back(seed);
        s.nodes_per_layer[1].push_back(nb);
        s.all_unique_nodes.push_back(seed);
        s.all_unique_nodes.push_back(nb);
        return s;
    }
};

TEST_F(SampleStorageCommitTest, FinalizeCommitsValidSample) {
    auto config = make_config("s_commit");
    {
        auto storage = SampleStorage::create(db_folder_, config);
        storage.write_sample(make_sample(0));
        storage.finalize();
    }
    EXPECT_TRUE(SampleStorage::exists(db_folder_, "s_commit"));
    auto st = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "s_commit"));
    EXPECT_EQ(st.get_catalog().total_batches, 1u);
    EXPECT_EQ(st.read_sample(0).batch_id, 0u);
}

TEST_F(SampleStorageCommitTest, DestroyWithoutFinalizeDiscardsPartialSample) {
    auto config = make_config("s_partial");
    auto path = SampleStorage::get_storage_path(db_folder_, "s_partial");
    {
        auto storage = SampleStorage::create(db_folder_, config);
        storage.write_sample(make_sample(0));
        // No finalize(): simulates a failed/cancelled run unwinding.
    }
    EXPECT_FALSE(SampleStorage::exists(db_folder_, "s_partial"));
    EXPECT_FALSE(fs::exists(path));
    EXPECT_THROW(SampleStorage::open(path), std::runtime_error);

    // A re-run with the same sample_name must not fail with "already exists".
    auto retry = SampleStorage::create(db_folder_, config);
    retry.write_sample(make_sample(0));
    retry.finalize();
    EXPECT_TRUE(SampleStorage::exists(db_folder_, "s_partial"));
}

TEST_F(SampleStorageCommitTest, ExplicitAbortDiscardsPartialSample) {
    auto config = make_config("s_abort");
    auto path = SampleStorage::get_storage_path(db_folder_, "s_abort");
    auto storage = SampleStorage::create(db_folder_, config);
    storage.write_sample(make_sample(0));
    storage.abort();
    EXPECT_FALSE(SampleStorage::exists(db_folder_, "s_abort"));
    EXPECT_FALSE(fs::exists(path));
    EXPECT_FALSE(storage.is_write_mode());
    // Writes after abort are rejected.
    EXPECT_THROW(storage.write_sample(make_sample(1)), std::runtime_error);
    // Idempotent: a second abort (and the destructor) must not resurrect it.
    storage.abort();
    EXPECT_FALSE(fs::exists(path));
}

TEST_F(SampleStorageCommitTest, AbortAfterFinalizeIsNoOp) {
    auto config = make_config("s_committed");
    auto storage = SampleStorage::create(db_folder_, config);
    storage.write_sample(make_sample(0));
    storage.finalize();
    storage.abort();  // must not discard a committed sample
    EXPECT_TRUE(SampleStorage::exists(db_folder_, "s_committed"));
    auto st = SampleStorage::open(
        SampleStorage::get_storage_path(db_folder_, "s_committed"));
    EXPECT_EQ(st.get_catalog().total_batches, 1u);
}
