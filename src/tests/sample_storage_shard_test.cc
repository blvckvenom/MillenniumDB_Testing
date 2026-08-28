// sample_storage_shard_test.cc
//
// Golden compare for the sharded (lock-free) parallel write: writing the SAME
// samples via the legacy single-writer path and via the sharded path (M shards)
// must produce BYTE-IDENTICAL batches.dat / batches.idx / frequency.dat and
// equal catalog statistics — across any shard count. This is the correctness
// gate for removing the single write mutex from the parallel sampling loop.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sample_fingerprint.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_config.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/object_id.h"

namespace fs = std::filesystem;
using namespace mdb::gnn;

namespace {

SamplingConfig make_config(const std::string& name) {
    SamplingConfig config;
    config.projection_name = "shard_test_proj";
    config.sample_name = name;
    config.fanouts = {2};
    config.batch_size = 2;
    return config;
}

// A sample whose node set + edges vary by batch_id, so the frequency tally and
// content fingerprint are non-trivial. Uses node ids in [1, kNodes].
constexpr uint64_t kNodes = 64;

GraphSample make_sample(uint64_t batch_id, SplitType split) {
    GraphSample sample;
    sample.batch_id = batch_id;
    sample.split = split;
    // Two seeds + a couple of neighbors that rotate through the node space.
    const uint64_t a = 1 + (batch_id * 3) % kNodes;
    const uint64_t b = 1 + (batch_id * 3 + 1) % kNodes;
    const uint64_t c = 1 + (batch_id * 5 + 2) % kNodes;
    const uint64_t d = 1 + (batch_id * 7 + 4) % kNodes;
    sample.nodes_per_layer = {
        {ObjectId(a), ObjectId(b)},  // seeds
        {ObjectId(c), ObjectId(d)},  // layer 1
    };
    LayerEdges edges;
    edges.src_indices = {0, 1};
    edges.dst_indices = {0, 1};
    edges.edge_ids = {ObjectId(100 + batch_id), ObjectId(200 + batch_id)};
    sample.edges_per_layer = {edges};
    sample.rebuild_unique_nodes();
    return sample;
}

// Read a whole file into a byte string.
std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    EXPECT_TRUE(in.good()) << "cannot open " << p;
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

class ShardWriteTest : public ::testing::Test {
protected:
    fs::path tmp_;
    void SetUp() override {
        tmp_ = fs::temp_directory_path()
             / ("shard_write_" + std::to_string(::getpid()) + "_"
                + ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(tmp_);
        fs::create_directories(tmp_);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_, ec);
    }

    RowMapping make_row_mapping() {
        std::vector<ObjectId> ids;
        ids.reserve(kNodes);
        for (uint64_t i = 1; i <= kNodes; ++i) ids.push_back(ObjectId(i));
        return RowMapping::create(tmp_ / "nodes.rmap", ids);
    }

    // batch_ids 0..N-1 laid out train -> val -> test (the engine's ordering).
    std::vector<GraphSample> make_samples(uint64_t n_train, uint64_t n_val,
                                          uint64_t n_test) {
        std::vector<GraphSample> out;
        uint64_t bid = 0;
        for (uint64_t i = 0; i < n_train; ++i) out.push_back(make_sample(bid++, SplitType::TRAIN));
        for (uint64_t i = 0; i < n_val; ++i)   out.push_back(make_sample(bid++, SplitType::VALIDATION));
        for (uint64_t i = 0; i < n_test; ++i)  out.push_back(make_sample(bid++, SplitType::TEST));
        return out;
    }

    // Legacy single-writer, dense (RowMapping) path.
    fs::path write_legacy(const RowMapping& rm,
                          const std::vector<GraphSample>& samples) {
        auto config = make_config("legacy");
        auto storage = SampleStorage::create(tmp_, config, rm, kNodes + 1);
        for (const auto& s : samples) storage.write_sample(s);
        storage.finalize();
        return SampleStorage::get_storage_path(tmp_, config.sample_name);
    }

    // Sharded path with `num_workers` shards. Distributes batches round-robin to
    // workers, each worker writing its batches in ascending batch_id order (as
    // the real monotone next_idx dispatch does).
    fs::path write_sharded(const RowMapping& rm,
                           const std::vector<GraphSample>& samples,
                           uint32_t num_workers) {
        auto config = make_config("sharded");
        auto storage = SampleStorage::create(tmp_, config, rm, kNodes + 1);
        storage.begin_sharded_write(num_workers);
        EXPECT_TRUE(storage.sharded_write_active());
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto& s = samples[i];
            std::ostringstream buf(std::ios::binary);
            s.serialize(buf);
            storage.shard_write(static_cast<uint32_t>(i % num_workers), s,
                                buf.str(), compute_batch_content_hash(s));
        }
        storage.merge_shards();
        return SampleStorage::get_storage_path(tmp_, config.sample_name);
    }

    void expect_byte_identical(const fs::path& a, const fs::path& b) {
        for (const char* f : {"batches.dat", "batches.idx", "frequency.dat"}) {
            EXPECT_EQ(read_file(a / f), read_file(b / f))
                << "file differs: " << f;
        }
        auto sa = SampleStorage::open(a);
        auto sb = SampleStorage::open(b);
        const auto& ca = sa.get_catalog();
        const auto& cb = sb.get_catalog();
        EXPECT_EQ(ca.total_batches, cb.total_batches);
        EXPECT_EQ(ca.train_batches, cb.train_batches);
        EXPECT_EQ(ca.validation_batches, cb.validation_batches);
        EXPECT_EQ(ca.test_batches, cb.test_batches);
        EXPECT_EQ(ca.unique_nodes, cb.unique_nodes);
        EXPECT_EQ(ca.total_edges, cb.total_edges);
        EXPECT_EQ(ca.sample_content_fp, cb.sample_content_fp);
    }
};

TEST_F(ShardWriteTest, SingleShardEqualsLegacy) {
    auto rm = make_row_mapping();
    auto samples = make_samples(10, 3, 5);
    auto legacy = write_legacy(rm, samples);
    auto sharded = write_sharded(rm, samples, 1);
    expect_byte_identical(legacy, sharded);
}

TEST_F(ShardWriteTest, FourShardsEqualLegacy) {
    auto rm = make_row_mapping();
    auto samples = make_samples(20, 6, 8);
    auto legacy = write_legacy(rm, samples);
    auto sharded = write_sharded(rm, samples, 4);
    expect_byte_identical(legacy, sharded);
}

TEST_F(ShardWriteTest, ManyShardsEqualLegacy) {
    auto rm = make_row_mapping();
    auto samples = make_samples(31, 7, 11);  // 49 batches over 8 shards
    auto legacy = write_legacy(rm, samples);
    auto sharded = write_sharded(rm, samples, 8);
    expect_byte_identical(legacy, sharded);
}

TEST_F(ShardWriteTest, ShardFilesDeletedAfterMerge) {
    auto rm = make_row_mapping();
    auto samples = make_samples(4, 1, 1);
    auto dir = write_sharded(rm, samples, 3);
    for (uint32_t w = 0; w < 3; ++w) {
        EXPECT_FALSE(fs::exists(dir / ("batches_shard_" + std::to_string(w) + ".dat")));
    }
}

// catalog.unique_nodes must report EVERY distinct sampled node (the documented
// "unique nodes across all samples"), even when the RowMapping covers only a
// subset of them. Without this, the dense path silently collapses the count to
// the feature-row subset and the metric depends on the storage mode.
TEST_F(ShardWriteTest, UniqueNodesCountsExpandedNotJustFeatureRows) {
    auto samples = make_samples(20, 6, 8);

    // Ground truth: distinct node values across every sample.
    std::set<uint64_t> distinct;
    for (const auto& s : samples) {
        for (const auto& n : s.all_unique_nodes) distinct.insert(n.get_value());
    }
    const uint64_t expanded = distinct.size();

    // RowMapping covering only the first few node values — far fewer than the
    // distinct nodes the samples touch.
    constexpr uint64_t kCover = 4;
    std::vector<ObjectId> ids;
    for (uint64_t i = 1; i <= kCover; ++i) ids.push_back(ObjectId(i));
    auto rm = RowMapping::create(tmp_ / "subset.rmap", ids);

    auto config = make_config("subset");
    auto storage = SampleStorage::create(tmp_, config, rm, kNodes + 1);
    storage.begin_sharded_write(4);
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        std::ostringstream buf(std::ios::binary);
        s.serialize(buf);
        storage.shard_write(static_cast<uint32_t>(i % 4), s,
                            buf.str(), compute_batch_content_hash(s));
    }
    storage.merge_shards();

    EXPECT_GT(expanded, kCover);  // the samples really do exceed the mapping
    EXPECT_EQ(storage.get_catalog().unique_nodes, expanded);
}

}  // namespace
