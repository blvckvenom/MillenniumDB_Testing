// src/gnn/tests/test_batch_timing_log.cc
#include "gnn/training/batch_timing_log.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace mdb::gnn;

namespace {

fs::path tmp_path() {
    fs::path p = fs::temp_directory_path() / ("mdb_btl_" + std::to_string(::getpid())
                  + "_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()))
                  + ".csv");
    return p;
}

std::string read_all(const fs::path& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

BatchTiming sample_record(uint64_t batch_id) {
    // CSV schema (15 fields total):
    //   batch_id, split,
    //   sample_read_us, load_features_us,
    //   l1_us, l2_us, l3_us, l4_us, assembler_kernel_us,
    //   rmap_lookup_us, active_us, edge_us, h2d_us,
    //   forward_us, backward_us
    return {
        batch_id, 0,
        100, 200,
        10, 20, 30, 40, 45,
        50, 60, 70, 80,
        90, 110
    };
}

} // namespace

TEST(BatchTimingLogTests, OpensAndWritesHeader) {
    auto p = tmp_path();
    { BatchTimingLog log(p.string()); }
    auto content = read_all(p);
    EXPECT_NE(content.find("batch_id,split,sample_read_us"), std::string::npos);
    fs::remove(p);
}

TEST(BatchTimingLogTests, AppendsRecord) {
    auto p = tmp_path();
    {
        BatchTimingLog log(p.string());
        log.append(sample_record(42));
        log.flush();
    }
    auto content = read_all(p);
    EXPECT_NE(content.find("42,0,100,200"), std::string::npos);
    fs::remove(p);
}

TEST(BatchTimingLogTests, FlushesOnDestruct) {
    auto p = tmp_path();
    {
        BatchTimingLog log(p.string());
        log.append(sample_record(1));
        // no explicit flush
    }
    auto content = read_all(p);
    EXPECT_NE(content.find("1,0,100,200"), std::string::npos);
    fs::remove(p);
}

TEST(BatchTimingLogTests, AutoFlushAtInterval) {
    auto p = tmp_path();
    BatchTimingLog log(p.string(), /*flush_interval=*/2);
    log.append(sample_record(1));
    log.append(sample_record(2));   // should auto-flush
    EXPECT_EQ(log.pending(), 0u);
    fs::remove(p);
}

TEST(BatchTimingLogTests, ThreadSafe) {
    auto p = tmp_path();
    BatchTimingLog log(p.string(), 1024);
    auto worker = [&](int base) {
        for (int i = 0; i < 100; ++i) log.append(sample_record(base + i));
    };
    std::thread t1(worker, 0), t2(worker, 1000), t3(worker, 2000);
    t1.join(); t2.join(); t3.join();
    log.flush();

    auto content = read_all(p);
    int newlines = 0;
    for (char c : content) if (c == '\n') ++newlines;
    EXPECT_EQ(newlines, 1 + 300); // header + 300 records
    fs::remove(p);
}

TEST(BatchTimingLogTests, RejectsBadPath) {
    EXPECT_THROW(
        BatchTimingLog log("/dev/null/this/cannot/exist/timing.csv"),
        std::runtime_error
    );
}

TEST(BatchTimingLogTests, CreatesParentDir) {
    auto base = tmp_path().parent_path() / "mdb_btl_subdir";
    fs::remove_all(base);
    auto p = base / "nested" / "timing.csv";
    { BatchTimingLog log(p.string()); }
    EXPECT_TRUE(fs::exists(p));
    fs::remove_all(base);
}

TEST(BatchTimingLogTests, MultipleSplits) {
    auto p = tmp_path();
    {
        BatchTimingLog log(p.string());
        BatchTiming train = sample_record(0); train.split = 0;
        BatchTiming val   = sample_record(1); val.split   = 1;
        BatchTiming test  = sample_record(2); test.split  = 2;
        log.append(train); log.append(val); log.append(test);
    }
    auto content = read_all(p);
    EXPECT_NE(content.find("0,0,"), std::string::npos);
    EXPECT_NE(content.find("1,1,"), std::string::npos);
    EXPECT_NE(content.find("2,2,"), std::string::npos);
    fs::remove(p);
}
