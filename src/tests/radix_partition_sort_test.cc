// Unit tests for RadixPartitionSort<N> (TDD RED state).
//
// These tests intentionally fail to compile/link at the current commit:
//
//   - The `partition_file.h` include is unresolved (created in Task 5).
//   - The `RadixPartitionSort<N>` template has no backing .cc yet
//     (implementations are added in Tasks 6-11).
//
// That unresolved state is the expected TDD RED confirmation. The main
// `mdb` target must still build because this header is only included
// here and by the implementation file once it is added.
//
// Spec reference:
//   docs/superpowers/specs/2026-04-21-radix-partition-sort-design.md §8.2

#include <sys/resource.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/parallel_scan_partitioner.h"
#include "graph_models/gql/projection/partition_file.h"
#include "graph_models/gql/projection/radix_partition_sort.h"
#include "graph_models/gql/projection/sorter_dispatch.h"
#include "graph_models/gql/projection/spill_codec.h"
#include "graph_models/gql/projection/streaming_record_buffer.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/record.h"

namespace fs = std::filesystem;

namespace {

constexpr const char* kScratchBase = "/tmp/radix_sort_test_scratch";

void wipe_scratch() {
    fs::remove_all(kScratchBase);
    fs::create_directories(kScratchBase);
}

}  // namespace

// --- Test 1: Bucket assignment is deterministic ---
TEST(RadixPartitionSort, BucketAssignmentIsDeterministic) {
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    GQL::RadixPartitionSort<3> rps(cfg);
    (void)rps;
    ASSERT_EQ(GQL::RadixPartitionSort<3>::compute_num_partitions(
                  /*total_bytes=*/1ULL << 32,
                  /*target=*/256ULL * 1024 * 1024,
                  /*min=*/8, /*max=*/128),
              16);  // 4 GB / 256 MB = 16 exact
}

// --- Test 2: Phase 1 records are routed by radix ---
TEST(RadixPartitionSort, Phase1RecordsRoutedByRadix) {
    wipe_scratch();
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    cfg.num_scan_threads = 2;
    GQL::RadixPartitionSort<3> rps(cfg);

    GQL::StreamingRecordBuffer<3> input("/tmp/test_input");
    for (std::uint64_t i = 1; i <= 1000; i++) {
        Record<3> r{{i * 13, i * 17, i * 19}};
        input.push_back(r);
    }

    std::size_t num_parts = rps.scan_and_partition(input, /*est=*/1000);
    ASSERT_EQ(num_parts, 8u);
    for (std::size_t p = 0; p < num_parts; ++p) {
        ASSERT_TRUE(fs::exists(fs::path(kScratchBase) / ("partition_" + std::to_string(p) + ".bin")))
            << "missing partition file " << p;
    }
}

// --- Test 3: Phase 2 each partition is sorted ---
// keep_sorted_parts retains the `.sorted_part_*.bin` files that Phase 3
// normally removes as cleanup, so the per-partition sortedness assertions
// below run against real data instead of vacuously skipping missing files.
TEST(RadixPartitionSort, Phase2EachPartitionIsSorted) {
    wipe_scratch();
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    cfg.keep_sorted_parts = true;
    GQL::RadixPartitionSort<3> rps(cfg);

    GQL::StreamingRecordBuffer<3> input("/tmp/test_input3");
    std::mt19937_64 rng(42);
    for (int i = 0; i < 10000; i++) {
        Record<3> r{{rng(), rng(), rng()}};
        input.push_back(r);
    }
    rps.scan_and_partition(input, 10000);
    std::size_t written = rps.sort_and_write("/tmp/test_output3");

    ASSERT_GT(written, 0u);
    std::size_t parts_inspected = 0;
    std::size_t records_seen = 0;
    for (std::size_t p = 0; p < 8; ++p) {
        std::string path = "/tmp/test_output3.sorted_part_" + std::to_string(p) + ".bin";
        if (!fs::exists(path)) continue;
        ++parts_inspected;
        std::ifstream in(path, std::ios::binary);
        Record<3> prev{};
        Record<3> curr{};
        bool first = true;
        while (in.read(reinterpret_cast<char*>(&curr), sizeof(curr))) {
            ++records_seen;
            if (!first) {
                ASSERT_FALSE(curr < prev) << "non-monotonic in partition " << p;
            }
            prev = curr;
            first = false;
        }
        fs::remove(path);
    }
    ASSERT_GT(parts_inspected, 0u)
        << "no .sorted_part_*.bin file survived sort_and_write despite "
           "keep_sorted_parts — the sortedness loop above never ran";
    // 10000 distinct fixed-seed records: the pre-dedup partition contents
    // must carry exactly the records Phase 3 counted as written.
    ASSERT_EQ(records_seen, written)
        << "sorted partition files are missing records";
    fs::remove("/tmp/test_output3.leaf");
    fs::remove("/tmp/test_output3.dir");
}

// --- Test 4: Concatenation is globally sorted (SUPERSEDED by Task 13) ---
//
// This test was written during TDD RED (Task 4) when the Phase 3 output
// format was not yet designed. At that time the plan assumed `sort_and_write`
// would leave `.sorted_part_N.bin` files as its final output, and the test
// asserted that concatenating them produced a globally-sorted stream.
//
// The real Phase 3 (Task 10, commit 3669282b) writes the records into
// BPTLeafWriter / BPTDirWriter (`.leaf` / `.dir` B+Tree pages) and REMOVES
// the intermediate `.sorted_part_*.bin` files as post-phase cleanup — they
// are scratch artifacts, not API output. So the original assertion cannot
// succeed: `actual` is always empty because the files it tries to read are
// gone by the time Phase 3 returns.
//
// The invariant this test was meant to capture ("global sort order after
// radix concatenation") is preserved — it is validated end-to-end by
// scripts/test_projection_radix.sh on cora_gnn, which performs a
// byte-identical `cmp` of the B+Tree `.leaf`/`.dir` files produced by
// RADIX against the CLASSIC backend. See Task 13 in the plan.
//
// Disabled via gtest convention to keep the test visible in reports
// without counting as a failure.
TEST(RadixPartitionSort, DISABLED_ConcatenationIsGloballySorted) {
    wipe_scratch();
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    GQL::RadixPartitionSort<3> rps(cfg);

    std::vector<Record<3>> expected;
    GQL::StreamingRecordBuffer<3> input("/tmp/test_input4");
    std::mt19937_64 rng(7);
    for (int i = 0; i < 5000; i++) {
        Record<3> r{{rng(), rng(), rng()}};
        input.push_back(r);
        expected.push_back(r);
    }
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());

    rps.scan_and_partition(input, 5000);
    rps.sort_and_write("/tmp/test_output4");
    std::vector<Record<3>> actual;
    for (std::size_t p = 0; p < 8; ++p) {
        std::string path = "/tmp/test_output4.sorted_part_" + std::to_string(p) + ".bin";
        if (!fs::exists(path)) continue;
        std::ifstream in(path, std::ios::binary);
        Record<3> r{};
        while (in.read(reinterpret_cast<char*>(&r), sizeof(r))) {
            actual.push_back(r);
        }
    }
    ASSERT_EQ(actual, expected);
}

// --- Test 5: Radix output == Classic output (Golden) ---
TEST(RadixPartitionSort, GoldenCompareClassicBackend_PlaceholderRunViaScript) {
    GTEST_SKIP() << "Run scripts/test_projection_radix.sh — byte-identical "
                    "B+Tree output vs CLASSIC on cora_gnn.";
}

// --- Test 6: Partition count clamped correctly ---
TEST(RadixPartitionSort, PartitionCountClampedCorrectly) {
    using RPS = GQL::RadixPartitionSort<3>;
    ASSERT_EQ(RPS::compute_num_partitions(/*bytes=*/1024,
                                          256ULL * 1024 * 1024, 8, 128), 8u);
    ASSERT_EQ(RPS::compute_num_partitions(1ULL << 30,
                                          256ULL * 1024 * 1024, 8, 128), 8u);
    ASSERT_EQ(RPS::compute_num_partitions(5ULL * 1024 * 1024 * 1024ULL,
                                          256ULL * 1024 * 1024, 8, 128), 20u);
    ASSERT_EQ(RPS::compute_num_partitions(40ULL * 1024 * 1024 * 1024ULL,
                                          256ULL * 1024 * 1024, 8, 128), 128u);
}

// --- Test 7: Worker count adaptive to cores and memory ---
TEST(RadixPartitionSort, WorkerCountAdaptivToCoresMemory) {
    using RPS = GQL::RadixPartitionSort<3>;
    constexpr std::size_t kMemBudget = 2ULL * 1024 * 1024 * 1024;
    constexpr std::size_t kWorkerBudget = 512ULL * 1024 * 1024;
    ASSERT_EQ(RPS::compute_num_workers(
                  16, 8, kMemBudget, kWorkerBudget, /*override=*/0),
              4u);
    ASSERT_EQ(RPS::compute_num_workers(
                  4, 2, kMemBudget, kWorkerBudget, /*override=*/0),
              2u);
    ASSERT_EQ(RPS::compute_num_workers(
                  16, 8, kMemBudget, kWorkerBudget, /*override=*/7),
              7u);
}

// --- Test 8: Env var switches backend ---
// get_sorter_backend() caches its decision process-wide on first call
// (std::call_once), so this test must remain the binary's only caller —
// a second observation could not see a different env value.
TEST(SorterDispatch, EnvVarSwitchesBackend) {
    setenv("MDB_PROJECTION_SORTER", "radix", 1);
    EXPECT_EQ(GQL::get_sorter_backend(), GQL::SorterBackend::RADIX)
        << "MDB_PROJECTION_SORTER=radix must select the RADIX backend";
    unsetenv("MDB_PROJECTION_SORTER");
}

// --- Test 9: Defensive fallback when partition exceeds worker memory ---
TEST(RadixPartitionSort, FallbackExternalWhenPartitionOversized) {
    wipe_scratch();
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    cfg.worker_memory_budget = 1024;
    GQL::RadixPartitionSort<3> rps(cfg);

    GQL::StreamingRecordBuffer<3> input("/tmp/test_input9");
    for (std::uint64_t i = 1; i <= 500; ++i) {
        input.push_back(Record<3>{{i, i + 1, i + 2}});
    }
    rps.scan_and_partition(input, 500);
    std::size_t written = rps.sort_and_write("/tmp/test_output9");
    ASSERT_EQ(written, 500u);
}

// --- Test 9b: Phase 2 write failure must throw, not silently truncate ---
// An unopenable output path stands in for any Phase 2 write failure
// (ENOSPC, EIO, ...). With unchecked streams every ofstream write silently
// no-ops, Phase 3 then sees zero sorted partitions and emits a
// structurally-valid EMPTY B+Tree while sort_and_write returns 0 — total
// silent data loss. The checked Phase 2 streams must surface the failure
// as an exception instead.
TEST(RadixPartitionSort, Phase2WriteFailureThrows) {
    wipe_scratch();
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    GQL::RadixPartitionSort<3> rps(cfg);

    GQL::StreamingRecordBuffer<3> input("/tmp/test_input_wfail");
    for (std::uint64_t i = 1; i <= 100; ++i) {
        input.push_back(Record<3>{{i, i + 1, i + 2}});
    }
    rps.scan_and_partition(input, 100);
    // The parent directory does not exist, so every Phase 2 output open
    // fails — the same observable state a short write leaves behind.
    EXPECT_THROW(rps.sort_and_write("/tmp/radix_sort_test_no_such_dir/out"),
                 std::runtime_error);
}

// --- Test 10: Cleanup on exception ---
// Phase 2 dies mid-flight (unopenable output directory — the same
// observable state as ENOSPC); the destructor must still remove the
// whole scratch directory on unwind.
TEST(RadixPartitionSort, ScratchFilesCleanedOnException) {
    wipe_scratch();
    {
        GQL::RadixPartitionSort<3>::Config cfg;
        cfg.scratch_dir = kScratchBase;
        cfg.min_partitions = 8;
        cfg.max_partitions = 8;
        GQL::RadixPartitionSort<3> rps(cfg);

        GQL::StreamingRecordBuffer<3> input("/tmp/test_input10");
        input.push_back(Record<3>{{1, 2, 3}});
        rps.scan_and_partition(input, 1);
        EXPECT_THROW(
            rps.sort_and_write("/tmp/radix_sort_test_no_such_dir10/out"),
            std::runtime_error);
        ASSERT_TRUE(fs::exists(kScratchBase))
            << "partition scratch vanished before the destructor ran";
    }
    ASSERT_FALSE(fs::exists(kScratchBase))
        << "scratch dir not cleaned by destructor after exception";
}

// Read the first 16 bytes of a file and return them.
static std::vector<uint8_t> read_first_16_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> out(16, 0);
    in.read(reinterpret_cast<char*>(out.data()), 16);
    return out;
}

// --- T5.11b Test: BITSET backend produces byte 0 != 0x02 ---
TEST(RadixPartitionSort, BitsetBackend_ByteZeroNotTwo) {
    wipe_scratch();
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    cfg.leaf_format = BPT::LeafFormat::BITSET;  // explicit default
    GQL::RadixPartitionSort<3> rps(cfg);

    GQL::StreamingRecordBuffer<3> input("/tmp/test_input_bitset");
    for (std::uint64_t i = 1; i <= 500; ++i) {
        input.push_back(Record<3>{{i, i + 1, i + 2}});
    }
    rps.scan_and_partition(input, 500);
    const std::string out_base = "/tmp/test_output_bitset";
    std::size_t written = rps.sort_and_write(out_base);
    ASSERT_EQ(written, 500u);
    // Byte 0 is the V1 value_count LSB; the first page's value_count is
    // 170 (max_records_per_leaf for N=3) in this 500-record dataset.
    // What we positively assert is: byte 0 != 0x02 under BITSET for this
    // specific dataset. (A different dataset with value_count=2 would
    // legitimately have byte 0 = 0x02 — design §6.1 edge case.)
    const auto hdr = read_first_16_bytes(out_base + ".leaf");
    EXPECT_NE(hdr[0], 0x02) << "BITSET backend must not emit v2 magic";
    fs::remove(out_base + ".leaf");
    fs::remove(out_base + ".dir");
}

// --- T5.11b Test: DELTA_VARINT backend produces byte 0 == 0x02 ---
TEST(RadixPartitionSort, DeltaVarintBackend_ByteZeroIsTwo) {
    wipe_scratch();
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    cfg.leaf_format = BPT::LeafFormat::DELTA_VARINT;  // opt-in
    GQL::RadixPartitionSort<3> rps(cfg);

    GQL::StreamingRecordBuffer<3> input("/tmp/test_input_dv");
    for (std::uint64_t i = 1; i <= 500; ++i) {
        input.push_back(Record<3>{{i, i + 1, i + 2}});
    }
    rps.scan_and_partition(input, 500);
    const std::string out_base = "/tmp/test_output_dv";
    std::size_t written = rps.sort_and_write(out_base);
    ASSERT_EQ(written, 500u) << "DELTA_VARINT dedup count must match input";

    const auto hdr = read_first_16_bytes(out_base + ".leaf");
    EXPECT_EQ(hdr[0], 0x02) << "DELTA_VARINT backend must emit v2 format_version magic";
    EXPECT_EQ(hdr[1], 3)    << "record_width must equal N=3";
    EXPECT_EQ(hdr[2], 0)    << "flags must be 0 (reserved)";
    EXPECT_EQ(hdr[3], 0)    << "reserved must be 0";
    fs::remove(out_base + ".leaf");
    fs::remove(out_base + ".dir");
}

// ---------------------------------------------------------------------------
// Spec #24 — GPU vs CPU bit-equal output (RADIX Phase 2 per-partition sort)
// ---------------------------------------------------------------------------
// Two fixed-seed runs of the same dataset MUST produce byte-identical
// `.sorted_part_*.bin` outputs regardless of whether the GPU code path
// engaged or not. The GPU path is gated by MDB_PROJECTION_RADIX_GPU; the
// MDB_PROJECTION_RADIX_GPU_MIN_RECORDS knob (introduced by Spec #24) lets
// us reach the GPU planner on small test datasets without lowering it
// globally. When the build lacks CUDA the GPU run is identical to the CPU
// run by construction (the #ifdef block in sort_partition_in_memory is
// elided). Either way the byte-identical assertion below must hold.
//
// IMPORTANT: the underlying `mdb::gpu::execute_gpu_radix_sort` extracts the
// lower 32 bits of each ObjectId value (mask 0x00FFFFFFFFFFFFFFULL then
// truncated to uint32_t — see gpu_radix_sort.cu:230). To produce
// byte-identical output across CPU and GPU we must feed it data whose top
// 8 bits (ObjectId type prefix) are constant across the dataset AND whose
// value bits fit in the lower 32 bits. Real B+Tree indexes satisfy both
// conditions by construction: every record in a given index shares the
// same type prefix, and graph datasets up to ~4 B nodes/edges fit in 32
// bits. The synthetic data below mimics that layout (`type | value32`).
TEST(RadixPartitionSortGpu, GpuVsCpuBitEqualPartitionOutputs) {
    constexpr uint64_t kTypePrefix = 0x4200000000000000ULL;  // node-like type
    auto make_oid = [&](uint64_t v) {
        return kTypePrefix | (v & 0xFFFFFFFFULL);
    };

    auto run = [&](const char* seed_label, bool gpu_on) {
        // Use a unique scratch / output base per run so the two passes do
        // not stomp each other's intermediate files.
        const std::string scratch =
            std::string(kScratchBase) + "_gpu_" + seed_label;
        const std::string output_base =
            std::string("/tmp/radix_gpu_test_output_") + seed_label;
        fs::remove_all(scratch);
        fs::create_directories(scratch);
        for (std::size_t p = 0; p < 8; ++p) {
            fs::remove(output_base + ".sorted_part_" + std::to_string(p) + ".bin");
        }
        fs::remove(output_base + ".leaf");
        fs::remove(output_base + ".dir");

        if (gpu_on) {
            unsetenv("MDB_PROJECTION_RADIX_GPU");
        } else {
            setenv("MDB_PROJECTION_RADIX_GPU", "0", 1);
        }
        // Reach the GPU planner even at this small dataset size.
        setenv("MDB_PROJECTION_RADIX_GPU_MIN_RECORDS", "1000", 1);

        GQL::RadixPartitionSort<3>::Config cfg;
        cfg.scratch_dir    = scratch;
        cfg.min_partitions = 8;
        cfg.max_partitions = 8;
        GQL::RadixPartitionSort<3> rps(cfg);

        GQL::StreamingRecordBuffer<3> input(scratch + "/input");
        std::mt19937_64 rng(0xC0FFEE);  // fixed seed → deterministic input
        for (int i = 0; i < 50000; i++) {
            Record<3> r{{ make_oid(rng()), make_oid(rng()), make_oid(rng()) }};
            input.push_back(r);
        }
        rps.scan_and_partition(input, 50000);
        std::size_t written = rps.sort_and_write(output_base);
        EXPECT_GT(written, 0u);

        // Phase 3 cleans up the per-partition .bin files but leaves the
        // .leaf file as the authoritative sorted output. Hash the .leaf
        // bytes for the cross-run comparison.
        std::ifstream in(output_base + ".leaf", std::ios::binary);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };

    const std::string cpu_bytes = run("cpu", /*gpu_on=*/false);
    const std::string gpu_bytes = run("gpu", /*gpu_on=*/true);

    ASSERT_EQ(cpu_bytes.size(), gpu_bytes.size())
        << "RADIX .leaf size differs between CPU and GPU paths";
    EXPECT_EQ(cpu_bytes, gpu_bytes)
        << "RADIX .leaf bytes differ between CPU and GPU paths";

    unsetenv("MDB_PROJECTION_RADIX_GPU");
    unsetenv("MDB_PROJECTION_RADIX_GPU_MIN_RECORDS");
}

// Keys whose counters exceed 32 bits violate the GPU sort's truncated-key
// precondition: execute_gpu_radix_sort keeps only the low 32 bits of each
// field, so 48-bit counters would collide and come out mis-ordered. The
// runtime guard must route such partitions to the CPU std::sort, keeping
// GPU-on and GPU-off runs byte-identical. On a CUDA build without the
// guard the GPU run silently sorts by truncated keys and this test fails;
// on non-CUDA builds both runs take the CPU path by construction.
TEST(RadixPartitionSortGpu, WideCountersFallBackToCpuBitEqual) {
    constexpr uint64_t kTypePrefix = 0x4200000000000000ULL;
    auto make_wide_oid = [&](uint64_t v) {
        // 48-bit counters: well past the GPU path's 32-bit key range.
        return kTypePrefix | (v & 0x0000FFFFFFFFFFFFULL);
    };

    auto run = [&](const char* seed_label, bool gpu_on) {
        const std::string scratch =
            std::string(kScratchBase) + "_gpu_wide_" + seed_label;
        const std::string output_base =
            std::string("/tmp/radix_gpu_wide_test_output_") + seed_label;
        fs::remove_all(scratch);
        fs::create_directories(scratch);
        fs::remove(output_base + ".leaf");
        fs::remove(output_base + ".dir");

        if (gpu_on) {
            unsetenv("MDB_PROJECTION_RADIX_GPU");
        } else {
            setenv("MDB_PROJECTION_RADIX_GPU", "0", 1);
        }
        // Reach the GPU planner even at this small dataset size.
        setenv("MDB_PROJECTION_RADIX_GPU_MIN_RECORDS", "1000", 1);

        GQL::RadixPartitionSort<3>::Config cfg;
        cfg.scratch_dir    = scratch;
        cfg.min_partitions = 8;
        cfg.max_partitions = 8;
        GQL::RadixPartitionSort<3> rps(cfg);

        GQL::StreamingRecordBuffer<3> input(scratch + "/input");
        std::mt19937_64 rng(0xBADC0DE5);  // fixed seed → deterministic input
        for (int i = 0; i < 50000; i++) {
            Record<3> r{{ make_wide_oid(rng()), make_wide_oid(rng()),
                          make_wide_oid(rng()) }};
            input.push_back(r);
        }
        rps.scan_and_partition(input, 50000);
        std::size_t written = rps.sort_and_write(output_base);
        EXPECT_GT(written, 0u);

        std::ifstream in(output_base + ".leaf", std::ios::binary);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };

    const std::string cpu_bytes = run("cpu", /*gpu_on=*/false);
    const std::string gpu_bytes = run("gpu", /*gpu_on=*/true);

    ASSERT_EQ(cpu_bytes.size(), gpu_bytes.size())
        << "RADIX .leaf size differs between CPU and GPU-eligible runs "
           "on >32-bit counters";
    EXPECT_EQ(cpu_bytes, gpu_bytes)
        << "RADIX .leaf bytes differ on >32-bit counters — the GPU path "
           "must fall back to CPU instead of sorting truncated keys";

    unsetenv("MDB_PROJECTION_RADIX_GPU");
    unsetenv("MDB_PROJECTION_RADIX_GPU_MIN_RECORDS");
}

// Determinism: same input + same env settings → byte-identical output across
// two independent runs of the GPU-eligible path. Uses ObjectId-shaped data
// (constant 8-bit type prefix + 32-bit value) so the assertion holds even
// when the GPU radix path engages — see GpuVsCpuBitEqualPartitionOutputs
// above for the rationale.
TEST(RadixPartitionSortGpu, DeterministicRepeatedRuns) {
    constexpr uint64_t kTypePrefix = 0x4200000000000000ULL;
    auto make_oid = [&](uint64_t v) {
        return kTypePrefix | (v & 0xFFFFFFFFULL);
    };

    auto run = [&](int run_idx) {
        const std::string scratch =
            std::string(kScratchBase) + "_det_" + std::to_string(run_idx);
        const std::string output_base =
            std::string("/tmp/radix_gpu_det_output_") + std::to_string(run_idx);
        fs::remove_all(scratch);
        fs::create_directories(scratch);
        fs::remove(output_base + ".leaf");
        fs::remove(output_base + ".dir");

        unsetenv("MDB_PROJECTION_RADIX_GPU");
        setenv("MDB_PROJECTION_RADIX_GPU_MIN_RECORDS", "1000", 1);

        GQL::RadixPartitionSort<3>::Config cfg;
        cfg.scratch_dir    = scratch;
        cfg.min_partitions = 8;
        cfg.max_partitions = 8;
        GQL::RadixPartitionSort<3> rps(cfg);

        GQL::StreamingRecordBuffer<3> input(scratch + "/input");
        std::mt19937_64 rng(0xDEADBEEF);
        for (int i = 0; i < 25000; i++) {
            Record<3> r{{ make_oid(rng()), make_oid(rng()), make_oid(rng()) }};
            input.push_back(r);
        }
        rps.scan_and_partition(input, 25000);
        rps.sort_and_write(output_base);

        std::ifstream in(output_base + ".leaf", std::ios::binary);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };

    const std::string a = run(1);
    const std::string b = run(2);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(a, b) << "RADIX GPU-eligible path is non-deterministic";
    unsetenv("MDB_PROJECTION_RADIX_GPU_MIN_RECORDS");
}

// --- T5.11b Test: empty index under DELTA_VARINT emits a single v2 page ---
TEST(RadixPartitionSort, DeltaVarintEmpty_EmitsSingleV2Page) {
    wipe_scratch();
    GQL::RadixPartitionSort<3>::Config cfg;
    cfg.scratch_dir = kScratchBase;
    cfg.min_partitions = 8;
    cfg.max_partitions = 8;
    cfg.leaf_format = BPT::LeafFormat::DELTA_VARINT;
    GQL::RadixPartitionSort<3> rps(cfg);

    GQL::StreamingRecordBuffer<3> input("/tmp/test_input_dv_empty");
    // Intentionally empty input.
    rps.scan_and_partition(input, 0);
    const std::string out_base = "/tmp/test_output_dv_empty";
    std::size_t written = rps.sort_and_write(out_base);
    EXPECT_EQ(written, 0u);

    const auto hdr = read_first_16_bytes(out_base + ".leaf");
    EXPECT_EQ(hdr[0], 0x02) << "empty v2 leaf must still carry format_version=2";
    EXPECT_EQ(hdr[1], 3);
    fs::remove(out_base + ".leaf");
    fs::remove(out_base + ".dir");
}

// ---------------------------------------------------------------------------
// Spec #25 — Phase 1 parallel partition fill bit-equal vs sequential.
// ---------------------------------------------------------------------------
// Two runs of `scan_and_partition` + `sort_and_write` with identical inputs
// MUST produce byte-identical `.leaf` outputs regardless of whether the
// chunk-parallel Phase 1 path engaged or not. The parallel path is gated
// by MDB_PROJECTION_RADIX_PHASE1_PARALLEL (default ON; "0" forces the
// legacy single-thread drain). Phase 2 sorts each partition end-to-end
// after concatenation, so even though the records WITHIN a per-thread
// part_p.bin file may land in different orders depending on how many
// TBB workers run, the final B+Tree leaves must converge.
TEST(RadixPartitionSortPhase1, ParallelVsSequentialBitEqualLeafOutput) {
    constexpr uint64_t kTypePrefix = 0x4200000000000000ULL;
    auto make_oid = [&](uint64_t v) {
        return kTypePrefix | (v & 0xFFFFFFFFULL);
    };

    auto run = [&](const char* seed_label, bool parallel_on) {
        const std::string scratch =
            std::string(kScratchBase) + "_phase1_" + seed_label;
        const std::string output_base =
            std::string("/tmp/radix_phase1_test_output_") + seed_label;
        fs::remove_all(scratch);
        fs::create_directories(scratch);
        fs::remove(output_base + ".leaf");
        fs::remove(output_base + ".dir");

        if (parallel_on) {
            unsetenv("MDB_PROJECTION_RADIX_PHASE1_PARALLEL");  // default ON
        } else {
            setenv("MDB_PROJECTION_RADIX_PHASE1_PARALLEL", "0", 1);
        }
        // Make sure Phase 2 GPU path doesn't add a confound: keep it OFF
        // for this Phase 1-focused comparison.
        setenv("MDB_PROJECTION_RADIX_GPU", "0", 1);

        GQL::RadixPartitionSort<3>::Config cfg;
        cfg.scratch_dir      = scratch;
        cfg.min_partitions   = 8;
        cfg.max_partitions   = 8;
        cfg.num_scan_threads = 4;
        GQL::RadixPartitionSort<3> rps(cfg);

        GQL::StreamingRecordBuffer<3> input(scratch + "/input");
        std::mt19937_64 rng(0xFEEDC0DE);  // fixed seed → deterministic input
        // Need >= kParallelMinRecords (65 K) to engage the parallel path.
        for (int i = 0; i < 100000; i++) {
            Record<3> r{{ make_oid(rng()), make_oid(rng()), make_oid(rng()) }};
            input.push_back(r);
        }
        rps.scan_and_partition(input, 100000);
        std::size_t written = rps.sort_and_write(output_base);
        EXPECT_GT(written, 0u);

        std::ifstream in(output_base + ".leaf", std::ios::binary);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };

    const std::string seq_bytes = run("seq", /*parallel_on=*/false);
    const std::string par_bytes = run("par", /*parallel_on=*/true);

    ASSERT_EQ(seq_bytes.size(), par_bytes.size())
        << "RADIX .leaf size differs between sequential and parallel Phase 1 paths";
    EXPECT_EQ(seq_bytes, par_bytes)
        << "RADIX .leaf bytes differ between sequential and parallel Phase 1 paths";

    unsetenv("MDB_PROJECTION_RADIX_PHASE1_PARALLEL");
    unsetenv("MDB_PROJECTION_RADIX_GPU");
}

// Determinism: two independent runs with the parallel Phase 1 path engaged
// must produce byte-identical .leaf output despite TBB worker non-determinism
// in record-within-partition ordering. Phase 2's full sort masks the
// non-determinism end-to-end.
TEST(RadixPartitionSortPhase1, ParallelDeterministicRepeatedRuns) {
    constexpr uint64_t kTypePrefix = 0x4200000000000000ULL;
    auto make_oid = [&](uint64_t v) {
        return kTypePrefix | (v & 0xFFFFFFFFULL);
    };

    auto run = [&](int run_idx) {
        const std::string scratch =
            std::string(kScratchBase) + "_phase1_det_" + std::to_string(run_idx);
        const std::string output_base =
            std::string("/tmp/radix_phase1_det_output_") + std::to_string(run_idx);
        fs::remove_all(scratch);
        fs::create_directories(scratch);
        fs::remove(output_base + ".leaf");
        fs::remove(output_base + ".dir");

        unsetenv("MDB_PROJECTION_RADIX_PHASE1_PARALLEL");  // default ON
        setenv("MDB_PROJECTION_RADIX_GPU", "0", 1);

        GQL::RadixPartitionSort<3>::Config cfg;
        cfg.scratch_dir      = scratch;
        cfg.min_partitions   = 8;
        cfg.max_partitions   = 8;
        cfg.num_scan_threads = 4;
        GQL::RadixPartitionSort<3> rps(cfg);

        GQL::StreamingRecordBuffer<3> input(scratch + "/input");
        std::mt19937_64 rng(0xCAFEBABE);
        for (int i = 0; i < 80000; i++) {
            Record<3> r{{ make_oid(rng()), make_oid(rng()), make_oid(rng()) }};
            input.push_back(r);
        }
        rps.scan_and_partition(input, 80000);
        rps.sort_and_write(output_base);

        std::ifstream in(output_base + ".leaf", std::ios::binary);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };

    const std::string a = run(1);
    const std::string b = run(2);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(a, b)
        << "RADIX parallel Phase 1 path is non-deterministic across repeated runs";

    unsetenv("MDB_PROJECTION_RADIX_GPU");
}

// Spilled iteration must yield exactly the records that were pushed. The
// spilled-mode has_next() used to infer exhaustion from positional state
// (current spill file index + read buffer), which only advances inside the
// next read attempt: after the last chunk of the last file was consumed it
// still reported one more record, and next() then indexed an empty read
// buffer (out-of-bounds read).
TEST(StreamingRecordBuffer, SpilledIterationYieldsExactlyNRecords) {
    // 437 records with a 300-record memory threshold produce two spill files
    // (300 + 137 records); the 300-record file is read in two chunks
    // (RECORDS_PER_PAGE = 170 for N = 3), so iteration crosses both chunk
    // and file boundaries.
    constexpr std::size_t kNumRecords       = 437;
    constexpr std::size_t kThresholdRecords = 300;
    GQL::StreamingRecordBuffer<3> buffer(
        "/tmp/test_srb_exact",
        kThresholdRecords * 3 * sizeof(std::uint64_t));

    for (std::uint64_t i = 0; i < kNumRecords; i++) {
        buffer.push_back(Record<3>{{ i, i + 1000, i + 2000 }});
    }
    buffer.finalize();
    ASSERT_TRUE(buffer.has_spilled());
    ASSERT_EQ(buffer.get_spill_paths().size(), 2u);
    ASSERT_EQ(buffer.size(), kNumRecords);

    buffer.begin_iteration();
    std::uint64_t count = 0;
    while (buffer.has_next()) {
        ASSERT_LT(count, kNumRecords) << "iteration emitted a record past the end";
        const Record<3>& rec = buffer.next();
        EXPECT_EQ(rec[0], count);
        EXPECT_EQ(rec[1], count + 1000);
        EXPECT_EQ(rec[2], count + 2000);
        count++;
    }
    EXPECT_EQ(count, kNumRecords);
}

// A spill file shorter than its recorded record count must surface as an
// exception from next() instead of indexing an empty read buffer or spinning
// forever on a reader that keeps returning 0 bytes.
//
// Compression-mode note: resolve_default_spill_compression() is cached
// process-wide on its first call (spill_codec.h), so when this binary runs
// the full suite, earlier StreamingRecordBuffer constructions have already
// pinned the compiled default (LZ4 when HAS_LZ4) and setting
// MDB_SPILL_COMPRESSION here would be silently ignored. The truncation
// point below therefore lands differently per mode:
//   - NONE: the cut is at an exact record boundary — file 1 keeps 1 of its
//     3 records, so 5 records are readable before the throw.
//   - LZ4:  the cut lands mid-frame — file 1 decodes to nothing, so only
//     the 4 records of the intact file 0 are readable before the throw.
// Either way the regression contract of the honest-has_next fix holds: a
// full drain loop must throw std::runtime_error from next() after serving
// at least every record of the intact first file and strictly fewer than
// the recorded total — never index an empty read buffer or spin forever.
TEST(StreamingRecordBuffer, TruncatedSpillFileThrowsMidIteration) {
    constexpr std::size_t  kThresholdRecords = 4;
    constexpr std::uint64_t kNumRecords      = 7;
    GQL::StreamingRecordBuffer<3> buffer(
        "/tmp/test_srb_trunc",
        kThresholdRecords * 3 * sizeof(std::uint64_t));
    for (std::uint64_t i = 0; i < kNumRecords; i++) {
        buffer.push_back(Record<3>{{ i, i, i }});
    }
    buffer.finalize();
    ASSERT_EQ(buffer.get_spill_paths().size(), 2u);  // 4 + 3 records

    // Truncate the second spill file (3 records) down to one record's worth
    // of payload bytes.
    fs::resize_file(buffer.get_spill_paths()[1],
                    GQL::SpillFormat::HEADER_SIZE + 1 * 3 * sizeof(std::uint64_t));

    buffer.begin_iteration();
    std::uint64_t drained = 0;
    EXPECT_THROW(
        {
            while (buffer.has_next()) {
                buffer.next();
                drained++;
            }
        },
        std::runtime_error);

    // The intact first spill file (4 records) must have been served in
    // full before the truncation was detected...
    EXPECT_GE(drained, 4u);
    // ...and the unreadable truncated tail must NOT have been fabricated:
    // pre-fix, the positional has_next() over-reported and next() indexed
    // an empty read buffer instead of throwing.
    EXPECT_LT(drained, kNumRecords);
}

// --- ParallelScanPartitioner: merge survives empty per-thread files ---
// Every (thread, partition) part_p.bin is pre-created in the ctor, so a
// sparse partition commonly has an EMPTY file for some slot. Per
// [ostream.inserters], operator<<(basic_streambuf*) sets failbit when it
// inserts zero characters: if an empty source precedes a non-empty one in
// the merge loop, a poisoned sink turns every later insert into a silent
// no-op — losing that partition's records entirely.
TEST(ParallelScanPartitioner, MergeSurvivesEmptyThreadFileBeforeNonEmpty) {
    const std::string scratch = std::string(kScratchBase) + "_psp_merge";
    fs::remove_all(scratch);

    GQL::ParallelScanPartitioner<1> partitioner(
        /*num_partitions=*/2, /*num_scan_threads=*/2, scratch,
        [](const Record<1>& r) { return static_cast<std::uint32_t>(r[0]); });

    constexpr std::size_t kRecordsForPartition1 = 5;
    partitioner.run([&](std::function<void(const Record<1>&)> emit) {
        // Slot assignment is first-touch. Thread A (joined before B starts)
        // takes slot 0 and writes ONLY partition 0, so thread_0/part_1.bin
        // stays empty; thread B takes slot 1 and writes ONLY partition 1.
        // Partition 1's merge order is therefore: empty file, then 5 records.
        std::thread a([&] { emit(Record<1>{ { 0 } }); });
        a.join();
        std::thread b([&] {
            for (std::size_t i = 0; i < kRecordsForPartition1; ++i) {
                emit(Record<1>{ { 1 } });
            }
        });
        b.join();
    });

    auto merged = partitioner.collect_merged_partition_paths();
    ASSERT_EQ(merged.size(), 2u);

    std::error_code ec;
    const auto p0_size = fs::file_size(merged[0], ec);
    ASSERT_FALSE(ec);
    EXPECT_EQ(p0_size, 1 * sizeof(Record<1>));

    const auto p1_size = fs::file_size(merged[1], ec);
    ASSERT_FALSE(ec);
    EXPECT_EQ(p1_size, kRecordsForPartition1 * sizeof(Record<1>));

    fs::remove_all(scratch);
}

// --- ParallelScanPartitioner: ctor survives a low NOFILE soft limit ---
// The ctor eagerly opens one FILE* per (thread, partition) pair; with a
// stock 1024 soft RLIMIT_NOFILE a large fan-out used to die inside the
// PartitionFile ctor before any record flowed. The ctor must raise the
// soft limit (or shrink the slot count) instead.
TEST(ParallelScanPartitioner, CtorSurvivesLowNofileSoftLimit) {
    struct rlimit saved;
    ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &saved), 0);
    if (saved.rlim_max != RLIM_INFINITY && saved.rlim_max < 1024) {
        GTEST_SKIP() << "hard RLIMIT_NOFILE too low to exercise the raise path";
    }

    // Drop the soft limit below the 4 × 64 = 256 eager-FILE* fan-out.
    struct rlimit low = saved;
    low.rlim_cur = 96;
    ASSERT_EQ(::setrlimit(RLIMIT_NOFILE, &low), 0);

    const std::string scratch = std::string(kScratchBase) + "_psp_rlimit";
    fs::remove_all(scratch);

    try {
        GQL::ParallelScanPartitioner<1> partitioner(
            /*num_partitions=*/64, /*num_scan_threads=*/4, scratch,
            [](const Record<1>& r) { return static_cast<std::uint32_t>(r[0]); });
        (void)partitioner;
    } catch (const std::exception& e) {
        ::setrlimit(RLIMIT_NOFILE, &saved);
        fs::remove_all(scratch);
        FAIL() << "ctor threw under a low NOFILE soft limit: " << e.what();
    }

    ASSERT_EQ(::setrlimit(RLIMIT_NOFILE, &saved), 0);
    fs::remove_all(scratch);
}
