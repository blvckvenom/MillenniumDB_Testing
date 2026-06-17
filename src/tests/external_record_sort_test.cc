// Regression test for the CLASSIC external-sort spill/merge path used by the
// GQL projection B+Tree index build (graph_project).
//
// The CLASSIC backend (GQL::ExternalRecordSort<N>) forms sorted "runs" on disk
// when the in-memory buffer overflows, then K-way-merges them in stream_sorted.
// A tbb::task_arena cap was added to bound the run-sort fan-out RSS (see the
// MEMORY BOUND comment in external_record_sort.h). This test pins the
// correctness of the spill -> sort_run_files -> merge_runs path so a future
// refactor cannot silently break the merge result or re-introduce the
// unbounded-memory OOM.
//
// The test deliberately drives MULTIPLE run files through disk: each run is
// written with SpillWriter (the exact format SpillReader/read_file_to_vector
// consumes inside the sorter) and registered via add_run(). A small explicit
// buffer_size forces fits_in_memory()==false, so stream_sorted routes through
// stream_external (Phase 1 sort_run_files + Phase 2 merge_runs) rather than the
// in-memory fast path.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "graph_models/gql/projection/external_record_sort.h"
#include "graph_models/gql/projection/spill_codec.h"
#include "storage/index/record.h"

namespace fs = std::filesystem;

namespace {

using Rec3 = Record<3>;

// Writes a chunk of records to a fresh spill file using the same on-disk
// format SpillReader (and therefore ExternalRecordSort::read_file_to_vector)
// understands. NONE compression is used so the bytes are deterministic and
// independent of whether the build linked liblz4.
void write_spill_run(const std::string& path, const std::vector<Rec3>& recs) {
    GQL::SpillWriter writer(path, GQL::SpillCompression::NONE);
    if (!recs.empty()) {
        writer.write(recs.data(), recs.size() * sizeof(Rec3));
    }
    writer.finalize();
}

}  // namespace

// Feeds many records through ExternalRecordSort<3> as several on-disk runs with
// a buffer too small to hold them all, forcing the spill/sort/merge path, then
// asserts the streamed output is globally sorted, count-preserving, and equal
// to a reference std::sort of the same input.
TEST(ExternalRecordSortSpill, MultiRunMergeIsSortedAndComplete) {
    const std::string tmp =
        (fs::temp_directory_path() / "mdb_ext_sort_spill_test").string();
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    constexpr size_t kN = 200000;       // 200k * 24 B = 4.8 MB total
    constexpr size_t kChunk = 20000;    // -> 10 separate run files on disk

    std::mt19937_64 rng(1234);
    std::vector<Rec3> input;
    input.reserve(kN);
    for (size_t i = 0; i < kN; ++i) {
        Rec3 r{};
        r[0] = rng() & 0x00FFFFFFFFFFFFFFULL;
        r[1] = rng() & 0x00FFFFFFFFFFFFFFULL;
        r[2] = rng() & 0x00FFFFFFFFFFFFFFULL;
        input.push_back(r);
    }

    // 1 MB buffer < 4.8 MB total -> fits_in_memory() is false -> external path.
    GQL::ExternalRecordSort<3> sorter(tmp, /*buffer_size=*/1u << 20);

    // Write the input as multiple spill runs and register each with add_run.
    std::vector<std::string> run_paths;
    for (size_t off = 0; off < kN; off += kChunk) {
        size_t end = std::min(off + kChunk, kN);
        std::vector<Rec3> chunk(input.begin() + off, input.begin() + end);
        std::string path = tmp + "/spill_" + std::to_string(off / kChunk);
        write_spill_run(path, chunk);
        run_paths.push_back(path);
        sorter.add_run(path, chunk.size());
    }

    // Precondition checks: the test must genuinely exercise the spill/merge
    // path, not the in-memory shortcut.
    ASSERT_GE(sorter.run_files().size(), 2u)
        << "expected multiple runs to force merge_runs";
    ASSERT_EQ(sorter.total_records(), kN);
    ASSERT_FALSE(sorter.fits_in_memory())
        << "buffer should be too small to hold all records in memory";

    std::vector<Rec3> out;
    out.reserve(kN);
    sorter.stream_sorted([&](const Rec3& r) { out.push_back(r); });

    // (b) count-preserving
    EXPECT_EQ(out.size(), kN);
    // (a) globally sorted
    EXPECT_TRUE(std::is_sorted(out.begin(), out.end()));
    // (c) equal to a reference std::sort of the same input
    std::vector<Rec3> expected = input;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(out, expected);

    fs::remove_all(tmp);
}

// Memory-bound invariant: feeding add_memory_records() a vector LARGER than the
// sort buffer must NOT keep it fully resident through the external sort. The
// stream_external path is required to chunk/sort/flush an oversized
// memory_records_ into bounded run files so that resident_memory_bytes() never
// exceeds buffer_size_. This pins the "bounded by construction" contract that
// the run_classic driver already respects (it only ever hands a <= 64 MB
// remainder), so a future direct caller cannot silently OOM by overwhelming the
// in-memory tail. Correctness (globally sorted, count-preserving) is asserted
// alongside the bound.
TEST(ExternalRecordSortSpill, MemoryRecordsRespectBufferCeiling) {
    const std::string tmp =
        (fs::temp_directory_path() / "mdb_ext_sort_spill_membound").string();
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    // 60k records * 24 B = 1.44 MB of in-memory records, fed through a 256 KB
    // buffer -> the resident tail is ~5.6x the buffer if held whole.
    constexpr size_t kN = 60000;
    const size_t kBuffer = 256u * 1024u;  // 256 KB

    std::mt19937_64 rng(424242);
    std::vector<Rec3> input;
    input.reserve(kN);
    for (size_t i = 0; i < kN; ++i) {
        Rec3 r{};
        r[0] = rng() & 0x00FFFFFFFFFFFFFFULL;
        r[1] = rng() & 0x00FFFFFFFFFFFFFFULL;
        r[2] = rng() & 0x00FFFFFFFFFFFFFFULL;
        input.push_back(r);
    }

    GQL::ExternalRecordSort<3> sorter(tmp, /*buffer_size=*/kBuffer);

    // The whole dataset arrives only as in-memory records (no add_run): this is
    // exactly the path the production driver caps at 64 MB but the API contract
    // must bound regardless of caller discipline.
    std::vector<Rec3> mem = input;
    sorter.add_memory_records(std::move(mem));

    ASSERT_EQ(sorter.total_records(), kN);
    ASSERT_FALSE(sorter.fits_in_memory())
        << "buffer should be too small to hold all records in memory";
    ASSERT_GT(sorter.total_records() * GQL::ExternalRecordSort<3>::RECORD_SIZE,
              sorter.buffer_size())
        << "test must feed strictly more than one buffer of memory records";

    std::vector<Rec3> out;
    out.reserve(kN);
    sorter.stream_sorted([&](const Rec3& r) { out.push_back(r); });

    // THE BOUND: after the sort drains, nothing oversized is held resident.
    EXPECT_LE(sorter.resident_memory_bytes(), sorter.buffer_size())
        << "memory_records_ must be flushed in bounded chunks, not held whole";

    // Correctness is preserved.
    EXPECT_EQ(out.size(), kN);
    EXPECT_TRUE(std::is_sorted(out.begin(), out.end()));
    std::vector<Rec3> expected = input;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(out, expected);

    fs::remove_all(tmp);
}

// Variant with an even smaller buffer and more runs, stressing the K-way merge
// with many concurrently-open run streams (the path bounded by the task_arena
// fan-out cap). Same correctness invariants.
TEST(ExternalRecordSortSpill, ManyRunsKWayMergeStress) {
    const std::string tmp =
        (fs::temp_directory_path() / "mdb_ext_sort_spill_test_many").string();
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    constexpr size_t kN = 100000;
    constexpr size_t kChunk = 2000;     // -> 50 run files

    std::mt19937_64 rng(98765);
    std::vector<Rec3> input;
    input.reserve(kN);
    for (size_t i = 0; i < kN; ++i) {
        Rec3 r{};
        // Narrow key domain on field 0 to produce duplicate keys across runs,
        // exercising the merge tie-break and adjacent-record ordering.
        r[0] = rng() % 5000;
        r[1] = rng() & 0x00FFFFFFFFFFFFFFULL;
        r[2] = rng() & 0x00FFFFFFFFFFFFFFULL;
        input.push_back(r);
    }

    GQL::ExternalRecordSort<3> sorter(tmp, /*buffer_size=*/256u * 1024u);

    for (size_t off = 0; off < kN; off += kChunk) {
        size_t end = std::min(off + kChunk, kN);
        std::vector<Rec3> chunk(input.begin() + off, input.begin() + end);
        std::string path = tmp + "/spill_" + std::to_string(off / kChunk);
        write_spill_run(path, chunk);
        sorter.add_run(path, chunk.size());
    }

    ASSERT_GE(sorter.run_files().size(), 10u);
    ASSERT_FALSE(sorter.fits_in_memory());

    std::vector<Rec3> out;
    out.reserve(kN);
    sorter.stream_sorted([&](const Rec3& r) { out.push_back(r); });

    EXPECT_EQ(out.size(), kN);
    EXPECT_TRUE(std::is_sorted(out.begin(), out.end()));
    std::vector<Rec3> expected = input;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(out, expected);

    fs::remove_all(tmp);
}
