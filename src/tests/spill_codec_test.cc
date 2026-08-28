// Unit tests for SpillCodec (SpillWriter / SpillReader).
//
// Scope: roundtrip fidelity for NONE and LZ4, legacy-format detection,
// compression ratio sanity on sorted integer data, empty-file handling.
//
// These tests use /tmp for ephemeral files so they do not pollute the DB
// directory. They follow the existing src/tests/*_test.cc convention:
// plain main(), try/catch, exit code 0 on success.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>  // getpid()
#include <vector>

#include "graph_models/gql/projection/spill_codec.h"

namespace fs = std::filesystem;
using GQL::SpillCompression;
using GQL::SpillFormat;
using GQL::SpillReader;
using GQL::SpillWriter;

namespace {

int   g_tests   = 0;
int   g_passed  = 0;
std::vector<std::string> g_failures;

#define CHECK(cond, name)                                                         \
    do {                                                                          \
        ++g_tests;                                                                \
        if (cond) {                                                               \
            ++g_passed;                                                           \
            std::cout << "  [PASS] " << (name) << "\n";                           \
        } else {                                                                  \
            g_failures.emplace_back(name);                                        \
            std::cerr << "  [FAIL] " << (name) << " (" << __FILE__ << ":"         \
                      << __LINE__ << ")\n";                                       \
        }                                                                         \
    } while (0)

/// Returns a unique temp path per test to avoid collisions across runs.
std::string make_temp_path(const std::string& tag) {
    std::string path = "/tmp/spill_codec_test_" + tag + "_" +
                       std::to_string(::getpid()) + "_" +
                       std::to_string(std::rand()) + ".bin";
    return path;
}

/// Writes raw bytes to `path` without the SpillCodec header. Used to simulate
/// a legacy spill file produced before SpillWriter was introduced, proving
/// that SpillReader's backward-compat branch works.
void write_legacy_raw(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
}

/// Reads full file into a byte vector. Used to verify file sizes for ratio
/// sanity checks.
std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<std::size_t>(sz));
    in.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

/// Creates sorted uint64 tuples representative of real spill content. Sorted
/// tuples compress well with LZ4 because consecutive records share leading
/// bytes (slow-growing from_id, clustered to_id, monotonic edge_id), which
/// is exactly the redundancy an LZ77-family codec exploits.
std::vector<uint64_t> make_sorted_tuples(std::size_t num_records, std::size_t record_width) {
    std::mt19937_64 rng(0xDEADBEEFCAFE1234ULL);
    std::vector<uint64_t> flat(num_records * record_width);
    for (std::size_t i = 0; i < num_records; ++i) {
        // Encode like a graph edge record: from_id grows slowly,
        // to_id varies more, edge_id monotonic. This mimics what
        // a sorted from_to_edge index looks like after external sort.
        flat[i * record_width + 0] = (i / 100) + 1;                  // from_id
        if (record_width >= 2) flat[i * record_width + 1] = rng() % 10000; // to_id
        if (record_width >= 3) flat[i * record_width + 2] = i;        // edge_id
    }
    // Post-sort by the full tuple to match real spill state.
    auto begin = flat.begin();
    auto end   = flat.end();
    auto cmp   = [record_width](auto a, auto b) {
        const uint64_t* pa = &(*a);
        const uint64_t* pb = &(*b);
        for (std::size_t k = 0; k < record_width; ++k) {
            if (pa[k] != pb[k]) return pa[k] < pb[k];
        }
        return false;
    };
    (void)begin; (void)end; (void)cmp;  // We skip actual re-sort since our
                                        // construction is already monotonic-ish
                                        // and the compressibility test doesn't
                                        // require strict sort order.
    return flat;
}

// ---------------------------------------------------------------------------
// Test: roundtrip with NONE compression
// ---------------------------------------------------------------------------
void test_roundtrip_none() {
    std::cout << "Test: roundtrip NONE" << std::endl;
    std::string path = make_temp_path("none_rt");

    // Write 10 KB of known pattern.
    std::vector<uint8_t> payload(10 * 1024);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<uint8_t>(i * 37 + 13);
    }

    {
        SpillWriter w(path, SpillCompression::NONE);
        w.write(payload.data(), payload.size());
        w.finalize();
    }

    // File should exist and contain: 8-byte header + payload.
    CHECK(fs::exists(path), "NONE: file created");
    CHECK(fs::file_size(path) == payload.size() + SpillFormat::HEADER_SIZE,
          "NONE: size matches header + payload");

    std::vector<uint8_t> out(payload.size());
    {
        SpillReader r(path);
        CHECK(r.compression() == SpillCompression::NONE, "NONE: reader detects NONE");
        CHECK(!r.legacy_format(),                        "NONE: reader says not legacy");
        std::size_t got = 0;
        while (got < out.size() && !r.eof()) {
            std::size_t n = r.read(out.data() + got, out.size() - got);
            if (n == 0) break;
            got += n;
        }
        CHECK(got == payload.size(), "NONE: all bytes read back");
    }
    CHECK(std::memcmp(out.data(), payload.data(), payload.size()) == 0,
          "NONE: byte-identical roundtrip");

    fs::remove(path);
}

// ---------------------------------------------------------------------------
// Test: roundtrip with LZ4 compression (only if HAS_LZ4)
// ---------------------------------------------------------------------------
void test_roundtrip_lz4() {
    std::cout << "Test: roundtrip LZ4" << std::endl;
#ifdef HAS_LZ4
    std::string path = make_temp_path("lz4_rt");

    // Use sorted uint64 tuples — the real workload pattern.
    auto flat = make_sorted_tuples(100'000, 3);
    std::size_t bytes = flat.size() * sizeof(uint64_t);

    {
        SpillWriter w(path, SpillCompression::LZ4);
        w.write(flat.data(), bytes);
        w.finalize();
    }

    CHECK(fs::exists(path), "LZ4: file created");

    // Read back and compare.
    std::vector<uint8_t> out(bytes);
    {
        SpillReader r(path);
        CHECK(r.compression() == SpillCompression::LZ4, "LZ4: reader detects LZ4");
        CHECK(!r.legacy_format(),                       "LZ4: reader says not legacy");
        std::size_t got = 0;
        while (got < out.size() && !r.eof()) {
            std::size_t n = r.read(out.data() + got, out.size() - got);
            if (n == 0) break;
            got += n;
        }
        CHECK(got == bytes, "LZ4: all bytes read back");
    }
    CHECK(std::memcmp(out.data(), flat.data(), bytes) == 0,
          "LZ4: byte-identical roundtrip");

    fs::remove(path);
#else
    std::cout << "  [SKIP] HAS_LZ4 not defined — skipping LZ4 tests" << std::endl;
#endif
}

// ---------------------------------------------------------------------------
// Test: backward compat — reader accepts legacy headerless files
// ---------------------------------------------------------------------------
void test_legacy_compat() {
    std::cout << "Test: legacy format (no header)" << std::endl;
    std::string path = make_temp_path("legacy");

    // Write raw bytes that do NOT start with the SpillFormat magic.
    // We pick a pattern whose first 4 bytes are not 'GSPL'.
    std::vector<uint8_t> payload(4096);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<uint8_t>(i ^ 0xA5);
    }
    // Force first 4 bytes to NOT match 'GSPL' (0x47 'G', 0x53 'S', 0x50 'P', 0x4C 'L' as bytes).
    payload[0] = 0x11; payload[1] = 0x22; payload[2] = 0x33; payload[3] = 0x44;
    write_legacy_raw(path, payload);

    std::vector<uint8_t> out(payload.size());
    {
        SpillReader r(path);
        CHECK(r.legacy_format(),                          "legacy: reader detects legacy");
        CHECK(r.compression() == SpillCompression::NONE,  "legacy: compression is NONE");
        std::size_t got = 0;
        while (got < out.size() && !r.eof()) {
            std::size_t n = r.read(out.data() + got, out.size() - got);
            if (n == 0) break;
            got += n;
        }
        CHECK(got == payload.size(), "legacy: all bytes read back");
    }
    CHECK(std::memcmp(out.data(), payload.data(), payload.size()) == 0,
          "legacy: byte-identical read");

    fs::remove(path);
}

// ---------------------------------------------------------------------------
// Test: compression ratio on sorted tuples
// ---------------------------------------------------------------------------
void test_compression_ratio() {
    std::cout << "Test: LZ4 compression ratio on sorted tuples" << std::endl;
#ifdef HAS_LZ4
    // 1M records × 24 bytes = 24 MB raw
    auto flat = make_sorted_tuples(1'000'000, 3);
    std::size_t raw_bytes = flat.size() * sizeof(uint64_t);

    std::string none_path = make_temp_path("ratio_none");
    std::string lz4_path  = make_temp_path("ratio_lz4");

    {
        SpillWriter w(none_path, SpillCompression::NONE);
        w.write(flat.data(), raw_bytes);
        w.finalize();
    }
    {
        SpillWriter w(lz4_path, SpillCompression::LZ4);
        w.write(flat.data(), raw_bytes);
        w.finalize();
    }

    std::size_t none_sz = fs::file_size(none_path);
    std::size_t lz4_sz  = fs::file_size(lz4_path);

    double ratio = static_cast<double>(none_sz) / static_cast<double>(lz4_sz);
    std::cout << "  NONE file: " << none_sz << " bytes\n"
              << "  LZ4 file : " << lz4_sz  << " bytes\n"
              << "  Ratio    : " << ratio   << "x\n";

    // Require at least 2x — a very conservative lower bound. Real runs
    // typically hit 3-5x on sorted integer tuples. Anything < 2x means
    // something is very wrong with the compressor.
    CHECK(ratio >= 2.0, "LZ4 achieves at least 2x compression on sorted tuples");

    fs::remove(none_path);
    fs::remove(lz4_path);
#else
    std::cout << "  [SKIP] HAS_LZ4 not defined" << std::endl;
#endif
}

// ---------------------------------------------------------------------------
// Test: empty file (header only)
// ---------------------------------------------------------------------------
void test_empty_file() {
    std::cout << "Test: empty spill (header only, no payload)" << std::endl;
    std::string path = make_temp_path("empty");

    {
        SpillWriter w(path, SpillCompression::NONE);
        w.finalize();
    }
    CHECK(fs::exists(path), "empty: file created");

    {
        SpillReader r(path);
        uint8_t buf[16];
        std::size_t n = r.read(buf, sizeof(buf));
        CHECK(n == 0,    "empty: read returns 0");
        CHECK(r.eof(),   "empty: eof() is true");
    }
    fs::remove(path);

#ifdef HAS_LZ4
    std::string path2 = make_temp_path("empty_lz4");
    {
        SpillWriter w(path2, SpillCompression::LZ4);
        w.finalize();
    }
    {
        SpillReader r(path2);
        uint8_t buf[16];
        std::size_t n = r.read(buf, sizeof(buf));
        CHECK(n == 0,    "empty LZ4: read returns 0");
        CHECK(r.eof(),   "empty LZ4: eof() is true");
    }
    fs::remove(path2);
#endif
}

// ---------------------------------------------------------------------------
// Test: multiple writes accumulate correctly
// ---------------------------------------------------------------------------
void test_multiple_writes() {
    std::cout << "Test: multiple writes roundtrip" << std::endl;
    std::string path = make_temp_path("multi");

#ifdef HAS_LZ4
    SpillCompression comp = SpillCompression::LZ4;
#else
    SpillCompression comp = SpillCompression::NONE;
#endif

    std::vector<uint8_t> part1(1024), part2(2048), part3(512);
    for (std::size_t i = 0; i < part1.size(); ++i) part1[i] = static_cast<uint8_t>(i);
    for (std::size_t i = 0; i < part2.size(); ++i) part2[i] = static_cast<uint8_t>(i + 1);
    for (std::size_t i = 0; i < part3.size(); ++i) part3[i] = static_cast<uint8_t>(i + 2);

    {
        SpillWriter w(path, comp);
        w.write(part1.data(), part1.size());
        w.write(part2.data(), part2.size());
        w.write(part3.data(), part3.size());
        w.finalize();
    }

    std::size_t total = part1.size() + part2.size() + part3.size();
    std::vector<uint8_t> expected;
    expected.reserve(total);
    expected.insert(expected.end(), part1.begin(), part1.end());
    expected.insert(expected.end(), part2.begin(), part2.end());
    expected.insert(expected.end(), part3.begin(), part3.end());

    std::vector<uint8_t> out(total);
    {
        SpillReader r(path);
        std::size_t got = 0;
        while (got < out.size() && !r.eof()) {
            std::size_t n = r.read(out.data() + got, out.size() - got);
            if (n == 0) break;
            got += n;
        }
        CHECK(got == total, "multi: all bytes read back");
    }
    CHECK(std::memcmp(out.data(), expected.data(), total) == 0,
          "multi: byte-identical concatenation");

    fs::remove(path);
}

} // anonymous namespace

int main() {
    std::cout << "=== SpillCodec tests ===" << std::endl;

    try {
        test_roundtrip_none();
        test_roundtrip_lz4();
        test_legacy_compat();
        test_compression_ratio();
        test_empty_file();
        test_multiple_writes();
    } catch (const std::exception& e) {
        std::cerr << "UNCAUGHT EXCEPTION: " << e.what() << std::endl;
        return 2;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << g_passed << "/" << g_tests << std::endl;
    if (!g_failures.empty()) {
        std::cout << "FAILED:" << std::endl;
        for (const auto& f : g_failures) {
            std::cout << "  - " << f << std::endl;
        }
        return 1;
    }
    return 0;
}
