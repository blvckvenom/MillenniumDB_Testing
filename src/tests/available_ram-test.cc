#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "misc/available_ram.h"

// Test semantics (matches src/tests/iri_prefixes-test.cc convention):
//   test function returns `true` = FAIL, `false` = PASS
//   main returns 0 on all-pass, 1 if any failed.

typedef bool TestFunction();

// Assertion helpers: on failure, log and return `true` (= FAIL) from the
// enclosing function. Caller must return bool.
#define EXPECT(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __func__ << " line " << __LINE__ \
                  << ": " << #cond << "\n"; \
        return true; \
    } \
} while (0)

#define EXPECT_EQ(a, b) do { \
    auto _aa = (a); auto _bb = (b); \
    if (!(_aa == _bb)) { \
        std::cerr << "FAIL " << __func__ << " line " << __LINE__ \
                  << ": " << #a << " (" << _aa << ") != " \
                  << #b << " (" << _bb << ")\n"; \
        return true; \
    } \
} while (0)

// ─── Fixture helpers ───────────────────────────────────────────────

static std::filesystem::path fixture_dir() {
    auto dir = std::filesystem::temp_directory_path() / "mdb_available_ram_test";
    std::filesystem::create_directories(dir);
    return dir;
}

static std::string write_fixture(const std::string& name, const std::string& contents) {
    auto path = fixture_dir() / name;
    std::ofstream f(path);
    f << contents;
    f.close();
    return path.string();
}

// RAII env var setter for tests (restores previous value on destruction).
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* prev = std::getenv(name);
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }
    ~ScopedEnv() {
        if (had_prev_) {
            setenv(name_, prev_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }
private:
    const char* name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

// ─── Tests ────────────────────────────────────────────────────────

// First test: MemAvailable parses correctly from a well-formed fixture.
bool test_meminfo_normal() {
    std::string path = write_fixture("meminfo_normal.txt",
        "MemTotal:     16777216 kB\n"
        "MemFree:      1048576 kB\n"
        "MemAvailable: 12345678 kB\n");

    uint64_t result = get_mem_available_from(path.c_str());
    EXPECT_EQ(result, static_cast<uint64_t>(12345678) * 1024);
    return false;
}

bool test_meminfo_missing_line() {
    std::string path = write_fixture("meminfo_missing.txt",
        "MemTotal:     16777216 kB\n"
        "MemFree:      1048576 kB\n");  // No MemAvailable line at all.

    EXPECT_EQ(get_mem_available_from(path.c_str()), uint64_t{0});
    return false;
}

bool test_meminfo_malformed() {
    std::string path = write_fixture("meminfo_malformed.txt",
        "MemAvailable: not_a_number kB\n");

    EXPECT_EQ(get_mem_available_from(path.c_str()), uint64_t{0});
    return false;
}

bool test_meminfo_empty() {
    std::string path = write_fixture("meminfo_empty.txt", "");
    EXPECT_EQ(get_mem_available_from(path.c_str()), uint64_t{0});
    return false;
}

bool test_meminfo_nonexistent() {
    EXPECT_EQ(get_mem_available_from("/nonexistent/path/meminfo"), uint64_t{0});
    return false;
}

bool test_meminfo_units_respected() {
    // MemAvailable is always reported in kB per /proc/meminfo convention;
    // make sure we multiply by 1024 correctly, not misread as bytes.
    std::string path = write_fixture("meminfo_kilo.txt",
        "MemAvailable: 1 kB\n");

    EXPECT_EQ(get_mem_available_from(path.c_str()), uint64_t{1024});
    return false;
}

bool test_core_floor_when_mem_zero() {
    // When MemAvailable=0 (non-Linux / sandbox) and no env var,
    // result must fall back to the floor.
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/0, /*env_value=*/nullptr, /*floor=*/256ULL * 1024 * 1024);
    EXPECT_EQ(r.bytes, 256ULL * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ADAPTIVE);
    return false;
}

bool test_core_adaptive_above_floor() {
    // 16 GB MemAvailable -> 12 GB buffer (75% of 16).
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/nullptr,
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT_EQ(r.bytes, 12ULL * 1024 * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ADAPTIVE);
    return false;
}

bool test_core_adaptive_below_floor_clamps() {
    // 300 MB MemAvailable -> 225 MB raw -> clamped to 256 MB floor.
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/300ULL * 1024 * 1024,
        /*env_value=*/nullptr,
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT_EQ(r.bytes, 256ULL * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ADAPTIVE);
    return false;
}

// ─── Harness ──────────────────────────────────────────────────────

int main() {
    std::vector<std::pair<const char*, TestFunction*>> tests = {
        {"test_meminfo_normal",           &test_meminfo_normal},
        {"test_meminfo_missing_line",     &test_meminfo_missing_line},
        {"test_meminfo_malformed",        &test_meminfo_malformed},
        {"test_meminfo_empty",            &test_meminfo_empty},
        {"test_meminfo_nonexistent",      &test_meminfo_nonexistent},
        {"test_meminfo_units_respected", &test_meminfo_units_respected},
        {"test_core_floor_when_mem_zero",         &test_core_floor_when_mem_zero},
        {"test_core_adaptive_above_floor",        &test_core_adaptive_above_floor},
        {"test_core_adaptive_below_floor_clamps", &test_core_adaptive_below_floor_clamps},
    };

    int failures = 0;
    for (auto& [name, fn] : tests) {
        if (fn()) {
            std::cerr << "[FAIL] " << name << "\n";
            ++failures;
        } else {
            std::cout << "[PASS] " << name << "\n";
        }
    }
    std::cout << (failures ? "FAILURES: " : "ALL PASS: ")
              << (tests.size() - failures) << "/" << tests.size() << "\n";
    return failures == 0 ? 0 : 1;
}
