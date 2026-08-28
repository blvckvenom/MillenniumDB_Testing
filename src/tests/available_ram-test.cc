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

bool test_core_env_exact_value() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/"4096",
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT_EQ(r.bytes, 4ULL * 1024 * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV);
    return false;
}

bool test_core_env_below_floor_clamps() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/"1",
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT_EQ(r.bytes, 256ULL * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV);  // Still tagged ENV even after clamp.
    return false;
}

bool test_core_env_whitespace_tolerated() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/0,
        /*env_value=*/"1024 ",  // trailing space
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT_EQ(r.bytes, 1024ULL * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV);
    return false;
}

bool test_core_env_non_numeric_invalid() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/"abc",
        /*floor=*/256ULL * 1024 * 1024);
    // Falls back to adaptive (12 GB) but tagged ENV_INVALID.
    EXPECT_EQ(r.bytes, 12ULL * 1024 * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV_INVALID);
    return false;
}

bool test_core_env_zero_invalid() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/"0",
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT_EQ(r.bytes, 12ULL * 1024 * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV_INVALID);
    return false;
}

bool test_core_env_negative_invalid() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/"-100",
        /*floor=*/256ULL * 1024 * 1024);
    // strtoull on "-100" sets errno=ERANGE (on most libcs) or returns a huge
    // wrapped value — both rejected by the `errno == 0` check.
    EXPECT(r.source == AdaptiveBufferResult::ENV_INVALID);
    return false;
}

bool test_core_env_overflow_invalid() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/"99999999999999999999999999",
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV_INVALID);
    return false;
}

bool test_core_env_trailing_garbage_invalid() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/"1024abc",
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV_INVALID);
    return false;
}

bool test_core_env_empty_invalid() {
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_core(
        /*mem_available=*/16ULL * 1024 * 1024 * 1024,
        /*env_value=*/"",
        /*floor=*/256ULL * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV_INVALID);
    return false;
}

bool test_public_wrapper_with_env() {
    ScopedEnv env("MDB_SORT_BUFFER_MB", "512");
    size_t result = compute_adaptive_sort_buffer();
    EXPECT_EQ(result, 512ULL * 1024 * 1024);
    return false;
}

bool test_public_wrapper_result_tag() {
    ScopedEnv env("MDB_SORT_BUFFER_MB", "512");
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_result();
    EXPECT_EQ(r.bytes, 512ULL * 1024 * 1024);
    EXPECT(r.source == AdaptiveBufferResult::ENV);
    return false;
}

// Regression guard: the floor must always equal 256 MB, matching the
// pre-adaptive hardcoded constant. The floor is what guarantees that no
// workload can end up with a smaller sort buffer than it had before the
// adaptive sizing existed; accidentally raising or lowering it silently
// breaks that guarantee.
bool test_default_floor_is_exactly_256_MB() {
    EXPECT_EQ(DEFAULT_SORT_BUFFER_MIN, 256ULL * 1024 * 1024);
    return false;
}

// Regression guard: invocation with an unset env var in a clean
// environment exercises the real /proc/meminfo read path end-to-end
// on Linux. Verifies only that the result is >= floor (the specific
// value depends on the host).
bool test_real_env_returns_at_least_floor() {
    ScopedEnv clear("MDB_SORT_BUFFER_MB", nullptr);
    size_t result = compute_adaptive_sort_buffer();
    EXPECT(result >= DEFAULT_SORT_BUFFER_MIN);
    return false;
}

// Regression guard: the source=env tag persists even when the resolved
// bytes are clamped to the floor (i.e., ENV wins semantically even if
// numerically indistinguishable from adaptive fallback at the floor).
bool test_env_tag_preserved_at_floor_clamp() {
    ScopedEnv env("MDB_SORT_BUFFER_MB", "10");  // 10 MB, well below 256 MB floor
    AdaptiveBufferResult r = compute_adaptive_sort_buffer_result();
    EXPECT_EQ(r.bytes, DEFAULT_SORT_BUFFER_MIN);
    EXPECT(r.source == AdaptiveBufferResult::ENV);
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
        {"test_core_env_exact_value",             &test_core_env_exact_value},
        {"test_core_env_below_floor_clamps",      &test_core_env_below_floor_clamps},
        {"test_core_env_whitespace_tolerated",    &test_core_env_whitespace_tolerated},
        {"test_core_env_non_numeric_invalid",     &test_core_env_non_numeric_invalid},
        {"test_core_env_zero_invalid",            &test_core_env_zero_invalid},
        {"test_core_env_negative_invalid",        &test_core_env_negative_invalid},
        {"test_core_env_overflow_invalid",        &test_core_env_overflow_invalid},
        {"test_core_env_trailing_garbage_invalid", &test_core_env_trailing_garbage_invalid},
        {"test_core_env_empty_invalid",           &test_core_env_empty_invalid},
        {"test_public_wrapper_with_env",          &test_public_wrapper_with_env},
        {"test_public_wrapper_result_tag",        &test_public_wrapper_result_tag},
        {"test_default_floor_is_exactly_256_MB",  &test_default_floor_is_exactly_256_MB},
        {"test_real_env_returns_at_least_floor",  &test_real_env_returns_at_least_floor},
        {"test_env_tag_preserved_at_floor_clamp", &test_env_tag_preserved_at_floor_clamp},
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
