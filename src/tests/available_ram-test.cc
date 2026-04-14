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

// ─── Harness ──────────────────────────────────────────────────────

int main() {
    std::vector<std::pair<const char*, TestFunction*>> tests = {
        {"test_meminfo_normal", &test_meminfo_normal},
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
