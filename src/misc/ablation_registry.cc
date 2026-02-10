#include "misc/ablation_registry.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <ostream>
#include <vector>

namespace {

struct Entry {
    std::string name;
    std::string resolved;
    std::string raw;        // what the environment actually held, if anything
    const char* source;     // "env" | "default"
    bool        honoured;   // false when the build cannot act on it
    std::string note;       // why it was rejected, or why it is inert
};

// A vector and not a map: the ORDER in which switches resolve is itself
// information. It says which decisions the run reached, and reading them in
// sequence reconstructs the path taken through the build.
std::vector<Entry>& entries() {
    static std::vector<Entry> e;
    return e;
}

std::mutex& lock() {
    static std::mutex m;
    return m;
}

// stderr and NOT stdout. `mdb list-projections` and `mdb dump` write DATA to
// stdout, and a decision line mixed into that corrupts anything that pipes it.
// The measurement scripts capture 2>&1, so the line still lands in every run
// log, and [RADIX], the closest analogue, already goes here.
void emit(const Entry& e) {
    std::cerr << "[ABLATION] " << e.name << '=' << e.resolved
              << " source=" << e.source;
    if (!e.raw.empty() && e.raw != e.resolved) {
        // The line that makes a typo visible: the operator wrote one thing and
        // the code settled on another.
        std::cerr << " raw=\"" << e.raw << '"';
    }
    if (!e.honoured) {
        std::cerr << " honoured=no";
    }
    if (!e.note.empty()) {
        std::cerr << " note=" << e.note;
    }
    std::cerr << std::endl;
}

// Resolution happens ONCE per name. A second call returns the first answer, so
// two call sites reading the same switch cannot disagree, and a switch cannot
// change meaning halfway through a long run.
// Returns BY VALUE, and that is the whole point. An earlier version handed back
// a `const Entry&` into the vector: the lock releases when this returns, so a
// concurrent resolve could push_back, reallocate, and leave the caller reading
// freed memory between the return and `.resolved`. It never fired because most
// call sites are function-local statics, but Ablation:: is now reached from a
// dozen files that spawn threads, so the window is real and the copy is one
// short string.
std::string record(const std::string& name, std::string resolved,
                   std::string raw, const char* source,
                   bool honoured = true, std::string note = {}) {
    std::lock_guard<std::mutex> g(lock());
    auto& all = entries();
    auto it = std::find_if(all.begin(), all.end(),
                           [&](const Entry& e) { return e.name == name; });
    if (it != all.end()) {
        return it->resolved;
    }
    all.push_back(Entry{name, std::move(resolved), std::move(raw), source,
                        honoured, std::move(note)});
    emit(all.back());
    return all.back().resolved;
}

const char* env_of(const char* name) { return std::getenv(name); }

}  // namespace

namespace Ablation {

bool flag(const char* name, bool fallback) {
    const char* v = env_of(name);
    if (v == nullptr) {
        return record(name, fallback ? "true" : "false", {}, "default") == "true";
    }
    std::string raw(v);
    std::string low = raw;
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool on = !(low == "0" || low == "false" || low == "no" || low.empty());
    return record(name, on ? "true" : "false", raw, "env") == "true";
}

long number(const char* name, long fallback) {
    const char* v = env_of(name);
    if (v == nullptr) {
        return std::stol(record(name, std::to_string(fallback), {}, "default"));
    }
    std::string raw(v);
    char* fin = nullptr;
    const long parsed = std::strtol(v, &fin, 10);
    if (fin == v || (fin != nullptr && *fin != '\0')) {
        // Falls back AND says so. Silently treating "eight" as 0 is the kind of
        // thing that turns an ablation arm into a different experiment.
        return std::stol(record(name, std::to_string(fallback), raw, "default",
                                true, "unparseable"));
    }
    return std::stol(record(name, std::to_string(parsed), raw, "env"));
}

std::string choice(const char* name, const char* fallback,
                   std::initializer_list<const char*> accepted) {
    const char* v = env_of(name);
    if (v == nullptr) {
        return record(name, fallback, {}, "default");
    }
    std::string raw(v);
    for (const char* a : accepted) {
        if (raw == a) {
            return record(name, raw, raw, "env");
        }
    }
    // The defect this closes: a switch that accepts anything that is not the
    // other branch turns a typo into a valid-looking arm.
    return record(name, fallback, raw, "default", true, "unrecognised");
}

std::string text(const char* name, const char* fallback) {
    const char* v = env_of(name);
    return v == nullptr ? record(name, fallback ? fallback : "", {}, "default")
                        : record(name, v, v, "env");
}

void inert(const char* name, const char* reason) {
    const char* v = env_of(name);
    record(name, v ? v : "(unset)", v ? v : "", v ? "env" : "default",
           /*honoured=*/false, reason ? reason : "not compiled in");
}

void report(std::ostream& os) {
    std::lock_guard<std::mutex> g(lock());
    for (const auto& e : entries()) {
        os << "[ABLATION] " << e.name << '=' << e.resolved
           << " source=" << e.source;
        if (!e.raw.empty() && e.raw != e.resolved) os << " raw=\"" << e.raw << '"';
        if (!e.honoured) os << " honoured=no";
        if (!e.note.empty()) os << " note=" << e.note;
        os << '\n';
    }
}

std::size_t resolved_count() {
    std::lock_guard<std::mutex> g(lock());
    return entries().size();
}

}  // namespace Ablation
