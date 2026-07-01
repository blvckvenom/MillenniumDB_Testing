#pragma once
// Single source of truth for the <feature>_vram_calib.txt file format.
//
// gnn_train writes this file after measuring the real peak VRAM of a training
// run (the dynamic-cache calibration): it records the max-safe L1 GPU cache for
// this machine+workload. gnn_build_feature_store reads it back when the caller
// passes autoCache:true, so a later build auto-sizes the L1 cache to what the
// previous train measured. Keeping the format (key names + line layout) in one
// place prevents the writer and reader from silently drifting apart.
//
// Format: one "key=value" pair per line, integer values (whole MB), trailing
// newline after each line. Unknown keys are ignored on read; missing keys read
// back as 0. Only recommended_l1_cache_mb is load-bearing for autoCache; the
// other three are informational (logged at apply time).

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace mdb::gnn {

struct VramCalib {
    uint64_t recommended_l1_cache_mb = 0;  // max-safe L1 GPU cache for autoCache
    uint64_t measured_peak_mb        = 0;  // peak VRAM observed during train
    uint64_t measured_reserve_mb     = 0;  // peak minus L1-resident (non-cache use)
    uint64_t total_vram_mb           = 0;  // device total VRAM at calibration time
};

// Serialize to the canonical text. Byte-identical to the layout gnn_train has
// written since the dynamic-cache calibration landed.
inline std::string format_vram_calib_text(const VramCalib& c) {
    std::ostringstream os;
    os << "recommended_l1_cache_mb=" << c.recommended_l1_cache_mb << "\n"
       << "measured_peak_mb="        << c.measured_peak_mb        << "\n"
       << "measured_reserve_mb="     << c.measured_reserve_mb     << "\n"
       << "total_vram_mb="           << c.total_vram_mb           << "\n";
    return os.str();
}

// Parse key=value lines. Tolerates trailing CR/whitespace and extra/unknown
// lines. Returns nullopt only when no recognized key was found (treat as "no
// usable calibration").
inline std::optional<VramCalib> parse_vram_calib_text(const std::string& text) {
    VramCalib c;
    bool any = false;
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        auto rtrim = [](std::string& s) {
            while (!s.empty() &&
                   (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
                s.pop_back();
            }
        };
        rtrim(key);
        rtrim(val);
        uint64_t n = 0;
        try {
            n = static_cast<uint64_t>(std::stoull(val));
        } catch (...) {
            continue;  // non-numeric value: skip this line, keep parsing
        }
        if      (key == "recommended_l1_cache_mb") { c.recommended_l1_cache_mb = n; any = true; }
        else if (key == "measured_peak_mb")        { c.measured_peak_mb        = n; any = true; }
        else if (key == "measured_reserve_mb")     { c.measured_reserve_mb     = n; any = true; }
        else if (key == "total_vram_mb")           { c.total_vram_mb           = n; any = true; }
    }
    if (!any) return std::nullopt;
    return c;
}

// Write the calibration file. Returns false if the file could not be opened or
// the stream errored.
inline bool write_vram_calib(const std::filesystem::path& path, const VramCalib& c) {
    std::ofstream f(path);
    if (!f) return false;
    f << format_vram_calib_text(c);
    return static_cast<bool>(f);
}

// Read and parse the calibration file. Returns nullopt if the file is absent,
// unreadable, or contains no recognized keys.
inline std::optional<VramCalib> read_vram_calib(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) return std::nullopt;
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_vram_calib_text(ss.str());
}

}  // namespace mdb::gnn
