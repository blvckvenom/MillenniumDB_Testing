// src/gnn/training/batch_timing_log.cc
#include "gnn/training/batch_timing_log.h"

#include <filesystem>
#include <stdexcept>

namespace mdb::gnn {

namespace fs = std::filesystem;

BatchTimingLog::BatchTimingLog(const std::string& path, size_t flush_interval)
    : path_(path), flush_interval_(flush_interval)
{
    fs::path p(path_);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }
    out_.open(path_, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!out_.is_open()) {
        throw std::runtime_error(
            "BatchTimingLog: cannot open '" + path_ + "' for writing");
    }
    write_csv_header_();
}

BatchTimingLog::~BatchTimingLog() {
    try { flush(); } catch (...) {}
    if (out_.is_open()) out_.close();
}

void BatchTimingLog::append(const BatchTiming& t) {
    std::lock_guard<std::mutex> g(mu_);
    buffer_.push_back(t);
    if (buffer_.size() >= flush_interval_) {
        write_buffer_locked_();
    }
}

void BatchTimingLog::flush() {
    std::lock_guard<std::mutex> g(mu_);
    write_buffer_locked_();
    out_.flush();
}

size_t BatchTimingLog::pending() const {
    std::lock_guard<std::mutex> g(mu_);
    return buffer_.size();
}

void BatchTimingLog::write_csv_header_() {
    out_ << "batch_id,split,sample_read_us,load_features_us,"
            "l1_us,l2_us,l3_us,l4_us,rmap_lookup_us,active_us,edge_us,"
            "h2d_us,forward_us,backward_us\n";
}

void BatchTimingLog::write_buffer_locked_() {
    for (const auto& t : buffer_) {
        out_ << t.batch_id << ','
             << static_cast<int>(t.split) << ','
             << t.sample_read_us << ','
             << t.load_features_us << ','
             << t.l1_us << ',' << t.l2_us << ','
             << t.l3_us << ',' << t.l4_us << ','
             << t.rmap_lookup_us << ','
             << t.active_us << ',' << t.edge_us << ','
             << t.h2d_us << ',' << t.forward_us << ','
             << t.backward_us << '\n';
    }
    buffer_.clear();
}

} // namespace mdb::gnn
