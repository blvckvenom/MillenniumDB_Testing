// src/gnn/training/batch_timing_log.h
#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace mdb::gnn {

/**
 * @brief Per-batch stage timing record (microseconds).
 *
 * Phase 0 profile instrumentation. All values in microseconds.
 * rmap_lookup_us is a SUB-counter already included in load_features_us +
 * active_us; tracked separately to quantify Phase 1 candidate savings.
 */
struct BatchTiming {
    uint64_t batch_id;
    uint8_t  split;            // 0=TRAIN, 1=VAL, 2=TEST
    uint64_t sample_read_us;
    uint64_t load_features_us;
    uint64_t l1_us;            // sub of load_features_us
    uint64_t l2_us;
    uint64_t l3_us;
    uint64_t l4_us;
    uint64_t assembler_kernel_us;  // Phase A: feature_store->load_batch_features (subsumes l1..l4)
    uint64_t rmap_lookup_us;   // sub-counter, NOT additive
    uint64_t active_us;
    uint64_t edge_us;
    uint64_t h2d_us;
    uint64_t forward_us;
    uint64_t backward_us;
};

/**
 * @brief Append-only CSV logger for per-batch timings.
 *
 * Thread-safe via internal mutex. Flushes every flush_interval entries
 * and on destruction. Writes directly to the user-supplied path (truncate +
 * append); partial CSVs after a crash are tolerated by the analysis pipeline.
 */
class BatchTimingLog {
public:
    /**
     * Open log at @p path; creates parent dir if missing. Writes CSV header
     * if file is new (does not check; truncates and overwrites).
     */
    explicit BatchTimingLog(const std::string& path, size_t flush_interval = 64);

    /** Destructor flushes remaining entries. */
    ~BatchTimingLog();

    BatchTimingLog(const BatchTimingLog&)            = delete;
    BatchTimingLog& operator=(const BatchTimingLog&) = delete;

    /** Append a timing record. Thread-safe. */
    void append(const BatchTiming& t);

    /** Force flush to disk. Thread-safe. */
    void flush();

    /** Number of records buffered (not yet flushed). */
    size_t pending() const;

private:
    void write_csv_header_();
    void write_buffer_locked_();

    std::string                path_;
    size_t                     flush_interval_;
    mutable std::mutex         mu_;
    std::ofstream              out_;
    std::vector<BatchTiming>   buffer_;
};

} // namespace mdb::gnn
