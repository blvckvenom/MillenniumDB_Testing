#pragma once

/**
 * @file streaming_aggregator.h
 * @brief O(1) memory streaming aggregation over sorted edge data.
 *
 * This is the key component that enables memory-efficient COUNT aggregation.
 * By processing a pre-sorted stream of edges, we can compute exact COUNTs
 * with O(1) memory per group instead of O(N) for the hash-based approach.
 *
 * ## Memory Analysis
 *
 * The StreamingEdgeAggregator maintains state for only ONE group at a time:
 * - Current group key: 24 bytes (from, to, type)
 * - Count: 8 bytes
 * - Representative edge: 8 bytes
 * - Sum value: 8 bytes
 * - **Total: 48 bytes constant**
 *
 * Compare to ParallelEdgeDetector hash table:
 * - 35M unique pairs × 224 bytes = 7.84 GB
 *
 * ## Algorithm
 *
 * 1. Receive sorted stream of EdgeAggregationRecords
 * 2. For each record:
 *    - If same group as current: aggregate (increment count, update sum)
 *    - If new group: emit current group, start new group
 * 3. After stream ends: emit final group
 *
 * The sort guarantees all records of the same (from, to, type) are consecutive,
 * so we never need to revisit a group.
 *
 * @see external_edge_sort.h for the sort phase
 * @see edge_aggregation_record.h for record format
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "graph_models/gql/projection/edge_aggregation_record.h"
#include "query/procedure/builtin/project_procedure.h"  // For Aggregation enum

namespace GQL {

// Use Aggregation from Procedures namespace
using Procedures::Aggregation;

/**
 * @brief Callback type for emitting aggregated results.
 *
 * @param representative_edge_id The edge ID to use for property storage
 * @param count Number of parallel edges in this group
 * @param aggregated_value SUM value (for SUM mode) or COUNT as double
 */
using AggregationEmitCallback = std::function<void(
    uint64_t representative_edge_id,
    uint64_t count,
    double aggregated_value
)>;

/**
 * @brief Streaming COUNT/SUM/MIN/MAX aggregation over sorted edge stream.
 *
 * Maintains state for only ONE group at a time—the key insight that enables
 * O(1) memory aggregation over sorted data.
 *
 * ## Usage
 *
 * ```cpp
 * StreamingEdgeAggregator aggregator(Aggregation::COUNT,
 *     [&storage](uint64_t edge_id, uint64_t count, double value) {
 *         storage->add_edge_property(ObjectId(edge_id), count_key_id,
 *                                    Conversions::pack_int(count));
 *     });
 *
 * external_sort.stream_sorted([&aggregator](const EdgeAggregationRecord& rec) {
 *     aggregator.process(rec);
 * });
 *
 * aggregator.finalize();  // Emit final group
 * ```
 *
 * ## Thread Safety
 *
 * Not thread-safe. Designed for single-threaded streaming processing.
 */
class StreamingEdgeAggregator {
public:
    /**
     * @brief Constructs a streaming aggregator.
     *
     * @param strategy Aggregation strategy (COUNT, SUM, MIN, MAX)
     * @param emit Callback to invoke when a group is complete
     */
    StreamingEdgeAggregator(Aggregation strategy, AggregationEmitCallback emit)
        : strategy_(strategy)
        , emit_(std::move(emit))
        , has_current_group_(false)
        , current_from_(0)
        , current_to_(0)
        , current_type_(0)
        , count_(0)
        , representative_edge_id_(0)
        , sum_value_(0.0)
        , min_value_(std::numeric_limits<double>::max())
        , max_value_(std::numeric_limits<double>::lowest())
    {}

    /**
     * @brief Processes a single record from the sorted stream.
     *
     * Records must arrive in sorted order by (from, to, type).
     * If a record belongs to the current group, it's aggregated.
     * If it's a new group, the current group is emitted first.
     *
     * @param record The edge record to process
     */
    void process(const EdgeAggregationRecord& record) {
        if (!has_current_group_) {
            // First record ever
            start_new_group(record);
            return;
        }

        // Check if same group
        if (record.from_node == current_from_ &&
            record.to_node == current_to_ &&
            record.type_id == current_type_) {
            // Same group - aggregate
            aggregate_into_group(record);
        } else {
            // New group - emit current and start new
            emit_group();
            start_new_group(record);
        }
    }

    /**
     * @brief Finalizes aggregation and emits the last group.
     *
     * Must be called after all records have been processed.
     */
    void finalize() {
        if (has_current_group_) {
            emit_group();
            has_current_group_ = false;
        }
    }

    /**
     * @brief Gets the count of groups emitted.
     */
    uint64_t groups_emitted() const { return groups_emitted_; }

    /**
     * @brief Gets the total count of records processed.
     */
    uint64_t records_processed() const { return records_processed_; }

private:
    /**
     * @brief Starts tracking a new aggregation group.
     */
    void start_new_group(const EdgeAggregationRecord& record) {
        has_current_group_ = true;
        current_from_ = record.from_node;
        current_to_ = record.to_node;
        current_type_ = record.type_id;
        count_ = 1;
        representative_edge_id_ = record.edge_id;

        // Initialize aggregation values
        double property_value = unpack_bits_to_double(record.property_bits);
        sum_value_ = property_value;
        min_value_ = property_value;
        max_value_ = property_value;

        records_processed_++;
    }

    /**
     * @brief Aggregates a record into the current group.
     */
    void aggregate_into_group(const EdgeAggregationRecord& record) {
        count_++;
        records_processed_++;

        double property_value = unpack_bits_to_double(record.property_bits);

        switch (strategy_) {
            case Aggregation::COUNT:
                // Just counting, no property needed
                break;

            case Aggregation::SUM:
                sum_value_ += property_value;
                break;

            case Aggregation::MIN:
                if (property_value < min_value_) {
                    min_value_ = property_value;
                    representative_edge_id_ = record.edge_id;
                }
                break;

            case Aggregation::MAX:
                if (property_value > max_value_) {
                    max_value_ = property_value;
                    representative_edge_id_ = record.edge_id;
                }
                break;

            case Aggregation::SINGLE:
                // SINGLE should have been handled before external sort
                // (throws on duplicates during collection)
                break;
        }
    }

    /**
     * @brief Emits the current group via callback.
     */
    void emit_group() {
        if (!has_current_group_) return;

        double aggregated_value = 0.0;

        switch (strategy_) {
            case Aggregation::COUNT:
                aggregated_value = static_cast<double>(count_);
                break;

            case Aggregation::SUM:
                aggregated_value = sum_value_;
                break;

            case Aggregation::MIN:
                aggregated_value = min_value_;
                break;

            case Aggregation::MAX:
                aggregated_value = max_value_;
                break;

            case Aggregation::SINGLE:
                aggregated_value = 0.0;
                break;
        }

        emit_(representative_edge_id_, count_, aggregated_value);
        groups_emitted_++;
    }

    Aggregation strategy_;
    AggregationEmitCallback emit_;

    // Current group state - only 48 bytes!
    bool has_current_group_;
    uint64_t current_from_;      // 8 bytes
    uint64_t current_to_;        // 8 bytes
    uint64_t current_type_;      // 8 bytes
    uint64_t count_;             // 8 bytes
    uint64_t representative_edge_id_;  // 8 bytes
    double sum_value_;           // 8 bytes
    double min_value_;           // 8 bytes (for MIN mode)
    double max_value_;           // 8 bytes (for MAX mode)

    // Statistics
    uint64_t groups_emitted_ = 0;
    uint64_t records_processed_ = 0;
};

/**
 * @brief Buffer for collecting edges before external sort.
 *
 * This is a thin wrapper around StreamingRecordBuffer<5> that provides
 * EdgeAggregationRecord-specific interface.
 *
 * ## Memory Model
 *
 * - Memory buffer: 64 MB default
 * - Spills to disk when threshold exceeded
 * - Peak memory: buffer_size + small overhead
 */
class EdgeAggregationBuffer {
public:
    /// @brief Default memory threshold (64 MB)
    static constexpr size_t DEFAULT_THRESHOLD = 64 * 1024 * 1024;

    /**
     * @brief Constructs an edge aggregation buffer.
     *
     * @param temp_file_prefix Prefix for spill files
     * @param memory_threshold Max memory before spilling (default 64 MB)
     */
    EdgeAggregationBuffer(const std::string& temp_file_prefix, size_t memory_threshold = DEFAULT_THRESHOLD)
        : temp_file_prefix_(temp_file_prefix)
        , memory_threshold_(memory_threshold)
        , max_records_in_memory_(memory_threshold / sizeof(EdgeAggregationRecord))
        , total_records_(0)
        , spill_count_(0)
        , finalized_(false)
    {
        memory_buffer_.reserve(max_records_in_memory_);
    }

    ~EdgeAggregationBuffer() {
        cleanup();
    }

    // Non-copyable
    EdgeAggregationBuffer(const EdgeAggregationBuffer&) = delete;
    EdgeAggregationBuffer& operator=(const EdgeAggregationBuffer&) = delete;

    /**
     * @brief Adds an edge record to the buffer.
     *
     * Spills to disk automatically when memory threshold is exceeded.
     */
    void push_back(const EdgeAggregationRecord& record) {
        memory_buffer_.push_back(record);
        total_records_++;

        if (memory_buffer_.size() >= max_records_in_memory_) {
            spill_to_disk();
        }
    }

    /**
     * @brief Finalizes the buffer (must call before getting spill info).
     */
    void finalize() {
        if (finalized_) return;
        finalized_ = true;
    }

    /**
     * @brief Gets paths to spill files.
     */
    const std::vector<std::string>& get_spill_paths() const {
        return spill_files_;
    }

    /**
     * @brief Gets record counts per spill file.
     */
    const std::vector<size_t>& get_spill_record_counts() const {
        return spill_record_counts_;
    }

    /**
     * @brief Gets records still in memory (not spilled).
     */
    std::vector<EdgeAggregationRecord>& get_memory_records() {
        return memory_buffer_;
    }

    /**
     * @brief Total records added.
     */
    size_t size() const { return total_records_; }

    /**
     * @brief Whether any data was spilled to disk.
     */
    bool has_spilled() const { return spill_count_ > 0; }

private:
    void spill_to_disk() {
        if (memory_buffer_.empty()) return;

        std::string spill_path = temp_file_prefix_ + "_spill_" + std::to_string(spill_count_);
        std::ofstream out(spill_path, std::ios::binary);

        if (!out) {
            throw std::runtime_error("Failed to create spill file: " + spill_path);
        }

        out.write(
            reinterpret_cast<const char*>(memory_buffer_.data()),
            memory_buffer_.size() * sizeof(EdgeAggregationRecord)
        );
        out.close();

        spill_files_.push_back(spill_path);
        spill_record_counts_.push_back(memory_buffer_.size());
        spill_count_++;

        memory_buffer_.clear();
    }

    void cleanup() {
        for (const auto& path : spill_files_) {
            std::filesystem::remove(path);
        }
        spill_files_.clear();
    }

    std::string temp_file_prefix_;
    size_t memory_threshold_;
    size_t max_records_in_memory_;
    size_t total_records_;
    size_t spill_count_;
    bool finalized_;

    std::vector<EdgeAggregationRecord> memory_buffer_;
    std::vector<std::string> spill_files_;
    std::vector<size_t> spill_record_counts_;
};

} // namespace GQL
