#pragma once

/**
 * @file streaming_record_buffer.h
 * @brief Memory-bounded streaming buffer for projection records.
 *
 * Provides a buffer that accumulates records in memory up to a configurable
 * threshold, then spills to disk. Enables building projections of arbitrary
 * size with bounded memory usage.
 *
 * @see DiskVector in src/import/disk_vector.h for inspiration
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "storage/index/record.h"
#include "storage/page/page.h"

namespace GQL {

/**
 * @brief Memory-bounded buffer that spills to disk when full.
 *
 * Provides streaming record collection with configurable memory budget.
 * Records are accumulated in memory until the threshold is reached,
 * then written to a temporary file. Multiple spill files may be created
 * for very large projections.
 *
 * ## Memory Model
 * - Memory buffer: Configurable (default 64 MB per index)
 * - Each record: N * 8 bytes
 * - Peak memory: buffer_size + small overhead
 *
 * ## Usage Pattern
 * ```cpp
 * StreamingRecordBuffer<3> buffer("/tmp/proj_edges", 64 * 1024 * 1024);
 * buffer.push_back({from, to, edge_id});  // Spills to disk automatically
 * buffer.finalize();  // Flush remaining
 * buffer.begin_iteration();
 * while (buffer.has_next()) {
 *     auto& record = buffer.next();
 *     // Process record
 * }
 * ```
 *
 * @tparam N Number of uint64_t values per record (1-5 supported)
 */
template<std::size_t N>
class StreamingRecordBuffer {
public:
    /// @brief Default memory threshold before spilling to disk (64 MB)
    static constexpr size_t DEFAULT_MEMORY_THRESHOLD = 64 * 1024 * 1024;

    /// @brief Records per buffer page (for efficient I/O)
    static constexpr size_t RECORDS_PER_PAGE = Page::SIZE / (N * sizeof(uint64_t));

    /**
     * @brief Constructs a streaming buffer with specified memory threshold.
     *
     * @param temp_file_prefix Prefix for temp file paths (e.g., "/tmp/proj_edges")
     * @param memory_threshold Max memory before spilling to disk (bytes)
     */
    StreamingRecordBuffer(const std::string& temp_file_prefix, size_t memory_threshold = DEFAULT_MEMORY_THRESHOLD)
        : temp_file_prefix_(temp_file_prefix)
        , memory_threshold_(memory_threshold)
        , max_records_in_memory_(memory_threshold / (N * sizeof(uint64_t)))
        , total_records_(0)
        , spill_file_count_(0)
        , finalized_(false)
        , iterating_(false)
        , current_spill_file_(0)
        , iter_memory_pos_(0)
        , iter_file_pos_(0)
    {
        // Reserve memory buffer capacity
        memory_buffer_.reserve(max_records_in_memory_);
    }

    ~StreamingRecordBuffer() {
        cleanup_temp_files();
    }

    // Non-copyable, movable
    StreamingRecordBuffer(const StreamingRecordBuffer&) = delete;
    StreamingRecordBuffer& operator=(const StreamingRecordBuffer&) = delete;
    StreamingRecordBuffer(StreamingRecordBuffer&&) = default;
    StreamingRecordBuffer& operator=(StreamingRecordBuffer&&) = default;

    /**
     * @brief Adds a record to the buffer.
     *
     * If memory threshold is exceeded, spills current buffer to disk.
     *
     * @param record The record to add
     */
    void push_back(const Record<N>& record) {
        memory_buffer_.push_back(record);
        total_records_++;

        if (memory_buffer_.size() >= max_records_in_memory_) {
            spill_to_disk();
        }
    }

    /**
     * @brief Adds a record to the buffer (move version).
     */
    void push_back(Record<N>&& record) {
        memory_buffer_.push_back(std::move(record));
        total_records_++;

        if (memory_buffer_.size() >= max_records_in_memory_) {
            spill_to_disk();
        }
    }

    /**
     * @brief Finalizes the buffer (must be called before iteration).
     *
     * Flushes any remaining records and prepares for iteration.
     */
    void finalize() {
        if (finalized_) return;

        // If we have spill files, flush remaining memory to disk too
        if (spill_file_count_ > 0 && !memory_buffer_.empty()) {
            spill_to_disk();
        }

        finalized_ = true;
    }

    /**
     * @brief Begins iteration over all records.
     *
     * After calling this, use has_next() and next() to iterate.
     */
    void begin_iteration() {
        if (!finalized_) {
            finalize();
        }

        iterating_ = true;
        current_spill_file_ = 0;
        iter_memory_pos_ = 0;
        iter_file_pos_ = 0;

        // Open first spill file if exists
        if (spill_file_count_ > 0) {
            open_spill_file_for_reading(0);
        }
    }

    /**
     * @brief Checks if more records are available.
     */
    bool has_next() const {
        if (spill_file_count_ == 0) {
            // All data in memory
            return iter_memory_pos_ < memory_buffer_.size();
        } else {
            // Data in spill files
            return current_spill_file_ < spill_file_count_ ||
                   iter_file_pos_ < file_read_buffer_.size();
        }
    }

    /**
     * @brief Returns the next record.
     *
     * @return Reference to the next record
     * @note Caller must check has_next() before calling
     */
    Record<N>& next() {
        if (spill_file_count_ == 0) {
            // All data in memory
            return memory_buffer_[iter_memory_pos_++];
        } else {
            // Data in spill files - need to read from disk
            if (iter_file_pos_ >= file_read_buffer_.size()) {
                read_next_chunk();
            }
            return file_read_buffer_[iter_file_pos_++];
        }
    }

    /**
     * @brief Returns total number of records.
     */
    size_t size() const { return total_records_; }

    /**
     * @brief Checks if any data was spilled to disk.
     */
    bool has_spilled() const { return spill_file_count_ > 0; }

    /**
     * @brief Clears all data and resets the buffer.
     */
    void clear() {
        cleanup_temp_files();
        memory_buffer_.clear();
        memory_buffer_.shrink_to_fit();
        memory_buffer_.reserve(max_records_in_memory_);
        total_records_ = 0;
        spill_file_count_ = 0;
        finalized_ = false;
        iterating_ = false;
    }

    /**
     * @brief Gets all records as a vector (for small datasets).
     *
     * @warning Only use when total_records < memory threshold.
     *          For large datasets, use iteration interface.
     * @return Vector of all records
     */
    std::vector<Record<N>> get_all_records() {
        if (spill_file_count_ == 0) {
            return memory_buffer_;
        }

        // Collect from all spill files
        std::vector<Record<N>> result;
        result.reserve(total_records_);

        begin_iteration();
        while (has_next()) {
            result.push_back(next());
        }

        return result;
    }

    /**
     * @brief Gets a mutable reference to the internal vector.
     *
     * @warning Only valid when no spill has occurred.
     *          Used for in-place sorting when data fits in memory.
     * @return Reference to internal memory buffer
     */
    std::vector<Record<N>>& get_memory_buffer() {
        return memory_buffer_;
    }

    /**
     * @brief Gets the list of spill file paths.
     *
     * Used by ExternalRecordSort to access spill files directly without
     * loading them all into memory first.
     *
     * @return Const reference to vector of spill file paths
     */
    const std::vector<std::string>& get_spill_paths() const {
        return spill_files_;
    }

    /**
     * @brief Gets the record counts for each spill file.
     *
     * The i-th element is the number of records in the i-th spill file.
     *
     * @return Const reference to vector of record counts per spill file
     */
    const std::vector<size_t>& get_spill_counts() const {
        return records_per_spill_;
    }

    /**
     * @brief Takes ownership of the memory buffer (move semantics).
     *
     * Efficiently transfers the internal memory buffer to the caller without
     * copying. The internal buffer is left empty after this call.
     *
     * Used by ExternalRecordSort to avoid copying when data fits in memory.
     *
     * @return The memory buffer contents (moved)
     */
    std::vector<Record<N>> take_memory_buffer() {
        auto result = std::move(memory_buffer_);
        memory_buffer_.clear();
        memory_buffer_.reserve(max_records_in_memory_);
        return result;
    }

    /**
     * @brief Gets the number of records currently in the memory buffer.
     *
     * This count does not include records that have been spilled to disk.
     * To get the total count including spilled records, use size().
     *
     * @return Number of records in memory (not spilled)
     */
    size_t memory_buffer_size() const {
        return memory_buffer_.size();
    }

private:
    /**
     * @brief Spills current memory buffer to a temporary file.
     */
    void spill_to_disk() {
        if (memory_buffer_.empty()) return;

        std::string spill_path = temp_file_prefix_ + "_spill_" + std::to_string(spill_file_count_);
        std::ofstream out(spill_path, std::ios::binary);

        if (!out) {
            throw std::runtime_error("Failed to create spill file: " + spill_path);
        }

        // Write records as raw bytes
        out.write(reinterpret_cast<const char*>(memory_buffer_.data()),
                  memory_buffer_.size() * N * sizeof(uint64_t));
        out.close();

        spill_files_.push_back(spill_path);
        records_per_spill_.push_back(memory_buffer_.size());
        spill_file_count_++;

        memory_buffer_.clear();
    }

    /**
     * @brief Opens a spill file for reading.
     */
    void open_spill_file_for_reading(size_t index) {
        if (current_input_stream_.is_open()) {
            current_input_stream_.close();
        }

        current_input_stream_.open(spill_files_[index], std::ios::binary);
        if (!current_input_stream_) {
            throw std::runtime_error("Failed to open spill file for reading: " + spill_files_[index]);
        }

        current_spill_records_remaining_ = records_per_spill_[index];
        file_read_buffer_.clear();
        iter_file_pos_ = 0;
    }

    /**
     * @brief Reads the next chunk of records from current spill file.
     */
    void read_next_chunk() {
        file_read_buffer_.clear();
        iter_file_pos_ = 0;

        // Try to read from current file
        while (current_spill_records_remaining_ == 0) {
            // Move to next spill file
            current_spill_file_++;
            if (current_spill_file_ >= spill_file_count_) {
                return;  // No more data
            }
            open_spill_file_for_reading(current_spill_file_);
        }

        // Read up to RECORDS_PER_PAGE records
        size_t to_read = std::min(current_spill_records_remaining_, RECORDS_PER_PAGE);
        file_read_buffer_.resize(to_read);

        current_input_stream_.read(
            reinterpret_cast<char*>(file_read_buffer_.data()),
            to_read * N * sizeof(uint64_t)
        );

        size_t actually_read = current_input_stream_.gcount() / (N * sizeof(uint64_t));
        file_read_buffer_.resize(actually_read);
        current_spill_records_remaining_ -= actually_read;
    }

    /**
     * @brief Removes all temporary files.
     */
    void cleanup_temp_files() {
        if (current_input_stream_.is_open()) {
            current_input_stream_.close();
        }

        for (const auto& path : spill_files_) {
            std::filesystem::remove(path);
        }
        spill_files_.clear();
        records_per_spill_.clear();
    }

    // Configuration
    std::string temp_file_prefix_;
    size_t memory_threshold_;
    size_t max_records_in_memory_;

    // State
    std::vector<Record<N>> memory_buffer_;
    uint64_t total_records_;
    size_t spill_file_count_;
    bool finalized_;

    // Spill file tracking
    std::vector<std::string> spill_files_;
    std::vector<size_t> records_per_spill_;

    // Iteration state
    bool iterating_;
    size_t current_spill_file_;
    size_t iter_memory_pos_;
    size_t iter_file_pos_;
    std::ifstream current_input_stream_;
    std::vector<Record<N>> file_read_buffer_;
    size_t current_spill_records_remaining_ = 0;
};

} // namespace GQL
