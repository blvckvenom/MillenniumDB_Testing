#pragma once

/**
 * @file file_gnn_tensor_store.h
 * @brief File-backed implementation of GnnTensorStore with memory-mapped access.
 *
 * This implementation persists tensors to disk using a binary format:
 *
 * Directory Structure:
 *   <base_path>/
 *     index.bin     - Index file with tensor metadata
 *     data_0.bin    - Data shard 0
 *     data_1.bin    - Data shard 1
 *     ...
 *
 * Index File Format (index.bin):
 *   [Header]
 *     magic: uint32_t (0x474E4E54 = "GNNT")
 *     version: uint32_t
 *     num_tensors: uint64_t
 *     num_shards: uint32_t
 *   [Entry] x num_tensors
 *     key_length: uint32_t
 *     key: char[key_length]
 *     dtype: uint8_t
 *     ndim: uint32_t
 *     shape: int64_t[ndim]
 *     shard_id: uint32_t
 *     offset: uint64_t
 *     byte_size: uint64_t
 *
 * Data Shards (data_N.bin):
 *   Raw tensor data, concatenated, byte-aligned
 *
 * Design Decisions:
 *   - Memory-mapped files allow accessing large tensors without loading into RAM
 *   - Sharding allows parallel writes and avoids single large file limitations
 *   - Index is always loaded into memory (small compared to tensor data)
 */

#include "gnn_tensor_store.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mdb::gnn {

/**
 * @brief File-backed GnnTensorStore using memory-mapped files.
 *
 * This implementation is designed for production use with large graph datasets
 * that don't fit entirely in memory.
 */
class FileGnnTensorStore : public GnnTensorStore {
public:
    /// Magic number for index file validation ("GNNT" in ASCII)
    static constexpr uint32_t MAGIC = 0x474E4E54;

    /// Current file format version
    static constexpr uint32_t VERSION = 1;

    /// Default maximum shard size (1 GB)
    static constexpr size_t DEFAULT_MAX_SHARD_SIZE = 1ULL << 30;

    /// Byte alignment for tensor data
    static constexpr size_t ALIGNMENT = 64;

    /**
     * @brief Construct a FileGnnTensorStore.
     *
     * @param base_path Directory for storing tensor files
     * @param max_shard_size Maximum size per data shard (default 1GB)
     * @param create_if_missing Create directory if it doesn't exist
     *
     * If base_path contains existing tensor data, it will be loaded.
     */
    explicit FileGnnTensorStore(const std::filesystem::path& base_path,
                                 size_t max_shard_size = DEFAULT_MAX_SHARD_SIZE,
                                 bool create_if_missing = true);

    ~FileGnnTensorStore() override;

    // Non-copyable, non-movable (owns file handles)
    FileGnnTensorStore(const FileGnnTensorStore&) = delete;
    FileGnnTensorStore& operator=(const FileGnnTensorStore&) = delete;
    FileGnnTensorStore(FileGnnTensorStore&&) = delete;
    FileGnnTensorStore& operator=(FileGnnTensorStore&&) = delete;

    // ========================================================================
    // Core Operations
    // ========================================================================

    bool store(const std::string& key,
               const void* data,
               const std::vector<int64_t>& shape,
               GnnDtype dtype) override;

    GnnTensorView load(const std::string& key) const override;

    bool remove(const std::string& key) override;

    bool exists(const std::string& key) const override;

    // ========================================================================
    // Metadata Operations
    // ========================================================================

    std::optional<GnnTensorMetadata> get_metadata(const std::string& key) const override;

    std::vector<std::string> list_keys() const override;

    size_t total_size() const override;

    size_t count() const override;

    // ========================================================================
    // Lifecycle Operations
    // ========================================================================

    void clear() override;

    void flush() override;

    // ========================================================================
    // File-Specific Operations
    // ========================================================================

    /**
     * @brief Get the base directory path.
     */
    const std::filesystem::path& base_path() const { return base_path_; }

    /**
     * @brief Compact the store by removing fragmentation.
     *
     * This rewrites all data files to remove gaps from deleted tensors.
     * The store remains usable during compaction via copy-on-write.
     *
     * @return Bytes reclaimed
     */
    size_t compact();

    /**
     * @brief Check if the store needs compaction.
     *
     * @param threshold Fragmentation ratio above which compaction is needed (0.0-1.0)
     * @return true if fragmentation exceeds threshold
     */
    bool needs_compaction(double threshold = 0.25) const;

private:
    /**
     * @brief Internal entry in the index.
     */
    struct IndexEntry {
        GnnTensorMetadata metadata;
        uint32_t shard_id;
        uint64_t offset;
        bool deleted = false;  // Soft delete for lazy compaction
    };

    /**
     * @brief Memory-mapped shard information.
     */
    struct MappedShard {
        void* data = nullptr;
        size_t size = 0;
        int fd = -1;

        ~MappedShard();
        MappedShard() = default;
        MappedShard(MappedShard&& other) noexcept;
        MappedShard& operator=(MappedShard&& other) noexcept;
        MappedShard(const MappedShard&) = delete;
        MappedShard& operator=(const MappedShard&) = delete;
    };

    // File paths
    std::filesystem::path base_path_;
    std::filesystem::path index_path() const { return base_path_ / "index.bin"; }
    std::filesystem::path shard_path(uint32_t shard_id) const;

    // Configuration
    size_t max_shard_size_;

    // In-memory index
    mutable std::shared_mutex mutex_;
    mutable std::mutex shard_mutex_;
    std::unordered_map<std::string, IndexEntry> index_;
    size_t total_bytes_ = 0;
    size_t deleted_bytes_ = 0;

    // Generation counter: incremented on every compact/remap
    std::atomic<uint64_t> generation_{0};

    // Memory-mapped shards
    mutable std::vector<MappedShard> shards_;
    std::vector<size_t> shard_sizes_;  // Current size of each shard

    // File I/O helpers
    void load_index();
    void save_index();
    void ensure_shard(uint32_t shard_id);
    void map_shard(uint32_t shard_id) const;
    void map_shard_impl(uint32_t shard_id) const;  // Caller must hold shard_mutex_
    void unmap_all_shards();

    // Alignment helper
    static size_t align_up(size_t size) {
        return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }
};

} // namespace mdb::gnn
