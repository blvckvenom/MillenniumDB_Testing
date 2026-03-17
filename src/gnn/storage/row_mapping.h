#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "graph_models/object_id.h"

namespace mdb::gnn {

/**
 * @brief Immutable mmap-backed array mapping row indices to ObjectIds.
 *
 * File layout: [Header: 16 bytes][ObjectId[0]: 8 bytes][ObjectId[1]: 8 bytes]...
 * Header: MAGIC(4) + VERSION(4) + count(8)
 *
 * Thread-safe for concurrent reads after construction. Both the mmap region
 * and the internal hash map (built lazily on first find()) are read-only
 * once initialized.
 *
 * Usage:
 *   auto rm = RowMapping::create("mapping.rmap", object_ids);
 *   auto rm = RowMapping::open("mapping.rmap");
 *   ObjectId oid = rm.get(row_index);
 *   auto idx = rm.find(some_oid);  // O(1) hash lookup
 */
class RowMapping {
public:
    static constexpr size_t   HEADER_SIZE = 16; // magic(4) + version(4) + count(8)

    static RowMapping create(const std::filesystem::path& path, const std::vector<ObjectId>& ids);
    static RowMapping open(const std::filesystem::path& path);

    // Move only (owns mmap)
    RowMapping(RowMapping&& other) noexcept;
    RowMapping& operator=(RowMapping&& other) noexcept;
    RowMapping(const RowMapping&) = delete;
    RowMapping& operator=(const RowMapping&) = delete;
    ~RowMapping();

    /// Get the ObjectId at a given row index. O(1).
    ObjectId get(uint64_t row_index) const;

    /// Hash map lookup for an ObjectId. Returns its row index if found. O(1).
    /// The index is built lazily on the first call (thread-safe via std::call_once).
    /// For duplicate ObjectIds, returns the first occurrence.
    std::optional<uint64_t> find(ObjectId target) const;

    /// Number of entries.
    uint64_t size() const { return count_; }

    const std::filesystem::path& path() const { return path_; }

private:
    static constexpr uint32_t MAGIC   = 0x524D4150; // "RMAP"
    static constexpr uint32_t VERSION = 1;

    RowMapping() = default;

    void build_index() const;

    std::filesystem::path path_;
    void*    mmap_ptr_  = nullptr;
    size_t   mmap_size_ = 0;
    uint64_t count_     = 0;

    /// Lazy init flag for id_to_row_. Built on first find() via std::call_once.
    mutable std::unique_ptr<std::once_flag> build_index_flag_ = std::make_unique<std::once_flag>();

    /// ObjectId.id → first row index. Built lazily on first find().
    mutable std::unordered_map<uint64_t, uint64_t> id_to_row_;

    const ObjectId* data_ptr() const;
};

} // namespace mdb::gnn
