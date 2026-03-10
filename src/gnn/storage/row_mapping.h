#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include "graph_models/object_id.h"

namespace fs = std::filesystem;

namespace mdb::gnn {

/**
 * @brief Immutable mmap-backed array mapping row indices to ObjectIds.
 *
 * File layout: [Header: 16 bytes][ObjectId[0]: 8 bytes][ObjectId[1]: 8 bytes]...
 * Header: MAGIC(4) + VERSION(4) + count(8)
 *
 * Thread-safe for concurrent reads (read-only mmap after construction).
 *
 * Usage:
 *   auto rm = RowMapping::create("mapping.rmap", object_ids);
 *   auto rm = RowMapping::open("mapping.rmap");
 *   ObjectId oid = rm.get(row_index);
 *   auto idx = rm.find(some_oid);  // linear search
 */
class RowMapping {
public:
    static constexpr uint32_t MAGIC   = 0x524D4150; // "RMAP"
    static constexpr uint32_t VERSION = 1;
    static constexpr size_t   HEADER_SIZE = 16; // magic(4) + version(4) + count(8)

    static RowMapping create(const fs::path& path, const std::vector<ObjectId>& ids);
    static RowMapping open(const fs::path& path);

    // Move only (owns mmap)
    RowMapping(RowMapping&& other) noexcept;
    RowMapping& operator=(RowMapping&& other) noexcept;
    RowMapping(const RowMapping&) = delete;
    RowMapping& operator=(const RowMapping&) = delete;
    ~RowMapping();

    /// Get the ObjectId at a given row index. O(1).
    ObjectId get(uint64_t row_index) const;

    /// Linear search for an ObjectId. Returns its row index if found. O(N).
    std::optional<uint64_t> find(ObjectId target) const;

    /// Number of entries.
    uint64_t size() const { return count_; }

    const fs::path& path() const { return path_; }

private:
    RowMapping() = default;

    fs::path path_;
    void*    mmap_ptr_  = nullptr;
    size_t   mmap_size_ = 0;
    uint64_t count_     = 0;

    const ObjectId* data_ptr() const;
};

} // namespace mdb::gnn
