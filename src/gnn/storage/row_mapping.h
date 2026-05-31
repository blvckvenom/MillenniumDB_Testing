#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <algorithm>
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
 * and the internal sorted index (built lazily on first find()) are read-only
 * once initialized.
 *
 * Usage:
 *   auto rm = RowMapping::create("mapping.rmap", object_ids);
 *   auto rm = RowMapping::open("mapping.rmap");
 *   ObjectId oid = rm.get(row_index);
 *   auto idx = rm.find(some_oid);  // O(log N) binary search lookup
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

    /// Binary search lookup for an ObjectId. Returns its row index if found. O(log N).
    /// The index is built lazily on the first call (thread-safe via std::call_once).
    /// For duplicate ObjectIds, returns the first occurrence.
    std::optional<uint64_t> find(ObjectId target) const;

    /// Number of entries.
    uint64_t size() const { return count_; }

    const std::filesystem::path& path() const { return path_; }

private:
    static constexpr uint32_t MAGIC   = 0x524D4150; // "RMAP"
    static constexpr uint32_t VERSION = 1;

    // Fix #17: persistent sorted index. Sidecar file `<path>.idx` stores
    // the (oid, row) pairs already sorted by oid. On open() we mmap it
    // and find() works directly off the mmap'd region — skipping the
    // O(N log N) build_index() that takes ~30 s on papers100M-scale.
    static constexpr uint32_t IDX_MAGIC   = 0x52494458; // "RIDX"
    static constexpr uint32_t IDX_VERSION = 1;

    RowMapping() = default;

    void build_index() const;
    void persist_sorted_index_() const;
    bool try_load_persisted_index_();

    std::filesystem::path path_;
    void*    mmap_ptr_  = nullptr;
    size_t   mmap_size_ = 0;
    uint64_t count_     = 0;

    // Fix #17: separate mmap region for the sidecar index. Non-null iff
    // we loaded the persisted index successfully.
    void*    idx_mmap_ptr_  = nullptr;
    size_t   idx_mmap_size_ = 0;
    const std::pair<uint64_t, uint64_t>* idx_data_ = nullptr;

    /// Lazy init flag for sorted_index_. Built on first find() via std::call_once.
    mutable std::unique_ptr<std::once_flag> build_index_flag_ = std::make_unique<std::once_flag>();

    /// (oid.id, row_idx) pairs sorted by oid.id. Built lazily on first find().
    /// Empty when idx_data_ points to a mmap'd persisted index.
    mutable std::vector<std::pair<uint64_t, uint64_t>> sorted_index_;

    const ObjectId* data_ptr() const;
};

} // namespace mdb::gnn
