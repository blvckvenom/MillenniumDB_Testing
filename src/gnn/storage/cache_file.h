#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "gnn/storage/gnn_dtype.h"

namespace mdb::gnn {

/**
 * @brief On-disk header for cache files (gpu_cache.bin, cpu_cache.bin). Always 32 bytes.
 *
 * File layout:
 *   [Header: 32 bytes]
 *   [ObjectId table: num_nodes * 8 bytes]
 *   [Feature data: num_nodes * feature_dim * dtype_size bytes]
 *
 * The ObjectId table enables reconstructing the ObjectId -> index HashMap at load time.
 * Feature data is contiguous for single-transfer load to GPU via torch::from_blob().
 */
struct CacheFileHeader {
    static constexpr uint32_t MAGIC   = 0x474E4E43; // "GNNC" (MSB-first, matches storage/ convention)
    static constexpr uint32_t VERSION = 1;
    static constexpr size_t   SIZE    = 32;

    uint32_t magic;
    uint32_t version;
    uint64_t num_nodes;
    uint64_t feature_dim;
    uint8_t  dtype;
    uint8_t  reserved[7];

    /// Create a valid header with the given parameters.
    static CacheFileHeader make(uint64_t nodes, uint64_t dim, GnnDtype dt) {
        if (static_cast<uint8_t>(dt) > static_cast<uint8_t>(GnnDtype::MAX_VALUE)) {
            throw std::invalid_argument("CacheFileHeader::make: invalid dtype");
        }
        CacheFileHeader h{};
        std::memset(&h, 0, sizeof(h));
        h.magic       = MAGIC;
        h.version     = VERSION;
        h.num_nodes   = nodes;
        h.feature_dim = dim;
        h.dtype       = static_cast<uint8_t>(dt);
        return h;
    }

    /// Validate header fields. num_nodes == 0 is valid (empty cache).
    bool is_valid() const {
        return magic == MAGIC && version == VERSION
            && feature_dim > 0
            && dtype <= static_cast<uint8_t>(GnnDtype::MAX_VALUE);
    }

    GnnDtype get_dtype() const {
        return static_cast<GnnDtype>(dtype);
    }

    /// Offset to ObjectId table (immediately after header).
    size_t oid_table_offset() const { return SIZE; }

    /// Size of ObjectId table in bytes.
    size_t oid_table_bytes() const { return num_nodes * sizeof(uint64_t); }

    /// Offset to feature data (after header + ObjectId table).
    size_t data_offset() const { return SIZE + oid_table_bytes(); }

    /// Feature data size in bytes. Throws on overflow.
    size_t data_bytes() const {
        size_t el = dtype_size(get_dtype());
        size_t row_bytes = feature_dim * el;
        if (feature_dim > 0 && row_bytes / feature_dim != el) {
            throw std::overflow_error("CacheFileHeader::data_bytes: dimension overflow");
        }
        size_t total = num_nodes * row_bytes;
        if (num_nodes > 0 && total / num_nodes != row_bytes) {
            throw std::overflow_error("CacheFileHeader::data_bytes: total size overflow");
        }
        return total;
    }

    /// Total file size (header + OID table + feature data).
    size_t total_file_size() const {
        return data_offset() + data_bytes();
    }
};

static_assert(sizeof(CacheFileHeader) == 32, "CacheFileHeader must be exactly 32 bytes");
static_assert(std::is_standard_layout_v<CacheFileHeader>,
              "CacheFileHeader must be standard layout for direct I/O");
static_assert(std::is_trivially_copyable_v<CacheFileHeader>,
              "CacheFileHeader must be trivially copyable for direct I/O");

} // namespace mdb::gnn
