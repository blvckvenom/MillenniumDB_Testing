#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/gnn_dtype.h"

namespace mdb::gnn {

/**
 * @brief On-disk header for packed batch files (.bin). Always 32 bytes, at offset 0.
 *
 * File layout: [Header: 32 bytes][Data: num_nodes * feature_dim * dtype_size bytes]
 * Data is row-major, contiguous, no padding between rows.
 */
struct PackedBatchHeader {
    static constexpr uint32_t MAGIC   = 0x474E4E42; // "GNNB"
    static constexpr uint32_t VERSION = 1;
    static constexpr size_t   SIZE    = 32;

    uint32_t magic;
    uint32_t version;
    uint64_t num_nodes;
    uint64_t feature_dim;
    uint8_t  dtype;
    uint8_t  reserved[7];

    static PackedBatchHeader make(uint64_t nodes, uint64_t dim, GnnDtype dt) {
        if (static_cast<uint8_t>(dt) > static_cast<uint8_t>(GnnDtype::MAX_VALUE)) {
            throw std::invalid_argument("PackedBatchHeader::make: invalid dtype");
        }
        PackedBatchHeader h{};
        std::memset(&h, 0, sizeof(h));
        h.magic       = MAGIC;
        h.version     = VERSION;
        h.num_nodes   = nodes;
        h.feature_dim = dim;
        h.dtype       = static_cast<uint8_t>(dt);
        return h;
    }

    /// Unlike FeatureMatrixHeader, num_nodes == 0 IS valid (empty batch).
    bool is_valid() const {
        return magic == MAGIC && version == VERSION
            && feature_dim > 0
            && dtype <= static_cast<uint8_t>(GnnDtype::MAX_VALUE);
    }

    GnnDtype get_dtype() const {
        return static_cast<GnnDtype>(dtype);
    }

    /// Returns data size in bytes. Throws std::overflow_error on overflow.
    size_t data_bytes() const {
        size_t dim_bytes = feature_dim * dtype_size(get_dtype());
        if (feature_dim > 0 && dim_bytes / feature_dim != dtype_size(get_dtype())) {
            throw std::overflow_error("PackedBatchHeader::data_bytes: dimension overflow");
        }
        size_t total = num_nodes * dim_bytes;
        if (num_nodes > 0 && total / num_nodes != dim_bytes) {
            throw std::overflow_error("PackedBatchHeader::data_bytes: total size overflow");
        }
        return total;
    }
};

static_assert(sizeof(PackedBatchHeader) == 32, "PackedBatchHeader must be exactly 32 bytes");
static_assert(std::is_standard_layout_v<PackedBatchHeader>,
              "PackedBatchHeader must be standard layout for direct I/O");
static_assert(std::is_trivially_copyable_v<PackedBatchHeader>,
              "PackedBatchHeader must be trivially copyable for direct I/O");

} // namespace mdb::gnn
