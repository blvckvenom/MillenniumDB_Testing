#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "gnn/storage/gnn_dtype.h"

namespace mdb::gnn {

/**
 * @brief On-disk header for .fmat files. Always 64 bytes, at offset 0.
 *
 * File layout: [Header: 64 bytes][Data: num_rows * num_cols * dtype_size bytes]
 * Data is row-major, contiguous, no padding between rows.
 */
struct FeatureMatrixHeader {
    static constexpr uint32_t MAGIC   = 0x474E4E46; // "GNNF"
    static constexpr uint32_t VERSION = 1;
    static constexpr size_t   SIZE    = 64;

    uint32_t magic;
    uint32_t version;
    uint64_t num_rows;
    uint64_t num_cols;
    uint8_t  dtype;
    uint8_t  reserved[39]; // pad to 64 bytes

    static FeatureMatrixHeader make(uint64_t rows, uint64_t cols, GnnDtype dt) {
        FeatureMatrixHeader h{};
        std::memset(&h, 0, sizeof(h));
        h.magic    = MAGIC;
        h.version  = VERSION;
        h.num_rows = rows;
        h.num_cols = cols;
        h.dtype    = static_cast<uint8_t>(dt);
        return h;
    }

    bool is_valid() const {
        return magic == MAGIC && version == VERSION
            && num_rows > 0 && num_cols > 0
            && dtype <= static_cast<uint8_t>(GnnDtype::MAX_VALUE);
    }

    GnnDtype get_dtype() const {
        return static_cast<GnnDtype>(dtype);
    }

    size_t row_bytes() const {
        return num_cols * dtype_size(get_dtype());
    }

    size_t data_bytes() const {
        return num_rows * row_bytes();
    }
};

static_assert(sizeof(FeatureMatrixHeader) == 64, "Header must be exactly 64 bytes");

} // namespace mdb::gnn
