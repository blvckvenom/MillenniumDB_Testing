#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "gnn/storage/gnn_dtype.h"

namespace mdb::gnn {

/**
 * @brief On-disk header for .fmat files. The header struct is always 64 bytes,
 *        at offset 0. The DATA section begins at get_data_offset() bytes.
 *
 * v1 (default) layout: [Header: 64 bytes][Data: num_rows*num_cols*dtype_size]
 *   Data immediately follows the 64-byte header (data offset == 64).
 *
 * v2 (page-aligned, opt-in via env MDB_FMAT_PAGE_ALIGN) layout:
 *   [Header: 64 bytes][zero padding][Data ... at offset 4096]
 *   The data section starts on a 4096-byte filesystem block boundary so
 *   O_DIRECT / cuFile / GPUDirect reads of the data section are block-aligned
 *   at the start. The explicit data offset is recorded in the header so every
 *   reader honors it instead of assuming the 64-byte legacy offset.
 *
 * Data is row-major, contiguous, no padding between rows (both versions).
 *
 * reserved[] byte layout (within the 39-byte tail):
 *   reserved[0..7]  : content fingerprint (set by create_reordered; 0 = absent)
 *   reserved[8..15] : data_offset (0 = legacy/implicit 64; else explicit start)
 *   reserved[16..38]: unused (zero)
 */
struct FeatureMatrixHeader {
    static constexpr uint32_t MAGIC   = 0x474E4E46; // "GNNF"
    static constexpr uint32_t VERSION = 1;          // default/legacy write version (data at offset 64)
    static constexpr uint32_t VERSION_PAGE_ALIGNED = 2;  // v2: explicit page-aligned data offset
    static constexpr uint32_t MAX_SUPPORTED_VERSION = 2; // highest version open() accepts
    static constexpr size_t   SIZE    = 64;         // on-disk header struct size (UNCHANGED across versions)
    static constexpr uint64_t DATA_OFFSET_V1 = 64;     // legacy: data immediately after the 64-byte header
    static constexpr uint64_t DATA_OFFSET_V2 = 4096;   // v2: page-aligned data section start

    uint32_t magic;
    uint32_t version;
    uint64_t num_rows;
    uint64_t num_cols;
    uint8_t  dtype;
    uint8_t  reserved[39]; // pad to 64 bytes; see reserved[] layout in the struct doc

    // Page-aligned data section is opt-in via env MDB_FMAT_PAGE_ALIGN.
    // Default OFF -> legacy v1 layout (data offset 64), byte-identical output.
    // NOT routed through the ablation registry: the registry resolves a name
    // once per process, and FeatureMatrixHeaderTest.PageAlignV2HeaderToggle
    // flips this variable from "0" to "1" inside a single test and asserts both
    // data offsets. Re-reading the environment on every call is load-bearing
    // here, so the switch keeps its own resolution until that test is reworked.
    static bool page_align_enabled() {
        const char* e = std::getenv("MDB_FMAT_PAGE_ALIGN");
        if (e == nullptr) return false;
        const std::string v(e);
        return v == "1" || v == "true" || v == "yes" || v == "on";
    }

    static FeatureMatrixHeader make(uint64_t rows, uint64_t cols, GnnDtype dt) {
        if (static_cast<uint8_t>(dt) > static_cast<uint8_t>(GnnDtype::MAX_VALUE)) {
            throw std::invalid_argument("FeatureMatrixHeader::make: invalid dtype");
        }
        FeatureMatrixHeader h{};
        std::memset(&h, 0, sizeof(h));
        h.magic    = MAGIC;
        h.num_rows = rows;
        h.num_cols = cols;
        h.dtype    = static_cast<uint8_t>(dt);
        // When page-alignment is OFF the header stays byte-identical to the
        // legacy v1 layout: version 1, reserved all-zero, so get_data_offset()
        // returns the implicit 64. When ON, record version 2 and an explicit
        // 4096-byte data offset (readers honor the recorded offset either way).
        if (page_align_enabled()) {
            h.version = VERSION_PAGE_ALIGNED;
            h.set_data_offset(DATA_OFFSET_V2);
        } else {
            h.version = VERSION;
        }
        return h;
    }

    bool is_valid() const {
        return magic == MAGIC
            && version >= VERSION && version <= MAX_SUPPORTED_VERSION
            && num_rows > 0 && num_cols > 0
            && dtype <= static_cast<uint8_t>(GnnDtype::MAX_VALUE)
            && get_data_offset() >= SIZE;  // data section cannot start inside the header
    }

    // Byte offset where the data section begins. v1 files (and any header
    // that did not record an explicit offset) store 0 in reserved[8..15] and the
    // data follows immediately after the 64-byte header. v2 files store the
    // page-aligned offset (4096) so the data section starts on a block boundary.
    uint64_t get_data_offset() const {
        uint64_t off = 0;
        std::memcpy(&off, reserved + 8, sizeof(off));  // reserved[8..15]
        return off == 0 ? DATA_OFFSET_V1 : off;
    }
    void set_data_offset(uint64_t off) {
        std::memcpy(reserved + 8, &off, sizeof(off));  // reserved[8..15]
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
static_assert(std::is_standard_layout_v<FeatureMatrixHeader>,
              "FeatureMatrixHeader must be standard layout for direct I/O");
static_assert(std::is_trivially_copyable_v<FeatureMatrixHeader>,
              "FeatureMatrixHeader must be trivially copyable for direct I/O");

} // namespace mdb::gnn
