// src/gnn/storage/addr_table.h
#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace mdb::gnn {

/**
 * @brief On-disk header for AddrTable sidecar files (v1 = 40 B, v2 = 56 B, LE).
 *
 * v1 file layout: [AddrTableHeader: 40 bytes][9 arrays back-to-back, no padding]
 *   l1_positions  [num_l1] uint32  — positions in output tensor for L1 nodes
 *   l1_indices    [num_l1] uint32  — gpu_cache row indices
 *   l2_positions  [num_l2] uint32  — positions in output tensor for L2 nodes
 *   l2_indices    [num_l2] uint32  — cpu_cache row indices
 *   l3_positions  [num_l3] uint32  — positions in output tensor for L3 nodes
 *   l3_row_idxs   [num_l3] uint64  — reordered FeatureMatrix row indices
 *   l4_positions  [num_l4] uint32  — positions in output tensor for L4 nodes
 *   l4_indices    [num_l4] uint32  — packed_slim file slot indices
 *   zero_positions[num_zero] uint32 — positions left as zeros (unresolved)
 *
 * v2 (DiskGNN-adoption Plan 1) appends `slim_offset`/`slim_length` (16 B, header
 * grows 40 -> 56) carrying this batch's payload location in the consolidated slim
 * file (see consolidated_slim.h). The 9 arrays then start at offset 56. v1 and v2
 * are distinguished by the `version` field; both round-trip (a v1-equivalent v2
 * header has slim_offset == slim_length == 0 but the writer still emits a 40-byte
 * v1 header for backwards-compat byte-identity). The struct is always 56 bytes in
 * memory; `header_bytes()` returns the on-disk size for the header's version, and
 * the writer/reader use it so v1 files stay byte-identical to pre-Plan-1 output.
 */
struct AddrTableHeader {
    static constexpr uint32_t MAGIC       = 0x41444452u;  // "ADDR" (MSB-first, matches storage/ convention)
    static constexpr uint32_t VERSION     = 1u;   // version make() writes (legacy default)
    static constexpr uint32_t VERSION_V2  = 2u;   // version make_v2() writes (consolidated-slim aware)
    static constexpr uint32_t MAX_VERSION = 2u;
    static constexpr size_t   SIZE_V1     = 40u;  // on-disk header bytes for v1
    static constexpr size_t   SIZE        = 56u;  // on-disk header bytes for v2 == in-memory struct size

    uint32_t magic;
    uint32_t version;
    uint32_t num_l1;
    uint32_t num_l2;
    uint32_t num_l3;
    uint32_t num_l4;
    uint32_t num_zero;
    uint32_t total;
    uint64_t meta_sha256_head;
    // --- v2 additions (zero for v1) ---
    uint64_t slim_offset;   // byte offset of this batch's payload in consolidated.slim
    uint64_t slim_length;   // payload bytes (full cold payload = batch_size * row_bytes); 0 = no consolidated file

    /// Build a valid v1 header. Precondition: l1 + l2 + l3 + l4 + zero <= UINT32_MAX
    /// (no runtime check — per-batch tier counts cannot realistically approach
    /// 2^32). is_valid() recomputes the same sum, so a wrapped total will not be
    /// detected by validation; callers must ensure this holds. slim_* are 0.
    static AddrTableHeader make(
        uint32_t l1, uint32_t l2, uint32_t l3, uint32_t l4, uint32_t zero,
        uint64_t meta_sha_head)
    {
        AddrTableHeader h{};
        h.magic            = MAGIC;
        h.version          = VERSION;
        h.num_l1           = l1;
        h.num_l2           = l2;
        h.num_l3           = l3;
        h.num_l4           = l4;
        h.num_zero         = zero;
        h.total            = l1 + l2 + l3 + l4 + zero;
        h.meta_sha256_head = meta_sha_head;
        return h;  // slim_offset == slim_length == 0
    }

    /// Build a valid v2 header carrying this batch's consolidated-slim location.
    static AddrTableHeader make_v2(
        uint32_t l1, uint32_t l2, uint32_t l3, uint32_t l4, uint32_t zero,
        uint64_t meta_sha_head, uint64_t slim_off, uint64_t slim_len)
    {
        AddrTableHeader h = make(l1, l2, l3, l4, zero, meta_sha_head);
        h.version     = VERSION_V2;
        h.slim_offset = slim_off;
        h.slim_length = slim_len;
        return h;
    }

    /// On-disk header size for THIS header's version (40 for v1, 56 for v2).
    size_t header_bytes() const {
        return version >= VERSION_V2 ? SIZE : SIZE_V1;
    }

    bool is_valid() const {
        return magic == MAGIC
            && version >= VERSION && version <= MAX_VERSION
            && total == (num_l1 + num_l2 + num_l3 + num_l4 + num_zero);
    }

    size_t expected_file_size() const {
        return header_bytes()
            + size_t(num_l1) * sizeof(uint32_t) * 2
            + size_t(num_l2) * sizeof(uint32_t) * 2
            + size_t(num_l3) * sizeof(uint32_t)
            + size_t(num_l3) * sizeof(uint64_t)
            + size_t(num_l4) * sizeof(uint32_t) * 2
            + size_t(num_zero) * sizeof(uint32_t);
    }
};

static_assert(sizeof(AddrTableHeader) == 56,
              "AddrTableHeader (v2 in-memory) must be exactly 56 bytes");
static_assert(AddrTableHeader::SIZE_V1 == 40,
              "AddrTableHeader v1 on-disk size must be 40 bytes (backwards-compat)");
static_assert(std::is_standard_layout_v<AddrTableHeader>,
              "AddrTableHeader must be standard layout for direct I/O");
static_assert(std::is_trivially_copyable_v<AddrTableHeader>,
              "AddrTableHeader must be trivially copyable");

} // namespace mdb::gnn
