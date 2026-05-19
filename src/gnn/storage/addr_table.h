// src/gnn/storage/addr_table.h
#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace mdb::gnn {

/**
 * @brief On-disk header for AddrTable sidecar files (40 bytes, little-endian).
 *
 * File layout: [AddrTableHeader: 40 bytes][9 arrays back-to-back, no padding]
 *   l1_positions  [num_l1] uint32  — positions in output tensor for L1 nodes
 *   l1_indices    [num_l1] uint32  — gpu_cache row indices
 *   l2_positions  [num_l2] uint32  — positions in output tensor for L2 nodes
 *   l2_indices    [num_l2] uint32  — cpu_cache row indices
 *   l3_positions  [num_l3] uint32  — positions in output tensor for L3 nodes
 *   l3_row_idxs   [num_l3] uint64  — reordered FeatureMatrix row indices
 *   l4_positions  [num_l4] uint32  — positions in output tensor for L4 nodes
 *   l4_indices    [num_l4] uint32  — packed_slim file slot indices
 *   zero_positions[num_zero] uint32 — positions left as zeros (unresolved)
 */
struct AddrTableHeader {
    static constexpr uint32_t MAGIC   = 0x41444452u;  // "ADDR" (MSB-first, matches storage/ convention)
    static constexpr uint32_t VERSION = 1u;
    static constexpr size_t   SIZE    = 40u;

    uint32_t magic;
    uint32_t version;
    uint32_t num_l1;
    uint32_t num_l2;
    uint32_t num_l3;
    uint32_t num_l4;
    uint32_t num_zero;
    uint32_t total;
    uint64_t meta_sha256_head;

    /// Build a valid header. Precondition: l1 + l2 + l3 + l4 + zero <= UINT32_MAX
    /// (no runtime check — per-batch tier counts cannot realistically approach
    /// 2^32). is_valid() recomputes the same sum, so a wrapped total will not be
    /// detected by validation; callers must ensure this holds.
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
        return h;
    }

    bool is_valid() const {
        return magic == MAGIC
            && version == VERSION
            && total == (num_l1 + num_l2 + num_l3 + num_l4 + num_zero);
    }

    size_t expected_file_size() const {
        return SIZE
            + size_t(num_l1) * sizeof(uint32_t) * 2
            + size_t(num_l2) * sizeof(uint32_t) * 2
            + size_t(num_l3) * sizeof(uint32_t)
            + size_t(num_l3) * sizeof(uint64_t)
            + size_t(num_l4) * sizeof(uint32_t) * 2
            + size_t(num_zero) * sizeof(uint32_t);
    }
};

static_assert(sizeof(AddrTableHeader) == 40,
              "AddrTableHeader must be exactly 40 bytes");
static_assert(std::is_standard_layout_v<AddrTableHeader>,
              "AddrTableHeader must be standard layout for direct I/O");
static_assert(std::is_trivially_copyable_v<AddrTableHeader>,
              "AddrTableHeader must be trivially copyable");

} // namespace mdb::gnn
