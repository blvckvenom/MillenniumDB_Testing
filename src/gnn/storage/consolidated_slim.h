// src/gnn/storage/consolidated_slim.h
#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace mdb::gnn {

/**
 * @brief On-disk header for the consolidated cold-feature file (64 bytes, LE).
 *
 * Consolidates the packed cold features of every batch into a single file:
 * instead of one `batch_NNNNNN.bin` per batch (thousands of small buffered
 * reads that contend for page cache — measured ~0.2 GB/s on a Gen4 NVMe),
 * all batches' cold-feature payloads are concatenated into ONE file
 * `<sample_dir>/packed_slim/consolidated.slim`, batch `b` at `slim_offset[b]`
 * (stored per-batch in the v2 AddrTableHeader), read with ONE O_DIRECT
 * sequential pread per batch (~1.4 GB/s on the same drive). The idea of
 * storing each mini-batch's features contiguously so one sequential read
 * serves the whole batch is "feature packing" (DiskGNN, SIGMOD'25 §5.2).
 *
 * File layout: [ConsolidatedSlimHeader: 64 B][pad to data_start][batch 0 payload]
 *              [pad to 4096][batch 1 payload][pad]... Each batch payload is the
 * full cold-data section (no per-batch header, no OID table): N_cold_b*row_bytes,
 * aligned up to `alignment`. Row order matches the per-batch `.bin` data section
 * (partition-iteration order), so addr_table.l4_indices[j] index rows identically.
 *
 * Stale-rejection: `perm_fingerprint` must equal the reordered `.rmap.idx`
 * permutation fingerprint (see row_mapping.h) and `meta_sha256_head` the
 * store's expected value; a mismatch at open => refuse the consolidated file,
 * fall back to the per-batch read. A stale file from a previous permutation
 * serves rows in the wrong order — training still runs to completion but
 * accuracy silently collapses — so the fingerprints are mandatory.
 */
struct ConsolidatedSlimHeader {
    static constexpr uint32_t MAGIC   = 0x43534C4Du;  // "CSLM" (MSB-first, matches storage/ convention)
    static constexpr uint32_t VERSION = 1u;
    static constexpr size_t   SIZE    = 64u;

    uint32_t magic;
    uint32_t version;
    uint64_t num_batches;       // == catalog total_batches
    uint64_t feature_dim;       // D
    uint8_t  dtype;             // GnnDtype enum value
    uint8_t  alignment_log2;    // 12 => 4096-byte O_DIRECT alignment
    uint8_t  reserved0[6];
    uint64_t perm_fingerprint;  // == reordered .rmap.idx IDX_VERSION-2 fingerprint (0 when reorder disabled)
    uint64_t meta_sha256_head;  // == addr_table meta_sha256_head / FourLevelStore expected_meta_sha_head_
    uint64_t data_start;        // byte offset of batch 0 payload == align_up(SIZE, alignment)
    uint64_t reserved1;

    /// Round `v` up to the next multiple of `a` (a power of two in practice).
    static uint64_t align_up(uint64_t v, uint64_t a) {
        return (a == 0) ? v : ((v + a - 1) / a) * a;
    }

    /// Build a valid header. `alignment_log2` defaults to 12 (4096-byte pages).
    static ConsolidatedSlimHeader make(uint64_t num_batches, uint64_t feature_dim,
                                       uint8_t dtype, uint64_t perm_fp,
                                       uint64_t meta_sha_head,
                                       uint8_t alignment_log2 = 12)
    {
        ConsolidatedSlimHeader h{};
        h.magic            = MAGIC;
        h.version          = VERSION;
        h.num_batches      = num_batches;
        h.feature_dim      = feature_dim;
        h.dtype            = dtype;
        h.alignment_log2   = alignment_log2;
        h.perm_fingerprint = perm_fp;
        h.meta_sha256_head = meta_sha_head;
        h.data_start       = align_up(SIZE, uint64_t(1) << alignment_log2);
        return h;
    }

    uint64_t alignment() const { return uint64_t(1) << alignment_log2; }

    bool is_valid() const {
        return magic == MAGIC
            && version == VERSION
            && feature_dim > 0
            && alignment_log2 >= 9 && alignment_log2 <= 30   // 512 B .. 1 GiB sane page size
            && data_start >= SIZE
            && data_start == align_up(SIZE, alignment());
    }
};

static_assert(sizeof(ConsolidatedSlimHeader) == 64,
              "ConsolidatedSlimHeader must be exactly 64 bytes");
static_assert(std::is_standard_layout_v<ConsolidatedSlimHeader>,
              "ConsolidatedSlimHeader must be standard layout for direct I/O");
static_assert(std::is_trivially_copyable_v<ConsolidatedSlimHeader>,
              "ConsolidatedSlimHeader must be trivially copyable for direct I/O");

} // namespace mdb::gnn
