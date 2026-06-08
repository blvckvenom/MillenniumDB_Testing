// src/gnn/storage/packed_full_format.h
#pragma once
#include <cstdint>
#include <cstddef>
#include <type_traits>
namespace mdb::gnn {

// Per-batch fully-packed feature store. Two files in <sample_dir>/packed_full/:
//   packed_full.idx : [PackedFullHeader 64B][PackedFullEntry x num_batches]
//   packed_full.dat : per-batch payloads, each = num_nodes_b * row_bytes, the
//                     features of all_unique_nodes[b] in order, row-major, in the
//                     feature dtype; each payload aligned up to `alignment` so the
//                     per-batch read is O_DIRECT-aligned.
// Train reads packed_full.dat[entry.offset : entry.offset+entry.length] with ONE
// O_DIRECT read and uses it directly as the batch feature tensor [N_b, D] - no
// 4-tier scatter, no addr_table. Keyed by store_fp (== mix_feature_store_fingerprint).
struct PackedFullEntry {
    uint64_t offset;     // byte offset into packed_full.dat (alignment-aligned)
    uint64_t length;     // payload bytes (== num_nodes * row_bytes)
    uint64_t num_nodes;  // N_b
};
static_assert(sizeof(PackedFullEntry) == 24, "PackedFullEntry must be 24 bytes");
static_assert(std::is_trivially_copyable_v<PackedFullEntry>, "trivially copyable");

struct PackedFullHeader {
    static constexpr uint32_t MAGIC     = 0x474E4650u;  // "GNFP"
    static constexpr uint32_t VERSION   = 1u;
    static constexpr size_t   SIZE      = 64u;
    static constexpr uint32_t ALIGNMENT = 4096u;        // O_DIRECT block alignment
    uint32_t magic;
    uint32_t version;
    uint64_t store_fp;       // == mix_feature_store_fingerprint(...); 0 = unknown
    uint64_t num_batches;
    uint64_t row_bytes;      // feature_dim * dtype_size
    uint32_t feature_dim;
    uint32_t dtype;          // GnnDtype as uint32
    uint32_t alignment;      // ALIGNMENT
    uint32_t reserved0;
    uint64_t reserved1[2];
    static PackedFullHeader make(uint64_t store_fp, uint32_t feature_dim, uint32_t dtype,
                                 uint64_t num_batches, uint64_t row_bytes) {
        PackedFullHeader h{};
        h.magic = MAGIC; h.version = VERSION;
        h.store_fp = store_fp; h.feature_dim = feature_dim; h.dtype = dtype;
        h.num_batches = num_batches; h.row_bytes = row_bytes; h.alignment = ALIGNMENT;
        return h;
    }
    bool is_valid() const {
        return magic == MAGIC && version == VERSION && row_bytes > 0 && alignment > 0;
    }
};
static_assert(sizeof(PackedFullHeader) == 64, "PackedFullHeader must be 64 bytes");
static_assert(std::is_trivially_copyable_v<PackedFullHeader>, "trivially copyable");
static_assert(std::is_standard_layout_v<PackedFullHeader>, "standard layout");
} // namespace mdb::gnn
