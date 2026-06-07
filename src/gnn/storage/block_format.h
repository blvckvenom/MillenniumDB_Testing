// src/gnn/storage/block_format.h
#pragma once
#include <cstdint>
#include <cstddef>
#include <type_traits>
namespace mdb::gnn {

// Per-batch baked computation-graph block. File layout (LE):
//   [BlockBatchHeader: 64 B]
//   [int64 M_k       for k in 0..num_layers]        (active-set size per node-layer; K+1 values)
//   [int64 E_k       for k in 0..num_layers-1]      (edge count per conv-layer; K values)
//   [int32 edge_index[k] : 2*E_k]  for k in 0..num_layers-1   (row-major [2,E_k]; local indices)
// active gather tensors are arange(M_k) — NOT stored. edge_index is int32 on disk,
// widened to int64 at load. Keyed by sample_fp for staleness (== compute_batch_content_hash).
struct BlockBatchHeader {
    static constexpr uint32_t MAGIC   = 0x474E424Bu;  // "GNBK"
    static constexpr uint32_t VERSION = 1u;
    static constexpr size_t   SIZE    = 64u;
    uint32_t magic;
    uint32_t version;
    uint64_t sample_fp;       // == compute_batch_content_hash(sample) at bake time
    uint32_t num_layers;      // == fanout length == nodes_per_layer.size()-1 (= K conv layers)
    uint32_t reserved0;
    uint64_t batch_id;
    uint64_t reserved1[4];
    static BlockBatchHeader make(uint32_t num_layers, uint64_t sample_fp, uint64_t batch_id = 0) {
        BlockBatchHeader h{};
        h.magic = MAGIC; h.version = VERSION;
        h.sample_fp = sample_fp; h.num_layers = num_layers; h.batch_id = batch_id;
        return h;
    }
    bool is_valid() const { return magic == MAGIC && version == VERSION && num_layers > 0; }
};
static_assert(sizeof(BlockBatchHeader) == 64, "BlockBatchHeader must be 64 bytes");
static_assert(std::is_trivially_copyable_v<BlockBatchHeader>, "trivially copyable");
static_assert(std::is_standard_layout_v<BlockBatchHeader>, "standard layout");
} // namespace mdb::gnn
