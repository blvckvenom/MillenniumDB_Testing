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
//   [uint64 seed_id  : num_seeds]                    (v2 self-contained: seed ObjectId.ids;
//                                                     == nodes_per_layer[0]; empty when num_seeds==0)
// active gather tensors are arange(M_k) — NOT stored. edge_index is int32 on disk,
// widened to int64 at load. Keyed by sample_fp for staleness (== compute_batch_content_hash).
//
// v2 (VERSION=2) additionally carries — in the header — what gnn_train needs from the
// sample so the train path can skip batches.dat: store_fp (sample-set fingerprint for
// setup-time staleness), num_unique_nodes (validated against addr_table.total), num_seeds
// (seed ids appended to the body, for label gather), and split. A v2 block with
// store_fp==0 is format-v2 but NOT self-contained (e.g. a non-self-contained writer call).
// v1 blocks (no seed bytes, all-zero v2 fields) remain readable for backward compat.
struct BlockBatchHeader {
    static constexpr uint32_t MAGIC   = 0x474E424Bu;  // "GNBK"
    static constexpr uint32_t VERSION = 2u;
    static constexpr size_t   SIZE    = 64u;
    uint32_t magic;
    uint32_t version;
    uint64_t sample_fp;          // == compute_batch_content_hash(sample) at bake time
    uint32_t num_layers;         // == fanout length == nodes_per_layer.size()-1 (= K conv layers)
    uint32_t reserved0;
    uint64_t batch_id;
    // --- v2 self-contained fields (32 B; replaces the former reserved1[4]) ---
    uint64_t store_fp;           // sample-set fingerprint (catalog.sample_content_fp) at bake; 0 = not self-contained
    uint64_t num_unique_nodes;   // all_unique_nodes.size() (v2 feature path validates against addr_table.total)
    uint64_t num_seeds;          // nodes_per_layer[0].size() (seeds appended to body as uint64 ids)
    uint32_t split;              // SplitType of the batch
    uint32_t reserved2;          // pad to keep 64 B
    // v2-format block that is NOT self-contained (store_fp/num_unique_nodes/num_seeds/split = 0).
    static BlockBatchHeader make(uint32_t num_layers, uint64_t sample_fp, uint64_t batch_id = 0) {
        BlockBatchHeader h{};
        h.magic = MAGIC; h.version = VERSION;
        h.sample_fp = sample_fp; h.num_layers = num_layers; h.batch_id = batch_id;
        return h;
    }
    // v2 self-contained block with all the train-needs-from-sample fields set.
    static BlockBatchHeader make_self_contained(uint32_t num_layers, uint64_t sample_fp,
                                                uint64_t batch_id, uint64_t store_fp,
                                                uint64_t num_unique_nodes, uint64_t num_seeds,
                                                uint32_t split) {
        BlockBatchHeader h{};
        h.magic = MAGIC; h.version = VERSION;
        h.sample_fp = sample_fp; h.num_layers = num_layers; h.batch_id = batch_id;
        h.store_fp = store_fp; h.num_unique_nodes = num_unique_nodes;
        h.num_seeds = num_seeds; h.split = split;
        return h;
    }
    // Accept BOTH v1 and v2 on-disk blocks for backward-compat reads.
    bool is_valid() const {
        return magic == MAGIC && (version == 1u || version == 2u) && num_layers > 0;
    }
    bool is_self_contained() const { return version >= 2u && store_fp != 0; }
};
static_assert(sizeof(BlockBatchHeader) == 64, "BlockBatchHeader must be 64 bytes");
static_assert(std::is_trivially_copyable_v<BlockBatchHeader>, "trivially copyable");
static_assert(std::is_standard_layout_v<BlockBatchHeader>, "standard layout");
} // namespace mdb::gnn
