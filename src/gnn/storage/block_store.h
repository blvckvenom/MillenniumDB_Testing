// src/gnn/storage/block_store.h
#pragma once
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <vector>
#include <torch/torch.h>
namespace mdb::gnn {

// Shared baked-block filename helper (DRY across the offline bake in
// four_level_store.cc and the train-time consume in batch_assembler.cc).
// Mirrors the addr_table naming convention: "block_%06...blk".
inline std::filesystem::path block_filename(const std::filesystem::path& dir, uint64_t batch_id) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "block_%06llu.blk",
                  static_cast<unsigned long long>(batch_id));
    return dir / buf;
}

struct LoadedBlock {
    std::vector<int64_t>       active_sizes;   // M_k, K+1 values
    std::vector<torch::Tensor> edge_indices;   // K tensors, each [2,E_k] int64 (widened from int32)
    // v2 self-contained fields (0 / empty for legacy v1 or non-self-contained v2 blocks).
    uint64_t                   store_fp = 0;           // catalog.sample_content_fp at bake (0 = not self-contained)
    uint64_t                   num_unique_nodes = 0;   // all_unique_nodes.size() (validated vs addr_table.total)
    uint32_t                   split = 0;              // SplitType of the batch
    std::vector<uint64_t>      seed_ids;               // seed ObjectId.ids (num_seeds == seed_ids.size())
};
struct BlockWriter {
    // edge_indices[k] must be a [2,E_k] int64 CPU tensor (values < 2^31). Written int32.
    // The trailing v2 self-contained params are defaulted so existing 5-arg callers are
    // unchanged: with default seed_ids (empty) / store_fp (0) the on-disk block carries
    // no seed bytes and is NOT self-contained.
    static void write(const std::filesystem::path& path, uint64_t sample_fp, uint64_t batch_id,
                      const std::vector<int64_t>& active_sizes,
                      const std::vector<torch::Tensor>& edge_indices,
                      uint64_t store_fp = 0, uint64_t num_unique_nodes = 0,
                      const std::vector<uint64_t>& seed_ids = {}, uint32_t split = 0);
};
struct BlockReader {
    // Returns nullopt if missing / bad magic-version / sample_fp mismatch (stale).
    // Populates the v2 self-contained fields from the header + appended seed body when
    // present; for a legacy v1 block they stay 0 / empty.
    static std::optional<LoadedBlock> open(const std::filesystem::path& path, uint64_t expected_sample_fp);

    // SC-3 self-contained open: validates via the STORE fingerprint
    // (catalog.sample_content_fp), NOT the per-batch sample_fp. Returns nullopt
    // unless the header is valid AND self-contained (version>=2 && store_fp!=0)
    // AND h.store_fp == expected_store_fp AND expected_store_fp != 0. On match
    // reads the full body exactly like open() (active_sizes, edge tensors widened
    // int32->int64) plus the seed_ids tail, and populates num_unique_nodes /
    // split / store_fp / seed_ids. Used by the train-time self-contained fast
    // path to build a minimal sample WITHOUT reading batches.dat.
    static std::optional<LoadedBlock> open_self_contained(
        const std::filesystem::path& path, uint64_t expected_store_fp);

    // Cheap freshness check: reads ONLY the 64-byte header and returns true iff
    // magic+version valid AND sample_fp == expected. Does NOT read the body.
    // For the bake-skip decision; a torn block (header ok, body truncated) is
    // impossible post-crash (atomic fsync+rename) and would still fall back to
    // online at train time via open() returning nullopt.
    static bool is_fresh(const std::filesystem::path& path, uint64_t expected_sample_fp);

    // Header-only read of the store-level fingerprint (catalog.sample_content_fp at bake).
    // Returns 0 on open-fail / short-read / invalid header. Used at train setup to compare
    // against catalog.sample_content_fp without reading the body.
    static uint64_t read_store_fp(const std::filesystem::path& path);
};
} // namespace mdb::gnn
