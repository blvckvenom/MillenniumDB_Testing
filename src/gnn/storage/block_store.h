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
};
struct BlockWriter {
    // edge_indices[k] must be a [2,E_k] int64 CPU tensor (values < 2^31). Written int32.
    static void write(const std::filesystem::path& path, uint64_t sample_fp, uint64_t batch_id,
                      const std::vector<int64_t>& active_sizes,
                      const std::vector<torch::Tensor>& edge_indices);
};
struct BlockReader {
    // Returns nullopt if missing / bad magic-version / sample_fp mismatch (stale).
    static std::optional<LoadedBlock> open(const std::filesystem::path& path, uint64_t expected_sample_fp);

    // Cheap freshness check: reads ONLY the 64-byte header and returns true iff
    // magic+version valid AND sample_fp == expected. Does NOT read the body.
    // For the bake-skip decision; a torn block (header ok, body truncated) is
    // impossible post-crash (atomic fsync+rename) and would still fall back to
    // online at train time via open() returning nullopt.
    static bool is_fresh(const std::filesystem::path& path, uint64_t expected_sample_fp);
};
} // namespace mdb::gnn
