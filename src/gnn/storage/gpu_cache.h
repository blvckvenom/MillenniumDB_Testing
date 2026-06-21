#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "gnn/storage/cache_file.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/object_id.h"

namespace mdb::gnn {

/**
 * @brief L1 GPU cache: stores top-frequency node features as a CUDA tensor.
 *
 * Uses the same GNNC file format as CpuCache. Constructor loads features
 * into a torch::Tensor on CUDA device via from_blob().to(kCUDA). Graceful
 * degradation: keeps tensor on CPU when no GPU is available.
 *
 * Thread-safety: lookup() is read-only after construction. Safe for
 * concurrent reads from multiple threads.
 */
class GpuCache {
public:
    struct Config {
        size_t budget_bytes = 2ULL * 1024 * 1024 * 1024; // 2 GB default
    };

    /// Write cache file in GNNC format (same as CpuCache).
    /// @param nodes     ObjectIds of the nodes to cache (may be empty).
    /// @param features  FeatureMatrix to read row data from.
    /// @param row_mapping Maps ObjectId -> row index in features.
    /// @param output_path Where to write the .bin file.
    static void build(
        const std::vector<ObjectId>& nodes,
        const FeatureMatrix&         features,
        const RowMapping&            row_mapping,
        const std::filesystem::path& output_path
    );

    /// Load cache file to GPU (or CPU fallback if CUDA unavailable).
    /// Empty tensor when file has 0 nodes.
    explicit GpuCache(const std::filesystem::path& cache_file);

    struct LookupResult {
        torch::Tensor             features;       // [num_hits, D] on same device as cache
        std::vector<uint32_t>     hit_positions;   // indices into input oids that were found
        std::vector<uint32_t>     miss_positions;  // indices into input oids that were NOT found
    };

    /// Lookup features for a batch of ObjectIds.
    /// Returns hits gathered into a contiguous tensor, plus position vectors.
    LookupResult lookup(const std::vector<ObjectId>& oids) const;

    /// Single-hash lookup (2026-05-15) returning the cache row index
    /// if present, nullopt otherwise. Used by FourLevelStore to eliminate the
    /// double hash on the L1 hit path (previously contains() then lookup()
    /// both call find()).
    std::optional<uint32_t> find_index(ObjectId oid) const;

    /// Gather feature rows by pre-validated cache row indices (2026-05-15)
    /// (e.g. from find_index). Skips the per-oid find loop in
    /// lookup() and just calls index_select(0, ...). Caller is responsible
    /// for ensuring all indices are valid (< num_nodes_).
    torch::Tensor gather_by_indices(const std::vector<uint32_t>& cache_indices) const;

    bool     contains(ObjectId oid) const;
    uint64_t num_nodes() const;
    uint64_t feature_dim() const;
    size_t   memory_bytes() const;
    bool     is_on_gpu() const;

private:
    torch::Tensor                          features_;     // [N, D] on CUDA or CPU
    std::unordered_map<uint64_t, uint32_t> oid_to_idx_;   // ObjectId.id -> row in features_
    uint64_t feature_dim_ = 0;
    uint64_t num_nodes_   = 0;
    bool     on_gpu_      = false;
};

} // namespace mdb::gnn
