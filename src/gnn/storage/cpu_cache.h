#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <vector>

#include "gnn/storage/cache_file.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/object_id.h"

namespace mdb::gnn {

/**
 * @brief L2 CPU cache — stores top-frequency node features in host memory.
 *
 * build() persists selected node features to a GNNC file on disk.
 * Constructor loads the file into CPU memory (pinned via cudaHostAlloc
 * when CUDA is available for UVA access from GPU, plain malloc otherwise).
 * lookup() returns hits/misses for batch feature assembly.
 *
 * File layout (GNNC format, defined in cache_file.h):
 *   [CacheFileHeader: 32 bytes]
 *   [ObjectId table:  num_nodes * 8 bytes]
 *   [Feature data:    num_nodes * feature_dim * elem_size bytes]
 */
class CpuCache {
public:
    struct Config {
        size_t budget_bytes = 4ULL * 1024 * 1024 * 1024; // 4 GB default
    };

    /// Write cache file from selected nodes.
    /// Reads features from FeatureMatrix via RowMapping, writes GNNC file.
    /// @param nodes      ObjectIds sorted by frequency (descending).
    /// @param features   FeatureMatrix to read rows from.
    /// @param row_mapping Maps ObjectId -> row index in features.
    /// @param output_path Destination file (will be created/truncated).
    static void build(
        const std::vector<ObjectId>& nodes,
        const FeatureMatrix&         features,
        const RowMapping&            row_mapping,
        const std::filesystem::path& output_path
    );

    /// Load cache file into CPU memory.
    /// Uses cudaHostAlloc (pinned) if CUDA available, malloc otherwise.
    explicit CpuCache(const std::filesystem::path& cache_file);
    ~CpuCache();

    // Move only (owns memory)
    CpuCache(CpuCache&& other) noexcept;
    CpuCache& operator=(CpuCache&& other) noexcept;
    CpuCache(const CpuCache&) = delete;
    CpuCache& operator=(const CpuCache&) = delete;

    /// Result of a batch lookup. Features are returned as raw bytes in
    /// input order for hits; hit/miss positions index into the original request.
    struct LookupResult {
        std::vector<char> features;            // [num_hits * D * elem_size] raw bytes
        uint64_t feature_dim;
        uint8_t  elem_size;
        std::vector<uint32_t> hit_positions;   // positions in original input that were found
        std::vector<uint32_t> miss_positions;  // positions in original input that were not found
    };

    /// Lookup a batch of ObjectIds. Returns features for hits and positions
    /// for both hits and misses (to allow the caller to fetch misses from L3/L4).
    LookupResult lookup(const std::vector<ObjectId>& oids) const;

    /// UVA-direct lookup: for each hit oid, returns a pointer into the pinned
    /// feature region. Caller MUST NOT free or modify the returned pointers;
    /// they remain valid for the CpuCache lifetime.
    ///
    /// Round 1B (2026-05-15): zero-copy variant of lookup() — avoids the
    /// per-call std::vector<char> allocation + memcpy that the standard
    /// lookup() performs. The DiskGNN paper accesses L2 features via UVA
    /// directly from the pinned host region; this method exposes the same
    /// path. Callers either (a) consume the pointers via memcpy into their
    /// own destination buffer, or (b) hand the pointers + dst buffer to a
    /// GPU kernel that reads through UVA.
    struct UvaLookupResult {
        std::vector<uint32_t>    hit_positions;  // positions in input oids[]
        std::vector<const void*> hit_pointers;   // pinned-memory ptrs (UVA-accessible)
        std::vector<uint32_t>    miss_positions; // positions in input oids[] that missed
    };

    UvaLookupResult lookup_uva(const std::vector<ObjectId>& oids) const;

    /// Round 1C (2026-05-15): single-hash lookup returning the cache row
    /// index if present, nullopt otherwise. Used by FourLevelStore to
    /// eliminate the double hash on the L2 hit path (previously contains()
    /// then lookup_uva() both call find()).
    std::optional<uint32_t> find_index(ObjectId oid) const;

    /// Round 1C (2026-05-15): return a UVA-accessible pointer to the row at
    /// the given pre-validated cache index. Same lifetime contract as
    /// lookup_uva: pointer is valid for the CpuCache lifetime, caller MUST
    /// NOT free or modify. No bounds-check is performed; caller is
    /// responsible for `idx < num_nodes_` (e.g. via prior find_index).
    const void* row_ptr(uint32_t idx) const;

    bool     contains(ObjectId oid) const;
    uint64_t num_nodes() const;
    uint64_t feature_dim() const;
    size_t   memory_bytes() const;

    /// Raw pointer to feature data (for UVA access from GPU).
    const void* data_ptr() const;
    bool is_pinned() const;

private:
    void* feature_data_ = nullptr;     // cudaHostAlloc or malloc
    size_t feature_data_size_ = 0;
    std::unordered_map<uint64_t, uint32_t> oid_to_idx_;
    uint64_t feature_dim_ = 0;
    uint64_t num_nodes_ = 0;
    uint8_t  elem_size_ = 0;
    bool pinned_ = false;              // true if cudaHostAlloc was used

    void free_data();
};

} // namespace mdb::gnn
