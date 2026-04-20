#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <torch/torch.h>

#include "gnn/core/feature_assembler.h"
#include "gnn/storage/cache_file.h"
#include "gnn/storage/cpu_cache.h"
#include "gnn/storage/direct_io_reader.h"
#include "gnn/storage/gpu_cache.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/feature_matrix_header.h"
#include "gnn/storage/gnn_dtype.h"
#include "gnn/storage/packed_batch_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sample_storage.h"
#include "graph_models/object_id.h"

namespace mdb::gnn {

// =============================================================================
// Store Metadata Header (GFLS format, 312 bytes)
// =============================================================================

/**
 * @brief On-disk header for the FourLevelStore metadata file (_store.meta).
 *
 * File layout:
 *   [StoreMetaHeader: 312 bytes]
 *
 * Written once by build(), read by the runtime constructor.
 */
struct StoreMetaHeader {
    static constexpr uint32_t MAGIC   = 0x47464C53; // "GFLS" (MSB-first)
    static constexpr uint32_t VERSION = 1;
    static constexpr size_t   SIZE    = 312;

    uint32_t magic;
    uint32_t version;
    uint64_t l1_count;
    uint64_t l2_count;
    uint64_t l3_count;
    uint64_t l4_count;
    uint64_t feature_dim;
    uint8_t  dtype;
    uint8_t  gpu_available;
    uint8_t  reserved_small[6];
    char     packed_slim_dir[256]; // null-terminated, fixed 256 bytes

    static StoreMetaHeader make(
        uint64_t l1, uint64_t l2, uint64_t l3, uint64_t l4,
        uint64_t dim, GnnDtype dt, bool gpu,
        const std::string& slim_dir)
    {
        StoreMetaHeader h{};
        std::memset(&h, 0, sizeof(h));
        h.magic         = MAGIC;
        h.version       = VERSION;
        h.l1_count      = l1;
        h.l2_count      = l2;
        h.l3_count      = l3;
        h.l4_count      = l4;
        h.feature_dim   = dim;
        h.dtype         = static_cast<uint8_t>(dt);
        h.gpu_available = gpu ? 1 : 0;

        if (slim_dir.size() >= sizeof(h.packed_slim_dir)) {
            throw std::invalid_argument(
                "StoreMetaHeader::make: packed_slim_dir too long (" +
                std::to_string(slim_dir.size()) + " >= 256)");
        }
        std::memcpy(h.packed_slim_dir, slim_dir.c_str(), slim_dir.size());
        // Rest already zero from memset
        return h;
    }

    bool is_valid() const {
        return magic == MAGIC && version == VERSION
            && feature_dim > 0
            && dtype <= static_cast<uint8_t>(GnnDtype::MAX_VALUE);
    }

    GnnDtype get_dtype() const {
        return static_cast<GnnDtype>(dtype);
    }

    std::string get_packed_slim_dir() const {
        // Find null terminator within bounds
        size_t len = strnlen(packed_slim_dir, sizeof(packed_slim_dir));
        return std::string(packed_slim_dir, len);
    }
};

static_assert(sizeof(StoreMetaHeader) == 312,
              "StoreMetaHeader must be exactly 312 bytes");
static_assert(std::is_standard_layout_v<StoreMetaHeader>,
              "StoreMetaHeader must be standard layout for direct I/O");
static_assert(std::is_trivially_copyable_v<StoreMetaHeader>,
              "StoreMetaHeader must be trivially copyable for direct I/O");

// =============================================================================
// FourLevelStore
// =============================================================================

/**
 * @brief Coordinator for four-level feature serving (L1 GPU, L2 CPU, L3 mmap, L4 disk).
 *
 * build() classifies nodes by access frequency into four tiers:
 *   L1 (GpuCache): hottest nodes, stored on GPU for zero-copy access
 *   L2 (CpuCache): warm nodes, stored in pinned CPU memory
 *   L3 (reordered FeatureMatrix): shared nodes (freq > 1), mmap fallback
 *   L4 (packed slim files): cold nodes (freq <= 1), per-batch v2 GNNB files
 *
 * Runtime: load_features() serves L1->L2->L3 for arbitrary node sets.
 * load_batch_features() serves all 4 levels for a specific batch.
 *
 * Thread-safety: After construction, all read methods are safe for concurrent
 * access from multiple threads. Stats are atomic.
 */
class FourLevelStore {
public:
    struct Config {
        GpuCache::Config gpu;
        CpuCache::Config cpu;
        bool   reorder = true;
        bool   force   = false;
        // After build() succeeds, delete the non-slim packed/ directory left
        // over by materialize_batches. This scratch is never read at runtime
        // (training reads packed_slim/) and on large graphs wastes tens of
        // GBs. Default true; set to false only for debugging or backward
        // compatibility with a training path that still reads packed/.
        bool   cleanup_materialize_scratch = true;
        MinHashReorderer::Config minhash;
    };

    struct BuildResult {
        uint64_t l1_nodes = 0, l2_nodes = 0;
        uint64_t l3_nodes = 0, l4_nodes = 0;
        uint64_t total_batches = 0;
        bool     gpu_available = false;
        int64_t  build_time_ms = 0;
        std::string packed_slim_dir;
    };

    /// Preprocessing: classify nodes by frequency, build caches, re-pack L4 slim.
    static BuildResult build(
        const FeatureMatrix&         features,
        const RowMapping&            row_mapping,
        SampleStorage&               samples,
        const Config&                config,
        const std::filesystem::path& db_folder,
        const std::string&           feature_name
    );

    /// Runtime: load all levels from persisted files.
    FourLevelStore(
        const std::filesystem::path& db_folder,
        const std::string&           feature_name,
        SampleStorage&               samples
    );

    /// Primitive: features for a set of nodes (L1 -> L2 -> L3 fallback, no L4).
    torch::Tensor load_features(const std::vector<ObjectId>& oids);

    /// Convenience: features for a batch (all 4 levels including L4).
    torch::Tensor load_batch_features(uint64_t batch_id);

    struct Stats {
        std::atomic<uint64_t> l1_hits{0}, l2_hits{0};
        std::atomic<uint64_t> l3_reads{0}, l4_reads{0};
        std::atomic<uint64_t> total_requests{0};
    };
    Stats& get_stats() { return stats_; }
    const Stats& get_stats() const { return stats_; }

    void reset_stats() {
        stats_.l1_hits.store(0);
        stats_.l2_hits.store(0);
        stats_.l3_reads.store(0);
        stats_.l4_reads.store(0);
        stats_.total_requests.store(0);
    }

    uint64_t feature_dim() const { return feature_dim_; }

private:
    std::unique_ptr<GpuCache> gpu_cache_;
    std::unique_ptr<CpuCache> cpu_cache_;

    // L3: prefer DirectIoReader (zero page cache via O_DIRECT), fallback to mmap
    std::unique_ptr<DirectIoReader> l3_reader_;     // io_uring + O_DIRECT
    std::optional<FeatureMatrix>    l3_mmap_fb_;    // mmap fallback
    std::optional<RowMapping>       reordered_rm_;

    // Assembly: prefer CUDA kernel, fallback to LibTorch index_copy_
    std::unique_ptr<FeatureAssembler> assembler_;

    SampleStorage* samples_ = nullptr;
    std::string    packed_slim_dir_;
    uint64_t       feature_dim_ = 0;
    uint8_t        elem_size_   = 0;
    GnnDtype       dtype_       = GnnDtype::FLOAT32;
    uint64_t       l3_header_size_ = FeatureMatrixHeader::SIZE; // data offset past header
    Stats          stats_;

    /// Map GnnDtype to torch scalar type.
    static torch::ScalarType to_torch_dtype(GnnDtype dt);
};

} // namespace mdb::gnn
