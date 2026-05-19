#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
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
#include "gnn/sampling/graph_sample.h"
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

        // Fix #15 (2026-05-13): granular force flags. When `force=true`,
        // these toggles let callers preserve specific outputs across a
        // rebuild. Useful for validating individual phases without
        // paying the cost of recomputing parts that are still valid.
        //
        // Each defaults to `true` so legacy callers passing only `force`
        // get the historical full-clobber behaviour. Set to `false` to
        // keep that output across a force rebuild — the matching build
        // phase will skip if the file already exists with the right
        // header (Fix #14 idempotency in L1/L2; existing reordered.fmat
        // bypasses the MinHash recompute in L3).
        bool   force_caches      = true;  // delete L1 gpu_cache + L2 cpu_cache
        bool   force_reorder     = true;  // delete reordered.fmat + .rmap
        bool   force_packed_slim = true;  // delete packed_slim/
        bool   force_meta        = true;  // delete store.meta

        // Phase 5 (2026-05-19): pre-classify every batch's unique nodes into
        // {L1, L2, L4, L3, zero} and persist the results as
        // addr_tables/batch_NNNNNN.addr sidecars next to packed_slim/.
        // Set to false only to skip Phase 5 (e.g., when rebuilding caches only).
        bool   build_addr_tables = true;

        // After build() succeeds, delete the non-slim packed/ directory left
        // over by materialize_batches. This scratch is never read at runtime
        // (training reads packed_slim/) and on large graphs wastes tens of
        // GBs. Default true; set to false only for debugging or backward
        // compatibility with a training path that still reads packed/.
        bool   cleanup_materialize_scratch = true;

        // Spec D (telemetry, 2026-05-07): disk space budget for the feature
        // store. 0 = unlimited (current behavior). When > 0, build() emits
        // a warning if the actual usage exceeds the budget. Future Spec C2
        // will use this to drive the heuristic search for segment_size that
        // satisfies the constraint. Mirrors DiskGNN's `disk_size` parameter
        // (cf. SIGMOD'25 §6 API `DiskGNN_train(..., disk_size, ...)`).
        size_t disk_budget_bytes = 0;
        MinHashReorderer::Config minhash;
    };

    struct BuildResult {
        uint64_t l1_nodes = 0, l2_nodes = 0;
        uint64_t l3_nodes = 0, l4_nodes = 0;
        uint64_t total_batches = 0;
        bool     gpu_available = false;
        int64_t  build_time_ms = 0;
        std::string packed_slim_dir;

        // Spec D telemetry: per-tier on-disk byte accounting. All values are
        // post-build, measured from the actual filesystem (NOT estimated).
        // - slim_bytes:       sum of packed_slim/*.bin file sizes
        // - reordered_bytes:  size of *_reordered.fmat (0 if reorder=false)
        // - gpu_cache_bytes:  size of *_gpu_cache.bin
        // - cpu_cache_bytes:  size of *_cpu_cache.bin
        // - total_disk_bytes: sum of the above (== feature store on-disk footprint)
        // - over_budget:      true iff config.disk_budget_bytes > 0 AND
        //                     total_disk_bytes > config.disk_budget_bytes.
        uint64_t slim_bytes       = 0;
        uint64_t reordered_bytes  = 0;
        uint64_t gpu_cache_bytes  = 0;
        uint64_t cpu_cache_bytes  = 0;
        uint64_t total_disk_bytes = 0;
        bool     over_budget      = false;

        // Phase 5 (2026-05-19): addr_table sidecar telemetry.
        // addr_tables_bytes — total bytes written to addr_tables/*.addr files.
        // addr_tables_built_ok — true iff Phase 5 completed without error.
        uint64_t addr_tables_bytes    = 0;
        bool     addr_tables_built_ok = false;
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

    /// Destructor: releases the persistent pinned host buffer (if any).
    ~FourLevelStore();

    /// Primitive: features for a set of nodes (L1 -> L2 -> L3 fallback, no L4).
    torch::Tensor load_features(const std::vector<ObjectId>& oids);

    /// Convenience: features for a batch (all 4 levels including L4).
    /// Legacy: looks up the sample by batch_id via SampleStorage, then dispatches
    /// to the overload below. Pays the deserialize cost once.
    torch::Tensor load_batch_features(uint64_t batch_id);

    /// Round 2B (2026-05-15): Load batch features given an already-deserialized
    /// GraphSample. Avoids re-reading + re-parsing the sample when the caller
    /// (e.g., BatchAssembler::assemble_from_sample) already has it in hand.
    torch::Tensor load_batch_features(const GraphSample& sample);

    struct Stats {
        // Per-tier node-count counters (existing).
        std::atomic<uint64_t> l1_hits{0}, l2_hits{0};
        std::atomic<uint64_t> l3_reads{0}, l4_reads{0};
        std::atomic<uint64_t> total_requests{0};

        // Spec A1 (2026-04-27): byte-level counters for paper-comparable
        // disk-traffic accounting (cf. DiskGNN SIGMOD'25 Table 1
        // "Disk access volume (GB)").
        //
        // *_bytes_served — bytes copied from RAM/GPU caches into output.
        //                  Useful for hit-rate by data volume vs node count.
        // l3_bytes_wanted — feature payload bytes extracted from L3.
        // l3_bytes_disk   — physical bytes read from disk via O_DIRECT.
        //                   Equals l3_bytes_wanted only if every row was
        //                   block-aligned and adjacent; otherwise reflects
        //                   alignment overhead (Spec A2 will reduce it).
        // l4_bytes_wanted — feature payload bytes extracted from L4 slim.
        // l4_bytes_disk   — bytes read from slim files (header + OID table
        //                   + feature data); counted once per batch read.
        std::atomic<uint64_t> l1_bytes_served{0};
        std::atomic<uint64_t> l2_bytes_served{0};
        std::atomic<uint64_t> l3_bytes_wanted{0};
        std::atomic<uint64_t> l3_bytes_disk{0};
        std::atomic<uint64_t> l4_bytes_wanted{0};
        std::atomic<uint64_t> l4_bytes_disk{0};
    };
    Stats& get_stats() { return stats_; }
    const Stats& get_stats() const { return stats_; }

    void reset_stats() {
        stats_.l1_hits.store(0);
        stats_.l2_hits.store(0);
        stats_.l3_reads.store(0);
        stats_.l4_reads.store(0);
        stats_.total_requests.store(0);
        stats_.l1_bytes_served.store(0);
        stats_.l2_bytes_served.store(0);
        stats_.l3_bytes_wanted.store(0);
        stats_.l3_bytes_disk.store(0);
        stats_.l4_bytes_wanted.store(0);
        stats_.l4_bytes_disk.store(0);
    }

    uint64_t feature_dim() const { return feature_dim_; }

    // Phase 0 (2026-05-17) profile instrumentation getters. Each returns the
    // per-tier microseconds captured during the most recent
    // load_batch_features() call. last_rmap_us() is a SUB-counter and its time
    // is already included inside the L1/L2/L3/L4 totals; it is tracked
    // separately to quantify the Phase 1 address-table candidate.
    //
    // Phase 0 (2026-05-17): public API still in microseconds (matches the
    // `_us` fields of BatchTiming). Internal storage uses nanoseconds for
    // precision; conversion happens here, once per batch (not per node).
    uint64_t last_l1_us() const   { return last_l1_ns_ / 1000; }
    uint64_t last_l2_us() const   { return last_l2_ns_ / 1000; }
    uint64_t last_l3_us() const   { return last_l3_ns_ / 1000; }
    uint64_t last_l4_us() const   { return last_l4_ns_ / 1000; }
    uint64_t last_rmap_us() const { return last_rmap_ns_ / 1000; }

    // Path 4 (2026-05-19): v2 runtime path telemetry.
    // last_addr_load_us() — microseconds to open + parse the addr_table sidecar
    //   for the most recent batch. 0 if the v2 path was not taken.
    // last_used_addr_tables() — whether the most recent load_batch_features
    //   was served by the v2 (addr_table) path.
    uint64_t last_addr_load_us() const { return last_addr_load_ns_ / 1000; }
    bool last_used_addr_tables() const { return last_used_v2_; }

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

    // Round 1A optimization (2026-05-15): persistent pinned host buffer reused
    // across load_batch_features() calls. Replaces per-batch
    // cudaHostAlloc + cudaFreeHost (each is a synchronous driver call
    // ~100-500 us). On papers100M with 1300+ batches/epoch the old path
    // burned ~150-600 ms/epoch on alloc churn alone.
    //
    // ensure_pinned_capacity(bytes) grows the buffer if needed (geometric
    // x1.5 plus 64-byte alignment headroom) and is thread-safe via
    // pinned_mutex_. The buffer outlives any single batch and is freed
    // exactly once in the destructor.
    mutable std::mutex pinned_mutex_;
    void*              pinned_ptr_      = nullptr;
    size_t             pinned_capacity_ = 0;

    // Phase 0 (2026-05-17) profile instrumentation. Per-call sub-timers in
    // nanoseconds (high precision to avoid integer-microsecond truncation of
    // sub-μs hash lookups). Accessors below convert to μs at the API boundary.
    // rmap_lookup_ns is a SUB-counter: already included in the L3 total;
    // tracked separately to quantify Phase 1 address-table candidate.
    mutable uint64_t last_l1_ns_   = 0;
    mutable uint64_t last_l2_ns_   = 0;
    mutable uint64_t last_l3_ns_   = 0;
    mutable uint64_t last_l4_ns_   = 0;
    mutable uint64_t last_rmap_ns_ = 0;

    // Path 4 (2026-05-19): v2 path dispatch state.
    //
    // use_addr_tables_ — set true at construction when addr_tables/ directory
    //   exists alongside the packed_slim/ dir. Controls whether the dispatcher
    //   probes for batch_NNNNNN.addr before falling back to legacy.
    //
    // expected_meta_sha_head_ — FNV-64 of gnn_meta.bin at construction time.
    //   Matches the hash written by build_addr_tables_ at build time.
    //   AddrTableReader::open rejects sidecars with a different hash
    //   (AddrTableStaleException). 0 disables staleness check.
    //
    // sample_dir_ — parent of packed_slim_dir_; addr_tables/ lives here.
    //
    // Per-call v2 telemetry:
    //   last_addr_load_ns_ — nanoseconds to open + parse the addr sidecar.
    //   last_used_v2_      — true if the most recent batch went through v2.
    bool                  use_addr_tables_    = false;
    uint64_t              expected_meta_sha_head_ = 0;
    std::filesystem::path sample_dir_;
    mutable uint64_t      last_addr_load_ns_  = 0;
    mutable bool          last_used_v2_       = false;

    /// Ensure the persistent pinned buffer is at least `bytes` long.
    /// Returns true on success, false if cudaHostAlloc failed (caller falls
    /// back to unpinned memory). No-op when GNN_CUDA_ENABLED is undefined.
    bool ensure_pinned_capacity(size_t bytes);

    /// Map GnnDtype to torch scalar type.
    static torch::ScalarType to_torch_dtype(GnnDtype dt);

    /// Legacy path: per-node hash classification across L1/L2/L4/L3, scatter
    /// via FeatureAssembler. Preserves pre-Path-4 behavior bit-identically.
    /// Public dispatcher falls through to this when the addr_table sidecar is
    /// missing, stale, or unreadable, or when the assembler gate is not met.
    torch::Tensor load_batch_features_legacy_(const GraphSample& sample);

    /// Path 4 fast path (2026-05-19): read addr_tables/batch_<bid>.addr,
    /// dispatch scatter-from-tier without per-node classification.
    /// Only called when assembler_ is active and gpu_cache_->is_on_gpu() —
    /// the CPU-only assembly case is served by legacy_ to avoid duplicating
    /// that path's memcpy logic. Throws AddrTableStaleException on meta_sha
    /// mismatch so the dispatcher can catch and fall back cleanly.
    torch::Tensor load_batch_features_v2_(const GraphSample& sample,
                                           const std::filesystem::path& addr_path);
};

} // namespace mdb::gnn
