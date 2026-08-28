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
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "gnn/core/feature_assembler.h"
#include "gnn/storage/addr_table_reader.h"
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

        // Granular force flags: when `force=true`, these toggles
        // let callers preserve specific outputs across a rebuild. Useful for
        // validating individual phases without paying the cost of recomputing
        // parts that are still valid.
        //
        // Each defaults to `true` so legacy callers passing only `force`
        // get the historical full-clobber behaviour. Set to `false` to
        // keep that output across a force rebuild — the matching build
        // phase will skip if the file already exists with the right
        // header (L1/L2 cache files are checked for validity before
        // overwriting; an existing reordered.fmat bypasses the MinHash
        // recompute in L3).
        bool   force_caches      = true;  // delete L1 gpu_cache + L2 cpu_cache
        bool   force_reorder     = true;  // delete reordered.fmat + .rmap
        bool   force_packed_slim = true;  // delete packed_slim/
        bool   force_meta        = true;  // delete store.meta

        // Pre-classify every batch's unique nodes into
        // {L1, L2, L4, L3, zero} and persist the results as
        // addr_tables/batch_NNNNNN.addr sidecars next to packed_slim/.
        // Set to false only to skip that stage (e.g., when rebuilding caches
        // only).
        bool   build_addr_tables = true;

        // Bake per-batch computation-graph blocks (blocks/block_NNNNNN.blk)
        // keyed by sample content hash. Default OFF; consumed by BatchAssembler
        // when present+fresh.
        bool   bake_blocks = false;

        // Packed-full build mode (additive): a single gather pass over
        // the source fmat into <sample_dir>/packed_full/ (one contiguous
        // [N_b, D] pack per batch, in all_unique_nodes order). Writes ONLY
        // packed_full/; never builds or deletes the 4-tier (reordered/caches/
        // packed_slim/addr_tables) or blocks/. Requires store.meta + blocks/ to
        // already exist (a prior bakeBlocks build). The train-time consumer
        // prefers this pack. Default OFF.
        bool   pack_full = false;

        // When true, also emit a single consolidated cold-feature file
        // (packed_slim/consolidated.slim) during the partitioned L4 pack, so
        // the runtime can serve each batch's cold features with ONE O_DIRECT
        // sequential pread instead of opening ~1512 small per-batch files. When
        // true, the addr_table sidecars are written as v2 (carrying per-batch
        // slim_offset/slim_length). Opt-in, default OFF; the per-batch .bin files
        // are still written so the legacy read path remains valid. Requires the
        // partitioned packer; ignored on the legacy worker-loop path.
        bool   write_consolidated_slim = false;

        // After build() succeeds, delete the non-slim packed/ directory left
        // over by materialize_batches. This scratch is never read at runtime
        // (training reads packed_slim/) and on large graphs wastes tens of
        // GBs. Default true; set to false only for debugging or backward
        // compatibility with a training path that still reads packed/.
        bool   cleanup_materialize_scratch = true;

        // Defer the L1/L2 cache (.bin) materialization to train startup.
        // When true, build() classifies tiers and writes
        // store.meta but does NOT write the gpu/cpu cache .bin; it drops a
        // "<feature>_cache.deferred" marker instead. The runtime constructor
        // then builds the L1/L2 caches lazily on first train, sized for THIS
        // host's VRAM/RAM (detect_resources), eliminating the build-host !=
        // train-host cache-size mismatch and keeping the .bin out of the build
        // artifacts. Default false (caches written at build, as before).
        bool   no_cache_bin = false;

        // Disk space budget for the Four-Level Feature Store:
        // 0 = unlimited (current behavior). When > 0, build() emits
        // a warning if the actual on-disk usage exceeds the budget. A future
        // heuristic-search phase would use this to find the disk-cache segment
        // size that satisfies the constraint. Mirrors DiskGNN's `disk_size`
        // parameter (cf. SIGMOD'25 §6 API `DiskGNN_train(..., disk_size, ...)`).
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

        // Per-tier on-disk byte accounting for the Four-Level Feature Store.
        // All values are post-build, measured from the actual filesystem (NOT
        // estimated).
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

        // addr_table sidecar telemetry.
        // addr_tables_bytes — total bytes written to addr_tables/*.addr files.
        // addr_tables_built_ok — true iff the addr-table stage completed
        //   without error.
        uint64_t addr_tables_bytes    = 0;
        bool     addr_tables_built_ok = false;

        // Baked computation-graph block telemetry.
        // blocks_bytes — total bytes written to blocks/*.blk files.
        // blocks_built_ok — true iff block baking completed without error.
        uint64_t blocks_bytes    = 0;
        bool     blocks_built_ok = false;

        // Packed-full feature pack telemetry.
        // packed_full_bytes — size of packed_full/packed_full.dat (0 if not built).
        uint64_t packed_full_bytes = 0;
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

    /// Rebuild addr_tables/ sidecars from this loaded runtime instance's
    /// already-resolved caches (gpu/cpu/reordered_rm).
    /// Used when the source FeatureMatrix is unavailable (placeholder /
    /// deleted), but the rest of the feature store is intact. Idempotent —
    /// overwrites any existing addr_tables/. Returns total bytes written.
    ///
    /// Side-effect: enables use_addr_tables_ and sets expected_meta_sha_head_
    /// so this instance's load_batch_features() can immediately serve via
    /// the v2 fast path (no need to re-construct the FourLevelStore).
    uint64_t rebuild_addr_tables(const std::filesystem::path& db_folder,
                                 bool bake_blocks = false,
                                 uint64_t* out_blocks_bytes = nullptr);

    /// Resolve the per-projection gnn_meta.bin path used as the addr-table
    /// staleness marker. gnn_meta.bin lives in the PROJECTION directory
    /// (<db_folder>/projections/<projection_name>/gnn_meta.bin), NOT the db
    /// root. Earlier code hashed <db_folder>/gnn_meta.bin, which never exists,
    /// so compute_meta_sha_head() always returned 0 and the staleness check was
    /// a silent no-op. Public + static so the staleness contract is unit-testable.
    static std::filesystem::path gnn_meta_path_for(
        const std::filesystem::path& db_folder,
        const std::string&           projection_name);

    /// Staleness check: return true iff a built feature store for `feature_name` exists
    /// AND its persisted content fingerprint (`<feature>_store.fp`) matches the
    /// given sample's content fingerprint mixed with the store's own identity
    /// (feature dim + dtype, read from store.meta). Used by the
    /// gnn_build_feature_store phase5-only fast path to refuse reusing
    /// addr_tables built from a different sample. Returns false if either
    /// sidecar is absent/unreadable or `sample_content_fp` is 0 (UNKNOWN).
    static bool store_matches_sample_fp(
        const std::filesystem::path& db_folder,
        const std::string&           feature_name,
        uint64_t                     sample_content_fp);

    /// Primitive: features for a set of nodes (L1 -> L2 -> L3 fallback, no L4).
    torch::Tensor load_features(const std::vector<ObjectId>& oids);

    /// Convenience: features for a batch (all 4 levels including L4).
    /// Legacy: looks up the sample by batch_id via SampleStorage, then dispatches
    /// to the overload below. Pays the deserialize cost once.
    torch::Tensor load_batch_features(uint64_t batch_id);

    /// Load batch features given an already-deserialized GraphSample.
    /// Avoids re-reading + re-parsing the sample when the caller
    /// (e.g., BatchAssembler::assemble_from_sample) already has it in hand.
    ///
    /// out_used_v2 (optional): receives whether THIS call was served by the v2
    /// (addr_table) path. Unlike last_used_addr_tables(), this per-call outcome
    /// cannot be overwritten by a concurrent prefetch worker, so any
    /// correctness decision (e.g., BatchAssembler's self-contained placeholder
    /// safety net) must consume it instead of the shared flag.
    torch::Tensor load_batch_features(const GraphSample& sample,
                                      bool* out_used_v2 = nullptr);

    struct Stats {
        // Per-tier node-count counters (existing).
        std::atomic<uint64_t> l1_hits{0}, l2_hits{0};
        std::atomic<uint64_t> l3_reads{0}, l4_reads{0};
        std::atomic<uint64_t> total_requests{0};

        // Byte-level counters for disk-traffic accounting comparable to
        // published systems (cf. DiskGNN SIGMOD'25 Table 1, row
        // "Disk access volume (GB)").
        //
        // *_bytes_served — bytes copied from RAM/GPU caches into output.
        //                  Useful for hit-rate by data volume vs node count.
        // l3_bytes_wanted — feature payload bytes extracted from L3.
        // l3_bytes_disk   — physical bytes read from disk via O_DIRECT.
        //                   Equals l3_bytes_wanted only if every row was
        //                   block-aligned and adjacent; otherwise reflects
        //                   alignment overhead (a future alignment-reduction
        //                   pass would narrow this gap).
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

    GnnDtype dtype() const { return dtype_; }
    /// Device the model should train on for this store. Mirrors the v2 gather's
    /// device decision: CUDA only for a float32 store on a GPU host (the v2
    /// assembler path), else CPU. Keeps the packed-full TrainingLoop device-probe
    /// consistent with the 4-tier path across dtypes (packed-full stores are
    /// float32 in practice; this just avoids a CPU/GPU mismatch on other dtypes).
    torch::Device feature_device() const {
        return (torch::cuda::is_available() && dtype_ == GnnDtype::FLOAT32)
                   ? torch::Device(torch::kCUDA)
                   : torch::Device(torch::kCPU);
    }

    // Profile instrumentation getters. Each returns the
    // per-tier microseconds captured during the most recent
    // load_batch_features() call. last_rmap_us() is a SUB-counter and its time
    // is already included inside the L1/L2/L3/L4 totals; it is tracked
    // separately to quantify how much an address-table fast path saves.
    //
    // Public API is in microseconds (matches the
    // `_us` fields of BatchTiming). Internal storage uses nanoseconds for
    // precision; conversion happens here, once per batch (not per node).
    uint64_t last_l1_us() const   { return last_l1_ns_.load() / 1000; }
    uint64_t last_l2_us() const   { return last_l2_ns_.load() / 1000; }
    uint64_t last_l3_us() const   { return last_l3_ns_.load() / 1000; }
    uint64_t last_l4_us() const   { return last_l4_ns_.load() / 1000; }
    uint64_t last_rmap_us() const { return last_rmap_ns_.load() / 1000; }

    // v2 (addr_table) runtime path telemetry.
    // last_addr_load_us() — microseconds to open + parse the addr_table sidecar
    //   for the most recent batch. 0 if the v2 path was not taken.
    // last_used_addr_tables() — whether the most recent load_batch_features
    //   was served by the v2 (addr_table) path. TELEMETRY-ONLY / APPROXIMATE:
    //   the flag is shared across prefetch workers, so under N>1 it reflects
    //   whichever call finished last, not necessarily the caller's own. For a
    //   per-call answer, pass `out_used_v2` to load_batch_features instead.
    uint64_t last_addr_load_us() const { return last_addr_load_ns_.load() / 1000; }
    bool last_used_addr_tables() const { return last_used_v2_.load(); }

    // Multi-worker prefetch support: the AsyncBatchPrefetcher
    // can drive load_batch_features() from N concurrent worker threads
    // (prefetchNumWorkers>1). Every shared mutable resource on the hot path
    // that is NOT thread-safe is replicated per worker so there is no
    // cross-worker data race on feature content:
    //   - the DirectIoReader (its own 4 io_uring rings, "one ring per thread"
    //     per direct_io_reader.h) — worker 0 uses the primary l3_reader_;
    //     workers 1..N-1 use extra_workers_[w-1].l3_reader.
    //   - the pinned host staging buffer the assemble CUDA kernel reads — same
    //     primary/extra split. assemble() is host-blocking
    //     (cudaStreamSynchronize), so each worker's buffer is safe to reuse on
    //     its next batch.
    // Read-only state (gpu_cache_, cpu_cache_, l3_mmap_fb_) and the atomic
    // Stats counters are already concurrency-safe and remain shared.
    //
    // Call prepare_worker_io(n) ONCE before constructing an N-worker
    // prefetcher (it is idempotent + grow-only + single-threaded). Each worker
    // thread binds its id via bind_worker_id() at thread start; the hot path
    // reads current_worker_id() to select its private resources.
    void prepare_worker_io(unsigned num_workers);
    static void     bind_worker_id(unsigned id) noexcept;
    static unsigned current_worker_id() noexcept;

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

    // Persistent pinned host buffer reused across load_batch_features()
    // calls. Replaces per-batch cudaHostAlloc + cudaFreeHost (each is a
    // synchronous driver call ~100-500 us). At 1300+ batches per epoch the
    // old path burned ~150-600 ms/epoch on alloc churn alone.
    //
    // ensure_pinned_capacity(bytes) grows the buffer if needed (geometric
    // x1.5 plus 64-byte alignment headroom) and is thread-safe via
    // pinned_mutex_. The buffer outlives any single batch and is freed
    // exactly once in the destructor.
    mutable std::mutex pinned_mutex_;
    void*              pinned_ptr_      = nullptr;
    size_t             pinned_capacity_ = 0;

    // Per-worker IO resources for safe N>1 prefetch.
    // Worker id 0 uses the primary l3_reader_ + pinned_ptr_ above (so the
    // single-worker path is byte-identical to pre-change). Workers 1..N-1 use
    // extra_workers_[id-1]. prepare_worker_io(N) populates this vector ONCE,
    // single-threaded, before any concurrent load_batch_features call.
    struct WorkerIo {
        std::unique_ptr<DirectIoReader> l3_reader;        // own io_uring rings
        void*  pinned_ptr      = nullptr;                 // own pinned staging
        size_t pinned_capacity = 0;
    };
    std::vector<WorkerIo> extra_workers_;
    std::filesystem::path reord_fmat_path_;   // source for per-worker readers

    // Select the DirectIoReader for the calling thread (primary for id 0,
    // per-worker otherwise). May return nullptr if O_DIRECT was unavailable at
    // construction — callers then use the shared (read-only) l3_mmap_fb_.
    DirectIoReader* l3_reader_for_current_worker_();

    // Ensure the calling worker's pinned buffer is >= bytes; on success sets
    // `out` to that worker's pinned pointer. Worker 0 grows pinned_ptr_; other
    // workers grow their own extra_workers_[id-1].pinned_ptr. No lock: each
    // worker touches only its own slot, and extra_workers_ is never resized
    // once workers are running.
    bool ensure_pinned_capacity_for_worker_(size_t bytes, void*& out);

    // Profile instrumentation. Per-call sub-timers in
    // nanoseconds (high precision to avoid integer-microsecond truncation of
    // sub-μs hash lookups). Accessors below convert to μs at the API boundary.
    // rmap_lookup_ns is a SUB-counter: already included in the L3 total;
    // tracked separately to quantify the address-table fast-path candidate.
    // Made atomic so concurrent multi-worker prefetch threads do not data-race
    // on these telemetry counters. Values are only meaningful single-worker
    // (under N>1 every worker accumulates into the same counter, so the sum is
    // not a per-batch figure) and are NOT read on the prefetcher path; the
    // atomics exist purely to keep the writes well-defined under N>1.
    mutable std::atomic<uint64_t> last_l1_ns_{0};
    mutable std::atomic<uint64_t> last_l2_ns_{0};
    mutable std::atomic<uint64_t> last_l3_ns_{0};
    mutable std::atomic<uint64_t> last_l4_ns_{0};
    mutable std::atomic<uint64_t> last_rmap_ns_{0};

    // v2 (addr_table) path dispatch state.
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
    //                        Shared across workers (telemetry-only); the
    //                        authoritative per-call outcome is reported via
    //                        load_batch_features' out_used_v2 out-param.
    bool                  use_addr_tables_    = false;
    uint64_t              expected_meta_sha_head_ = 0;
    std::filesystem::path sample_dir_;
    mutable std::atomic<uint64_t> last_addr_load_ns_{0};
    mutable std::atomic<bool>     last_used_v2_{false};

    // Consolidated cold-feature read path: when `use_consolidated_slim_`,
    // load_batch_features_v2_ serves each batch's cold features with ONE pread
    // of [slim_offset, slim_length) (from the v2 addr_table header) from the
    // single consolidated.slim file, instead of opening the per-batch .bin.
    // This collapses ~1512 small file opens per epoch into one sequential read
    // per batch. The O_DIRECT fd is shared across prefetch workers (pread is
    // offset-based, so concurrent reads at distinct offsets are safe); each
    // read uses a private posix_memalign'd buffer. consolidated_buf_fd_ is the
    // buffered fallback when O_DIRECT is unavailable/fails. Opt-in (env
    // MDB_GNN_CONSOLIDATED_SLIM), validated against the perm/meta fingerprints
    // at ctor; any mismatch leaves use_consolidated_slim_ false → per-batch read.
    bool   use_consolidated_slim_ = false;
    int    consolidated_od_fd_    = -1;     // O_DIRECT fd (shared)
    int    consolidated_buf_fd_   = -1;     // buffered fallback fd (shared)
    size_t cons_block_align_      = 4096;

    /// Ensure the persistent pinned buffer is at least `bytes` long.
    /// Returns true on success, false if cudaHostAlloc failed (caller falls
    /// back to unpinned memory). No-op when GNN_CUDA_ENABLED is undefined.
    bool ensure_pinned_capacity(size_t bytes);

    /// Map GnnDtype to torch scalar type.
    static torch::ScalarType to_torch_dtype(GnnDtype dt);

    /// Legacy path: per-node hash classification across L1/L2/L4/L3, scatter
    /// via FeatureAssembler. Preserves the pre-addr-table behavior
    /// bit-identically.
    /// Public dispatcher falls through to this when the addr_table sidecar is
    /// missing, stale, or unreadable, or when the assembler gate is not met.
    torch::Tensor load_batch_features_legacy_(const GraphSample& sample);

    /// Per-tier buckets produced by the legacy classification loop: for each
    /// tier, the output row positions and the matching source indices
    /// (cache rows for L1/L2, reordered-FM rows for L3, slim slots for L4).
    /// Positions across all buckets partition [0, total) minus the
    /// zero-filled misses.
    struct LegacyTierBuckets {
        std::vector<uint32_t> l1_input_positions;  // positions in oids[] for L1 lookup
        std::vector<uint32_t> l1_indices;          // L1 cache row indices for the hits
        std::vector<uint32_t> l2_positions;        // output positions resolved from L2
        std::vector<uint32_t> l2_indices;          // L2 cache row indices for the hits
        std::vector<uint32_t> l3_positions;        // output positions resolved from L3
        std::vector<uint64_t> l3_row_indices;      // corresponding rows in reordered FM
        std::vector<uint32_t> l4_positions;        // output positions resolved from L4
        std::vector<uint32_t> l4_slim_indices;     // index into slim_data per L4 node
    };

    /// Legacy step 1: read this batch's packed_slim .bin (one bulk read,
    /// O_DIRECT opt-in) and parse it into an OID->slot map plus the raw
    /// feature payload. A missing/invalid/truncated file leaves both outputs
    /// empty (identical to the historical no-file branch). Accumulates
    /// last_l4_ns_ and stats_.l4_bytes_disk.
    void load_l4_slim_for_batch_(uint64_t batch_id,
                                 std::unordered_map<uint64_t, uint32_t>& slim_oid_to_idx,
                                 std::vector<char>& slim_data);

    /// Legacy step 2: classify every node of the batch into L1/L2/L4/L3
    /// buckets by hash lookup (L4 checked before L3 because slim data is
    /// exact). Unresolved nodes are counted as zero-filled L3 misses and
    /// warned about. Accumulates the per-tier lookup timers and hit stats.
    LegacyTierBuckets classify_batch_nodes_legacy_(
        const std::vector<ObjectId>& oids,
        uint64_t batch_id,
        size_t row_bytes,
        const std::unordered_map<uint64_t, uint32_t>& slim_oid_to_idx,
        const std::vector<char>& slim_data);

    /// Legacy step 3: batch-read the classified L3 rows from the reordered FM
    /// (per-worker O_DIRECT reader, mmap fallback) into one contiguous buffer
    /// ordered like l3_row_indices. Empty input yields an empty buffer.
    /// Accumulates last_l3_ns_ and stats_.l3_bytes_disk.
    std::vector<char> read_l3_rows_legacy_(
        const std::vector<uint64_t>& l3_row_indices,
        size_t row_bytes);

    /// Legacy step 4, GPU arm: gather L1 rows on-GPU, combine L2/L3/L4 rows
    /// into one pinned (or heap-fallback) host buffer, and scatter everything
    /// into the [total, D] output via FeatureAssembler. Preconditions: the
    /// assembler gate held (float32 dtype, GPU-resident L1 cache).
    torch::Tensor assemble_batch_gpu_legacy_(
        uint64_t total,
        const LegacyTierBuckets& buckets,
        const std::vector<char>& l3_buf,
        const std::vector<char>& slim_data);

    /// Legacy step 4, CPU arm (no GPU, or non-float32 dtype): memcpy each
    /// tier's rows straight into a zero-initialized CPU tensor. Unresolved
    /// positions stay zero.
    torch::Tensor assemble_batch_cpu_legacy_(
        uint64_t total,
        const LegacyTierBuckets& buckets,
        const std::vector<char>& l3_buf,
        const std::vector<char>& slim_data,
        size_t row_bytes);

    /// addr_table fast path: read addr_tables/batch_<bid>.addr,
    /// dispatch scatter-from-tier without per-node classification.
    /// Only called when assembler_ is active and gpu_cache_->is_on_gpu() —
    /// the CPU-only assembly case is served by legacy_ to avoid duplicating
    /// that path's memcpy logic. Throws AddrTableStaleException on meta_sha
    /// mismatch so the dispatcher can catch and fall back cleanly.
    torch::Tensor load_batch_features_v2_(const GraphSample& sample,
                                           const std::filesystem::path& addr_path);

    /// v2 step 1: read this batch's L4 cold-feature payload — ONE pread of
    /// [slim_offset, slim_length) from consolidated.slim when that path is
    /// active, otherwise the whole per-batch .bin (O_DIRECT opt-in). On
    /// return the payload lives in slim_owner at
    /// [slim_data_offset, slim_data_offset + slim_data_bytes). No-op when
    /// the addr table has no L4 rows (out-params must arrive empty/zero).
    /// Throws so the dispatcher falls back to legacy on any read failure.
    /// Accumulates last_l4_ns_ and stats_.l4_bytes_disk.
    void read_l4_slim_payload_v2_(const GraphSample& sample,
                                  const AddrTableReader::Result& addr,
                                  size_t row_bytes,
                                  std::vector<char>& slim_owner,
                                  size_t& slim_data_offset,
                                  size_t& slim_data_bytes);

    /// v2 step 2: batch-read the pre-resolved L3 rows (addr.l3_row_idxs)
    /// from the reordered FM — per-worker O_DIRECT reader, mmap fallback —
    /// into one contiguous buffer in l3_row_idxs order. Empty when the addr
    /// table has no L3 rows or no reader is available (the caller accounts
    /// that case). Accumulates last_l3_ns_ and stats_.l3_bytes_disk.
    std::vector<char> read_l3_rows_v2_(const AddrTableReader::Result& addr,
                                       size_t row_bytes);

    /// v2 step 3: GPU gather of the pre-resolved L1 cache rows. Bounds-checks
    /// every index against the GPU cache on the host (an out-of-range index
    /// from a stale-but-format-valid addr_table would fire a device assert
    /// and poison the CUDA context) and throws so the dispatcher falls back.
    /// Fills gpu_features ([num_l1, D] CUDA tensor) and gpu_positions;
    /// leaves both untouched when the addr table has no L1 rows.
    /// Accumulates last_l1_ns_ and the L1 hit stats.
    void gather_l1_rows_v2_(const AddrTableReader::Result& addr,
                            size_t row_bytes,
                            torch::Tensor& gpu_features,
                            std::vector<uint32_t>& gpu_positions);

    /// v2 steps 4+5: assemble the [total, D] output from the pre-gathered
    /// tiers — L2 rows by pre-resolved cache indices (kernel-direct when the
    /// fused assembler is enabled), L3 rows from l3_buf, L4 rows from the
    /// slim payload at [slim_data_offset, +slim_data_bytes) in slim_owner —
    /// written straight into the worker's pinned buffer (heap fallback) and
    /// scattered together with the L1 GPU rows via FeatureAssembler.
    /// Bounds-checks L2/L4 indices and throws so the dispatcher falls back
    /// to legacy; sets last_used_v2_ only after a successful assemble.
    torch::Tensor assemble_batch_gpu_v2_(
        const GraphSample&             sample,
        const AddrTableReader::Result& addr,
        const torch::Tensor&           gpu_features,
        const std::vector<uint32_t>&   gpu_positions,
        const std::vector<char>&       l3_buf,
        const std::vector<char>&       slim_owner,
        size_t                         slim_data_offset,
        size_t                         slim_data_bytes,
        uint64_t                       total,
        size_t                         row_bytes);
};

} // namespace mdb::gnn
