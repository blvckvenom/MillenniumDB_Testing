#include "gnn/storage/four_level_store.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/resource.h>  // getrlimit/setrlimit — raise NOFILE for Lever B

#ifdef GNN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include "gnn/common/page_cache_hint.h"
#include "gnn/common/pipeline_overlap.h"
#include "gnn/common/posix_io.h"
#include "gnn/core/feature_assembler.h"
#include "gnn/storage/addr_table.h"
#include "gnn/storage/addr_table_reader.h"
#include "gnn/storage/addr_table_writer.h"
#include "gnn/storage/block_store.h"
#include "gnn/training/graph_block_builder.h"
#include "gnn/storage/cache_file.h"
#include "gnn/storage/direct_io_reader.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/feature_matrix_header.h"
#include "gnn/storage/gnn_dtype.h"
#include "gnn/storage/packed_batch_store.h"
#include "gnn/storage/packed_full_store.h"
#include "gnn/storage/consolidated_slim_reader.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sample_fingerprint.h"  // STEP 8 content fingerprint
#include "gnn/sampling/sample_storage.h"
#include "graph_models/object_id.h"
// query_context.h must be included after liburing (via four_level_store.h ->
// direct_io_reader.h -> liburing.h -> linux/fs.h) which defines BLOCK_SIZE
// as a macro.  string_manager.h (included by query_context.h) uses the same
// name as a constexpr member, which causes a parse error unless we undef the
// macro first.
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif
#include "query/query_context.h"

namespace fs = std::filesystem;

namespace mdb::gnn {

// =============================================================================
// Phase 5 helpers (anonymous namespace — internal to this TU)
// =============================================================================
namespace {

// ---------------------------------------------------------------------------
// OidIdxAdapter — satisfies AddrTableWriter::build<CacheA,CacheB> contract.
//
// AddrTableWriter::build is templated on CacheA/CacheB and calls only
// find_index(oid) -> std::optional<uint32_t>. We need the mapping from
// an OID to its row index in the cache file. GpuCache/CpuCache::build()
// sorts entries by FeatureMatrix row before writing to disk, so the cache
// file's OID table (not the original l1_nodes/l2_nodes input order) is
// the ground truth for what find_index() returns. We read that table
// directly at Phase 5 build time rather than loading the full cache
// (which would allocate GPU memory and take seconds).
// ---------------------------------------------------------------------------
struct OidIdxAdapter {
    std::unordered_map<uint64_t, uint32_t> oid_to_idx;

    std::optional<uint32_t> find_index(ObjectId oid) const {
        auto it = oid_to_idx.find(oid.id);
        if (it == oid_to_idx.end()) return std::nullopt;
        return it->second;
    }
};

// Build an OidIdxAdapter by reading a GNNC cache file's OID table.
// Returns an empty adapter (zero entries) when the path does not exist.
OidIdxAdapter build_oid_idx_adapter(const fs::path& cache_path)
{
    OidIdxAdapter adapter;
    if (!fs::exists(cache_path)) return adapter;

    int fd = ::open(cache_path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "FourLevelStore Phase 5: cannot open cache file " +
            cache_path.string() + ": " + safe_strerror(errno));
    }
    struct FdCleanup { int fd; ~FdCleanup() { if (fd >= 0) ::close(fd); } } cleanup{fd};

    CacheFileHeader hdr{};
    read_all(fd, &hdr, sizeof(hdr), cache_path.string());
    if (!hdr.is_valid() || hdr.num_nodes == 0) return adapter;

    std::vector<uint64_t> oid_table(hdr.num_nodes);
    read_all(fd, oid_table.data(), hdr.num_nodes * sizeof(uint64_t), cache_path.string());

    adapter.oid_to_idx.reserve(hdr.num_nodes);
    for (uint32_t i = 0; i < static_cast<uint32_t>(hdr.num_nodes); ++i) {
        adapter.oid_to_idx[oid_table[i]] = i;
    }
    return adapter;
}

// Read the OID table from a packed_slim v2 file and return a map
// oid.id -> slot_index (position in file, 0-based). Used as the L4
// lookup map passed to AddrTableWriter::build.
// Returns an empty map for v1 files (no OID table) or missing files.
std::unordered_map<uint64_t, uint32_t> read_slim_oid_table(const fs::path& slim_path)
{
    std::unordered_map<uint64_t, uint32_t> result;
    if (!fs::exists(slim_path)) return result;

    int fd = ::open(slim_path.c_str(), O_RDONLY);
    if (fd < 0) return result;
    struct FdCleanup { int fd; ~FdCleanup() { if (fd >= 0) ::close(fd); } } cleanup{fd};

    PackedBatchHeader hdr{};
    ssize_t r = ::read(fd, &hdr, sizeof(hdr));
    if (r != static_cast<ssize_t>(sizeof(hdr)) || !hdr.is_valid()) return result;
    if (!hdr.has_oid_table() || hdr.num_nodes == 0) return result;

    std::vector<uint64_t> oid_table(hdr.num_nodes);
    read_all(fd, oid_table.data(), hdr.num_nodes * sizeof(uint64_t), slim_path.string());

    result.reserve(hdr.num_nodes);
    for (uint32_t i = 0; i < static_cast<uint32_t>(hdr.num_nodes); ++i) {
        result[oid_table[i]] = i;
    }
    return result;
}

// Compute meta_sha_head: first 8 bytes of gnn_meta.bin XOR file size.
// This is a fast content-derived identifier used to detect stale addr_table
// sidecars at runtime (AddrTableReader::open validates it when != 0).
// The same formula must be used both here (build) and in the runtime
// consumer. Convention: if the file does not exist or cannot be read,
// returns 0 (which disables staleness detection in AddrTableReader::open).
uint64_t compute_meta_sha_head(const fs::path& gnn_meta_path)
{
    // FNV-64 over the full file contents — detects any byte-level change
    // including same-size rewrites. The plan (Task 4 Step 3 helper notes)
    // explicitly allows FNV-64 in lieu of SHA-256 because the hash is only
    // used as a staleness marker, not for cryptographic integrity.
    if (!fs::exists(gnn_meta_path)) return 0;
    int fd = ::open(gnn_meta_path.c_str(), O_RDONLY);
    if (fd < 0) return 0;
    struct FdCleanup { int fd; ~FdCleanup() { if (fd >= 0) ::close(fd); } } c{fd};

    constexpr uint64_t FNV_PRIME  = 0x00000100000001B3ULL;
    constexpr uint64_t FNV_OFFSET = 0xCBF29CE484222325ULL;
    uint64_t hash = FNV_OFFSET;
    unsigned char buf[4096];
    ssize_t n;
    while ((n = ::read(fd, buf, sizeof(buf))) > 0) {
        for (ssize_t i = 0; i < n; ++i) {
            hash ^= static_cast<uint64_t>(buf[i]);
            hash *= FNV_PRIME;
        }
    }
    if (n < 0) return 0;  // read error — disable staleness check at runtime

    // 0 is the sentinel for "staleness check disabled" in AddrTableReader::open.
    // FNV-64 of a non-empty file basically never hashes to 0, but guard anyway.
    return hash == 0 ? 1 : hash;
}

// Nodes a load could not resolve in any tier are served as all-zero feature
// rows: training keeps running, but a stale/incomplete store degrades model
// quality invisibly. Make that state loud — one line per affected load,
// mirroring the dispatcher's per-batch "falling back to legacy" logging.
void warn_zero_filled(const std::string& context, uint64_t zero_filled,
                      uint64_t total)
{
    std::cerr << "[FourLevelStore] WARNING (" << context << "): "
              << zero_filled << "/" << total
              << " nodes unresolved in any tier — their features are"
                 " ZERO-FILLED. The feature store is likely stale for this"
                 " sample; re-run gnn_build_feature_store.\n";
}

// STEP 8: feature-store content fingerprint sidecar ("<feature>_store.fp").
// A 24-byte sibling record kept next to store.meta so StoreMetaHeader stays
// byte-identical (no format migration / version bump). Layout:
//   {uint32 magic 'GFFP', uint32 version 1, uint64 fingerprint, uint64 reserved}
constexpr uint32_t STORE_FP_MAGIC   = 0x47464650u; // 'GFFP'
constexpr uint32_t STORE_FP_VERSION = 1u;

struct StoreFpRecord {
    uint32_t magic;
    uint32_t version;
    uint64_t fingerprint;
    uint64_t reserved;
};
static_assert(sizeof(StoreFpRecord) == 24, "store.fp record must be 24 bytes");

// Returns the persisted fingerprint, or 0 (UNKNOWN) if the sidecar is absent,
// short, or has a bad magic/version — matching compute_meta_sha_head's
// "0 disables the staleness check" convention (the safe direction: recompute).
uint64_t read_store_fp(const fs::path& path)
{
    if (!fs::exists(path)) return 0;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return 0;
    struct FdClose { int fd; ~FdClose() { if (fd >= 0) ::close(fd); } } c{fd};
    StoreFpRecord rec{};
    ssize_t n = ::read(fd, &rec, sizeof(rec));
    if (n != static_cast<ssize_t>(sizeof(rec))) return 0;
    if (rec.magic != STORE_FP_MAGIC || rec.version != STORE_FP_VERSION) return 0;
    return rec.fingerprint;
}

// Atomically write the sidecar (open|write_all|fsync|fsync_dir), mirroring the
// store.meta write. Throws on failure so the build's cleanup catch removes it.
void write_store_fp(const fs::path& path, uint64_t fingerprint)
{
    StoreFpRecord rec{};
    rec.magic       = STORE_FP_MAGIC;
    rec.version     = STORE_FP_VERSION;
    rec.fingerprint = fingerprint;
    rec.reserved    = 0;
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "FourLevelStore: cannot create " + path.string() + ": " +
            safe_strerror(errno));
    }
    FdGuard guard(fd);
    write_all(fd, &rec, sizeof(rec), path.string());
    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "FourLevelStore: fsync failed on store.fp: " + safe_strerror(errno));
    }
    fsync_directory(path);
}

// Format an addr_table filename for a given batch_id.
// Mirrors the packed_slim naming: "batch_%06lu.addr".
fs::path addr_table_filename(const fs::path& addr_tables_dir, uint64_t batch_id)
{
    char fname[32];
    std::snprintf(fname, sizeof(fname), "batch_%06lu.addr",
                  static_cast<unsigned long>(batch_id));
    return addr_tables_dir / fname;
}

// Baked-block filename: use the shared mdb::gnn::block_filename free function
// (block_store.h) so the train-time consume in batch_assembler.cc derives the
// exact same name. Previously a local copy lived here.

// Packed-full build (additive): gather each batch's all_unique_nodes features
// from the source fmat into a contiguous [N_b, D] payload (in all_unique_nodes
// order) and append to <sample_dir>/packed_full/. One sequential gather pass;
// NO reorder/caches/tiering. Keyed by store_fp (the mixed feature-store fp).
// Single-threaded v1 (the gather is the cost; measure before parallelizing).
uint64_t build_packed_full_(SampleStorage& samples, const FeatureMatrix& fm,
                            const RowMapping& rmap, const fs::path& sample_dir,
                            uint64_t store_fp, uint64_t total_batches) {
    const uint64_t D         = fm.num_cols();
    const uint64_t row_bytes = fm.row_bytes();

    // Disk guard: estimate total packed_full size from batch 0 and refuse early
    // if it won't fit (suggest deleting the 4-tier to free space).
    // batch-0 size extrapolation assumes uniform fanout across batches.
    if (total_batches > 0) {
        auto s0 = samples.read_sample(0);
        uint64_t est_per_batch = static_cast<uint64_t>(s0.all_unique_nodes.size()) * row_bytes;
        uint64_t est_total     = est_per_batch * total_batches;
        std::error_code sec;
        auto space = fs::space(sample_dir, sec);
        if (!sec && space.available < est_total + est_total / 20) {  // +5% margin
            throw std::runtime_error(
                "build_packed_full_: insufficient disk for packed_full (need ~" +
                std::to_string(est_total / (1024ULL * 1024)) + " MiB, have " +
                std::to_string(space.available / (1024ULL * 1024)) + " MiB free in " +
                sample_dir.string() + "). Delete the 4-tier (reordered.fmat, "
                "packed_slim/, addr_tables/) to free space — keep store.meta + blocks/.");
        }
    }

    PackedFullWriter w(sample_dir / "packed_full", store_fp,
                       static_cast<uint32_t>(D), static_cast<uint32_t>(fm.dtype()), row_bytes);
    std::vector<char> buf;
    std::vector<uint64_t> rows;
    for (uint64_t b = 0; b < total_batches; ++b) {
        auto sample = samples.read_sample(b);
        const auto& uniq = sample.all_unique_nodes;
        rows.clear();
        rows.reserve(uniq.size());
        for (const auto& oid : uniq) {
            auto r = rmap.find(oid);
            if (!r) {
                throw std::runtime_error(
                    "build_packed_full_: node not in RowMapping: " + std::to_string(oid.id));
            }
            rows.push_back(*r);
        }
        buf.resize(rows.size() * row_bytes);
        fm.extract_rows(rows, buf.data());          // gather in all_unique_nodes order
        w.write_batch(b, buf.data(), rows.size());
    }
    w.finalize();
    std::error_code ec;
    auto sz = fs::file_size(sample_dir / "packed_full" / "packed_full.dat", ec);
    return ec ? 0 : static_cast<uint64_t>(sz);
}

// ---------------------------------------------------------------------------
// build_addr_tables_ — Phase 5 of FourLevelStore::build()
//
// For each batch in [0, total_batches):
//   1. Read sample.all_unique_nodes (classify input).
//   2. Read packed_slim OID table (L4 lookup map).
//   3. Classify all_unique_nodes into {L1,L2,L4,L3,zero}.
//   4. Atomically write batch_NNNNNN.addr to addr_tables/.
//
// Workers are dispatched via a SALIENT-style atomic counter. QueryContext
// is propagated from the primary thread so BPT lookups inside read_sample
// do not null-deref the thread_local context pointer.
// ---------------------------------------------------------------------------
void build_addr_tables_(
    SampleStorage&                              samples,
    const fs::path&                             addr_tables_dir,
    const fs::path&                             packed_slim_dir,
    const OidIdxAdapter&                        l1_adapter,
    const OidIdxAdapter&                        l2_adapter,
    const std::optional<RowMapping>&            reordered_rm_holder,
    uint64_t                                    meta_sha_head,
    uint64_t                                    total_batches,
    uint64_t&                                   out_addr_tables_bytes,
    bool                                        write_addr_tables,
    bool                                        bake_blocks,
    const fs::path&                             blocks_dir,
    uint64_t&                                   out_blocks_bytes,
    uint64_t                                    store_fp,
    const std::vector<uint64_t>*                cons_offsets = nullptr,
    const std::vector<uint64_t>*                cons_lengths = nullptr)
{
    if (write_addr_tables) fs::create_directories(addr_tables_dir);
    if (bake_blocks)       fs::create_directories(blocks_dir);

    unsigned num_workers = std::thread::hardware_concurrency();
    if (num_workers == 0) num_workers = 4;
    if (num_workers > 20) num_workers = 20;
    if (const char* env = std::getenv("MDB_GNN_ADDR_TABLE_WORKERS")) {
        try {
            int v = std::stoi(env);
            if (v > 0) num_workers = static_cast<unsigned>(v);
        } catch (...) {}
    }
    if (num_workers > std::thread::hardware_concurrency() &&
        std::thread::hardware_concurrency() > 0)
    {
        num_workers = std::thread::hardware_concurrency();
    }
    if (num_workers > total_batches && total_batches > 0)
        num_workers = static_cast<unsigned>(total_batches);
    if (num_workers == 0) num_workers = 1;

    // Build rmap_find closure from the reordered RowMapping.
    // Falls back to returning nullopt (L3 treated as zero) when absent.
    AddrTableWriter::RmapFind rmap_find;
    if (reordered_rm_holder.has_value()) {
        const RowMapping* rm_ptr = &*reordered_rm_holder;
        rmap_find = [rm_ptr](ObjectId oid) -> std::optional<uint64_t> {
            return rm_ptr->find(oid);
        };
        // Warm the RowMapping lazy index before workers spawn.
        if (reordered_rm_holder->size() > 0) {
            (void) rm_ptr->find(rm_ptr->get(0));
        }
    } else {
        rmap_find = [](ObjectId) -> std::optional<uint64_t> { return std::nullopt; };
    }

    std::atomic<uint64_t> next_bid{0};
    std::atomic<uint64_t> total_bytes_acc{0};
    std::atomic<uint64_t> blocks_bytes_acc{0};
    std::exception_ptr    first_exc;
    std::mutex            exc_mutex;

    // Capture the primary thread's QueryContext so workers can propagate it.
    QueryContext* primary_ctx = QueryContext::_query_ctx;

    auto worker_fn = [&]() {
        QueryContext::set_query_ctx(primary_ctx);

        try {
            while (true) {
                uint64_t b = next_bid.fetch_add(1, std::memory_order_relaxed);
                if (b >= total_batches) break;

                // 1. Read sample for the classify-input list.
                auto sample = samples.read_sample(b);

                if (write_addr_tables) {
                    // 2. Build L4 lookup map from packed_slim OID table.
                    char fname[32];
                    std::snprintf(fname, sizeof(fname), "batch_%06lu.bin",
                                  static_cast<unsigned long>(b));
                    auto slim_path = packed_slim_dir / fname;
                    auto slim_oid_to_idx = read_slim_oid_table(slim_path);

                    // 3. Classify each unique node into a tier.
                    AddrTableBuffers buf;
                    AddrTableWriter::build(
                        sample.all_unique_nodes,
                        &l1_adapter,
                        &l2_adapter,
                        slim_oid_to_idx,
                        rmap_find,
                        meta_sha_head,
                        buf);

                    // DiskGNN-adoption Plan 1: when a consolidated cold-feature
                    // file was written, upgrade this batch's header to v2 carrying
                    // its (slim_offset, slim_length) so the runtime can pread it
                    // directly.
                    if (cons_offsets && cons_lengths) {
                        buf.header = AddrTableHeader::make_v2(
                            buf.header.num_l1, buf.header.num_l2, buf.header.num_l3,
                            buf.header.num_l4, buf.header.num_zero, meta_sha_head,
                            (*cons_offsets)[b], (*cons_lengths)[b]);
                    }

                    // 4. Atomically write the addr_table sidecar.
                    auto addr_path = addr_table_filename(addr_tables_dir, b);
                    AddrTableWriter::write_atomic(addr_path, buf);

                    total_bytes_acc.fetch_add(buf.total_bytes(),
                                              std::memory_order_relaxed);
                }

                // Task 6: bake the per-batch computation-graph block. Additive
                // and independent of addr_tables. Idempotent via content hash:
                // skip when a fresh block with a matching sample_fp exists.
                // Thread-safe: each batch writes its own file via the atomic
                // tmp+rename BlockWriter, the byte counter is atomic, and
                // graph_block::build_* are pure functions on the local sample.
                if (bake_blocks) {
                    uint64_t fp = mdb::gnn::compute_batch_content_hash(sample);
                    auto blk_path = block_filename(blocks_dir, b);
                    // Cheap header-only freshness peek (no body read) for the
                    // idempotent re-bake skip; train-time open() still validates
                    // the full block.
                    if (!BlockReader::is_fresh(blk_path, fp)) {
                        // Build oid_to_global identically to BatchAssembler.
                        std::unordered_map<uint64_t, int64_t> oid_to_global;
                        oid_to_global.reserve(sample.all_unique_nodes.size());
                        for (int64_t i = 0;
                             i < static_cast<int64_t>(sample.all_unique_nodes.size());
                             ++i) {
                            oid_to_global[sample.all_unique_nodes[i].id] = i;
                        }
                        auto active = mdb::gnn::graph_block::build_active_indices(
                            sample, oid_to_global);
                        auto edges = mdb::gnn::graph_block::build_edge_indices(
                            sample, active);
                        // SC-2: stamp the v2 self-contained fields so a later
                        // train run (SC-3) can skip batches.dat. store_fp is the
                        // store-level staleness key; the rest come from the full
                        // sample (seeds = nodes_per_layer[0] for label gather).
                        const uint64_t num_unique_nodes =
                            sample.all_unique_nodes.size();
                        std::vector<uint64_t> seed_ids;
                        if (!sample.nodes_per_layer.empty()) {
                            const auto& seeds = sample.nodes_per_layer[0];
                            seed_ids.reserve(seeds.size());
                            for (const auto& oid : seeds)
                                seed_ids.push_back(oid.id);
                        }
                        const uint32_t split =
                            static_cast<uint32_t>(sample.split);
                        BlockWriter::write(blk_path, fp, b,
                                           active.sizes_per_layer, edges,
                                           store_fp, num_unique_nodes,
                                           seed_ids, split);
                        std::error_code sz_ec;
                        auto bsz = fs::file_size(blk_path, sz_ec);
                        if (!sz_ec)
                            blocks_bytes_acc.fetch_add(bsz,
                                                       std::memory_order_relaxed);
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(exc_mutex);
            if (!first_exc) first_exc = std::current_exception();
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for (unsigned i = 0; i < num_workers; ++i) {
        workers.emplace_back(worker_fn);
    }
    for (auto& t : workers) t.join();
    if (first_exc) std::rethrow_exception(first_exc);

    if (write_addr_tables) fsync_directory(addr_tables_dir);
    if (bake_blocks)       fsync_directory(blocks_dir);
    out_addr_tables_bytes = total_bytes_acc.load(std::memory_order_relaxed);
    out_blocks_bytes      = blocks_bytes_acc.load(std::memory_order_relaxed);
}

} // anonymous namespace

// =============================================================================
// Static: resolve the per-projection gnn_meta.bin path (addr-table staleness
// marker). gnn_meta.bin is written by graph_project into the PROJECTION dir,
// not the db root — the historical db-root path never existed, so
// compute_meta_sha_head() always returned 0 and the staleness check was a
// silent no-op. Public + static so the contract is unit-testable.
// =============================================================================
fs::path FourLevelStore::gnn_meta_path_for(const fs::path&    db_folder,
                                           const std::string& projection_name)
{
    return db_folder / "projections" / projection_name / "gnn_meta.bin";
}

bool FourLevelStore::store_matches_sample_fp(
    const fs::path&    db_folder,
    const std::string& feature_name,
    uint64_t           sample_content_fp)
{
    auto gnn_dir       = db_folder / "gnn_features";
    auto meta_path     = gnn_dir / (feature_name + "_store.meta");
    auto store_fp_path = gnn_dir / (feature_name + "_store.fp");
    if (!fs::exists(meta_path)) return false;

    // Read store.meta for the feature dim + dtype that mix into the key.
    StoreMetaHeader meta{};
    {
        int fd = ::open(meta_path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        FdGuard guard(fd);
        try {
            read_all(fd, &meta, sizeof(meta), meta_path.string());
        } catch (...) {
            return false;
        }
    }
    if (!meta.is_valid()) return false;

    uint64_t want = mix_feature_store_fingerprint(
        sample_content_fp, feature_name, meta.feature_dim, meta.dtype);
    uint64_t have = read_store_fp(store_fp_path);
    return want != 0 && have == want;
}

// =============================================================================
// Helper: Map GnnDtype -> torch::ScalarType
// =============================================================================

torch::ScalarType FourLevelStore::to_torch_dtype(GnnDtype dt) {
    switch (dt) {
        case GnnDtype::FLOAT32: return torch::kFloat32;
        case GnnDtype::FLOAT64: return torch::kFloat64;
        case GnnDtype::INT32:   return torch::kInt32;
        case GnnDtype::INT64:   return torch::kInt64;
        case GnnDtype::UINT8:   return torch::kUInt8;
        case GnnDtype::BOOL:    return torch::kBool;
    }
    throw std::invalid_argument(
        "FourLevelStore: unknown GnnDtype " + std::to_string(static_cast<int>(dt)));
}

// =============================================================================
// Persistent pinned-host buffer pool (Round 1A, 2026-05-15)
// =============================================================================
//
// load_batch_features() previously did cudaHostAlloc + cudaFreeHost per
// batch. Each call is a synchronous driver entry (~100-500 us); on
// papers100M training (1300+ batches/epoch) that burned 150-600 ms/epoch
// purely on alloc churn. The fix is to keep one persistent pinned buffer
// that grows monotonically and is released once in the destructor.
//
// Growth policy: when the request exceeds capacity we allocate
// max(requested, 1.5 x current) so consecutive growths amortize like
// std::vector. Ownership invariant: each worker slot is single-owner —
// exactly one thread is bound to a given worker id while loads run, and the
// store must outlive every thread calling load_batch_features. pinned_mutex_
// guards only the shared slot-0 fields (pinned_ptr_/pinned_capacity_) so a
// grow can never race the destructor's teardown free.
// =============================================================================

// Round 3B-mw (2026-06-01): per-thread worker id. Default 0 = primary
// (main thread / single-worker / non-prefetch callers). AsyncBatchPrefetcher
// binds 0..N-1 at each worker thread's start via bind_worker_id().
namespace { thread_local unsigned t_gnn_worker_id = 0; }

void FourLevelStore::bind_worker_id(unsigned id) noexcept { t_gnn_worker_id = id; }
unsigned FourLevelStore::current_worker_id() noexcept { return t_gnn_worker_id; }

DirectIoReader* FourLevelStore::l3_reader_for_current_worker_() {
    const unsigned w = t_gnn_worker_id;
    if (w == 0 || w > extra_workers_.size()) {
        return l3_reader_.get();              // primary (also the clamp fallback)
    }
    return extra_workers_[w - 1].l3_reader.get();  // may be null -> mmap fallback
}

void FourLevelStore::prepare_worker_io(unsigned num_workers) {
    // Idempotent + grow-only + single-threaded: MUST be called before any
    // concurrent load_batch_features (i.e. before an N>1 prefetcher exists).
    if (num_workers <= 1) return;
    const unsigned want_extra = num_workers - 1;
    while (extra_workers_.size() < want_extra) {
        WorkerIo io;
        // Only the O_DIRECT reader is per-worker-unsafe and thus replicated.
        // If the primary fell back to the shared mmap (l3_reader_ == null),
        // workers share l3_mmap_fb_ (read-only, concurrency-safe) and need no
        // private reader.
        if (l3_reader_ && !reord_fmat_path_.empty()) {
            // The primary uses O_DIRECT, so l3_mmap_fb_ is NOT populated; a
            // worker whose reader failed to open would silently read zeros for
            // L3 rows (corrupt features). Fail loud instead — the caller
            // (TrainingLoop) catches and falls back to a single worker.
            try {
                io.l3_reader = std::make_unique<DirectIoReader>(reord_fmat_path_);
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    std::string("FourLevelStore::prepare_worker_io: cannot open "
                                "per-worker O_DIRECT reader for ")
                    + reord_fmat_path_.string() + " (refusing to enable "
                      "multi-worker with a worker that would read zeros): "
                    + e.what());
            }
        }
        extra_workers_.push_back(std::move(io));
    }
}

bool FourLevelStore::ensure_pinned_capacity_for_worker_(size_t bytes, void*& out) {
#ifdef GNN_CUDA_ENABLED
    if (bytes == 0) { out = nullptr; return true; }
    const unsigned w = t_gnn_worker_id;
    // Slot 0 is shared with the destructor's teardown free, so guard it with
    // pinned_mutex_. Slots 1..N are single-owner (each prefetch worker binds
    // a unique id and extra_workers_ is sized before any worker runs) and
    // are freed only after the prefetcher joins — no lock needed.
    std::unique_lock<std::mutex> lock;
    void**  pp;
    size_t* pc;
    if (w == 0 || w > extra_workers_.size()) {
        lock = std::unique_lock<std::mutex>(pinned_mutex_);
        pp = &pinned_ptr_;
        pc = &pinned_capacity_;
    } else {
        pp = &extra_workers_[w - 1].pinned_ptr;
        pc = &extra_workers_[w - 1].pinned_capacity;
    }
    if (bytes <= *pc) { out = *pp; return true; }
    size_t new_cap = bytes;
    const size_t grown = *pc + (*pc >> 1);
    if (grown > new_cap) new_cap = grown;
    void* new_ptr = nullptr;
    if (cudaHostAlloc(&new_ptr, new_cap, cudaHostAllocDefault) != cudaSuccess
        || new_ptr == nullptr) {
        return false;  // caller falls back to unpinned memory
    }
    if (*pp != nullptr) cudaFreeHost(*pp);
    *pp = new_ptr;
    *pc = new_cap;
    out = new_ptr;
    return true;
#else
    (void)bytes; out = nullptr; return false;
#endif
}

FourLevelStore::~FourLevelStore() {
#ifdef GNN_CUDA_ENABLED
    std::lock_guard<std::mutex> lock(pinned_mutex_);
    if (pinned_ptr_ != nullptr) {
        cudaFreeHost(pinned_ptr_);
        pinned_ptr_      = nullptr;
        pinned_capacity_ = 0;
    }
    // Round 3B-mw: free per-worker pinned staging buffers.
    for (auto& io : extra_workers_) {
        if (io.pinned_ptr != nullptr) {
            cudaFreeHost(io.pinned_ptr);
            io.pinned_ptr      = nullptr;
            io.pinned_capacity = 0;
        }
    }
#endif
    // DiskGNN-adoption Plan 1 Phase 2: close the consolidated.slim fds.
    if (consolidated_od_fd_  >= 0) { ::close(consolidated_od_fd_);  consolidated_od_fd_  = -1; }
    if (consolidated_buf_fd_ >= 0) { ::close(consolidated_buf_fd_); consolidated_buf_fd_ = -1; }
}

// =============================================================================
// Path 4 (2026-05-20): rebuild addr_tables on a loaded runtime instance
// =============================================================================
// Used when source FeatureMatrix is unavailable (e.g. placeholder / deleted)
// but the rest of the feature store (L1/L2/L3/L4 + caches + reordered_rm) is
// intact. Re-runs Phase 5 against the already-loaded caches and updates this
// instance's v2-dispatch state so subsequent load_batch_features() calls can
// immediately use the v2 fast path.
uint64_t FourLevelStore::rebuild_addr_tables(const fs::path& db_folder,
                                             bool bake_blocks,
                                             uint64_t* out_blocks_bytes) {
    if (!samples_) {
        throw std::runtime_error(
            "FourLevelStore::rebuild_addr_tables: no SampleStorage bound");
    }

    const auto& catalog = samples_->get_catalog();
    uint64_t total_batches = catalog.total_batches;

    uint64_t meta_sha_head = compute_meta_sha_head(
        gnn_meta_path_for(db_folder, catalog.projection_name));

    // Reconstruct OidIdxAdapters from the already-written GNNC cache files
    // (same approach as FourLevelStore::build's Phase 5 step at line 880+).
    auto gnn_dir       = db_folder / "gnn_features";
    auto gpu_cache_path = gnn_dir / "node_features_gpu_cache.bin";
    auto cpu_cache_path = gnn_dir / "node_features_cpu_cache.bin";
    // Note: feature_name is fixed to "node_features" here — single-feature
    // pipeline as of 2026-05-20. Generalize if multi-feature lands.
    auto l1_adapter = build_oid_idx_adapter(gpu_cache_path);
    auto l2_adapter = build_oid_idx_adapter(cpu_cache_path);

    // Match the path convention used by FourLevelStore::build() at the Phase 5
    // wiring (line 460): addr_tables_dir = sample_dir / "addr_tables".
    auto addr_tables_dir = sample_dir_ / "addr_tables";
    auto blocks_dir      = sample_dir_ / "blocks";

    uint64_t out_bytes    = 0;
    uint64_t blocks_bytes = 0;
    build_addr_tables_(
        *samples_,
        addr_tables_dir,
        fs::path(packed_slim_dir_),
        l1_adapter,
        l2_adapter,
        reordered_rm_,
        meta_sha_head,
        total_batches,
        out_bytes,
        /*write_addr_tables=*/true,
        bake_blocks,
        blocks_dir,
        blocks_bytes,
        /*store_fp=*/catalog.sample_content_fp);

    if (out_blocks_bytes) *out_blocks_bytes = blocks_bytes;

    // After Phase 5 completes, this instance's v2 dispatch can serve
    // load_batch_features() immediately. (The runtime ctor would have
    // disabled use_addr_tables_ if addr_tables/ was absent at construction
    // time — flip it on now that we've created the sidecars.)
    use_addr_tables_ = true;
    expected_meta_sha_head_ = meta_sha_head;

    return out_bytes;
}

// =============================================================================
// build() — The Core Pipeline
// =============================================================================

FourLevelStore::BuildResult FourLevelStore::build(
    const FeatureMatrix&         features,
    const RowMapping&            row_mapping,
    SampleStorage&               samples,
    const Config&                config,
    const fs::path&              db_folder,
    const std::string&           feature_name)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    BuildResult result;

    auto gnn_dir         = db_folder / "gnn_features";
    auto gpu_cache_path  = gnn_dir / (feature_name + "_gpu_cache.bin");
    auto cpu_cache_path  = gnn_dir / (feature_name + "_cpu_cache.bin");
    auto meta_path       = gnn_dir / (feature_name + "_store.meta");
    auto reordered_fmat  = gnn_dir / (feature_name + "_reordered.fmat");
    auto reordered_rmap  = gnn_dir / (feature_name + "_reordered.rmap");

    const auto& catalog  = samples.get_catalog();
    auto sample_dir      = SampleStorage::get_storage_path(
                               db_folder.string(), catalog.sample_name);
    auto packed_slim_dir  = fs::path(sample_dir) / "packed_slim";
    auto addr_tables_dir  = fs::path(sample_dir) / "addr_tables";
    auto store_fp_path    = gnn_dir / (feature_name + "_store.fp");
    result.packed_slim_dir = packed_slim_dir.string();

    // STEP 8: the feature store's reuse key = the sample's persisted content
    // fingerprint (catalog) mixed with this store's identity (name + dim +
    // dtype). 0 = UNKNOWN (legacy/pre-STEP8 sample) → always recompute.
    const uint64_t cur_fp = mix_feature_store_fingerprint(
        catalog.sample_content_fp, feature_name,
        features.num_cols(), static_cast<uint8_t>(features.dtype()));

    // Packed-full build mode (additive, PS-class): a single gather pass over the
    // source fmat into <sample_dir>/packed_full/, keyed by cur_fp. Writes ONLY
    // packed_full/; never builds/deletes the 4-tier or blocks/. Requires
    // store.meta + blocks/ to already exist (a prior bakeBlocks build). The
    // train-time consumer (BatchAssembler packed-full mode) prefers this pack.
    if (config.pack_full) {
        if (cur_fp == 0) {
            throw std::runtime_error(
                "packFullFeatures: sample has no content fingerprint (legacy/pre-STEP8 sample); "
                "the packed-full pack would never be adopted at train. Re-run gnn_offline_sample "
                "so the sample carries a content fingerprint, then bakeBlocks, then packFullFeatures.");
        }
        std::cerr << "[FourLevelStore] packed-full build start (additive)\n";
        result.packed_full_bytes = build_packed_full_(
            samples, features, row_mapping, fs::path(sample_dir),
            cur_fp, catalog.total_batches);
        std::cerr << "[FourLevelStore] packed-full build done ("
                  << (result.packed_full_bytes / (1024ULL * 1024)) << " MiB)\n";
        auto pf_end = std::chrono::high_resolution_clock::now();
        result.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            pf_end - total_start).count();
        return result;
    }

    // --- Force cleanup ---
    // Fix #15: granular flags let callers preserve specific outputs.
    // Pre-Fix-#15 callers (force=true only) get unchanged behaviour
    // because the new flags default to true.
    if (config.force) {
        std::error_code ec;
        if (config.force_packed_slim) fs::remove_all(packed_slim_dir, ec);
        if (config.force_packed_slim) fs::remove_all(addr_tables_dir, ec);
        // Task 6: a force rebuild also clears stale baked blocks so they are
        // rebaked fresh against the new sample content.
        if (config.force_packed_slim) fs::remove_all(fs::path(sample_dir) / "blocks", ec);
        if (config.force_caches)      fs::remove(gpu_cache_path, ec);
        if (config.force_caches)      fs::remove(cpu_cache_path, ec);
        if (config.force_meta)        fs::remove(meta_path, ec);
        if (config.force_meta)        fs::remove(store_fp_path, ec);  // STEP 8 sidecar
        if (config.force_reorder)     fs::remove(reordered_fmat, ec);
        if (config.force_reorder)     fs::remove(reordered_rmap, ec);
    }

    // --- Pre-condition checks (STEP 8 reuse-or-recompute gate) ---
    // Replaces the legacy unconditional "already exists" throw: when a store is
    // present and we are NOT forcing, compare its content fingerprint to cur_fp.
    // Match → reuse (skip the whole rebuild). Mismatch/UNKNOWN → delete the
    // stale artifacts and fall through to a full recompute (never silent reuse).
    if (fs::exists(meta_path)) {
        if (config.force) {
            // force + force_meta=false intentionally preserved meta — keep the
            // historical behaviour for this contradictory combination.
            throw std::runtime_error(
                "Feature store already exists at: " + meta_path.string() + "\n"
                "force:1 with force_meta:false preserves store.meta, which "
                "conflicts with a full rebuild. Pass force_meta:true (the "
                "default) to overwrite, or drop force to reuse-or-recompute "
                "by fingerprint.");
        }

        uint64_t prior_fp = read_store_fp(store_fp_path);
        if (prior_fp != 0 && prior_fp == cur_fp) {
            std::cerr << "[FourLevelStore] feature store fingerprint matches (fp="
                      << prior_fp << ") — reusing existing artifacts\n" << std::flush;

            StoreMetaHeader meta{};
            bool meta_ok = false;
            int fd = ::open(meta_path.c_str(), O_RDONLY);
            if (fd >= 0) {
                FdGuard guard(fd);
                read_all(fd, &meta, sizeof(meta), meta_path.string());
                meta_ok = meta.is_valid();
            }
            // Reuse is only valid while the artifacts store.meta promises are
            // still on disk. A fingerprint match with missing tier files
            // (partial cleanup, packed-full surgery) must recompute — a
            // silently reused broken store would zero-fill features at train.
            bool artifacts_ok = false;
            if (meta_ok) {
                std::error_code ae;
                artifacts_ok = true;
                if (meta.l1_count > 0 && !fs::exists(gpu_cache_path, ae)) {
                    artifacts_ok = false;
                }
                if (meta.l2_count > 0 && !fs::exists(cpu_cache_path, ae)) {
                    artifacts_ok = false;
                }
                if ((meta.l3_count + meta.l4_count) > 0 &&
                    !fs::exists(packed_slim_dir, ae)) {
                    artifacts_ok = false;
                }
                // reordered.fmat/.rmap are written together; exactly one
                // present means a partially-deleted artifact.
                if (fs::exists(reordered_fmat, ae) != fs::exists(reordered_rmap, ae)) {
                    artifacts_ok = false;
                }
            }
            if (meta_ok && artifacts_ok) {
                auto fsz = [](const fs::path& p) -> uint64_t {
                    std::error_code e; auto s = fs::file_size(p, e);
                    return e ? 0 : static_cast<uint64_t>(s);
                };
                result.l1_nodes       = meta.l1_count;
                result.l2_nodes       = meta.l2_count;
                result.l3_nodes       = meta.l3_count;
                result.l4_nodes       = meta.l4_count;
                result.gpu_available  = meta.gpu_available != 0;
                result.packed_slim_dir = meta.get_packed_slim_dir();
                result.total_batches  = catalog.total_batches;
                result.gpu_cache_bytes = fsz(gpu_cache_path);
                result.cpu_cache_bytes = fsz(cpu_cache_path);
                result.reordered_bytes = config.reorder ? fsz(reordered_fmat) : 0;
                uint64_t slim = 0;
                std::error_code de;
                if (fs::exists(packed_slim_dir, de)) {
                    for (const auto& ent :
                         fs::directory_iterator(packed_slim_dir, de)) {
                        std::error_code se;
                        auto s = fs::file_size(ent.path(), se);
                        if (!se) slim += static_cast<uint64_t>(s);
                    }
                }
                result.slim_bytes = slim;
                result.total_disk_bytes = result.slim_bytes + result.gpu_cache_bytes
                                        + result.cpu_cache_bytes + result.reordered_bytes;
                result.addr_tables_built_ok = fs::exists(addr_tables_dir, de);

                // An explicitly requested Phase 5 must not be dropped by the
                // reuse fast path: bake_blocks (and a missing addr_tables/)
                // can be satisfied against the reused tier artifacts without
                // any rebuild. Baking is idempotent per block via
                // BlockReader::is_fresh, so re-requesting it on a fully baked
                // store only re-reads the samples.
                const bool need_addr = config.build_addr_tables
                                       && !result.addr_tables_built_ok;
                if (need_addr || config.bake_blocks) {
                    try {
                        auto l1_adapter = build_oid_idx_adapter(gpu_cache_path);
                        auto l2_adapter = build_oid_idx_adapter(cpu_cache_path);
                        auto meta_sha_head = compute_meta_sha_head(
                            gnn_meta_path_for(db_folder, catalog.projection_name));
                        std::optional<RowMapping> reordered_rm_reuse;
                        if (fs::exists(reordered_rmap)) {
                            reordered_rm_reuse.emplace(RowMapping::open(reordered_rmap));
                        }
                        auto blocks_dir = fs::path(sample_dir) / "blocks";
                        build_addr_tables_(
                            samples,
                            addr_tables_dir,
                            packed_slim_dir,
                            l1_adapter,
                            l2_adapter,
                            reordered_rm_reuse,
                            meta_sha_head,
                            catalog.total_batches,
                            result.addr_tables_bytes,
                            /*write_addr_tables=*/need_addr,
                            /*bake_blocks=*/config.bake_blocks,
                            blocks_dir,
                            result.blocks_bytes,
                            /*store_fp=*/catalog.sample_content_fp);
                        if (need_addr) result.addr_tables_built_ok = true;
                        if (config.bake_blocks) result.blocks_built_ok = true;
                    } catch (const std::exception& ex) {
                        std::cerr << "FourLevelStore::build: WARNING Phase 5 "
                                     "addr_tables failed on the reuse path: "
                                  << ex.what()
                                  << " — runtime will use per-batch lookup fallback.\n";
                    }
                }

                auto now = std::chrono::high_resolution_clock::now();
                result.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                           now - total_start).count();
                return result;  // full reuse — no rebuild
            }
            std::cerr << (meta_ok
                          ? "[FourLevelStore] fingerprint matches but tier"
                            " artifacts are missing — recomputing\n"
                          : "[FourLevelStore] existing store.meta unreadable/invalid"
                            " — recomputing\n")
                      << std::flush;
        } else {
            std::cerr << "[FourLevelStore] feature store fingerprint mismatch (prior="
                      << prior_fp << " cur=" << cur_fp
                      << ") — recomputing stale artifacts\n" << std::flush;
        }

        // Recompute: delete the same set force would, plus store.fp, then fall
        // through to the normal full rebuild below.
        std::error_code ec;
        fs::remove_all(packed_slim_dir, ec);
        fs::remove_all(addr_tables_dir, ec);
        fs::remove_all(fs::path(sample_dir) / "blocks", ec);  // Task 6: stale blocks
        fs::remove(gpu_cache_path, ec);
        fs::remove(cpu_cache_path, ec);
        fs::remove(meta_path, ec);
        fs::remove(reordered_fmat, ec);
        fs::remove(reordered_rmap, ec);
        fs::remove(store_fp_path, ec);
    }

    auto log_phase = [&total_start](const std::string& tag) {
        auto now = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(now - total_start).count();
        std::cerr << "[FourLevelStore] T+" << std::fixed << std::setprecision(1)
                  << t << "s " << tag << "\n" << std::flush;
    };
    log_phase("build() start");

    // --- Step 1: Read frequencies ---
    std::vector<uint64_t> freq_dense;
    try {
        freq_dense = samples.get_dense_frequencies(row_mapping);
    } catch (...) {
        // Fallback: convert from hash map
        auto freq_map = samples.get_node_frequencies();
        uint64_t N = row_mapping.size();
        freq_dense.assign(N, 0);
        for (const auto& [oid_id, count] : freq_map) {
            auto row = row_mapping.find(ObjectId(oid_id));
            if (row.has_value() && *row < N) {
                freq_dense[*row] = count;
            }
        }
    }

    // --- Step 2: Classify nodes ---
    uint64_t N = features.num_rows();
    uint64_t D = features.num_cols();
    GnnDtype dt = features.dtype();
    size_t elem = dtype_size(dt);
    size_t row_bytes = D * elem;

    // Sort by frequency descending
    struct NodeFreq { uint64_t row; uint64_t freq; };
    std::vector<NodeFreq> sorted_nodes;
    sorted_nodes.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        if (i < freq_dense.size() && freq_dense[i] > 0) {
            sorted_nodes.push_back({i, freq_dense[i]});
        }
    }
    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
              [](const auto& a, const auto& b) { return a.freq > b.freq; });

    // Determine cache capacities
    bool gpu_available = torch::cuda::is_available();
    size_t gpu_budget = gpu_available ? config.gpu.budget_bytes : 0;
    size_t cpu_budget = config.cpu.budget_bytes;

    uint64_t K1 = 0;
    uint64_t K2 = 0;
    if (row_bytes > 0) {
        K1 = std::min(static_cast<uint64_t>(gpu_budget / row_bytes),
                      static_cast<uint64_t>(sorted_nodes.size()));
        K2 = std::min(static_cast<uint64_t>(cpu_budget / row_bytes),
                      static_cast<uint64_t>(sorted_nodes.size()) - K1);
    }

    // If no GPU, redistribute L1 nodes to L2
    if (!gpu_available && K1 > 0) {
        K2 += K1;
        K1 = 0;
    }

    // Build node sets for cache building and for L4 filtering
    std::vector<ObjectId> l1_nodes, l2_nodes;
    std::unordered_set<uint64_t> cached_oid_set;  // L1 + L2 OIDs

    for (uint64_t i = 0; i < sorted_nodes.size(); ++i) {
        auto oid = row_mapping.get(sorted_nodes[i].row);
        if (i < K1) {
            l1_nodes.push_back(oid);
            cached_oid_set.insert(oid.id);
        } else if (i < K1 + K2) {
            l2_nodes.push_back(oid);
            cached_oid_set.insert(oid.id);
        } else if (sorted_nodes[i].freq > 1) {
            result.l3_nodes++;
        } else {
            result.l4_nodes++;
        }
    }

    // Nodes with freq == 0 (never accessed) are also L4
    if (freq_dense.size() < N) {
        result.l4_nodes += (N - freq_dense.size());
    } else {
        for (uint64_t i = 0; i < N; ++i) {
            if (freq_dense[i] == 0) {
                result.l4_nodes++;
            }
        }
    }

    result.l1_nodes     = K1;
    result.l2_nodes     = l2_nodes.size();
    result.gpu_available = gpu_available;
    result.total_batches = catalog.total_batches;

    // Track which artifacts THIS invocation (re)writes so the failure-cleanup
    // catch below removes only this run's partial outputs. Artifacts reused
    // fresh (e.g. a fingerprint-matched reordered.fmat) survive a later-phase
    // failure — their embedded fingerprints make the retry's reuse-or-recompute
    // decision safe.
    bool wrote_caches  = false;
    bool wrote_reorder = false;
    bool wrote_slim    = false;
    bool wrote_addr    = false;
    bool wrote_meta    = false;

    // --- Cleanup guard for partial outputs ---
    try {

    // --- Step 3: Build cache files ---
    fs::create_directories(gnn_dir);
    wrote_caches = true;
    log_phase("L1 GpuCache::build start (" + std::to_string(l1_nodes.size()) + " nodes)");
    GpuCache::build(l1_nodes, features, row_mapping, gpu_cache_path);
    log_phase("L1 GpuCache::build done");
    log_phase("L2 CpuCache::build start (" + std::to_string(l2_nodes.size()) + " nodes)");
    CpuCache::build(l2_nodes, features, row_mapping, cpu_cache_path);
    log_phase("L2 CpuCache::build done");

    // --- Step 4: Ensure L3 reordered FM exists ---
    const FeatureMatrix* active_fm = &features;
    std::optional<FeatureMatrix> reordered_holder;
    const RowMapping* active_rm = &row_mapping;
    std::optional<RowMapping> reordered_rm_holder;

    // STEP 8: reordered.fmat is "fresh" only if it exists AND its embedded
    // fingerprint matches cur_fp (with cur_fp known). A reordered.fmat left by a
    // different sample (e.g. force + force_reorder=false) is detected here and
    // rebuilt instead of being opened with the wrong shape.
    bool reordered_fresh = false;
    if (config.reorder && cur_fp != 0 &&
        fs::exists(reordered_fmat) && fs::exists(reordered_rmap)) {
        try {
            reordered_fresh =
                (FeatureMatrix::open(reordered_fmat).fingerprint() == cur_fp);
        } catch (...) {
            reordered_fresh = false;
        }
    }
    if (config.reorder && !reordered_fresh) {
        wrote_reorder = true;
        if (fs::exists(reordered_fmat)) {
            std::error_code rec;
            fs::remove(reordered_fmat, rec);
            fs::remove(reordered_rmap, rec);
            // Drop the orphaned sorted-index sidecar too: a fresh permutation
            // is about to be written, and a stale <rmap>.idx from the OLD
            // permutation would otherwise be silently adopted at open() and
            // serve wrong feature rows (root cause of the 2026-06-01 L4
            // corruption). RowMapping::create now also removes it, but make
            // the intent explicit at the rebuild site.
            fs::remove(fs::path(reordered_rmap.string() + ".idx"), rec);
        }
        log_phase("L3 MinHash build_access_graph start");
        MinHashReorderer reorderer(config.minhash);
        reorderer.build_access_graph(catalog.total_batches,
            [&](uint64_t batch_id) -> std::vector<uint64_t> {
                auto sample = samples.read_sample(batch_id);
                std::vector<uint64_t> rows;
                rows.reserve(sample.all_unique_nodes.size());
                for (const auto& oid : sample.all_unique_nodes) {
                    auto row = row_mapping.find(oid);
                    if (row.has_value()) rows.push_back(*row);
                }
                return rows;
            });
        log_phase("L3 MinHash compute_permutation start");
        auto perm = reorderer.compute_permutation(N);
        log_phase("L3 MinHash compute_permutation done");

        log_phase("L3 FeatureMatrix::create_reordered start (Fix #12)");
        FeatureMatrix::create_reordered(features, perm, reordered_fmat, cur_fp);
        log_phase("L3 FeatureMatrix::create_reordered done");

        // Create reordered RowMapping
        log_phase("L3 RowMapping::create start");
        std::vector<ObjectId> reordered_ids(N);
        for (uint64_t i = 0; i < N; ++i) {
            reordered_ids[i] = row_mapping.get(perm[i]);
        }
        RowMapping::create(reordered_rmap, reordered_ids);
        log_phase("L3 RowMapping::create done");
    } else if (config.reorder) {
        log_phase("L3 reorder fresh (fingerprint match) — reusing reordered.fmat");
    } else {
        log_phase("L3 reorder disabled");
    }

    if (config.reorder && fs::exists(reordered_fmat)) {
        reordered_holder.emplace(FeatureMatrix::open(reordered_fmat));
        active_fm = &*reordered_holder;
        reordered_rm_holder.emplace(RowMapping::open(reordered_rmap));
        active_rm = &*reordered_rm_holder;
    }

    // --- Step 5: Re-pack L4 slim ---
    // For each batch: filter out L1/L2 nodes, write only remaining nodes as v2.
    // L3 nodes are INCLUDED in slim files because load_batch_features() reads
    // the OID table to identify L4 nodes -- excluding L3 here would lose them.
    // Instead, we keep all non-cached nodes in slim files (L3+L4), and at
    // runtime load_batch_features() resolves L3 from the reordered FM while
    // L4 features come directly from the slim file data section.
    wrote_slim = true;
    fs::create_directories(packed_slim_dir);

    // Per-entry pair so we can sort by row ascending. Reading the reordered
    // FM in row-monotone order keeps mmap access sequential and avoids
    // page-cache thrash when the FM exceeds host RAM (papers100M: 56 GB
    // FM vs 30 GB host saw the writeback collapse with random reads).
    struct SlimEntry { uint64_t row; ObjectId oid; };

    log_phase("L4 packed_slim start");
    auto l4_start = std::chrono::high_resolution_clock::now();
    std::atomic<uint64_t> batches_done{0};
    // Parallel L4 materialisation: each batch writes its own .bin file, so
    // workers can run independently. Default 4 workers — caps at
    // hardware_concurrency() and is overridable via env MDB_GNN_L4_WORKERS
    // for tuning. The reads from `samples` (per-call ifstream) and
    // `active_fm` (mmap) and `active_rm` (built-once index then read-only)
    // are all thread-safe.
    unsigned num_l4_workers = 4;
    if (const char* env = std::getenv("MDB_GNN_L4_WORKERS")) {
        try {
            int parsed = std::stoi(env);
            if (parsed > 0) num_l4_workers = static_cast<unsigned>(parsed);
        } catch (...) {
            // ignore malformed env value, keep default
        }
    }
    if (num_l4_workers > std::thread::hardware_concurrency() &&
        std::thread::hardware_concurrency() > 0)
    {
        num_l4_workers = std::thread::hardware_concurrency();
    }
    if (num_l4_workers > catalog.total_batches) {
        num_l4_workers = static_cast<unsigned>(catalog.total_batches);
    }
    if (num_l4_workers == 0) num_l4_workers = 1;

    std::atomic<uint64_t> next_batch{0};
    std::atomic<uint64_t> shared_slim_bytes{0};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;

    // Touch RowMapping index once before workers spawn — RowMapping::find()
    // lazy-builds a sorted index under std::call_once on first call. If we
    // didn't force-init here, the first worker pays an O(N log N) sort
    // while the rest stall on the call_once barrier.
    if (N > 0) {
        (void) active_rm->find(active_rm->get(0));
    }

    bool use_pipeline_overlap = false;
    if (const char* env = std::getenv("MDB_GNN_PIPELINE_OVERLAP")) {
        std::string s(env);
        if (s == "1" || s == "true" || s == "yes") use_pipeline_overlap = true;
    }

    auto worker_fn = [&]() {
        // PackedBatch and pipeline state live OUTSIDE the try so the RAII
        // guard below can reach them on any unwind path.
        struct PackedBatch {
            std::vector<char>     out_buf;
            std::filesystem::path batch_path;
            uint64_t              batch_id;
        };

        ChunkPipeline<PackedBatch> pipe(2);
        std::thread                writer;

        // RAII guard: when the lambda returns by any path (normal return,
        // throw, or stack unwinding), close the pipe and join the writer.
        // Skipping this would leave a joinable std::thread to be destroyed,
        // which invokes std::terminate() per [thread.thread.destr].
        struct WriterGuard {
            ChunkPipeline<PackedBatch>* pipe;
            std::thread*                writer;
            bool                        active;
            ~WriterGuard() {
                if (!active) return;
                try {
                    pipe->close();
                } catch (...) { /* swallow — destructors must not throw */ }
                if (writer->joinable()) {
                    try { writer->join(); } catch (...) {}
                }
            }
        };
        WriterGuard guard{&pipe, &writer, false};

        try {
            if (use_pipeline_overlap) {
                writer = std::thread([&]() {
                    try {
                        while (auto pb_opt = pipe.pop()) {
                            auto& pb = *pb_opt;
                            int wfd = ::open(pb.batch_path.c_str(),
                                             O_WRONLY | O_CREAT | O_TRUNC, 0644);
                            if (wfd < 0) {
                                throw std::runtime_error(
                                    "FourLevelStore::build: cannot create " +
                                    pb.batch_path.string() + ": " +
                                    safe_strerror(errno));
                            }
                            FdGuard fdg(wfd);
                            write_all(wfd, pb.out_buf.data(),
                                      pb.out_buf.size(), pb.batch_path.string());
                            // Persist the file data before store.meta can mark
                            // the build complete (matches the partitioned
                            // packer's per-fd fsync in Phase 3).
                            if (::fsync(wfd) < 0) {
                                throw std::runtime_error(
                                    "FourLevelStore::build: fsync failed on " +
                                    pb.batch_path.string() + ": " +
                                    safe_strerror(errno));
                            }
                            // Fix #22: hint kernel that this .bin's pages can
                            // leave the cache. The fsync above made them clean,
                            // so this frees them immediately.
                            fadvise_dontneed(wfd, 0,
                                static_cast<off_t>(pb.out_buf.size()));
                        }
                    } catch (...) {
                        pipe.set_error(std::current_exception());
                    }
                });
                guard.active = true;  // arm the guard now that writer exists
            }

            while (true) {
                uint64_t b = next_batch.fetch_add(1, std::memory_order_relaxed);
                if (b >= catalog.total_batches) break;

                auto sample = samples.read_sample(b);

                std::vector<SlimEntry> entries;
                entries.reserve(sample.all_unique_nodes.size());
                for (const auto& oid : sample.all_unique_nodes) {
                    if (cached_oid_set.count(oid.id) != 0) continue;
                    auto row = active_rm->find(oid);
                    if (row.has_value()) {
                        entries.push_back({*row, oid});
                    }
                }
                std::sort(entries.begin(), entries.end(),
                          [](const SlimEntry& a, const SlimEntry& b) {
                              return a.row < b.row;
                          });

                const size_t n = entries.size();
                const size_t oid_block  = n * sizeof(uint64_t);
                const size_t data_block = n * row_bytes;

                auto header = PackedBatchHeader::make_v2(n, D, dt);

                std::vector<char> out_buf(sizeof(header) + oid_block + data_block);
                std::memcpy(out_buf.data(), &header, sizeof(header));

                auto* oid_ptr  = reinterpret_cast<uint64_t*>(
                    out_buf.data() + sizeof(header));
                char*  feat_ptr = out_buf.data() + sizeof(header) + oid_block;
                for (size_t i = 0; i < n; ++i) {
                    oid_ptr[i] = entries[i].oid.id;
                    std::memcpy(feat_ptr + i * row_bytes,
                                active_fm->row(entries[i].row),
                                row_bytes);
                }

                char fname[32];
                std::snprintf(fname, sizeof(fname), "batch_%06lu.bin",
                              static_cast<unsigned long>(b));
                auto batch_path = packed_slim_dir / fname;

                const size_t this_bytes = out_buf.size();

                if (use_pipeline_overlap) {
                    pipe.push(PackedBatch{std::move(out_buf), batch_path, b});
                } else {
                    int fd = ::open(batch_path.c_str(),
                                    O_WRONLY | O_CREAT | O_TRUNC, 0644);
                    if (fd < 0) {
                        throw std::runtime_error(
                            "FourLevelStore::build: cannot create " +
                            batch_path.string() + ": " + safe_strerror(errno));
                    }
                    FdGuard guard(fd);
                    write_all(fd, out_buf.data(), out_buf.size(), batch_path.string());
                    // Persist the file data before store.meta can mark the
                    // build complete (matches the partitioned packer).
                    if (::fsync(fd) < 0) {
                        throw std::runtime_error(
                            "FourLevelStore::build: fsync failed on " +
                            batch_path.string() + ": " + safe_strerror(errno));
                    }
                    // Fix #22: same hint as in the pipeline writer branch.
                    fadvise_dontneed(fd, 0, static_cast<off_t>(out_buf.size()));
                }

                shared_slim_bytes.fetch_add(this_bytes, std::memory_order_relaxed);

                uint64_t done = batches_done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (done % 100 == 0 || done == catalog.total_batches) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double dt = std::chrono::duration<double>(now - l4_start).count();
                    std::cerr << "[L4] " << done << "/" << catalog.total_batches
                              << " (" << std::fixed << std::setprecision(1)
                              << dt << "s, "
                              << (done / dt) << " batches/s)\n" << std::flush;
                }
            }

            // Normal-exit cleanup: close + join is now redundant with the
            // guard, but explicit close() lets the writer exit promptly so
            // we can observe any writer-side error before disarming. If
            // join() succeeds cleanly, disarm so the destructor is a no-op.
            if (use_pipeline_overlap) {
                pipe.close();
                if (writer.joinable()) writer.join();
                guard.active = false;  // disarm — already joined cleanly
            }
        } catch (...) {
            // Guard destructor will close + join. Just record the exception.
            std::lock_guard<std::mutex> lk(exception_mutex);
            if (!first_exception) first_exception = std::current_exception();
        }
    };

    // Lever B (2026-06-01): opt-in partitioned packed_slim. The default
    // worker loop above gathers each batch's cold (non-L1/L2) rows RANDOMLY
    // from the reordered FM; on papers100M (54 GB FM > 30 GB RAM) those rows
    // are by definition the on-disk cold tier, so every gather faults a page
    // -> ~0.3 batches/s page-cache thrash (IOPS-bound; more workers do not
    // help). MDB_GNN_SLIM_PARTITIONED=1 instead reuses the proven Spec-B1
    // partitioned packer: ONE sequential .fmat scan in row-range partitions +
    // scatter-pwrite each row into the batch files that need it. Output is the
    // identical v2 [header][OID table][features] format read_slim_oid_table
    // consumes; the OID table travels with each row so AddrTable resolution is
    // positionally agnostic (correctness independent of within-batch order).
    // DEFAULT ON since 2026-06-01: the partitioned packer is ~5.7x faster on
    // papers100M-scale (one sequential .fmat scan vs per-batch random gather of
    // cold on-disk rows) and produces BIT-IDENTICAL output (validated cora
    // testAccuracy 0.86240786 classic vs partitioned). Disable with
    // MDB_GNN_SLIM_PARTITIONED=0.
    bool use_slim_partitioned = true;
    if (const char* env = std::getenv("MDB_GNN_SLIM_PARTITIONED")) {
        std::string s(env);
        if (s == "0" || s == "false" || s == "no")       use_slim_partitioned = false;
        else if (s == "1" || s == "true" || s == "yes")  use_slim_partitioned = true;
    }

    // The partitioned packer keeps ONE fd open per batch through its final
    // phase, so it needs NOFILE >= total_batches + headroom. Raise the soft
    // limit toward the hard limit; if even the hard limit can't cover it, fall
    // back to the (slower but unbounded-fd) worker loop instead of failing the
    // build with EMFILE. This makes the fast path a safe default regardless of
    // the inherited ulimit. The raise is build-phase-only: the inherited soft
    // limit is restored once the pack finishes (or throws), so the long-running
    // server keeps its operator-configured fd tripwire.
    struct NofileRestore {
        bool          armed = false;
        struct rlimit saved {};
        void restore() noexcept {
            if (armed) {
                ::setrlimit(RLIMIT_NOFILE, &saved);  // best-effort
                armed = false;
            }
        }
        ~NofileRestore() { restore(); }
    } nofile_restore;
    if (use_slim_partitioned) {
        const rlim_t need = static_cast<rlim_t>(catalog.total_batches) + 64;
        struct rlimit rl;
        if (::getrlimit(RLIMIT_NOFILE, &rl) == 0) {
            if (rl.rlim_cur < need) {
                nofile_restore.saved = rl;
                nofile_restore.armed = true;
                rl.rlim_cur = (rl.rlim_max == RLIM_INFINITY)
                                  ? need
                                  : std::min<rlim_t>(need, rl.rlim_max);
                ::setrlimit(RLIMIT_NOFILE, &rl);   // best-effort
                ::getrlimit(RLIMIT_NOFILE, &rl);   // re-read what we actually got
            }
            if (rl.rlim_cur < need) {
                log_phase("L4 packed_slim: NOFILE soft limit " +
                          std::to_string(static_cast<uint64_t>(rl.rlim_cur)) +
                          " < needed " + std::to_string(static_cast<uint64_t>(need)) +
                          " (total_batches+64) — falling back to non-partitioned packer");
                use_slim_partitioned = false;
            }
        }
    }

    // DiskGNN-adoption Plan 1: consolidated cold-feature file (opt-in). Computed
    // before the partitioned pack so the writer can stamp the perm/meta
    // fingerprints; per-batch (offset, length) are captured for the v2 addr_tables.
    // Only the partitioned (Lever B) path writes it; the legacy worker loop ignores it.
    const bool write_consolidated =
        config.write_consolidated_slim && use_slim_partitioned;
    fs::path              consolidated_path =
        write_consolidated ? (packed_slim_dir / "consolidated.slim") : fs::path{};
    uint64_t              cons_perm_fp  = 0;
    uint64_t              cons_meta_sha = 0;
    std::vector<uint64_t> cons_offsets, cons_lengths;
    if (write_consolidated) {
        cons_perm_fp = reordered_rm_holder.has_value()
                         ? reordered_rm_holder->perm_fingerprint() : 0;
        cons_meta_sha = compute_meta_sha_head(
            gnn_meta_path_for(db_folder, catalog.projection_name));
        log_phase("L4 consolidated.slim: enabled (Plan 1)");
    }

    if (use_slim_partitioned) {
        log_phase("L4 packed_slim: partitioned sequential-scan path (Lever B)");
        size_t slim_partition_bytes = 256ULL * 1024 * 1024;
        if (const char* mb = std::getenv("MDB_GNN_SLIM_PARTITION_MB")) {
            try {
                long v = std::stol(mb);
                if (v > 0) slim_partition_bytes = static_cast<size_t>(v) * 1024 * 1024;
            } catch (...) { /* keep default */ }
        }
        // Cold-node entries for batch b, sorted by row ascending — mirrors the
        // worker-loop collection exactly so output is row-order-equivalent.
        // The partitioned packer pairs oid_provider[k] with row_provider[k],
        // so BOTH lambdas must return the same row-sorted order.
        auto cold_entries = [&](uint64_t b) {
            auto sample = samples.read_sample(b);
            std::vector<SlimEntry> e;
            e.reserve(sample.all_unique_nodes.size());
            for (const auto& oid : sample.all_unique_nodes) {
                if (cached_oid_set.count(oid.id) != 0) continue;
                auto row = active_rm->find(oid);
                if (row.has_value()) e.push_back({*row, oid});
            }
            std::sort(e.begin(), e.end(),
                      [](const SlimEntry& a, const SlimEntry& c) { return a.row < c.row; });
            return e;
        };
        generate_packed_batches_partitioned(
            *active_fm,
            catalog.total_batches,
            [&](uint64_t b) {
                auto e = cold_entries(b);
                std::vector<uint64_t> rows;
                rows.reserve(e.size());
                for (const auto& x : e) rows.push_back(x.row);
                return rows;
            },
            packed_slim_dir,
            slim_partition_bytes,
            [&](uint64_t b) {
                auto e = cold_entries(b);
                std::vector<ObjectId> oids;
                oids.reserve(e.size());
                for (const auto& x : e) oids.push_back(x.oid);
                return oids;
            },
            consolidated_path,
            cons_perm_fp,
            cons_meta_sha,
            write_consolidated ? &cons_offsets : nullptr,
            write_consolidated ? &cons_lengths : nullptr);
        // Account slim bytes from the written files (the partitioned packer
        // does not thread through shared_slim_bytes).
        std::error_code de_ec;
        uint64_t acc = 0;
        for (auto& de : fs::directory_iterator(packed_slim_dir, de_ec)) {
            if (de.path().extension() == ".bin") {
                std::error_code sz_ec;
                auto sz = fs::file_size(de.path(), sz_ec);
                if (!sz_ec) acc += sz;
            }
        }
        shared_slim_bytes.store(acc, std::memory_order_relaxed);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(num_l4_workers);
        for (unsigned i = 0; i < num_l4_workers; ++i) {
            workers.emplace_back(worker_fn);
        }
        for (auto& t : workers) t.join();
        if (first_exception) std::rethrow_exception(first_exception);
    }

    // The per-batch fds are closed now — drop the NOFILE raise immediately
    // (the RAII destructor also covers the throw paths above).
    nofile_restore.restore();

    uint64_t slim_bytes_acc = shared_slim_bytes.load(std::memory_order_relaxed);

    log_phase("L4 packed_slim done, fsync_directory next");
    // Each batch file's data pages were fsync'd at write time; this directory
    // fsync persists the directory entries so the file names survive a crash.
    fsync_directory(packed_slim_dir);
    log_phase("L4 fsync_directory done");

    // --- Phase 5: Build AddrTable sidecars ---
    // Pre-classify every batch's unique nodes into {L1, L2, L4, L3, zero}
    // and write addr_tables/batch_NNNNNN.addr next to packed_slim/. This
    // lets runtime load_batch_features() skip the per-node hash lookups on
    // the hot path, amortising them to build time instead.
    //
    // We reconstruct OidIdxAdapters by reading the already-written GNNC cache
    // files' OID tables rather than loading them into GPU/CPU memory.
    // GpuCache/CpuCache::build() sorts entries by FeatureMatrix row before
    // writing, so the on-disk OID table is the ground truth for find_index().
    if (config.build_addr_tables || config.bake_blocks) {
        log_phase("Phase 5 addr_tables start");
        if (config.build_addr_tables) wrote_addr = true;
        try {
            auto l1_adapter = build_oid_idx_adapter(gpu_cache_path);
            auto l2_adapter = build_oid_idx_adapter(cpu_cache_path);

            auto meta_sha_head = compute_meta_sha_head(
                gnn_meta_path_for(db_folder, catalog.projection_name));

            auto blocks_dir = fs::path(sample_dir) / "blocks";

            build_addr_tables_(
                samples,
                addr_tables_dir,
                packed_slim_dir,
                l1_adapter,
                l2_adapter,
                reordered_rm_holder,
                meta_sha_head,
                catalog.total_batches,
                result.addr_tables_bytes,
                /*write_addr_tables=*/config.build_addr_tables,
                /*bake_blocks=*/config.bake_blocks,
                blocks_dir,
                result.blocks_bytes,
                /*store_fp=*/catalog.sample_content_fp,
                write_consolidated ? &cons_offsets : nullptr,
                write_consolidated ? &cons_lengths : nullptr);

            result.addr_tables_built_ok = true;
            if (config.bake_blocks) result.blocks_built_ok = true;
            log_phase("Phase 5 addr_tables done");
        } catch (const std::exception& ex) {
            // Phase 5 is best-effort: a failure logs a warning but does not
            // abort the feature store build. The runtime consumer falls back
            // to the legacy per-batch hash-lookup path.
            std::cerr << "FourLevelStore::build: WARNING Phase 5 addr_tables failed: "
                      << ex.what() << " — runtime will use per-batch lookup fallback.\n";
            result.addr_tables_built_ok = false;
            result.blocks_built_ok = false;
        }
    }

    // Fix #22: source FM and reordered FM are no longer needed by this
    // procedure. Hint the kernel so the next caller (typically gnn_train
    // running back-to-back) starts with a clean page-cache budget instead
    // of inheriting ~110 GB of stale source+reordered pages.
    features.release_cache();
    if (config.reorder && reordered_holder.has_value()) {
        reordered_holder->release_cache();
    }

    // --- Step 5b: Spec D telemetry — measure on-disk footprint ---
    // We measure files that ALREADY exist on disk (caches were written in
    // Step 3, slim was just written in Step 5, reordered.fmat was written
    // in Step 4 if reorder=true). All measurements are post-fsync.
    auto safe_size = [](const fs::path& p) -> uint64_t {
        std::error_code ec;
        if (!fs::exists(p, ec)) return 0;
        auto sz = fs::file_size(p, ec);
        return ec ? 0 : static_cast<uint64_t>(sz);
    };
    result.slim_bytes      = slim_bytes_acc;
    result.gpu_cache_bytes = safe_size(gpu_cache_path);
    result.cpu_cache_bytes = safe_size(cpu_cache_path);
    result.reordered_bytes = config.reorder ? safe_size(reordered_fmat) : 0;
    result.total_disk_bytes = result.slim_bytes
                            + result.gpu_cache_bytes
                            + result.cpu_cache_bytes
                            + result.reordered_bytes;

    if (config.disk_budget_bytes > 0 &&
        result.total_disk_bytes > config.disk_budget_bytes)
    {
        result.over_budget = true;
        std::cerr << "FourLevelStore::build: warning, on-disk footprint "
                  << result.total_disk_bytes << " B"
                  << " exceeds disk_budget " << config.disk_budget_bytes << " B."
                  << " Future Spec C2 will adjust segment_size automatically;"
                  << " for now, retry with a smaller fanout, smaller batch,"
                  << " or larger disk budget.\n";
    }

    // --- Step 6: Write metadata ---
    auto meta = StoreMetaHeader::make(
        result.l1_nodes, result.l2_nodes, result.l3_nodes, result.l4_nodes,
        D, dt, gpu_available, packed_slim_dir.string());

    {
        wrote_meta = true;
        int fd = ::open(meta_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            throw std::runtime_error(
                "FourLevelStore::build: cannot create " + meta_path.string() +
                ": " + safe_strerror(errno));
        }
        FdGuard guard(fd);
        write_all(fd, &meta, sizeof(meta), meta_path.string());
        if (::fsync(fd) < 0) {
            throw std::runtime_error(
                "FourLevelStore::build: fsync failed on metadata: " +
                safe_strerror(errno));
        }
        fsync_directory(meta_path);
    }

    // STEP 8: write the content-fingerprint sidecar next to store.meta. Written
    // after the meta fsync so the two together mark a fully built store bound to
    // this specific sample+feature. The failure-cleanup catch removes it too.
    write_store_fp(store_fp_path, cur_fp);

    } catch (...) {
        // Best-effort cleanup of THIS run's partial outputs only (see the
        // wrote_* tracking above).
        std::error_code ec;
        if (wrote_slim || wrote_addr) {
            // addr_tables index into packed_slim slots, so removing the slim
            // dir makes any addr_tables stale — they go together.
            fs::remove_all(addr_tables_dir, ec);
        }
        if (wrote_slim) fs::remove_all(packed_slim_dir, ec);
        if (wrote_caches) {
            fs::remove(gpu_cache_path, ec);
            fs::remove(cpu_cache_path, ec);
        }
        if (wrote_meta) {
            fs::remove(meta_path, ec);
            fs::remove(store_fp_path, ec);  // STEP 8 sidecar
        }
        if (wrote_reorder) {
            fs::remove(reordered_fmat, ec);
            fs::remove(reordered_rmap, ec);
            fs::remove(fs::path(reordered_rmap.string() + ".idx"), ec);
        }
        throw;
    }

    // --- Step 6: Cleanup materialize_batches scratch ---
    // The non-slim packed/ directory is scratch intermediate: it was written
    // by gnn_materialize_batches but is NEVER read at runtime (training reads
    // packed_slim/ instead). On large graphs it can occupy tens of GBs.
    // Delete it here unless the caller opted out via Config::cleanup_materialize_scratch=false.
    if (config.cleanup_materialize_scratch) {
        auto packed_scratch = fs::path(sample_dir) / "packed";
        if (fs::exists(packed_scratch)) {
            std::error_code cleanup_ec;
            auto removed_count = fs::remove_all(packed_scratch, cleanup_ec);
            if (cleanup_ec) {
                // Non-fatal: log via cerr but don't abort the build.
                std::cerr << "FourLevelStore::build: warning, failed to cleanup "
                          << packed_scratch.string() << ": " << cleanup_ec.message()
                          << " (you can remove it manually)\n";
            } else if (removed_count > 0) {
                std::cout << "  [cleanup] removed " << (removed_count - 1)
                          << " files from materialize_batches scratch dir "
                          << packed_scratch.string() << "\n";
            }
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start).count();

    return result;
}

// =============================================================================
// Runtime Constructor
// =============================================================================

FourLevelStore::FourLevelStore(
    const fs::path& db_folder,
    const std::string& feature_name,
    SampleStorage& samples)
    : samples_(&samples)
{
    auto gnn_dir     = db_folder / "gnn_features";
    auto gpu_path    = gnn_dir / (feature_name + "_gpu_cache.bin");
    auto cpu_path    = gnn_dir / (feature_name + "_cpu_cache.bin");
    auto meta_path   = gnn_dir / (feature_name + "_store.meta");
    auto reord_fmat  = gnn_dir / (feature_name + "_reordered.fmat");
    auto reord_rmap  = gnn_dir / (feature_name + "_reordered.rmap");
    reord_fmat_path_ = reord_fmat;   // Round 3B-mw: source for per-worker readers

    // Read and validate metadata
    if (!fs::exists(meta_path)) {
        throw std::runtime_error(
            "FourLevelStore: metadata not found at " + meta_path.string() +
            ". Run FourLevelStore::build() first.");
    }

    StoreMetaHeader meta{};
    {
        int fd = ::open(meta_path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(
                "FourLevelStore: cannot open " + meta_path.string() +
                ": " + safe_strerror(errno));
        }
        FdGuard guard(fd);
        read_all(fd, &meta, sizeof(meta), meta_path.string());
    }

    if (!meta.is_valid()) {
        throw std::runtime_error(
            "FourLevelStore: invalid GFLS header in " + meta_path.string());
    }

    packed_slim_dir_ = meta.get_packed_slim_dir();
    feature_dim_     = meta.feature_dim;
    dtype_           = meta.get_dtype();
    elem_size_       = static_cast<uint8_t>(dtype_size(dtype_));

    // Cross-check the tier files store.meta promises before serving anything.
    // reordered.fmat/.rmap are written together — exactly one present is a
    // partially-deleted artifact (L3 rows would be classified but unreadable,
    // or unresolvable, and silently zero-filled). Both cold sources absent
    // (no packed_slim/ and no reordered pair) while meta records cold nodes
    // means NO cold node can be resolved at all.
    {
        const bool have_fmat = fs::exists(reord_fmat);
        const bool have_rmap = fs::exists(reord_rmap);
        if (have_fmat != have_rmap) {
            throw std::runtime_error(
                "FourLevelStore: reordered artifacts are inconsistent: "
                + (have_fmat ? reord_rmap.string() : reord_fmat.string())
                + " is missing while its sibling exists. The store is partially"
                  " deleted — re-run gnn_build_feature_store.");
        }
        // Exception: a packed-full pack legitimately replaces the cold tiers
        // (its build instructions say to delete them while keeping store.meta);
        // the consumer (BatchAssembler) serves features from the pack and
        // never touches L3/L4, so only require the cold tiers when no pack
        // is present.
        const bool packed_full_present = fs::exists(
            fs::path(packed_slim_dir_).parent_path() / "packed_full"
            / "packed_full.dat");
        if ((meta.l3_count + meta.l4_count) > 0 && !have_fmat
            && !fs::exists(fs::path(packed_slim_dir_)) && !packed_full_present)
        {
            throw std::runtime_error(
                "FourLevelStore: store.meta records "
                + std::to_string(meta.l3_count + meta.l4_count)
                + " L3/L4 nodes but neither " + packed_slim_dir_ + " nor "
                + reord_fmat.string()
                + " exists — every cold node would be silently zero-filled."
                  " Re-run gnn_build_feature_store.");
        }
    }

    // Path 4 (2026-05-19): probe for addr_tables sidecar directory.
    // sample_dir_ is one level up from packed_slim_dir_. addr_tables/ is a
    // sibling of packed_slim/.
    sample_dir_ = fs::path(packed_slim_dir_).parent_path();
    auto addr_dir = sample_dir_ / "addr_tables";
    use_addr_tables_ = fs::exists(addr_dir) && fs::is_directory(addr_dir);

    if (use_addr_tables_) {
        // Compute the same FNV-64 hash over gnn_meta.bin that build_addr_tables_
        // used at build time. 0 disables the staleness check.
        auto gnn_meta_path = gnn_meta_path_for(
            db_folder, samples.get_catalog().projection_name);
        expected_meta_sha_head_ = compute_meta_sha_head(gnn_meta_path);
        if (expected_meta_sha_head_ == 0) {
            // Fail-open is the convention (legacy stores may predate
            // gnn_meta.bin), but the disabled state must be visible: with no
            // marker, an addr_table from a different projection generation is
            // adopted subject only to the per-batch node-count guard.
            std::cerr << "[FourLevelStore] WARNING: " << gnn_meta_path.string()
                      << " is missing or unreadable — addr_table staleness"
                         " validation DISABLED.\n";
        }
    }

    // Load GPU cache (L1)
    if (fs::exists(gpu_path)) {
        gpu_cache_ = std::make_unique<GpuCache>(gpu_path);
    }

    // Load CPU cache (L2)
    if (fs::exists(cpu_path)) {
        cpu_cache_ = std::make_unique<CpuCache>(cpu_path);
    }

    // Load L3: try DirectIoReader first (zero page cache via O_DIRECT),
    // fall back to mmap FeatureMatrix if O_DIRECT is not supported
    if (fs::exists(reord_fmat)) {
        // I8: Read and validate actual header before opening reader
        {
            std::ifstream hdr_stream(reord_fmat, std::ios::binary);
            if (hdr_stream.good()) {
                FeatureMatrixHeader fmat_hdr{};
                hdr_stream.read(reinterpret_cast<char*>(&fmat_hdr), sizeof(fmat_hdr));
                if (hdr_stream.good() && fmat_hdr.is_valid()) {
                    l3_header_size_ = FeatureMatrixHeader::SIZE;
                    feature_dim_    = fmat_hdr.num_cols;
                    elem_size_      = static_cast<uint8_t>(dtype_size(fmat_hdr.get_dtype()));
                }
            }
        }

        // I10: Only catch std::runtime_error (expected on tmpfs, NFS, etc.
        // where O_DIRECT is unsupported). Fatal errors (std::bad_alloc,
        // std::logic_error, etc.) must propagate.
        try {
            l3_reader_ = std::make_unique<DirectIoReader>(reord_fmat);
        } catch (const std::runtime_error&) {
            // O_DIRECT not available — fall back to mmap
            l3_reader_.reset();
        }

        if (!l3_reader_) {
            // Mmap fallback
            l3_mmap_fb_.emplace(FeatureMatrix::open(reord_fmat));
        }
    }
    if (fs::exists(reord_rmap)) {
        reordered_rm_.emplace(RowMapping::open(reord_rmap));
    }

    // DiskGNN-adoption Plan 1 Phase 2: probe packed_slim/consolidated.slim.
    // Opt-in (env MDB_GNN_CONSOLIDATED_SLIM). Validate magic/version/dim/dtype +
    // the perm + meta stale-rejection fingerprints; only then open the O_DIRECT
    // + buffered fds and flip use_consolidated_slim_. Any mismatch/absence leaves
    // it false → the per-batch read path. NEVER throws out of the ctor for this.
    {
        const char* ce = std::getenv("MDB_GNN_CONSOLIDATED_SLIM");
        const bool cons_enabled = ce && (std::strcmp(ce, "1") == 0 ||
                                          std::strcmp(ce, "true") == 0 ||
                                          std::strcmp(ce, "yes") == 0);
        auto cons_path = fs::path(packed_slim_dir_) / "consolidated.slim";
        if (cons_enabled && fs::exists(cons_path)) {
            try {
                ConsolidatedSlimHeader ch{};
                bool hdr_ok = false;
                {
                    int fd = ::open(cons_path.c_str(), O_RDONLY);
                    if (fd >= 0) {
                        FdGuard g(fd);
                        struct stat st{};
                        if (::fstat(fd, &st) == 0 &&
                            static_cast<size_t>(st.st_size) >= sizeof(ch)) {
                            read_all(fd, &ch, sizeof(ch), cons_path.string());
                            hdr_ok = true;
                            if (st.st_blksize > 0 &&
                                (st.st_blksize & (st.st_blksize - 1)) == 0) {
                                cons_block_align_ = static_cast<size_t>(st.st_blksize);
                            }
                        }
                    }
                }
                const uint64_t expect_perm =
                    reordered_rm_.has_value() ? reordered_rm_->perm_fingerprint() : 0;
                if (hdr_ok && validate_consolidated_header(
                                  ch, feature_dim_, static_cast<uint8_t>(dtype_),
                                  expect_perm, expected_meta_sha_head_)) {
                    int bf = ::open(cons_path.c_str(), O_RDONLY);
                    if (bf >= 0) {
                        consolidated_buf_fd_   = bf;
                        consolidated_od_fd_    = ::open(cons_path.c_str(),
                                                        O_RDONLY | O_DIRECT);  // -1 if unsupported
                        use_consolidated_slim_ = true;
                    }
                }
            } catch (...) {
                use_consolidated_slim_ = false;  // any failure → per-batch read
            }
        }
    }

    // Feature assembler (always created; dispatches to CUDA kernel or LibTorch
    // index_copy_ internally depending on ENABLE_CUDA_ASSEMBLER + GPU availability)
    assembler_ = std::make_unique<FeatureAssembler>(static_cast<int64_t>(feature_dim_));
}

// =============================================================================
// load_features() — Primitive (L1 -> L2 -> L3 only, no L4)
// =============================================================================

torch::Tensor FourLevelStore::load_features(const std::vector<ObjectId>& oids) {
    uint64_t total = oids.size();
    if (total == 0) {
        return torch::empty(
            {0, static_cast<int64_t>(feature_dim_)},
            torch::TensorOptions().dtype(to_torch_dtype(dtype_)));
    }

    stats_.total_requests += total;

    size_t row_bytes = feature_dim_ * elem_size_;

    // Allocate output tensor [total, D] on CPU
    auto output = torch::zeros(
        {static_cast<int64_t>(total), static_cast<int64_t>(feature_dim_)},
        torch::TensorOptions().dtype(to_torch_dtype(dtype_)));
    char* out_ptr = static_cast<char*>(output.data_ptr());

    uint64_t zero_filled = 0;
    for (uint32_t i = 0; i < total; ++i) {
        const auto& oid = oids[i];

        // Try L1 (GPU cache)
        if (gpu_cache_ && gpu_cache_->contains(oid)) {
            // Single-node lookup through GpuCache
            auto lr = gpu_cache_->lookup({oid});
            if (!lr.hit_positions.empty()) {
                auto cpu_feats = lr.features.cpu().contiguous();
                std::memcpy(out_ptr + i * row_bytes,
                            cpu_feats.data_ptr(),
                            row_bytes);
                stats_.l1_hits++;
                stats_.l1_bytes_served += row_bytes;
                continue;
            }
        }

        // Try L2 (CPU cache)
        if (cpu_cache_ && cpu_cache_->contains(oid)) {
            auto lr = cpu_cache_->lookup({oid});
            if (!lr.hit_positions.empty()) {
                std::memcpy(out_ptr + i * row_bytes,
                            lr.features.data(),
                            row_bytes);
                stats_.l2_hits++;
                stats_.l2_bytes_served += row_bytes;
                continue;
            }
        }

        // Fallback to L3 (reordered FeatureMatrix)
        if (reordered_rm_.has_value()) {
            auto row = reordered_rm_->find(oid);
            if (row.has_value()) {
                if (DirectIoReader* l3_rdr = l3_reader_for_current_worker_()) {
                    // DirectIoReader path: read single row via O_DIRECT
                    std::vector<uint64_t> single_row = {*row};
                    auto result = l3_rdr->read_rows(single_row, row_bytes, l3_header_size_);
                    std::memcpy(out_ptr + i * row_bytes, result.data.get(), row_bytes);
                    stats_.l3_bytes_disk += result.bytes_disk;
                } else if (l3_mmap_fb_.has_value()) {
                    // Mmap fallback path: page cache mediates, count row_bytes
                    // as approximation (no aligned-region amplification here).
                    std::memcpy(out_ptr + i * row_bytes,
                                l3_mmap_fb_->row(*row),
                                row_bytes);
                    stats_.l3_bytes_disk += row_bytes;
                }
                stats_.l3_reads++;
                stats_.l3_bytes_wanted += row_bytes;
                continue;
            }
        }

        // Node not found in any level; leave as zeros (no read happened,
        // so bytes_wanted/bytes_disk remain unchanged — only l3_reads
        // increments, mirroring legacy "miss counts as read" semantics).
        stats_.l3_reads++;
        zero_filled++;
    }

    if (zero_filled > 0) {
        warn_zero_filled("load_features", zero_filled, total);
    }

    return output;
}

// =============================================================================
// load_batch_features() — The Runtime Hot Path (all 4 levels)
// =============================================================================

torch::Tensor FourLevelStore::load_batch_features(uint64_t batch_id) {
    // Round 2B (2026-05-15): legacy path — read the sample here and dispatch
    // to the GraphSample overload. Hot callers (BatchAssembler) skip this and
    // call the overload directly with their already-deserialized sample to
    // avoid the double-deserialize that previously cost ~55 MB read + parse
    // per batch on papers100M-scale runs.
    auto sample = samples_->read_sample(batch_id);
    return load_batch_features(sample);
}

// Public dispatcher: attempts v2 path (addr_table sidecar) when eligible;
// falls through to legacy on miss/stale/error or when the GPU assembler
// gate is not met (CPU-only, non-float32).
torch::Tensor FourLevelStore::load_batch_features(const GraphSample& sample,
                                                  bool* out_used_v2) {
    // Reset per-call v2 telemetry. The legacy path resets its own per-tier
    // timers internally (at the top of load_batch_features_legacy_).
    last_addr_load_ns_ = 0;
    last_used_v2_ = false;
    // Per-call dispatch outcome: starts false, flipped only on a successful v2
    // return below. Concurrent workers each receive their own answer here —
    // the shared last_used_v2_ flag remains telemetry-only.
    if (out_used_v2) *out_used_v2 = false;

    // Gate: v2 only when the GPU assembler path is active. The CPU-only path
    // (no gpu_cache, or non-float32 dtype) is served by legacy_ to avoid
    // duplicating the memcpy assembly logic.
    bool assembler_gate = (assembler_ != nullptr)
                          && (dtype_ == GnnDtype::FLOAT32)
                          && gpu_cache_
                          && gpu_cache_->is_on_gpu();

    if (use_addr_tables_ && assembler_gate) {
        char fname[32];
        std::snprintf(fname, sizeof(fname), "batch_%06lu.addr",
                      static_cast<unsigned long>(sample.batch_id));
        auto addr_path = sample_dir_ / "addr_tables" / fname;
        if (fs::exists(addr_path)) {
            try {
                auto result = load_batch_features_v2_(sample, addr_path);
                if (out_used_v2) *out_used_v2 = true;
                return result;
            } catch (const AddrTableStaleException& e) {
                std::cerr << "[FourLevelStore] addr_table stale for batch "
                          << sample.batch_id << " (" << e.what()
                          << ") — falling back to legacy\n";
            } catch (const std::exception& e) {
                std::cerr << "[FourLevelStore] addr_table read failed for batch "
                          << sample.batch_id << " (" << e.what()
                          << ") — falling back to legacy\n";
            }
        }
    }

    return load_batch_features_legacy_(sample);
}

// ---------------------------------------------------------------------------
// Round 3B-mw follow-up (2026-06-01): opt-in O_DIRECT whole-file read of an L4
// packed_slim .bin (env MDB_GNN_L4_O_DIRECT, default OFF). The buffered read
// path caps ~1.2 GB/s per file via page cache + readahead and makes N
// concurrent prefetch workers contend on a shared page-cache budget under RAM
// pressure. O_DIRECT bypasses the page cache, so workers reading DIFFERENT
// .bin files parallelize toward the NVMe ceiling. It reads the SAME bytes as
// the buffered path → features are bit-identical (validated on cora). Returns
// true with file_buf filled to exactly file_size; false to fall back to the
// buffered read (O_DIRECT unsupported on the fs, alignment/short-read failure).
static bool l4_o_direct_enabled() {
    static const bool on = []() {
        const char* e = std::getenv("MDB_GNN_L4_O_DIRECT");
        return e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "true") == 0 ||
                     std::strcmp(e, "yes") == 0);
    }();
    return on;
}

static bool read_slim_file_o_direct(const std::string& path, size_t file_size,
                                    std::vector<char>& file_buf) {
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) return false;
    FdGuard guard(fd);
    constexpr size_t ALIGN = 4096;
    const size_t cap = ((file_size + ALIGN - 1) / ALIGN) * ALIGN;  // aligned-up
    void* abuf = nullptr;
    if (::posix_memalign(&abuf, ALIGN, cap) != 0 || abuf == nullptr) return false;
    char*  dst = static_cast<char*>(abuf);
    size_t got = 0;
    bool   ok  = true;
    while (got < file_size) {
        // O_DIRECT requires aligned offset+buffer; intermediate reads return
        // block multiples (offset stays aligned) and the final read at EOF
        // returns the partial tail.
        ssize_t r = ::pread(fd, dst + got, cap - got, static_cast<off_t>(got));
        if (r < 0) { if (errno == EINTR) continue; ok = false; break; }
        if (r == 0) break;  // EOF
        got += static_cast<size_t>(r);
    }
    if (ok && got >= file_size) {
        file_buf.assign(dst, dst + file_size);
    } else {
        ok = false;
    }
    std::free(abuf);
    return ok;
}

torch::Tensor FourLevelStore::load_batch_features_legacy_(const GraphSample& sample) {
    const auto& oids = sample.all_unique_nodes;
    const uint64_t batch_id = sample.batch_id;
    uint64_t total = oids.size();

    if (total == 0) {
        return torch::empty(
            {0, static_cast<int64_t>(feature_dim_)},
            torch::TensorOptions().dtype(to_torch_dtype(dtype_)));
    }

    stats_.total_requests += total;

    // Phase 0 (2026-05-17): reset per-call profile timers. Each tier wraps its
    // own block with steady_clock::now() and accumulates nanoseconds into the
    // corresponding last_*_ns_ member. last_rmap_ns_ is a SUB-counter that
    // tracks the (single, on the L3-fallback path) reordered_rm_->find() call
    // and is also already included inside last_l3_ns_.
    last_l1_ns_   = 0;
    last_l2_ns_   = 0;
    last_l3_ns_   = 0;
    last_l4_ns_   = 0;
    last_rmap_ns_ = 0;

    size_t row_bytes = feature_dim_ * elem_size_;

    // Step 1: Read L4 packed_slim for this batch to get OID table
    auto t_l4_step1_start = std::chrono::steady_clock::now();
    char fname[32];
    std::snprintf(fname, sizeof(fname), "batch_%06lu.bin",
                  static_cast<unsigned long>(batch_id));
    auto slim_path = fs::path(packed_slim_dir_) / fname;

    // Read slim file header and OID table
    std::unordered_map<uint64_t, uint32_t> slim_oid_to_idx;
    std::vector<char> slim_data;
    uint64_t slim_nodes = 0;

    if (fs::exists(slim_path)) {
        int fd = ::open(slim_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            FdGuard guard(fd);

            // Round 3A (2026-05-15): single bulk read of the entire .bin file
            // into a stack-local buffer, then parse header/OID-table/features
            // from it. Eliminates 2 syscalls per batch (3 reads -> 1) and the
            // two intervening kernel/user copies. At 1300+ batches × 50 epochs
            // = 65k+ batch-loads per E2E run, the saved syscall round-trips
            // accumulate to several seconds of wall-clock.
            //
            // The kernel SEQUENTIAL hint pre-fetches aggressively for the
            // single bulk read; the DONTNEED hint after consume tells the
            // kernel these pages are no longer hot, preventing the L4 working
            // set from squeezing out productive pages (FeatureMatrix /
            // batches.dat) on memory-pressured runs (papers100M scale).
            //
            // Follow-up (deferred): we could elide the `slim_data` copy by
            // moving `file_buf` into scope and indexing into it at the
            // post-header offset; that's a bigger refactor for marginal gain.

            struct stat st{};
            if (::fstat(fd, &st) < 0) {
                throw std::runtime_error(
                    "FourLevelStore: fstat failed on " + slim_path.string() +
                    ": " + safe_strerror(errno));
            }
            const size_t file_size = static_cast<size_t>(st.st_size);

            if (file_size >= sizeof(PackedBatchHeader)) {
                std::vector<char> file_buf;
                if (!(l4_o_direct_enabled() &&
                      read_slim_file_o_direct(slim_path.string(), file_size, file_buf))) {
                    // Buffered fallback: sequential hint + single bulk read.
                    ::posix_fadvise(fd, 0, static_cast<off_t>(file_size),
                                    POSIX_FADV_SEQUENTIAL);
                    file_buf.resize(file_size);
                    read_all(fd, file_buf.data(), file_size, slim_path.string());
                }

                PackedBatchHeader hdr{};
                std::memcpy(&hdr, file_buf.data(), sizeof(hdr));

                if (hdr.is_valid() && hdr.has_oid_table()) {
                    uint64_t hdr_nodes  = hdr.num_nodes;
                    size_t   data_bytes = hdr.data_bytes();
                    size_t   oid_bytes  = hdr_nodes * sizeof(uint64_t);

                    // Bounds-check the header's claims against the actual
                    // file size — a truncated/corrupted file would have
                    // tripped read_all's "unexpected EOF" in the old 3-read
                    // path; we replicate that safety here without throwing
                    // (treat as missing, identical to the no-file branch).
                    bool sizes_ok = (sizeof(hdr) + oid_bytes + data_bytes)
                                    <= file_size;
                    if (sizes_ok) {
                        slim_nodes = hdr_nodes;

                        // Parse OID table (memcpy from file_buf).
                        const char* p = file_buf.data() + sizeof(hdr);
                        std::vector<uint64_t> oid_table(slim_nodes);
                        if (slim_nodes > 0) {
                            std::memcpy(oid_table.data(), p, oid_bytes);
                            p += oid_bytes;
                        }
                        for (uint32_t j = 0; j < slim_nodes; ++j) {
                            slim_oid_to_idx[oid_table[j]] = j;
                        }

                        // Parse feature data (memcpy from file_buf).
                        if (data_bytes > 0) {
                            slim_data.assign(p, p + data_bytes);
                        }

                        // Spec A1: account for the batch-wide disk traffic
                        // of this slim file (header + OID table + features).
                        // Per-node L4 payload bytes (l4_bytes_wanted) are
                        // accumulated inside the partition loop below; this
                        // is the actual physical read.
                        stats_.l4_bytes_disk += sizeof(hdr)
                                              + oid_bytes
                                              + data_bytes;
                    }
                }

                // Hint kernel that these pages can be evicted now — same
                // hygiene as line 510 (cache-warm path) and the rest of
                // Fix #22's DONTNEED policy in this file.
                fadvise_dontneed(fd, 0, static_cast<off_t>(file_size));
            }
        }
    }
    {
        auto t_l4_step1_end = std::chrono::steady_clock::now();
        last_l4_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l4_step1_end - t_l4_step1_start)
                           .count();
    }

    // Step 2: Partition nodes into L1/L2/L3/L4 buckets.
    // Each bucket stores (output_position, source_data) for later assembly.
    //
    // Round 1C (2026-05-15): store cache row indices directly (l1_indices,
    // l2_indices) instead of re-hashing later. The classification loop now
    // performs one hash per oid (via find_index) on the L1/L2 hit path,
    // down from two (contains() + lookup()/lookup_uva()).
    std::vector<uint32_t> l1_input_positions;    // positions in oids[] for L1 lookup
    std::vector<uint32_t> l1_indices;            // L1 cache row indices for the hits

    std::vector<uint32_t> l2_positions;          // output positions resolved from L2
    std::vector<uint32_t> l2_indices;            // L2 cache row indices for the hits
    std::vector<uint32_t> l3_positions;          // output positions resolved from L3
    std::vector<uint64_t> l3_row_indices;        // corresponding row indices in reordered FM

    std::vector<uint32_t> l4_positions;          // output positions resolved from L4
    std::vector<uint32_t> l4_slim_indices;       // index into slim_data for each L4 node

    uint64_t zero_filled = 0;
    for (uint32_t i = 0; i < total; ++i) {
        const auto& oid = oids[i];

        // Try L1 (GPU cache) — single hash via find_index (Round 1C).
        // Phase 0 (2026-05-17): time the L1 lookup independently of L2/L3/L4.
        if (gpu_cache_) {
            auto t_l1_lookup_start = std::chrono::steady_clock::now();
            auto idx = gpu_cache_->find_index(oid);
            auto t_l1_lookup_end = std::chrono::steady_clock::now();
            last_l1_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t_l1_lookup_end - t_l1_lookup_start)
                               .count();
            if (idx.has_value()) {
                l1_input_positions.push_back(i);
                l1_indices.push_back(*idx);
                stats_.l1_hits++;
                stats_.l1_bytes_served += row_bytes;
                continue;
            }
        }

        // Try L2 (CPU cache) — single hash via find_index (Round 1C).
        // Phase 0 (2026-05-17): time the L2 lookup independently.
        if (cpu_cache_) {
            auto t_l2_lookup_start = std::chrono::steady_clock::now();
            auto idx = cpu_cache_->find_index(oid);
            auto t_l2_lookup_end = std::chrono::steady_clock::now();
            last_l2_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t_l2_lookup_end - t_l2_lookup_start)
                               .count();
            if (idx.has_value()) {
                l2_positions.push_back(i);
                l2_indices.push_back(*idx);
                stats_.l2_hits++;
                stats_.l2_bytes_served += row_bytes;
                continue;
            }
        }

        // Try L4 (slim file data -- check first since slim has exact data)
        // Phase 0 (2026-05-17): time the L4 slim-table classification.
        {
            auto t_l4_cls_start = std::chrono::steady_clock::now();
            auto slim_it = slim_oid_to_idx.find(oid.id);
            bool l4_hit = false;
            if (slim_it != slim_oid_to_idx.end()) {
                uint32_t idx = slim_it->second;
                size_t offset = static_cast<size_t>(idx) * row_bytes;
                if (offset + row_bytes <= slim_data.size()) {
                    l4_positions.push_back(i);
                    l4_slim_indices.push_back(idx);
                    stats_.l4_reads++;
                    stats_.l4_bytes_wanted += row_bytes;
                    l4_hit = true;
                }
            }
            auto t_l4_cls_end = std::chrono::steady_clock::now();
            last_l4_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t_l4_cls_end - t_l4_cls_start)
                               .count();
            if (l4_hit) {
                continue;
            }
        }

        // Fallback to L3 (reordered FeatureMatrix)
        // Phase 0 (2026-05-17): time the L3 reordered_rm_->find() call. This
        // is the only RowMapping lookup in this function — last_rmap_ns_ is a
        // SUB-counter that captures it, and the same elapsed is also rolled
        // into last_l3_ns_.
        {
            auto t_l3_cls_start = std::chrono::steady_clock::now();
            bool l3_hit = false;
            if (reordered_rm_.has_value()) {
                auto t_rmap_start = std::chrono::steady_clock::now();
                auto row = reordered_rm_->find(oid);
                auto t_rmap_end = std::chrono::steady_clock::now();
                last_rmap_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     t_rmap_end - t_rmap_start)
                                     .count();
                if (row.has_value()) {
                    l3_positions.push_back(i);
                    l3_row_indices.push_back(*row);
                    stats_.l3_reads++;
                    stats_.l3_bytes_wanted += row_bytes;
                    l3_hit = true;
                }
            }
            if (!l3_hit) {
                // Node not resolved -- leave as zeros, count as L3 miss.
                // No bytes_wanted increment: nothing was actually read.
                stats_.l3_reads++;
                zero_filled++;
            }
            auto t_l3_cls_end = std::chrono::steady_clock::now();
            last_l3_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t_l3_cls_end - t_l3_cls_start)
                               .count();
        }
    }

    if (zero_filled > 0) {
        warn_zero_filled("legacy batch " + std::to_string(batch_id),
                         zero_filled, total);
    }

    // Step 3: Batch-read L3 rows via DirectIoReader (zero page cache)
    // or mmap fallback. Result is a contiguous buffer of l3_row_indices.size() rows.
    // Phase 0 (2026-05-17): time the L3 disk read.
    std::vector<char> l3_buf;
    {
        auto t_l3_read_start = std::chrono::steady_clock::now();
        if (!l3_row_indices.empty()) {
            DirectIoReader* l3_rdr = l3_reader_for_current_worker_();
            if (l3_rdr) {
                auto result = l3_rdr->read_rows(l3_row_indices, row_bytes, l3_header_size_);
                l3_buf.assign(result.data.get(), result.data.get() + result.size);
                // Spec A1: capture O_DIRECT physical bytes (>= wanted due to
                // 4 KB block alignment overhead — Spec A2 will reduce this).
                stats_.l3_bytes_disk += result.bytes_disk;
            } else if (l3_mmap_fb_.has_value()) {
                l3_buf.resize(l3_row_indices.size() * row_bytes);
                l3_mmap_fb_->extract_rows(l3_row_indices, l3_buf.data());
                // Mmap fallback: page cache mediates, so we count row-level
                // bytes as approximation (no aligned-region amplification visible).
                stats_.l3_bytes_disk += l3_row_indices.size() * row_bytes;
            }
        }
        auto t_l3_read_end = std::chrono::steady_clock::now();
        last_l3_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l3_read_end - t_l3_read_start)
                           .count();
    }

    // Step 4: Assemble output tensor.
    // Use FeatureAssembler when dtype is FLOAT32 and GPU cache is present
    // (assembler targets GPU output via CUDA kernel / index_copy_).
    // Otherwise fall back to direct CPU memcpy.
    //
    // TODO: extend FeatureAssembler for float64 support.
    bool use_assembler = (assembler_ != nullptr)
                         && (dtype_ == GnnDtype::FLOAT32)
                         && gpu_cache_
                         && gpu_cache_->is_on_gpu();

    if (use_assembler) {
        // --- GPU-accelerated assembly path ---

        // L1: gather rows by pre-computed cache indices (Round 1C).
        // The classification loop already validated each L1 hit, so
        // every l1_indices[k] is in-range -- skip the second hash pass.
        // Phase 0 (2026-05-17): time the L1 GPU gather.
        torch::Tensor gpu_features;
        std::vector<uint32_t> gpu_positions;
        if (!l1_indices.empty()) {
            auto t_l1_gather_start = std::chrono::steady_clock::now();
            gpu_features = gpu_cache_->gather_by_indices(l1_indices);
            auto t_l1_gather_end = std::chrono::steady_clock::now();
            last_l1_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t_l1_gather_end - t_l1_gather_start)
                               .count();
            // Every entry in l1_indices was already a hit; positions
            // match 1:1 with l1_input_positions.
            gpu_positions = l1_input_positions;
        }

        // Combine L2 + L3 + L4 CPU features into one contiguous float buffer
        std::vector<float> cpu_combined;
        std::vector<uint32_t> cpu_combined_positions;

        size_t cpu_total = l2_positions.size() + l3_positions.size() + l4_positions.size();
        cpu_combined.reserve(cpu_total * feature_dim_);
        cpu_combined_positions.reserve(cpu_total);

        // L2 features
        //
        // Round 1B (2026-05-15): use CpuCache::lookup_uva() to avoid the
        // per-batch std::vector<char> allocation + memcpy that the standard
        // lookup() does. lookup_uva() returns pointers into the already-
        // pinned feature region; we still memcpy from those pointers into
        // cpu_combined here (so the assembler kernel sees one contiguous
        // pinned buffer), but the row data is copied exactly once (pinned
        // L2 region -> cpu_combined growth -> pinned_ptr_) instead of twice
        // (pinned L2 region -> lr.features heap vector -> cpu_combined ->
        // pinned_ptr_). Saves ~num_l2_hits row-byte memcpys per batch.
        //
        // Round 1C (2026-05-15): use pre-computed l2_indices via row_ptr()
        // directly -- skip the lookup_uva() find loop. Each L2 hit was
        // already validated in the classification loop.
        // Phase 0 (2026-05-17): time the L2 pinned-row copy.
        if (!l2_positions.empty()) {
            auto t_l2_copy_start = std::chrono::steady_clock::now();
            for (size_t h = 0; h < l2_positions.size(); ++h) {
                cpu_combined_positions.push_back(l2_positions[h]);
                const float* row_ptr_f = static_cast<const float*>(
                    cpu_cache_->row_ptr(l2_indices[h]));
                cpu_combined.insert(cpu_combined.end(),
                                    row_ptr_f,
                                    row_ptr_f + feature_dim_);
            }
            auto t_l2_copy_end = std::chrono::steady_clock::now();
            last_l2_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t_l2_copy_end - t_l2_copy_start)
                               .count();
        }

        // L3 features (already in l3_buf, in order of l3_row_indices)
        // Phase 0 (2026-05-17): time the L3 copy into the combined buffer.
        if (!l3_positions.empty() && !l3_buf.empty()) {
            auto t_l3_copy_start = std::chrono::steady_clock::now();
            const float* l3_data = reinterpret_cast<const float*>(l3_buf.data());
            for (size_t j = 0; j < l3_positions.size(); ++j) {
                cpu_combined_positions.push_back(l3_positions[j]);
                cpu_combined.insert(cpu_combined.end(),
                                    l3_data + j * feature_dim_,
                                    l3_data + (j + 1) * feature_dim_);
            }
            auto t_l3_copy_end = std::chrono::steady_clock::now();
            last_l3_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t_l3_copy_end - t_l3_copy_start)
                               .count();
        }

        // L4 features (from slim_data)
        // Phase 0 (2026-05-17): time the L4 copy into the combined buffer.
        if (!l4_positions.empty()) {
            auto t_l4_copy_start = std::chrono::steady_clock::now();
            const float* slim_float = reinterpret_cast<const float*>(slim_data.data());
            for (size_t j = 0; j < l4_positions.size(); ++j) {
                cpu_combined_positions.push_back(l4_positions[j]);
                uint32_t idx = l4_slim_indices[j];
                cpu_combined.insert(cpu_combined.end(),
                                    slim_float + idx * feature_dim_,
                                    slim_float + (idx + 1) * feature_dim_);
            }
            auto t_l4_copy_end = std::chrono::steady_clock::now();
            last_l4_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t_l4_copy_end - t_l4_copy_start)
                               .count();
        }

        // C2: Pin the CPU combined buffer for UVA access from the CUDA
        // assembler kernel.  Unpinned heap memory forces the GPU to read
        // through UVA at ~120 MB/s; pinned memory achieves ~12 GB/s.
        //
        // Round 1A (2026-05-15): use the persistent pinned-host buffer pool
        // instead of cudaHostAlloc+cudaFreeHost per batch. The buffer grows
        // geometrically and is freed once in ~FourLevelStore().
        const float* assembler_data = cpu_combined.data();
#ifdef GNN_CUDA_ENABLED
        size_t cpu_combined_bytes = cpu_combined.size() * sizeof(float);
        void* worker_pinned = nullptr;
        if (cpu_combined_bytes > 0
            && ensure_pinned_capacity_for_worker_(cpu_combined_bytes, worker_pinned)
            && worker_pinned != nullptr) {
            // Per-worker pinned buffer (Round 3B-mw): the calling worker owns
            // this slot, so the memcpy + the assemble kernel that reads it
            // cannot race another worker. assemble() is host-blocking, so the
            // buffer is safe to reuse on this worker's next batch.
            std::memcpy(worker_pinned, cpu_combined.data(), cpu_combined_bytes);
            assembler_data = reinterpret_cast<const float*>(worker_pinned);
        }
#endif

        auto result_tensor = assembler_->assemble(
            static_cast<int64_t>(total),
            gpu_features, gpu_positions,
            assembler_data,
            static_cast<int64_t>(cpu_combined_positions.size()),
            cpu_combined_positions
        );

        // No per-batch cudaFreeHost — pinned_ptr_ is reused; freed in dtor.
        return result_tensor;
    }

    // --- CPU-only assembly path (no GPU, or non-float32 dtype) ---
    auto output = torch::zeros(
        {static_cast<int64_t>(total), static_cast<int64_t>(feature_dim_)},
        torch::TensorOptions().dtype(to_torch_dtype(dtype_)));
    char* out_ptr = static_cast<char*>(output.data_ptr());

    // L1 features (GPU cache -> CPU copy)
    //
    // Round 1C (2026-05-15): batch via gather_by_indices(l1_indices) +
    // single .cpu().contiguous() + strided memcpy. Pre-Round-1C this loop
    // did one gpu_cache_->lookup({l1_input_oids[k]}) per hit -- each call
    // built a one-element index tensor, ran index_select on GPU, and
    // copied back to host. Now one gather/copy serves all L1 hits.
    // Phase 0 (2026-05-17): time the CPU-path L1 gather+copy.
    if (!l1_indices.empty()) {
        auto t_l1_cpu_start = std::chrono::steady_clock::now();
        auto l1_feats = gpu_cache_->gather_by_indices(l1_indices)
                            .cpu().contiguous();
        const char* l1_src = static_cast<const char*>(l1_feats.data_ptr());
        for (size_t k = 0; k < l1_indices.size(); ++k) {
            std::memcpy(out_ptr + l1_input_positions[k] * row_bytes,
                        l1_src + k * row_bytes, row_bytes);
        }
        auto t_l1_cpu_end = std::chrono::steady_clock::now();
        last_l1_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l1_cpu_end - t_l1_cpu_start)
                           .count();
    }

    // L2 features
    //
    // Round 1B (2026-05-15): batch + zero-copy via lookup_uva. Pre-fix did
    // one cpu_cache_->lookup({oids[pos]}) per L2 hit, each call allocating
    // a fresh std::vector<char>(row_bytes) and memcpying the row into it,
    // then we memcpyed from that vector into out_ptr. Now we ask for all
    // L2 oids in one call, get UVA pointers back, and memcpy directly from
    // the pinned region into out_ptr.
    //
    // Round 1C (2026-05-15): use the pre-computed l2_indices + row_ptr()
    // directly so we skip the find loop inside lookup_uva.
    // Phase 0 (2026-05-17): time the CPU-path L2 copy.
    if (!l2_positions.empty()) {
        auto t_l2_cpu_start = std::chrono::steady_clock::now();
        for (size_t h = 0; h < l2_positions.size(); ++h) {
            std::memcpy(out_ptr + l2_positions[h] * row_bytes,
                        cpu_cache_->row_ptr(l2_indices[h]), row_bytes);
        }
        auto t_l2_cpu_end = std::chrono::steady_clock::now();
        last_l2_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l2_cpu_end - t_l2_cpu_start)
                           .count();
    }

    // L3 features (from batched read in l3_buf)
    // Phase 0 (2026-05-17): time the CPU-path L3 copy.
    if (!l3_positions.empty()) {
        auto t_l3_cpu_start = std::chrono::steady_clock::now();
        for (size_t j = 0; j < l3_positions.size(); ++j) {
            std::memcpy(out_ptr + l3_positions[j] * row_bytes,
                        l3_buf.data() + j * row_bytes, row_bytes);
        }
        auto t_l3_cpu_end = std::chrono::steady_clock::now();
        last_l3_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l3_cpu_end - t_l3_cpu_start)
                           .count();
    }

    // L4 features (from slim_data)
    // Phase 0 (2026-05-17): time the CPU-path L4 copy.
    if (!l4_positions.empty()) {
        auto t_l4_cpu_start = std::chrono::steady_clock::now();
        for (size_t j = 0; j < l4_positions.size(); ++j) {
            uint32_t idx = l4_slim_indices[j];
            size_t offset = static_cast<size_t>(idx) * row_bytes;
            std::memcpy(out_ptr + l4_positions[j] * row_bytes,
                        slim_data.data() + offset, row_bytes);
        }
        auto t_l4_cpu_end = std::chrono::steady_clock::now();
        last_l4_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l4_cpu_end - t_l4_cpu_start)
                           .count();
    }

    return output;
}

// =============================================================================
// load_batch_features_v2_() — Path 4 fast path (2026-05-19)
// =============================================================================
//
// Reads the pre-classified addr_table sidecar for this batch and assembles
// the output tensor using the same GPU assembler path as legacy_ — but
// without the per-node hash lookup loop (Steps 1-2 of legacy_).
//
// Only called when assembler_ is active and gpu_cache_->is_on_gpu() (the
// dispatcher enforces this gate). On meta_sha mismatch throws
// AddrTableStaleException so the dispatcher catches and falls back cleanly.
//
// Stats accounting matches legacy_ for paper-comparable I/O reporting.
// =============================================================================

torch::Tensor FourLevelStore::load_batch_features_v2_(
    const GraphSample& sample,
    const fs::path&    addr_path)
{
    // Reset per-tier timers (v2 does not time individual hash lookups, but
    // we zero them so stale ns values from a previous batch do not bleed
    // through if the caller inspects last_l?_us() after a v2 serve).
    last_l1_ns_   = 0;
    last_l2_ns_   = 0;
    last_l3_ns_   = 0;
    last_l4_ns_   = 0;
    last_rmap_ns_ = 0;

    // --- Open and validate the addr_table sidecar ---
    auto t_addr_start = std::chrono::steady_clock::now();
    auto addr = AddrTableReader::open(addr_path, expected_meta_sha_head_);
    auto t_addr_end = std::chrono::steady_clock::now();
    last_addr_load_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t_addr_end - t_addr_start).count();

    // C3: validate that the addr_table was built for this exact sample.
    // If gnn_offline_sample was re-run without rebuilding addr_tables, the
    // header total will disagree with the sample's actual node count — the
    // resulting tensor would silently have the wrong shape.  Throw
    // AddrTableStaleException so the dispatcher logs "stale" and falls back
    // to legacy cleanly.
    if (addr.header.total != static_cast<uint32_t>(sample.all_unique_nodes.size())) {
        throw AddrTableStaleException(
            "v2: addr_table total " + std::to_string(addr.header.total)
            + " != sample.all_unique_nodes.size() "
            + std::to_string(sample.all_unique_nodes.size())
            + " — sample regenerated without rebuilding addr_tables");
    }

    const uint64_t total    = addr.header.total;
    const size_t   row_bytes = feature_dim_ * elem_size_;

    if (total == 0) {
        return torch::empty(
            {0, static_cast<int64_t>(feature_dim_)},
            torch::TensorOptions().dtype(to_torch_dtype(dtype_)));
    }

    stats_.total_requests += total;

    // --- Step 1 (v2): Read L4 packed_slim file for this batch ---
    //
    // Same as legacy_ Step 1: read the entire .bin file, parse header + OID
    // table, extract slim_data. We need slim_data to supply L4 features by
    // their pre-classified l4_indices. The OID table lookup is not needed
    // (addr table already has indices), but we still read the file for the
    // feature payload and for l4_bytes_disk accounting parity with legacy_.
    auto t_l4_read_start = std::chrono::steady_clock::now();
    // The L4 data section lives inside slim_owner at
    // [slim_data_offset, slim_data_offset + slim_data_bytes). For the per-batch
    // .bin path we MOVE the whole file buffer in and index it (no separate copy
    // of the data section); the consolidated path owns its exact-range buffer at
    // offset 0.
    std::vector<char> slim_owner;
    size_t slim_data_offset = 0;
    size_t slim_data_bytes  = 0;
    if (addr.header.num_l4 > 0) {
      if (use_consolidated_slim_ &&
          addr.header.version >= AddrTableHeader::VERSION_V2 &&
          addr.header.slim_length > 0) {
        // DiskGNN-adoption Plan 1: ONE pread of this batch's payload from the
        // single consolidated.slim at [slim_offset, slim_length). The payload is
        // byte-identical to the per-batch .bin data section (same partition
        // order), so the l4_indices below index it identically. O_DIRECT first
        // (page-cache bypass → N workers parallelize); buffered fallback; on
        // total failure throw so the dispatcher falls back to legacy.
        const uint64_t off = addr.header.slim_offset;
        const uint64_t len = addr.header.slim_length;
        if (len % row_bytes != 0) {
            throw std::runtime_error(
                "v2: consolidated slim_length " + std::to_string(len)
                + " not a multiple of row_bytes " + std::to_string(row_bytes));
        }
        const size_t aligned_len =
            ((len + cons_block_align_ - 1) / cons_block_align_) * cons_block_align_;
        void* abuf = nullptr;
        if (::posix_memalign(&abuf, cons_block_align_, aligned_len) != 0 || !abuf) {
            throw std::runtime_error("v2: consolidated posix_memalign failed");
        }
        bool ok = (consolidated_od_fd_ >= 0) &&
                  pread_exact(consolidated_od_fd_, abuf, aligned_len, off);
        if (!ok) {
            // O_DIRECT unavailable/failed → buffered read of the EXACT range.
            ok = pread_exact(consolidated_buf_fd_, abuf, len, off);
        }
        if (!ok) {
            std::free(abuf);
            throw std::runtime_error(
                "v2: consolidated read failed at offset " + std::to_string(off));
        }
        // Consolidated read uses an O_DIRECT-aligned buffer that must be freed,
        // so one copy out of it is unavoidable; own the exact range at offset 0.
        slim_owner.assign(static_cast<char*>(abuf), static_cast<char*>(abuf) + len);
        slim_data_offset = 0;
        slim_data_bytes  = len;
        std::free(abuf);
        stats_.l4_bytes_disk += len;
      } else {
        // C1: any failure here is a hard error — slim_data must be populated
        // when num_l4 > 0, otherwise the L4 copy loop reads an empty buffer
        // (UB).  Throw std::runtime_error so the dispatcher catches, logs
        // "read failed", and falls back to legacy.
        char fname[32];
        std::snprintf(fname, sizeof(fname), "batch_%06lu.bin",
                      static_cast<unsigned long>(sample.batch_id));
        auto slim_path = fs::path(packed_slim_dir_) / fname;

        int fd = ::open(slim_path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("v2: cannot open slim file "
                                     + slim_path.string()
                                     + ": " + std::strerror(errno));
        }
        FdGuard guard(fd);

        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            throw std::runtime_error("v2: fstat failed on "
                                     + slim_path.string()
                                     + ": " + std::strerror(errno));
        }
        const size_t file_size = static_cast<size_t>(st.st_size);
        if (file_size <= sizeof(PackedBatchHeader)) {
            throw std::runtime_error("v2: slim file too small: "
                                     + slim_path.string());
        }

        // Round 3B-mw: opt-in O_DIRECT whole-file read (bypasses page cache so
        // N workers' per-file reads parallelize). Falls back to buffered.
        std::vector<char> file_buf;
        if (!(l4_o_direct_enabled() &&
              read_slim_file_o_direct(slim_path.string(), file_size, file_buf))) {
            // Fix #22: SEQUENTIAL hint before read, DONTNEED after (page-cache
            // relief so late-phase L4 throughput does not collapse on 30 GB hosts).
            ::posix_fadvise(fd, 0, static_cast<off_t>(file_size),
                            POSIX_FADV_SEQUENTIAL);
            file_buf.resize(file_size);
            read_all(fd, file_buf.data(), file_size, slim_path.string());
        }

        PackedBatchHeader hdr{};
        std::memcpy(&hdr, file_buf.data(), sizeof(hdr));
        if (!hdr.is_valid() || !hdr.has_oid_table()) {
            throw std::runtime_error("v2: invalid slim header in "
                                     + slim_path.string());
        }

        const size_t oid_bytes   = hdr.num_nodes * sizeof(uint64_t);
        const size_t data_offset = sizeof(hdr) + oid_bytes;
        const size_t data_bytes  = hdr.data_bytes();
        if (data_offset + data_bytes > file_size) {
            throw std::runtime_error(
                "v2: slim file size invariant violated: " + slim_path.string());
        }

        // Move the file buffer in and index its data section directly — avoids
        // copying ~data_bytes (the per-batch feature payload, tens of MB on
        // papers100M) into a separate slim_data vector.
        slim_owner       = std::move(file_buf);
        slim_data_offset = data_offset;
        slim_data_bytes  = data_bytes;
        stats_.l4_bytes_disk += sizeof(hdr) + oid_bytes + data_bytes;

        fadvise_dontneed(fd, 0, static_cast<off_t>(file_size));
      }  // end else (per-batch .bin read)
    }
    {
        auto t_l4_read_end = std::chrono::steady_clock::now();
        last_l4_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l4_read_end - t_l4_read_start).count();
    }

    // --- Step 2 (v2): Read L3 rows from disk ---
    //
    // Use addr.l3_row_idxs (pre-resolved row indices in the reordered FM)
    // directly — skips the per-node RowMapping::find() call.
    std::vector<char> l3_buf;
    {
        auto t_l3_read_start = std::chrono::steady_clock::now();
        if (addr.header.num_l3 > 0) {
            // Convert ConstView<uint64_t> -> vector<uint64_t> for read_rows.
            std::vector<uint64_t> l3_row_indices(
                addr.l3_row_idxs.data,
                addr.l3_row_idxs.data + addr.header.num_l3);
            DirectIoReader* l3_rdr = l3_reader_for_current_worker_();
            if (l3_rdr) {
                auto result = l3_rdr->read_rows(
                    l3_row_indices, row_bytes, l3_header_size_);
                l3_buf.assign(result.data.get(),
                              result.data.get() + result.size);
                stats_.l3_bytes_disk += result.bytes_disk;
            } else if (l3_mmap_fb_.has_value()) {
                l3_buf.resize(addr.header.num_l3 * row_bytes);
                l3_mmap_fb_->extract_rows(l3_row_indices, l3_buf.data());
                stats_.l3_bytes_disk += addr.header.num_l3 * row_bytes;
            }
        }
        auto t_l3_read_end = std::chrono::steady_clock::now();
        last_l3_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l3_read_end - t_l3_read_start).count();
    }

    // --- Step 3 (v2): GPU gather for L1 ---
    //
    // l1_indices from the addr_table are gpu_cache row indices; gather_by_indices
    // returns a GPU tensor of shape [num_l1, D].
    torch::Tensor gpu_features;
    std::vector<uint32_t> gpu_positions;
    if (addr.header.num_l1 > 0) {
        auto t_l1_start = std::chrono::steady_clock::now();
        std::vector<uint32_t> l1_indices(
            addr.l1_indices.data,
            addr.l1_indices.data + addr.header.num_l1);
        // Bounds-check every L1 index, symmetric with the L2/L4 tiers below.
        // gather_by_indices takes pre-validated indices and runs index_select
        // on CUDA — an out-of-range index from a stale-but-format-valid
        // addr_table would fire a device-side assert and poison the CUDA
        // context for the whole process. Throw on the host instead so the
        // dispatcher falls back to legacy.
        const uint64_t l1_rows = gpu_cache_->num_nodes();
        for (uint32_t l1_idx : l1_indices) {
            if (static_cast<uint64_t>(l1_idx) >= l1_rows) {
                throw std::runtime_error(
                    "v2: L1 index " + std::to_string(l1_idx)
                    + " out of bounds for GPU cache ("
                    + std::to_string(l1_rows) + " rows)");
            }
        }
        gpu_features = gpu_cache_->gather_by_indices(l1_indices);
        gpu_positions.assign(addr.l1_positions.data,
                             addr.l1_positions.data + addr.header.num_l1);
        auto t_l1_end = std::chrono::steady_clock::now();
        last_l1_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l1_end - t_l1_start).count();
        stats_.l1_hits        += addr.header.num_l1;
        stats_.l1_bytes_served += addr.header.num_l1 * row_bytes;
    }

    // --- Step 4+5 (v2): assemble L2 + L3 + L4 directly into the pinned buffer ---
    //
    // Previously this built a std::vector<float> cpu_combined (row-by-row
    // insert, reallocating as it grew) and then memcpy'd the whole thing into
    // the worker pinned buffer — two ~(cpu_total×D) passes over hundreds of MB
    // per batch on papers100M. We now size the pinned destination up front and
    // write each tier's rows STRAIGHT into it (one pass), mirroring DiskGNN's
    // read→scatter-by-precomputed-index pattern. Byte-identical output: same
    // rows, same order (L2→L3→L4), same positions; only the intermediate vector
    // + the final memcpy are gone. Falls back to a heap buffer when a pinned
    // destination is unavailable (CPU-only build / pinned alloc failure) — that
    // fallback path is byte-identical to the pre-refactor heap path.
    std::vector<uint32_t> cpu_combined_positions;

    size_t cpu_total = addr.header.num_l2
                     + addr.header.num_l3
                     + addr.header.num_l4;
    cpu_combined_positions.reserve(cpu_total);

    const size_t cpu_total_floats = cpu_total * feature_dim_;
    float* dst = nullptr;                 // contiguous [rows,D] destination
    std::vector<float> cpu_fallback;      // used only when pinned is unavailable
#ifdef GNN_CUDA_ENABLED
    void* worker_pinned = nullptr;
    const size_t cpu_total_bytes = cpu_total_floats * sizeof(float);
    if (cpu_total_bytes > 0
        && ensure_pinned_capacity_for_worker_(cpu_total_bytes, worker_pinned)
        && worker_pinned != nullptr) {
        // Per-worker pinned buffer (Round 3B-mw). Sized once; the pointer is
        // stable for the whole assembly (no realloc between here and assemble).
        dst = reinterpret_cast<float*>(worker_pinned);
    }
#endif
    if (dst == nullptr && cpu_total_floats > 0) {
        cpu_fallback.resize(cpu_total_floats);
        dst = cpu_fallback.data();
    }
    size_t cursor = 0;  // running float offset into dst (advances in row order)

    // L2: copy rows from cpu_cache_ by pre-resolved cache indices.
    if (addr.header.num_l2 > 0) {
        auto t_l2_start = std::chrono::steady_clock::now();
        // Bounds-check every L2 index, symmetric with the L4 tier below. A
        // stale-but-format-valid addr_table can carry l2_indices that exceed
        // the CPU cache's row count; CpuCache::row_ptr does NOT validate (the
        // caller owns idx < num_nodes), so an unchecked deref would walk past
        // the cache buffer (UB / silent wrong-node features).
        const uint64_t l2_rows = cpu_cache_->num_nodes();
        for (uint32_t h = 0; h < addr.header.num_l2; ++h) {
            cpu_combined_positions.push_back(addr.l2_positions[h]);
            const uint32_t l2_idx = addr.l2_indices[h];
            if (static_cast<uint64_t>(l2_idx) >= l2_rows) {
                throw std::runtime_error(
                    "v2: L2 index " + std::to_string(l2_idx)
                    + " out of bounds for CPU cache ("
                    + std::to_string(l2_rows) + " rows)");
            }
            const float* row_ptr_f = static_cast<const float*>(
                cpu_cache_->row_ptr(l2_idx));
            std::memcpy(dst + cursor, row_ptr_f, feature_dim_ * sizeof(float));
            cursor += feature_dim_;
        }
        auto t_l2_end = std::chrono::steady_clock::now();
        last_l2_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l2_end - t_l2_start).count();
        stats_.l2_hits        += addr.header.num_l2;
        stats_.l2_bytes_served += addr.header.num_l2 * row_bytes;
    }

    // L3: rows are in l3_buf in the order of l3_row_idxs (same order as
    // addr.l3_positions), so we append them in sequence.
    if (addr.header.num_l3 > 0 && !l3_buf.empty()) {
        auto t_l3_copy_start = std::chrono::steady_clock::now();
        const float* l3_data = reinterpret_cast<const float*>(l3_buf.data());
        for (uint32_t j = 0; j < addr.header.num_l3; ++j) {
            cpu_combined_positions.push_back(addr.l3_positions[j]);
            std::memcpy(dst + cursor, l3_data + j * feature_dim_,
                        feature_dim_ * sizeof(float));
            cursor += feature_dim_;
        }
        auto t_l3_copy_end = std::chrono::steady_clock::now();
        last_l3_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l3_copy_end - t_l3_copy_start).count();
        stats_.l3_reads       += addr.header.num_l3;
        stats_.l3_bytes_wanted += addr.header.num_l3 * row_bytes;
    } else if (addr.header.num_l3 > 0 && l3_buf.empty()) {
        // L3 rows requested but nothing was readable (no reader + no mmap).
        // Count as reads with no data — same semantics as legacy_ line 1414.
        // (No row written and no position pushed — identical to the old
        // insert-of-nothing; dst was over-sized by num_l3 rows, harmless.)
        stats_.l3_reads += addr.header.num_l3;
    }

    // Zero-classified misses (unresolved nodes): just count as l3_reads with
    // no data, matching legacy_'s "leave as zeros" path (line 1414).
    stats_.l3_reads += addr.header.num_zero;
    if (addr.header.num_zero > 0) {
        warn_zero_filled("v2 batch " + std::to_string(sample.batch_id),
                         addr.header.num_zero, total);
    }

    // L4: addr.l4_indices are packed_slim file slot indices.
    if (addr.header.num_l4 > 0) {
        auto t_l4_copy_start = std::chrono::steady_clock::now();
        // C2: bounds-check every L4 index.  A stale-but-format-valid addr_table
        // can carry l4_indices that exceed the slim file's row count; an unchecked
        // read would silently walk past the end of slim_data (UB).
        const float*  slim_float           = reinterpret_cast<const float*>(
                                                 slim_owner.data() + slim_data_offset);
        const size_t  slim_capacity_floats = slim_data_bytes / sizeof(float);
        for (uint32_t j = 0; j < addr.header.num_l4; ++j) {
            cpu_combined_positions.push_back(addr.l4_positions[j]);
            uint32_t     idx           = addr.l4_indices[j];
            // Widen BEFORE adding: idx + 1 in uint32 wraps to 0 for
            // idx == UINT32_MAX, which would pass the bounds check below.
            size_t       row_end_floats = (static_cast<size_t>(idx) + 1) * feature_dim_;
            if (row_end_floats > slim_capacity_floats) {
                throw std::runtime_error(
                    "v2: L4 index " + std::to_string(idx)
                    + " out of bounds for slim file ("
                    + std::to_string(slim_capacity_floats / feature_dim_)
                    + " rows)");
            }
            std::memcpy(dst + cursor, slim_float + idx * feature_dim_,
                        feature_dim_ * sizeof(float));
            cursor += feature_dim_;
        }
        auto t_l4_copy_end = std::chrono::steady_clock::now();
        last_l4_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_l4_copy_end - t_l4_copy_start).count();
        stats_.l4_reads       += addr.header.num_l4;
        stats_.l4_bytes_wanted += addr.header.num_l4 * row_bytes;
    }

    // Step 5 (v2): the destination is already the pinned (or fallback) buffer —
    // no separate pin/memcpy pass. `dst` is nullptr only when cpu_total==0, in
    // which case assemble() reads zero CPU rows and never dereferences it.
    const float* assembler_data = dst;

    // I1: set the flag AFTER assemble() returns successfully.  If assemble()
    // throws, the flag stays false so the dispatcher's catch block and any
    // telemetry consumer see the correct "legacy actually served" state.
    auto result = assembler_->assemble(
        static_cast<int64_t>(total),
        gpu_features, gpu_positions,
        assembler_data,
        static_cast<int64_t>(cpu_combined_positions.size()),
        cpu_combined_positions
    );
    last_used_v2_ = true;
    return result;
}

} // namespace mdb::gnn
