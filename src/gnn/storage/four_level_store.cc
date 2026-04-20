#include "gnn/storage/four_level_store.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <fcntl.h>

#ifdef GNN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include "gnn/common/posix_io.h"
#include "gnn/core/feature_assembler.h"
#include "gnn/storage/cache_file.h"
#include "gnn/storage/direct_io_reader.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/feature_matrix_header.h"
#include "gnn/storage/gnn_dtype.h"
#include "gnn/storage/packed_batch_store.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/sampling/sample_storage.h"
#include "graph_models/object_id.h"

namespace fs = std::filesystem;

namespace mdb::gnn {

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
    auto packed_slim_dir = fs::path(sample_dir) / "packed_slim";
    result.packed_slim_dir = packed_slim_dir.string();

    // --- Force cleanup ---
    if (config.force) {
        std::error_code ec;
        fs::remove_all(packed_slim_dir, ec);
        fs::remove(gpu_cache_path, ec);
        fs::remove(cpu_cache_path, ec);
        fs::remove(meta_path, ec);
        fs::remove(reordered_fmat, ec);
        fs::remove(reordered_rmap, ec);
    }

    // --- Pre-condition checks ---
    if (fs::exists(meta_path)) {
        throw std::runtime_error(
            "Feature store already exists at: " + meta_path.string() + "\n"
            "Use force:1 to overwrite.");
    }

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

    // --- Cleanup guard for partial outputs ---
    try {

    // --- Step 3: Build cache files ---
    fs::create_directories(gnn_dir);
    GpuCache::build(l1_nodes, features, row_mapping, gpu_cache_path);
    CpuCache::build(l2_nodes, features, row_mapping, cpu_cache_path);

    // --- Step 4: Ensure L3 reordered FM exists ---
    const FeatureMatrix* active_fm = &features;
    std::optional<FeatureMatrix> reordered_holder;
    const RowMapping* active_rm = &row_mapping;
    std::optional<RowMapping> reordered_rm_holder;

    if (config.reorder && !fs::exists(reordered_fmat)) {
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
        auto perm = reorderer.compute_permutation(N);

        FeatureMatrix::create_reordered(features, perm, reordered_fmat);

        // Create reordered RowMapping
        std::vector<ObjectId> reordered_ids(N);
        for (uint64_t i = 0; i < N; ++i) {
            reordered_ids[i] = row_mapping.get(perm[i]);
        }
        RowMapping::create(reordered_rmap, reordered_ids);
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
    fs::create_directories(packed_slim_dir);

    for (uint64_t b = 0; b < catalog.total_batches; ++b) {
        auto sample = samples.read_sample(b);

        // Partition: keep only non-cached nodes (not in L1/L2)
        std::vector<ObjectId> slim_oids;
        slim_oids.reserve(sample.all_unique_nodes.size());
        for (const auto& oid : sample.all_unique_nodes) {
            if (cached_oid_set.count(oid.id) == 0) {
                slim_oids.push_back(oid);
            }
        }

        // Translate to row indices in active FM
        std::vector<uint64_t> slim_rows;
        slim_rows.reserve(slim_oids.size());
        for (const auto& oid : slim_oids) {
            auto row = active_rm->find(oid);
            if (row.has_value()) {
                slim_rows.push_back(*row);
            }
        }

        // Write v2 packed file: header + OID table + features
        char fname[32];
        std::snprintf(fname, sizeof(fname), "batch_%06lu.bin",
                      static_cast<unsigned long>(b));
        auto batch_path = packed_slim_dir / fname;

        auto header = PackedBatchHeader::make_v2(slim_oids.size(), D, dt);

        int fd = ::open(batch_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            throw std::runtime_error(
                "FourLevelStore::build: cannot create " + batch_path.string() +
                ": " + safe_strerror(errno));
        }
        FdGuard guard(fd);

        // Write header
        write_all(fd, &header, sizeof(header), batch_path.string());

        // Write ObjectId table
        for (const auto& oid : slim_oids) {
            uint64_t id = oid.id;
            write_all(fd, &id, sizeof(id), batch_path.string());
        }

        // Write feature data
        std::vector<char> buf(row_bytes);
        for (auto row : slim_rows) {
            std::memcpy(buf.data(), active_fm->row(row), row_bytes);
            write_all(fd, buf.data(), row_bytes, batch_path.string());
        }

        if (::fsync(fd) < 0) {
            throw std::runtime_error(
                "FourLevelStore::build: fsync failed: " + safe_strerror(errno));
        }
    }

    // --- Step 6: Write metadata ---
    auto meta = StoreMetaHeader::make(
        result.l1_nodes, result.l2_nodes, result.l3_nodes, result.l4_nodes,
        D, dt, gpu_available, packed_slim_dir.string());

    {
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

    } catch (...) {
        // Best-effort cleanup of partial outputs
        std::error_code ec;
        fs::remove_all(packed_slim_dir, ec);
        fs::remove(gpu_cache_path, ec);
        fs::remove(cpu_cache_path, ec);
        fs::remove(meta_path, ec);
        if (config.reorder) {
            fs::remove(reordered_fmat, ec);
            fs::remove(reordered_rmap, ec);
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
                continue;
            }
        }

        // Fallback to L3 (reordered FeatureMatrix)
        if (reordered_rm_.has_value()) {
            auto row = reordered_rm_->find(oid);
            if (row.has_value()) {
                if (l3_reader_) {
                    // DirectIoReader path: read single row via O_DIRECT
                    std::vector<uint64_t> single_row = {*row};
                    auto result = l3_reader_->read_rows(single_row, row_bytes, l3_header_size_);
                    std::memcpy(out_ptr + i * row_bytes, result.data.get(), row_bytes);
                } else if (l3_mmap_fb_.has_value()) {
                    // Mmap fallback path
                    std::memcpy(out_ptr + i * row_bytes,
                                l3_mmap_fb_->row(*row),
                                row_bytes);
                }
                stats_.l3_reads++;
                continue;
            }
        }

        // Node not found in any level; leave as zeros
        stats_.l3_reads++;
    }

    return output;
}

// =============================================================================
// load_batch_features() — The Runtime Hot Path (all 4 levels)
// =============================================================================

torch::Tensor FourLevelStore::load_batch_features(uint64_t batch_id) {
    auto sample = samples_->read_sample(batch_id);
    const auto& oids = sample.all_unique_nodes;
    uint64_t total = oids.size();

    if (total == 0) {
        return torch::empty(
            {0, static_cast<int64_t>(feature_dim_)},
            torch::TensorOptions().dtype(to_torch_dtype(dtype_)));
    }

    stats_.total_requests += total;

    size_t row_bytes = feature_dim_ * elem_size_;

    // Step 1: Read L4 packed_slim for this batch to get OID table
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
            PackedBatchHeader hdr{};
            read_all(fd, &hdr, sizeof(hdr), slim_path.string());

            if (hdr.is_valid() && hdr.has_oid_table()) {
                slim_nodes = hdr.num_nodes;

                // Read OID table
                std::vector<uint64_t> oid_table(slim_nodes);
                if (slim_nodes > 0) {
                    read_all(fd, oid_table.data(),
                             slim_nodes * sizeof(uint64_t),
                             slim_path.string());
                }
                for (uint32_t j = 0; j < slim_nodes; ++j) {
                    slim_oid_to_idx[oid_table[j]] = j;
                }

                // Read feature data
                size_t data_bytes = hdr.data_bytes();
                if (data_bytes > 0) {
                    slim_data.resize(data_bytes);
                    read_all(fd, slim_data.data(), data_bytes,
                             slim_path.string());
                }
            }
        }
    }

    // Step 2: Partition nodes into L1/L2/L3/L4 buckets.
    // Each bucket stores (output_position, source_data) for later assembly.
    std::vector<uint32_t> l1_input_positions;    // positions in oids[] for L1 lookup
    std::vector<ObjectId> l1_input_oids;

    std::vector<uint32_t> l2_positions;          // output positions resolved from L2
    std::vector<uint32_t> l3_positions;          // output positions resolved from L3
    std::vector<uint64_t> l3_row_indices;        // corresponding row indices in reordered FM

    std::vector<uint32_t> l4_positions;          // output positions resolved from L4
    std::vector<uint32_t> l4_slim_indices;       // index into slim_data for each L4 node

    for (uint32_t i = 0; i < total; ++i) {
        const auto& oid = oids[i];

        // Try L1 (GPU cache)
        if (gpu_cache_ && gpu_cache_->contains(oid)) {
            l1_input_positions.push_back(i);
            l1_input_oids.push_back(oid);
            stats_.l1_hits++;
            continue;
        }

        // Try L2 (CPU cache)
        if (cpu_cache_ && cpu_cache_->contains(oid)) {
            l2_positions.push_back(i);
            stats_.l2_hits++;
            continue;
        }

        // Try L4 (slim file data -- check first since slim has exact data)
        auto slim_it = slim_oid_to_idx.find(oid.id);
        if (slim_it != slim_oid_to_idx.end()) {
            uint32_t idx = slim_it->second;
            size_t offset = static_cast<size_t>(idx) * row_bytes;
            if (offset + row_bytes <= slim_data.size()) {
                l4_positions.push_back(i);
                l4_slim_indices.push_back(idx);
                stats_.l4_reads++;
                continue;
            }
        }

        // Fallback to L3 (reordered FeatureMatrix)
        if (reordered_rm_.has_value()) {
            auto row = reordered_rm_->find(oid);
            if (row.has_value()) {
                l3_positions.push_back(i);
                l3_row_indices.push_back(*row);
                stats_.l3_reads++;
                continue;
            }
        }

        // Node not resolved -- leave as zeros, count as L3 miss
        stats_.l3_reads++;
    }

    // Step 3: Batch-read L3 rows via DirectIoReader (zero page cache)
    // or mmap fallback. Result is a contiguous buffer of l3_row_indices.size() rows.
    std::vector<char> l3_buf;
    if (!l3_row_indices.empty()) {
        if (l3_reader_) {
            auto result = l3_reader_->read_rows(l3_row_indices, row_bytes, l3_header_size_);
            l3_buf.assign(result.data.get(), result.data.get() + result.size);
        } else if (l3_mmap_fb_.has_value()) {
            l3_buf.resize(l3_row_indices.size() * row_bytes);
            l3_mmap_fb_->extract_rows(l3_row_indices, l3_buf.data());
        }
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

        // L1: batch lookup returns [K1_hits, D] tensor on GPU
        torch::Tensor gpu_features;
        std::vector<uint32_t> gpu_positions;
        if (!l1_input_oids.empty()) {
            auto lr = gpu_cache_->lookup(l1_input_oids);
            gpu_features = lr.features;
            // Map hit_positions (indices into l1_input_oids) back to output positions
            gpu_positions.reserve(lr.hit_positions.size());
            for (auto hp : lr.hit_positions) {
                gpu_positions.push_back(l1_input_positions[hp]);
            }
        }

        // Combine L2 + L3 + L4 CPU features into one contiguous float buffer
        std::vector<float> cpu_combined;
        std::vector<uint32_t> cpu_combined_positions;

        size_t cpu_total = l2_positions.size() + l3_positions.size() + l4_positions.size();
        cpu_combined.reserve(cpu_total * feature_dim_);
        cpu_combined_positions.reserve(cpu_total);

        // L2 features
        if (!l2_positions.empty()) {
            std::vector<ObjectId> l2_oids;
            l2_oids.reserve(l2_positions.size());
            for (auto pos : l2_positions) l2_oids.push_back(oids[pos]);
            auto lr = cpu_cache_->lookup(l2_oids);
            const float* l2_data = reinterpret_cast<const float*>(lr.features.data());
            for (size_t h = 0; h < lr.hit_positions.size(); ++h) {
                cpu_combined_positions.push_back(l2_positions[lr.hit_positions[h]]);
                cpu_combined.insert(cpu_combined.end(),
                                    l2_data + h * feature_dim_,
                                    l2_data + (h + 1) * feature_dim_);
            }
        }

        // L3 features (already in l3_buf, in order of l3_row_indices)
        if (!l3_positions.empty() && !l3_buf.empty()) {
            const float* l3_data = reinterpret_cast<const float*>(l3_buf.data());
            for (size_t j = 0; j < l3_positions.size(); ++j) {
                cpu_combined_positions.push_back(l3_positions[j]);
                cpu_combined.insert(cpu_combined.end(),
                                    l3_data + j * feature_dim_,
                                    l3_data + (j + 1) * feature_dim_);
            }
        }

        // L4 features (from slim_data)
        if (!l4_positions.empty()) {
            const float* slim_float = reinterpret_cast<const float*>(slim_data.data());
            for (size_t j = 0; j < l4_positions.size(); ++j) {
                cpu_combined_positions.push_back(l4_positions[j]);
                uint32_t idx = l4_slim_indices[j];
                cpu_combined.insert(cpu_combined.end(),
                                    slim_float + idx * feature_dim_,
                                    slim_float + (idx + 1) * feature_dim_);
            }
        }

        // C2: Pin the CPU combined buffer for UVA access from the CUDA
        // assembler kernel.  Unpinned heap memory forces the GPU to read
        // through UVA at ~120 MB/s; pinned memory achieves ~12 GB/s.
        const float* assembler_data = cpu_combined.data();
        float* pinned_buf = nullptr;
#ifdef GNN_CUDA_ENABLED
        size_t cpu_combined_bytes = cpu_combined.size() * sizeof(float);
        if (cpu_combined_bytes > 0) {
            if (cudaHostAlloc(reinterpret_cast<void**>(&pinned_buf),
                              cpu_combined_bytes,
                              cudaHostAllocDefault) == cudaSuccess) {
                std::memcpy(pinned_buf, cpu_combined.data(), cpu_combined_bytes);
                assembler_data = pinned_buf;
            } else {
                pinned_buf = nullptr;  // fallback to unpinned
            }
        }
#endif

        auto result_tensor = assembler_->assemble(
            static_cast<int64_t>(total),
            gpu_features, gpu_positions,
            assembler_data,
            static_cast<int64_t>(cpu_combined_positions.size()),
            cpu_combined_positions
        );

#ifdef GNN_CUDA_ENABLED
        if (pinned_buf) {
            cudaFreeHost(pinned_buf);
        }
#endif
        return result_tensor;
    }

    // --- CPU-only assembly path (no GPU, or non-float32 dtype) ---
    auto output = torch::zeros(
        {static_cast<int64_t>(total), static_cast<int64_t>(feature_dim_)},
        torch::TensorOptions().dtype(to_torch_dtype(dtype_)));
    char* out_ptr = static_cast<char*>(output.data_ptr());

    // L1 features (GPU cache -> CPU copy, one-by-one)
    for (size_t k = 0; k < l1_input_oids.size(); ++k) {
        auto lr = gpu_cache_->lookup({l1_input_oids[k]});
        if (!lr.hit_positions.empty()) {
            auto cpu_feats = lr.features.cpu().contiguous();
            std::memcpy(out_ptr + l1_input_positions[k] * row_bytes,
                        cpu_feats.data_ptr(), row_bytes);
        }
    }

    // L2 features
    for (auto pos : l2_positions) {
        auto lr = cpu_cache_->lookup({oids[pos]});
        if (!lr.hit_positions.empty()) {
            std::memcpy(out_ptr + pos * row_bytes,
                        lr.features.data(), row_bytes);
        }
    }

    // L3 features (from batched read in l3_buf)
    for (size_t j = 0; j < l3_positions.size(); ++j) {
        std::memcpy(out_ptr + l3_positions[j] * row_bytes,
                    l3_buf.data() + j * row_bytes, row_bytes);
    }

    // L4 features (from slim_data)
    for (size_t j = 0; j < l4_positions.size(); ++j) {
        uint32_t idx = l4_slim_indices[j];
        size_t offset = static_cast<size_t>(idx) * row_bytes;
        std::memcpy(out_ptr + l4_positions[j] * row_bytes,
                    slim_data.data() + offset, row_bytes);
    }

    return output;
}

} // namespace mdb::gnn
