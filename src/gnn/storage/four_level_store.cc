#include "gnn/storage/four_level_store.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <fcntl.h>

#include "gnn/common/posix_io.h"
#include "gnn/storage/cache_file.h"
#include "gnn/storage/feature_matrix.h"
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

    // Load L3 (mmap fallback for now, DirectIoReader added in Task 14)
    if (fs::exists(reord_fmat)) {
        l3_fm_.emplace(FeatureMatrix::open(reord_fmat));
    }
    if (fs::exists(reord_rmap)) {
        reordered_rm_.emplace(RowMapping::open(reord_rmap));
    }
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

        // Fallback to L3 (reordered FeatureMatrix via mmap)
        if (l3_fm_.has_value() && reordered_rm_.has_value()) {
            auto row = reordered_rm_->find(oid);
            if (row.has_value()) {
                std::memcpy(out_ptr + i * row_bytes,
                            l3_fm_->row(*row),
                            row_bytes);
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

    // Step 2: Allocate output tensor [total, D] on CPU
    auto output = torch::zeros(
        {static_cast<int64_t>(total), static_cast<int64_t>(feature_dim_)},
        torch::TensorOptions().dtype(to_torch_dtype(dtype_)));
    char* out_ptr = static_cast<char*>(output.data_ptr());

    // Step 3: Partition and fill features from each level
    for (uint32_t i = 0; i < total; ++i) {
        const auto& oid = oids[i];

        // Try L1 (GPU cache)
        if (gpu_cache_ && gpu_cache_->contains(oid)) {
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

        // Try L4 (slim file data — check first since slim has exact data)
        auto slim_it = slim_oid_to_idx.find(oid.id);
        if (slim_it != slim_oid_to_idx.end()) {
            uint32_t idx = slim_it->second;
            size_t offset = static_cast<size_t>(idx) * row_bytes;
            if (offset + row_bytes <= slim_data.size()) {
                std::memcpy(out_ptr + i * row_bytes,
                            slim_data.data() + offset,
                            row_bytes);
                stats_.l4_reads++;
                continue;
            }
        }

        // Fallback to L3 (reordered FeatureMatrix)
        if (l3_fm_.has_value() && reordered_rm_.has_value()) {
            auto row = reordered_rm_->find(oid);
            if (row.has_value()) {
                std::memcpy(out_ptr + i * row_bytes,
                            l3_fm_->row(*row),
                            row_bytes);
                stats_.l3_reads++;
                continue;
            }
        }

        // Node not resolved — leave as zeros, count as L3 miss
        stats_.l3_reads++;
    }

    return output;
}

} // namespace mdb::gnn
