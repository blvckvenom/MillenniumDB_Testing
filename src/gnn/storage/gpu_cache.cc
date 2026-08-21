#include "gnn/storage/gpu_cache.h"

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "gnn/storage/cache_file.h"
#include "gnn/storage/gnn_dtype.h"
#include "misc/ablation_registry.h"

namespace mdb::gnn {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// RAII helpers (same pattern as FeatureMatrix / RowMapping)
// ---------------------------------------------------------------------------
namespace {

class FdGuard {
public:
    explicit FdGuard(int fd) : fd_(fd) {}
    ~FdGuard() { if (fd_ >= 0) ::close(fd_); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
    int get() const { return fd_; }
private:
    int fd_;
};

void write_all(int fd, const void* buf, size_t count) {
    const char* p = static_cast<const char*>(buf);
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t written = ::write(fd, p, remaining);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "GpuCache::build: write failed: " + std::string(std::strerror(errno)));
        }
        if (written == 0) {
            throw std::runtime_error(
                "GpuCache::build: write returned 0 — disk full or I/O error");
        }
        p += written;
        remaining -= static_cast<size_t>(written);
    }
}

void read_all(int fd, void* buf, size_t count) {
    char* p = static_cast<char*>(buf);
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t n = ::read(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "GpuCache: read failed: " + std::string(std::strerror(errno)));
        }
        if (n == 0) {
            throw std::runtime_error(
                "GpuCache: unexpected EOF during read");
        }
        p += n;
        remaining -= static_cast<size_t>(n);
    }
}

/// Map GnnDtype to torch::ScalarType.
torch::ScalarType to_torch_dtype(GnnDtype dt) {
    switch (dt) {
        case GnnDtype::FLOAT32: return torch::kFloat32;
        case GnnDtype::FLOAT64: return torch::kFloat64;
        case GnnDtype::INT32:   return torch::kInt32;
        case GnnDtype::INT64:   return torch::kInt64;
        case GnnDtype::UINT8:   return torch::kUInt8;
        case GnnDtype::BOOL:    return torch::kBool;
    }
    throw std::invalid_argument("GpuCache: unknown GnnDtype " +
                                std::to_string(static_cast<int>(dt)));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// build() — write GNNC file
// ---------------------------------------------------------------------------

// Skip rebuild if the existing cache file's header already matches the
// requested node count, feature dimension, and dtype. Same pattern as cpu_cache.cc.
static bool gpu_cache_matches(
    const fs::path& path,
    uint64_t        expected_N,
    uint64_t        expected_D,
    GnnDtype        expected_dtype)
{
    if (!fs::exists(path)) return false;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    CacheFileHeader h{};
    ssize_t r = ::read(fd, &h, sizeof(h));
    ::close(fd);
    if (r != static_cast<ssize_t>(sizeof(h))) return false;
    if (!h.is_valid()) return false;
    return h.num_nodes == expected_N
        && h.feature_dim == expected_D
        && h.get_dtype() == expected_dtype;
}

void GpuCache::build(
    const std::vector<ObjectId>& nodes,
    const FeatureMatrix&         features,
    const RowMapping&            row_mapping,
    const fs::path&              output_path)
{
    const uint64_t N = nodes.size();
    const uint64_t D = features.num_cols();
    const GnnDtype dt = features.dtype();

    if (gpu_cache_matches(output_path, N, D, dt)) {
        // Existing cache header matches the requested dims → reusable. Skip rebuild.
        return;
    }

    auto header = CacheFileHeader::make(N, D, dt);
    const size_t row_bytes = features.row_bytes();

    // Resolve OID -> row and sort by row to make the source mmap reads
    // sequential. The on-disk index is reordered but oid_table[i] still
    // maps to data[i] so the cache reader is semantically unchanged.
    struct Entry { uint64_t row; ObjectId oid; };
    std::vector<Entry> entries;
    entries.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        auto row_idx = row_mapping.find(nodes[i]);
        if (!row_idx.has_value()) {
            char hex[17];
            std::snprintf(hex, sizeof(hex), "%016llx",
                          static_cast<unsigned long long>(nodes[i].id));
            throw std::runtime_error(
                std::string("GpuCache::build: ObjectId 0x") + hex +
                " has no corresponding row in RowMapping");
        }
        entries.push_back({*row_idx, nodes[i]});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.row < b.row; });

    // Single contiguous output buffer: header + OID table + features.
    const size_t oid_block  = N * sizeof(uint64_t);
    const size_t data_block = N * row_bytes;
    std::vector<char> out_buf(sizeof(header) + oid_block + data_block);
    std::memcpy(out_buf.data(), &header, sizeof(header));
    auto* oid_ptr  = reinterpret_cast<uint64_t*>(out_buf.data() + sizeof(header));
    char*  feat_ptr = out_buf.data() + sizeof(header) + oid_block;
    // Optional multi-threaded gather (env MDB_GNN_CACHE_WORKERS, default 1 = sequential):
    // each row scatter writes to a disjoint output slot (oid_ptr[i], feat_ptr[i*..])
    // and reads from the const mmap source via features.row(), so splitting the
    // i-range across threads is both thread-safe and bit-identical.
    {
        // Shares one resolution with the CPU cache builder: same switch, same
        // answer, so an L1/L2 build cannot end up half parallel and half not.
        static const long env_cache_workers = Ablation::number("MDB_GNN_CACHE_WORKERS", 1);
        unsigned cache_workers = 1;
        if (env_cache_workers > 1) cache_workers = static_cast<unsigned>(env_cache_workers);
        const unsigned hw = std::thread::hardware_concurrency();
        if (hw > 0 && cache_workers > hw) cache_workers = hw;
        auto gather_range = [&](uint64_t lo, uint64_t hi) {
            for (uint64_t i = lo; i < hi; ++i) {
                oid_ptr[i] = entries[i].oid.id;
                std::memcpy(feat_ptr + i * row_bytes,
                            features.row(entries[i].row), row_bytes);
            }
        };
        if (cache_workers <= 1 || N < cache_workers) {
            gather_range(0, N);
        } else {
            std::vector<std::thread> gw;
            const uint64_t chunk = (N + cache_workers - 1) / cache_workers;
            for (unsigned w = 0; w < cache_workers; ++w) {
                const uint64_t lo = static_cast<uint64_t>(w) * chunk;
                const uint64_t hi = std::min<uint64_t>(lo + chunk, N);
                if (lo >= hi) break;
                gw.emplace_back(gather_range, lo, hi);
            }
            for (auto& t : gw) t.join();
        }
    }

    int fd = ::open(output_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "GpuCache::build: cannot open " + output_path.string() +
            ": " + std::strerror(errno));
    }
    FdGuard guard(fd);
    write_all(fd, out_buf.data(), out_buf.size());

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "GpuCache::build: fsync failed: " + std::string(std::strerror(errno)));
    }

    // Best-effort parent directory fsync for crash consistency
    {
        int dir_fd = ::open(output_path.parent_path().c_str(), O_RDONLY);
        if (dir_fd >= 0) {
            ::fsync(dir_fd);
            ::close(dir_fd);
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor — load GNNC file to GPU (or CPU fallback)
// ---------------------------------------------------------------------------

GpuCache::GpuCache(const fs::path& cache_file) {
    if (!fs::exists(cache_file)) {
        throw std::runtime_error(
            "GpuCache: file not found: " + cache_file.string());
    }

    auto file_size = fs::file_size(cache_file);
    if (file_size < CacheFileHeader::SIZE) {
        throw std::runtime_error(
            "GpuCache: file too small for header (" +
            std::to_string(file_size) + " bytes): " + cache_file.string());
    }

    int fd = ::open(cache_file.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "GpuCache: cannot open " + cache_file.string() +
            ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    // Read and validate header
    CacheFileHeader header;
    read_all(fd, &header, sizeof(header));

    // For empty caches (0 nodes), header.is_valid() still requires feature_dim > 0,
    // which build() guarantees (it writes features.num_cols() even when N=0).
    if (!header.is_valid()) {
        throw std::runtime_error(
            "GpuCache: invalid GNNC header in " + cache_file.string());
    }

    num_nodes_   = header.num_nodes;
    feature_dim_ = header.feature_dim;

    // Empty cache: no data to load
    if (num_nodes_ == 0) {
        features_ = torch::empty({0, 0});
        on_gpu_ = false;
        return;
    }

    // Validate file size against header
    size_t expected_size = header.total_file_size();
    if (file_size < expected_size) {
        throw std::runtime_error(
            "GpuCache: file truncated — expected " +
            std::to_string(expected_size) + " bytes, got " +
            std::to_string(file_size) + ": " + cache_file.string());
    }

    // Read ObjectId table -> build oid_to_idx_ HashMap
    std::vector<uint64_t> oid_table(num_nodes_);
    read_all(fd, oid_table.data(), num_nodes_ * sizeof(uint64_t));

    oid_to_idx_.reserve(num_nodes_);
    for (uint32_t i = 0; i < num_nodes_; ++i) {
        oid_to_idx_[oid_table[i]] = i;
    }

    // Read feature data into CPU buffer
    size_t data_bytes = header.data_bytes();
    std::vector<char> cpu_buffer(data_bytes);
    read_all(fd, cpu_buffer.data(), data_bytes);

    // Convert to torch::Tensor
    auto torch_dtype = to_torch_dtype(header.get_dtype());

    // from_blob does NOT own cpu_buffer. On the CUDA path we skip the clone():
    // .to(kCUDA) copies the data into GPU storage (owned by the result) while
    // cpu_buffer is still alive, so the extra full-size host clone() (a second
    // data_bytes heap copy at cache load) is pure waste. Only the CPU-only path
    // needs an owning clone() to outlive cpu_buffer's scope.
    auto cpu_view = torch::from_blob(
        cpu_buffer.data(),
        {static_cast<int64_t>(num_nodes_), static_cast<int64_t>(feature_dim_)},
        torch_dtype
    );

    if (torch::cuda::is_available()) {
        features_ = cpu_view.to(torch::kCUDA);
        on_gpu_ = true;
    } else {
        features_ = cpu_view.clone();
        on_gpu_ = false;
    }
}

// ---------------------------------------------------------------------------
// lookup()
// ---------------------------------------------------------------------------

GpuCache::LookupResult GpuCache::lookup(const std::vector<ObjectId>& oids) const {
    LookupResult result;
    std::vector<int64_t> cache_indices; // indices into features_ tensor

    for (uint32_t i = 0; i < static_cast<uint32_t>(oids.size()); ++i) {
        auto it = oid_to_idx_.find(oids[i].id);
        if (it != oid_to_idx_.end()) {
            result.hit_positions.push_back(i);
            cache_indices.push_back(static_cast<int64_t>(it->second));
        } else {
            result.miss_positions.push_back(i);
        }
    }

    if (cache_indices.empty()) {
        // No hits: return empty tensor on the same device as features_
        auto device = (features_.defined() && features_.numel() > 0)
                          ? features_.device()
                          : torch::Device(torch::kCPU);
        auto stype = (features_.defined() && features_.numel() > 0)
                         ? features_.scalar_type()
                         : torch::kFloat32;
        result.features = torch::empty(
            {0, static_cast<int64_t>(feature_dim_)},
            torch::TensorOptions().dtype(stype).device(device));
    } else {
        // Gather rows by index using index_select
        auto idx_tensor = torch::tensor(cache_indices, torch::kInt64);
        if (on_gpu_) {
            idx_tensor = idx_tensor.to(torch::kCUDA);
        }
        result.features = features_.index_select(0, idx_tensor);
    }

    return result;
}

// ---------------------------------------------------------------------------
// find_index() / gather_by_indices()
// ---------------------------------------------------------------------------
//
// Split the work of lookup() into a single-hash find_index() (used in the
// Four-Level Feature Store classification loop) and a gather_by_indices()
// that takes pre-validated cache row indices and returns a feature tensor
// on the cache device. Together these replace the previous double-hash
// pattern (contains() then lookup()) with one hash lookup per L1 hit.

std::optional<uint32_t> GpuCache::find_index(ObjectId oid) const {
    auto it = oid_to_idx_.find(oid.id);
    if (it == oid_to_idx_.end()) return std::nullopt;
    return it->second;
}

torch::Tensor GpuCache::gather_by_indices(
    const std::vector<uint32_t>& cache_indices) const
{
    if (cache_indices.empty()) {
        // Return an empty tensor on the same device/dtype as features_.
        auto device = (features_.defined() && features_.numel() > 0)
                          ? features_.device()
                          : torch::Device(torch::kCPU);
        auto stype = (features_.defined() && features_.numel() > 0)
                         ? features_.scalar_type()
                         : torch::kFloat32;
        return torch::empty(
            {0, static_cast<int64_t>(feature_dim_)},
            torch::TensorOptions().dtype(stype).device(device));
    }

    // Build int64 index tensor for index_select. Same body as the back
    // half of lookup() — no find loop needed.
    std::vector<int64_t> idx64;
    idx64.reserve(cache_indices.size());
    for (uint32_t i : cache_indices) idx64.push_back(static_cast<int64_t>(i));

    auto idx_tensor = torch::tensor(idx64, torch::kInt64);
    if (on_gpu_) {
        idx_tensor = idx_tensor.to(torch::kCUDA);
    }
    return features_.index_select(0, idx_tensor);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

bool GpuCache::contains(ObjectId oid) const {
    return oid_to_idx_.count(oid.id) > 0;
}

uint64_t GpuCache::num_nodes() const {
    return num_nodes_;
}

uint64_t GpuCache::feature_dim() const {
    return feature_dim_;
}

size_t GpuCache::memory_bytes() const {
    if (!features_.defined() || features_.numel() == 0) {
        return 0;
    }
    return static_cast<size_t>(features_.nbytes());
}

bool GpuCache::is_on_gpu() const {
    return on_gpu_;
}

} // namespace mdb::gnn
