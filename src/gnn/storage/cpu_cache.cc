#include "gnn/storage/cpu_cache.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "gnn/common/posix_io.h"
#include "gnn/storage/cache_file.h"
#include "gnn/storage/gnn_dtype.h"

#ifdef GNN_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace mdb::gnn {

namespace fs = std::filesystem;

// =============================================================================
// build() — write GNNC file from selected nodes
// =============================================================================

// Fix #14 (2026-05-13): skip rebuild if an existing cache file's header
// already matches the requested (num_nodes, feature_dim, dtype). Cuts
// 19 min off papers100M L2 build on idempotent re-runs.
static bool cache_header_matches(
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

void CpuCache::build(
    const std::vector<ObjectId>& nodes,
    const FeatureMatrix&         features,
    const RowMapping&            row_mapping,
    const fs::path&              output_path)
{
    auto dtype = features.dtype();
    uint64_t D = features.num_cols();
    uint64_t N = nodes.size();
    size_t elem = dtype_size(dtype);
    size_t row_bytes = D * elem;

    if (cache_header_matches(output_path, N, D, dtype)) {
        // Fix #14: header matches expected dims → existing cache is
        // reusable. Skipping the rebuild avoids re-reading the source
        // FM (random row access, expensive on cold cache).
        return;
    }

    auto header = CacheFileHeader::make(N, D, dtype);

    // Resolve OID -> row before opening the output file so any error
    // surfaces without leaving a partial file behind. While we're at
    // it, sort by row_index to convert random source reads into
    // sequential ones (same trick as Fix #12 in create_reordered).
    // The (oid, row) pairing on disk is preserved — the on-disk index
    // is arbitrary as long as oid_table[i] still maps to data[i].
    struct Entry { uint64_t row; ObjectId oid; };
    std::vector<Entry> entries;
    entries.reserve(N);
    for (const auto& oid : nodes) {
        auto row = row_mapping.find(oid);
        if (!row.has_value()) {
            throw std::runtime_error(
                "CpuCache::build: ObjectId " + std::to_string(oid.id) +
                " not in RowMapping");
        }
        entries.push_back({*row, oid});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.row < b.row; });

    // Single contiguous output buffer: header + OID table + features.
    // Replaces 2N+1 small write_all calls with one write_all per cache.
    const size_t oid_block  = N * sizeof(uint64_t);
    const size_t data_block = N * row_bytes;
    std::vector<char> out_buf(sizeof(header) + oid_block + data_block);
    std::memcpy(out_buf.data(), &header, sizeof(header));
    auto* oid_ptr  = reinterpret_cast<uint64_t*>(out_buf.data() + sizeof(header));
    char*  feat_ptr = out_buf.data() + sizeof(header) + oid_block;
    for (uint64_t i = 0; i < N; ++i) {
        oid_ptr[i] = entries[i].oid.id;
        std::memcpy(feat_ptr + i * row_bytes,
                    features.row(entries[i].row), row_bytes);
    }

    int fd = ::open(output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "CpuCache::build: cannot create " + output_path.string() +
            ": " + safe_strerror(errno));
    }
    FdGuard guard(fd);
    write_all(fd, out_buf.data(), out_buf.size(), output_path.string());

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "CpuCache::build: fsync failed: " + safe_strerror(errno));
    }
    fsync_directory(output_path);
}

// =============================================================================
// Constructor — load GNNC file into CPU memory
// =============================================================================

CpuCache::CpuCache(const fs::path& cache_file) {
    if (!fs::exists(cache_file)) {
        throw std::runtime_error(
            "CpuCache: file not found: " + cache_file.string());
    }

    auto file_size = fs::file_size(cache_file);
    if (file_size < CacheFileHeader::SIZE) {
        throw std::runtime_error(
            "CpuCache: file too small for header (" +
            std::to_string(file_size) + " bytes): " + cache_file.string());
    }

    int fd = ::open(cache_file.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "CpuCache: cannot open " + cache_file.string() +
            ": " + safe_strerror(errno));
    }
    FdGuard guard(fd);

    // Read and validate header
    CacheFileHeader header{};
    read_all(fd, &header, sizeof(header), cache_file.string());

    if (!header.is_valid()) {
        throw std::runtime_error(
            "CpuCache: invalid header in " + cache_file.string());
    }

    num_nodes_   = header.num_nodes;
    feature_dim_ = header.feature_dim;
    elem_size_   = static_cast<uint8_t>(dtype_size(header.get_dtype()));

    // Validate file size matches header expectations
    size_t expected_size = header.total_file_size();
    if (file_size < expected_size) {
        throw std::runtime_error(
            "CpuCache: file truncated — expected " +
            std::to_string(expected_size) + " bytes, got " +
            std::to_string(file_size) + ": " + cache_file.string());
    }

    // Read ObjectId table -> build HashMap
    oid_to_idx_.reserve(num_nodes_);
    for (uint32_t i = 0; i < num_nodes_; ++i) {
        uint64_t oid_id;
        read_all(fd, &oid_id, sizeof(oid_id), cache_file.string());
        oid_to_idx_[oid_id] = i;
    }

    // Allocate feature data
    feature_data_size_ = header.data_bytes();

    if (feature_data_size_ == 0) {
        // Empty cache: no allocation needed
        feature_data_ = nullptr;
        pinned_ = false;
        return;
    }

    // Try pinned memory first (for UVA), fall back to malloc
#ifdef GNN_CUDA_ENABLED
    if (cudaHostAlloc(&feature_data_, feature_data_size_, cudaHostAllocDefault) == cudaSuccess) {
        pinned_ = true;
    } else
#endif
    {
        feature_data_ = std::malloc(feature_data_size_);
        if (feature_data_ == nullptr) {
            throw std::runtime_error(
                "CpuCache: malloc failed for " +
                std::to_string(feature_data_size_) + " bytes");
        }
        pinned_ = false;
    }

    // Read feature data into allocated buffer
    read_all(fd, feature_data_, feature_data_size_, cache_file.string());
}

// =============================================================================
// Destructor
// =============================================================================

CpuCache::~CpuCache() {
    free_data();
}

void CpuCache::free_data() {
    if (feature_data_ == nullptr) return;

#ifdef GNN_CUDA_ENABLED
    if (pinned_) {
        cudaFreeHost(feature_data_);
    } else
#endif
    {
        std::free(feature_data_);
    }
    feature_data_ = nullptr;
    feature_data_size_ = 0;
}

// =============================================================================
// Move semantics
// =============================================================================

CpuCache::CpuCache(CpuCache&& other) noexcept
    : feature_data_(other.feature_data_),
      feature_data_size_(other.feature_data_size_),
      oid_to_idx_(std::move(other.oid_to_idx_)),
      feature_dim_(other.feature_dim_),
      num_nodes_(other.num_nodes_),
      elem_size_(other.elem_size_),
      pinned_(other.pinned_)
{
    other.feature_data_      = nullptr;
    other.feature_data_size_ = 0;
    other.feature_dim_       = 0;
    other.num_nodes_         = 0;
    other.elem_size_         = 0;
    other.pinned_            = false;
}

CpuCache& CpuCache::operator=(CpuCache&& other) noexcept {
    if (this != &other) {
        free_data();
        feature_data_      = other.feature_data_;
        feature_data_size_ = other.feature_data_size_;
        oid_to_idx_        = std::move(other.oid_to_idx_);
        feature_dim_       = other.feature_dim_;
        num_nodes_         = other.num_nodes_;
        elem_size_         = other.elem_size_;
        pinned_            = other.pinned_;

        other.feature_data_      = nullptr;
        other.feature_data_size_ = 0;
        other.feature_dim_       = 0;
        other.num_nodes_         = 0;
        other.elem_size_         = 0;
        other.pinned_            = false;
    }
    return *this;
}

// =============================================================================
// lookup()
// =============================================================================

CpuCache::LookupResult CpuCache::lookup(const std::vector<ObjectId>& oids) const {
    LookupResult result;
    result.feature_dim = feature_dim_;
    result.elem_size   = elem_size_;

    size_t row_bytes = feature_dim_ * elem_size_;

    // Pre-scan to count hits for reservation
    for (uint32_t i = 0; i < static_cast<uint32_t>(oids.size()); ++i) {
        auto it = oid_to_idx_.find(oids[i].id);
        if (it != oid_to_idx_.end()) {
            result.hit_positions.push_back(i);
        } else {
            result.miss_positions.push_back(i);
        }
    }

    // Allocate and fill feature data for hits
    if (!result.hit_positions.empty() && row_bytes > 0) {
        result.features.resize(result.hit_positions.size() * row_bytes);
        char* out = result.features.data();
        const char* base = static_cast<const char*>(feature_data_);

        for (uint32_t pos : result.hit_positions) {
            auto it = oid_to_idx_.find(oids[pos].id);
            uint32_t idx = it->second;
            std::memcpy(out, base + static_cast<size_t>(idx) * row_bytes, row_bytes);
            out += row_bytes;
        }
    }

    return result;
}

// =============================================================================
// lookup_uva() — zero-copy pointer-based lookup
// =============================================================================
//
// Round 1B (2026-05-15): unlike lookup(), this does not allocate or memcpy
// row bytes. Each hit returns a const void* into the already-pinned
// feature_data_ region. The pointer is valid for the lifetime of this
// CpuCache (or until the cache is moved-from). Callers either memcpy into
// their own destination or pass the pointer to a CUDA kernel for UVA reads.

CpuCache::UvaLookupResult CpuCache::lookup_uva(const std::vector<ObjectId>& oids) const {
    UvaLookupResult result;
    result.hit_positions.reserve(oids.size());
    result.hit_pointers.reserve(oids.size());

    size_t row_bytes = feature_dim_ * elem_size_;
    const char* base = static_cast<const char*>(feature_data_);

    for (uint32_t i = 0; i < static_cast<uint32_t>(oids.size()); ++i) {
        auto it = oid_to_idx_.find(oids[i].id);
        if (it != oid_to_idx_.end()) {
            result.hit_positions.push_back(i);
            result.hit_pointers.push_back(
                base + static_cast<size_t>(it->second) * row_bytes);
        } else {
            result.miss_positions.push_back(i);
        }
    }
    return result;
}

// =============================================================================
// find_index() / row_ptr() — Round 1C
// =============================================================================
//
// Round 1C (2026-05-15): split the work of lookup_uva() into a single-hash
// find_index() (used in the FourLevelStore classification loop) and a
// row_ptr() accessor that returns the UVA pointer to the row at a
// pre-validated cache index. Together these replace the previous
// double-hash pattern (contains() then lookup_uva()) with one hash per
// L2 hit.

std::optional<uint32_t> CpuCache::find_index(ObjectId oid) const {
    auto it = oid_to_idx_.find(oid.id);
    if (it == oid_to_idx_.end()) return std::nullopt;
    return it->second;
}

const void* CpuCache::row_ptr(uint32_t idx) const {
    const size_t row_bytes = feature_dim_ * elem_size_;
    return static_cast<const char*>(feature_data_)
         + static_cast<size_t>(idx) * row_bytes;
}

// =============================================================================
// Accessors
// =============================================================================

bool CpuCache::contains(ObjectId oid) const {
    return oid_to_idx_.count(oid.id) > 0;
}

uint64_t CpuCache::num_nodes() const {
    return num_nodes_;
}

uint64_t CpuCache::feature_dim() const {
    return feature_dim_;
}

size_t CpuCache::memory_bytes() const {
    return feature_data_size_;
}

const void* CpuCache::data_ptr() const {
    return feature_data_;
}

bool CpuCache::is_pinned() const {
    return pinned_;
}

} // namespace mdb::gnn
