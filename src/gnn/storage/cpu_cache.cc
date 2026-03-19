#include "gnn/storage/cpu_cache.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>

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

    auto header = CacheFileHeader::make(N, D, dtype);

    int fd = ::open(output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "CpuCache::build: cannot create " + output_path.string() +
            ": " + safe_strerror(errno));
    }
    FdGuard guard(fd);

    // Write header
    write_all(fd, &header, sizeof(header), output_path.string());

    // Write ObjectId table
    for (const auto& oid : nodes) {
        uint64_t id = oid.id;
        write_all(fd, &id, sizeof(id), output_path.string());
    }

    // Write feature data: for each node, look up row in FeatureMatrix
    std::vector<char> row_buf(row_bytes);
    for (const auto& oid : nodes) {
        auto row = row_mapping.find(oid);
        if (!row.has_value()) {
            throw std::runtime_error(
                "CpuCache::build: ObjectId " + std::to_string(oid.id) +
                " not in RowMapping");
        }
        std::memcpy(row_buf.data(), features.row(*row), row_bytes);
        write_all(fd, row_buf.data(), row_bytes, output_path.string());
    }

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
