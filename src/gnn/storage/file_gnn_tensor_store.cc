#include "file_gnn_tensor_store.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

// POSIX headers for memory mapping
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mdb::gnn {

// ============================================================================
// MappedShard Implementation
// ============================================================================

FileGnnTensorStore::MappedShard::~MappedShard() {
    if (data != nullptr && data != MAP_FAILED) {
        munmap(data, size);
    }
    if (fd >= 0) {
        close(fd);
    }
}

FileGnnTensorStore::MappedShard::MappedShard(MappedShard&& other) noexcept
    : data(other.data), size(other.size), fd(other.fd) {
    other.data = nullptr;
    other.size = 0;
    other.fd = -1;
}

FileGnnTensorStore::MappedShard& FileGnnTensorStore::MappedShard::operator=(MappedShard&& other) noexcept {
    if (this != &other) {
        if (data != nullptr && data != MAP_FAILED) {
            munmap(data, size);
        }
        if (fd >= 0) {
            close(fd);
        }
        data = other.data;
        size = other.size;
        fd = other.fd;
        other.data = nullptr;
        other.size = 0;
        other.fd = -1;
    }
    return *this;
}

// ============================================================================
// FileGnnTensorStore Construction/Destruction
// ============================================================================

FileGnnTensorStore::FileGnnTensorStore(const std::filesystem::path& base_path,
                                       size_t max_shard_size,
                                       bool create_if_missing)
    : base_path_(base_path), max_shard_size_(max_shard_size) {

    if (create_if_missing && !std::filesystem::exists(base_path_)) {
        std::filesystem::create_directories(base_path_);
    }

    if (!std::filesystem::is_directory(base_path_)) {
        throw std::runtime_error("FileGnnTensorStore: path is not a directory: " +
                                 base_path_.string());
    }

    // Load existing index if present
    if (std::filesystem::exists(index_path())) {
        load_index();
    }
}

FileGnnTensorStore::~FileGnnTensorStore() {
    try {
        flush();
    } catch (...) {
        // Suppress exceptions in destructor
    }
    unmap_all_shards();
}

std::filesystem::path FileGnnTensorStore::shard_path(uint32_t shard_id) const {
    return base_path_ / ("data_" + std::to_string(shard_id) + ".bin");
}

// ============================================================================
// Core Operations
// ============================================================================

bool FileGnnTensorStore::store(const std::string& key,
                               const void* data,
                               const std::vector<int64_t>& shape,
                               GnnDtype dtype) {
    if (data == nullptr || shape.empty()) {
        return false;
    }

    // Validate shape
    for (auto dim : shape) {
        if (dim <= 0) {
            return false;
        }
    }

    // Compute metadata
    GnnTensorMetadata metadata;
    metadata.shape = shape;
    metadata.dtype = dtype;
    metadata.byte_size = metadata.compute_byte_size();

    size_t aligned_size = align_up(metadata.byte_size);

    std::unique_lock lock(mutex_);

    // Mark old entry as deleted if exists
    auto it = index_.find(key);
    if (it != index_.end()) {
        deleted_bytes_ += align_up(it->second.metadata.byte_size);
        total_bytes_ -= it->second.metadata.byte_size;
    }

    // Find shard with enough space
    uint32_t shard_id = 0;
    size_t offset = 0;

    if (shard_sizes_.empty()) {
        shard_sizes_.push_back(0);
    }

    // Find existing shard with enough space
    bool found = false;
    for (size_t i = 0; i < shard_sizes_.size(); ++i) {
        if (shard_sizes_[i] + aligned_size <= max_shard_size_) {
            shard_id = static_cast<uint32_t>(i);
            offset = shard_sizes_[i];
            found = true;
            break;
        }
    }

    if (!found) {
        // No existing shard has space — create a new one.
        // If aligned_size > max_shard_size_, the tensor gets its own oversized shard.
        shard_sizes_.push_back(0);
        shard_id = static_cast<uint32_t>(shard_sizes_.size() - 1);
        offset = 0;
    }

    // Write data to shard file
    ensure_shard(shard_id);

    std::filesystem::path shard_file = shard_path(shard_id);
    int fd = open(shard_file.c_str(), O_WRONLY | O_CREAT, 0644);
    if (fd < 0) {
        return false;
    }

    // Seek to offset and write
    if (lseek(fd, static_cast<off_t>(offset), SEEK_SET) < 0) {
        close(fd);
        return false;
    }

    ssize_t written = write(fd, data, metadata.byte_size);
    close(fd);

    if (written != static_cast<ssize_t>(metadata.byte_size)) {
        return false;
    }

    // Update shard size
    shard_sizes_[shard_id] = offset + aligned_size;

    // Invalidate memory mapping for this shard (will be re-mapped on next load)
    {
        std::lock_guard<std::mutex> shard_lock(shard_mutex_);
        if (shard_id < shards_.size() && shards_[shard_id].data != nullptr) {
            shards_[shard_id] = MappedShard();  // Unmap
        }
    }

    // Update index
    IndexEntry entry;
    entry.metadata = metadata;
    entry.shard_id = shard_id;
    entry.offset = offset;
    entry.deleted = false;

    index_[key] = entry;
    total_bytes_ += metadata.byte_size;

    return true;
}

GnnTensorView FileGnnTensorStore::load(const std::string& key) const {
    std::shared_lock lock(mutex_);

    auto it = index_.find(key);
    if (it == index_.end() || it->second.deleted) {
        return GnnTensorView();  // Invalid view
    }

    const IndexEntry& entry = it->second;

    // Hold shard_mutex_ across both map AND read to prevent vector reallocation race
    std::lock_guard<std::mutex> shard_lock(shard_mutex_);
    map_shard_impl(entry.shard_id);

    if (entry.shard_id >= shards_.size() || shards_[entry.shard_id].data == nullptr) {
        return GnnTensorView();  // Mapping failed
    }

    const void* data_ptr = static_cast<const char*>(shards_[entry.shard_id].data) + entry.offset;
    return GnnTensorView(data_ptr, entry.metadata);
}

bool FileGnnTensorStore::remove(const std::string& key) {
    std::unique_lock lock(mutex_);

    auto it = index_.find(key);
    if (it == index_.end() || it->second.deleted) {
        return false;
    }

    // Soft delete - mark as deleted for lazy compaction
    deleted_bytes_ += align_up(it->second.metadata.byte_size);
    total_bytes_ -= it->second.metadata.byte_size;
    it->second.deleted = true;

    return true;
}

bool FileGnnTensorStore::exists(const std::string& key) const {
    std::shared_lock lock(mutex_);
    auto it = index_.find(key);
    return it != index_.end() && !it->second.deleted;
}

// ============================================================================
// Metadata Operations
// ============================================================================

std::optional<GnnTensorMetadata> FileGnnTensorStore::get_metadata(const std::string& key) const {
    std::shared_lock lock(mutex_);

    auto it = index_.find(key);
    if (it == index_.end() || it->second.deleted) {
        return std::nullopt;
    }

    return it->second.metadata;
}

std::vector<std::string> FileGnnTensorStore::list_keys() const {
    std::shared_lock lock(mutex_);

    std::vector<std::string> keys;
    keys.reserve(index_.size());
    for (const auto& [key, entry] : index_) {
        if (!entry.deleted) {
            keys.push_back(key);
        }
    }
    return keys;
}

size_t FileGnnTensorStore::total_size() const {
    std::shared_lock lock(mutex_);
    return total_bytes_;
}

size_t FileGnnTensorStore::count() const {
    std::shared_lock lock(mutex_);
    size_t active = 0;
    for (const auto& [_, entry] : index_) {
        if (!entry.deleted) {
            ++active;
        }
    }
    return active;
}

// ============================================================================
// Lifecycle Operations
// ============================================================================

void FileGnnTensorStore::clear() {
    std::unique_lock lock(mutex_);

    unmap_all_shards();
    index_.clear();
    shard_sizes_.clear();
    total_bytes_ = 0;
    deleted_bytes_ = 0;

    // Remove all files
    if (std::filesystem::exists(index_path())) {
        std::filesystem::remove(index_path());
    }

    for (uint32_t i = 0; ; ++i) {
        auto path = shard_path(i);
        if (!std::filesystem::exists(path)) {
            break;
        }
        std::filesystem::remove(path);
    }
}

void FileGnnTensorStore::flush() {
    std::shared_lock lock(mutex_);
    save_index();
}

// ============================================================================
// File-Specific Operations
// ============================================================================

size_t FileGnnTensorStore::compact() {
    std::unique_lock lock(mutex_);

    size_t reclaimed = deleted_bytes_;
    if (reclaimed == 0) {
        return 0;
    }

    // Create new store in temporary location
    auto temp_path = base_path_ / ".compact_temp";
    std::filesystem::create_directories(temp_path);

    // Copy all non-deleted tensors
    std::vector<size_t> new_shard_sizes;
    new_shard_sizes.push_back(0);

    std::unordered_map<std::string, IndexEntry> new_index;

    for (auto& [key, entry] : index_) {
        if (entry.deleted) {
            continue;
        }

        // Read old data
        map_shard(entry.shard_id);
        const char* old_data = static_cast<const char*>(shards_[entry.shard_id].data) + entry.offset;

        size_t aligned_size = align_up(entry.metadata.byte_size);

        // Find shard in new store
        uint32_t new_shard_id = 0;
        size_t new_offset = 0;

        bool compact_found = false;
        for (size_t i = 0; i < new_shard_sizes.size(); ++i) {
            if (new_shard_sizes[i] + aligned_size <= max_shard_size_) {
                new_shard_id = static_cast<uint32_t>(i);
                new_offset = new_shard_sizes[i];
                compact_found = true;
                break;
            }
        }

        if (!compact_found) {
            new_shard_sizes.push_back(0);
            new_shard_id = static_cast<uint32_t>(new_shard_sizes.size() - 1);
            new_offset = 0;
        }

        // Write to new shard
        auto new_shard_path = temp_path / ("data_" + std::to_string(new_shard_id) + ".bin");
        int fd = open(new_shard_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            std::filesystem::remove_all(temp_path);
            throw std::runtime_error("compact: failed to create shard: " + new_shard_path.string());
        }

        if (lseek(fd, static_cast<off_t>(new_offset), SEEK_SET) < 0) {
            close(fd);
            std::filesystem::remove_all(temp_path);
            throw std::runtime_error("compact: lseek failed");
        }

        const char* write_ptr = static_cast<const char*>(old_data);
        size_t remaining = entry.metadata.byte_size;
        while (remaining > 0) {
            ssize_t written = ::write(fd, write_ptr, remaining);
            if (written < 0) {
                close(fd);
                std::filesystem::remove_all(temp_path);
                throw std::runtime_error("compact: write failed: " + std::string(strerror(errno)));
            }
            write_ptr += written;
            remaining -= static_cast<size_t>(written);
        }

        fsync(fd);
        close(fd);

        new_shard_sizes[new_shard_id] = new_offset + aligned_size;

        // Update entry for new location
        IndexEntry new_entry = entry;
        new_entry.shard_id = new_shard_id;
        new_entry.offset = new_offset;
        new_index[key] = new_entry;
    }

    // Phase 1: Unmap old shards
    unmap_all_shards();

    // Phase 2: Rename old files to .bak (recoverable)
    for (uint32_t i = 0; ; ++i) {
        auto path = shard_path(i);
        if (!std::filesystem::exists(path)) break;
        std::filesystem::rename(path, std::filesystem::path(path.string() + ".bak"));
    }
    auto idx_path = index_path();
    if (std::filesystem::exists(idx_path)) {
        std::filesystem::rename(idx_path, std::filesystem::path(idx_path.string() + ".bak"));
    }

    // Phase 3: Move new files into place
    for (const auto& dir_entry : std::filesystem::directory_iterator(temp_path)) {
        std::filesystem::rename(dir_entry.path(), base_path_ / dir_entry.path().filename());
    }
    std::filesystem::remove(temp_path);

    // Phase 4: Delete .bak files (safe — new files in place)
    for (uint32_t i = 0; ; ++i) {
        auto bak = std::filesystem::path(shard_path(i).string() + ".bak");
        if (!std::filesystem::exists(bak)) break;
        std::filesystem::remove(bak);
    }
    if (auto idx_bak = std::filesystem::path(idx_path.string() + ".bak"); std::filesystem::exists(idx_bak)) {
        std::filesystem::remove(idx_bak);
    }

    // Update state
    index_ = std::move(new_index);
    shard_sizes_ = std::move(new_shard_sizes);
    deleted_bytes_ = 0;

    // Save new index
    save_index();

    return reclaimed;
}

bool FileGnnTensorStore::needs_compaction(double threshold) const {
    std::shared_lock lock(mutex_);

    size_t total_disk_size = 0;
    for (auto sz : shard_sizes_) {
        total_disk_size += sz;
    }

    if (total_disk_size == 0) {
        return false;
    }

    double fragmentation = static_cast<double>(deleted_bytes_) / static_cast<double>(total_disk_size);
    return fragmentation > threshold;
}

// ============================================================================
// File I/O Helpers
// ============================================================================

void FileGnnTensorStore::load_index() {
    std::ifstream file(index_path(), std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open index file: " + index_path().string());
    }

    // Read header
    uint32_t magic, version;
    uint64_t num_tensors;
    uint32_t num_shards;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
    file.read(reinterpret_cast<char*>(&num_shards), sizeof(num_shards));

    if (magic != MAGIC) {
        throw std::runtime_error("Invalid index file magic number");
    }

    if (version > VERSION) {
        throw std::runtime_error("Index file version " + std::to_string(version) +
                                 " is newer than supported version " + std::to_string(VERSION));
    }

    // Read shard sizes
    shard_sizes_.resize(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        file.read(reinterpret_cast<char*>(&shard_sizes_[i]), sizeof(size_t));
    }

    // Read entries
    for (uint64_t i = 0; i < num_tensors; ++i) {
        // Key
        uint32_t key_length;
        file.read(reinterpret_cast<char*>(&key_length), sizeof(key_length));
        std::string key(key_length, '\0');
        file.read(key.data(), key_length);

        IndexEntry entry;

        // Dtype
        uint8_t dtype_val;
        file.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val));
        entry.metadata.dtype = static_cast<GnnDtype>(dtype_val);

        // Shape
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        entry.metadata.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            file.read(reinterpret_cast<char*>(&entry.metadata.shape[d]), sizeof(int64_t));
        }

        // Location
        file.read(reinterpret_cast<char*>(&entry.shard_id), sizeof(entry.shard_id));
        file.read(reinterpret_cast<char*>(&entry.offset), sizeof(entry.offset));
        file.read(reinterpret_cast<char*>(&entry.metadata.byte_size), sizeof(entry.metadata.byte_size));

        // Deleted flag
        uint8_t deleted;
        file.read(reinterpret_cast<char*>(&deleted), sizeof(deleted));
        entry.deleted = (deleted != 0);

        if (!entry.deleted) {
            total_bytes_ += entry.metadata.byte_size;
        } else {
            deleted_bytes_ += align_up(entry.metadata.byte_size);
        }

        index_[key] = entry;
    }
}

void FileGnnTensorStore::save_index() {
    std::ofstream file(index_path(), std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Failed to write index file: " + index_path().string());
    }

    // Write header
    uint32_t magic = MAGIC;
    uint32_t version = VERSION;
    uint64_t num_tensors = index_.size();
    uint32_t num_shards = static_cast<uint32_t>(shard_sizes_.size());

    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));
    file.write(reinterpret_cast<const char*>(&num_shards), sizeof(num_shards));

    // Write shard sizes
    for (auto sz : shard_sizes_) {
        file.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    }

    // Write entries
    for (const auto& [key, entry] : index_) {
        // Key
        uint32_t key_length = static_cast<uint32_t>(key.size());
        file.write(reinterpret_cast<const char*>(&key_length), sizeof(key_length));
        file.write(key.data(), key_length);

        // Dtype
        uint8_t dtype_val = static_cast<uint8_t>(entry.metadata.dtype);
        file.write(reinterpret_cast<const char*>(&dtype_val), sizeof(dtype_val));

        // Shape
        uint32_t ndim = static_cast<uint32_t>(entry.metadata.shape.size());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (auto dim : entry.metadata.shape) {
            file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }

        // Location
        file.write(reinterpret_cast<const char*>(&entry.shard_id), sizeof(entry.shard_id));
        file.write(reinterpret_cast<const char*>(&entry.offset), sizeof(entry.offset));
        file.write(reinterpret_cast<const char*>(&entry.metadata.byte_size), sizeof(entry.metadata.byte_size));

        // Deleted flag
        uint8_t deleted = entry.deleted ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&deleted), sizeof(deleted));
    }
}

void FileGnnTensorStore::ensure_shard(uint32_t shard_id) {
    auto path = shard_path(shard_id);
    if (!std::filesystem::exists(path)) {
        // Create empty file
        std::ofstream file(path, std::ios::binary);
    }

    // Ensure shard_sizes_ is large enough
    while (shard_sizes_.size() <= shard_id) {
        shard_sizes_.push_back(0);
    }
}

void FileGnnTensorStore::map_shard(uint32_t shard_id) const {
    std::lock_guard<std::mutex> shard_lock(shard_mutex_);
    map_shard_impl(shard_id);
}

void FileGnnTensorStore::map_shard_impl(uint32_t shard_id) const {
    // NOTE: Caller must hold shard_mutex_

    // Ensure shards_ vector is large enough
    while (shards_.size() <= shard_id) {
        shards_.emplace_back();
    }

    if (shards_[shard_id].data != nullptr) {
        return;  // Already mapped
    }

    auto path = shard_path(shard_id);
    if (!std::filesystem::exists(path)) {
        return;  // Shard doesn't exist yet
    }

    size_t file_size = std::filesystem::file_size(path);
    if (file_size == 0) {
        return;  // Empty file
    }

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return;  // Failed to open
    }

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return;
    }

    // Advise kernel we'll read sequentially (optimization for batch loading)
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    shards_[shard_id].data = mapped;
    shards_[shard_id].size = file_size;
    shards_[shard_id].fd = fd;
}

void FileGnnTensorStore::unmap_all_shards() {
    std::lock_guard<std::mutex> shard_lock(shard_mutex_);
    shards_.clear();
}

} // namespace mdb::gnn
