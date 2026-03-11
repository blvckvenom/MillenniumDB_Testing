#include "gnn/storage/row_mapping.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace mdb::gnn {

namespace fs = std::filesystem;

namespace {

class FdGuard {
public:
    explicit FdGuard(int fd) : fd_(fd) {}
    ~FdGuard() { if (fd_ >= 0) ::close(fd_); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
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
                "RowMapping: write failed: " + std::string(std::strerror(errno)));
        }
        if (written == 0) {
            throw std::runtime_error(
                "RowMapping: write returned 0 for non-zero count — disk full or I/O error");
        }
        p += written;
        remaining -= static_cast<size_t>(written);
    }
}

} // anonymous namespace

// --- Move semantics ---

RowMapping::RowMapping(RowMapping&& other) noexcept
    : path_(std::move(other.path_)),
      mmap_ptr_(other.mmap_ptr_),
      mmap_size_(other.mmap_size_),
      count_(other.count_)
{
    other.mmap_ptr_  = nullptr;
    other.mmap_size_ = 0;
    other.count_     = 0;
}

RowMapping& RowMapping::operator=(RowMapping&& other) noexcept {
    if (this != &other) {
        if (mmap_ptr_ != nullptr) {
            ::munmap(mmap_ptr_, mmap_size_);
        }
        path_      = std::move(other.path_);
        mmap_ptr_  = other.mmap_ptr_;
        mmap_size_ = other.mmap_size_;
        count_     = other.count_;
        other.mmap_ptr_  = nullptr;
        other.mmap_size_ = 0;
        other.count_     = 0;
    }
    return *this;
}

RowMapping::~RowMapping() {
    if (mmap_ptr_ != nullptr) {
        ::munmap(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
    }
}

const ObjectId* RowMapping::data_ptr() const {
    return reinterpret_cast<const ObjectId*>(
        static_cast<const char*>(mmap_ptr_) + HEADER_SIZE);
}

// --- create() ---

RowMapping RowMapping::create(const fs::path& path, const std::vector<ObjectId>& ids) {
    // Overflow check
    size_t data_size = ids.size() * sizeof(ObjectId);
    if (!ids.empty() && data_size / sizeof(ObjectId) != ids.size()) {
        throw std::overflow_error("RowMapping::create: data size would overflow");
    }
    if (data_size > SIZE_MAX - HEADER_SIZE) {
        throw std::overflow_error("RowMapping::create: total file size would overflow");
    }
    size_t file_size = HEADER_SIZE + data_size;

    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "RowMapping::create: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    // Write header: magic(4) + version(4) + count(8)
    uint32_t magic   = MAGIC;
    uint32_t version = VERSION;
    uint64_t count   = ids.size();
    write_all(fd, &magic, sizeof(magic));
    write_all(fd, &version, sizeof(version));
    write_all(fd, &count, sizeof(count));

    // Write ObjectId array (each ObjectId is 8 bytes — just its uint64_t id)
    static_assert(sizeof(ObjectId) == sizeof(uint64_t),
                  "ObjectId must be 8 bytes for direct serialization");
    if (!ids.empty()) {
        write_all(fd, ids.data(), data_size);
    }

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "RowMapping::create: fsync failed: " + std::string(std::strerror(errno)));
    }

    // mmap read-only
    void* ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "RowMapping::create: mmap failed: " + std::string(std::strerror(errno)));
    }

    RowMapping rm;
    rm.path_      = path;
    rm.mmap_ptr_  = ptr;
    rm.mmap_size_ = file_size;
    rm.count_     = count;
    return rm;
}

// --- open() ---

RowMapping RowMapping::open(const fs::path& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("RowMapping::open: file not found: " + path.string());
    }

    auto file_size = fs::file_size(path);
    if (file_size < HEADER_SIZE) {
        throw std::runtime_error(
            "RowMapping::open: file too small (" + std::to_string(file_size) +
            " bytes): " + path.string());
    }

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "RowMapping::open: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    void* ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "RowMapping::open: mmap failed: " + std::string(std::strerror(errno)));
    }

    // Validate header
    const char* base = static_cast<const char*>(ptr);
    uint32_t magic, version;
    uint64_t count;
    std::memcpy(&magic, base, sizeof(magic));
    std::memcpy(&version, base + 4, sizeof(version));
    std::memcpy(&count, base + 8, sizeof(count));

    if (magic != MAGIC || version != VERSION) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "RowMapping::open: invalid header in " + path.string());
    }

    // Overflow check: prevent count * sizeof(ObjectId) from wrapping
    if (count > (SIZE_MAX - HEADER_SIZE) / sizeof(ObjectId)) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "RowMapping::open: count in header would overflow size computation: " +
            path.string());
    }

    size_t expected = HEADER_SIZE + count * sizeof(ObjectId);
    if (file_size < expected) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "RowMapping::open: file truncated — expected " +
            std::to_string(expected) + " bytes, got " +
            std::to_string(file_size) + ": " + path.string());
    }

    RowMapping rm;
    rm.path_      = path;
    rm.mmap_ptr_  = ptr;
    rm.mmap_size_ = file_size;
    rm.count_     = count;
    return rm;
}

// --- get() ---

ObjectId RowMapping::get(uint64_t row_index) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("RowMapping::get: not mapped");
    }
    if (row_index >= count_) {
        throw std::out_of_range(
            "RowMapping::get: index " + std::to_string(row_index) +
            " out of range [0, " + std::to_string(count_) + ")");
    }
    return data_ptr()[row_index];
}

// --- find() ---

std::optional<uint64_t> RowMapping::find(ObjectId target) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("RowMapping::find: not mapped");
    }
    const ObjectId* arr = data_ptr();
    for (uint64_t i = 0; i < count_; ++i) {
        if (arr[i].id == target.id) {
            return i;
        }
    }
    return std::nullopt;
}

} // namespace mdb::gnn
