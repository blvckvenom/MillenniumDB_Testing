#include "gnn/training/split_store.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mdb::gnn {

namespace fs = std::filesystem;

// ============================================================================
// Anonymous helpers
// ============================================================================

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
                "SplitStore: write failed: " + std::string(std::strerror(errno)));
        }
        if (written == 0) {
            throw std::runtime_error(
                "SplitStore: write returned 0 — disk full or I/O error");
        }
        p         += written;
        remaining -= static_cast<size_t>(written);
    }
}

} // anonymous namespace

// ============================================================================
// Move semantics + destructor
// ============================================================================

SplitStore::SplitStore(SplitStore&& other) noexcept
    : mmap_ptr_ (other.mmap_ptr_),
      mmap_size_(other.mmap_size_),
      num_nodes_(other.num_nodes_)
{
    other.mmap_ptr_  = nullptr;
    other.mmap_size_ = 0;
    other.num_nodes_ = 0;
}

SplitStore& SplitStore::operator=(SplitStore&& other) noexcept {
    if (this != &other) {
        if (mmap_ptr_ != nullptr) {
            ::munmap(mmap_ptr_, mmap_size_);
        }
        mmap_ptr_  = other.mmap_ptr_;
        mmap_size_ = other.mmap_size_;
        num_nodes_ = other.num_nodes_;

        other.mmap_ptr_  = nullptr;
        other.mmap_size_ = 0;
        other.num_nodes_ = 0;
    }
    return *this;
}

SplitStore::~SplitStore() {
    if (mmap_ptr_ != nullptr) {
        ::munmap(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
    }
}

// ============================================================================
// data_ptr()
// ============================================================================

const uint8_t* SplitStore::data_ptr() const {
    return reinterpret_cast<const uint8_t*>(
        static_cast<const char*>(mmap_ptr_) + HEADER_SIZE);
}

// ============================================================================
// write()
// ============================================================================

void SplitStore::write(const fs::path& path, const std::vector<uint8_t>& splits) {
    // Overflow guard: splits.size() * 1 must not wrap when added to HEADER_SIZE
    if (splits.size() > SIZE_MAX - HEADER_SIZE) {
        throw std::overflow_error("SplitStore::write: total file size would overflow");
    }

    // Create parent directories if needed
    auto parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }

    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "SplitStore::write: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    // Header: magic(8) + version(4) + reserved(4) + num_nodes(8) = 24 bytes
    write_all(fd, MAGIC, 8);

    uint32_t version  = VERSION;
    uint32_t reserved = 0;
    write_all(fd, &version,  sizeof(version));
    write_all(fd, &reserved, sizeof(reserved));

    uint64_t num_nodes = splits.size();
    write_all(fd, &num_nodes, sizeof(num_nodes));

    // Data: uint8[N]
    if (!splits.empty()) {
        write_all(fd, splits.data(), splits.size() * sizeof(uint8_t));
    }

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "SplitStore::write: fsync failed: " + std::string(std::strerror(errno)));
    }

    // Best-effort parent directory fsync for crash consistency
    {
        int dir_fd = ::open(parent.empty() ? "." : parent.c_str(), O_RDONLY);
        if (dir_fd >= 0) {
            ::fsync(dir_fd);
            ::close(dir_fd);
        }
    }
}

// ============================================================================
// open()
// ============================================================================

SplitStore SplitStore::open(const fs::path& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("SplitStore::open: file not found: " + path.string());
    }

    auto file_size = fs::file_size(path);
    if (file_size < HEADER_SIZE) {
        throw std::runtime_error(
            "SplitStore::open: file too small (" + std::to_string(file_size) +
            " bytes): " + path.string());
    }

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "SplitStore::open: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    void* ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "SplitStore::open: mmap failed: " + std::string(std::strerror(errno)));
    }

    // Validate header
    const char* base = static_cast<const char*>(ptr);

    uint8_t  magic_buf[8];
    uint32_t version;
    uint32_t reserved_field;
    uint64_t num_nodes;

    std::memcpy(magic_buf,       base,      8);
    std::memcpy(&version,        base + 8,  4);
    std::memcpy(&reserved_field, base + 12, 4);
    std::memcpy(&num_nodes,      base + 16, 8);

    if (std::memcmp(magic_buf, MAGIC, 8) != 0 || version != VERSION) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "SplitStore::open: invalid header in " + path.string());
    }

    // Overflow guard: num_nodes * 1 added to HEADER_SIZE
    if (num_nodes > SIZE_MAX - HEADER_SIZE) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "SplitStore::open: num_nodes in header would overflow size computation: " +
            path.string());
    }

    size_t expected = HEADER_SIZE + num_nodes * sizeof(uint8_t);
    if (file_size < expected) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "SplitStore::open: file truncated — expected " +
            std::to_string(expected) + " bytes, got " +
            std::to_string(file_size) + ": " + path.string());
    }

    SplitStore ss;
    ss.mmap_ptr_  = ptr;
    ss.mmap_size_ = file_size;
    ss.num_nodes_ = num_nodes;
    return ss;
}

// ============================================================================
// parse_split_string()
// ============================================================================

SplitStore::Split SplitStore::parse_split_string(const std::string& s) {
    if (s == "train")      return TRAIN;
    if (s == "val")        return VAL;
    if (s == "valid")      return VAL;  // OGB convention
    if (s == "validation") return VAL;
    if (s == "test")       return TEST;
    return UNLABELED;
}

// ============================================================================
// get()
// ============================================================================

SplitStore::Split SplitStore::get(uint64_t row_index) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("SplitStore::get: store is not open");
    }
    if (row_index >= num_nodes_) {
        throw std::out_of_range(
            "SplitStore::get: index " + std::to_string(row_index) +
            " out of range [0, " + std::to_string(num_nodes_) + ")");
    }
    return static_cast<Split>(data_ptr()[row_index]);
}

// ============================================================================
// gather_mask()
// ============================================================================

torch::Tensor SplitStore::gather_mask(const std::vector<uint64_t>& row_indices,
                                       Split target) const {
    auto tensor = torch::empty({static_cast<int64_t>(row_indices.size())}, torch::kBool);
    auto acc    = tensor.accessor<bool, 1>();
    for (size_t i = 0; i < row_indices.size(); i++) {
        acc[i] = (get(row_indices[i]) == target);
    }
    return tensor;
}

} // namespace mdb::gnn
