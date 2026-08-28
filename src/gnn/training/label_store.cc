#include "gnn/training/label_store.h"

#include "gnn/common/posix_io.h"

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
// Move semantics + destructor
// ============================================================================

LabelStore::LabelStore(LabelStore&& other) noexcept
    : mmap_ptr_   (other.mmap_ptr_),
      mmap_size_  (other.mmap_size_),
      num_nodes_  (other.num_nodes_),
      num_classes_(other.num_classes_)
{
    other.mmap_ptr_    = nullptr;
    other.mmap_size_   = 0;
    other.num_nodes_   = 0;
    other.num_classes_ = 0;
}

LabelStore& LabelStore::operator=(LabelStore&& other) noexcept {
    if (this != &other) {
        if (mmap_ptr_ != nullptr) {
            ::munmap(mmap_ptr_, mmap_size_);
        }
        mmap_ptr_    = other.mmap_ptr_;
        mmap_size_   = other.mmap_size_;
        num_nodes_   = other.num_nodes_;
        num_classes_ = other.num_classes_;

        other.mmap_ptr_    = nullptr;
        other.mmap_size_   = 0;
        other.num_nodes_   = 0;
        other.num_classes_ = 0;
    }
    return *this;
}

LabelStore::~LabelStore() {
    if (mmap_ptr_ != nullptr) {
        ::munmap(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
    }
}

// ============================================================================
// data_ptr()
// ============================================================================

const int64_t* LabelStore::data_ptr() const {
    return reinterpret_cast<const int64_t*>(
        static_cast<const char*>(mmap_ptr_) + HEADER_SIZE);
}

// ============================================================================
// write()
// ============================================================================

void LabelStore::write(const fs::path& path,
                       const std::vector<int64_t>& labels,
                       uint64_t num_classes)
{
    // Overflow guard: labels.size() * 8 must not wrap
    size_t data_size = labels.size() * sizeof(int64_t);
    if (!labels.empty() && data_size / sizeof(int64_t) != labels.size()) {
        throw std::overflow_error("LabelStore::write: data size would overflow");
    }
    if (data_size > SIZE_MAX - HEADER_SIZE) {
        throw std::overflow_error("LabelStore::write: total file size would overflow");
    }

    // Create parent directories if needed
    auto parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }

    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "LabelStore::write: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    // Header: magic(8) + version(4) + reserved(4) + num_nodes(8) + num_classes(8) = 32 bytes
    write_all(fd, MAGIC, 8);

    uint32_t version  = VERSION;
    uint32_t reserved = 0;
    write_all(fd, &version,  sizeof(version));
    write_all(fd, &reserved, sizeof(reserved));

    uint64_t num_nodes = labels.size();
    write_all(fd, &num_nodes,   sizeof(num_nodes));
    write_all(fd, &num_classes, sizeof(num_classes));

    // Data: int64[N]
    if (!labels.empty()) {
        write_all(fd, labels.data(), data_size);
    }

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "LabelStore::write: fsync failed: " + std::string(std::strerror(errno)));
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

LabelStore LabelStore::open(const fs::path& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("LabelStore::open: file not found: " + path.string());
    }

    auto file_size = fs::file_size(path);
    if (file_size < HEADER_SIZE) {
        throw std::runtime_error(
            "LabelStore::open: file too small (" + std::to_string(file_size) +
            " bytes): " + path.string());
    }

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "LabelStore::open: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    void* ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "LabelStore::open: mmap failed: " + std::string(std::strerror(errno)));
    }

    // Validate header
    const char* base = static_cast<const char*>(ptr);

    uint8_t  magic_buf[8];
    uint32_t version;
    uint32_t reserved_field;
    uint64_t num_nodes;
    uint64_t num_classes;

    std::memcpy(magic_buf,       base,      8);
    std::memcpy(&version,        base + 8,  4);
    std::memcpy(&reserved_field, base + 12, 4);
    std::memcpy(&num_nodes,      base + 16, 8);
    std::memcpy(&num_classes,    base + 24, 8);

    if (std::memcmp(magic_buf, MAGIC, 8) != 0 || version != VERSION) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "LabelStore::open: invalid header in " + path.string());
    }

    // Overflow guard: num_nodes * 8 must not wrap
    if (num_nodes > (SIZE_MAX - HEADER_SIZE) / sizeof(int64_t)) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "LabelStore::open: num_nodes in header would overflow size computation: " +
            path.string());
    }

    size_t expected = HEADER_SIZE + num_nodes * sizeof(int64_t);
    if (file_size < expected) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "LabelStore::open: file truncated — expected " +
            std::to_string(expected) + " bytes, got " +
            std::to_string(file_size) + ": " + path.string());
    }

    LabelStore ls;
    ls.mmap_ptr_    = ptr;
    ls.mmap_size_   = file_size;
    ls.num_nodes_   = num_nodes;
    ls.num_classes_ = num_classes;
    return ls;
}

// ============================================================================
// get()
// ============================================================================

int64_t LabelStore::get(uint64_t row_index) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("LabelStore::get: store is not open");
    }
    if (row_index >= num_nodes_) {
        throw std::out_of_range(
            "LabelStore::get: index " + std::to_string(row_index) +
            " out of range [0, " + std::to_string(num_nodes_) + ")");
    }
    return data_ptr()[row_index];
}

// ============================================================================
// gather()
// ============================================================================

torch::Tensor LabelStore::gather(const std::vector<uint64_t>& row_indices) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("LabelStore::gather: store is not open");
    }
    auto tensor = torch::empty({static_cast<int64_t>(row_indices.size())}, torch::kInt64);
    auto acc    = tensor.accessor<int64_t, 1>();
    const int64_t* d = data_ptr();
    for (size_t i = 0; i < row_indices.size(); i++) {
        const uint64_t r = row_indices[i];
        // BatchAssembler injects UINT64_MAX as an "unknown seed" sentinel and
        // documents that LabelStore treats it as -1 (the unlabeled label that
        // mini.label_mask filters out). Honour that contract: map the sentinel
        // and any out-of-range index to -1 instead of throwing (get() throws),
        // so a sample with an unmapped seed degrades gracefully not crash-train.
        acc[i] = (r < num_nodes_) ? d[r] : static_cast<int64_t>(-1);
    }
    return tensor;
}

} // namespace mdb::gnn
