#include "gnn/storage/feature_matrix.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace mdb::gnn {

// --- RAII file descriptor guard ---
namespace {

class FdGuard {
public:
    explicit FdGuard(int fd) : fd_(fd) {}
    ~FdGuard() { if (fd_ >= 0) ::close(fd_); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
    int get() const { return fd_; }
    void release() { fd_ = -1; }
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
                "FeatureMatrix: write failed: " + std::string(std::strerror(errno)));
        }
        p += written;
        remaining -= static_cast<size_t>(written);
    }
}

void* mmap_file_readonly(const fs::path& path, size_t expected_size) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "FeatureMatrix: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    void* ptr = ::mmap(nullptr, expected_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "FeatureMatrix: mmap failed for " + path.string() + ": " + std::strerror(errno));
    }
    return ptr;
}

} // anonymous namespace

// --- Move semantics ---

FeatureMatrix::FeatureMatrix(FeatureMatrix&& other) noexcept
    : header_(other.header_),
      path_(std::move(other.path_)),
      mmap_ptr_(other.mmap_ptr_),
      mmap_size_(other.mmap_size_)
{
    other.mmap_ptr_  = nullptr;
    other.mmap_size_ = 0;
}

FeatureMatrix& FeatureMatrix::operator=(FeatureMatrix&& other) noexcept {
    if (this != &other) {
        // Release current mapping
        if (mmap_ptr_ != nullptr) {
            ::munmap(mmap_ptr_, mmap_size_);
        }
        header_    = other.header_;
        path_      = std::move(other.path_);
        mmap_ptr_  = other.mmap_ptr_;
        mmap_size_ = other.mmap_size_;
        other.mmap_ptr_  = nullptr;
        other.mmap_size_ = 0;
    }
    return *this;
}

FeatureMatrix::~FeatureMatrix() {
    if (mmap_ptr_ != nullptr) {
        ::munmap(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
    }
}

const void* FeatureMatrix::data_ptr() const {
    return static_cast<const char*>(mmap_ptr_) + FeatureMatrixHeader::SIZE;
}

// --- create() ---

FeatureMatrix FeatureMatrix::create(
    const fs::path& path,
    uint64_t num_rows,
    uint64_t num_cols,
    GnnDtype dtype,
    const void* data)
{
    if (num_rows == 0 || num_cols == 0) {
        throw std::invalid_argument("FeatureMatrix::create: num_rows and num_cols must be > 0");
    }
    if (data == nullptr) {
        throw std::invalid_argument("FeatureMatrix::create: data pointer must not be null");
    }

    auto header = FeatureMatrixHeader::make(num_rows, num_cols, dtype);

    // Overflow check: num_rows * row_bytes
    size_t rb = header.row_bytes();
    if (rb > 0 && num_rows > SIZE_MAX / rb) {
        throw std::overflow_error("FeatureMatrix::create: data size would overflow size_t");
    }
    size_t data_size = header.data_bytes();
    if (data_size > SIZE_MAX - FeatureMatrixHeader::SIZE) {
        throw std::overflow_error("FeatureMatrix::create: total file size would overflow size_t");
    }
    size_t file_size = FeatureMatrixHeader::SIZE + data_size;

    // Write file: header + data
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "FeatureMatrix::create: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    write_all(fd, &header, sizeof(header));
    write_all(fd, data, data_size);

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "FeatureMatrix::create: fsync failed: " + std::string(std::strerror(errno)));
    }

    // mmap the written file read-only
    FeatureMatrix fm;
    fm.header_    = header;
    fm.path_      = path;
    fm.mmap_size_ = file_size;
    fm.mmap_ptr_  = mmap_file_readonly(path, file_size);
    return fm;
}

// --- open() ---

FeatureMatrix FeatureMatrix::open(const fs::path& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("FeatureMatrix::open: file not found: " + path.string());
    }

    auto file_size = fs::file_size(path);
    if (file_size < FeatureMatrixHeader::SIZE) {
        throw std::runtime_error(
            "FeatureMatrix::open: file too small for header (" +
            std::to_string(file_size) + " bytes): " + path.string());
    }

    // mmap the entire file
    void* ptr = mmap_file_readonly(path, file_size);

    // Read and validate header from mapped memory
    FeatureMatrixHeader header;
    std::memcpy(&header, ptr, sizeof(header));

    if (!header.is_valid()) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "FeatureMatrix::open: invalid header in " + path.string());
    }

    // Validate file size matches header expectations
    size_t expected_size = FeatureMatrixHeader::SIZE + header.data_bytes();
    if (file_size < expected_size) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "FeatureMatrix::open: file truncated — expected " +
            std::to_string(expected_size) + " bytes, got " +
            std::to_string(file_size) + ": " + path.string());
    }

    FeatureMatrix fm;
    fm.header_    = header;
    fm.path_      = path;
    fm.mmap_ptr_  = ptr;
    fm.mmap_size_ = file_size;
    return fm;
}

// --- row() ---

const void* FeatureMatrix::row(uint64_t row_id) const {
    if (row_id >= header_.num_rows) {
        throw std::out_of_range(
            "FeatureMatrix::row: row_id " + std::to_string(row_id) +
            " out of range [0, " + std::to_string(header_.num_rows) + ")");
    }
    return static_cast<const char*>(data_ptr()) + row_id * header_.row_bytes();
}

// --- scan() ---

void FeatureMatrix::scan(RowCallback callback) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("FeatureMatrix::scan: not mapped");
    }

    // Hint: sequential access pattern for the full data region
    ::madvise(mmap_ptr_, mmap_size_, MADV_SEQUENTIAL);

    for (uint64_t i = 0; i < header_.num_rows; ++i) {
        callback(i, row(i));
    }

    // Restore default policy
    ::madvise(mmap_ptr_, mmap_size_, MADV_NORMAL);
}

// --- extract_rows() ---

void FeatureMatrix::extract_rows(
    const std::vector<uint64_t>& row_ids, void* out) const
{
    if (row_ids.empty()) return;

    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("FeatureMatrix::extract_rows: not mapped");
    }

    const size_t rb = header_.row_bytes();
    char* out_bytes = static_cast<char*>(out);

    // Build (row_id, original_position) pairs, then sort by row_id
    // for sequential disk access
    std::vector<std::pair<uint64_t, size_t>> sorted_ids;
    sorted_ids.reserve(row_ids.size());
    for (size_t i = 0; i < row_ids.size(); ++i) {
        sorted_ids.emplace_back(row_ids[i], i);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end());

    // Prefetch hint for the data region
    ::madvise(mmap_ptr_, mmap_size_, MADV_WILLNEED);

    // Copy rows in sorted order, placing each at its original position in output
    for (const auto& [rid, orig_pos] : sorted_ids) {
        const void* src = row(rid); // bounds-checked
        std::memcpy(out_bytes + orig_pos * rb, src, rb);
    }

    ::madvise(mmap_ptr_, mmap_size_, MADV_NORMAL);
}

// --- create_streaming() ---

FeatureMatrix FeatureMatrix::create_streaming(
    const fs::path& /*path*/, uint64_t /*num_rows*/, uint64_t /*num_cols*/,
    GnnDtype /*dtype*/, RowWriter /*writer*/)
{
    throw std::runtime_error("FeatureMatrix::create_streaming not implemented yet");
}

// --- create_reordered() ---

FeatureMatrix FeatureMatrix::create_reordered(
    const FeatureMatrix& /*source*/,
    const std::vector<uint64_t>& /*permutation*/,
    const fs::path& /*output_path*/)
{
    throw std::runtime_error("FeatureMatrix::create_reordered not implemented yet");
}

} // namespace mdb::gnn
