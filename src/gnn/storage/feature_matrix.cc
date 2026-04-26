#include "gnn/storage/feature_matrix.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

namespace mdb::gnn {

namespace fs = std::filesystem;

// --- RAII helpers ---
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

// RAII guard that restores madvise hint on scope exit (even if exception thrown)
class MadviseGuard {
public:
    MadviseGuard(void* addr, size_t len, int advice)
        : addr_(addr), len_(len)
    {
        // madvise is advisory; ignore failures (e.g. ENOMEM under memory pressure)
        ::madvise(addr_, len_, advice);
    }
    ~MadviseGuard() {
        ::madvise(addr_, len_, MADV_NORMAL);
    }
    MadviseGuard(const MadviseGuard&) = delete;
    MadviseGuard& operator=(const MadviseGuard&) = delete;
private:
    void*  addr_;
    size_t len_;
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
        if (written == 0) {
            throw std::runtime_error(
                "FeatureMatrix: write returned 0 for non-zero count — disk full or I/O error");
        }
        p += written;
        remaining -= static_cast<size_t>(written);
    }
}

// Positional write of `count` bytes at file offset `offset`. Thread-safe across
// workers writing to disjoint [offset, offset+count) ranges (POSIX pwrite does
// not modify the shared file pointer). Loops over short writes and EINTR.
void pwrite_all(int fd, const void* buf, size_t count, off_t offset) {
    const char* p = static_cast<const char*>(buf);
    size_t remaining = count;
    off_t  off       = offset;
    while (remaining > 0) {
        ssize_t written = ::pwrite(fd, p, remaining, off);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "FeatureMatrix: pwrite failed: " + std::string(std::strerror(errno)));
        }
        if (written == 0) {
            throw std::runtime_error(
                "FeatureMatrix: pwrite returned 0 for non-zero count — disk full or I/O error");
        }
        p        += written;
        off      += written;
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
    other.header_    = {};  // zero out moved-from header (num_rows=0)
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
        other.header_    = {};
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

    // Overflow check: num_cols * dtype_size (before row_bytes computes it unchecked)
    size_t ds = dtype_size(dtype);
    if (ds > 0 && num_cols > SIZE_MAX / ds) {
        throw std::overflow_error("FeatureMatrix::create: num_cols * dtype_size overflow");
    }

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

    // Best-effort parent directory fsync for crash consistency
    {
        int dir_fd = ::open(path.parent_path().c_str(), O_RDONLY);
        if (dir_fd >= 0) {
            ::fsync(dir_fd);
            ::close(dir_fd);
        }
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
        char detail[256];
        std::snprintf(detail, sizeof(detail),
            "magic=0x%08X (expected 0x%08X), version=%u (expected %u), "
            "num_rows=%llu, num_cols=%llu, dtype=%u (max %u)",
            header.magic, FeatureMatrixHeader::MAGIC,
            header.version, FeatureMatrixHeader::VERSION,
            static_cast<unsigned long long>(header.num_rows),
            static_cast<unsigned long long>(header.num_cols),
            header.dtype, static_cast<unsigned>(GnnDtype::MAX_VALUE));
        throw std::runtime_error(
            "FeatureMatrix::open: invalid header in " + path.string() +
            " — " + detail);
    }

    // Overflow checks on untrusted header values before computing expected size
    size_t ds = dtype_size(header.get_dtype());
    if (ds > 0 && header.num_cols > SIZE_MAX / ds) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "FeatureMatrix::open: num_cols * dtype_size overflows: " + path.string());
    }
    size_t rb = header.row_bytes();
    if (rb > 0 && header.num_rows > SIZE_MAX / rb) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "FeatureMatrix::open: num_rows * row_bytes overflows: " + path.string());
    }
    size_t db = header.data_bytes();
    if (db > SIZE_MAX - FeatureMatrixHeader::SIZE) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "FeatureMatrix::open: header_size + data_bytes overflows: " + path.string());
    }

    // Validate file size matches header expectations
    size_t expected_size = FeatureMatrixHeader::SIZE + db;
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
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("FeatureMatrix::row: not mapped (moved-from or uninitialized)");
    }
    if (row_id >= header_.num_rows) {
        throw std::out_of_range(
            "FeatureMatrix::row: row_id " + std::to_string(row_id) +
            " out of range [0, " + std::to_string(header_.num_rows) + ")");
    }
    return static_cast<const char*>(data_ptr()) + row_id * header_.row_bytes();
}

// --- scan() ---

// NOTE: scan() and extract_rows() set process-wide madvise hints on the mmap region.
// Calling both concurrently on the SAME FeatureMatrix instance causes conflicting hints.
// This does not corrupt data, but may degrade readahead performance.
// Current callers (HNSW build, import) never do this — they use row() directly.

void FeatureMatrix::scan(RowCallback callback) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("FeatureMatrix::scan: not mapped");
    }

    // RAII guard: sets MADV_SEQUENTIAL now, restores MADV_NORMAL on scope exit
    // (even if callback throws an exception)
    MadviseGuard madvise_guard(mmap_ptr_, mmap_size_, MADV_SEQUENTIAL);

    for (uint64_t i = 0; i < header_.num_rows; ++i) {
        callback(i, row(i));
    }
}

// --- extract_rows() ---

void FeatureMatrix::extract_rows(
    const std::vector<uint64_t>& row_ids, void* out) const
{
    if (row_ids.empty()) return;

    if (out == nullptr) {
        throw std::invalid_argument(
            "FeatureMatrix::extract_rows: output pointer must not be null");
    }

    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("FeatureMatrix::extract_rows: not mapped");
    }

    // Validate all row_ids before doing any work
    for (size_t i = 0; i < row_ids.size(); ++i) {
        if (row_ids[i] >= header_.num_rows) {
            throw std::out_of_range(
                "FeatureMatrix::extract_rows: row_ids[" + std::to_string(i) +
                "] = " + std::to_string(row_ids[i]) +
                " out of range [0, " + std::to_string(header_.num_rows) + ")");
        }
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

    // RAII guard: prefetch hint, restored to NORMAL on exit
    MadviseGuard madvise_guard(mmap_ptr_, mmap_size_, MADV_WILLNEED);

    // Copy rows in sorted order, placing each at its original position in output
    const char* base = static_cast<const char*>(data_ptr());
    for (const auto& [rid, orig_pos] : sorted_ids) {
        std::memcpy(out_bytes + orig_pos * rb, base + rid * rb, rb);
    }
}

// --- create_streaming() ---

FeatureMatrix FeatureMatrix::create_streaming(
    const fs::path& path,
    uint64_t num_rows,
    uint64_t num_cols,
    GnnDtype dtype,
    RowWriter writer)
{
    if (num_rows == 0 || num_cols == 0) {
        throw std::invalid_argument(
            "FeatureMatrix::create_streaming: num_rows and num_cols must be > 0");
    }

    auto header = FeatureMatrixHeader::make(num_rows, num_cols, dtype);

    // Overflow check: num_cols * dtype_size (before row_bytes computes it unchecked)
    size_t ds = dtype_size(dtype);
    if (ds > 0 && num_cols > SIZE_MAX / ds) {
        throw std::overflow_error("FeatureMatrix::create_streaming: num_cols * dtype_size overflow");
    }

    const size_t rb = header.row_bytes();

    // Overflow check
    if (rb > 0 && num_rows > SIZE_MAX / rb) {
        throw std::overflow_error(
            "FeatureMatrix::create_streaming: data size would overflow size_t");
    }
    size_t data_size = header.data_bytes();
    if (data_size > SIZE_MAX - FeatureMatrixHeader::SIZE) {
        throw std::overflow_error(
            "FeatureMatrix::create_streaming: total file size would overflow size_t");
    }
    size_t file_size = FeatureMatrixHeader::SIZE + data_size;

    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "FeatureMatrix::create_streaming: cannot open " + path.string() +
            ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    // Write header
    write_all(fd, &header, sizeof(header));

    // Write rows one at a time via callback.
    // If the writer callback throws, clean up the partially-written file.
    try {
        std::vector<char> row_buf(rb);
        for (uint64_t i = 0; i < num_rows; ++i) {
            writer(i, row_buf.data(), rb);
            write_all(fd, row_buf.data(), rb);
        }
    } catch (...) {
        ::close(fd);
        guard.release();  // prevent double-close
        std::error_code ec;
        fs::remove(path, ec);
        throw;
    }

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "FeatureMatrix::create_streaming: fsync failed: " +
            std::string(std::strerror(errno)));
    }

    // Best-effort parent directory fsync for crash consistency
    {
        int dir_fd = ::open(path.parent_path().c_str(), O_RDONLY);
        if (dir_fd >= 0) {
            ::fsync(dir_fd);
            ::close(dir_fd);
        }
    }

    // mmap the result
    FeatureMatrix fm;
    fm.header_    = header;
    fm.path_      = path;
    fm.mmap_size_ = file_size;
    fm.mmap_ptr_  = mmap_file_readonly(path, file_size);
    return fm;
}

// --- create_parallel() ---
//
// Pre-allocates the output file and dispatches num_workers std::threads to
// write disjoint contiguous row ranges via pwrite(). The supplied RowWriter
// is invoked concurrently on different (row_id, dest) pairs and MUST be
// thread-safe. Each worker uses a private row buffer, so dest pointers do
// not alias across threads. pwrite() is atomic per call against the file
// descriptor's offset — workers writing to non-overlapping byte ranges do
// not interfere.
FeatureMatrix FeatureMatrix::create_parallel(
    const fs::path& path,
    uint64_t num_rows,
    uint64_t num_cols,
    GnnDtype dtype,
    RowWriter writer,
    unsigned num_workers)
{
    if (num_rows == 0 || num_cols == 0) {
        throw std::invalid_argument(
            "FeatureMatrix::create_parallel: num_rows and num_cols must be > 0");
    }
    if (!writer) {
        throw std::invalid_argument(
            "FeatureMatrix::create_parallel: writer must be non-null");
    }

    // Single-thread fallback — preserves existing semantics exactly.
    if (num_workers <= 1) {
        return create_streaming(path, num_rows, num_cols, dtype, std::move(writer));
    }

    auto header = FeatureMatrixHeader::make(num_rows, num_cols, dtype);

    size_t ds = dtype_size(dtype);
    if (ds > 0 && num_cols > SIZE_MAX / ds) {
        throw std::overflow_error("FeatureMatrix::create_parallel: num_cols * dtype_size overflow");
    }

    const size_t rb = header.row_bytes();
    if (rb > 0 && num_rows > SIZE_MAX / rb) {
        throw std::overflow_error(
            "FeatureMatrix::create_parallel: data size would overflow size_t");
    }
    size_t data_size = header.data_bytes();
    if (data_size > SIZE_MAX - FeatureMatrixHeader::SIZE) {
        throw std::overflow_error(
            "FeatureMatrix::create_parallel: total file size would overflow size_t");
    }
    size_t file_size = FeatureMatrixHeader::SIZE + data_size;

    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "FeatureMatrix::create_parallel: cannot open " + path.string() +
            ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    // Pre-allocate the full file. ftruncate only extends; sparse holes will be
    // filled in by the parallel pwrite() pass that follows.
    if (::ftruncate(fd, static_cast<off_t>(file_size)) != 0) {
        throw std::runtime_error(
            "FeatureMatrix::create_parallel: ftruncate(" + std::to_string(file_size) +
            ") failed: " + std::string(std::strerror(errno)));
    }

    // Header at offset 0. Workers only touch offsets >= FeatureMatrixHeader::SIZE.
    pwrite_all(fd, &header, sizeof(header), 0);

    // Cap workers at num_rows so every worker has at least 1 row.
    const uint64_t W64 = std::min<uint64_t>(num_workers, num_rows);
    const unsigned W   = static_cast<unsigned>(W64);
    const uint64_t base_chunk = num_rows / W;
    const uint64_t remainder  = num_rows % W;

    std::vector<std::thread>      threads;
    std::vector<std::exception_ptr> errors(W, nullptr);
    threads.reserve(W);

    for (unsigned w = 0; w < W; ++w) {
        // Distribute remainder across the first `remainder` workers so chunks
        // differ by at most one row. Avoids a single straggler at the tail.
        uint64_t start = w * base_chunk + std::min<uint64_t>(w, remainder);
        uint64_t end   = start + base_chunk + (w < remainder ? 1 : 0);

        threads.emplace_back([&, w, start, end]() {
            try {
                std::vector<char> row_buf(rb);
                for (uint64_t i = start; i < end; ++i) {
                    writer(i, row_buf.data(), rb);
                    off_t off = static_cast<off_t>(FeatureMatrixHeader::SIZE +
                                                   i * rb);
                    pwrite_all(fd, row_buf.data(), rb, off);
                }
            } catch (...) {
                errors[w] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();

    // Surface the first error (if any), removing the partial file.
    for (unsigned w = 0; w < W; ++w) {
        if (errors[w]) {
            ::close(fd);
            guard.release();
            std::error_code ec;
            fs::remove(path, ec);
            std::rethrow_exception(errors[w]);
        }
    }

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "FeatureMatrix::create_parallel: fsync failed: " +
            std::string(std::strerror(errno)));
    }

    {
        int dir_fd = ::open(path.parent_path().c_str(), O_RDONLY);
        if (dir_fd >= 0) {
            ::fsync(dir_fd);
            ::close(dir_fd);
        }
    }

    FeatureMatrix fm;
    fm.header_    = header;
    fm.path_      = path;
    fm.mmap_size_ = file_size;
    fm.mmap_ptr_  = mmap_file_readonly(path, file_size);
    return fm;
}

// --- create_reordered() ---

FeatureMatrix FeatureMatrix::create_reordered(
    const FeatureMatrix& source,
    const std::vector<uint64_t>& permutation,
    const fs::path& output_path)
{
    if (permutation.size() != source.num_rows()) {
        throw std::invalid_argument(
            "FeatureMatrix::create_reordered: permutation size (" +
            std::to_string(permutation.size()) + ") != source num_rows (" +
            std::to_string(source.num_rows()) + ")");
    }

    // Validate all indices before writing anything
    for (size_t i = 0; i < permutation.size(); ++i) {
        if (permutation[i] >= source.num_rows()) {
            throw std::out_of_range(
                "FeatureMatrix::create_reordered: permutation[" + std::to_string(i) +
                "] = " + std::to_string(permutation[i]) +
                " out of range [0, " + std::to_string(source.num_rows()) + ")");
        }
    }

    return create_streaming(
        output_path,
        source.num_rows(),
        source.num_cols(),
        source.dtype(),
        [&](uint64_t row_id, void* dest, uint64_t rb) {
            std::memcpy(dest, source.row(permutation[row_id]), rb);
        });
}

} // namespace mdb::gnn
