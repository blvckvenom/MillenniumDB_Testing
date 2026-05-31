#include "gnn/storage/feature_matrix.h"

#include "gnn/common/page_cache_hint.h"   // Fix #22
#include "gnn/common/pipeline_overlap.h"  // Fix #21

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
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

// Fix #20: physical pre-allocation. fallocate(2) reserves disk blocks
// up-front, so subsequent pwrites don't have to allocate extents on the
// fly. Falls back to ftruncate if fallocate is unsupported (e.g. tmpfs).
// Returns 0 on success, errno on failure.
int reserve_file_size(int fd, off_t size) {
#if defined(__linux__) && defined(FALLOC_FL_KEEP_SIZE)
    // mode=0 → allocate blocks AND grow the file. EOPNOTSUPP on
    // filesystems that don't support fallocate (e.g. older tmpfs)
    // falls through to the ftruncate path below.
    int rc = ::fallocate(fd, 0, 0, size);
    if (rc == 0) return 0;
    if (errno != EOPNOTSUPP && errno != ENOSYS) {
        return errno;
    }
#endif
    if (::ftruncate(fd, size) != 0) return errno;
    return 0;
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

// --- release_cache() ---

void FeatureMatrix::release_cache() const {
    if (mmap_ptr_ == nullptr) return;
    madvise_dontneed(mmap_ptr_, mmap_size_);
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

    // Pre-allocate via fallocate (Fix #20) to reserve physical blocks,
    // avoiding per-pwrite extent allocation. Falls back to ftruncate
    // on filesystems that don't support fallocate.
    if (int err = reserve_file_size(fd, static_cast<off_t>(file_size))) {
        throw std::runtime_error(
            "FeatureMatrix::create_parallel: reserve(" + std::to_string(file_size) +
            ") failed: " + std::string(std::strerror(err)));
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
//
// Spec D L3 reorder is the hottest path in `gnn_build_feature_store`: it
// writes the entire feature matrix (papers100M: 56.86 GB) in a new row
// permutation. The original implementation drove `create_streaming` with a
// callback that did `source.row(perm[output_row])` — random reads keyed by
// the destination row, which is uncorrelated with the source position.
// With 30 GB RAM hosting a 56 GB source mmap, the page cache thrashed and
// the phase took 3 h 21 min wall-clock on celebi.
//
// This implementation supports two strategies, selected by env var
// MDB_GNN_REORDER_STRATEGY:
//   chunked         (default, Fix #13): chunked output writes, random
//                   source reads. Best when source FM fits comfortably
//                   in RAM (e.g. < ~30% of total RAM).
//   external_sort   (Fix #15, 2026-05-13): two-pass radix partition.
//                   Pass 1 reads source sequentially and scatters rows
//                   into per-bucket temp files. Pass 2 reads each
//                   bucket sequentially, assembles the output range
//                   in memory and writes it with a single pwrite.
//                   Both passes are sequential I/O; eliminates the
//                   source page-cache thrash that dominates `chunked`
//                   when the FM exceeds ~70% of RAM.
//
// The legacy `chunked` strategy uses CHUNKED OUTPUT WRITES:
//
// Earlier history for posterity:
//   Original (pre-Fix #12, 2026-04-22): per-row write_all with random
//     mmap reads keyed by `perm[output_row]`. 3 h 21 min on papers100M.
//   Fix #12 (2026-05-13, intermediate): sorted (src,out) moves, per-row
//     pwrite to scattered offsets. 1 h 24 min — write amplification (22×)
//     dominated.
//   Fix #13 (current): chunked sequential writes. Target sub-30 min.
// Fix #15 helper: two-pass external radix sort. Used when env var
// MDB_GNN_REORDER_STRATEGY=external_sort. Pass 1 scans source mmap
// sequentially and scatters rows into per-bucket temp files keyed by
// the output position. Pass 2 reads each bucket sequentially, places
// rows into an in-memory chunk buffer at their out_local positions, and
// pwrites the contiguous chunk to the final output. Both source and
// output I/O become sequential; the only random access is across the
// (few-dozen) bucket file handles, which the kernel batches well.
FeatureMatrix FeatureMatrix::create_reordered_external_sort_(
    const FeatureMatrix& source,
    const std::vector<uint64_t>& permutation,
    const std::filesystem::path& output_path,
    uint64_t fingerprint)
{
    namespace fs = std::filesystem;
    const uint64_t N  = source.num_rows();
    const uint64_t D  = source.num_cols();
    const GnnDtype dt = source.dtype();

    auto header = FeatureMatrixHeader::make(N, D, dt);
    // STEP 8: embed the sample/feature content fingerprint in reserved[0..7].
    std::memcpy(header.reserved, &fingerprint, sizeof(fingerprint));
    const size_t rb = header.row_bytes();
    size_t data_size = header.data_bytes();
    if (data_size > SIZE_MAX - FeatureMatrixHeader::SIZE) {
        throw std::overflow_error(
            "create_reordered_external_sort: file size would overflow");
    }
    size_t file_size = FeatureMatrixHeader::SIZE + data_size;

    // Output file: open and pre-allocate
    int out_fd = ::open(output_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        throw std::runtime_error(
            "create_reordered_external_sort: cannot open " + output_path.string() +
            ": " + std::strerror(errno));
    }
    FdGuard out_guard(out_fd);
    if (int err = reserve_file_size(out_fd, static_cast<off_t>(file_size))) {
        throw std::runtime_error(
            "create_reordered_external_sort: reserve failed: " +
            std::string(std::strerror(err)));
    }
    pwrite_all(out_fd, &header, sizeof(header), 0);

    // Bucket sizing: ~1 GB per bucket → 2M rows for D=128 float32. With
    // 56 GB output that's 56 buckets, well under the FD limit.
    size_t bucket_bytes = 1ULL * 1024 * 1024 * 1024;
    if (const char* env = std::getenv("MDB_GNN_REORDER_BUCKET_MB")) {
        try {
            long long parsed = std::stoll(env);
            if (parsed > 0) bucket_bytes = static_cast<size_t>(parsed) * 1024ULL * 1024ULL;
        } catch (...) { /* ignore */ }
    }
    if (bucket_bytes < rb) bucket_bytes = rb;
    const uint64_t bucket_rows = std::max<uint64_t>(1, bucket_bytes / rb);
    const uint64_t num_buckets = (N + bucket_rows - 1) / bucket_rows;
    if (bucket_rows > static_cast<uint64_t>(UINT32_MAX)) {
        throw std::overflow_error(
            "create_reordered_external_sort: bucket_rows exceeds uint32_t");
    }

    auto temp_dir = output_path.parent_path() / ".reorder_tmp";
    fs::remove_all(temp_dir);
    fs::create_directories(temp_dir);

    auto t_start = std::chrono::high_resolution_clock::now();
    std::cerr << "[create_reordered/ext] N=" << N << " D=" << D
              << " bucket_rows=" << bucket_rows
              << " num_buckets=" << num_buckets
              << " temp_dir=" << temp_dir.string() << "\n" << std::flush;

    // --- Pass 1: split ---
    // For each src in [0, N), write (out_local: u32, row_data) to its
    // bucket file. Per-bucket 1 MB buffer to amortise small writes.
    const size_t entry_size = sizeof(uint32_t) + rb;
    const size_t per_bucket_buf = 1ULL * 1024 * 1024;

    struct Bucket {
        int fd;
        std::vector<char> buf;
        size_t used;
        uint64_t entries;
    };
    std::vector<Bucket> buckets(num_buckets);
    try {
        for (uint64_t b = 0; b < num_buckets; ++b) {
            auto p = temp_dir / ("bucket_" + std::to_string(b) + ".tmp");
            int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                throw std::runtime_error(
                    "create_reordered_external_sort: cannot create " + p.string() +
                    ": " + std::strerror(errno));
            }
            buckets[b].fd      = fd;
            buckets[b].buf.assign(per_bucket_buf, 0);
            buckets[b].used    = 0;
            buckets[b].entries = 0;
        }

        // Pass 1 reads source.row(permutation[out]) — i.e. RANDOM source rows
        // in permutation order, NOT sequential. MADV_RANDOM tells the kernel
        // to (a) skip the 128 KB readahead window (would waste I/O on pages
        // we never touch) and (b) avoid aggressive eviction (lets pages stay
        // cached for likely re-access). Bug fix 2026-05-17: previously this
        // was MADV_SEQUENTIAL which actively hurt the random pattern below.
        MadviseGuard madvise_guard(source.mmap_ptr_, source.mmap_size_, MADV_RANDOM);

        auto report_every = std::max<uint64_t>(1, N / 20);

        // Iterate by OUTPUT row (which is what perm is indexed on).
        // For each output row, read source[perm[out]] — random mmap access.
        for (uint64_t out = 0; out < N; ++out) {
            uint64_t src = permutation[out];
            uint64_t b   = out / bucket_rows;
            uint32_t out_local = static_cast<uint32_t>(out - b * bucket_rows);

            Bucket& bk = buckets[b];
            if (bk.used + entry_size > bk.buf.size()) {
                write_all(bk.fd, bk.buf.data(), bk.used);
                bk.used = 0;
            }
            std::memcpy(bk.buf.data() + bk.used, &out_local, sizeof(out_local));
            std::memcpy(bk.buf.data() + bk.used + sizeof(out_local),
                        source.row(src), rb);
            bk.used += entry_size;
            ++bk.entries;

            if ((out + 1) % report_every == 0 || out + 1 == N) {
                auto now = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration<double>(now - t_start).count();
                std::cerr << "[create_reordered/ext] pass1 " << (out + 1)
                          << "/" << N << " (" << std::fixed
                          << std::setprecision(1) << dt << "s)\n" << std::flush;
            }
        }
        for (auto& bk : buckets) {
            if (bk.used > 0) write_all(bk.fd, bk.buf.data(), bk.used);
            if (::fsync(bk.fd) < 0) {
                // Non-fatal — temp files; just log
            }
            // Fix #22: bucket file is sequential append-only and now fully
            // flushed; clean pages can be evicted before Pass 2 starts
            // reading bucket files. Without this hint the kernel keeps the
            // 56+57 GB of bucket-file pages resident, evicting productive
            // reordered.fmat pages under memory pressure.
            fadvise_dontneed(bk.fd, 0, 0);  // 0,0 = "entire file"
            ::close(bk.fd);
            bk.fd = -1;
        }
    } catch (...) {
        for (auto& bk : buckets) {
            if (bk.fd >= 0) ::close(bk.fd);
        }
        fs::remove_all(temp_dir);
        ::close(out_fd);
        out_guard.release();
        std::error_code ec;
        fs::remove(output_path, ec);
        throw;
    }

    auto t_pass1 = std::chrono::high_resolution_clock::now();
    std::cerr << "[create_reordered/ext] pass1 done ("
              << std::chrono::duration<double>(t_pass1 - t_start).count()
              << "s)\n" << std::flush;

    // Fix #22: source FM was scanned sequentially in Pass 1; pages are
    // no longer needed for Pass 2 (which only reads bucket files).
    // Release them before allocating the per-chunk buffer for Pass 2.
    // On papers100M (56 GB source) this is the single biggest page-cache
    // liberation in the whole feature-store build.
    madvise_dontneed(source.mmap_ptr_, source.mmap_size_);

    // Fix #21: opt-in pipelined Pass 2. Producer (open + read + scatter)
    // overlaps with consumer (pwrite + close + remove + log) via a
    // bounded queue of capacity 2 (DiskGNN §6). Disabled by default to
    // preserve byte-identical behavior with the legacy serial path.
    bool use_pipeline_overlap = false;
    if (const char* env = std::getenv("MDB_GNN_PIPELINE_OVERLAP")) {
        std::string s(env);
        if (s == "1" || s == "true" || s == "yes") use_pipeline_overlap = true;
    }

    // --- Pass 2: merge ---
    // For each bucket: open file, read all entries, scatter into
    // chunk_buf at out_local positions, sequential pwrite of chunk.
    if (use_pipeline_overlap) {
        struct ReadyBucket {
            std::vector<char>      chunk_buf;
            uint64_t               bucket_id;
            uint64_t               chunk_start;
            uint64_t               actual_rows;
            std::filesystem::path  bucket_path;
        };

        ChunkPipeline<ReadyBucket> pipe(2);

        std::thread producer([&]() {
            try {
                for (uint64_t b = 0; b < num_buckets; ++b) {
                    auto bp = temp_dir / ("bucket_" + std::to_string(b) + ".tmp");
                    int in_fd = ::open(bp.c_str(), O_RDONLY);
                    if (in_fd < 0) {
                        throw std::runtime_error(
                            "create_reordered_external_sort: cannot read bucket " +
                            bp.string());
                    }
                    FdGuard in_guard(in_fd);

                    std::vector<char> chunk_buf(bucket_rows * rb, 0);

                    std::vector<char> read_buf(4 * 1024 * 1024);
                    size_t buf_pos = 0, buf_end = 0;
                    auto fill_buf = [&](size_t need) -> bool {
                        if (buf_end - buf_pos >= need) return true;
                        size_t remaining = buf_end - buf_pos;
                        std::memmove(read_buf.data(),
                                     read_buf.data() + buf_pos, remaining);
                        buf_pos = 0; buf_end = remaining;
                        while (buf_end < need && buf_end < read_buf.size()) {
                            ssize_t r = ::read(in_fd, read_buf.data() + buf_end,
                                               read_buf.size() - buf_end);
                            if (r < 0) {
                                if (errno == EINTR) continue;
                                throw std::runtime_error("ext_sort: read failed");
                            }
                            if (r == 0) break;
                            buf_end += r;
                        }
                        return buf_end - buf_pos >= need;
                    };

                    for (uint64_t e = 0; e < buckets[b].entries; ++e) {
                        if (!fill_buf(entry_size)) {
                            throw std::runtime_error("ext_sort: truncated bucket");
                        }
                        uint32_t out_local;
                        std::memcpy(&out_local, read_buf.data() + buf_pos,
                                    sizeof(out_local));
                        buf_pos += sizeof(out_local);
                        std::memcpy(chunk_buf.data() + out_local * rb,
                                    read_buf.data() + buf_pos, rb);
                        buf_pos += rb;
                    }

                    // Fix #22: bucket file fully consumed — release its pages
                    // before close (close() doesn't itself flush page cache).
                    fadvise_dontneed(in_fd, 0, 0);  // 0,0 = "entire file"

                    uint64_t chunk_start = b * bucket_rows;
                    uint64_t actual = std::min(bucket_rows, N - chunk_start);
                    pipe.push(ReadyBucket{
                        std::move(chunk_buf), b, chunk_start, actual, bp});
                }
                pipe.close();
            } catch (...) {
                pipe.set_error(std::current_exception());
            }
        });

        try {
            while (auto rb_ready_opt = pipe.pop()) {
                auto& rb_ready = *rb_ready_opt;
                off_t off = static_cast<off_t>(
                    FeatureMatrixHeader::SIZE + rb_ready.chunk_start * rb);
                pwrite_all(out_fd, rb_ready.chunk_buf.data(),
                           rb_ready.actual_rows * rb, off);
                // Fix #22: the chunk's output pages are now flushed; hint
                // for eviction before the next bucket arrives.
                fadvise_dontneed(out_fd, off,
                                 static_cast<off_t>(rb_ready.actual_rows * rb));
                // Now safe to remove the consumed bucket file.
                std::filesystem::remove(rb_ready.bucket_path);

                auto now = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration<double>(now - t_start).count();
                std::cerr << "[create_reordered/ext] pass2 bucket "
                          << (rb_ready.bucket_id + 1) << "/" << num_buckets
                          << " (pipeline, " << std::fixed << std::setprecision(1)
                          << dt << "s)\n" << std::flush;
            }
            producer.join();
        } catch (...) {
            // Critical: drain + join even on error so the std::thread destructor
            // doesn't trigger std::terminate(). Same pattern as Task 3.
            pipe.set_error(std::current_exception());
            if (producer.joinable()) producer.join();
            fs::remove_all(temp_dir);
            ::close(out_fd);
            out_guard.release();
            std::error_code ec;
            fs::remove(output_path, ec);
            throw;
        }
    } else {
        std::vector<char> chunk_buf(bucket_rows * rb);
        try {
            for (uint64_t b = 0; b < num_buckets; ++b) {
                // Zero-fill in case the bucket has fewer entries than bucket_rows
                // (mostly for the final bucket; also guards against missing
                // entries from a non-injective permutation).
                std::memset(chunk_buf.data(), 0, chunk_buf.size());

                auto p = temp_dir / ("bucket_" + std::to_string(b) + ".tmp");
                int in_fd = ::open(p.c_str(), O_RDONLY);
                if (in_fd < 0) {
                    throw std::runtime_error(
                        "create_reordered_external_sort: cannot read bucket " +
                        p.string());
                }
                FdGuard in_guard(in_fd);

                // Buffered sequential read
                std::vector<char> read_buf(4 * 1024 * 1024);
                size_t buf_pos = 0;
                size_t buf_end = 0;
                auto fill_buf = [&](size_t need) -> bool {
                    if (buf_end - buf_pos >= need) return true;
                    size_t remaining = buf_end - buf_pos;
                    std::memmove(read_buf.data(), read_buf.data() + buf_pos, remaining);
                    buf_pos = 0;
                    buf_end = remaining;
                    while (buf_end < need && buf_end < read_buf.size()) {
                        ssize_t r = ::read(in_fd, read_buf.data() + buf_end,
                                           read_buf.size() - buf_end);
                        if (r < 0) {
                            if (errno == EINTR) continue;
                            throw std::runtime_error("ext_sort: read failed");
                        }
                        if (r == 0) break;
                        buf_end += r;
                    }
                    return buf_end - buf_pos >= need;
                };

                for (uint64_t e = 0; e < buckets[b].entries; ++e) {
                    if (!fill_buf(entry_size)) {
                        throw std::runtime_error("ext_sort: truncated bucket");
                    }
                    uint32_t out_local;
                    std::memcpy(&out_local, read_buf.data() + buf_pos, sizeof(out_local));
                    buf_pos += sizeof(out_local);
                    std::memcpy(chunk_buf.data() + out_local * rb,
                                read_buf.data() + buf_pos, rb);
                    buf_pos += rb;
                }
                ::close(in_fd);
                in_guard.release();
                fs::remove(p);

                uint64_t chunk_start = b * bucket_rows;
                uint64_t chunk_actual_rows = std::min(bucket_rows, N - chunk_start);
                off_t off = static_cast<off_t>(
                    FeatureMatrixHeader::SIZE + chunk_start * rb);
                pwrite_all(out_fd, chunk_buf.data(), chunk_actual_rows * rb, off);
                // Fix #22: hint kernel that this chunk's output pages can
                // leave the page cache. Mirrors the pipeline-branch hint.
                fadvise_dontneed(out_fd, off,
                                 static_cast<off_t>(chunk_actual_rows * rb));

                auto now = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration<double>(now - t_start).count();
                std::cerr << "[create_reordered/ext] pass2 bucket " << (b + 1)
                          << "/" << num_buckets << " (" << std::fixed
                          << std::setprecision(1) << dt << "s)\n" << std::flush;
            }
        } catch (...) {
            fs::remove_all(temp_dir);
            ::close(out_fd);
            out_guard.release();
            std::error_code ec;
            fs::remove(output_path, ec);
            throw;
        }
    }

    auto t_pass2 = std::chrono::high_resolution_clock::now();
    std::cerr << "[create_reordered/ext] pass2 done ("
              << std::chrono::duration<double>(t_pass2 - t_pass1).count()
              << "s)\n" << std::flush;

    if (::fsync(out_fd) < 0) {
        throw std::runtime_error(
            "create_reordered_external_sort: fsync failed");
    }
    {
        int dir_fd = ::open(output_path.parent_path().c_str(), O_RDONLY);
        if (dir_fd >= 0) { ::fsync(dir_fd); ::close(dir_fd); }
    }
    fs::remove_all(temp_dir);

    ::close(out_fd);
    out_guard.release();
    return FeatureMatrix::open(output_path);
}

FeatureMatrix FeatureMatrix::create_reordered(
    const FeatureMatrix& source,
    const std::vector<uint64_t>& permutation,
    const fs::path& output_path,
    uint64_t fingerprint)
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

    const uint64_t N  = source.num_rows();
    const uint64_t D  = source.num_cols();
    const GnnDtype dt = source.dtype();

    // Strategy selector — env-var driven. Default `chunked` (Fix #13),
    // which is safer when temp disk space is tight (external_sort
    // requires ~N * row_bytes extra disk for the bucket files).
    bool use_external_sort = false;
    if (const char* env = std::getenv("MDB_GNN_REORDER_STRATEGY")) {
        std::string s(env);
        if (s == "external_sort" || s == "ext_sort" || s == "external") {
            use_external_sort = true;
        }
    }
    if (use_external_sort && N > 0) {
        return create_reordered_external_sort_(source, permutation, output_path, fingerprint);
    }

    if (N == 0) {
        // Degenerate case — fall back to the simple writer (which throws
        // on N==0 to preserve the historical contract).
        return create_streaming(output_path, N, D, dt,
            [&](uint64_t, void*, uint64_t) { /* unreachable */ });
    }

    auto header = FeatureMatrixHeader::make(N, D, dt);
    // STEP 8: embed the sample/feature content fingerprint in reserved[0..7].
    std::memcpy(header.reserved, &fingerprint, sizeof(fingerprint));
    const size_t rb = header.row_bytes();
    if (rb > 0 && N > SIZE_MAX / rb) {
        throw std::overflow_error(
            "FeatureMatrix::create_reordered: data size would overflow size_t");
    }
    size_t data_size = header.data_bytes();
    if (data_size > SIZE_MAX - FeatureMatrixHeader::SIZE) {
        throw std::overflow_error(
            "FeatureMatrix::create_reordered: total file size would overflow size_t");
    }
    size_t file_size = FeatureMatrixHeader::SIZE + data_size;

    int fd = ::open(output_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "FeatureMatrix::create_reordered: cannot open " + output_path.string() +
            ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    if (int err = reserve_file_size(fd, static_cast<off_t>(file_size))) {
        throw std::runtime_error(
            "FeatureMatrix::create_reordered: reserve(" + std::to_string(file_size) +
            ") failed: " + std::string(std::strerror(err)));
    }
    pwrite_all(fd, &header, sizeof(header), 0);

    try {
        auto t_start = std::chrono::high_resolution_clock::now();
        std::cerr << "[create_reordered] N=" << N << " D=" << D
                  << " row_bytes=" << rb
                  << " file_size_mb=" << (file_size / 1024 / 1024)
                  << "\n" << std::flush;

        // --- Chunk sizing ---
        // Default 1 GB chunk = ~1M rows for D=256/float32. Smaller chunks
        // reduce peak memory at the cost of more output syscalls (still
        // tiny compared to 111M one-row pwrites). Larger chunks improve
        // sequential write throughput but eat RAM that competes with the
        // source mmap page cache.
        size_t chunk_bytes = 1ULL * 1024 * 1024 * 1024;  // 1 GB
        if (const char* env = std::getenv("MDB_GNN_REORDER_CHUNK_MB")) {
            try {
                long long parsed = std::stoll(env);
                if (parsed > 0) {
                    chunk_bytes = static_cast<size_t>(parsed) * 1024ULL * 1024ULL;
                }
            } catch (...) {
                // ignore malformed env value
            }
        }
        if (chunk_bytes < rb) chunk_bytes = rb;  // at least one row
        uint64_t chunk_rows = std::max<uint64_t>(1, chunk_bytes / rb);
        uint64_t num_chunks = (N + chunk_rows - 1) / chunk_rows;

        // --- Worker count ---
        // Default 2 workers — each holds a chunk_bytes buffer, so total
        // worker heap = 2 * chunk_bytes (= 2 GB by default). Override
        // via env MDB_GNN_REORDER_WORKERS. More workers help when the
        // page cache has free room AND chunks are small enough that
        // their buffers + the source working-set fit in RAM together.
        unsigned num_workers = 2;
        if (const char* env = std::getenv("MDB_GNN_REORDER_WORKERS")) {
            try {
                int parsed = std::stoi(env);
                if (parsed > 0) num_workers = static_cast<unsigned>(parsed);
            } catch (...) {
                // ignore malformed env value
            }
        }
        if (num_workers > std::thread::hardware_concurrency() &&
            std::thread::hardware_concurrency() > 0)
        {
            num_workers = std::thread::hardware_concurrency();
        }
        if (num_workers > num_chunks) {
            num_workers = static_cast<unsigned>(num_chunks);
        }
        if (num_workers == 0) num_workers = 1;

        std::cerr << "[create_reordered] chunk_rows=" << chunk_rows
                  << " chunk_bytes_mb=" << (chunk_bytes / 1024 / 1024)
                  << " num_chunks=" << num_chunks
                  << " num_workers=" << num_workers
                  << "\n" << std::flush;

        // Random access pattern across source — hint the kernel so it
        // skips wasteful readahead. The default (MADV_NORMAL) does a
        // 128 KB readahead on every page fault, which is pure waste
        // when consecutive reads land on unrelated pages.
        MadviseGuard madvise_guard(source.mmap_ptr_, source.mmap_size_, MADV_RANDOM);

        // Fix #21: opt-in pipeline overlap mode.
        // One producer thread packs the next chunk while the main consumer
        // pwrites the previous one. Bounded queue (capacity 2, DiskGNN §6).
        // MDB_GNN_REORDER_WORKERS is ignored in this mode — pipeline already
        // provides I/O/compute overlap without the kernel-side pwrite contention
        // that multi-worker concurrent pwrites cause on a single NVMe.
        bool use_pipeline_overlap = false;
        if (const char* env = std::getenv("MDB_GNN_PIPELINE_OVERLAP")) {
            std::string s(env);
            if (s == "1" || s == "true" || s == "yes") use_pipeline_overlap = true;
        }

        auto t_write_start_outer = std::chrono::high_resolution_clock::now();

        if (use_pipeline_overlap) {
            struct ReadyChunk {
                std::vector<char> buf;
                uint64_t          out_start;
                uint64_t          rows;
            };

            ChunkPipeline<ReadyChunk> pipe(2);
            std::atomic<uint64_t> chunks_done{0};
            auto t_write_start = std::chrono::high_resolution_clock::now();
            t_write_start_outer = t_write_start;

            std::thread producer([&]() {
                try {
                    for (uint64_t c = 0; c < num_chunks; ++c) {
                        uint64_t out_start   = c * chunk_rows;
                        uint64_t out_end     = std::min(out_start + chunk_rows, N);
                        uint64_t actual_rows = out_end - out_start;
                        ReadyChunk rc{
                            std::vector<char>(actual_rows * rb),
                            out_start,
                            actual_rows
                        };
                        for (uint64_t i = 0; i < actual_rows; ++i) {
                            uint64_t src = permutation[out_start + i];
                            std::memcpy(rc.buf.data() + i * rb, source.row(src), rb);
                        }
                        pipe.push(std::move(rc));
                    }
                    pipe.close();
                } catch (...) {
                    pipe.set_error(std::current_exception());
                }
            });

            try {
                while (auto rc_opt = pipe.pop()) {
                    auto& rc = *rc_opt;
                    off_t off = static_cast<off_t>(
                        FeatureMatrixHeader::SIZE + rc.out_start * rb);
                    pwrite_all(fd, rc.buf.data(), rc.rows * rb, off);
                    // Fix #22: this chunk's output pages are now clean and on-disk;
                    // hint the kernel that they can leave the page cache so
                    // subsequent chunks aren't competing with stale pages.
                    fadvise_dontneed(fd, off, static_cast<off_t>(rc.rows * rb));

                    uint64_t done = chunks_done.fetch_add(1,
                        std::memory_order_relaxed) + 1;
                    auto now = std::chrono::high_resolution_clock::now();
                    double dt = std::chrono::duration<double>(now - t_write_start).count();
                    double rate_mbs = (done * chunk_bytes / 1048576.0) / dt;
                    std::cerr << "[create_reordered] chunk " << done
                              << "/" << num_chunks << " (pipeline, "
                              << std::fixed << std::setprecision(1)
                              << dt << "s elapsed, "
                              << rate_mbs << " MB/s avg logical)\n"
                              << std::flush;
                }
                producer.join();
            } catch (...) {
                // Signal producer to stop (in case it's blocked on push), then
                // join to honor the std::thread destruction contract. Both calls
                // are no-throw on the ChunkPipeline / std::thread side once
                // notified, so we can safely re-throw the original exception.
                pipe.set_error(std::current_exception());
                if (producer.joinable()) producer.join();
                throw;
            }
        } else {
            // Legacy multi-worker path — preserved exactly as-is below.
            std::vector<std::thread>      threads;
            std::vector<std::exception_ptr> errors(num_workers, nullptr);
            threads.reserve(num_workers);

            std::atomic<uint64_t> next_chunk{0};
            std::atomic<uint64_t> chunks_done{0};
            auto t_write_start = std::chrono::high_resolution_clock::now();
            t_write_start_outer = t_write_start;

            for (unsigned w = 0; w < num_workers; ++w) {
                threads.emplace_back([&, w]() {
                    try {
                        // Pre-allocate chunk buffer (max possible size). Reused
                        // across chunks to avoid per-chunk allocator churn.
                        std::vector<char> chunk_buf(chunk_rows * rb);
                        while (true) {
                            uint64_t c = next_chunk.fetch_add(1, std::memory_order_relaxed);
                            if (c >= num_chunks) break;

                            uint64_t out_start = c * chunk_rows;
                            uint64_t out_end   = std::min(out_start + chunk_rows, N);
                            uint64_t actual_rows = out_end - out_start;

                            // Gather random source rows into the contiguous
                            // chunk buffer. Reads are scattered over `source`
                            // mmap; this is where most of the wall-clock for
                            // a chunk goes (page faults on cold source pages).
                            for (uint64_t i = 0; i < actual_rows; ++i) {
                                uint64_t src = permutation[out_start + i];
                                std::memcpy(chunk_buf.data() + i * rb,
                                            source.row(src), rb);
                            }

                            // One large pwrite per chunk. Replaces ~1M small
                            // pwrites and avoids the 22× write amplification
                            // observed on the per-row variant.
                            off_t off = static_cast<off_t>(
                                FeatureMatrixHeader::SIZE + out_start * rb);
                            pwrite_all(fd, chunk_buf.data(), actual_rows * rb, off);
                            // Fix #22: hint kernel that this chunk's output
                            // pages can leave the page cache. Mirrors the
                            // pipeline-branch hint for symmetric coverage.
                            fadvise_dontneed(fd, off,
                                             static_cast<off_t>(actual_rows * rb));

                            uint64_t done = chunks_done.fetch_add(1,
                                std::memory_order_relaxed) + 1;
                            auto now = std::chrono::high_resolution_clock::now();
                            double dt = std::chrono::duration<double>(
                                now - t_write_start).count();
                            double rate_mbs = (done * chunk_bytes / 1048576.0) / dt;
                            std::cerr << "[create_reordered] chunk " << done
                                      << "/" << num_chunks << " (w" << w
                                      << ", rows " << out_start << ".."
                                      << out_end << ", "
                                      << std::fixed << std::setprecision(1)
                                      << dt << "s elapsed, "
                                      << rate_mbs << " MB/s avg logical)\n"
                                      << std::flush;
                        }
                    } catch (...) {
                        errors[w] = std::current_exception();
                    }
                });
            }

            for (auto& t : threads) t.join();
            for (unsigned w = 0; w < num_workers; ++w) {
                if (errors[w]) std::rethrow_exception(errors[w]);
            }
        }

        auto t_write_end = std::chrono::high_resolution_clock::now();
        double total_dt = std::chrono::duration<double>(
            t_write_end - t_write_start_outer).count();
        double final_rate_mbs = (N * rb / 1048576.0) / total_dt;
        std::cerr << "[create_reordered] write phase done ("
                  << std::fixed << std::setprecision(1)
                  << total_dt << "s, " << final_rate_mbs
                  << " MB/s avg logical)\n" << std::flush;
    } catch (...) {
        ::close(fd);
        guard.release();
        std::error_code ec;
        fs::remove(output_path, ec);
        throw;
    }

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "FeatureMatrix::create_reordered: fsync failed: " +
            std::string(std::strerror(errno)));
    }
    {
        int dir_fd = ::open(output_path.parent_path().c_str(), O_RDONLY);
        if (dir_fd >= 0) {
            ::fsync(dir_fd);
            ::close(dir_fd);
        }
    }

    FeatureMatrix fm;
    fm.header_    = header;
    fm.path_      = output_path;
    fm.mmap_size_ = file_size;
    fm.mmap_ptr_  = mmap_file_readonly(output_path, file_size);
    return fm;
}

} // namespace mdb::gnn
