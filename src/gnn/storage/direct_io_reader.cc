#include "gnn/storage/direct_io_reader.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef ENABLE_IO_URING
#include <liburing.h>
#endif

namespace fs = std::filesystem;

namespace mdb::gnn {

// =============================================================================
// Static helpers
// =============================================================================

AlignedBuffer DirectIoReader::alloc_aligned(size_t size, size_t alignment) {
    if (size == 0) {
        return AlignedBuffer(nullptr);
    }
    void* ptr = nullptr;
    int rc = ::posix_memalign(&ptr, alignment, size);
    if (rc != 0) {
        throw std::runtime_error(
            "DirectIoReader: posix_memalign failed (size=" +
            std::to_string(size) + ", align=" + std::to_string(alignment) +
            "): " + std::strerror(rc));
    }
    return AlignedBuffer(static_cast<char*>(ptr));
}

uint64_t DirectIoReader::align_down(uint64_t val, uint64_t align) {
    return val & ~(align - 1);
}

uint64_t DirectIoReader::align_up(uint64_t val, uint64_t align) {
    return (val + align - 1) & ~(align - 1);
}

// =============================================================================
// Constructor
// =============================================================================

DirectIoReader::DirectIoReader(const fs::path& file_path) {
    if (!fs::exists(file_path)) {
        throw std::runtime_error(
            "DirectIoReader: file not found: " + file_path.string());
    }

    file_size_ = fs::file_size(file_path);

    // Detect filesystem block size for alignment
    struct stat st;
    if (::stat(file_path.c_str(), &st) == 0 && st.st_blksize > 0) {
        block_align_ = static_cast<size_t>(st.st_blksize);
        // Ensure block_align_ is a power of two (required for align_down/align_up)
        if ((block_align_ & (block_align_ - 1)) != 0) {
            block_align_ = 4096;  // fallback to safe default
        }
    }

    // Try O_DIRECT first
    fd_ = ::open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ >= 0) {
        direct_ = true;
    } else {
        // Fallback: some filesystems (tmpfs, some NFS) don't support O_DIRECT.
        // Also EINVAL on some older kernels for unaligned file sizes.
        fd_ = ::open(file_path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error(
                "DirectIoReader: cannot open " + file_path.string() +
                ": " + std::strerror(errno));
        }
        direct_ = false;
    }

#ifdef ENABLE_IO_URING
    if (::io_uring_queue_init(QUEUE_DEPTH, &ring_, 0) == 0) {
        ring_initialized_ = true;
        io_uring_active_  = true;
    }
    // If io_uring_queue_init fails (e.g., kernel too old, seccomp), fall back
    // silently to pread. This is not an error.
#endif
}

// =============================================================================
// Destructor
// =============================================================================

DirectIoReader::~DirectIoReader() {
#ifdef ENABLE_IO_URING
    if (ring_initialized_) {
        ::io_uring_queue_exit(&ring_);
    }
#endif
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

// =============================================================================
// pread_all — synchronous fallback
// =============================================================================

void DirectIoReader::pread_all(void* buf, size_t count, off_t offset) {
    char* p = static_cast<char*>(buf);
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t n = ::pread(fd_, p, remaining, offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "DirectIoReader: pread failed at offset " +
                std::to_string(offset) + ": " + std::strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error(
                "DirectIoReader: unexpected EOF at offset " +
                std::to_string(offset) + " (wanted " +
                std::to_string(remaining) + " more bytes)");
        }
        p += n;
        offset += static_cast<off_t>(n);
        remaining -= static_cast<size_t>(n);
    }
}

// =============================================================================
// advise_dontneed — page cache eviction hint (non-O_DIRECT mode)
// =============================================================================

void DirectIoReader::advise_dontneed(off_t offset, size_t len) {
    // Advisory: failures are silently ignored.
    ::posix_fadvise(fd_, offset, static_cast<off_t>(len), POSIX_FADV_DONTNEED);
}

// =============================================================================
// read_aligned_region — O_DIRECT single-region helper
// =============================================================================

size_t DirectIoReader::read_aligned_region(
    char* dest,
    uint64_t file_offset,
    uint64_t wanted_bytes)
{
    if (wanted_bytes == 0) return 0;

    if (!direct_) {
        // No alignment needed — just pread directly
        pread_all(dest, wanted_bytes, static_cast<off_t>(file_offset));
        advise_dontneed(static_cast<off_t>(file_offset), wanted_bytes);
        return wanted_bytes;
    }

    // O_DIRECT path: align offset down, size up
    uint64_t aligned_off  = align_down(file_offset, block_align_);
    uint64_t skip         = file_offset - aligned_off;
    uint64_t end          = file_offset + wanted_bytes;
    // I7: Clamp to actual file size boundary, not the aligned file size.
    // O_DIRECT requires aligned read sizes, so align_up to block boundary,
    // but never exceed the file-size-aligned ceiling. For non-O_DIRECT reads
    // that somehow reach here, clamp to exact file size.
    if (end > file_size_) end = file_size_;
    uint64_t aligned_end  = align_up(end, block_align_);
    uint64_t aligned_fs   = align_up(file_size_, block_align_);
    if (aligned_end > aligned_fs) {
        aligned_end = aligned_fs;
    }
    uint64_t aligned_size = aligned_end - aligned_off;

    if (aligned_size == 0) return 0;

    auto tmp = alloc_aligned(aligned_size, block_align_);
    pread_all(tmp.get(), aligned_size, static_cast<off_t>(aligned_off));
    std::memcpy(dest, tmp.get() + skip, wanted_bytes);
    return static_cast<size_t>(aligned_size);
}

// =============================================================================
// io_uring batch submission
// =============================================================================

#ifdef ENABLE_IO_URING

void DirectIoReader::submit_io_uring(
    const std::vector<IoOp>& ops,
    char* aligned_buf,
    char* out_buf)
{
    size_t submitted = 0;
    size_t total_ops = ops.size();

    while (submitted < total_ops) {
        // Submit up to QUEUE_DEPTH at a time
        size_t batch_end = std::min(submitted + QUEUE_DEPTH, total_ops);
        size_t batch_size = batch_end - submitted;

        for (size_t i = submitted; i < batch_end; ++i) {
            const auto& op = ops[i];
            struct io_uring_sqe* sqe = ::io_uring_get_sqe(&ring_);
            if (!sqe) {
                // Queue full — should not happen within QUEUE_DEPTH, but handle it
                ::io_uring_submit(&ring_);
                // Drain completions for what we submitted so far in this batch
                for (size_t j = submitted; j < i; ++j) {
                    struct io_uring_cqe* cqe;
                    if (::io_uring_wait_cqe(&ring_, &cqe) < 0) {
                        throw std::runtime_error(
                            "DirectIoReader: io_uring_wait_cqe failed");
                    }
                    if (cqe->res < 0) {
                        ::io_uring_cqe_seen(&ring_, cqe);
                        throw std::runtime_error(
                            "DirectIoReader: io_uring read failed: " +
                            std::string(std::strerror(-cqe->res)));
                    }
                    ::io_uring_cqe_seen(&ring_, cqe);
                }
                // Retry getting SQE
                sqe = ::io_uring_get_sqe(&ring_);
                if (!sqe) {
                    throw std::runtime_error(
                        "DirectIoReader: io_uring_get_sqe failed after drain");
                }
            }
            ::io_uring_prep_read(
                sqe, fd_,
                aligned_buf + op.buf_offset,
                static_cast<unsigned>(op.aligned_size),
                static_cast<off_t>(op.file_offset));
            sqe->user_data = static_cast<uint64_t>(i);
        }

        int ret = ::io_uring_submit(&ring_);
        if (ret < 0) {
            throw std::runtime_error(
                "DirectIoReader: io_uring_submit failed: " +
                std::string(std::strerror(-ret)));
        }

        // Collect completions
        for (size_t i = 0; i < batch_size; ++i) {
            struct io_uring_cqe* cqe;
            if (::io_uring_wait_cqe(&ring_, &cqe) < 0) {
                throw std::runtime_error(
                    "DirectIoReader: io_uring_wait_cqe failed");
            }
            if (cqe->res < 0) {
                int err = -cqe->res;
                ::io_uring_cqe_seen(&ring_, cqe);
                throw std::runtime_error(
                    "DirectIoReader: io_uring read error: " +
                    std::string(std::strerror(err)));
            }
            ::io_uring_cqe_seen(&ring_, cqe);
        }

        // Copy wanted bytes from aligned scratch to output
        for (size_t i = submitted; i < batch_end; ++i) {
            const auto& op = ops[i];
            std::memcpy(
                out_buf + op.dest_offset,
                aligned_buf + op.buf_offset + op.skip,
                op.wanted);
        }

        submitted = batch_end;
    }
}

#endif // ENABLE_IO_URING

// =============================================================================
// read_rows — main row-reading API
// =============================================================================

DirectIoReader::ReadResult DirectIoReader::read_rows(
    const std::vector<uint64_t>& row_indices,
    uint64_t row_bytes,
    uint64_t data_offset)
{
    if (row_indices.empty() || row_bytes == 0) {
        return {AlignedBuffer(nullptr), 0, 0, 0};
    }

    size_t total_bytes = row_indices.size() * row_bytes;
    auto out = alloc_aligned(total_bytes, block_align_);
    // Zero-fill to ensure deterministic output for any edge case
    std::memset(out.get(), 0, total_bytes);

    // Build read operations sorted by file offset for sequential I/O
    struct ReadOp {
        uint64_t file_offset;
        size_t   out_offset;  // position in output buffer (original order)
    };
    std::vector<ReadOp> ops;
    ops.reserve(row_indices.size());
    for (size_t i = 0; i < row_indices.size(); ++i) {
        ops.push_back({
            data_offset + row_indices[i] * row_bytes,
            i * static_cast<size_t>(row_bytes)
        });
    }
    // Sort by file offset for better I/O scheduling
    std::sort(ops.begin(), ops.end(),
              [](const auto& a, const auto& b) {
                  return a.file_offset < b.file_offset;
              });

#ifdef ENABLE_IO_URING
    if (io_uring_active_ && direct_) {
        // Build aligned I/O operations for io_uring batch submission
        std::vector<IoOp> io_ops;
        io_ops.reserve(ops.size());
        size_t scratch_offset = 0;

        for (const auto& op : ops) {
            uint64_t aligned_off  = align_down(op.file_offset, block_align_);
            uint64_t skip         = op.file_offset - aligned_off;
            uint64_t end          = op.file_offset + row_bytes;
            uint64_t aligned_end  = align_up(end, block_align_);
            // Clamp to aligned file size
            uint64_t aligned_fs   = align_up(file_size_, block_align_);
            if (aligned_end > aligned_fs) aligned_end = aligned_fs;
            uint64_t aligned_size = aligned_end - aligned_off;

            // C3: Overflow check before accumulating scratch_offset
            if (scratch_offset > SIZE_MAX - aligned_size) {
                throw std::overflow_error(
                    "DirectIoReader: scratch buffer size overflow");
            }

            io_ops.push_back({
                aligned_off,
                aligned_size,
                scratch_offset,
                op.out_offset,
                skip,
                row_bytes
            });
            scratch_offset += aligned_size;
        }

        // Allocate a single aligned scratch buffer for all aligned reads
        auto scratch = alloc_aligned(scratch_offset, block_align_);
        submit_io_uring(io_ops, scratch.get(), out.get());

        // bytes_disk = scratch_offset (sum of all aligned read sizes
        // submitted to io_uring). Equals total_bytes only if every row
        // happened to be block-aligned and contiguous; otherwise reflects
        // alignment overhead — exactly the metric Spec A2 will reduce.
        return {std::move(out), total_bytes, row_indices.size(), scratch_offset};
    }
#endif

    // Synchronous path: pread (with alignment handling if O_DIRECT).
    // read_aligned_region returns the OS-level bytes read per call; sum
    // them for paper-comparable bytes_disk accounting.
    size_t bytes_disk_total = 0;
    for (const auto& op : ops) {
        bytes_disk_total += read_aligned_region(
            out.get() + op.out_offset,
            op.file_offset,
            row_bytes);
    }

    return {std::move(out), total_bytes, row_indices.size(), bytes_disk_total};
}

// =============================================================================
// read_range — contiguous byte range
// =============================================================================

DirectIoReader::ReadResult DirectIoReader::read_range(uint64_t offset, uint64_t size) {
    if (size == 0) {
        return {AlignedBuffer(nullptr), 0, 0, 0};
    }

    auto out = alloc_aligned(size, block_align_);

    if (!direct_) {
        // Simple pread path — no alignment overhead, bytes_disk == size.
        pread_all(out.get(), size, static_cast<off_t>(offset));
        advise_dontneed(static_cast<off_t>(offset), size);
        return {std::move(out), size, 0, size};
    }

    // O_DIRECT path: read aligned region, extract wanted bytes
    uint64_t aligned_off  = align_down(offset, block_align_);
    uint64_t skip         = offset - aligned_off;
    uint64_t aligned_end  = align_up(offset + size, block_align_);
    uint64_t aligned_fs   = align_up(file_size_, block_align_);
    if (aligned_end > aligned_fs) aligned_end = aligned_fs;
    uint64_t aligned_size = aligned_end - aligned_off;

#ifdef ENABLE_IO_URING
    if (io_uring_active_) {
        auto scratch = alloc_aligned(aligned_size, block_align_);

        struct io_uring_sqe* sqe = ::io_uring_get_sqe(&ring_);
        if (!sqe) {
            throw std::runtime_error(
                "DirectIoReader: io_uring_get_sqe failed");
        }
        ::io_uring_prep_read(
            sqe, fd_,
            scratch.get(),
            static_cast<unsigned>(aligned_size),
            static_cast<off_t>(aligned_off));
        sqe->user_data = 0;

        int ret = ::io_uring_submit(&ring_);
        if (ret < 0) {
            throw std::runtime_error(
                "DirectIoReader: io_uring_submit failed: " +
                std::string(std::strerror(-ret)));
        }

        struct io_uring_cqe* cqe;
        if (::io_uring_wait_cqe(&ring_, &cqe) < 0) {
            throw std::runtime_error(
                "DirectIoReader: io_uring_wait_cqe failed");
        }
        if (cqe->res < 0) {
            int err = -cqe->res;
            ::io_uring_cqe_seen(&ring_, cqe);
            throw std::runtime_error(
                "DirectIoReader: io_uring range read error: " +
                std::string(std::strerror(err)));
        }
        ::io_uring_cqe_seen(&ring_, cqe);

        std::memcpy(out.get(), scratch.get() + skip, size);
        return {std::move(out), size, 0, static_cast<size_t>(aligned_size)};
    }
#endif

    // Synchronous O_DIRECT fallback
    auto scratch = alloc_aligned(aligned_size, block_align_);
    pread_all(scratch.get(), aligned_size, static_cast<off_t>(aligned_off));
    std::memcpy(out.get(), scratch.get() + skip, size);

    return {std::move(out), size, 0, static_cast<size_t>(aligned_size)};
}

} // namespace mdb::gnn
