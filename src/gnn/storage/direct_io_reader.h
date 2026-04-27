#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

#ifdef ENABLE_IO_URING
#include <liburing.h>
#endif

namespace mdb::gnn {

/// Custom deleter for aligned memory (allocated with posix_memalign).
struct AlignedDeleter {
    void operator()(char* ptr) const { std::free(ptr); }
};
using AlignedBuffer = std::unique_ptr<char[], AlignedDeleter>;

/**
 * @brief Reads feature rows from disk bypassing OS page cache.
 *
 * Uses O_DIRECT to prevent page cache pollution (critical for RAM-bounded
 * training where L3 mmap pages would compete with L2 for host memory).
 *
 * When ENABLE_IO_URING is defined and liburing is available at runtime,
 * uses io_uring for async batch submission (single syscall for N reads).
 * Falls back to synchronous pread() + POSIX_FADV_DONTNEED otherwise.
 *
 * O_DIRECT requires buffer addresses, read offsets, and read sizes to be
 * aligned to the filesystem block size (typically 512 or 4096 bytes).
 * Feature rows are typically NOT aligned (e.g., 1433 dims x 4 bytes = 5732).
 * This reader handles alignment transparently: for each row, it reads
 * the containing aligned region and extracts the needed bytes.
 *
 * Thread-safety: NOT thread-safe (single io_uring ring). Use one per thread.
 */
class DirectIoReader {
public:
    explicit DirectIoReader(const std::filesystem::path& file_path);
    ~DirectIoReader();

    // Not copyable, not movable (owns fd and io_uring ring)
    DirectIoReader(const DirectIoReader&) = delete;
    DirectIoReader& operator=(const DirectIoReader&) = delete;
    DirectIoReader(DirectIoReader&&) = delete;
    DirectIoReader& operator=(DirectIoReader&&) = delete;

    /// Result of a read operation.
    struct ReadResult {
        AlignedBuffer data;        ///< aligned buffer containing the read data
        size_t        size;        ///< useful bytes in data (rows * row_bytes, or `size` for read_range)
        size_t        num_rows;    ///< number of rows read (0 for read_range)
        /// Physical bytes read from disk. With O_DIRECT this counts the
        /// aligned-up read regions, so bytes_disk >= size; the difference
        /// is read amplification due to block alignment (typically 4 KB).
        /// With non-O_DIRECT (mmap fallback / tmpfs / NFS), bytes_disk
        /// equals size — page-cache hits are not measured.
        size_t        bytes_disk;
    };

    /// Read specific rows from a feature file.
    /// @param row_indices  Which rows to read (0-based).
    /// @param row_bytes    Bytes per row.
    /// @param data_offset  Byte offset from file start to first row (past header).
    /// @return Aligned buffer containing rows in the ORDER of row_indices.
    ReadResult read_rows(
        const std::vector<uint64_t>& row_indices,
        uint64_t row_bytes,
        uint64_t data_offset
    );

    /// Read a contiguous byte range.
    ReadResult read_range(uint64_t offset, uint64_t size);

    /// Whether O_DIRECT is active (may be disabled on tmpfs, etc.).
    bool is_direct() const { return direct_; }

    /// Whether io_uring is active.
    bool is_io_uring() const { return io_uring_active_; }

    /// File size in bytes.
    size_t file_size() const { return file_size_; }

private:
    int    fd_               = -1;
    size_t file_size_        = 0;
    bool   direct_           = false;
    bool   io_uring_active_  = false;

    /// Block alignment used for O_DIRECT reads. Detected at open time.
    size_t block_align_ = 4096;

#ifdef ENABLE_IO_URING
    struct io_uring ring_;
    bool ring_initialized_ = false;
    static constexpr unsigned QUEUE_DEPTH = 64;
#endif

    /// Allocate aligned memory (required for O_DIRECT).
    /// Returns nullptr-unique_ptr on size == 0.
    static AlignedBuffer alloc_aligned(size_t size, size_t alignment = 4096);

    /// Read using pread, retrying on EINTR and partial reads.
    void pread_all(void* buf, size_t count, off_t offset);

    /// Issue POSIX_FADV_DONTNEED for a range (best-effort, non-O_DIRECT mode).
    void advise_dontneed(off_t offset, size_t len);

    /// Aligned read helpers for O_DIRECT.
    /// Round value down to nearest multiple of alignment.
    static uint64_t align_down(uint64_t val, uint64_t align);
    /// Round value up to nearest multiple of alignment.
    static uint64_t align_up(uint64_t val, uint64_t align);

    /// Internal: read a single aligned region and copy the relevant sub-range
    /// into the destination buffer. Used when O_DIRECT is active.
    /// Returns the number of bytes actually read at OS level (== aligned_size
    /// in O_DIRECT mode, == wanted_bytes otherwise). Caller sums these into
    /// ReadResult::bytes_disk for paper-comparable disk-traffic accounting.
    size_t read_aligned_region(
        char* dest,
        uint64_t file_offset,
        uint64_t wanted_bytes
    );

    /// Spec A2 (2026-04-27): aligned span used by the read_rows dedup path.
    /// Multiple wanted rows can share a single AlignedSpan when their
    /// aligned regions overlap or are adjacent — typical for papers100M
    /// where 8 rows of 512 B share a single 4 KB page after MinHash reorder.
    struct AlignedSpan {
        uint64_t aligned_off;   ///< block-aligned start offset in file
        uint64_t aligned_size;  ///< block-aligned size (always multiple of block_align_)
        size_t   buf_offset;    ///< offset within scratch buffer
    };

    /// Spec A2: per-row scatter copy from a deduped scratch span to output.
    struct CopyOp {
        size_t   span_idx;      ///< index into AlignedSpan vector
        size_t   src_in_span;   ///< byte offset within the merged span
        size_t   dest_offset;   ///< byte offset in output buffer
        size_t   wanted;        ///< bytes to copy (equals row_bytes)
    };

#ifdef ENABLE_IO_URING
    /// Batch-submit reads via io_uring (one read per AlignedSpan) and
    /// collect completions. Caller is responsible for the subsequent
    /// scatter copies via the matching CopyOp list.
    void submit_io_uring_spans(
        const std::vector<AlignedSpan>& spans,
        char* scratch_buf
    );
#endif
};

} // namespace mdb::gnn
