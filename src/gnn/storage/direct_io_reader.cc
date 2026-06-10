#include "gnn/storage/direct_io_reader.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

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
    // Spec A3: initialize NUM_RINGS rings (DiskGNN config: 4 rings × 1024 SQEs).
    // Each ring is independent — one may fail (e.g., RLIMIT_MEMLOCK on old
    // kernels) while others succeed. We use whichever rings init'd; if zero,
    // fall back to pread silently. MDB_GNN_NO_IO_URING=1 skips ring init
    // entirely, forcing the synchronous pread path.
    const char* no_uring = std::getenv("MDB_GNN_NO_IO_URING");
    if (!(no_uring && std::strcmp(no_uring, "0") != 0)) {
        for (auto& slot : rings_) {
            if (::io_uring_queue_init(QUEUE_DEPTH, &slot.ring, 0) == 0) {
                slot.initialized  = true;
                io_uring_active_  = true;
            }
        }
    }
#endif
}

// =============================================================================
// Destructor
// =============================================================================

DirectIoReader::~DirectIoReader() {
#ifdef ENABLE_IO_URING
    for (auto& slot : rings_) {
        if (slot.initialized) {
            ::io_uring_queue_exit(&slot.ring);
        }
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
// pread_all_tail_aware — synchronous read of a block-aligned region
// =============================================================================

void DirectIoReader::pread_all_tail_aware(void* buf, size_t count, off_t offset) {
    // O_DIRECT requires block-aligned read sizes, so an aligned region at the
    // file tail extends past EOF whenever the file size is not block-aligned
    // (feature rows rarely are). The kernel returns the partial tail on the
    // final read and 0 thereafter; only bytes that exist in the file within
    // [offset, offset + count) are required.
    const uint64_t off_u   = static_cast<uint64_t>(offset);
    const size_t   logical = off_u >= file_size_
        ? 0
        : static_cast<size_t>(std::min<uint64_t>(count, file_size_ - off_u));

    char* p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < count) {
        ssize_t n = ::pread(fd_, p + got, count - got,
                            offset + static_cast<off_t>(got));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "DirectIoReader: pread failed at offset " +
                std::to_string(offset + static_cast<off_t>(got)) +
                ": " + std::strerror(errno));
        }
        if (n == 0) break;  // EOF — valid at the aligned tail
        got += static_cast<size_t>(n);
    }
    if (got < logical) {
        throw std::runtime_error(
            "DirectIoReader: unexpected EOF at offset " +
            std::to_string(offset + static_cast<off_t>(got)) + " (wanted " +
            std::to_string(logical - got) + " more bytes)");
    }
    if (got < count) {
        // Keep the unread tail deterministic — scatter copies of a row that
        // was clamped at EOF read from this region.
        std::memset(p + got, 0, count - got);
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
    pread_all_tail_aware(tmp.get(), aligned_size, static_cast<off_t>(aligned_off));
    std::memcpy(dest, tmp.get() + skip, wanted_bytes);
    return static_cast<size_t>(aligned_size);
}

// =============================================================================
// io_uring batch submission
// =============================================================================

#ifdef ENABLE_IO_URING

void DirectIoReader::submit_to_ring(
    struct io_uring& ring,
    const std::vector<AlignedSpan>& spans,
    char* scratch_buf,
    size_t begin,
    size_t end)
{
    size_t submitted = begin;

    while (submitted < end) {
        // Submit up to QUEUE_DEPTH SQEs at a time. One span per SQE.
        size_t batch_end  = std::min(submitted + QUEUE_DEPTH, end);
        size_t batch_size = batch_end - submitted;

        for (size_t i = submitted; i < batch_end; ++i) {
            const auto& s = spans[i];
            struct io_uring_sqe* sqe = ::io_uring_get_sqe(&ring);
            if (!sqe) {
                // Queue full — should not happen within QUEUE_DEPTH, but handle it
                ::io_uring_submit(&ring);
                for (size_t j = submitted; j < i; ++j) {
                    struct io_uring_cqe* cqe;
                    if (::io_uring_wait_cqe(&ring, &cqe) < 0) {
                        throw std::runtime_error(
                            "DirectIoReader: io_uring_wait_cqe failed");
                    }
                    if (cqe->res < 0) {
                        ::io_uring_cqe_seen(&ring, cqe);
                        throw std::runtime_error(
                            "DirectIoReader: io_uring read failed: " +
                            std::string(std::strerror(-cqe->res)));
                    }
                    ::io_uring_cqe_seen(&ring, cqe);
                }
                sqe = ::io_uring_get_sqe(&ring);
                if (!sqe) {
                    throw std::runtime_error(
                        "DirectIoReader: io_uring_get_sqe failed after drain");
                }
            }
            ::io_uring_prep_read(
                sqe, fd_,
                scratch_buf + s.buf_offset,
                static_cast<unsigned>(s.aligned_size),
                static_cast<off_t>(s.aligned_off));
            sqe->user_data = static_cast<uint64_t>(i);
        }

        int ret = ::io_uring_submit(&ring);
        if (ret < 0) {
            throw std::runtime_error(
                "DirectIoReader: io_uring_submit failed: " +
                std::string(std::strerror(-ret)));
        }

        for (size_t i = 0; i < batch_size; ++i) {
            struct io_uring_cqe* cqe;
            if (::io_uring_wait_cqe(&ring, &cqe) < 0) {
                throw std::runtime_error(
                    "DirectIoReader: io_uring_wait_cqe failed");
            }
            if (cqe->res < 0) {
                int err = -cqe->res;
                ::io_uring_cqe_seen(&ring, cqe);
                throw std::runtime_error(
                    "DirectIoReader: io_uring read error: " +
                    std::string(std::strerror(err)));
            }
            ::io_uring_cqe_seen(&ring, cqe);
        }

        submitted = batch_end;
    }
}

void DirectIoReader::submit_io_uring_spans(
    const std::vector<AlignedSpan>& spans,
    char* scratch_buf)
{
    if (spans.empty()) return;

    // Spec A3: collect active rings, partition spans into contiguous chunks
    // (preserving sequential file_offset ordering within each ring), and
    // dispatch one std::async thread per ring. Each ring is driven by
    // exactly one thread — io_uring rings are not internally thread-safe.
    std::vector<struct io_uring*> active_rings;
    active_rings.reserve(NUM_RINGS);
    for (auto& slot : rings_) {
        if (slot.initialized) active_rings.push_back(&slot.ring);
    }
    if (active_rings.empty()) {
        // Should not happen: io_uring_active_ would be false.
        throw std::runtime_error(
            "DirectIoReader: submit_io_uring_spans called without active rings");
    }

    const size_t N           = active_rings.size();
    const size_t total_spans = spans.size();

    // Single-ring fast path — avoids the std::async overhead for small reads.
    if (N == 1 || total_spans <= QUEUE_DEPTH) {
        submit_to_ring(*active_rings[0], spans, scratch_buf, 0, total_spans);
        return;
    }

    // Partition spans into N contiguous chunks. Contiguous (not round-robin)
    // preserves sequential file_offset ordering within each ring — the
    // kernel sees N separate sequential streams, each prefetch-friendly.
    std::vector<std::future<void>> futures;
    futures.reserve(N);
    for (size_t r = 0; r < N; ++r) {
        size_t begin =  r        * total_spans / N;
        size_t end   = (r + 1)   * total_spans / N;
        if (begin == end) continue;
        struct io_uring* ring_ptr = active_rings[r];
        futures.push_back(std::async(std::launch::async,
            [this, ring_ptr, &spans, scratch_buf, begin, end]() {
                submit_to_ring(*ring_ptr, spans, scratch_buf, begin, end);
            }));
    }

    // Wait for all rings, propagate the first exception (others continue
    // safely — each writes to a disjoint slice of scratch, no shared state).
    std::exception_ptr first_exception = nullptr;
    for (auto& f : futures) {
        try {
            f.get();
        } catch (...) {
            if (!first_exception) first_exception = std::current_exception();
        }
    }
    if (first_exception) std::rethrow_exception(first_exception);
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

    // ----- Non-direct: per-row pread (no alignment overhead, no dedup) -----
    // tmpfs / NFS / older kernels fall here. bytes_disk == size in this mode
    // because the OS hides page-cache reads behind pread.
    if (!direct_) {
        size_t bytes_disk_total = 0;
        for (const auto& op : ops) {
            bytes_disk_total += read_aligned_region(
                out.get() + op.out_offset,
                op.file_offset,
                row_bytes);
        }
        return {std::move(out), total_bytes, row_indices.size(), bytes_disk_total};
    }

    // ----- O_DIRECT: page-level dedup (Spec A2, 2026-04-27) -----
    // Walk sorted ops, merging consecutive aligned regions when they overlap
    // or are adjacent. Each merged AlignedSpan becomes one io_uring/pread
    // read; CopyOps record where to scatter wanted bytes back to output.
    std::vector<AlignedSpan> spans;
    std::vector<CopyOp>      copies;
    spans.reserve(ops.size());
    copies.reserve(ops.size());

    const uint64_t aligned_fs = align_up(file_size_, block_align_);

    for (const auto& op : ops) {
        uint64_t aligned_off = align_down(op.file_offset, block_align_);
        uint64_t end         = op.file_offset + row_bytes;
        if (end > file_size_) {
            // This row extends past EOF — its out-of-range bytes stay zero (the
            // `out` buffer was zero-filled above). The L4 caller bounds-checks
            // indices, so this normally never fires; a stale/corrupt row index
            // would land here and yield a silently zero-filled feature row.
            // Warn ONCE rather than throw (changing to a hard error is
            // behaviour-risky and the caller already validates indices).
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                std::cerr << "[DirectIoReader] WARNING: row at offset "
                          << op.file_offset << " (+" << row_bytes
                          << " B) exceeds file size " << file_size_
                          << " — out-of-range bytes zero-filled "
                             "(further such warnings suppressed)\n";
            }
            end = file_size_;
        }
        uint64_t aligned_end = align_up(end, block_align_);
        if (aligned_end > aligned_fs) aligned_end = aligned_fs;

        if (!spans.empty()) {
            auto& last = spans.back();
            uint64_t last_end = last.aligned_off + last.aligned_size;
            if (aligned_off <= last_end) {
                // Merge: extend the existing span to cover this row's region.
                // The `<=` (not `<`) collapses adjacent spans where one ends
                // exactly where the next begins — common after sort + MinHash
                // reorder, when hot rows cluster within / across page borders.
                uint64_t new_end = std::max(last_end, aligned_end);
                last.aligned_size = new_end - last.aligned_off;
                size_t src = static_cast<size_t>(op.file_offset - last.aligned_off);
                copies.push_back({spans.size() - 1, src, op.out_offset,
                                  static_cast<size_t>(row_bytes)});
                continue;
            }
        }
        // Disjoint — open a new span.
        spans.push_back({aligned_off, aligned_end - aligned_off, 0});
        size_t src = static_cast<size_t>(op.file_offset - aligned_off);
        copies.push_back({spans.size() - 1, src, op.out_offset,
                          static_cast<size_t>(row_bytes)});
    }

    // Assign cumulative buf_offset to each span (overflow-checked).
    size_t scratch_total = 0;
    for (auto& s : spans) {
        if (scratch_total > SIZE_MAX - s.aligned_size) {
            throw std::overflow_error(
                "DirectIoReader: scratch buffer size overflow");
        }
        s.buf_offset = scratch_total;
        scratch_total += s.aligned_size;
    }

    auto scratch = alloc_aligned(scratch_total, block_align_);

#ifdef ENABLE_IO_URING
    if (io_uring_active_) {
        submit_io_uring_spans(spans, scratch.get());
    } else
#endif
    {
        // Synchronous direct: pread per merged span. Spans are block-aligned,
        // so the span at the file tail may extend past EOF — tail-aware.
        for (const auto& s : spans) {
            pread_all_tail_aware(scratch.get() + s.buf_offset,
                                 static_cast<size_t>(s.aligned_size),
                                 static_cast<off_t>(s.aligned_off));
        }
    }

    // Scatter wanted bytes from scratch to output positions.
    for (const auto& c : copies) {
        const auto& s = spans[c.span_idx];
        std::memcpy(out.get() + c.dest_offset,
                    scratch.get() + s.buf_offset + c.src_in_span,
                    c.wanted);
    }

    // bytes_disk == scratch_total == sum of merged aligned sizes. With
    // perfect dedup (all rows share one page) this approaches one block;
    // without dedup (rows in distinct pages) it equals the per-row sum.
    // The l3_read_amplification metric (bytes_disk / bytes_wanted)
    // typically moves from ~8× pre-A2 to ~2-4× after MinHash clustering.
    return {std::move(out), total_bytes, row_indices.size(), scratch_total};
}

// =============================================================================
// read_range — contiguous byte range
// =============================================================================

DirectIoReader::ReadResult DirectIoReader::read_range(uint64_t offset, uint64_t size) {
    if (size == 0) {
        return {AlignedBuffer(nullptr), 0, 0, 0};
    }
    // Overflow-safe past-EOF check. In the O_DIRECT path below, the scratch
    // region is clamped to align_up(file_size_), so a past-EOF range would
    // make the final memcpy read past the scratch allocation.
    if (offset > file_size_ || size > file_size_ - offset) {
        throw std::out_of_range(
            "DirectIoReader: read_range past EOF (offset=" +
            std::to_string(offset) + ", size=" + std::to_string(size) +
            ", file_size=" + std::to_string(file_size_) + ")");
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
        // read_range is a single contiguous read — no parallelism benefit.
        // Use the first available ring; the others stay idle for this call.
        struct io_uring* ring = nullptr;
        for (auto& slot : rings_) {
            if (slot.initialized) { ring = &slot.ring; break; }
        }
        if (!ring) {
            throw std::runtime_error(
                "DirectIoReader: read_range called without active rings");
        }

        auto scratch = alloc_aligned(aligned_size, block_align_);

        struct io_uring_sqe* sqe = ::io_uring_get_sqe(ring);
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

        int ret = ::io_uring_submit(ring);
        if (ret < 0) {
            throw std::runtime_error(
                "DirectIoReader: io_uring_submit failed: " +
                std::string(std::strerror(-ret)));
        }

        struct io_uring_cqe* cqe;
        if (::io_uring_wait_cqe(ring, &cqe) < 0) {
            throw std::runtime_error(
                "DirectIoReader: io_uring_wait_cqe failed");
        }
        if (cqe->res < 0) {
            int err = -cqe->res;
            ::io_uring_cqe_seen(ring, cqe);
            throw std::runtime_error(
                "DirectIoReader: io_uring range read error: " +
                std::string(std::strerror(err)));
        }
        ::io_uring_cqe_seen(ring, cqe);

        std::memcpy(out.get(), scratch.get() + skip, size);
        return {std::move(out), size, 0, static_cast<size_t>(aligned_size)};
    }
#endif

    // Synchronous O_DIRECT fallback — aligned region may extend past EOF.
    auto scratch = alloc_aligned(aligned_size, block_align_);
    pread_all_tail_aware(scratch.get(), aligned_size, static_cast<off_t>(aligned_off));
    std::memcpy(out.get(), scratch.get() + skip, size);

    return {std::move(out), size, 0, static_cast<size_t>(aligned_size)};
}

} // namespace mdb::gnn
