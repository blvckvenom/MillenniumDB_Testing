#include "gnn/storage/packed_batch_store.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <unistd.h>

#include "gnn/common/posix_io.h"

namespace mdb::gnn {

namespace fs = std::filesystem;

namespace {

std::string make_batch_filename(uint64_t batch_id) {
    // "batch_" (6) + up to 20 digits + ".bin" (4) + NUL = max 31 chars
    char buf[64];
    std::snprintf(buf, sizeof(buf), "batch_%06" PRIu64 ".bin", batch_id);
    return std::string(buf);
}

} // anonymous namespace

// ===========================================================================
// PackedBatchWriter
// ===========================================================================

PackedBatchWriter::PackedBatchWriter(const fs::path& dir,
                                     uint64_t feature_dim,
                                     GnnDtype dtype)
    : dir_(dir), feature_dim_(feature_dim), dtype_(dtype)
{
    if (feature_dim_ == 0) {
        throw std::invalid_argument("PackedBatchWriter: feature_dim must be > 0");
    }
    fs::create_directories(dir_);
}

std::filesystem::path PackedBatchWriter::batch_path(uint64_t batch_id) const {
    return dir_ / make_batch_filename(batch_id);
}

void PackedBatchWriter::write_batch(uint64_t batch_id, const void* data,
                                    uint64_t num_nodes) {
    if (batch_id != batches_written_) {
        throw std::invalid_argument(
            "PackedBatchWriter::write_batch: expected batch_id " +
            std::to_string(batches_written_) + ", got " + std::to_string(batch_id));
    }
    if (num_nodes > 0 && data == nullptr) {
        throw std::invalid_argument(
            "PackedBatchWriter::write_batch: data is null but num_nodes > 0");
    }

    auto header = PackedBatchHeader::make(num_nodes, feature_dim_, dtype_);
    auto path = batch_path(batch_id);
    std::string path_str = path.string();

    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "PackedBatchWriter: cannot open " + path_str + ": " + safe_strerror(errno));
    }

    try {
        FdGuard guard(fd);

        write_all(fd, &header, sizeof(header), path_str);
        if (num_nodes > 0) {
            write_all(fd, data, header.data_bytes(), path_str);
        }

        if (::fsync(fd) < 0) {
            throw std::runtime_error(
                "PackedBatchWriter: fsync failed for " + path_str + ": " + safe_strerror(errno));
        }
    } catch (...) {
        // Clean up partial file on failure (Fix #10)
        fs::remove(path);
        throw;
    }

    // Best-effort parent directory fsync for crash consistency (Fix #3)
    fsync_directory(path);

    ++batches_written_;
}

// ===========================================================================
// PackedBatchReader
// ===========================================================================

PackedBatchReader::PackedBatchReader(const fs::path& dir,
                                     uint64_t num_batches,
                                     uint64_t feature_dim,
                                     GnnDtype dtype)
    : dir_(dir), num_batches_(num_batches), feature_dim_(feature_dim), dtype_(dtype)
{
    if (!fs::exists(dir_)) {
        throw std::runtime_error(
            "PackedBatchReader: directory not found: " + dir_.string());
    }
    // Fix #5: symmetric with Writer validation
    if (feature_dim_ == 0) {
        throw std::invalid_argument(
            "PackedBatchReader: feature_dim must be > 0");
    }
}

std::filesystem::path PackedBatchReader::batch_path(uint64_t batch_id) const {
    return dir_ / make_batch_filename(batch_id);
}

PackedBatchHeader PackedBatchReader::read_header(uint64_t batch_id) const {
    if (batch_id >= num_batches_) {
        throw std::out_of_range(
            "PackedBatchReader: batch_id " + std::to_string(batch_id) +
            " >= num_batches " + std::to_string(num_batches_));
    }

    auto path = batch_path(batch_id);
    std::string path_str = path.string();

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "PackedBatchReader: cannot open " + path_str + ": " + safe_strerror(errno));
    }
    FdGuard guard(fd);

    PackedBatchHeader header;
    read_all(fd, &header, sizeof(header), path_str);

    if (!header.is_valid()) {
        throw std::runtime_error(
            "PackedBatchReader: invalid header in " + path_str);
    }
    if (header.feature_dim != feature_dim_) {
        throw std::runtime_error(
            "PackedBatchReader: feature_dim mismatch in " + path_str +
            ": expected " + std::to_string(feature_dim_) +
            ", got " + std::to_string(header.feature_dim));
    }
    if (header.get_dtype() != dtype_) {
        throw std::runtime_error(
            "PackedBatchReader: dtype mismatch in " + path_str);
    }

    return header;
}

uint64_t PackedBatchReader::read_batch(uint64_t batch_id, void* out,
                                       size_t out_capacity) const {
    // Fix #1: Single-open — read header + data from the same fd.
    if (batch_id >= num_batches_) {
        throw std::out_of_range(
            "PackedBatchReader: batch_id " + std::to_string(batch_id) +
            " >= num_batches " + std::to_string(num_batches_));
    }

    auto path = batch_path(batch_id);
    std::string path_str = path.string();

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "PackedBatchReader: cannot open " + path_str + ": " + safe_strerror(errno));
    }
    FdGuard guard(fd);

    // Read and validate header
    PackedBatchHeader header;
    read_all(fd, &header, sizeof(header), path_str);

    if (!header.is_valid()) {
        throw std::runtime_error(
            "PackedBatchReader: invalid header in " + path_str);
    }
    if (header.feature_dim != feature_dim_) {
        throw std::runtime_error(
            "PackedBatchReader: feature_dim mismatch in " + path_str +
            ": expected " + std::to_string(feature_dim_) +
            ", got " + std::to_string(header.feature_dim));
    }
    if (header.get_dtype() != dtype_) {
        throw std::runtime_error(
            "PackedBatchReader: dtype mismatch in " + path_str);
    }

    // For v2 files, seek past the ObjectId table to reach feature data
    if (header.has_oid_table()) {
        off_t data_start = static_cast<off_t>(header.data_offset());
        if (::lseek(fd, data_start, SEEK_SET) < 0) {
            throw std::runtime_error(
                "PackedBatchReader::read_batch: lseek past OID table failed: " +
                std::string(std::strerror(errno)));
        }
    }

    size_t data_size = header.data_bytes(); // overflow-checked

    if (data_size == 0) {
        return 0;
    }

    if (out_capacity < data_size) {
        throw std::runtime_error(
            "PackedBatchReader: buffer too small for " + path_str +
            " — need " + std::to_string(data_size) +
            " bytes, got " + std::to_string(out_capacity));
    }

    // Read data directly — fd is positioned after header (v1) or after OID table (v2)
    read_all(fd, out, data_size, path_str);

    return header.num_nodes;
}

// ===========================================================================
// generate_packed_batches
// ===========================================================================

void generate_packed_batches(
    const FeatureMatrix& features,
    uint64_t num_batches,
    std::function<std::vector<uint64_t>(uint64_t batch_id)> batch_provider,
    const fs::path& output_dir)
{
    PackedBatchWriter writer(output_dir, features.num_cols(), features.dtype());

    size_t row_bytes = features.row_bytes();
    std::vector<char> buffer;

    // Progress reporting cadence: print every 5% of batches or 100 batches.
    const uint64_t progress_step = std::max<uint64_t>(num_batches / 20, 100);
    auto t0 = std::chrono::steady_clock::now();

    std::cout << "[Materialize] packing " << num_batches
              << " batches → " << output_dir.string() << "\n" << std::flush;

    for (uint64_t b = 0; b < num_batches; ++b) {
        auto row_ids = batch_provider(b);
        uint64_t N = row_ids.size();

        if (N == 0) {
            writer.write_batch(b, nullptr, 0);
            continue;
        }

        // Fix #2: Overflow guard before buffer allocation
        if (row_bytes > 0 && N > SIZE_MAX / row_bytes) {
            throw std::overflow_error(
                "generate_packed_batches: batch " + std::to_string(b) +
                " buffer size overflow (" + std::to_string(N) +
                " nodes * " + std::to_string(row_bytes) + " bytes/row)");
        }

        size_t needed = N * row_bytes;
        if (buffer.size() < needed) {
            buffer.resize(needed);
        }

        features.extract_rows(row_ids, buffer.data());
        writer.write_batch(b, buffer.data(), N);

        if ((b + 1) % progress_step == 0 || b + 1 == num_batches) {
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            double rate = (b + 1) / elapsed;
            double eta = (num_batches - b - 1) / rate;
            std::cout << "[Materialize] packed " << (b + 1) << "/"
                      << num_batches
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * (b + 1) / num_batches) << "%)"
                      << "  rate=" << std::fixed << std::setprecision(1) << rate << "/s"
                      << "  elapsed=" << std::fixed << std::setprecision(0) << elapsed << "s"
                      << "  ETA=" << std::fixed << std::setprecision(0) << eta << "s"
                      << std::endl;
        }
    }
}

void generate_packed_batches(
    const FeatureMatrix& features,
    const std::vector<std::vector<uint64_t>>& batch_assignments,
    const fs::path& output_dir)
{
    generate_packed_batches(
        features,
        static_cast<uint64_t>(batch_assignments.size()),
        [&](uint64_t batch_id) { return batch_assignments.at(batch_id); },
        output_dir
    );
}

// ===========================================================================
// generate_packed_batches_partitioned (Spec B1 — DiskGNN-style inverted loop)
// ===========================================================================

void generate_packed_batches_partitioned(
    const FeatureMatrix& features,
    uint64_t num_batches,
    std::function<std::vector<uint64_t>(uint64_t batch_id)> batch_provider,
    const fs::path& output_dir,
    size_t partition_bytes)
{
    fs::create_directories(output_dir);

    const uint64_t row_bytes = features.row_bytes();
    if (row_bytes == 0) {
        throw std::runtime_error(
            "generate_packed_batches_partitioned: row_bytes == 0");
    }

    const uint64_t total_rows = features.num_rows();
    if (total_rows == 0) {
        throw std::runtime_error(
            "generate_packed_batches_partitioned: empty FeatureMatrix");
    }

    // Partition geometry: target partition_bytes, round down to whole rows.
    uint64_t partition_rows = std::max<uint64_t>(1, partition_bytes / row_bytes);
    if (partition_rows > total_rows) partition_rows = total_rows;
    const uint64_t num_partitions =
        (total_rows + partition_rows - 1) / partition_rows;

    std::cout << "[Materialize] partitioned packer (B1): "
              << num_partitions << " partitions × " << partition_rows
              << " rows ("
              << (partition_rows * row_bytes / (1024ULL * 1024)) << " MB each), "
              << num_batches << " batches\n" << std::flush;

    // RowRef: 16 bytes — one inverted-index entry per (row, batch, position).
    // batch_id < 2^32 covers any realistic dataset; pos_in_batch < 2^32 too
    // (a 4 B-node batch would need terabytes of features per single batch,
    // way past any practical workload).
    struct RowRef {
        uint32_t batch_id;
        uint32_t pos_in_batch;
        uint64_t fmat_row;
    };
    static_assert(sizeof(RowRef) == 16, "RowRef must be 16 bytes");

    // ----- Phase 0+1: pre-write headers, ftruncate, build inverted index -----
    auto t_phase01_start = std::chrono::steady_clock::now();

    std::vector<std::vector<RowRef>> inverted(num_partitions);

    {
        const uint64_t feature_dim = features.num_cols();
        const auto     dtype       = features.dtype();

        for (uint64_t b = 0; b < num_batches; ++b) {
            auto row_ids = batch_provider(b);
            const uint64_t N = row_ids.size();

            // Phase 0a: write header
            auto path = output_dir / make_batch_filename(b);
            std::string path_str = path.string();

            int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                throw std::runtime_error(
                    "generate_packed_batches_partitioned: cannot create " +
                    path_str + ": " + safe_strerror(errno));
            }
            FdGuard guard(fd);

            auto header = PackedBatchHeader::make(N, feature_dim, dtype);
            write_all(fd, &header, sizeof(header), path_str);

            // Phase 0b: ftruncate to total size, leaving the features region
            // as a sparse hole. ext4/xfs do not allocate physical blocks for
            // the hole; Phase 2's pwrites populate them. For N==0 we leave
            // the file at 32 bytes (header only) — same on-disk size as the
            // classic path.
            if (N > 0) {
                const off_t total_size =
                    static_cast<off_t>(sizeof(header) + N * row_bytes);
                if (::ftruncate(fd, total_size) < 0) {
                    throw std::runtime_error(
                        "generate_packed_batches_partitioned: ftruncate failed for " +
                        path_str + ": " + safe_strerror(errno));
                }
            }

            // Phase 1: bucket each row_id into its partition's inverted list.
            for (size_t i = 0; i < N; ++i) {
                const uint64_t fmat_row = row_ids[i];
                if (fmat_row >= total_rows) {
                    throw std::out_of_range(
                        "generate_packed_batches_partitioned: row " +
                        std::to_string(fmat_row) +
                        " out of range [0, " + std::to_string(total_rows) + ")");
                }
                const uint64_t partition_id = fmat_row / partition_rows;
                inverted[partition_id].push_back({
                    static_cast<uint32_t>(b),
                    static_cast<uint32_t>(i),
                    fmat_row
                });
            }
            // FdGuard closes the fd; row_ids vector is freed at end of scope.
        }
    }

    auto t_phase01_end = std::chrono::steady_clock::now();

    uint64_t total_refs = 0;
    for (const auto& bucket : inverted) total_refs += bucket.size();
    std::cout << "[Materialize] phase 0+1 done in "
              << std::chrono::duration_cast<std::chrono::seconds>(
                     t_phase01_end - t_phase01_start).count()
              << "s — inverted index: " << total_refs << " refs across "
              << num_partitions << " partitions ("
              << (total_refs * sizeof(RowRef) / (1024 * 1024)) << " MB)"
              << std::endl;

    // ----- Phase 2: sequential scan + scatter pwrites -----
    auto t_phase2_start = std::chrono::steady_clock::now();

    // Phase 2 reads each partition in row order via FeatureMatrix::row(),
    // which returns a pointer into the mmap'd .fmat past the header. The
    // pointer is stable for the lifetime of the FeatureMatrix; we use it as
    // a byte-arithmetic base for memcpy. Kernel default readahead handles
    // sequential I/O — a future optimization can expose a public
    // madvise_sequential() helper on FeatureMatrix to widen the window.
    std::vector<char> partition_buf;
    partition_buf.reserve(partition_rows * row_bytes);

    const uint64_t progress_step = std::max<uint64_t>(num_partitions / 20, 4);

    for (uint64_t p = 0; p < num_partitions; ++p) {
        auto& refs = inverted[p];
        if (refs.empty()) continue;

        const uint64_t row_start = p * partition_rows;
        const uint64_t row_end   = std::min(row_start + partition_rows, total_rows);
        const uint64_t this_rows = row_end - row_start;
        const size_t   this_bytes = static_cast<size_t>(this_rows * row_bytes);

        // Sequential read: memcpy the partition from mmap into local buf.
        // features.row(row_start) yields a pointer into the mmap'd .fmat at
        // the start of this partition (byte offset row_start * row_bytes
        // past the data section). Kernel default readahead handles I/O.
        partition_buf.resize(this_bytes);
        std::memcpy(partition_buf.data(),
                    features.row(row_start),
                    this_bytes);

        // Sort refs by batch_id so each output file is opened/closed exactly
        // once per partition. Writes within a batch group are random offsets
        // (bounded by N[b] * row_bytes), but the BATCH file handle is reused.
        std::sort(refs.begin(), refs.end(),
                  [](const RowRef& a, const RowRef& b) {
                      return a.batch_id < b.batch_id;
                  });

        size_t i = 0;
        while (i < refs.size()) {
            const uint32_t bid = refs[i].batch_id;
            size_t j = i + 1;
            while (j < refs.size() && refs[j].batch_id == bid) ++j;

            auto path = output_dir / make_batch_filename(bid);
            std::string path_str = path.string();

            int fd = ::open(path.c_str(), O_WRONLY);
            if (fd < 0) {
                throw std::runtime_error(
                    "generate_packed_batches_partitioned: cannot open " +
                    path_str + " for scatter: " + safe_strerror(errno));
            }
            FdGuard guard(fd);

            for (size_t k = i; k < j; ++k) {
                const auto& ref = refs[k];
                const size_t buf_offset =
                    static_cast<size_t>((ref.fmat_row - row_start) * row_bytes);
                off_t off_curr = static_cast<off_t>(
                    sizeof(PackedBatchHeader) + ref.pos_in_batch * row_bytes);

                // pwrite loop: handle EINTR + partial writes.
                size_t remaining  = static_cast<size_t>(row_bytes);
                const char* src   = partition_buf.data() + buf_offset;
                while (remaining > 0) {
                    ssize_t n = ::pwrite(fd, src, remaining, off_curr);
                    if (n < 0) {
                        if (errno == EINTR) continue;
                        throw std::runtime_error(
                            "generate_packed_batches_partitioned: pwrite failed at " +
                            path_str + ": " + safe_strerror(errno));
                    }
                    if (n == 0) {
                        throw std::runtime_error(
                            "generate_packed_batches_partitioned: pwrite returned 0 at " +
                            path_str);
                    }
                    src       += n;
                    off_curr  += n;
                    remaining -= static_cast<size_t>(n);
                }
            }

            i = j;
        }

        // Free the partition's inverted bucket to release memory before the
        // next partition. swap-with-empty forces deallocation.
        std::vector<RowRef>().swap(refs);

        if ((p + 1) % progress_step == 0 || p + 1 == num_partitions) {
            auto t_now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(
                t_now - t_phase2_start).count();
            double rate = (p + 1) / elapsed;
            double eta  = (num_partitions - p - 1) / rate;
            std::cout << "[Materialize] phase 2 partition " << (p + 1) << "/"
                      << num_partitions
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * (p + 1) / num_partitions) << "%)"
                      << "  elapsed=" << std::fixed << std::setprecision(0)
                      << elapsed << "s"
                      << "  ETA=" << std::fixed << std::setprecision(0)
                      << eta << "s"
                      << std::endl;
        }
    }

    auto t_phase2_end = std::chrono::steady_clock::now();

    // ----- Phase 3: fsync all batch files for crash-consistent durability -----
    auto t_phase3_start = std::chrono::steady_clock::now();
    for (uint64_t b = 0; b < num_batches; ++b) {
        auto path = output_dir / make_batch_filename(b);
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) continue;  // non-fatal: file may be 0-row empty
        FdGuard guard(fd);
        if (::fsync(fd) < 0) {
            throw std::runtime_error(
                "generate_packed_batches_partitioned: fsync failed for " +
                path.string() + ": " + safe_strerror(errno));
        }
    }
    if (num_batches > 0) {
        fsync_directory(output_dir / make_batch_filename(0));
    }
    auto t_phase3_end = std::chrono::steady_clock::now();

    std::cout << "[Materialize] partitioned packer DONE — total "
              << std::chrono::duration_cast<std::chrono::seconds>(
                     t_phase3_end - t_phase01_start).count() << "s ("
              << "phase01=" << std::chrono::duration_cast<std::chrono::seconds>(
                     t_phase01_end - t_phase01_start).count() << "s, "
              << "phase2=" << std::chrono::duration_cast<std::chrono::seconds>(
                     t_phase2_end - t_phase2_start).count() << "s, "
              << "phase3=" << std::chrono::duration_cast<std::chrono::seconds>(
                     t_phase3_end - t_phase3_start).count() << "s)"
              << std::endl;
}

} // namespace mdb::gnn
