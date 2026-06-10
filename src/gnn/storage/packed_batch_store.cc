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
#include "gnn/storage/consolidated_slim.h"

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
    std::function<std::vector<uint64_t>(uint64_t batch_id)> row_provider,
    const fs::path& output_dir,
    size_t partition_bytes,
    std::function<std::vector<ObjectId>(uint64_t batch_id)> oid_provider,
    const fs::path& consolidated_path,
    uint64_t consolidated_perm_fp,
    uint64_t consolidated_meta_sha,
    std::vector<uint64_t>* out_consolidated_offsets,
    std::vector<uint64_t>* out_consolidated_lengths)
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

    const uint64_t feature_dim = features.num_cols();
    const auto     dtype       = features.dtype();
    const bool     has_oids    = static_cast<bool>(oid_provider);

    // --- DiskGNN-adoption Plan 1: optional consolidated cold-feature file ---
    // One file = ConsolidatedSlimHeader + every batch's data section, batch b
    // at a 4096-aligned offset. Written in the SAME .fmat scan as the per-batch
    // .bin files (which are left unchanged). cons_offset[b] is the byte offset
    // of batch b's payload; the payload is cons_length[b] = batch_size[b]*row_bytes.
    const bool write_consolidated = !consolidated_path.empty();
    ConsolidatedSlimHeader cons_header{};
    uint64_t cons_align = 4096;
    if (write_consolidated) {
        cons_header = ConsolidatedSlimHeader::make(
            num_batches, feature_dim, static_cast<uint8_t>(dtype),
            consolidated_perm_fp, consolidated_meta_sha);
        cons_align = cons_header.alignment();
    }
    std::vector<uint64_t> cons_offset(write_consolidated ? num_batches : 0, 0);
    uint64_t cons_cursor = write_consolidated ? cons_header.data_start : 0;
    int      cons_fd     = -1;
    if (out_consolidated_offsets) out_consolidated_offsets->assign(num_batches, 0);
    if (out_consolidated_lengths) out_consolidated_lengths->assign(num_batches, 0);

    // Partition geometry: target partition_bytes, round down to whole rows.
    uint64_t partition_rows = std::max<uint64_t>(1, partition_bytes / row_bytes);
    if (partition_rows > total_rows) partition_rows = total_rows;
    const uint64_t num_partitions =
        (total_rows + partition_rows - 1) / partition_rows;

    std::cout << "[Materialize] partitioned packer (B1): "
              << num_partitions << " partitions × " << partition_rows
              << " rows ("
              << (partition_rows * row_bytes / (1024ULL * 1024)) << " MB each), "
              << num_batches << " batches, format="
              << (has_oids ? "v2" : "v1")
              << "\n" << std::flush;

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

    // ----- Phase 0+1: build inverted index, OID lists, open all batch FDs -----
    //
    // We open all num_batches output files up-front and KEEP them open until
    // Phase 3. Relies on ulimit -n >= num_batches + small overhead.
    //
    // Pre-fix B1 (the in-RAM buffer approach) needed total_refs * row_bytes
    // of RAM (~36 GB on papers100M), which OOM-killed celebi. The persistent
    // FD design lets each (partition × batch) contribution land in its final
    // file slot via pwrite using a per-batch cursor — no full-batch buffering.
    auto t_phase01_start = std::chrono::steady_clock::now();

    std::vector<std::vector<RowRef>>    inverted(num_partitions);
    std::vector<uint64_t>               batch_size(num_batches, 0);
    std::vector<std::vector<ObjectId>>  batch_oids(has_oids ? num_batches : 0);
    std::vector<int>                    fds(num_batches, -1);

    auto cleanup_fds = [&]() {
        for (auto& fd : fds) {
            if (fd >= 0) {
                ::close(fd);
                fd = -1;
            }
        }
        if (cons_fd >= 0) {
            ::close(cons_fd);
            cons_fd = -1;
        }
    };

    try {
        for (uint64_t b = 0; b < num_batches; ++b) {
            auto rows = row_provider(b);
            const uint64_t N = rows.size();
            batch_size[b] = N;

            if (write_consolidated) {
                const uint64_t payload = N * row_bytes;
                cons_offset[b] = cons_cursor;
                if (out_consolidated_offsets) (*out_consolidated_offsets)[b] = cons_cursor;
                if (out_consolidated_lengths) (*out_consolidated_lengths)[b] = payload;
                cons_cursor += ConsolidatedSlimHeader::align_up(payload, cons_align);
            }

            if (has_oids) {
                batch_oids[b] = oid_provider(b);
                if (batch_oids[b].size() != N) {
                    throw std::runtime_error(
                        "generate_packed_batches_partitioned: OID/row count "
                        "mismatch for batch " + std::to_string(b) + " (oids=" +
                        std::to_string(batch_oids[b].size()) + ", rows=" +
                        std::to_string(N) + ")");
                }
            }

            // Bucket each row into its partition.
            for (size_t i = 0; i < N; ++i) {
                const uint64_t fmat_row = rows[i];
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

            // Open file, write header, ftruncate to total size.
            auto path = output_dir / make_batch_filename(b);
            std::string path_str = path.string();

            int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                throw std::runtime_error(
                    "generate_packed_batches_partitioned: cannot create " +
                    path_str + ": " + safe_strerror(errno));
            }
            fds[b] = fd;

            auto header = has_oids
                ? PackedBatchHeader::make_v2(N, feature_dim, dtype)
                : PackedBatchHeader::make(N, feature_dim, dtype);
            write_all(fd, &header, sizeof(header), path_str);

            if (N > 0) {
                const off_t total_size = static_cast<off_t>(
                    sizeof(header)
                    + (has_oids ? N * sizeof(uint64_t) : 0)
                    + N * row_bytes);
                if (::ftruncate(fd, total_size) < 0) {
                    throw std::runtime_error(
                        "generate_packed_batches_partitioned: ftruncate failed for " +
                        path_str + ": " + safe_strerror(errno));
                }
            }
        }

        // Consolidated file: write the header and size the whole file once
        // (header + sum of 4096-aligned per-batch payloads). Payloads are
        // pwritten during Phase 2 alongside the per-batch data.
        if (write_consolidated) {
            std::string cpath = consolidated_path.string();
            cons_fd = ::open(cpath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (cons_fd < 0) {
                throw std::runtime_error(
                    "generate_packed_batches_partitioned: cannot create consolidated " +
                    cpath + ": " + safe_strerror(errno));
            }
            write_all(cons_fd, &cons_header, sizeof(cons_header), cpath);
            if (cons_cursor > cons_header.data_start) {
                if (::ftruncate(cons_fd, static_cast<off_t>(cons_cursor)) < 0) {
                    throw std::runtime_error(
                        "generate_packed_batches_partitioned: ftruncate consolidated "
                        "failed for " + cpath + ": " + safe_strerror(errno));
                }
            }
        }
    } catch (...) {
        cleanup_fds();
        throw;
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

    // ----- Phase 2: single pass over .fmat, pwrite per (partition × batch) ----
    //
    // For each partition:
    //   1. memcpy the partition from mmap'd .fmat into a local buffer
    //   2. sort refs by batch_id so each batch's contribution is contiguous
    //   3. for each batch group: build contiguous OID and data slices,
    //      pwrite them at offset = base + cursor[batch] * stride
    //   4. cursor[batch] += group_size
    //
    // OID/data layout in v2: header || OID_table[0..N-1] || data[0..N-1].
    // The OID table records the same partition-iteration order as the data
    // section, so oid_table[i] ↔ data[i] is preserved. The on-disk row order
    // is NOT sample-input order (drops bit-identity vs classic), but the
    // (oid, row) pairing is correct for any consumer that uses an OID lookup.
    auto t_phase2_start = std::chrono::steady_clock::now();

    std::vector<uint64_t> cursor(num_batches, 0);
    std::vector<char>     partition_buf;
    partition_buf.reserve(partition_rows * row_bytes);

    // Reusable per-(partition, batch) contribution buffers.
    std::vector<char>     contrib_data;
    std::vector<uint64_t> contrib_oids;

    const uint64_t progress_step = std::max<uint64_t>(num_partitions / 20, 4);

    try {
        for (uint64_t p = 0; p < num_partitions; ++p) {
            auto& refs = inverted[p];
            if (refs.empty()) continue;

            const uint64_t row_start = p * partition_rows;
            const uint64_t row_end   = std::min(row_start + partition_rows, total_rows);
            const uint64_t this_rows = row_end - row_start;
            const size_t   this_bytes = static_cast<size_t>(this_rows * row_bytes);

            // (1) Sequential memcpy from mmap.
            partition_buf.resize(this_bytes);
            std::memcpy(partition_buf.data(),
                        features.row(row_start),
                        this_bytes);

            // (2) Sort refs by (batch_id, pos_in_batch) so each batch's
            // contribution is one contiguous group — one open/pwrite per batch
            // contribution. pos_in_batch is part of the key because std::sort
            // is unstable: keyed on batch_id alone it may permute equal-key
            // refs, breaking the v1 positional contract (rows within a group
            // must keep row_provider order).
            std::sort(refs.begin(), refs.end(),
                      [](const RowRef& a, const RowRef& b) {
                          return a.batch_id != b.batch_id
                                   ? a.batch_id < b.batch_id
                                   : a.pos_in_batch < b.pos_in_batch;
                      });

            size_t i = 0;
            while (i < refs.size()) {
                const uint32_t bid = refs[i].batch_id;
                size_t j = i + 1;
                while (j < refs.size() && refs[j].batch_id == bid) ++j;

                const size_t group_size = j - i;

                // Build contiguous data + OID slices for this contribution.
                contrib_data.resize(group_size * row_bytes);
                if (has_oids) contrib_oids.resize(group_size);

                for (size_t k = 0; k < group_size; ++k) {
                    const auto& ref = refs[i + k];
                    const size_t buf_offset =
                        static_cast<size_t>((ref.fmat_row - row_start) * row_bytes);
                    std::memcpy(contrib_data.data() + k * row_bytes,
                                partition_buf.data() + buf_offset,
                                row_bytes);
                    if (has_oids) {
                        contrib_oids[k] = batch_oids[bid][ref.pos_in_batch].id;
                    }
                }

                // Compute target offsets in this batch's file.
                const uint64_t batch_oid_off  = sizeof(PackedBatchHeader)
                                              + cursor[bid] * sizeof(uint64_t);
                const uint64_t batch_data_off = sizeof(PackedBatchHeader)
                                              + (has_oids
                                                 ? batch_size[bid] * sizeof(uint64_t)
                                                 : 0)
                                              + cursor[bid] * row_bytes;

                // pwrite OID slice (v2 only).
                if (has_oids) {
                    size_t       remaining = group_size * sizeof(uint64_t);
                    const char*  src       = reinterpret_cast<const char*>(contrib_oids.data());
                    off_t        off_curr  = static_cast<off_t>(batch_oid_off);
                    while (remaining > 0) {
                        ssize_t n = ::pwrite(fds[bid], src, remaining, off_curr);
                        if (n < 0) {
                            if (errno == EINTR) continue;
                            throw std::runtime_error(
                                "generate_packed_batches_partitioned: pwrite OIDs failed: " +
                                safe_strerror(errno));
                        }
                        if (n == 0) {
                            throw std::runtime_error(
                                "generate_packed_batches_partitioned: pwrite OIDs returned 0");
                        }
                        src       += n;
                        off_curr  += n;
                        remaining -= static_cast<size_t>(n);
                    }
                }

                // pwrite data slice.
                {
                    size_t       remaining = group_size * row_bytes;
                    const char*  src       = contrib_data.data();
                    off_t        off_curr  = static_cast<off_t>(batch_data_off);
                    while (remaining > 0) {
                        ssize_t n = ::pwrite(fds[bid], src, remaining, off_curr);
                        if (n < 0) {
                            if (errno == EINTR) continue;
                            throw std::runtime_error(
                                "generate_packed_batches_partitioned: pwrite data failed: " +
                                safe_strerror(errno));
                        }
                        if (n == 0) {
                            throw std::runtime_error(
                                "generate_packed_batches_partitioned: pwrite data returned 0");
                        }
                        src       += n;
                        off_curr  += n;
                        remaining -= static_cast<size_t>(n);
                    }
                }

                // Consolidated file: same contrib_data at this batch's aligned
                // base + the same row cursor, so the consolidated payload is
                // byte-identical to the per-batch .bin data section.
                if (write_consolidated) {
                    size_t       remaining = group_size * row_bytes;
                    const char*  src       = contrib_data.data();
                    off_t        off_curr  = static_cast<off_t>(
                        cons_offset[bid] + cursor[bid] * row_bytes);
                    while (remaining > 0) {
                        ssize_t n = ::pwrite(cons_fd, src, remaining, off_curr);
                        if (n < 0) {
                            if (errno == EINTR) continue;
                            throw std::runtime_error(
                                "generate_packed_batches_partitioned: pwrite consolidated failed: " +
                                safe_strerror(errno));
                        }
                        if (n == 0) {
                            throw std::runtime_error(
                                "generate_packed_batches_partitioned: pwrite consolidated returned 0");
                        }
                        src       += n;
                        off_curr  += n;
                        remaining -= static_cast<size_t>(n);
                    }
                }

                cursor[bid] += group_size;
                i = j;
            }

            // Free this partition's inverted bucket.
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

        // Sanity: each batch should have its full row count written.
        for (uint64_t b = 0; b < num_batches; ++b) {
            if (cursor[b] != batch_size[b]) {
                throw std::runtime_error(
                    "generate_packed_batches_partitioned: batch " +
                    std::to_string(b) + " cursor mismatch: " +
                    std::to_string(cursor[b]) + " written, expected " +
                    std::to_string(batch_size[b]));
            }
        }
    } catch (...) {
        cleanup_fds();
        throw;
    }

    auto t_phase2_end = std::chrono::steady_clock::now();

    // ----- Phase 3: fsync + close all batch FDs -----
    auto t_phase3_start = std::chrono::steady_clock::now();
    try {
        for (uint64_t b = 0; b < num_batches; ++b) {
            if (fds[b] < 0) continue;
            if (::fsync(fds[b]) < 0) {
                throw std::runtime_error(
                    "generate_packed_batches_partitioned: fsync failed for batch " +
                    std::to_string(b) + ": " + safe_strerror(errno));
            }
            ::close(fds[b]);
            fds[b] = -1;
        }
        if (num_batches > 0) {
            fsync_directory(output_dir / make_batch_filename(0));
        }
        if (write_consolidated && cons_fd >= 0) {
            if (::fsync(cons_fd) < 0) {
                throw std::runtime_error(
                    "generate_packed_batches_partitioned: fsync consolidated failed: " +
                    safe_strerror(errno));
            }
            ::close(cons_fd);
            cons_fd = -1;
            fsync_directory(consolidated_path);
        }
    } catch (...) {
        cleanup_fds();
        throw;
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
