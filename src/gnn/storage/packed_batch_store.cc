#include "gnn/storage/packed_batch_store.h"

#include <cerrno>
#include <cinttypes>
#include <cstring>
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

    // Read data directly — fd is already positioned right after the header
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

} // namespace mdb::gnn
