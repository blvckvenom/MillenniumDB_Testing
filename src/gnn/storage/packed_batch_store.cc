#include "gnn/storage/packed_batch_store.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

namespace mdb::gnn {

namespace fs = std::filesystem;

namespace {

class FdGuard {
public:
    explicit FdGuard(int fd) : fd_(fd) {}
    ~FdGuard() { if (fd_ >= 0) ::close(fd_); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
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
                "PackedBatchStore: write failed: " + std::string(std::strerror(errno)));
        }
        if (written == 0) {
            throw std::runtime_error(
                "PackedBatchStore: write returned 0 — disk full or I/O error");
        }
        p += written;
        remaining -= static_cast<size_t>(written);
    }
}

void read_all(int fd, void* buf, size_t count, const std::string& context) {
    char* p = static_cast<char*>(buf);
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t n = ::read(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "PackedBatchStore: read failed: " + std::string(std::strerror(errno)));
        }
        if (n == 0) {
            throw std::runtime_error(
                "PackedBatchStore: unexpected EOF in " + context);
        }
        p += n;
        remaining -= static_cast<size_t>(n);
    }
}

fs::path make_batch_path(const fs::path& dir, uint64_t batch_id) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "batch_%06" PRIu64 ".bin", batch_id);
    return dir / buf;
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
    return make_batch_path(dir_, batch_id);
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

    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "PackedBatchWriter: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    write_all(fd, &header, sizeof(header));
    if (num_nodes > 0) {
        write_all(fd, data, header.data_bytes());
    }

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "PackedBatchWriter: fsync failed: " + std::string(std::strerror(errno)));
    }

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
}

std::filesystem::path PackedBatchReader::batch_path(uint64_t batch_id) const {
    return make_batch_path(dir_, batch_id);
}

PackedBatchHeader PackedBatchReader::read_header(uint64_t batch_id) const {
    if (batch_id >= num_batches_) {
        throw std::out_of_range(
            "PackedBatchReader: batch_id " + std::to_string(batch_id) +
            " >= num_batches " + std::to_string(num_batches_));
    }

    auto path = batch_path(batch_id);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "PackedBatchReader: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    PackedBatchHeader header;
    read_all(fd, &header, sizeof(header), path.string());

    if (!header.is_valid()) {
        throw std::runtime_error(
            "PackedBatchReader: invalid header in " + path.string());
    }
    if (header.feature_dim != feature_dim_) {
        throw std::runtime_error(
            "PackedBatchReader: feature_dim mismatch in " + path.string() +
            ": expected " + std::to_string(feature_dim_) +
            ", got " + std::to_string(header.feature_dim));
    }
    if (header.get_dtype() != dtype_) {
        throw std::runtime_error(
            "PackedBatchReader: dtype mismatch in " + path.string());
    }

    return header;
}

uint64_t PackedBatchReader::read_batch(uint64_t batch_id, void* out,
                                       size_t out_capacity) const {
    auto header = read_header(batch_id);
    size_t data_size = header.data_bytes();

    if (data_size == 0) {
        return 0;
    }

    if (out_capacity < data_size) {
        throw std::runtime_error(
            "PackedBatchReader: buffer too small — need " +
            std::to_string(data_size) + " bytes, got " + std::to_string(out_capacity));
    }

    // Re-open file and skip header (read_header already validated)
    auto path = batch_path(batch_id);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "PackedBatchReader: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    if (::lseek(fd, PackedBatchHeader::SIZE, SEEK_SET) < 0) {
        throw std::runtime_error(
            "PackedBatchReader: lseek failed: " + std::string(std::strerror(errno)));
    }

    read_all(fd, out, data_size, path.string());

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
        batch_assignments.size(),
        [&](uint64_t batch_id) { return batch_assignments[batch_id]; },
        output_dir
    );
}

} // namespace mdb::gnn
