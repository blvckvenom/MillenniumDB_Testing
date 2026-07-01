// src/gnn/storage/packed_full_store.cc
#include "gnn/storage/packed_full_store.h"
#include "gnn/common/posix_io.h"
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>
namespace mdb::gnn {

static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

PackedFullWriter::PackedFullWriter(const std::filesystem::path& dir, uint64_t store_fp,
                                   uint32_t feature_dim, uint32_t dtype, uint64_t row_bytes)
    : dir_(dir),
      hdr_(PackedFullHeader::make(store_fp, feature_dim, dtype, /*num_batches=*/0, row_bytes)) {
    std::filesystem::create_directories(dir_);
    auto dat = dir_ / "packed_full.dat";
    dat_fd_ = ::open(dat.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (dat_fd_ < 0)
        throw std::runtime_error("PackedFullWriter: open " + dat.string() + ": " + safe_strerror(errno));
}
PackedFullWriter::~PackedFullWriter() { if (dat_fd_ >= 0) ::close(dat_fd_); }

void PackedFullWriter::write_batch(uint64_t batch_id, const void* payload, uint64_t num_nodes) {
    if (finalized_) throw std::runtime_error("PackedFullWriter: write_batch after finalize");
    const uint64_t len = num_nodes * hdr_.row_bytes;
    const uint64_t off = align_up(cur_offset_, hdr_.alignment);
    if (off != cur_offset_) {  // pad
        static const char zeros[PackedFullHeader::ALIGNMENT] = {0};
        uint64_t pad = off - cur_offset_;
        while (pad > 0) { uint64_t n = pad < sizeof(zeros) ? pad : sizeof(zeros);
                          write_all(dat_fd_, zeros, n, "packed_full.dat pad"); pad -= n; }
    }
    write_all(dat_fd_, payload, len, "packed_full.dat");
    cur_offset_ = off + len;
    if (entries_.size() <= batch_id) entries_.resize(batch_id + 1);
    entries_[batch_id] = PackedFullEntry{off, len, num_nodes};
}

void PackedFullWriter::finalize() {
    if (finalized_) return;
    if (::fsync(dat_fd_) != 0)
        throw std::runtime_error(std::string("PackedFullWriter: fsync dat: ") + safe_strerror(errno));
    ::close(dat_fd_); dat_fd_ = -1;
    hdr_.num_batches = entries_.size();
    auto idx = dir_ / "packed_full.idx";
    auto tmp = idx; tmp += ".tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw std::runtime_error("PackedFullWriter: open idx tmp: " + safe_strerror(errno));
    try {
        write_all(fd, &hdr_, sizeof(hdr_), tmp.string());
        if (!entries_.empty())
            write_all(fd, entries_.data(), entries_.size() * sizeof(PackedFullEntry), tmp.string());
        if (::fsync(fd) != 0) throw std::runtime_error(std::string("fsync idx: ") + safe_strerror(errno));
    } catch (...) { ::close(fd); ::unlink(tmp.c_str()); throw; }
    ::close(fd);
    std::error_code rec;
    std::filesystem::rename(tmp, idx, rec);
    if (rec) { ::unlink(tmp.c_str()); throw std::runtime_error("PackedFullWriter: rename idx: " + rec.message()); }
    fsync_directory(idx);  // fsync_directory fsyncs idx's parent (dir_), the dir
                           // that actually contains the renamed .idx (cf. block_store.cc)
    finalized_ = true;
}

std::optional<PackedFullReader> PackedFullReader::open(const std::filesystem::path& dir,
                                                       uint64_t expected_store_fp) {
    PackedFullReader r; r.dir_ = dir;
    auto idx = dir / "packed_full.idx";
    int fd = ::open(idx.c_str(), O_RDONLY);
    if (fd < 0) return std::nullopt;
    FdGuard guard(fd);
    if (::read(fd, &r.hdr_, sizeof(r.hdr_)) != (ssize_t)sizeof(r.hdr_)) return std::nullopt;
    if (!r.hdr_.is_valid() || expected_store_fp == 0 || r.hdr_.store_fp != expected_store_fp)
        return std::nullopt;
    r.entries_.resize(r.hdr_.num_batches);
    if (r.hdr_.num_batches > 0) {
        size_t bytes = r.hdr_.num_batches * sizeof(PackedFullEntry);
        if (::read(fd, r.entries_.data(), bytes) != (ssize_t)bytes) return std::nullopt;
    }
    return r;
}

void PackedFullReader::read_payload_for_test(uint64_t batch_id, void* out) const {
    auto e = entries_.at(batch_id);
    int fd = ::open(dat_path().c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("read_payload_for_test: open dat");
    FdGuard guard(fd);
    if (::pread(fd, out, e.length, e.offset) != (ssize_t)e.length)
        throw std::runtime_error("read_payload_for_test: short pread");
}
} // namespace mdb::gnn
