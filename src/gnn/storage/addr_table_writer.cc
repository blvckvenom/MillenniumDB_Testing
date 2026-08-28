// src/gnn/storage/addr_table_writer.cc
#include "gnn/storage/addr_table_writer.h"
#include "gnn/common/posix_io.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace mdb::gnn {

namespace fs = std::filesystem;

namespace {

void write_vec_u32(int fd, const std::vector<uint32_t>& v, const std::string& ctx) {
    if (!v.empty()) mdb::gnn::write_all(fd, v.data(), v.size() * sizeof(uint32_t), ctx);
}

void write_vec_u64(int fd, const std::vector<uint64_t>& v, const std::string& ctx) {
    if (!v.empty()) mdb::gnn::write_all(fd, v.data(), v.size() * sizeof(uint64_t), ctx);
}

} // namespace

void AddrTableWriter::write_atomic(const fs::path& path,
                                    const AddrTableBuffers& buf)
{
    auto tmp = path.string() + ".tmp";

    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("AddrTableWriter: cannot create " + tmp
                                 + ": " + std::strerror(errno));
    }

    try {
        // Version-aware header size: v1 writes 40 bytes (byte-identical to
        // pre-v2 output), v2 writes 56 bytes (incl. slim_offset/slim_length).
        // sizeof(buf.header) is always 56 (the in-memory struct), so write only
        // header_bytes() to keep v1 sidecars unchanged on disk.
        mdb::gnn::write_all(fd, &buf.header, buf.header.header_bytes(), tmp);
        write_vec_u32(fd, buf.l1_positions, tmp);
        write_vec_u32(fd, buf.l1_indices, tmp);
        write_vec_u32(fd, buf.l2_positions, tmp);
        write_vec_u32(fd, buf.l2_indices, tmp);
        write_vec_u32(fd, buf.l3_positions, tmp);
        write_vec_u64(fd, buf.l3_row_idxs, tmp);
        write_vec_u32(fd, buf.l4_positions, tmp);
        write_vec_u32(fd, buf.l4_indices, tmp);
        write_vec_u32(fd, buf.zero_positions, tmp);

        if (::fsync(fd) < 0) {
            throw std::runtime_error("AddrTableWriter: fsync failed on " + tmp
                                     + ": " + std::strerror(errno));
        }
    } catch (...) {
        ::close(fd);
        ::unlink(tmp.c_str());
        throw;
    }
    ::close(fd);

    if (::rename(tmp.c_str(), path.c_str()) < 0) {
        ::unlink(tmp.c_str());
        throw std::runtime_error("AddrTableWriter: rename failed " + tmp
                                 + " -> " + path.string()
                                 + ": " + std::strerror(errno));
    }
    // Durability: ensure the directory entry is fsync'd so the renamed
    // file survives a crash (matches the convention used by every other
    // atomic-write site in the codebase — feature_matrix.cc, gpu_cache.cc,
    // model_checkpoint.cc, etc.).
    mdb::gnn::fsync_directory(path);
}

} // namespace mdb::gnn
