// src/gnn/storage/addr_table_writer.cc
#include "gnn/storage/addr_table_writer.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace mdb::gnn {

namespace fs = std::filesystem;

namespace {

void write_all(int fd, const void* buf, size_t n, const std::string& ctx) {
    const auto* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("AddrTableWriter: write failed on " + ctx
                                     + ": " + std::strerror(errno));
        }
        p += static_cast<size_t>(w);
        n -= static_cast<size_t>(w);
    }
}

void write_vec_u32(int fd, const std::vector<uint32_t>& v, const std::string& ctx) {
    if (!v.empty()) write_all(fd, v.data(), v.size() * sizeof(uint32_t), ctx);
}

void write_vec_u64(int fd, const std::vector<uint64_t>& v, const std::string& ctx) {
    if (!v.empty()) write_all(fd, v.data(), v.size() * sizeof(uint64_t), ctx);
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
        write_all(fd, &buf.header, sizeof(buf.header), tmp);
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
}

} // namespace mdb::gnn
