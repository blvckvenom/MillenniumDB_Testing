// src/gnn/storage/addr_table_reader.cc
#include "gnn/storage/addr_table_reader.h"
#include "gnn/common/posix_io.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace mdb::gnn {

namespace fs = std::filesystem;

AddrTableReader::Result
AddrTableReader::open(const fs::path& path, uint64_t expected_meta_sha_head)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("AddrTableReader: cannot open " + path.string()
                                 + ": " + std::strerror(errno));
    }
    struct stat st{};
    if (::fstat(fd, &st) < 0) {
        int saved = errno;
        ::close(fd);
        throw std::runtime_error("AddrTableReader: fstat failed on " + path.string()
                                 + ": " + std::strerror(saved));
    }
    const size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < AddrTableHeader::SIZE) {
        ::close(fd);
        throw std::runtime_error("AddrTableReader: file too small for header: "
                                 + path.string());
    }

    Result res;
    res.data.resize(file_size);
    try {
        mdb::gnn::read_all(fd, res.data.data(), file_size, path.string());
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);

    std::memcpy(&res.header, res.data.data(), sizeof(res.header));

    if (!res.header.is_valid()) {
        throw std::runtime_error("AddrTableReader: invalid header in " + path.string());
    }
    if (res.header.expected_file_size() != file_size) {
        throw std::runtime_error("AddrTableReader: size mismatch in " + path.string()
                                 + " (header says " + std::to_string(res.header.expected_file_size())
                                 + ", actual " + std::to_string(file_size) + ")");
    }
    if (expected_meta_sha_head != 0 &&
        res.header.meta_sha256_head != expected_meta_sha_head)
    {
        throw AddrTableStaleException("AddrTableReader: meta_sha mismatch in "
                                       + path.string());
    }

    // Wire ConstView pointers into res.data.
    // Layout mirrors AddrTableHeader::expected_file_size():
    //   l1_positions[num_l1] l1_indices[num_l1]
    //   l2_positions[num_l2] l2_indices[num_l2]
    //   l3_positions[num_l3] l3_row_idxs[num_l3] (uint64)
    //   l4_positions[num_l4] l4_indices[num_l4]
    //   zero_positions[num_zero]
    const unsigned char* p = res.data.data() + sizeof(res.header);

    auto take_u32 = [&](uint32_t n) -> const uint32_t* {
        const auto* r = reinterpret_cast<const uint32_t*>(p);
        p += static_cast<size_t>(n) * sizeof(uint32_t);
        return r;
    };
    auto take_u64 = [&](uint32_t n) -> const uint64_t* {
        const auto* r = reinterpret_cast<const uint64_t*>(p);
        p += static_cast<size_t>(n) * sizeof(uint64_t);
        return r;
    };

    res.l1_positions   = {take_u32(res.header.num_l1), res.header.num_l1};
    res.l1_indices     = {take_u32(res.header.num_l1), res.header.num_l1};
    res.l2_positions   = {take_u32(res.header.num_l2), res.header.num_l2};
    res.l2_indices     = {take_u32(res.header.num_l2), res.header.num_l2};
    res.l3_positions   = {take_u32(res.header.num_l3), res.header.num_l3};
    res.l3_row_idxs    = {take_u64(res.header.num_l3), res.header.num_l3};
    res.l4_positions   = {take_u32(res.header.num_l4), res.header.num_l4};
    res.l4_indices     = {take_u32(res.header.num_l4), res.header.num_l4};
    res.zero_positions = {take_u32(res.header.num_zero), res.header.num_zero};

    return res;
}

} // namespace mdb::gnn
