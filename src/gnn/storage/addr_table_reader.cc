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
    if (file_size < AddrTableHeader::SIZE_V1) {
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

    // Version-aware header read. The common 40-byte prefix (magic, version,
    // tier counts, total, meta_sha) is identical across v1/v2 and always present
    // (file_size >= SIZE_V1 checked above). Validate magic+version+total from it
    // BEFORE deciding whether to pull the 16-byte v2 extension, so a corrupt or
    // unknown-version file cannot drive an over-read. v1 leaves slim_* == 0
    // (zero-init below); v2 fills slim_offset/slim_length from bytes [40,56).
    std::memset(&res.header, 0, sizeof(res.header));
    std::memcpy(&res.header, res.data.data(), AddrTableHeader::SIZE_V1);

    if (!res.header.is_valid()) {
        throw std::runtime_error("AddrTableReader: invalid header in " + path.string());
    }
    if (res.header.version >= AddrTableHeader::VERSION_V2) {
        if (file_size < AddrTableHeader::SIZE) {
            throw std::runtime_error("AddrTableReader: file too small for v2 header: "
                                     + path.string());
        }
        std::memcpy(&res.header, res.data.data(), AddrTableHeader::SIZE);
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
    const unsigned char* p = res.data.data() + res.header.header_bytes();

    auto take_u32 = [&](uint32_t n) -> const uint32_t* {
        const auto* r = reinterpret_cast<const uint32_t*>(p);
        p += static_cast<size_t>(n) * sizeof(uint32_t);
        return r;
    };
    // l3_row_idxs is the only uint64 array. Its byte offset is
    //   40 + 8*num_l1 + 8*num_l2 + 4*num_l3
    // which is 4-aligned but only 8-aligned when num_l3 is even.
    // Copy into an owned, properly-aligned vector to avoid UB.
    res.l1_positions   = {take_u32(res.header.num_l1), res.header.num_l1};
    res.l1_indices     = {take_u32(res.header.num_l1), res.header.num_l1};
    res.l2_positions   = {take_u32(res.header.num_l2), res.header.num_l2};
    res.l2_indices     = {take_u32(res.header.num_l2), res.header.num_l2};
    res.l3_positions   = {take_u32(res.header.num_l3), res.header.num_l3};

    res.l3_row_idxs_storage.resize(res.header.num_l3);
    if (res.header.num_l3 > 0) {
        std::memcpy(res.l3_row_idxs_storage.data(), p,
                    static_cast<size_t>(res.header.num_l3) * sizeof(uint64_t));
    }
    p += static_cast<size_t>(res.header.num_l3) * sizeof(uint64_t);
    res.l3_row_idxs = {res.l3_row_idxs_storage.data(), res.header.num_l3};

    res.l4_positions   = {take_u32(res.header.num_l4), res.header.num_l4};
    res.l4_indices     = {take_u32(res.header.num_l4), res.header.num_l4};
    res.zero_positions = {take_u32(res.header.num_zero), res.header.num_zero};

    return res;
}

} // namespace mdb::gnn
