// src/gnn/storage/block_store.cc
#include "gnn/storage/block_store.h"
#include "gnn/storage/block_format.h"
#include "gnn/common/posix_io.h"

#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace mdb::gnn {

void BlockWriter::write(const std::filesystem::path& path, uint64_t sample_fp, uint64_t batch_id,
                        const std::vector<int64_t>& active_sizes,
                        const std::vector<torch::Tensor>& edge_indices) {
    const uint32_t K = static_cast<uint32_t>(edge_indices.size());      // conv layers
    // node layers = K+1; active_sizes must have K+1 entries.
    if (active_sizes.size() != static_cast<size_t>(K) + 1)
        throw std::runtime_error("BlockWriter: active_sizes.size() must == edge_indices.size()+1");
    auto h = BlockBatchHeader::make(K, sample_fp, batch_id);

    // Atomic write: tmp -> fsync fd -> close -> rename -> fsync parent dir.
    // Mirrors addr_table_writer.cc / four_level_store.cc / model_checkpoint.cc
    // so a crash mid-bake cannot leave a torn block that passes the magic+fp
    // check, and no stale .tmp leaks on a mid-write exception.
    auto tmp = path.string() + ".tmp";

    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("BlockWriter: cannot create " + tmp
                                 + ": " + safe_strerror(errno));
    }

    try {
        write_all(fd, &h, sizeof(h), tmp);
        write_all(fd, active_sizes.data(), active_sizes.size() * sizeof(int64_t), tmp);
        std::vector<int64_t> E(K);
        for (uint32_t k = 0; k < K; ++k) E[k] = edge_indices[k].size(1);
        if (K > 0) write_all(fd, E.data(), static_cast<size_t>(K) * sizeof(int64_t), tmp);
        for (uint32_t k = 0; k < K; ++k) {
            auto t = edge_indices[k].to(torch::kInt32).contiguous();        // [2,E_k] int32
            write_all(fd, t.data_ptr<int32_t>(),
                      static_cast<size_t>(t.numel()) * sizeof(int32_t), tmp);
        }
        if (::fsync(fd) < 0) {
            throw std::runtime_error("BlockWriter: fsync failed on " + tmp
                                     + ": " + safe_strerror(errno));
        }
    } catch (...) {
        ::close(fd);
        ::unlink(tmp.c_str());
        throw;
    }
    ::close(fd);

    if (::rename(tmp.c_str(), path.c_str()) < 0) {
        ::unlink(tmp.c_str());
        throw std::runtime_error("BlockWriter: rename failed " + tmp
                                 + " -> " + path.string()
                                 + ": " + safe_strerror(errno));
    }
    // Durability: fsync the parent directory so the renamed file survives a
    // crash (matches every other atomic-write site in the codebase).
    fsync_directory(path);
}

std::optional<LoadedBlock> BlockReader::open(const std::filesystem::path& path, uint64_t expected_sample_fp) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return std::nullopt;
    BlockBatchHeader h{};
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!is || !h.is_valid() || h.sample_fp != expected_sample_fp) return std::nullopt;
    const uint32_t K = h.num_layers;
    LoadedBlock out;
    out.active_sizes.resize(static_cast<size_t>(K) + 1);
    is.read(reinterpret_cast<char*>(out.active_sizes.data()),
            static_cast<std::streamsize>(out.active_sizes.size() * sizeof(int64_t)));
    std::vector<int64_t> E(K);
    is.read(reinterpret_cast<char*>(E.data()), static_cast<std::streamsize>(K * sizeof(int64_t)));
    if (!is) return std::nullopt;
    out.edge_indices.reserve(K);
    for (uint32_t k = 0; k < K; ++k) {
        const int64_t Ek = E[k];
        auto t32 = torch::empty({2, Ek}, torch::kInt32);
        is.read(reinterpret_cast<char*>(t32.data_ptr<int32_t>()),
                static_cast<std::streamsize>(t32.numel() * sizeof(int32_t)));
        if (!is) return std::nullopt;
        out.edge_indices.push_back(t32.to(torch::kInt64));   // widen
    }
    return out;
}

bool BlockReader::is_fresh(const std::filesystem::path& path, uint64_t expected_sample_fp) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    BlockBatchHeader h{};
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!is) return false;  // short read / open failure
    return h.is_valid() && h.sample_fp == expected_sample_fp;
}
} // namespace mdb::gnn
