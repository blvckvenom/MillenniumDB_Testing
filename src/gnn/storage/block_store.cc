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
                        const std::vector<torch::Tensor>& edge_indices,
                        uint64_t store_fp, uint64_t num_unique_nodes,
                        const std::vector<uint64_t>& seed_ids, uint32_t split) {
    const uint32_t K = static_cast<uint32_t>(edge_indices.size());      // conv layers
    // node layers = K+1; active_sizes must have K+1 entries.
    if (active_sizes.size() != static_cast<size_t>(K) + 1)
        throw std::runtime_error("BlockWriter: active_sizes.size() must == edge_indices.size()+1");
    auto h = BlockBatchHeader::make_self_contained(K, sample_fp, batch_id, store_fp,
                                                   num_unique_nodes, seed_ids.size(), split);

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
        // v2 self-contained tail: seed ObjectId.ids (num_seeds × uint64). Empty for the
        // default / legacy callers (num_seeds==0) — no seed bytes written in that case.
        if (!seed_ids.empty()) {
            write_all(fd, seed_ids.data(), seed_ids.size() * sizeof(uint64_t), tmp);
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

namespace {
// Shared body read for open() / open_self_contained(): given an ifstream
// positioned right after a validated 64-byte header, decode active_sizes, the
// per-conv-layer edge tensors (int32 on disk, widened to int64), and the v2
// seed_ids tail. Populates the v2 header-derived fields from `h`. Returns
// nullopt on any short read so a torn/truncated block falls back to online.
std::optional<LoadedBlock> read_block_body(std::ifstream& is, const BlockBatchHeader& h) {
    const uint32_t K = h.num_layers;
    // The header counts come straight off disk and drive allocations below.
    // The magic + fingerprint checks do not cover them, so a corrupt count
    // (bit rot, tampering) must degrade to nullopt — the graceful online
    // fallback — instead of throwing bad_alloc / c10::Error out of assemble.
    constexpr uint32_t MAX_LAYERS = 64;          // fanout depth bound
    constexpr int64_t  MAX_EDGES  = (1ll << 31); // edge indices are int32 on disk
    if (K > MAX_LAYERS) return std::nullopt;
    // Remaining bytes after the 64 B header; every count below must add up
    // to exactly this (the writer emits header + body with no padding).
    const std::streamoff body_start = is.tellg();
    is.seekg(0, std::ios::end);
    const std::streamoff file_end = is.tellg();
    is.seekg(body_start, std::ios::beg);
    if (!is || body_start < 0 || file_end < body_start) return std::nullopt;
    const uint64_t remaining = static_cast<uint64_t>(file_end - body_start);
    const uint64_t fixed_bytes = (2ull * K + 1) * sizeof(int64_t);  // M_k + E_k arrays
    if (remaining < fixed_bytes) return std::nullopt;
    if (h.num_seeds > remaining / sizeof(uint64_t)) return std::nullopt;
    LoadedBlock out;
    // v2 self-contained header fields (0 for legacy v1 / non-self-contained v2 blocks).
    out.store_fp         = h.store_fp;
    out.num_unique_nodes = h.num_unique_nodes;
    out.split            = h.split;
    out.active_sizes.resize(static_cast<size_t>(K) + 1);
    is.read(reinterpret_cast<char*>(out.active_sizes.data()),
            static_cast<std::streamsize>(out.active_sizes.size() * sizeof(int64_t)));
    std::vector<int64_t> E(K);
    is.read(reinterpret_cast<char*>(E.data()), static_cast<std::streamsize>(K * sizeof(int64_t)));
    if (!is) return std::nullopt;
    uint64_t edge_bytes = 0;
    for (uint32_t k = 0; k < K; ++k) {
        if (E[k] < 0 || E[k] >= MAX_EDGES) return std::nullopt;
        edge_bytes += 2ull * static_cast<uint64_t>(E[k]) * sizeof(int32_t);
    }
    if (fixed_bytes + edge_bytes + h.num_seeds * sizeof(uint64_t) != remaining) {
        return std::nullopt;
    }
    out.edge_indices.reserve(K);
    for (uint32_t k = 0; k < K; ++k) {
        const int64_t Ek = E[k];
        auto t32 = torch::empty({2, Ek}, torch::kInt32);
        is.read(reinterpret_cast<char*>(t32.data_ptr<int32_t>()),
                static_cast<std::streamsize>(t32.numel() * sizeof(int32_t)));
        if (!is) return std::nullopt;
        out.edge_indices.push_back(t32.to(torch::kInt64));   // widen
    }
    // v2 self-contained tail: num_seeds uint64 seed ids (header.num_seeds==0 for v1).
    if (h.num_seeds > 0) {
        out.seed_ids.resize(static_cast<size_t>(h.num_seeds));
        is.read(reinterpret_cast<char*>(out.seed_ids.data()),
                static_cast<std::streamsize>(out.seed_ids.size() * sizeof(uint64_t)));
        if (!is) return std::nullopt;  // short read on the seed tail
    }
    return out;
}
} // namespace

std::optional<LoadedBlock> BlockReader::open(const std::filesystem::path& path, uint64_t expected_sample_fp) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return std::nullopt;
    BlockBatchHeader h{};
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!is || !h.is_valid() || h.sample_fp != expected_sample_fp) return std::nullopt;
    return read_block_body(is, h);
}

std::optional<LoadedBlock> BlockReader::open_self_contained(
    const std::filesystem::path& path, uint64_t expected_store_fp) {
    // Validate via the STORE fingerprint, not the per-batch sample_fp. The block
    // must be self-contained (version>=2 && store_fp!=0) and its store_fp must
    // match the caller's catalog fingerprint; a 0 expected_store_fp is rejected
    // (UNKNOWN provenance must never adopt a block silently).
    if (expected_store_fp == 0) return std::nullopt;
    std::ifstream is(path, std::ios::binary);
    if (!is) return std::nullopt;
    BlockBatchHeader h{};
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!is || !h.is_valid() || !h.is_self_contained()
        || h.store_fp != expected_store_fp) {
        return std::nullopt;
    }
    return read_block_body(is, h);
}

bool BlockReader::is_fresh(const std::filesystem::path& path, uint64_t expected_sample_fp,
                           uint64_t expected_store_fp) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    BlockBatchHeader h{};
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!is) return false;  // short read / open failure
    // The bake always writes the current format: a valid-but-older block must
    // NOT count as fresh, or the re-bake would silently keep it and the
    // self-contained train path would never become eligible.
    if (!h.is_valid() || h.version < BlockBatchHeader::VERSION) return false;
    if (h.sample_fp != expected_sample_fp) return false;
    // A self-contained bake (expected_store_fp != 0) further requires the
    // existing block to carry the SAME store fingerprint; otherwise
    // open_self_contained would reject it at train time and the speedup
    // would silently degrade to the batches.dat fallback.
    if (expected_store_fp != 0
        && (!h.is_self_contained() || h.store_fp != expected_store_fp)) {
        return false;
    }
    return true;
}

uint64_t BlockReader::read_store_fp(const std::filesystem::path& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return 0;  // open failure
    BlockBatchHeader h{};
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!is) return 0;  // short read
    return h.is_valid() ? h.store_fp : 0;
}
} // namespace mdb::gnn
