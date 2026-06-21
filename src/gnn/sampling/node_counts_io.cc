#include "gnn/sampling/node_counts_io.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace mdb::gnn::node_counts_io {

namespace {

constexpr uint8_t kMagic[8] = {'N','O','D','E','C','N','T','0'};

void fsync_directory_(const std::filesystem::path& dir) {
    int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[node_counts_io] WARNING: cannot open dir for "
                  << "fsync " << dir.string() << " (errno=" << errno
                  << "); node_counts.bin may not be durable.\n";
        return;
    }
    if (::fsync(fd) != 0) {
        std::cerr << "[node_counts_io] WARNING: fsync(dir) failed "
                  << "for " << dir.string() << " (errno=" << errno
                  << ").\n";
    }
    ::close(fd);
}

}  // namespace

void persist(const std::filesystem::path&  projection_dir,
             const std::vector<uint64_t>&  counts,
             EdgeOrientation               orientation)
{
    if (projection_dir.empty()) return;
    if (counts.empty())         return;

    // Part 2 reproducibility fix (c), opt-in via env MDB_GNN_FREEZE_NODE_COUNTS=1 (default OFF):
    // once node_counts.bin exists, do NOT overwrite it. The frequency-based four-level TIER ASSIGNMENT
    // reads these per-node access counts; rewriting them each run with THIS run's post-sample counts makes
    // the tier assignment — and thus the served neighbor sets — drift run-to-run, so a fixed seed yields a
    // DIFFERENT sample each warm run (root cause: cold-start reproduces 46f7, warm-start drifts — see
    // docs/research/2026-06-18-feature-store-study/FINAL_REPORT.md). Freezing the counts gives every warm run
    // an identical tier layout => bit-reproducible sample (det4-proven: identical counts => identical sample;
    // and validated to be the SUFFICIENT fix where canonical neighbor order alone was not). Default OFF =>
    // unchanged behavior. Seed deterministic counts by doing ONE cold-start build (no node_counts.bin) first.
    {
        static const bool kFreeze = []() {
            const char* e = std::getenv("MDB_GNN_FREEZE_NODE_COUNTS");
            return e != nullptr && std::string(e) == "1";
        }();
        if (kFreeze) {
            std::error_code fec;
            if (std::filesystem::exists(projection_dir / "node_counts.bin", fec)) {
                std::cerr << "[node_counts_io] MDB_GNN_FREEZE_NODE_COUNTS: node_counts.bin "
                             "exists — skipping overwrite (frozen for reproducible sampling).\n";
                return;
            }
        }
    }

    std::error_code ec;
    std::filesystem::create_directories(projection_dir, ec);
    auto target = projection_dir / "node_counts.bin";
    auto tmp    = projection_dir / "node_counts.bin.tmp";

    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            std::cerr << "[node_counts_io] WARNING: cannot open "
                      << tmp.string() << " for write (errno="
                      << errno << "); node_counts.bin not persisted.\n";
            return;
        }

        const uint64_t num_nodes = static_cast<uint64_t>(counts.size());
        uint64_t direction_bitmask = 0;
        switch (orientation) {
            case EdgeOrientation::NATURAL:    direction_bitmask = 1; break;
            case EdgeOrientation::REVERSE:    direction_bitmask = 2; break;
            case EdgeOrientation::UNDIRECTED: direction_bitmask = 3; break;
        }

        f.write(reinterpret_cast<const char*>(kMagic), 8);
        f.write(reinterpret_cast<const char*>(&num_nodes),         sizeof(num_nodes));
        f.write(reinterpret_cast<const char*>(&direction_bitmask), sizeof(direction_bitmask));
        f.write(reinterpret_cast<const char*>(counts.data()),
                static_cast<std::streamsize>(num_nodes * sizeof(uint64_t)));

        if (!f) {
            std::cerr << "[node_counts_io] WARNING: write failed on "
                      << tmp.string() << " (errno=" << errno
                      << "); node_counts.bin not persisted.\n";
            std::filesystem::remove(tmp, ec);
            return;
        }

        // Flush user-space buffers to the kernel before fsync().
        f.flush();
        if (!f) {
            std::cerr << "[node_counts_io] WARNING: flush failed on "
                      << tmp.string() << " (errno=" << errno
                      << "); node_counts.bin not persisted.\n";
            std::filesystem::remove(tmp, ec);
            return;
        }
    }

    // fsync the temp file, then rename + fsync the directory.
    int fd = ::open(tmp.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }

    std::filesystem::rename(tmp, target, ec);
    if (ec) {
        std::cerr << "[node_counts_io] WARNING: rename "
                  << tmp.string() << " -> " << target.string()
                  << " failed (" << ec.message()
                  << "); node_counts.bin not persisted.\n";
        std::filesystem::remove(tmp, ec);
        return;
    }

    fsync_directory_(projection_dir);
}

}  // namespace mdb::gnn::node_counts_io
