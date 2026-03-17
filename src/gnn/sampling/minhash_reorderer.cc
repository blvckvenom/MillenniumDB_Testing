#include "gnn/sampling/minhash_reorderer.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <unistd.h>

#include "gnn/common/posix_io.h"

namespace mdb::gnn {

// ===========================================================================
// Constructor + hash
// ===========================================================================

MinHashReorderer::MinHashReorderer(const Config& config) : config_(config) {
    if (config_.num_hashes == 0) {
        throw std::invalid_argument("MinHashReorderer: num_hashes must be > 0");
    }
    if (config_.hashes_per_pass == 0) {
        throw std::invalid_argument("MinHashReorderer: hashes_per_pass must be > 0");
    }

    // Large prime for universal hash family h(x) = (a*x + b) mod p
    // Broder et al. (2000): this is a sound approximation for min-wise independence
    prime_ = 4294967311ULL; // 2^32 + 15

    std::mt19937_64 rng(config_.random_seed);
    std::uniform_int_distribution<uint64_t> dist(1, prime_ - 1);
    hash_coeffs_.resize(config_.num_hashes);
    for (auto& [a, b] : hash_coeffs_) {
        a = dist(rng);
        b = dist(rng);
    }
}

uint64_t MinHashReorderer::hash(uint64_t x, uint32_t hash_idx) const {
    auto [a, b] = hash_coeffs_[hash_idx];
    return (a * x + b) % prime_;
}

// ===========================================================================
// build_access_graph (dispatch)
// ===========================================================================

void MinHashReorderer::build_access_graph(uint64_t num_batches, BatchProvider provider) {
    if (graph_built_) {
        throw std::runtime_error("MinHashReorderer: build_access_graph already called");
    }
    total_batches_ = num_batches;
    graph_built_ = true;

    switch (config_.strategy) {
        case Strategy::SEGMENTED:
            build_segmented(num_batches, provider);
            break;
        case Strategy::MULTIPASS_BOUNDED:
            build_multipass(num_batches, provider);
            break;
    }
}

// ===========================================================================
// Strategy A: Segmented (DiskGNN Algorithm 1)
// ===========================================================================

void MinHashReorderer::build_segmented(uint64_t num_batches, BatchProvider& provider) {
    for (uint64_t batch_id = 0; batch_id < num_batches; ++batch_id) {
        auto row_ids = provider(batch_id);

        // Determine segment for this batch
        uint32_t seg_size = config_.segment_size;
        uint64_t segment_id = (seg_size > 0 && seg_size < num_batches)
            ? batch_id / seg_size : 0;

        for (uint64_t row_id : row_ids) {
            // Grow arrays if needed
            if (row_id >= hash_values_.size()) {
                hash_values_.resize(row_id + 1, UINT64_MAX);
                accessed_.resize(row_id + 1, false);
            }
            accessed_[row_id] = true;

            // Apply all k hash functions, accumulate min into composite key
            // Composite: segment_id in high 32 bits, hash value in low 32 bits
            // This ensures nodes group by segment first, then by MinHash
            for (uint32_t h = 0; h < config_.num_hashes; ++h) {
                uint64_t hval = hash(batch_id, h);
                uint64_t composite = (segment_id << 32) | (hval & 0xFFFFFFFF);
                hash_values_[row_id] = std::min(hash_values_[row_id], composite);
            }
        }
    }
}

// ===========================================================================
// Strategy B: Multi-pass Bounded (original contribution)
// ===========================================================================

void MinHashReorderer::build_multipass(uint64_t num_batches, BatchProvider& provider) {
    namespace fs = std::filesystem;

    // Phase 0: Materialize batch assignments to temp file
    temp_dir_ = fs::temp_directory_path()
        / ("mdb_minhash_" + std::to_string(getpid()) + "_" + std::to_string(config_.random_seed));
    fs::create_directories(temp_dir_);
    auto temp_path = temp_dir_ / "batches.tmp";

    {
        int fd = ::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            throw std::runtime_error(
                "MinHashReorderer: cannot create temp file: " + temp_path.string());
        }
        FdGuard guard(fd);

        for (uint64_t b = 0; b < num_batches; ++b) {
            auto row_ids = provider(b);
            uint64_t count = row_ids.size();
            write_all(fd, &b, 8, temp_path.string());
            write_all(fd, &count, 8, temp_path.string());
            if (count > 0) {
                write_all(fd, row_ids.data(), count * 8, temp_path.string());
                for (uint64_t rid : row_ids) {
                    if (rid >= accessed_.size()) {
                        accessed_.resize(rid + 1, false);
                    }
                    accessed_[rid] = true;
                }
            }
        }

        if (::fsync(fd) < 0) {
            throw std::runtime_error(
                "MinHashReorderer: fsync failed: " + safe_strerror(errno));
        }
    }

    // Phase 1..P: Multi-pass MinHash over temp file
    uint64_t N = accessed_.size();
    hash_values_.resize(N, 0); // fingerprint accumulator

    uint32_t total_hashes = config_.num_hashes;
    uint32_t K = config_.hashes_per_pass;
    std::vector<uint64_t> min_val;

    for (uint32_t h_start = 0; h_start < total_hashes; h_start += K) {
        uint32_t h_end = std::min(h_start + K, total_hashes);
        uint32_t num_h = h_end - h_start;

        // Initialize min_val: one array per hash function in this pass
        // Interleaved: min_val[row * num_h + h] for better cache behavior
        min_val.assign(N * num_h, UINT64_MAX);

        // Sequential scan of temp file
        int fd = ::open(temp_path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(
                "MinHashReorderer: cannot open temp file: " + temp_path.string());
        }
        FdGuard guard(fd);

        for (uint64_t b = 0; b < num_batches; ++b) {
            uint64_t batch_id, count;
            read_all(fd, &batch_id, 8, temp_path.string());
            read_all(fd, &count, 8, temp_path.string());

            // Compute hash values for this batch
            std::vector<uint64_t> hvals(num_h);
            for (uint32_t h = 0; h < num_h; ++h) {
                hvals[h] = hash(batch_id, h_start + h);
            }

            // Read row_ids and update min_val
            for (uint64_t i = 0; i < count; ++i) {
                uint64_t rid;
                read_all(fd, &rid, 8, temp_path.string());
                for (uint32_t h = 0; h < num_h; ++h) {
                    uint64_t& mv = min_val[rid * num_h + h];
                    mv = std::min(mv, hvals[h]);
                }
            }
        }

        // Mix each hash function's min_val into fingerprint
        for (uint64_t i = 0; i < N; ++i) {
            if (accessed_[i]) {
                for (uint32_t h = 0; h < num_h; ++h) {
                    // FNV-like mixing (Fowler-Noll-Vo inspired)
                    hash_values_[i] = hash_values_[i] * 2654435761ULL + min_val[i * num_h + h];
                }
            }
        }
    }

    // Cleanup temp files
    fs::remove_all(temp_dir_);
}

// ===========================================================================
// compute_permutation
// ===========================================================================

std::vector<uint64_t> MinHashReorderer::compute_permutation(uint64_t total_rows) const {
    if (!graph_built_) {
        throw std::runtime_error(
            "MinHashReorderer::compute_permutation: build_access_graph not called");
    }

    // Validate total_rows covers all accessed rows
    for (size_t i = total_rows; i < accessed_.size(); ++i) {
        if (accessed_[i]) {
            throw std::invalid_argument(
                "MinHashReorderer: total_rows " + std::to_string(total_rows) +
                " but accessed row " + std::to_string(i) + " exists");
        }
    }

    // Separate accessed and unaccessed rows
    std::vector<uint64_t> accessed_rows;
    std::vector<uint64_t> unaccessed_rows;
    for (uint64_t i = 0; i < total_rows; ++i) {
        if (i < accessed_.size() && accessed_[i]) {
            accessed_rows.push_back(i);
        } else {
            unaccessed_rows.push_back(i);
        }
    }

    // Sort accessed rows by their hash value
    std::sort(accessed_rows.begin(), accessed_rows.end(),
        [this](uint64_t a, uint64_t b) {
            return hash_values_[a] < hash_values_[b];
        });

    // Build permutation: accessed (sorted) + unaccessed
    std::vector<uint64_t> permutation;
    permutation.reserve(total_rows);
    permutation.insert(permutation.end(), accessed_rows.begin(), accessed_rows.end());
    permutation.insert(permutation.end(), unaccessed_rows.begin(), unaccessed_rows.end());

    return permutation;
}

// ===========================================================================
// compute_inverse
// ===========================================================================

std::vector<uint64_t> MinHashReorderer::compute_inverse(const std::vector<uint64_t>& permutation) {
    std::vector<uint64_t> inverse(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        inverse[permutation[i]] = i;
    }
    return inverse;
}

// ===========================================================================
// get_stats
// ===========================================================================

MinHashReorderer::Stats MinHashReorderer::get_stats() const {
    if (!graph_built_) {
        throw std::runtime_error("MinHashReorderer::get_stats: build_access_graph not called");
    }
    uint64_t accessed = 0;
    for (size_t i = 0; i < accessed_.size(); ++i) {
        if (accessed_[i]) ++accessed;
    }
    return Stats{
        accessed,
        total_batches_,
        accessed > 0 ? static_cast<double>(total_batches_) / accessed : 0.0
    };
}

} // namespace mdb::gnn
