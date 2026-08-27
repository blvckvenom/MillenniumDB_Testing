#include "gnn/sampling/minhash_reorderer.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#include "gnn/common/posix_io.h"
#include "misc/ablation_registry.h"

namespace mdb::gnn {

// ===========================================================================
// Constructor + Destructor
// ===========================================================================

MinHashReorderer::MinHashReorderer(const Config& config) : config_(config) {
    if (config_.num_hashes == 0) {
        throw std::invalid_argument("MinHashReorderer: num_hashes must be > 0");
    }
    if (config_.strategy == Strategy::MULTIPASS_BOUNDED && config_.hashes_per_pass == 0) {
        throw std::invalid_argument("MinHashReorderer: hashes_per_pass must be > 0");
    }

    // Choose the largest prime below 2^32 so all hash outputs fit in 32 bits
    // without truncation when used as the low 32 bits of composite keys.
    // Broder et al. (2000): h(x) = (a*x + b) mod p is a sound approximation
    // for min-wise independence.
    prime_ = 4294967291ULL; // 2^32 - 5

    std::mt19937_64 rng(config_.random_seed);
    std::uniform_int_distribution<uint64_t> dist(1, prime_ - 1);
    hash_coeffs_.resize(config_.num_hashes);
    for (auto& [a, b] : hash_coeffs_) {
        a = dist(rng);
        b = dist(rng);
    }
}

MinHashReorderer::~MinHashReorderer() {
    // Clean up temp files created by Strategy B (MULTIPASS_BOUNDED), if any
    cleanup_temp_dir();
}

void MinHashReorderer::cleanup_temp_dir() noexcept {
    if (!temp_dir_.empty()) {
        try {
            std::filesystem::remove_all(temp_dir_);
        } catch (...) {
            // Best effort — ignore cleanup failures in destructor
        }
        temp_dir_.clear();
    }
}

uint64_t MinHashReorderer::hash(uint64_t x, uint32_t hash_idx) const {
    assert(hash_idx < hash_coeffs_.size());
    auto [a, b] = hash_coeffs_[hash_idx];
    return (a * x + b) % prime_;
}

// ===========================================================================
// build_access_graph (dispatch)
// ===========================================================================

void MinHashReorderer::build_access_graph(uint64_t num_batches, const BatchProvider& provider) {
    if (graph_built_) {
        throw std::runtime_error("MinHashReorderer: build_access_graph already called");
    }
    total_batches_ = num_batches;

    switch (config_.strategy) {
        case Strategy::SEGMENTED:
            build_segmented(num_batches, provider);
            break;
        case Strategy::MULTIPASS_BOUNDED:
            build_multipass(num_batches, provider);
            break;
    }

    // Set graph_built_ only after the strategy call completes successfully.
    // If the strategy throws, graph_built_ remains false so callers can detect the failure.
    graph_built_ = true;
}

// ===========================================================================
// Strategy A: Segmented (DiskGNN Algorithm 1)
// ===========================================================================

void MinHashReorderer::build_segmented(uint64_t num_batches, const BatchProvider& provider) {
    // The per-batch loop is parallelised. Each worker owns a local min-hash
    // array, accumulating updates independently; we merge post-pass with a
    // sequential reduction. This avoids the CAS-min contention that would
    // arise from atomic shared-state updates.
    //
    // Worker count default 4 (configurable via MDB_GNN_MINHASH_WORKERS),
    // capped at hardware_concurrency() and num_batches. Each worker holds an
    // N-sized uint64 array (8 bytes per node), so heap grows linearly with
    // the worker count; the default 4 workers cost ~32 bytes per node, which
    // stays a small fraction of host RAM even at 100M-node scale.
    // number() replaces stoi-with-swallowed-exception: a value that did not
    // parse used to leave the default standing with nothing said, so a run asked
    // for 16 workers reported exactly like one that got 4. The >0 test and the
    // int range test reproduce what stoi did (a non-positive value was ignored,
    // an out-of-range one threw and kept the default); only a value with
    // trailing garbage now reads differently, and it falls back declaring so.
    static const long minhash_workers_env =
        Ablation::number("MDB_GNN_MINHASH_WORKERS", 4);
    unsigned num_workers = 4;
    if (minhash_workers_env > 0 && minhash_workers_env <= INT_MAX) {
        num_workers = static_cast<unsigned>(minhash_workers_env);
    }
    if (num_workers > std::thread::hardware_concurrency() &&
        std::thread::hardware_concurrency() > 0)
    {
        num_workers = std::thread::hardware_concurrency();
    }
    if (num_workers > num_batches) {
        num_workers = static_cast<unsigned>(num_batches);
    }
    if (num_workers == 0) num_workers = 1;

    if (num_workers == 1) {
        // Serial path — preserves the historical behaviour exactly for
        // single-thread callers and small workloads (tests).
        for (uint64_t batch_id = 0; batch_id < num_batches; ++batch_id) {
            auto row_ids = provider(batch_id);
            uint32_t seg_size = config_.segment_size;
            uint64_t segment_id = (seg_size > 0 && seg_size < num_batches)
                ? batch_id / seg_size : 0;
            for (uint64_t row_id : row_ids) {
                if (row_id >= hash_values_.size()) {
                    hash_values_.resize(row_id + 1, UINT64_MAX);
                    accessed_.resize(row_id + 1, 0);
                }
                accessed_[row_id] = 1;
                ++total_accesses_;
                for (uint32_t h = 0; h < config_.num_hashes; ++h) {
                    uint64_t hval = hash(batch_id, h);
                    uint64_t composite = (segment_id << 32) | (hval & 0xFFFFFFFF);
                    hash_values_[row_id] = std::min(hash_values_[row_id], composite);
                }
            }
        }
        return;
    }

    // Parallel path: workers share atomic batch-dispatch counter; each
    // accumulates into its own (hash_values, accessed) arrays. Final
    // reduction merges into the canonical members.
    //
    // We pre-size to a conservative upper bound (probe the first batch to
    // discover the max row_id). Workers grow their local arrays lock-free
    // because each owns its own.
    std::atomic<uint64_t> next_batch{0};
    std::vector<std::vector<uint64_t>> local_hash_values(num_workers);
    std::vector<std::vector<uint8_t>>  local_accessed(num_workers);
    std::vector<uint64_t>              local_total(num_workers, 0);
    std::vector<std::exception_ptr>    errors(num_workers, nullptr);

    auto worker_fn = [&](unsigned w) {
        try {
            auto& lh = local_hash_values[w];
            auto& la = local_accessed[w];
            while (true) {
                uint64_t batch_id = next_batch.fetch_add(1, std::memory_order_relaxed);
                if (batch_id >= num_batches) break;
                auto row_ids = provider(batch_id);
                uint32_t seg_size = config_.segment_size;
                uint64_t segment_id = (seg_size > 0 && seg_size < num_batches)
                    ? batch_id / seg_size : 0;
                for (uint64_t row_id : row_ids) {
                    if (row_id >= lh.size()) {
                        lh.resize(row_id + 1, UINT64_MAX);
                        la.resize(row_id + 1, 0);
                    }
                    la[row_id] = 1;
                    ++local_total[w];
                    for (uint32_t h = 0; h < config_.num_hashes; ++h) {
                        uint64_t hval = hash(batch_id, h);
                        uint64_t composite = (segment_id << 32) | (hval & 0xFFFFFFFF);
                        lh[row_id] = std::min(lh[row_id], composite);
                    }
                }
            }
        } catch (...) {
            errors[w] = std::current_exception();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_workers);
    for (unsigned w = 0; w < num_workers; ++w) {
        threads.emplace_back(worker_fn, w);
    }
    for (auto& t : threads) t.join();
    for (unsigned w = 0; w < num_workers; ++w) {
        if (errors[w]) std::rethrow_exception(errors[w]);
    }

    // Reduction: take min across workers' local hash_values_.
    size_t max_size = 0;
    for (const auto& l : local_hash_values) max_size = std::max(max_size, l.size());
    hash_values_.assign(max_size, UINT64_MAX);
    accessed_.assign(max_size, 0);
    for (unsigned w = 0; w < num_workers; ++w) {
        const auto& lh = local_hash_values[w];
        const auto& la = local_accessed[w];
        total_accesses_ += local_total[w];
        for (size_t i = 0; i < lh.size(); ++i) {
            hash_values_[i] = std::min(hash_values_[i], lh[i]);
            accessed_[i] |= la[i];
        }
    }
}

// ===========================================================================
// Strategy B: Multi-pass Bounded (original contribution)
// ===========================================================================

void MinHashReorderer::build_multipass(uint64_t num_batches, const BatchProvider& provider) {
    namespace fs = std::filesystem;

    // Pass 0: Materialize batch assignments to temp file
    temp_dir_ = fs::temp_directory_path()
        / ("mdb_minhash_" + std::to_string(getpid()) + "_" + std::to_string(config_.random_seed));
    fs::create_directories(temp_dir_);
    auto temp_path = temp_dir_ / "batches.tmp";

    {
        int fd = ::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            throw std::runtime_error(
                "MinHashReorderer: cannot create temp file: " + temp_path.string()
                + ": " + safe_strerror(errno));
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
                        accessed_.resize(rid + 1, 0);
                    }
                    accessed_[rid] = 1;
                    ++total_accesses_;
                }
            }
        }

        if (::fsync(fd) < 0) {
            throw std::runtime_error(
                "MinHashReorderer: fsync failed (" + temp_path.string()
                + "): " + safe_strerror(errno));
        }
    }

    fsync_directory(temp_path);

    // Pass 1..P: Multi-pass MinHash over temp file
    uint64_t N = accessed_.size();
    hash_values_.resize(N, 0); // fingerprint accumulator

    uint32_t total_hashes = config_.num_hashes;
    uint32_t K = config_.hashes_per_pass;

    for (uint32_t h_start = 0; h_start < total_hashes; h_start += K) {
        uint32_t h_end = std::min(h_start + K, total_hashes);
        uint32_t num_h = h_end - h_start;

        // Guard against size_t overflow before allocating N * num_h * sizeof(uint64_t) bytes
        if (N > 0 && num_h > SIZE_MAX / N / sizeof(uint64_t)) {
            throw std::overflow_error(
                "MinHashReorderer: min_val allocation would overflow ("
                + std::to_string(N) + " * " + std::to_string(num_h) + ")");
        }

        std::vector<uint64_t> min_val(N * num_h, UINT64_MAX);

        // Sequential scan of temp file
        int fd = ::open(temp_path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(
                "MinHashReorderer: cannot open temp file: " + temp_path.string()
                + ": " + safe_strerror(errno));
        }
        FdGuard guard(fd);

        // Pre-allocate a reusable buffer outside the batch loop so we bulk-read all row_ids at once
        std::vector<uint64_t> row_id_buf;

        for (uint64_t b = 0; b < num_batches; ++b) {
            uint64_t batch_id, count;
            read_all(fd, &batch_id, 8, temp_path.string());
            read_all(fd, &count, 8, temp_path.string());

            // Compute hash values for this batch
            std::vector<uint64_t> hvals(num_h);
            for (uint32_t h = 0; h < num_h; ++h) {
                hvals[h] = hash(batch_id, h_start + h);
            }

            if (count == 0) continue;

            // Bulk read all row_ids for this batch in a single read_all call
            row_id_buf.resize(count);
            read_all(fd, row_id_buf.data(), count * 8, temp_path.string());

            // Update min_val for each row
            for (uint64_t i = 0; i < count; ++i) {
                uint64_t rid = row_id_buf[i];

                // Bounds-check each row_id read from the temp file against N to catch corruption
                if (rid >= N) {
                    throw std::runtime_error(
                        "MinHashReorderer: corrupted temp file — row_id "
                        + std::to_string(rid) + " >= N=" + std::to_string(N));
                }

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

    // Cleanup temp files (also done in destructor as safety net)
    cleanup_temp_dir();
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
    // avg_batches_per_node is total_accesses divided by accessed_nodes (not total_batches / accessed);
    // total_accesses counts every (node, batch) occurrence, giving the true mean co-occurrence rate
    return Stats{
        accessed,
        total_batches_,
        total_accesses_,
        accessed > 0 ? static_cast<double>(total_accesses_) / accessed : 0.0
    };
}

} // namespace mdb::gnn
