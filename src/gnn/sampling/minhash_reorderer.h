#pragma once

// MinHash reordering algorithm based on DiskGNN (Liu et al., SIGMOD 2025)
// which uses HashOrder (Zhang et al., 2024) for disk cache reordering.
// Original: https://github.com/Liu-rj/DiskGNN (MIT License)
// Ported from CUDA/Python to CPU C++17. Extended with memory-bounded variant.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mdb::gnn {

class MinHashReorderer {
public:
    enum class Strategy {
        SEGMENTED,         // DiskGNN Algorithm 1 (validated)
        MULTIPASS_BOUNDED  // Memory-bounded variant (original contribution)
    };

    struct Config {
        Strategy strategy = Strategy::SEGMENTED;
        uint32_t num_hashes = 2;       // k hash functions (DiskGNN GitHub code uses 2; paper says "k")
        uint32_t hashes_per_pass = 8;  // Strategy B only: RAM vs I/O tradeoff
        uint32_t segment_size = 100;   // Strategy A only: batches per segment (0 = global)
        uint64_t random_seed = 42;     // reproducibility
    };

    /// Throws std::invalid_argument if num_hashes == 0 or hashes_per_pass == 0.
    explicit MinHashReorderer(const Config& config);

    using BatchProvider = std::function<std::vector<uint64_t>(uint64_t batch_id)>;

    /// Build access graph. Can only be called once per instance.
    /// Duplicate row IDs in a batch are harmless (min is idempotent).
    void build_access_graph(uint64_t num_batches, BatchProvider provider);

    /// Compute permutation. total_rows = FeatureMatrix::num_rows().
    /// permutation[i] = source row for output position i.
    /// Accessed nodes sorted by MinHash, unaccessed appended at end.
    std::vector<uint64_t> compute_permutation(uint64_t total_rows) const;

    /// Compute inverse: inverse[old_row] = new_position.
    static std::vector<uint64_t> compute_inverse(const std::vector<uint64_t>& permutation);

    struct Stats {
        uint64_t accessed_nodes;
        uint64_t total_batches;
        double   avg_batches_per_node;
    };
    /// Available after build_access_graph().
    Stats get_stats() const;

private:
    Config config_;

    // Hash function coefficients: h_j(x) = (a_j * x + b_j) mod prime_
    std::vector<std::pair<uint64_t, uint64_t>> hash_coeffs_;
    uint64_t prime_;

    bool graph_built_ = false;
    uint64_t total_batches_ = 0;

    // Strategy A state: hash_values_[row_id] = accumulated min-hash value
    std::vector<uint64_t> hash_values_;
    std::vector<bool> accessed_;

    // Strategy B state
    std::filesystem::path temp_dir_;

    uint64_t hash(uint64_t x, uint32_t hash_idx) const;
    void build_segmented(uint64_t num_batches, BatchProvider& provider);
    void build_multipass(uint64_t num_batches, BatchProvider& provider);
};

} // namespace mdb::gnn
