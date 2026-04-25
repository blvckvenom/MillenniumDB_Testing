#pragma once

/**
 * @file bloom_filter.h
 * @brief Memory-efficient probabilistic set membership testing.
 *
 * Implements a Bloom filter for O(1) probabilistic duplicate detection with
 * configurable false positive rate. Used to replace hash sets during projection
 * creation to reduce memory from O(n) to O(1).
 *
 * @see Bloom, B.H. "Space/time trade-offs in hash coding with allowable errors" (1970)
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "third_party/hashes/hash_function_wrapper.h"

namespace GQL {

/**
 * @brief Space-efficient probabilistic set for duplicate detection.
 *
 * Provides O(1) insertion and lookup with configurable false positive rate.
 * No false negatives - if the filter says "not present", it's definitely not present.
 *
 * ## Memory Usage
 * For FPR (false positive rate) p and n elements:
 * - Optimal bits: m = -n * ln(p) / (ln(2)^2)
 * - Optimal hash count: k = (m/n) * ln(2)
 *
 * Example for 123M edges with 1% FPR:
 * - m = ~1.2 billion bits = ~150 MB
 * - k = 7 hash functions
 *
 * ## Usage Pattern
 * ```cpp
 * BloomFilter filter(expected_elements, 0.01);  // 1% FPR
 * if (!filter.probably_contains(key)) {
 *     filter.add(key);
 *     // First occurrence - process it
 * }
 * // Final deduplication: std::unique() on sorted data
 * ```
 *
 * @note False positives mean some duplicates slip through.
 *       Final correctness guaranteed by std::unique() during index build.
 */
class BloomFilter {
public:
    /**
     * @brief Constructs a Bloom filter optimized for given parameters.
     *
     * @param expected_elements Expected number of unique elements
     * @param false_positive_rate Desired false positive rate (0.0 to 1.0)
     *
     * Memory allocation: O(expected_elements / false_positive_rate) bits
     */
    BloomFilter(size_t expected_elements, double false_positive_rate = 0.01)
        : num_elements_(0)
    {
        // Clamp FPR to reasonable range
        if (false_positive_rate <= 0.0) false_positive_rate = 0.001;
        if (false_positive_rate >= 1.0) false_positive_rate = 0.5;

        // Calculate optimal parameters
        // m = -n * ln(p) / (ln(2)^2)
        double ln2_squared = 0.4804530139182014246671025263266649717305529515945455;
        double ln_fpr = std::log(false_positive_rate);
        size_t optimal_bits = static_cast<size_t>(-static_cast<double>(expected_elements) * ln_fpr / ln2_squared);

        // Minimum 64 bits, align to 64 for efficiency
        num_bits_ = std::max(optimal_bits, size_t{64});
        num_bits_ = (num_bits_ + 63) & ~size_t{63};  // Round up to 64-bit boundary

        // k = (m/n) * ln(2)
        double ln2 = 0.693147180559945309417232121458176568075500134360255;
        num_hashes_ = std::max(1u, static_cast<unsigned>(
            static_cast<double>(num_bits_) / expected_elements * ln2 + 0.5
        ));

        // Cap hash functions at 10 (diminishing returns beyond this)
        num_hashes_ = std::min(num_hashes_, 10u);

        // Allocate bit array
        bits_.resize((num_bits_ + 63) / 64, 0);
    }

    /**
     * @brief Checks if an element is probably in the set.
     *
     * @param data Pointer to key data
     * @param length Length of key data in bytes
     * @return true if element might be present (possible false positive)
     * @return false if element is definitely not present (no false negatives)
     */
    bool probably_contains(const void* data, size_t length) const {
        for (unsigned i = 0; i < num_hashes_; ++i) {
            size_t bit_index = hash_with_seed(data, length, i) % num_bits_;
            size_t word_index = bit_index / 64;
            size_t bit_offset = bit_index % 64;

            if (!(bits_[word_index] & (1ULL << bit_offset))) {
                return false;  // Definitely not present
            }
        }
        return true;  // Probably present
    }

    /**
     * @brief Checks if a 24-byte EdgeKey is probably in the set.
     *
     * Optimized overload for the common case of EdgeKey (from, to, edge_id).
     *
     * @param from_node Source node ID
     * @param to_node Target node ID
     * @param edge_id Edge identifier
     * @return true if probably present, false if definitely not present
     */
    bool probably_contains_edge(uint64_t from_node, uint64_t to_node, uint64_t edge_id) const {
        uint64_t key[3] = {from_node, to_node, edge_id};
        return probably_contains(key, sizeof(key));
    }

    /**
     * @brief Adds an element to the filter.
     *
     * @param data Pointer to key data
     * @param length Length of key data in bytes
     */
    void add(const void* data, size_t length) {
        for (unsigned i = 0; i < num_hashes_; ++i) {
            size_t bit_index = hash_with_seed(data, length, i) % num_bits_;
            size_t word_index = bit_index / 64;
            size_t bit_offset = bit_index % 64;

            bits_[word_index] |= (1ULL << bit_offset);
        }
        num_elements_++;
    }

    /**
     * @brief Adds a 24-byte EdgeKey to the filter.
     *
     * Optimized overload for EdgeKey (from, to, edge_id).
     */
    void add_edge(uint64_t from_node, uint64_t to_node, uint64_t edge_id) {
        uint64_t key[3] = {from_node, to_node, edge_id};
        add(key, sizeof(key));
    }

    /**
     * @brief Clears all elements from the filter.
     *
     * O(m/64) where m is number of bits.
     */
    void clear() {
        std::fill(bits_.begin(), bits_.end(), 0);
        num_elements_ = 0;
    }

    /// @brief Returns number of elements added (for statistics)
    size_t size() const { return num_elements_; }

    /// @brief Returns memory usage in bytes
    size_t memory_bytes() const { return bits_.size() * sizeof(uint64_t); }

    /// @brief Returns number of hash functions used
    unsigned num_hash_functions() const { return num_hashes_; }

    /// @brief Returns estimated false positive rate based on actual fill
    double estimated_fpr() const {
        if (num_elements_ == 0) return 0.0;
        // FPR ≈ (1 - e^(-kn/m))^k
        double fill_ratio = static_cast<double>(num_hashes_ * num_elements_) / num_bits_;
        double prob_bit_set = 1.0 - std::exp(-fill_ratio);
        double fpr = std::pow(prob_bit_set, num_hashes_);
        return fpr;
    }

private:
    /**
     * @brief Computes hash with a seed for multiple hash functions.
     *
     * Uses double-hashing technique: h_i(x) = h1(x) + i * h2(x)
     * This gives k independent hash functions from 2 base hashes.
     */
    uint64_t hash_with_seed(const void* data, size_t length, unsigned seed) const {
        // Create seeded input by appending seed
        if (seed == 0) {
            return HashFunctionWrapper(data, length);
        }

        // Double hashing technique for additional hash functions
        // h_i(x) = (h1(x) + i * h2(x)) mod m
        uint64_t h1 = HashFunctionWrapper(data, length);

        // Compute h2 by hashing with a modified key
        char seeded_data[32];  // Max 24 bytes for EdgeKey + 4 bytes seed
        size_t total_len = std::min(length, size_t{28}) + sizeof(uint32_t);
        std::memcpy(seeded_data, data, std::min(length, size_t{28}));
        uint32_t seed_val = seed;
        std::memcpy(seeded_data + std::min(length, size_t{28}), &seed_val, sizeof(seed_val));
        uint64_t h2 = HashFunctionWrapper(seeded_data, total_len);

        // Ensure h2 is odd for better distribution
        h2 |= 1;

        return h1 + seed * h2;
    }

    std::vector<uint64_t> bits_;  ///< Bit array storage
    size_t num_bits_;              ///< Total number of bits
    unsigned num_hashes_;          ///< Number of hash functions
    size_t num_elements_;          ///< Elements added (for statistics)
};

} // namespace GQL
