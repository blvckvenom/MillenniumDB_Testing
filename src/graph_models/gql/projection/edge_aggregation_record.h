#pragma once

/**
 * @file edge_aggregation_record.h
 * @brief Record format for external sort-aggregate algorithm.
 *
 * This file defines the EdgeAggregationRecord structure used for memory-efficient
 * COUNT aggregation. Instead of holding all unique (from, to, type) pairs in a
 * hash table (O(N) memory), we stream edges to disk and use external merge-sort
 * followed by streaming aggregation (O(B) memory where B is buffer size).
 *
 * ## Memory Analysis
 *
 * Previous approach (ParallelEdgeDetector hash table):
 * - ParallelEdgeKey: 24 bytes
 * - EdgeAggregator: 64 bytes
 * - Hash table overhead: ~30%
 * - Total per unique pair: ~224 bytes
 * - For 35M pairs: ~7.8 GB
 *
 * New approach (External sort-aggregate):
 * - EdgeAggregationRecord: 40 bytes
 * - Disk space for 35M edges: ~1.4 GB
 * - Peak memory: ~320 MB (collection + sort + aggregate buffers)
 *
 * ## Algorithm Overview
 *
 * 1. **Collection**: Stream edges to disk via EdgeAggregationBuffer
 *    - Memory: 64 MB buffer, spills to disk when full
 *
 * 2. **Sort**: External K-way merge-sort by (from, to, type) key
 *    - Memory: 256 MB merge buffers
 *    - Reuses DiskVector pattern from import/disk_vector.h
 *
 * 3. **Aggregate**: Single pass over sorted stream
 *    - Memory: O(1) - tracks only current group
 *    - Emits (edge_id, count) when group changes
 *
 * @see external_edge_sort.h for sort implementation
 * @see streaming_aggregator.h for aggregation implementation
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <tuple>

namespace GQL {

/**
 * @brief Record format for external sort-aggregate.
 *
 * Sorted by (from_node, to_node, type_id) to enable streaming aggregation.
 * The edge_id field identifies the representative edge for property writes.
 *
 * ## Layout (40 bytes total)
 *
 * ```
 * +----------+----------+----------+----------+---------------+
 * | from_node| to_node  | type_id  | edge_id  | property_bits |
 * | 8 bytes  | 8 bytes  | 8 bytes  | 8 bytes  | 8 bytes       |
 * +----------+----------+----------+----------+---------------+
 * ```
 *
 * ## Sort Key
 * - Primary: from_node (ascending)
 * - Secondary: to_node (ascending)
 * - Tertiary: type_id (ascending)
 *
 * This ordering groups all parallel edges together for streaming aggregation.
 */
struct EdgeAggregationRecord {
    uint64_t from_node;      ///< Source node ID (sort key 1)
    uint64_t to_node;        ///< Target node ID (sort key 2)
    uint64_t type_id;        ///< Relationship type ID (sort key 3)
    uint64_t edge_id;        ///< Edge identifier (representative selection uses MIN)
    uint64_t property_bits;  ///< Union: double for SUM, unused for COUNT

    /**
     * @brief Comparison for sorting.
     *
     * Lexicographic ordering by (from_node, to_node, type_id).
     */
    bool operator<(const EdgeAggregationRecord& other) const {
        if (from_node != other.from_node) return from_node < other.from_node;
        if (to_node != other.to_node) return to_node < other.to_node;
        return type_id < other.type_id;
    }

    /**
     * @brief Extracts sort key for radix sort (ska_sort).
     *
     * Returns a tuple of (from_node, to_node, type_id) which ska_sort can
     * efficiently radix-sort by processing each uint64_t field byte-by-byte.
     * This achieves O(N) time complexity vs O(N log N) for comparison sort.
     */
    std::tuple<uint64_t, uint64_t, uint64_t> get_sort_key() const {
        return std::make_tuple(from_node, to_node, type_id);
    }

    /**
     * @brief Equality for duplicate detection.
     *
     * Two records belong to the same aggregation group if they have
     * the same (from_node, to_node, type_id) triple.
     */
    bool same_group(const EdgeAggregationRecord& other) const {
        return from_node == other.from_node &&
               to_node == other.to_node &&
               type_id == other.type_id;
    }

    /**
     * @brief Converts to array format for StreamingRecordBuffer<5>.
     */
    std::array<uint64_t, 5> to_array() const {
        return {from_node, to_node, type_id, edge_id, property_bits};
    }

    /**
     * @brief Constructs from array format.
     */
    static EdgeAggregationRecord from_array(const std::array<uint64_t, 5>& arr) {
        return EdgeAggregationRecord{arr[0], arr[1], arr[2], arr[3], arr[4]};
    }
};

/**
 * @brief Hash functor for EdgeAggregationRecord.
 *
 * Uses FNV-1a hash for good distribution across buckets.
 * Only used for small-graph in-memory path (< 1M edges).
 */
struct EdgeAggregationRecordHash {
    std::size_t operator()(const EdgeAggregationRecord& r) const {
        // FNV-1a hash constants for 64-bit
        constexpr uint64_t FNV_PRIME = 0x100000001b3;
        constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325;

        uint64_t hash = FNV_OFFSET;
        hash ^= r.from_node;
        hash *= FNV_PRIME;
        hash ^= r.to_node;
        hash *= FNV_PRIME;
        hash ^= r.type_id;
        hash *= FNV_PRIME;

        return static_cast<std::size_t>(hash);
    }
};

/**
 * @brief Equality functor for EdgeAggregationRecord group membership.
 */
struct EdgeAggregationRecordGroupEqual {
    bool operator()(const EdgeAggregationRecord& a, const EdgeAggregationRecord& b) const {
        return a.same_group(b);
    }
};

/**
 * @brief Utility to pack a double into uint64_t property_bits.
 */
inline uint64_t pack_double_to_bits(double value) {
    uint64_t bits;
    static_assert(sizeof(double) == sizeof(uint64_t), "double must be 8 bytes");
    std::memcpy(&bits, &value, sizeof(double));
    return bits;
}

/**
 * @brief Utility to unpack uint64_t property_bits to double.
 */
inline double unpack_bits_to_double(uint64_t bits) {
    double value;
    static_assert(sizeof(double) == sizeof(uint64_t), "double must be 8 bytes");
    std::memcpy(&value, &bits, sizeof(double));
    return value;
}

} // namespace GQL
