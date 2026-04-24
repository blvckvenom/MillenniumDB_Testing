#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace HNSW {

// Bitset-based visited set for HNSW search.
// Uses a flat bit-vector for O(1) test/set and a tracking list for O(visited) reset,
// avoiding the hash overhead of boost::unordered_set<uint32_t>.
class HNSWVisitedSet {
    std::vector<bool> bits_;
    std::vector<uint32_t> visited_list_; // Tracks set bits for O(visited) reset

public:
    HNSWVisitedSet() = default;

    explicit HNSWVisitedSet(size_t num_nodes) :
        bits_(num_nodes, false)
    {
        visited_list_.reserve(std::min(num_nodes, size_t(1024)));
    }

    // Pre-allocate the bitset to hold at least num_nodes entries.
    // Matches the old unordered_set::reserve() call site semantics.
    void reserve(size_t num_nodes) {
        if (num_nodes > bits_.size()) {
            bits_.resize(num_nodes, false);
        }
        visited_list_.reserve(std::min(num_nodes, size_t(1024)));
    }

    // Mark node_id as visited. No-op if already visited.
    // Matches the old unordered_set::emplace() call site semantics.
    void emplace(uint32_t node_id) {
        if (node_id >= bits_.size()) {
            bits_.resize(static_cast<size_t>(node_id) + 1, false);
        }
        if (!bits_[node_id]) {
            bits_[node_id] = true;
            visited_list_.push_back(node_id);
        }
    }

    // Check if node_id has been visited.
    // Matches the old unordered_set::contains() call site semantics.
    bool contains(uint32_t node_id) const {
        if (node_id >= bits_.size()) {
            return false;
        }
        return bits_[node_id];
    }

    // Atomically test and mark node_id as visited.
    // Returns true if the node was already visited (skip), false if newly visited (process).
    bool test_and_set(uint32_t node_id) {
        if (node_id >= bits_.size()) {
            bits_.resize(static_cast<size_t>(node_id) + 1, false);
        }
        if (bits_[node_id]) {
            return true;
        }
        bits_[node_id] = true;
        visited_list_.push_back(node_id);
        return false;
    }

    // Reset all visited bits in O(visited) time.
    // Matches the old unordered_set::clear() call site semantics.
    void clear() {
        for (uint32_t id : visited_list_) {
            bits_[id] = false;
        }
        visited_list_.clear();
    }
};

} // namespace HNSW
