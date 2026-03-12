#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <shared_mutex>

#include <boost/unordered/unordered_flat_map.hpp>

#include "storage/catalog/catalog.h"
#include "storage/index/hnsw/hnsw_index.h"
#include "storage/index/hnsw/hnsw_metric.h"

namespace HNSW {

class HNSWIndexManager {
public:
    struct HNSWIndexMetadata {
        MetricType metric_type;
        std::string predicate;

        friend std::ostream& operator<<(std::ostream& os, const HNSWIndexMetadata& metadata)
        {
            os << "{\"metric_type\": " << metadata.metric_type;
            os << ", \"predicate\": " << metadata.predicate << "}";
            return os;
        }
    };

    static constexpr char HNSW_INDEX_DIR[] = "hnsw_index";

    // Initialize the hnsw index manager
    void init();

    ~HNSWIndexManager();

    void load_hnsw_index(const std::string& name, const HNSWIndexMetadata& metadata);

    void write_to_disk(const std::string& name, HNSWIndex* index);

    // Returns nullptr if the hnsw index was not found
    HNSWIndex* get_hnsw_index(const std::string& name);

    // Create a new hnsw index with the given name and predicate
    template<Catalog::ModelID model_id>
    uint_fast32_t create_hnsw_index(
        const std::string& name,
        const std::string& predicate,
        uint64_t dimension,
        uint64_t max_edges,
        uint64_t num_candidates,
        MetricType metric_type
    );

    boost::unordered_flat_map<std::string, HNSWIndexMetadata> get_name2metadata() const
    {
        std::shared_lock<std::shared_mutex> lck(name2hnsw_index_mutex);
        return name2metadata;
    }

    bool has_changes() const
    {
        return has_changes_.load(std::memory_order_acquire);
    }

    boost::unordered_flat_map<std::string, std::vector<std::string>> get_predicate2names() const
    {
        std::shared_lock<std::shared_mutex> lck(name2hnsw_index_mutex);
        return predicate2names;
    }

    std::vector<std::string> get_index_names()
    {
        std::shared_lock lck(name2hnsw_index_mutex);

        std::vector<std::string> res;
        res.reserve(name2hnsw_index.size());
        for (const auto& [name, _] : name2hnsw_index) {
            res.emplace_back(name);
        }

        return res;
    }

    // Drop (delete) an HNSW index by name
    // Returns true if the index was found and dropped, false otherwise
    bool drop_hnsw_index(const std::string& name);

    std::size_t num_hnsw_indexes() const
    {
        std::shared_lock<std::shared_mutex> lck(name2hnsw_index_mutex);
        return name2hnsw_index.size();
    }

private:
    std::atomic<bool> has_changes_ { false };

    // Prevents concurrent access to name2hnsw_index and related maps.
    // Lock ordering: update_execution_mutex (server) → name2hnsw_index_mutex.
    // Never acquire update_execution_mutex while holding this mutex.
    mutable std::shared_mutex name2hnsw_index_mutex;

    // Name to hnsw index
    boost::unordered_flat_map<std::string, std::unique_ptr<HNSWIndex>> name2hnsw_index;
    // Name to hnsw index metadata (dimension, max_edges, metric, etc...)
    boost::unordered_flat_map<std::string, HNSWIndexMetadata> name2metadata;
    // Predicate to hnsw index name
    boost::unordered_flat_map<std::string, std::vector<std::string>> predicate2names;
};
} // namespace HNSW
