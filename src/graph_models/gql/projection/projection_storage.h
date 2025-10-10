#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph_models/object_id.h"

template<std::size_t N>
class BPlusTree;

namespace GQL {

class ProjectionCatalog;

// Represents a projected edge
struct ProjectedEdge {
    ObjectId from_node;
    ObjectId to_node;
    ObjectId edge_id;
    bool is_directed;

    // Optional properties
    std::unordered_map<std::string, ObjectId> properties;
};

// Represents a projected node
struct ProjectedNode {
    ObjectId node_id;

    // Optional properties
    std::unordered_map<std::string, ObjectId> properties;
};

class ProjectionStorage {
public:
    // Batch write configuration
    static constexpr size_t BATCH_SIZE = 1000;  // Flush after this many nodes/edges
    static constexpr size_t INITIAL_CAPACITY = 10000;  // Pre-allocate for better performance

    // projection_dir should be the FULL path (e.g., "test_db/projections/test_projection")
    // db_folder should be the database root (e.g., "test_db") for constructing relative paths
    ProjectionStorage(const std::string& projection_dir, const std::string& db_folder);
    ~ProjectionStorage();

    // Initialize storage (create B+trees)
    void init();

    // Add a node to the projection (buffered)
    void add_node(const ProjectedNode& node);

    // Add an edge to the projection (buffered)
    void add_edge(const ProjectedEdge& edge);

    // Check if a node exists
    bool has_node(ObjectId node_id) const;

    // Check if an edge exists
    bool has_edge(ObjectId from, ObjectId to) const;

    // Get node count
    uint64_t get_node_count() const { return node_count; }

    // Get edge count
    uint64_t get_edge_count() const { return edge_count; }

    // Get directed edge count
    uint64_t get_directed_edge_count() const { return directed_edge_count; }

    // Get undirected edge count
    uint64_t get_undirected_edge_count() const { return undirected_edge_count; }

    // Flush any pending writes (including batched data)
    void flush();

private:
    // Flush batched nodes to B+tree
    void flush_node_batch();

    // Flush batched edges to B+tree
    void flush_edge_batch();
    std::string projection_dir;  // Full path like "test_db/projections/test_projection"
    std::string rel_dir;          // Relative path from db_folder like "projections/test_projection"

    // B+tree indexes for nodes and edges
    std::unique_ptr<BPlusTree<1>> nodes_index;           // {node_id}
    std::unique_ptr<BPlusTree<3>> from_to_edge_index;    // {from, to, edge_id}
    std::unique_ptr<BPlusTree<3>> to_from_edge_index;    // {to, from, edge_id}
    std::unique_ptr<BPlusTree<2>> edge_direction_index;  // {edge_id, is_directed}

    // Property storage (optional)
    std::unique_ptr<BPlusTree<3>> node_properties_index; // {node_id, prop_key, prop_value}
    std::unique_ptr<BPlusTree<4>> edge_properties_index; // {edge_id, prop_key, prop_value}

    // Statistics
    uint64_t node_count = 0;
    uint64_t edge_count = 0;
    uint64_t directed_edge_count = 0;
    uint64_t undirected_edge_count = 0;

    // Keep track of inserted nodes/edges to avoid duplicates
    std::unordered_set<uint64_t> inserted_nodes;
    std::unordered_set<uint64_t> inserted_edges;

    // Batch write buffers for performance
    std::vector<ProjectedNode> node_batch;
    std::vector<ProjectedEdge> edge_batch;
};

} // namespace GQL
