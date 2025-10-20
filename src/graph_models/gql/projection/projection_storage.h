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

    // Feature flags for optional indexes
    struct Features {
        bool include_node_labels = false;
        bool include_edge_labels = false;
        bool include_node_properties = false;
        bool include_edge_properties = false;
    };

    // projection_dir should be the FULL path (e.g., "test_db/projections/test_projection")
    // db_folder should be the database root (e.g., "test_db") for constructing relative paths
    ProjectionStorage(const std::string& projection_dir, const std::string& db_folder);

    // Constructor with projection name (for catalog creation)
    ProjectionStorage(const std::string& projection_dir, const std::string& db_folder, const std::string& projection_name);

    // Constructor with feature flags (for creating projections with optional indexes)
    ProjectionStorage(const std::string& projection_dir, const std::string& db_folder, const std::string& projection_name, const Features& features);

    ~ProjectionStorage();

    // Initialize storage (create NEW B+trees - for writing)
    void init();

    // Open existing storage (for reading)
    void open();

    // Add a node to the projection (buffered)
    void add_node(const ProjectedNode& node);

    // Add an edge to the projection (buffered)
    void add_edge(const ProjectedEdge& edge);

    // Add a node label to the projection (requires INCLUDE LABELS)
    void add_node_label(ObjectId node_id, ObjectId label_id);

    // Add an edge label to the projection (requires INCLUDE LABELS)
    void add_edge_label(ObjectId edge_id, ObjectId label_id);

    // Add a node property to the projection (requires INCLUDE PROPERTIES)
    void add_node_property(ObjectId node_id, ObjectId key_id, ObjectId value_id);

    // Add an edge property to the projection (requires INCLUDE PROPERTIES)
    void add_edge_property(ObjectId edge_id, ObjectId key_id, ObjectId value_id);

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

    // Read methods for inspecting projection data
    std::vector<ObjectId> get_all_node_ids() const;
    std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>> get_all_edges_info() const;

    // Getters for required indexes (used by ProjectionQueryContext)
    BPlusTree<1>* get_nodes_index() { return nodes_index.get(); }
    BPlusTree<3>* get_from_to_edge_index() { return from_to_edge_index.get(); }
    BPlusTree<3>* get_to_from_edge_index() { return to_from_edge_index.get(); }
    BPlusTree<2>* get_edge_direction_index() { return edge_direction_index.get(); }

    // Getters for optional label indexes (may be null if not included)
    BPlusTree<2>* get_node_label_index() { return node_label_index.get(); }
    BPlusTree<2>* get_label_node_index() { return label_node_index.get(); }
    BPlusTree<2>* get_edge_label_index() { return edge_label_index.get(); }
    BPlusTree<2>* get_label_edge_index() { return label_edge_index.get(); }

    // Getters for optional property indexes (may be null if not included)
    BPlusTree<3>* get_node_key_value_index() { return node_key_value_index.get(); }
    BPlusTree<3>* get_key_value_node_index() { return key_value_node_index.get(); }
    BPlusTree<3>* get_edge_key_value_index() { return edge_key_value_index.get(); }
    BPlusTree<3>* get_key_value_edge_index() { return key_value_edge_index.get(); }

    // Const versions for read-only access
    const BPlusTree<1>* get_nodes_index() const { return nodes_index.get(); }
    const BPlusTree<3>* get_from_to_edge_index() const { return from_to_edge_index.get(); }
    const BPlusTree<3>* get_to_from_edge_index() const { return to_from_edge_index.get(); }
    const BPlusTree<2>* get_edge_direction_index() const { return edge_direction_index.get(); }

    const BPlusTree<2>* get_node_label_index() const { return node_label_index.get(); }
    const BPlusTree<2>* get_label_node_index() const { return label_node_index.get(); }
    const BPlusTree<2>* get_edge_label_index() const { return edge_label_index.get(); }
    const BPlusTree<2>* get_label_edge_index() const { return label_edge_index.get(); }
    const BPlusTree<3>* get_node_key_value_index() const { return node_key_value_index.get(); }
    const BPlusTree<3>* get_key_value_node_index() const { return key_value_node_index.get(); }
    const BPlusTree<3>* get_edge_key_value_index() const { return edge_key_value_index.get(); }
    const BPlusTree<3>* get_key_value_edge_index() const { return key_value_edge_index.get(); }

private:
    // Flush batched nodes to B+tree
    void flush_node_batch();

    // Flush batched edges to B+tree
    void flush_edge_batch();

    // Save catalog file with projection metadata
    void save_catalog();

    std::string projection_dir;  // Full path like "test_db/projections/test_projection"
    std::string rel_dir;          // Relative path from db_folder like "projections/test_projection"
    std::string projection_name;  // Name of the projection (extracted from path or provided)

    // Feature flags determining which optional indexes to create
    Features features;

    // Required indexes (always present)
    std::unique_ptr<BPlusTree<1>> nodes_index;           // {node_id}
    std::unique_ptr<BPlusTree<3>> from_to_edge_index;    // {from, to, edge_id}
    std::unique_ptr<BPlusTree<3>> to_from_edge_index;    // {to, from, edge_id}
    std::unique_ptr<BPlusTree<2>> edge_direction_index;  // {edge_id, is_directed}

    // Optional label indexes (only if INCLUDE LABELS specified)
    std::unique_ptr<BPlusTree<2>> node_label_index;      // {node_id, label_id} - given node, find labels
    std::unique_ptr<BPlusTree<2>> label_node_index;      // {label_id, node_id} - given label, find nodes
    std::unique_ptr<BPlusTree<2>> edge_label_index;      // {edge_id, label_id} - given edge, find labels
    std::unique_ptr<BPlusTree<2>> label_edge_index;      // {label_id, edge_id} - given label, find edges

    // Optional property indexes (only if INCLUDE PROPERTIES specified)
    std::unique_ptr<BPlusTree<3>> node_key_value_index;  // {node_id, key_id, value_id} - given node, find properties
    std::unique_ptr<BPlusTree<3>> key_value_node_index;  // {key_id, value_id, node_id} - given property, find nodes
    std::unique_ptr<BPlusTree<3>> edge_key_value_index;  // {edge_id, key_id, value_id} - given edge, find properties
    std::unique_ptr<BPlusTree<3>> key_value_edge_index;  // {key_id, value_id, edge_id} - given property, find edges

    // Legacy property storage (deprecated - for backward compatibility)
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
