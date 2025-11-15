#include "projection_storage.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>

#include "projection_catalog.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/bplus_tree/bpt_mem_import.h"
#include "storage/index/record.h"

namespace GQL {

// Helper function to initialize an empty BPlusTree
template<std::size_t N>
static void init_empty_bptree(const std::string& base_name) {
    BPTLeafWriter<N> leaf_writer(base_name + ".leaf");
    leaf_writer.make_empty();

    BPTDirWriter<N> dir_writer(base_name + ".dir");
    // dir_writer automatically creates a root page when destroyed
}

ProjectionStorage::ProjectionStorage(const std::string& projection_dir_, const std::string& db_folder)
    : projection_dir(projection_dir_)
{
    // Calculate relative path from db_folder
    // E.g., if projection_dir = "test_db/projections/test_projection" and db_folder = "test_db"
    // then rel_dir = "projections/test_projection"
    if (projection_dir.find(db_folder) == 0) {
        rel_dir = projection_dir.substr(db_folder.length());
        // Remove leading slash if present
        if (!rel_dir.empty() && rel_dir[0] == '/') {
            rel_dir = rel_dir.substr(1);
        }
    } else {
        // Fallback: use full path
        rel_dir = projection_dir;
    }

    // Extract projection name from path (last component)
    size_t last_slash = projection_dir.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        projection_name = projection_dir.substr(last_slash + 1);
    } else {
        projection_name = projection_dir;
    }

    // Pre-allocate for better performance
    inserted_nodes.reserve(INITIAL_CAPACITY);
    inserted_edges.reserve(INITIAL_CAPACITY);
    node_batch.reserve(BATCH_SIZE);
    edge_batch.reserve(BATCH_SIZE);
}

ProjectionStorage::ProjectionStorage(const std::string& projection_dir_, const std::string& db_folder, const std::string& projection_name_)
    : projection_dir(projection_dir_), projection_name(projection_name_)
{
    // Calculate relative path from db_folder
    if (projection_dir.find(db_folder) == 0) {
        rel_dir = projection_dir.substr(db_folder.length());
        if (!rel_dir.empty() && rel_dir[0] == '/') {
            rel_dir = rel_dir.substr(1);
        }
    } else {
        rel_dir = projection_dir;
    }

    // Pre-allocate for better performance
    inserted_nodes.reserve(INITIAL_CAPACITY);
    inserted_edges.reserve(INITIAL_CAPACITY);
    node_batch.reserve(BATCH_SIZE);
    edge_batch.reserve(BATCH_SIZE);
}

ProjectionStorage::ProjectionStorage(const std::string& projection_dir_, const std::string& db_folder, const std::string& projection_name_, const Features& features_)
    : projection_dir(projection_dir_), projection_name(projection_name_), features(features_)
{
    // Calculate relative path from db_folder
    if (projection_dir.find(db_folder) == 0) {
        rel_dir = projection_dir.substr(db_folder.length());
        if (!rel_dir.empty() && rel_dir[0] == '/') {
            rel_dir = rel_dir.substr(1);
        }
    } else {
        rel_dir = projection_dir;
    }

    // Pre-allocate for better performance
    inserted_nodes.reserve(INITIAL_CAPACITY);
    inserted_edges.reserve(INITIAL_CAPACITY);
    node_batch.reserve(BATCH_SIZE);
    edge_batch.reserve(BATCH_SIZE);
}

ProjectionStorage::~ProjectionStorage() {
    flush();
}

void ProjectionStorage::init() {
    // Initialize B+tree indexes using relative paths
    // rel_dir is relative to file_manager's db_folder (e.g., "projections/test_projection")

    // Initialize required indexes (always created)
    init_empty_bptree<1>(projection_dir + "/nodes");
    init_empty_bptree<3>(projection_dir + "/from_to_edge");
    init_empty_bptree<3>(projection_dir + "/to_from_edge");
    init_empty_bptree<2>(projection_dir + "/edge_direction");

    nodes_index = std::make_unique<BPlusTree<1>>(rel_dir + "/nodes");
    from_to_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/from_to_edge");
    to_from_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/to_from_edge");
    edge_direction_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_direction");

    // Initialize optional label indexes if requested
    if (features.include_node_labels) {
        init_empty_bptree<2>(projection_dir + "/node_label");
        init_empty_bptree<2>(projection_dir + "/label_node");  // Auxiliary index for label->node lookup
        node_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/node_label");
        label_node_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_node");
    }

    if (features.include_edge_labels) {
        init_empty_bptree<2>(projection_dir + "/edge_label");
        init_empty_bptree<2>(projection_dir + "/label_edge");  // Auxiliary index for label->edge lookup
        edge_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_label");
        label_edge_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_edge");
    }

    // Initialize optional property indexes if requested
    if (features.include_node_properties) {
        init_empty_bptree<3>(projection_dir + "/node_key_value");
        init_empty_bptree<3>(projection_dir + "/key_value_node");  // Auxiliary index for key/value->node lookup
        node_key_value_index = std::make_unique<BPlusTree<3>>(rel_dir + "/node_key_value");
        key_value_node_index = std::make_unique<BPlusTree<3>>(rel_dir + "/key_value_node");
    }

    if (features.include_edge_properties) {
        init_empty_bptree<3>(projection_dir + "/edge_key_value");
        init_empty_bptree<3>(projection_dir + "/key_value_edge");  // Auxiliary index for key/value->edge lookup
        edge_key_value_index = std::make_unique<BPlusTree<3>>(rel_dir + "/edge_key_value");
        key_value_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/key_value_edge");
    }
}

void ProjectionStorage::open() {
    // Open existing BPlusTree objects (using relative paths for file_manager)
    // Do NOT call init_empty_bptree - the files already exist!

    // Load catalog to restore statistics and metadata
    std::filesystem::path proj_path(projection_dir);
    if (std::filesystem::exists(proj_path / "catalog.dat")) {
        ProjectionCatalog catalog(projection_dir);
        catalog.load();

        // Restore statistics from catalog
        node_count = catalog.node_count;
        edge_count = catalog.edge_count;
        directed_edge_count = catalog.directed_edge_count;
        undirected_edge_count = catalog.undirected_edge_count;
    }

    // Open required indexes (always present)
    nodes_index = std::make_unique<BPlusTree<1>>(rel_dir + "/nodes");
    from_to_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/from_to_edge");
    to_from_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/to_from_edge");
    edge_direction_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_direction");

    // Open optional label indexes if they exist
    if (std::filesystem::exists(proj_path / "node_label.leaf")) {
        node_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/node_label");
        features.include_node_labels = true;
        // Also open auxiliary index if it exists
        if (std::filesystem::exists(proj_path / "label_node.leaf")) {
            label_node_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_node");
        }
    }

    if (std::filesystem::exists(proj_path / "edge_label.leaf")) {
        edge_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_label");
        features.include_edge_labels = true;
        // Also open auxiliary index if it exists
        if (std::filesystem::exists(proj_path / "label_edge.leaf")) {
            label_edge_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_edge");
        }
    }

    // Open optional property indexes if they exist
    if (std::filesystem::exists(proj_path / "node_key_value.leaf")) {
        node_key_value_index = std::make_unique<BPlusTree<3>>(rel_dir + "/node_key_value");
        features.include_node_properties = true;
        // Also open auxiliary index if it exists
        if (std::filesystem::exists(proj_path / "key_value_node.leaf")) {
            key_value_node_index = std::make_unique<BPlusTree<3>>(rel_dir + "/key_value_node");
        }
    }

    if (std::filesystem::exists(proj_path / "edge_key_value.leaf")) {
        edge_key_value_index = std::make_unique<BPlusTree<3>>(rel_dir + "/edge_key_value");
        features.include_edge_properties = true;
        // Also open auxiliary index if it exists
        if (std::filesystem::exists(proj_path / "key_value_edge.leaf")) {
            key_value_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/key_value_edge");
        }
    }
}

void ProjectionStorage::add_node(const ProjectedNode& node) {
    uint64_t node_id_val = node.node_id.id;

    // Check if already inserted
    if (inserted_nodes.find(node_id_val) != inserted_nodes.end()) {
        return;
    }

    // Add to batch
    node_batch.push_back(node);
    inserted_nodes.insert(node_id_val);

    // Flush batch if it reaches threshold
    if (node_batch.size() >= BATCH_SIZE) {
        flush_node_batch();
    }
}

void ProjectionStorage::add_edge(const ProjectedEdge& edge) {
    uint64_t edge_id = edge.edge_id.id;

    // Check if already inserted
    if (inserted_edges.find(edge_id) != inserted_edges.end()) {
        std::cerr << "[ProjectionStorage] Skipping duplicate edge 0x" << std::hex << edge_id << std::dec << std::endl;
        return;
    }

    std::cerr << "[ProjectionStorage] Adding edge 0x" << std::hex << edge_id << std::dec
              << " to batch (size before: " << edge_batch.size() << ")" << std::endl;

    // Add to batch
    edge_batch.push_back(edge);
    inserted_edges.insert(edge_id);

    // Flush batch if it reaches threshold
    if (edge_batch.size() >= BATCH_SIZE) {
        std::cerr << "[ProjectionStorage] Batch full, flushing " << edge_batch.size() << " edges" << std::endl;
        flush_edge_batch();
    }
}

void ProjectionStorage::add_node_label(ObjectId node_id, ObjectId label_id) {
    // Only insert if label index is enabled
    if (!node_label_index) {
        return;
    }

    // Write to primary index: {node_id, label_id}
    Record<2> node_label_record;
    node_label_record[0] = node_id.id;
    node_label_record[1] = label_id.id;
    node_label_index->insert(node_label_record);

    // Write to auxiliary index: {label_id, node_id} (for efficient label->nodes queries)
    if (label_node_index) {
        Record<2> label_node_record;
        label_node_record[0] = label_id.id;
        label_node_record[1] = node_id.id;
        label_node_index->insert(label_node_record);
    }
}

void ProjectionStorage::add_edge_label(ObjectId edge_id, ObjectId label_id) {
    // Only insert if label index is enabled
    if (!edge_label_index) {
        return;
    }

    // Write to primary index: {edge_id, label_id}
    Record<2> edge_label_record;
    edge_label_record[0] = edge_id.id;
    edge_label_record[1] = label_id.id;
    edge_label_index->insert(edge_label_record);

    // Write to auxiliary index: {label_id, edge_id} (for efficient label->edges queries)
    if (label_edge_index) {
        Record<2> label_edge_record;
        label_edge_record[0] = label_id.id;
        label_edge_record[1] = edge_id.id;
        label_edge_index->insert(label_edge_record);
    }
}

void ProjectionStorage::add_node_property(ObjectId node_id, ObjectId key_id, ObjectId value_id) {
    // Only insert if property index is enabled
    if (!node_key_value_index) {
        return;
    }

    // Write to primary index: {node_id, key_id, value_id}
    Record<3> node_prop_record;
    node_prop_record[0] = node_id.id;
    node_prop_record[1] = key_id.id;
    node_prop_record[2] = value_id.id;
    node_key_value_index->insert(node_prop_record);

    // Write to auxiliary index: {key_id, value_id, node_id} (for efficient property->nodes queries)
    if (key_value_node_index) {
        Record<3> key_value_node_record;
        key_value_node_record[0] = key_id.id;
        key_value_node_record[1] = value_id.id;
        key_value_node_record[2] = node_id.id;
        key_value_node_index->insert(key_value_node_record);
    }
}

void ProjectionStorage::add_edge_property(ObjectId edge_id, ObjectId key_id, ObjectId value_id) {
    // Only insert if property index is enabled
    if (!edge_key_value_index) {
        return;
    }

    // Write to primary index: {edge_id, key_id, value_id}
    Record<3> edge_prop_record;
    edge_prop_record[0] = edge_id.id;
    edge_prop_record[1] = key_id.id;
    edge_prop_record[2] = value_id.id;
    edge_key_value_index->insert(edge_prop_record);

    // Write to auxiliary index: {key_id, value_id, edge_id} (for efficient property->edges queries)
    if (key_value_edge_index) {
        Record<3> key_value_edge_record;
        key_value_edge_record[0] = key_id.id;
        key_value_edge_record[1] = value_id.id;
        key_value_edge_record[2] = edge_id.id;
        key_value_edge_index->insert(key_value_edge_record);
    }
}

bool ProjectionStorage::has_node(ObjectId node_id) const {
    if (!nodes_index) {
        return false;
    }

    Record<1> search_record;
    search_record[0] = node_id.id;

    // Use range query to check existence
    bool interruption_requested = false;
    auto iter = nodes_index->get_range(&interruption_requested, search_record, search_record);
    return iter.next() != nullptr;
}

bool ProjectionStorage::has_edge(ObjectId from, ObjectId to) const {
    if (!from_to_edge_index) {
        return false;
    }

    Record<3> min_record;
    min_record[0] = from.id;
    min_record[1] = to.id;
    min_record[2] = 0;

    Record<3> max_record;
    max_record[0] = from.id;
    max_record[1] = to.id;
    max_record[2] = UINT64_MAX;

    bool interruption_requested = false;
    auto iter = from_to_edge_index->get_range(&interruption_requested, min_record, max_record);
    return iter.next() != nullptr;
}

void ProjectionStorage::flush() {
    // Flush any pending batched writes
    flush_node_batch();
    flush_edge_batch();

    // Save catalog with projection metadata
    save_catalog();

    // B+trees are automatically flushed when they go out of scope
    // This method is here for explicit control if needed
}

void ProjectionStorage::flush_node_batch() {
    if (node_batch.empty()) {
        return;
    }

    // OPTIMIZATION: Collect node records and sort before insertion
    // Sorted insertion improves B+Tree performance via better cache locality and fewer page splits
    std::vector<Record<1>> node_records;
    std::vector<Record<3>> node_property_records;

    node_records.reserve(node_batch.size());

    for (const auto& node : node_batch) {
        uint64_t node_id_val = node.node_id.id;

        // Collect node record
        Record<1> node_record;
        node_record[0] = node_id_val;
        node_records.push_back(node_record);

        // Collect node properties if present
        if (!node.properties.empty()) {
            for (const auto& [prop_name, prop_value] : node.properties) {
                Record<3> prop_record;
                prop_record[0] = node_id_val;
                prop_record[1] = std::hash<std::string>{}(prop_name); // Simplified
                prop_record[2] = prop_value.id;
                node_property_records.push_back(prop_record);
            }
        }
    }

    // Sort records by key for optimal B+Tree insertion
    std::sort(node_records.begin(), node_records.end());

    // Insert sorted node records
    for (const auto& record : node_records) {
        if (nodes_index->insert(record)) {
            node_count++;
        }
    }

    // Handle node properties if any were collected
    if (!node_property_records.empty()) {
        if (!node_properties_index) {
            init_empty_bptree<3>(projection_dir + "/node_properties");
            node_properties_index = std::make_unique<BPlusTree<3>>(
                rel_dir + "/node_properties"
            );
        }

        // Sort property records by (node_id, property_name_hash, value)
        std::sort(node_property_records.begin(), node_property_records.end());

        // Insert sorted property records
        for (const auto& record : node_property_records) {
            node_properties_index->insert(record);
        }
    }

    node_batch.clear();
}

void ProjectionStorage::flush_edge_batch() {
    if (edge_batch.empty()) {
        return;
    }

    std::cerr << "[ProjectionStorage] flush_edge_batch: Flushing " << edge_batch.size() << " edges to B+tree" << std::endl;

    // OPTIMIZATION: Collect all edge records and sort before insertion
    // Separate vectors for each index to enable sorted batch insertion
    std::vector<Record<3>> from_to_records;
    std::vector<Record<3>> to_from_records;
    std::vector<Record<2>> direction_records;
    std::vector<Record<4>> edge_property_records;

    from_to_records.reserve(edge_batch.size());
    to_from_records.reserve(edge_batch.size());
    direction_records.reserve(edge_batch.size());

    for (const auto& edge : edge_batch) {
        uint64_t from_id = edge.from_node.id;
        uint64_t to_id = edge.to_node.id;
        uint64_t edge_id = edge.edge_id.id;

        // For undirected edges, normalize the ordering to avoid duplicates
        // Always store with lower node ID first to ensure consistent (from, to) pairs
        if (!edge.is_directed && from_id > to_id) {
            std::swap(from_id, to_id);
        }

        // Collect from->to record
        Record<3> from_to_record;
        from_to_record[0] = from_id;
        from_to_record[1] = to_id;
        from_to_record[2] = edge_id;
        from_to_records.push_back(from_to_record);

        // Collect to->from record
        Record<3> to_from_record;
        to_from_record[0] = to_id;
        to_from_record[1] = from_id;
        to_from_record[2] = edge_id;
        to_from_records.push_back(to_from_record);

        // Collect direction record
        Record<2> direction_record;
        direction_record[0] = edge_id;
        direction_record[1] = edge.is_directed ? 1 : 0;
        direction_records.push_back(direction_record);

        // Collect edge properties if present
        if (!edge.properties.empty()) {
            for (const auto& [prop_name, prop_value] : edge.properties) {
                Record<4> prop_record;
                prop_record[0] = edge_id;
                prop_record[1] = std::hash<std::string>{}(prop_name); // Simplified
                prop_record[2] = prop_value.id;
                prop_record[3] = 0; // Reserved
                edge_property_records.push_back(prop_record);
            }
        }
    }

    // Sort all record collections by key for optimal B+Tree insertion
    std::sort(from_to_records.begin(), from_to_records.end());
    std::sort(to_from_records.begin(), to_from_records.end());
    std::sort(direction_records.begin(), direction_records.end());

    // Insert sorted from->to records
    for (const auto& record : from_to_records) {
        bool inserted = from_to_edge_index->insert(record);
        if (inserted) {
            edge_count++;
            std::cerr << "[ProjectionStorage] Inserted edge 0x" << std::hex << record[2] << std::dec
                      << " into from_to_edge_index (" << record[0] << " -> " << record[1]
                      << "), edge_count=" << edge_count << std::endl;
        }
    }

    // Update directed/undirected counts (must iterate original batch for is_directed flag)
    for (const auto& edge : edge_batch) {
        if (edge.is_directed) {
            directed_edge_count++;
        } else {
            undirected_edge_count++;
        }
    }

    // Insert sorted to->from records
    for (const auto& record : to_from_records) {
        to_from_edge_index->insert(record);
    }

    // Insert sorted direction records
    for (const auto& record : direction_records) {
        edge_direction_index->insert(record);
    }

    // Handle edge properties if any were collected
    if (!edge_property_records.empty()) {
        if (!edge_properties_index) {
            init_empty_bptree<4>(projection_dir + "/edge_properties");
            edge_properties_index = std::make_unique<BPlusTree<4>>(
                rel_dir + "/edge_properties"
            );
        }

        // Sort property records by (edge_id, property_name_hash, value, reserved)
        std::sort(edge_property_records.begin(), edge_property_records.end());

        // Insert sorted property records
        for (const auto& record : edge_property_records) {
            edge_properties_index->insert(record);
        }
    }

    edge_batch.clear();
}

std::vector<ObjectId> ProjectionStorage::get_all_node_ids() const {
    std::vector<ObjectId> node_ids;

    if (!nodes_index) {
        return node_ids;
    }

    // Scan all nodes in the B+tree
    Record<1> min_record;
    min_record[0] = 0;

    Record<1> max_record;
    max_record[0] = UINT64_MAX;

    bool interruption_requested = false;
    auto iter = nodes_index->get_range(&interruption_requested, min_record, max_record);

    const Record<1>* record;
    while ((record = iter.next()) != nullptr) {
        node_ids.push_back(ObjectId((*record)[0]));
    }

    return node_ids;
}

std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>> ProjectionStorage::get_all_edges_info() const {
    std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>> edges;

    if (!from_to_edge_index || !edge_direction_index) {
        return edges;
    }

    // Scan all edges in the from_to_edge B+tree
    Record<3> min_record;
    min_record[0] = 0;
    min_record[1] = 0;
    min_record[2] = 0;

    Record<3> max_record;
    max_record[0] = UINT64_MAX;
    max_record[1] = UINT64_MAX;
    max_record[2] = UINT64_MAX;

    bool interruption_requested = false;
    auto iter = from_to_edge_index->get_range(&interruption_requested, min_record, max_record);

    const Record<3>* record;
    while ((record = iter.next()) != nullptr) {
        ObjectId from_node((*record)[0]);
        ObjectId to_node((*record)[1]);
        ObjectId edge_id((*record)[2]);

        // Look up direction for this edge
        Record<2> dir_min;
        dir_min[0] = edge_id.id;
        dir_min[1] = 0;

        Record<2> dir_max;
        dir_max[0] = edge_id.id;
        dir_max[1] = 1;

        bool interruption = false;
        auto dir_iter = edge_direction_index->get_range(&interruption, dir_min, dir_max);
        const Record<2>* dir_record = dir_iter.next();

        bool is_directed = true;
        if (dir_record != nullptr) {
            is_directed = ((*dir_record)[1] == 1);
        }

        edges.push_back(std::make_tuple(from_node, to_node, edge_id, is_directed));
    }

    return edges;
}

void ProjectionStorage::save_catalog() {
    // Don't create catalog if projection name is empty (e.g., when opening existing projection)
    if (projection_name.empty()) {
        return;
    }

    ProjectionCatalog catalog(projection_dir);

    // Set projection metadata
    catalog.projection_name = projection_name;
    catalog.creation_timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    // Set statistics
    catalog.node_count = node_count;
    catalog.edge_count = edge_count;
    catalog.directed_edge_count = directed_edge_count;
    catalog.undirected_edge_count = undirected_edge_count;

    // Set legacy configuration flags (v1.0 compatibility)
    catalog.has_node_properties = (node_properties_index != nullptr);
    catalog.has_edge_properties = (edge_properties_index != nullptr);
    catalog.undirected_relationships = (undirected_edge_count > 0);

    // Set v1.1 feature flags (optional indexes)
    catalog.includes_node_labels = features.include_node_labels;
    catalog.includes_edge_labels = features.include_edge_labels;
    catalog.includes_node_properties = features.include_node_properties;
    catalog.includes_edge_properties = features.include_edge_properties;

    // Save to disk
    catalog.save();
}

} // namespace GQL
