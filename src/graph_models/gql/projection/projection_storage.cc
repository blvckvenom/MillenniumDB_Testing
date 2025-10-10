#include "projection_storage.h"

#include <stdexcept>

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

    // Initialize empty BPlusTree files (using absolute paths)
    init_empty_bptree<1>(projection_dir + "/nodes");
    init_empty_bptree<3>(projection_dir + "/from_to_edge");
    init_empty_bptree<3>(projection_dir + "/to_from_edge");
    init_empty_bptree<2>(projection_dir + "/edge_direction");

    // Now create BPlusTree objects (using relative paths for file_manager)
    nodes_index = std::make_unique<BPlusTree<1>>(rel_dir + "/nodes");
    from_to_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/from_to_edge");
    to_from_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/to_from_edge");
    edge_direction_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_direction");

    // Property indexes will be created on demand
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
        return;
    }

    // Add to batch
    edge_batch.push_back(edge);
    inserted_edges.insert(edge_id);

    // Flush batch if it reaches threshold
    if (edge_batch.size() >= BATCH_SIZE) {
        flush_edge_batch();
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

    // B+trees are automatically flushed when they go out of scope
    // This method is here for explicit control if needed
}

void ProjectionStorage::flush_node_batch() {
    if (node_batch.empty()) {
        return;
    }

    for (const auto& node : node_batch) {
        uint64_t node_id_val = node.node_id.id;

        // Insert into nodes index
        Record<1> node_record;
        node_record[0] = node_id_val;

        if (nodes_index->insert(node_record)) {
            node_count++;
        }

        // Handle node properties if present
        if (!node.properties.empty()) {
            if (!node_properties_index) {
                init_empty_bptree<3>(projection_dir + "/node_properties");
                node_properties_index = std::make_unique<BPlusTree<3>>(
                    rel_dir + "/node_properties"
                );
            }

            for (const auto& [prop_name, prop_value] : node.properties) {
                // Store property (simplified - in production would need property name encoding)
                Record<3> prop_record;
                prop_record[0] = node_id_val;
                prop_record[1] = std::hash<std::string>{}(prop_name); // Simplified
                prop_record[2] = prop_value.id;

                node_properties_index->insert(prop_record);
            }
        }
    }

    node_batch.clear();
}

void ProjectionStorage::flush_edge_batch() {
    if (edge_batch.empty()) {
        return;
    }

    for (const auto& edge : edge_batch) {
        uint64_t from_id = edge.from_node.id;
        uint64_t to_id = edge.to_node.id;
        uint64_t edge_id = edge.edge_id.id;

        // Insert into from->to index
        Record<3> from_to_record;
        from_to_record[0] = from_id;
        from_to_record[1] = to_id;
        from_to_record[2] = edge_id;

        if (from_to_edge_index->insert(from_to_record)) {
            edge_count++;

            if (edge.is_directed) {
                directed_edge_count++;
            } else {
                undirected_edge_count++;
            }
        }

        // Insert into to->from index
        Record<3> to_from_record;
        to_from_record[0] = to_id;
        to_from_record[1] = from_id;
        to_from_record[2] = edge_id;

        to_from_edge_index->insert(to_from_record);

        // Store edge direction
        Record<2> direction_record;
        direction_record[0] = edge_id;
        direction_record[1] = edge.is_directed ? 1 : 0;

        edge_direction_index->insert(direction_record);

        // Handle edge properties if present
        if (!edge.properties.empty()) {
            if (!edge_properties_index) {
                init_empty_bptree<4>(projection_dir + "/edge_properties");
                edge_properties_index = std::make_unique<BPlusTree<4>>(
                    rel_dir + "/edge_properties"
                );
            }

            for (const auto& [prop_name, prop_value] : edge.properties) {
                Record<4> prop_record;
                prop_record[0] = edge_id;
                prop_record[1] = std::hash<std::string>{}(prop_name); // Simplified
                prop_record[2] = prop_value.id;
                prop_record[3] = 0; // Reserved

                edge_properties_index->insert(prop_record);
            }
        }
    }

    edge_batch.clear();
}

} // namespace GQL
